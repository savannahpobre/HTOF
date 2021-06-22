"""
  Module for parsing intermediate data from Hipparcos and Gaia.
  For Hipparcos (both reductions) and Gaia, the scan angle theta is the angle between the north
  equitorial pole (declination) and the along-scan axis, defined as positive if east of the north pole
  (positive for increasing RA).

  Author:
    G. Mirek Brandt
    Daniel Michalik
"""

import numpy as np
import pandas as pd
from scipy import stats, special
import warnings
from ast import literal_eval
import os
import re
import glob
import itertools
from math import ceil, floor
import pkg_resources

from astropy.time import Time
from astropy.table import QTable, Column, Table

from htof import settings as st
from htof.utils.data_utils import merge_consortia, safe_concatenate
from htof.utils.parse_utils import gaia_obmt_to_tcb_julian_year

import abc


class DataParser(object):
    """
    Base class for parsing Hip1, Hip2 and Gaia data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as pandas.DataFrame. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, parallax_factors=None):
        self.scan_angle = pd.Series(scan_angle, dtype=np.float64)
        self._epoch = pd.DataFrame(epoch, dtype=np.float64)
        self.residuals = pd.Series(residuals, dtype=np.float64)
        self.parallax_factors = pd.Series(parallax_factors, dtype=np.float64)
        self.along_scan_errs = pd.Series(along_scan_errs, dtype=np.float64)
        self.inverse_covariance_matrix = inverse_covariance_matrix

    @staticmethod
    def read_intermediate_data_file(star_id: str, intermediate_data_directory: str, skiprows, header, sep):
        star_id = str(star_id)
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # search for the star id with leading zeros stripped
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
            filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # search for files with the full 6 digit hipparcos string
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.zfill(6) + '*')
            filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) != 1:
            # take the file with which contains only the hip id if there are multiple matches
            filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_id.lstrip('0') + '*')
            filepath_list = match_filename(glob.glob(filepath, recursive=True), star_id)
        if len(filepath_list) == 0:
            raise FileNotFoundError('No file with name containing {0} or {1} or {2} found in {3}'
                                    ''.format(star_id, star_id.lstrip('0'), star_id.zfill(6), intermediate_data_directory))
        if len(filepath_list) > 1:
            raise FileNotFoundError('Unable to find the correct file among the {0} files containing {1}'
                                    'found in {2}'.format(len(filepath_list), star_id, intermediate_data_directory))
        data = pd.read_csv(filepath_list[0], sep=sep, skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id: str, intermediate_data_parent_directory: str, **kwargs):
        pass    # pragma: no cover

    def julian_day_epoch(self):
        return self._epoch.values.flatten()

    @property
    def epoch(self):
        return self._epoch.values.flatten()

    def calculate_inverse_covariance_matrices(self, cross_scan_along_scan_var_ratio=np.inf):
        self.inverse_covariance_matrix = calc_inverse_covariance_matrices(self.scan_angle,
                                                                          cross_scan_along_scan_var_ratio=cross_scan_along_scan_var_ratio,
                                                                          along_scan_errs=self.along_scan_errs)

    def write(self, path: str, *args, **kwargs):
        """
        :param path: str. filepath to write out the processed data.
        :param args: arguments for astropy.table.Table.write()
        :param kwargs: keyword arguments for astropy.table.Table.write()
        :return: None

        Note: The IntermediateDataParser.inverse_covariance_matrix are added to the table as strings
        so that they are easily writable. The icov matrix is saved a string.
        Each element of t['icov'] can be recovered with ast.literal_eval(t['icov'][i])
        where i is the index. ast.literal_eval(t['icov'][i]) will return a 2x2 list.
        """
        t = self.as_table()
        # transform icov matrices as writable strings.
        t['icov'] = [str(icov.tolist()) for icov in t['icov']]
        t.write(path, fast_writer=False, *args, **kwargs)

    def as_table(self):
        """
        :return: astropy.table.QTable
                 The IntermediateDataParser object tabulated.
                 This table has as columns all of the attributes of IntermediateDataParser.

                 For any attribute which is empty or None, the column will contain zeros.
        """
        cols = [self.scan_angle, self.julian_day_epoch(), self.residuals, self.along_scan_errs, self.inverse_covariance_matrix]
        cols = [Column(col) for col in cols]
        # replacing incorrect length columns with empties.
        cols = [col if len(col) == len(self) else Column(None, length=len(self)) for col in cols]

        t = QTable(cols, names=['scan_angle', 'julian_day_epoch', 'residuals', 'along_scan_errs', 'icov'])
        return t

    def __add__(self, other):
        all_scan_angles = pd.concat([self.scan_angle, other.scan_angle])
        all_epoch = pd.concat([pd.DataFrame(self.julian_day_epoch()), pd.DataFrame(other.julian_day_epoch())])
        all_residuals = pd.concat([self.residuals, other.residuals])
        all_along_scan_errs = pd.concat([self.along_scan_errs, other.along_scan_errs])
        # TODO: add parallax factors. Tricky because gaia missions do not have them.
        all_inverse_covariance_matrix = safe_concatenate(self.inverse_covariance_matrix,
                                                         other.inverse_covariance_matrix)

        return DataParser(scan_angle=all_scan_angles, epoch=all_epoch, residuals=all_residuals,
                          inverse_covariance_matrix=all_inverse_covariance_matrix,
                          along_scan_errs=all_along_scan_errs)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __len__(self):
        return len(self._epoch)


class GaiaData(DataParser):
    DEAD_TIME_TABLE_NAME = None

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=-np.inf, max_epoch=np.inf, along_scan_errs=None):
        super(GaiaData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                       epoch=epoch, residuals=residuals,
                                       inverse_covariance_matrix=inverse_covariance_matrix)
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def parse(self, star_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=0, header='infer', sep=r'\s*,\s*')
        data = self.trim_data(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'],
                              data, self.min_epoch, self.max_epoch)
        data = self.reject_dead_times(data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], data)
        self._epoch = data['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        self.scan_angle = data['scanAngle[rad]']

    def trim_data(self, epochs, data, min_mjd, max_mjd):
        valid = np.logical_and(epochs >= min_mjd, epochs <= max_mjd)
        return data[valid].dropna()

    def reject_dead_times(self, epochs, data):
        # there will be different astrometric gaps for gaia DR2 and DR3 because rejection criteria may change.
        # hence we have the appropriate parsers have different values for DEAD_TIME_TABLE_NAME.
        if self.DEAD_TIME_TABLE_NAME is None:
            # return the data if there is no dead time table specified.
            return data
        dead_time_table = Table.read(self.DEAD_TIME_TABLE_NAME)
        # convert on board mission time (OBMT) to julian day
        for col, newcol in zip(['start', 'end'], ['start_tcb_jd', 'end_tcb_jd']):
            dead_time_table[newcol] = gaia_obmt_to_tcb_julian_year(dead_time_table[col]).jd
        # make a mask of the epochs. Those that are within a dead time window have a value of 0 (masked)
        valid = np.ones(len(data), dtype=bool)
        for entry in dead_time_table:
            valid[np.logical_and(epochs >= entry['start_tcb_jd'], epochs <= entry['end_tcb_jd'])] = 0
        # reject the epochs which fall within a dead time window
        data = data[valid].dropna()
        return data


class DecimalYearData(DataParser):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(DecimalYearData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                              epoch=epoch, residuals=residuals,
                                              inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass  # pragma: no cover

    def julian_day_epoch(self):
        return Time(self._epoch.values.flatten(), format='decimalyear').jd


def calc_inverse_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=np.inf,
                                     along_scan_errs=None):
    """
    :param scan_angles: pandas.DataFrame.
            data frame with scan angles, e.g. as-is from IntermediateDataParser.read_intermediate_data_file.
            scan_angles.values is a numpy array with the scan angles
    :param cross_scan_along_scan_var_ratio: var_cross_scan / var_along_scan
    :param along_scan_errs: array. array of len(scan_angles), the errors in the along scan direction, one for each
    scan in scan_angles.
    :return An ndarray with shape (len(scan_angles), 2, 2), e.g. an array of covariance matrices in the same order
    as the scan angles
    """
    if along_scan_errs is None or len(along_scan_errs) == 0:
        along_scan_errs = np.ones_like(scan_angles.values.flatten())
    if np.any(np.isclose(along_scan_errs, 0)):
        warnings.warn('Along scan error found that is zero. This is unphysical. Cannot '
                      'compute the inverse covariance matrices. Setting this AL error '
                      'to a large number and continuing.', RuntimeWarning)
        along_scan_errs[np.isclose(along_scan_errs, 0)] = 1000
    icovariance_matrices = []
    icov_matrix_in_scan_basis = np.array([[1, 0],
                                         [0, 1/cross_scan_along_scan_var_ratio]])
    for theta, err in zip(scan_angles.values.flatten(), along_scan_errs):
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[s, -c], [c, s]])
        icov_matrix_in_ra_dec_basis = np.matmul(np.matmul(1/(err ** 2) * Rot, icov_matrix_in_scan_basis), Rot.T)
        icovariance_matrices.append(icov_matrix_in_ra_dec_basis)
    return np.array(icovariance_matrices)


class HipparcosOriginalData(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                    epoch=epoch, residuals=residuals,
                                                    inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, data_choice='MERGED'):
        """
        :param star_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC', 'BOTH', or 'MERGED. The standard is 'MERGED' which does a merger
        of the 'NDAC' and 'FAST' data reductions in the same way as the hipparcos 1991.25 catalog. 'BOTH' keeps
        both consortia's data in the IAD, which would be unphysical and is just for debugging. 'FAST' would keep
        only the FAST consortia data, likewise only NDAC would be kept if you selected 'NDAC'.
        """
        if (data_choice != 'NDAC') and (data_choice != 'FAST') and (data_choice != 'MERGED')\
                and (data_choice != 'BOTH'):
            raise ValueError('data choice has to be either NDAC or FAST or MERGED or BOTH.')
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=10, header='infer', sep=r'\s*\|\s*')
        data = self._fix_unnamed_column(data)
        data = self._select_data(data, data_choice)
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1998
        #  10.1051/aas:1998218, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians, arctan2(sin, cos)
        # Use the larger denominator when computing the epoch offset. 
        # This increases numerical precision and avoids NaNs if one of the two fields (IA3, IA4) is exactly zero.
        self._epoch = 1991.25 + (data['IA6'] / data['IA3']).where(abs(data['IA3']) > abs(data['IA4']), (data['IA7'] / data['IA4']))
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)
        self.along_scan_errs = data['IA9']  # unit milli-arcseconds
        self.parallax_factors = data['IA5']

    @staticmethod
    def _select_data(data, data_choice):
        # restrict intermediate data to either NDAC, FAST, or merge the NDAC and FAST results.
        if data_choice == 'MERGED':
            data = merge_consortia(data)
        elif data_choice != 'BOTH':
            data = data[data['IA2'].str.upper() == {'NDAC': 'N', 'FAST': 'F'}[data_choice]]
        return data

    @staticmethod
    def _fix_unnamed_column(data, correct_key='IA2', col_idx=1):
        data.rename(columns={data.columns[col_idx]: correct_key}, inplace=True)
        return data


class HipparcosRereductionDVDBook(DecimalYearData):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, catalog_f2=None, calculated_f2=None, catalog_soltype=None):
        super(HipparcosRereductionDVDBook, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                          epoch=epoch, residuals=residuals,
                                                          inverse_covariance_matrix=inverse_covariance_matrix)
        self._additional_rejected_epochs = {}  # epochs that need to be rejected due to the write out bug.
        self._rejected_epochs = {}  # epochs that are known rejects, e.g.,
        # those that have negative AL errors in the java tool
        self.catalog_f2 = catalog_f2
        self.calculated_f2 = calculated_f2
        self.catalog_soltype = catalog_soltype  # TODO actually populate

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, header_rows=1,
              attempt_adhoc_rejection=True, **kwargs):
        """
        :param: star_id:
        :param: intermediate_data_directory:
        :param: error_inflate: True if the along-scan errors are to be corrected by the inflation factor
        according to Appendix B of D. Michalik et al. 2014. Only turn this off for tests, or if the parameters
        required to compute the error inflation are unavailable.
        :param: header_rows: int.
        :return:

        Compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        see also Figure 2.1, section 2.5.1, and section 4.1.2
        NOTE: that the Hipparcos re-reduction book and the figures therein describe the
        scan angle against the north ecliptic pole.
        NOTE: In the actual intermediate astrometry data on the CD the scan angle
        is given as east of the north equatorial pole, as for the original
        Hipparcos and Gaia (Source: private communication between Daniel
        Michalik and Floor van Leeuwen, April 2019).
        """
        header = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                  skiprows=0, header=None, sep=r'\s+')
        data = self.read_intermediate_data_file(star_id, intermediate_data_directory,
                                                skiprows=header_rows, header=None, sep=r'\s+')
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(theta) = cos(psi), data[4] = cos(theta) = sin(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)
        self.parallax_factors = data[2]
        self.catalog_f2 = header.iloc[0][6]
        # TODO need to calculate f2 newly using htof. Like we do in the java tool.
        n_transits, nparam, percent_rejected = header.iloc[0][2], get_nparam(header.iloc[0][4]), header.iloc[0][7]
        if type(self) is HipparcosRereductionDVDBook and attempt_adhoc_rejection:
            warnings.warn(f"Print htof is attempting to find the epochs for this DVD IAD that need to be rejected."
                          f" However if this is a source with the write out bug, then this will fail. Preferably"
                          f" switch to using the Java Tool data.", UserWarning)
            self.rejected_epochs = find_epochs_to_reject_DVD(self, n_transits, percent_rejected, nparam, self.catalog_f2)
        if error_inflate:
            # adjust the along scan errors so that the errors on the best fit parameters match the catalog.
            self.along_scan_errs *= self.error_inflation_factor(n_transits, nparam, self.catalog_f2)
        return header, data

    @staticmethod
    def error_inflation_factor(ntr, nparam, f2):
        """
        :param ntr: int. Number of transits used in the catalog solution. I.e. this should be
        N_transit_total - N_reject. So if N_reject is unknown, then the error inflation factor will be slightly wrong.
        :param nparam: int. Number of parameters used in the solution (e.g. 5, 7, 9..)
        :param f2: float. Goodness of fit metric. field F2 in the Hipparcos Re-reduction catalog.
        :return: u. float.
        The errors are to be scaled by u = Sqrt(Q/v) in equation B.4 of D. Michalik et al. 2014.
        (Title: Joint astrometric solution of Hipparcos and Gaia)
        NOTE: ntr (the number of transits) given in the header of the Hip2 IAD, is not necessarily
        the number of transits used.
        """
        num_transits_used = ntr
        nu = num_transits_used - nparam  # equation B.1 of D. Michalik et al. 2014
        Q = nu * (np.sqrt(2/(9*nu))*f2 + 1 - 2/(9*nu))**3  # equation B.3
        u = np.sqrt(Q/nu)  # equation B.4. This is the chi squared statistic of the fit.
        return u

    def _reject_epochs(self, attr_to_set, value):
        residuals_to_reject, orbits_to_reject = value['residual/along_scan_error'], value['orbit/scan_angle/time']
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, residuals_to_reject, False)
        self.residuals, self.along_scan_errs = self.residuals[not_outlier], self.along_scan_errs[not_outlier]
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, orbits_to_reject, False)
        self._epoch, self.scan_angle = self._epoch[not_outlier], self.scan_angle[not_outlier]
        self.parallax_factors = self.parallax_factors[not_outlier]
        setattr(self, attr_to_set, value)

    @property
    def additional_rejected_epochs(self):
        return self._additional_rejected_epochs

    @additional_rejected_epochs.setter
    def additional_rejected_epochs(self, value):
        self._reject_epochs('_additional_rejected_epochs', value)

    @property
    def rejected_epochs(self):
        return self._rejected_epochs

    @rejected_epochs.setter
    def rejected_epochs(self, value):
        self._reject_epochs('_rejected_epochs', value)


class HipparcosRereductionJavaTool(HipparcosRereductionDVDBook):
    EPOCHREJECTLIST = Table.read(pkg_resources.resource_filename('htof',
                                                                 'data/epoch_reject_shortlist.csv'), format='ascii')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, catalog_f2=None, calculated_f2=None, catalog_soltype=None):
        super(HipparcosRereductionJavaTool, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                           epoch=epoch, residuals=residuals,
                                                           inverse_covariance_matrix=inverse_covariance_matrix,
                                                           catalog_f2=catalog_f2, calculated_f2=calculated_f2,
                                                           catalog_soltype=catalog_soltype)

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, attempt_adhoc_rejection=True,
              reject_known=True, **kwargs):
        header, raw_data = super(HipparcosRereductionJavaTool, self).parse(star_id, intermediate_data_directory,
                                                                           error_inflate=False, header_rows=6,
                                                                           attempt_adhoc_rejection=False)
        n_transits, n_expected_transits = header.iloc[0][2], header.iloc[1][4]
        n_additional_reject = int(n_transits) - int(n_expected_transits)
        max_n_auto_reject = 4
        if attempt_adhoc_rejection:
            if 3 >= n_additional_reject > 0:
                # Note: you could use find_epochs_to_reject_java_large here and it would be faster for n_additional_reject=3
                # and basically just as good. The only draw back is it will find slightly different rows (within the same
                # orbit) to reject, compared to find_epochs_to_reject_java
                self.additional_rejected_epochs = find_epochs_to_reject_java(self, n_additional_reject)
            if max_n_auto_reject >= n_additional_reject > 3:
                orbit_number = raw_data[0].values
                self.additional_rejected_epochs = find_epochs_to_reject_java_large(self, n_additional_reject, orbit_number)
            if n_additional_reject > max_n_auto_reject:
                # These take too long to do automatically, pull the epochs to reject from the file that we computed
                correct_id = header.iloc[0][0]
                t = self.EPOCHREJECTLIST[self.EPOCHREJECTLIST['hip_id'] == int(correct_id)]
                if len(t) == 1:
                    self.additional_rejected_epochs = {'residual/along_scan_error': literal_eval(t['residual/along_scan_error'][0]),
                                                       'orbit/scan_angle/time': literal_eval(t['orbit/scan_angle/time'][0])}
                else:
                    warnings.warn(f'Cannot fix {star_id}. It has more than {max_n_auto_reject} bugged epochs, and'
                                  f' the correct epochs to reject are not in the known list '
                                  f'(epoch_reject_shortlist.csv)', UserWarning)    # pragma: no cover
        if not attempt_adhoc_rejection and n_additional_reject > 0:
            warnings.warn(f"attempt_adhoc_rejection = False and {star_id} is a bugged source. "
                          "You are foregoing the write out bug "
                          "correction for this Java tool source. The IAD will not correspond exactly "
                          "to the best fit solution. ", UserWarning)
        epochs_to_reject = np.where(self.along_scan_errs <= 0)[0] # note that we have to reject
        # the epochs with negative along scan errors (the formally known epochs that need to be rejected)
        # AFTER we have done the bug correction (rejected the epochs from the write out bug). This order
        # is important because the write out bug correction shuffles the orbits.
        if len(epochs_to_reject) > 0 and reject_known:
            self.rejected_epochs = {'residual/along_scan_error': list(epochs_to_reject),
                                    'orbit/scan_angle/time': list(epochs_to_reject)}
        # compute f2 of the residuals (with ad-hoc correction where applicable)
        nparam = get_nparam(header.iloc[0][4])
        Q = np.sum((self.residuals/self.along_scan_errs)**2)
        n_transits_final = len(self)
        # note that n_transits_final = n_expected_transits - number of indicated rejects (By negative AL errors)
        self.calculated_f2 = special.erfcinv(stats.chi2.sf(Q, n_transits_final  - nparam)*2)*np.sqrt(2)
        # TODO move error_inflate out of parse. Has nothing to do with parsing.
        if error_inflate:
            self.along_scan_errs *= self.error_inflation_factor(n_transits_final, nparam, self.calculated_f2)
        return header, raw_data
        # setting self.rejected_epochs also rejects the epochs (see the @setter)


class GaiaDR2(GaiaData):
    DEAD_TIME_TABLE_NAME = pkg_resources.resource_filename('htof', 'data/astrometric_gaps_gaiadr2_08252020.csv')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaDR2_min_epoch, max_epoch=st.GaiaDR2_max_epoch, along_scan_errs=None):
        super(GaiaDR2, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)


class GaiaeDR3(GaiaData):
    DEAD_TIME_TABLE_NAME = pkg_resources.resource_filename('htof', 'data/astrometric_gaps_gaiaedr3_12232020.csv')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 min_epoch=st.GaiaeDR3_min_epoch, max_epoch=st.GaiaeDR3_max_epoch, along_scan_errs=None):
        super(GaiaeDR3, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                      epoch=epoch, residuals=residuals,
                                      inverse_covariance_matrix=inverse_covariance_matrix,
                                      min_epoch=min_epoch, max_epoch=max_epoch)


def digits_only(x: str):
    return re.sub("[^0-9]", "", x)


def match_filename(paths, star_id):
    return [f for f in paths if digits_only(os.path.basename(f).split('.')[0]).zfill(6) == star_id.zfill(6)]


def find_epochs_to_reject_DVD(data: DataParser, n_transits, percent_rejected, nparam, catalog_f2):
    # just looks for combinations of orbits within the dvd IAD that yield a stationary point of chisquared.
    # Note that this does not work for sources with the write out bug.
    chi2_thresh = 1
    possible_rejects = np.arange(len(data))
    min_n_reject = max(floor((percent_rejected - 1) / 100 * n_transits), 0)
    max_n_reject = max(ceil((percent_rejected + 1) / 100 * n_transits), 1)
    max_n_reject = min(max_n_reject, 3)  # limit to three rejected sources so that combinatorics dont blow up.
    # calculate the chisquared partials
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    rows_to_keep = np.ones(len(data), dtype=bool)
    orbit_factors = np.array([data.parallax_factors.values, sin_scan, cos_scan, dt * sin_scan, dt * cos_scan])
    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)
    chi2_vector = (2 * residual_factors * orbit_factors).T
    sum_chisquared_partials_norejects = np.sqrt(np.sum(np.sum(chi2_vector, axis=0) ** 2))
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we should be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_row_rejects_pern = [[]]
    candidate_row_chisquared_partials_pern = [sum_chisquared_partials_norejects]
    n_reject = max(min_n_reject, 1)
    while n_reject < max_n_reject:
        candidate_row_rejects = []
        candidate_row_chisquared_partials = []
        combinations = list(set(itertools.combinations(possible_rejects, int(n_reject))))
        for rows_to_reject in combinations:
            rows_to_keep[list(rows_to_reject)] = False
            # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
            sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[rows_to_keep], axis=0) ** 2))
            candidate_row_rejects.append(rows_to_reject)
            candidate_row_chisquared_partials.append(sum_chisquared_partials)
            # reset for the next loop:
            rows_to_keep[list(rows_to_reject)] = True
        n_reject += 1
        candidate_row_rejects_pern.append(np.array(candidate_row_rejects)[np.argmin(candidate_row_chisquared_partials)])
        candidate_row_chisquared_partials_pern.append(np.min(candidate_row_chisquared_partials))
    # see if any of the rejections are viable (i.e., check if this IAD is messed up in an unrepairable way)
    if np.min(candidate_row_chisquared_partials_pern) > chi2_thresh:
        warnings.warn(f"Attempted to find which rows to reject, but the chisquared partials "
                      f"are larger than {chi2_thresh}. "
                      "This is likely a bugged source. Aborting rejection routine. ", UserWarning)    # pragma: no cover
        return {'residual/along_scan_error': [], 'orbit/scan_angle/time': []}
    # exclude any rejections that do not yield stationary points.
    viable_rejections = np.where(np.array(candidate_row_chisquared_partials_pern) < chi2_thresh)[0]
    candidate_row_rejects_pern = [candidate_row_rejects_pern[v] for v in viable_rejections]
    candidate_row_chisquared_partials_pern = [candidate_row_chisquared_partials_pern[v] for v in viable_rejections]
    # calculate f2 values for all the viable rejections
    candidate_row_f2_vals_pern = []
    data_minus_model_squared = ((data.residuals.values / data.along_scan_errs.values) ** 2)
    for r in candidate_row_rejects_pern:
        rows_to_keep[list(r)] = False
        chisquared = np.sum(data_minus_model_squared[rows_to_keep])
        candidate_row_f2_vals_pern.append(compute_f2(n_transits - nparam, chisquared))
        rows_to_keep[list(r)] = True
    # restrict viable choices to the one that best matches f2
    reject_idx = candidate_row_rejects_pern[np.argmin(np.abs(np.array(candidate_row_f2_vals_pern) - catalog_f2))]
    return {'residual/along_scan_error': list(reject_idx), 'orbit/scan_angle/time': list(reject_idx)}


def find_epochs_to_reject_java(data: DataParser, n_additional_reject):
    # Note there are degeneracies in the best epochs to reject. E.g. for hip 39, as long as the last
    #  residual is rejected, basically any of the 1426 orbits (Because they are all similar)
    #  can be rejected and they result in a very similar chisquared.
    possible_rejects = np.arange(len(data))
    # calculate the chisquared partials
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    resid_reject_idx = [len(data) - 1 - i for i in range(int(n_additional_reject))]  # always reject the repeated observations.
    # need to iterate over popping orbit combinations
    orbits_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False

    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
    mask_rejected_resid = (data.along_scan_errs.values > 0).astype(bool)[residuals_to_keep]
    _orbit_factors = np.array([data.parallax_factors.values, sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we sghould be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []
    for orbit_to_reject in itertools.combinations(possible_rejects, int(n_additional_reject)):
        orbits_to_keep[list(orbit_to_reject)] = False
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.
        # this pops the orbits out and shifts all the orbits after:
        orbit_factors = _orbit_factors[orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        chi2_vector = (2 * residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[mask_rejected_resid], axis=0) ** 2))
        candidate_orbit_rejects.append(orbit_to_reject)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
        # reset for the next loop:
        orbits_to_keep[list(orbit_to_reject)] = True
    orbit_reject_idx = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        warnings.warn("Attempted to fix the write out bug, but the chisquared partials are larger than 0.5. There are "
                      "likely more additional rejected epochs than htof can handle.", UserWarning)    # pragma: no cover

    return {'residual/along_scan_error': list(resid_reject_idx),
            'orbit/scan_angle/time': list(orbit_reject_idx)}


def find_epochs_to_reject_java_large(data: DataParser, n_additional_reject, orbit_number):
    # this is for any java tool object where n_additional_reject is greater than 3.
    # we assume the scan angles and times of rows in the same orbit are similar, therefore we only have
    # to try all combinations of distributing n_additional_reject rejected epochs among N orbits
    # calculate the chisquared partials
    orbit_prototypes, orbit_index, orbit_multiplicity = np.unique(orbit_number, return_index=True, return_counts=True)
    num_unique_orbits = len(orbit_prototypes)
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    resid_reject_idx = [len(data) - 1 - i for i in range(int(n_additional_reject))]  # always reject the repeated observations.
    # need to iterate over popping orbit combinations
    orbits_to_keep = np.zeros(len(data), dtype=bool)
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False

    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
    mask_rejected_resid = (data.along_scan_errs.values > 0).astype(bool)[residuals_to_keep]
    _orbit_factors = np.array([sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we sghould be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []
    for rejects_from_each_orbit in partitions(n_additional_reject, num_unique_orbits):
        if np.any(rejects_from_each_orbit > orbit_multiplicity):
            # ignore any trials of rejects that put e.g. 10 rejects into an orbit with only 4 observations.
            continue
        end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
        for s, e in zip(orbit_index, end_index):
            orbits_to_keep[s:e] = True
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.
        # this pops the orbits out and shifts all the orbits after:
        orbit_factors = _orbit_factors[orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        chi2_vector = (2 * residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[mask_rejected_resid], axis=0) ** 2))
        candidate_orbit_rejects.append(rejects_from_each_orbit)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
        # reset for the next loop:
        orbits_to_keep[:] = False
    rejects_from_each_orbit = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    # now transform rejects_from_each_orbit into actual orbit indices that we are going to reject.
    end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
    for s, e in zip(orbit_index, end_index):
        orbits_to_keep[s:e] = True
    orbit_reject_idx = np.where(~orbits_to_keep)[0]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        warnings.warn("Attempted to fix the write out bug, but the chisquared partials are larger than 0.5. There are "
                      "likely more additional rejected epochs than htof can handle.", UserWarning)    # pragma: no cover

    return {'residual/along_scan_error': list(resid_reject_idx),
            'orbit/scan_angle/time': list(orbit_reject_idx)}


def partitions(n, k):
    """
    yield all possible weighs to distribute n rejected rows among k orbits.
    This is just the solution to the "stars and bars" problem.
    Theorem 2: https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29

    From https://stackoverflow.com/questions/28965734/general-bars-and-stars
    """
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


def get_nparam(nparam_header_val):
    # strip the solution type (5, 7, or 9) from the solution type, which is a number 10xd+s consisting of
    # two parts: d and s. see Note 1 on Vizier for the Hipparcos re-reduction.
    return int(str(int(nparam_header_val))[-1])


def compute_f2(nu, chisquared):
    # equation B.2 of D. Michalik et al. 2014. Joint astrometric solution of Hipparcos and Gaia
    return (9*nu/2)**(1/2)*((chisquared/nu)**(1/3) + 2/(9*nu) - 1)
