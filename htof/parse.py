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
import warnings
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
                 along_scan_errs=None):
        self.scan_angle = pd.Series(scan_angle, dtype=np.float64)
        self._epoch = pd.DataFrame(epoch, dtype=np.float64)
        self.residuals = pd.Series(residuals, dtype=np.float64)
        self.along_scan_errs = pd.Series(along_scan_errs, dtype=np.float64)
        self.inverse_covariance_matrix = inverse_covariance_matrix
        self._rejected_epochs = {}

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

    @property
    def rejected_epochs(self):
        return self._rejected_epochs

    @rejected_epochs.setter
    def rejected_epochs(self, value):
        residuals_to_reject, orbits_to_reject = value['residual/along_scan_error'], value['orbit/scan_angle/time']
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, residuals_to_reject, False)
        self.residuals, self.along_scan_errs = self.residuals[not_outlier], self.along_scan_errs[not_outlier]
        not_outlier = np.ones(len(self), dtype=bool)
        np.put(not_outlier, orbits_to_reject, False)
        self._epoch, self.scan_angle = self._epoch[not_outlier], self.scan_angle[not_outlier]
        self._rejected_epochs = value

    def __add__(self, other):
        all_scan_angles = pd.concat([self.scan_angle, other.scan_angle])
        all_epoch = pd.concat([pd.DataFrame(self.julian_day_epoch()), pd.DataFrame(other.julian_day_epoch())])
        all_residuals = pd.concat([self.residuals, other.residuals])
        all_along_scan_errs = pd.concat([self.along_scan_errs, other.along_scan_errs])

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
                 along_scan_errs=None):
        super(HipparcosRereductionDVDBook, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                          epoch=epoch, residuals=residuals,
                                                          inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, error_inflate=True, header_rows=1, **kwargs):
        """
        :param: star_id:
        :param: intermediate_data_directory:
        :param: error_inflate: True if the along-scan errors are to be corrected by the inflation factor
        according to equation B.1 of D. Michalik et al. 2014. Only turn this off for tests, or if the parameters
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
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = sin(psi), data[4] = cos(psi)
        self._epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)
        self.along_scan_errs = data[6]  # unit milli-arcseconds (mas)
        n_transits, nparam, catalog_f2, percent_rejected = header.iloc[0][2], get_nparam(header.iloc[0][4]), header.iloc[0][6], header.iloc[0][7]
        if percent_rejected > 0:
            # TODO there are 10,000 sources with more than one rejected epoch, and only 700 of them have
            # this write out bug. We could try and reverse engineer the rejected epochs (like we did before)
            # for any sources that do not have the write out bug.
            # we should probably do this, since I imagnie we will have ~15000 or so sources that we cannot refit
            # from the DVD if we do not do anything.
            warnings.warn(f"You are parsing a DVD source that has {percent_rejected} percent "
                          f"rejected observations. Note that the write out bug, plus the issue that "
                          f"rejected observations are not indicated, makes it difficult to algorithmically reject "
                          f"the correct epochs and assign residuals correctly. The IAD for this source will "
                          f"not reflect the best-fit solution. Please use the java tool data instead, or correct "
                          f"the data file yourself.", UserWarning)
        if error_inflate:
            # adjust the along scan errors so that the errors on the best fit parameters match the catalog.
            self.along_scan_errs *= self.error_inflation_factor(n_transits, nparam, catalog_f2)
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


class HipparcosRereductionJavaTool(HipparcosRereductionDVDBook):
    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None):
        super(HipparcosRereductionJavaTool, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                         epoch=epoch, residuals=residuals,
                                                         inverse_covariance_matrix=inverse_covariance_matrix)

    def parse(self, star_id, intermediate_data_directory, attempt_adhoc_rejection=True, **kwargs):
        # TODO set error error_inflate=True when the F2 value is available in the headers of 2.1 data.
        header, raw_data = super(HipparcosRereductionJavaTool, self).parse(star_id, intermediate_data_directory,
                                                                    error_inflate=False, header_rows=5,
                                                                    attempt_adhoc_rejection=False)
        n_transits, n_expected_transits = header.iloc[1][4], header.iloc[0][2]
        n_additional_reject = n_transits - n_expected_transits
        print(n_additional_reject)
        if attempt_adhoc_rejection and n_additional_reject > 0:
            epochs_to_reject = find_epochs_to_reject_java(self, n_additional_reject)
        if not attempt_adhoc_rejection and n_additional_reject > 0:
            warnings.warn("attempt_adhoc_rejection = False and this is a bugged source. "
                          "You are foregoing the write out bug "
                          "correction for this Java tool source. The IAD will not correspond exactly "
                          "to the best fit solution. ", UserWarning)
        if n_additional_reject == 0:
            epochs_to_reject = np.where(self.along_scan_errs < 0)[0]
            epochs_to_reject = {'residual/along_scan_error': list(epochs_to_reject),
                                'orbit/scan_angle/time': list(epochs_to_reject)}
        self.rejected_epochs = epochs_to_reject
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
    orbit_reject_idx = []  # index of the orbit/scan_angle/time set to throw out.

    # need to iterate over popping orbit combinations
    # then after popping, set the contributions to 0 of those rows with negative AL errors
    # then calculate chisquared, saving that chisquared value.
    # take the best chisquared value
    # change rejected epochs to additionl rejected epochs
    # in the @setter for rejected epochs, delete the ones with negative AL errors. or something.
    orbits_to_keep = np.ones(len(data), dtype=bool)
    orbit_combinations = list(set(itertools.combinations(possible_rejects, int(n_additional_reject))))
    #
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False
    known_rejected_residuals = (data.residuals.values < 0).astype(bool)
    # we should be able to do the orbit reject calculation fairly easily in memory.
    # for 100 choose 3 we have like 250,000 combinations of orbits -- we sghould be able to
    # do those in 10,000 orbit chunks in memory and gain a factor of 10,000 speed up.
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []
    for orbit_to_reject in orbit_combinations:
        orbits_to_keep[list(orbit_to_reject)] = False
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.

        # this pops the orbits out and shifts all the orbits after upwards:
        orbit_factors = np.array([sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T[orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
        chi2_vector = (2 * residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector * (known_rejected_residuals[residuals_to_keep]).reshape(-1, 1),
                                                        axis=0) ** 2))
        candidate_orbit_rejects.append(orbit_to_reject)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
            # this is a good enough stationary point and also matches the f2 value (by construction)
            # that this combination is probably the right combination.
        # reset for the next loop:
        orbits_to_keep[list(orbit_to_reject)] = True

    orbit_reject_idx = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    if np.min(candidate_orbit_chisquared_partials) > 5:
        warnings.warn("Attempted to fix the write out bug, but this is one of the few sources that "
                      "htof cannot fix. ", UserWarning)

    return {'residual/along_scan_error': list(resid_reject_idx), 'orbit/scan_angle/time': list(orbit_reject_idx)}


def get_nparam(nparam_header_val):
    # strip the solution type (5, 7, or 9) from the solution type, which is a number 10xd+s consisting of
    # two parts: d and s. see Note 1 on Vizier for the Hipparcos re-reduction.
    return int(str(int(nparam_header_val))[-1])


def compute_f2(nu, chisquared):
    # equation B.2 of D. Michalik et al. 2014. Joint astrometric solution of Hipparcos and Gaia
    return (9*nu/2)**(1/2)*((chisquared/nu)**(1/3) + 2/(9*nu) - 1)
