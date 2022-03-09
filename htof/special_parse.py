import numpy as np
import warnings
import pkg_resources
from pandas import DataFrame, Series
from shutil import copy

from astropy.time import Time
from astropy.coordinates import Angle
from scipy import stats, special
from astropy.table import Table
from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionJavaTool


class Hipparcos2Recalibrated(HipparcosRereductionJavaTool):
    """
    A parser class which re-calibrates the hipparcos 2 data according to Brandt et al. 2022

    This is a monstrous parser class because well it is doing a lot.

    Note that this "Parser" class actually refits the data. This is programmatically poorly set up,
    in the sense that we parse the data and fit it, mixing this class with an AstrometricFitter class... For now,
    this is OK because the objective of this class is mainly so other users can save and inspect the recalibrated
    hipparcos 2 IAD.

    TODO: need to update raw_data, instead of just setting it to None.

    """
    EPOCHREJECTLIST = Table.read(pkg_resources.resource_filename('htof',
                                                                 'data/epoch_reject_shortlist.csv'), format='ascii')

    def __init__(self, scan_angle=None, epoch=None, residuals=None, inverse_covariance_matrix=None,
                 along_scan_errs=None, meta=None, residual_offset=0.145, cosmic_dispersion=2.15):
        super(Hipparcos2Recalibrated, self).__init__(scan_angle=scan_angle, along_scan_errs=along_scan_errs,
                                                     epoch=epoch, residuals=residuals,
                                                     inverse_covariance_matrix=inverse_covariance_matrix,
                                                     meta=meta)
        self.residual_offset = residual_offset
        self.cosmic_dispersion = cosmic_dispersion
        self.recalibrated_header = None
        self.recalibrated_data = None

    def parse(self, star_id, intermediate_data_directory, attempt_adhoc_rejection=True,
              reject_known=True, **kwargs):
        # important that the 2007 error inflation is turned off (error_inflate=False)
        header, raw_data = super(Hipparcos2Recalibrated, self).parse(star_id, intermediate_data_directory,
                                                                     error_inflate=False,
                                                                     attempt_adhoc_rejection=attempt_adhoc_rejection,
                                                                     reject_known=reject_known, **kwargs)
        if self.meta['catalog_soltype'] != 5:
            warnings.warn(f'This source has a solution type with {self.meta["catalog_soltype"]} free parameters. '
                          f'Only five-parameter sources can be '
                          'recalibrated currently. No recalibration will be performed.')
            return header, raw_data
        elif attempt_adhoc_rejection is False or reject_known is False:
            warnings.warn('We have not tested recalibration without rejecting any of the flagged or bugged '
                          'observations. No recalibration will be performed.')
            return header, raw_data
        else:
            # apply the calibrations
            self.residuals += self.residual_offset  # note that this modifies the raw_data column also.
            self.along_scan_errs = np.sqrt(self.along_scan_errs**2 + self.cosmic_dispersion**2)
            self.calculate_inverse_covariance_matrices()
            # munge the parallax factors into the correct form. Note that we are using
            # the parallax factors from the catalog here to keep everything consistent.
            ra_motion, dec_motion = to_ra_dec_basis(self.parallax_factors.values, self.scan_angle.values)
            parallactic_perturbations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}
            # refit the data, calculate the new residuals, and parameters.
            fitter = AstrometricFitter(inverse_covariance_matrices=self.inverse_covariance_matrix,
                                       epoch_times=Time(Time(self.julian_day_epoch(), format='jd'), format='jyear').value,
                                       central_epoch_dec=1991.25,
                                       central_epoch_ra=1991.25,
                                       fit_degree=1,
                                       use_parallax=True,
                                       parallactic_pertubations=parallactic_perturbations)
            # get residuals in ra and dec.
            ra = Angle(self.residuals.values * np.sin(self.scan_angle.values), unit='mas')
            dec = Angle(self.residuals.values * np.cos(self.scan_angle.values), unit='mas')
            # fit the residuals
            coeffs, errors, Q, new_residuals = fitter.fit_line(ra.mas, dec.mas, return_all=True)
            # compute the along-scan residuals
            new_residuals = to_along_scan_basis(new_residuals[:, 0], new_residuals[:, 1], self.scan_angle.values)
            self.residuals = Series(new_residuals, index=self.residuals.index)
            """
            update the header with the new statistics
            """
            ntransits, nparam = len(self), 5
            header['second']['NOB'] = len(self)
            header['second']['NR'] = 0  # because we automatically remove any "flagged as rejected" observations.
            header['first']['F1'] = 0
            header['first']['NRES'] = len(self)
            header['first']['F2'] = special.erfcinv(stats.chi2.sf(Q, ntransits - nparam)*2)*np.sqrt(2)
            if np.isfinite(header['first']['F2']):
                header['first']['F2'] = np.round(header['first']['F2'], 4)
            # update the best fit parameters with the new values
            for i, key, dp, in zip(np.arange(5), ['Plx', 'RAdeg', 'DEdeg', 'pm_RA', 'pm_DE'], [2, 8, 8, 2, 2]):
                header['third'][key] = np.round(header['third'][key] + coeffs[i], dp)
            # update the errors with the new errors
            for i, key, dp, in zip(np.arange(5), ['e_Plx', 'e_RA', 'e_DE', 'e_pmRA', 'e_pmDE'], [2, 2, 2, 2, 2]):
                header['third'][key] = np.round(errors[i], dp)
            # save the modified header to the class, because these will be
            # used by self.write_as_javatool_format()
            self.recalibrated_header = header
            """
            update the raw_data columns with the new data. Note that rejected/bugged epochs are already
            taken care of.
            """
            # data order in Java tool data: IORB   EPOCH    PARF    CPSI    SPSI     RES   SRES
            recalibrated_data = DataFrame({'1': self._iorb, '2': self._epoch - 1991.25, '3': self.parallax_factors,
                                       '4': self._cpsi, '5': self._spsi, '6': np.round(self.residuals.values, 3),
                                       '7': np.round(self.along_scan_errs.values, 3)})
            self.recalibrated_data = recalibrated_data
            return None, None  # the raw header and raw data have been modified, so do not return them.

    def write_as_javatool_format(self, path: str):
        """
        Variant of the .write() method that is specially for this Parser. It writes out the fixed data
        with the exact same format as the input JavaTool IAD (i.e., the .d files).
        We recommend using the normal .write() format. This method is for users who do not want to adopt the
        htof file format output by the normal .write() format.

        :param: path: path to write the file to. filename should end with .d . ".d" is the same file ending as the
        java tool IAD. But in principle, most common ascii file extensions should work (e.g., ".txt" etc).
        """
        if self.recalibrated_header is None or self.recalibrated_data is None:
            warnings.warn('This source was NOT recalibrated, see earlier warnings as to why. Will not save any '
                          f'output file at {path} because no recalibration was done.')
            return None
        ks1 = ['HIP', 'MCE', 'NRES', 'NC', 'isol_n', 'SCE', 'F2', 'F1']
        ks2 = ['Hp', 'B-V', 'VarAnn', 'NOB', 'NR']
        ks3 = ['RAdeg', 'DEdeg', 'Plx', 'pm_RA', 'pm_DE', 'e_RA', 'e_DE', 'e_Plx', 'e_pmRA', 'e_pmDE',
               'dpmRA', 'dpmDE', 'e_dpmRA', 'e_dpmDE', 'ddpmRA', 'ddpmDE', 'e_ddpmRA', 'e_ddpmDE',
               'upsRA', 'upsDE', 'e_upsRA', 'e_upsDE', 'var']
        header_template_fpath = pkg_resources.resource_filename('htof', 'data/hip2_recalibrated_header.txt')
        copy(header_template_fpath, path)  # copy the template file to the output path.
        # populate the header lines.
        f = open(path, 'r')
        lines = f.readlines()
        for idx, hline, ks in zip([6, 8, 10], ['first', 'second', 'third'], [ks1, ks2, ks3]):
            matter = "  ".join([str(self.recalibrated_header[hline][key]) for key in ks])
            if hline == 'third':
                matter = "  ".join([str(self.recalibrated_header[hline][key]) for key in ks[:3]])
                matter += "  " + " ".join([" {:<6}".format(self.recalibrated_header[hline][key]) for key in ks[3:]])
            lines[idx] = '# ' + matter + '\n'
            lines[idx] = lines[idx].replace('nan', '---')
        # populate the IAD entries
        lines[-1] += '\n'
        for line in self.recalibrated_data.to_numpy():
            # fixed width formatting
            vals_to_write = ["{0: .4f}".format(i) for i in line[1:]]
            line = '   {0: <5}'.format(int(line[0])) + "  ".join(vals_to_write) + "\n"
            lines.append(line)
        f = open(path, 'w')
        f.writelines(lines)
        f.close()
        return None


def to_ra_dec_basis(value, scan_angle):
    """
    Convert values (e.g. residuals) along the direction of the scan to the same value in RA and Dec. I.e. assume the value
    has a zero across_scan component and all of the value is in the along-scan direction, then convert that value to a vector in
    RA and Dec.
    these maths are just from https://en.wikipedia.org/wiki/Rotation_of_axes  .
    """
    dec_value, ra_value = value * np.cos(scan_angle), value * np.sin(scan_angle)
    return ra_value, dec_value


def to_along_scan_basis(ra_value, dec_value, scan_angle):
    """
    Convert values (e.g. residuals) in RA and Dec to the same value along the direction of the scan. I.e convert from RA, DEC to
    Along-scan , Across-scan basis, then keep only the along-scan component.
    these maths are just from https://en.wikipedia.org/wiki/Rotation_of_axes  .

    note that the inputs ra_value and dec_value may have nonzero across scan components. This function will zero
    out the cross-scan component. to_ra_dec_basis and to_along_scan_basis() are ONLY inverse transforms for data that
    has components solely in the across-scan direction.
    """
    along_scan_value = dec_value * np.cos(scan_angle) + ra_value * np.sin(scan_angle)
    return along_scan_value

