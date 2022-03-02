import numpy as np
import warnings
import pkg_resources

from astropy.time import Time
from astropy.coordinates import Angle
from scipy import stats, special
from astropy.table import Table
from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionJavaTool, DataParser


class Hipparcos2Recalibrated(HipparcosRereductionJavaTool):
    """
    A parser class which re-calibrates the hipparcos 2 data according to Brandt et al. 2022

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

    def parse(self, star_id, intermediate_data_directory, attempt_adhoc_rejection=True,
              reject_known=True, **kwargs):
        # important that the 2007 error inflation is turned off (error_inflate=False)
        header, raw_data = super(Hipparcos2Recalibrated, self).parse(star_id, intermediate_data_directory,
                                                                     error_inflate=False,
                                                                     attempt_adhoc_rejection=attempt_adhoc_rejection,
                                                                     reject_known=reject_known, **kwargs)
        if self.meta['catalog_soltype'] != 5:
            warnings.warn('This source has a solution type with {} free parameters. Only five-parameter sources can be '
                          'recalibrated currently. No recalibration will be performed.')
            return header, raw_data
        else:
            # apply the calibrations
            self.residuals += self.residual_offset
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
            # update the header with the new parameters and errors.
            ntransits, nparam = len(self), 5
            header['first']['F2'] = special.erfcinv(stats.chi2.sf(Q, ntransits - nparam)*2)*np.sqrt(2)
            import pdb; pdb.set_trace()

            raw_data = None  # because we do not recalibrate the raw data, so to protect the user we delete it.
        return header, raw_data


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

