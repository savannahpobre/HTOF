import numpy as np
import warnings
import pkg_resources

from astropy.time import Time
from astropy.table import Table
from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionJavaTool


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
            # refit the data, calculate the new residuals, and parameters.
            self.calculate_inverse_covariance_matrices()
            pf = self.parallax_factors
            ra_pert = None
            dec_pert = None
            parallactic_perturbations = {'ra_plx': ra_pert, 'dec_plx': dec_pert}
            fitter = AstrometricFitter(inverse_covariance_matrices=self.inverse_covariance_matrix,
                                       epoch_times=Time(Time(self.julian_day_epoch(), format='jd'), format='jyear').value,
                                       central_epoch_dec=1991.25,
                                       central_epoch_ra=1991.25,
                                       fit_degree=1,
                                       use_parallax=True,
                                       parallactic_pertubations=parallactic_perturbations)

            # update the header with the new parameters

            raw_data = None  # because we do not recalibrate the raw data, so to protect the user we delete it.
        return header, raw_data


def along_scan_residuals(ra_residual, dec_residual, scan_angle):
    """
    Convert residuals in RA and Dec to a residual along the direction of the scan. I.e convert from RA, DEC to
    Along-scan , Across-scan basis, then keep only the along-scan component.
    these maths are just from https://en.wikipedia.org/wiki/Rotation_of_axes  .
    """
    along_scan_residual = dec_residual * np.cos(scan_angle) + ra_residual * np.sin(scan_angle)
    return along_scan_residual


def calculate_new_residuals(data: Hipparcos2Recalibrated, depoch, dplx, dra, ddec, dmura, dmudec):
    """
    Recomputes new along-scan residuals, expressed against a perturbed five-parameter catalog solution. If
    the catalog parallax is plx, the catalog ra and dec are ra and dec, and the catalog proper motions are mudec and mura,
    then this function will return new residuals expressed against:
    [plx + dplx, ra + dra, dec + ddec, mudec + dmudec, mura + dmura]

    :param depoch: the observation epoch in years - 1991.25
    :param dplx: the change to the catalog parallax in mas (milli-arcseconds)
    :param dra: the change to the central ra in mas
    :param ddec: the change to the central dec in mas
    :param dmura: the change to the proper motion in ra in mas/yr
    :param dmudec: the change to the proper motion in dec in mas/yr
    """
    ra_residuals = data.residuals * np.sin(data.scan_angle) - dra - dmura * depoch
    dec_residuals = data.residuals * np.cos(data.scan_angle) - ddec - dmudec * depoch
    new_residuals = along_scan_residuals(ra_residuals, dec_residuals, data.scan_angle) - dplx * data.parallax_factors
    return new_residuals
