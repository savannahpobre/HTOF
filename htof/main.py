"""
Driver script for htof.
The Astrometry class is what a user should use to both parse and fit the intermediate astrometric data.

Author: G. Mirek Brandt
"""

from astropy.time import Time
from astropy.coordinates import Angle
import warnings

from htof.fit import AstrometricFitter
from htof.special_parse import to_ra_dec_basis, Hipparcos2Recalibrated, Hipparcos2ParserFactory
from htof.parse import GaiaeDR3, GaiaDR2, GaiaData
from htof.parse import HipparcosOriginalData, HipparcosRereductionJavaTool, HipparcosRereductionDVDBook
from htof.sky_path import parallactic_motion, earth_ephemeris, earth_sun_l2_ephemeris


class Astrometry(object):
    """
    General wrapper for the different fitting and parsing classes.

    :param use_catalog_parallax_factors: True if you want to load and use the hipparcos catalog parallax factors given
    with the IAD, or the gaia parallax factors shipped with the GOST scanning law.
    Set to False if you want to recompute the parallax factors. Default is False.

    data_choice hip2or21 will attempt to see if the input data file is Hip2 (DVD) or Hip21 (java tool), then proceed accordingly.
    """
    parsers = {'gaiaedr3': GaiaeDR3, 'gaiadr2': GaiaDR2, 'gaia': GaiaData, 'hip21': HipparcosRereductionJavaTool,
               'hip1': HipparcosOriginalData, 'hip2': HipparcosRereductionDVDBook, 'hip2or21': Hipparcos2ParserFactory,
               'hip2recalibrated': Hipparcos2Recalibrated}
    ephemeri = {'gaiadr2': earth_sun_l2_ephemeris, 'gaia': earth_sun_l2_ephemeris, 'gaiaedr3': earth_sun_l2_ephemeris,
                'hip1': earth_ephemeris, 'hip2': earth_ephemeris, 'hip21': earth_ephemeris, 'hip2or21': earth_ephemeris,
                'hip2recalibrated': Hipparcos2Recalibrated}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None,
                 central_epoch_ra=0, central_epoch_dec=0, format='jd', fit_degree=1,
                 use_parallax=False, central_ra=None, central_dec=None,
                 use_catalog_parallax_factors=False, along_scan_error_scaling=1.0, **kwargs):
        self.along_scan_error_scaling = along_scan_error_scaling
        if data_choice.lower() == 'hip2recalibrated':
            warnings.warn(f'You have selected {data_choice}, the recalibrated Hipparcos 2 data. Note that for this,'
                          f' you should be feeding in the filepaths to the Hip21 (Hip2 java tool data), because'
                          f' htof applies the recalibration on-the-fly for each file. As well, be sure to read'
                          f' Brandt et al. 2022 to understand the limitations of using the recalibrated data. ')     # pragma: no cover

        if data is None:
            ThisDataParser = self.parsers[data_choice.lower()]
            data = ThisDataParser.parse_and_instantiate(star_id=star_id,
                                                        intermediate_data_directory=intermediate_data_directory)
            data.scale_along_scan_errs(self.along_scan_error_scaling)
            data.calculate_inverse_covariance_matrices()

        parallactic_pertubations = None
        if use_parallax:
            if not use_catalog_parallax_factors:
                # recompute the parallax factors at the new central_ra and central_ra epoch.
                if not (isinstance(central_ra, Angle) and isinstance(central_dec, Angle)):
                    raise ValueError('Cannot compute parallax factors. central_ra and central_dec must be instances'
                                     ' of astropy.coordinates.Angle.')     # pragma: no cover
                if central_epoch_dec != central_epoch_ra:
                    warnings.warn('central_epoch_dec != central_epoch_ra. '
                                  'Using central_epoch_ra as the central_epoch to compute the parallax motion.',
                                  UserWarning)    # pragma: no cover
                ra_motion, dec_motion = parallactic_motion(Time(data.julian_day_epoch(), format='jd').jyear,
                                                           central_ra.mas, central_dec.mas, 'mas',
                                                           Time(central_epoch_ra, format=format).jyear,
                                                           ephemeris=self.ephemeri[data_choice.lower()])
            else:
                ra_motion, dec_motion = to_ra_dec_basis(data.parallax_factors.values, data.scan_angle.values)
            parallactic_pertubations = {'ra_plx': ra_motion, 'dec_plx': dec_motion}

        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=Time(Time(data.julian_day_epoch(), format='jd'), format=format).value,
                                       central_epoch_dec=Time(central_epoch_dec, format=format).value,
                                       central_epoch_ra=Time(central_epoch_ra, format=format).value,
                                       fit_degree=fit_degree,
                                       use_parallax=use_parallax,
                                       parallactic_pertubations=parallactic_pertubations)
        self.data = data
        self.fitter = fitter

    def fit(self, ra_vs_epoch, dec_vs_epoch, return_all=False):
        return self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch, dec_vs_epoch=dec_vs_epoch, return_all=return_all)

    def optimal_central_epochs(self):
        return {'ra': self.fitter.find_optimal_central_epoch('ra'), 'dec': self.fitter.find_optimal_central_epoch('dec')}
