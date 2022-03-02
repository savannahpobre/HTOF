import numpy as np
from astropy.coordinates import Angle
from htof.special_parse import to_ra_dec_basis, to_along_scan_basis, Hipparcos2Recalibrated
from htof.main import Astrometry
import pytest


class TestHip2RecalibratedParser:
    def test_parse_with_no_changes(self):
        data = Hipparcos2Recalibrated(cosmic_dispersion=0, residual_offset=0)
        data.parse('27321', 'htof/test/data_for_tests/Hip21/')
        # todo test that the new residuals are equal to the old residuals.

    def test_parse(self):
        data = Hipparcos2Recalibrated()
        data.parse('27321', 'htof/test/data_for_tests/Hip21/')

@pytest.mark.e2e
class TestParallaxFactors:
    """
    Tests the computation of the parallax factors anew, as well as tests the basis transformations from along-scan
    to ra and dec.
    """
    # Hip 27321 parameters from the Hipparcos 1 catalogue via Vizier
    cntr_ra, cntr_dec = Angle(86.82118054, 'degree'), Angle(-51.06671341, 'degree')
    # generate fitter and parse intermediate data
    astro = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=1991.25,
                       central_epoch_dec=1991.25, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec)
    ra_motion = astro.fitter.parallactic_pertubations['ra_plx']
    dec_motion = astro.fitter.parallactic_pertubations['dec_plx']
    along_scan_parallax_factors = astro.data.parallax_factors.values
    scan_angle = astro.data.scan_angle.values

    def test_parallax_factors_forward_transform(self):
        assert np.allclose(self.along_scan_parallax_factors,
                           to_along_scan_basis(self.ra_motion, self.dec_motion, self.scan_angle), atol=0.03)

    def test_parallax_factors_backward_transform(self):
        # note that this is redundant with test_basis_change_consistency()
        ra_motion, dec_motion = to_ra_dec_basis(self.along_scan_parallax_factors, self.scan_angle)
        # convert the calculated parallax factors to the along-scan basis and back to null out the AC component.
        al_parallax_factor = to_along_scan_basis(self.ra_motion, self.dec_motion, self.scan_angle)
        ra_motion_acnull, dec_motion_acnull = to_ra_dec_basis(al_parallax_factor, self.scan_angle)
        assert np.allclose(ra_motion, ra_motion_acnull, atol=0.03)
        assert np.allclose(dec_motion, dec_motion_acnull, atol=0.03)


def test_basis_change_consistency():
    """
    Tests that to_ra_dec_basis() and to_along_scan_basis() are inverse transforms in the case where
    the data being transformed has a zero across-scan component.

    Note that these two functions are NOT inverse transforms if the data has a non-zero across scan component.
    """
    al_value = np.array([-5, 0, 1, 5, 10] * 100)  # [-5, 0, 1, 5, 10, -5, 0, 1, 5, 10, -5, 0,...]
    scan_angle = np.array([[i]*5 for i in np.linspace(-2*np.pi, 2*np.pi, 100)]).flatten()
    ra_val, dec_val = to_ra_dec_basis(al_value, scan_angle)
    al_value_check = to_along_scan_basis(ra_val, dec_val, scan_angle)
    assert np.allclose(al_value_check, al_value)
