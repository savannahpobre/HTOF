import numpy as np
from astropy.coordinates import Angle
from htof.special_parse import to_ra_dec_basis, to_along_scan_basis, Hipparcos2Recalibrated, \
    Hipparcos2ParserFactory, get_datatype
from htof.parse import HipparcosRereductionJavaTool, HipparcosRereductionDVDBook
from htof.main import Astrometry
import pytest
import os
import tempfile


class TestHip2RecalibratedParser:
    def test_parse_residuals_with_no_changes(self):
        data_recalb = Hipparcos2Recalibrated(cosmic_dispersion=0, residual_offset=0)
        data_recalb.parse('27321', 'htof/test/data_for_tests/Hip21/')
        data = HipparcosRereductionJavaTool()
        data.parse('27321', 'htof/test/data_for_tests/Hip21/', error_inflate=False)
        # test that the new residuals are equal to the old residuals, within 0.01 (round off).
        assert np.allclose(data_recalb.residuals, data.residuals, atol=0.01)
        assert np.allclose(data_recalb.along_scan_errs, data.along_scan_errs, atol=0.01)

    def test_recal_residuals(self):
        # this tests that a handful of residuals are equal to those calculated by hand
        # when we do the recalibration. This assumes the default recalibration parameters of 2.15 CD and 0.145
        # residual offset, so this test will fail if those default params change. This is by design.
        data = Hipparcos2Recalibrated()
        data.parse('27321', 'htof/test/data_for_tests/Hip21/')
        comparison = [-0.12720199, -0.81727459, -1.60732196, 0.2189357, -0.35112427, -2.77118401]
        resids = np.hstack([data.residuals.values[:3], data.residuals.values[-3:]])
        assert np.allclose(comparison, resids, atol=0.001)

    def test_recal_parameters(self):
        """
        hip 27321 Java tool params
        # RAdeg        DEdeg        Plx      pm_RA    pm_DE    e_RA   e_DE   e_Plx  e_pmRA e_pmDE
        # 86.82118073  -51.06671341 51.44    4.65     83.10    0.10   0.11   0.12   0.11   0.15
        """
        # this tests that the astrometric parameters are identical to those calculated by hand.
        # when we do the recalibration
        data = Hipparcos2Recalibrated()
        data.parse('27321', 'htof/test/data_for_tests/Hip21/')
        params = [data.recalibrated_header['third'][key] for key in ['Plx', 'RAdeg', 'DEdeg', 'pm_RA', 'pm_DE']]
        param_errors = [data.recalibrated_header['third'][key] for key in ['e_Plx', 'e_RA', 'e_DE', 'e_pmRA', 'e_pmDE']]

        comparison = np.array([-0.02631759, 0.01228683, -0.00729088, 0.05298548, 0.01326707])
        comparison += np.array([51.44, 86.82118073, -51.06671341, 4.65, 83.10])
        comparison_errors = np.array([0.34975472, 0.3020452, 0.33768252, 0.33901742, 0.45255402])
        assert np.allclose(params, comparison, atol=0.01)
        assert np.allclose(param_errors, comparison_errors, atol=0.01)

    @pytest.mark.e2e
    def test_parse_and_write_recalibrated_data(self):
        hip_ids = [39, 651, 4427, 17447, 21000, 27100, 27321,
                   37515, 44050, 94046, 94312, 114114, 10160, 581]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for hip_id in hip_ids:
                data = Hipparcos2Recalibrated()
                data.parse(hip_id, 'htof/test/data_for_tests/Hip21/')
                outpath = os.path.join(tmp_dir, f'{hip_id}_recalibrated.d')
                data.write_as_javatool_format(outpath)

    def test_parse_and_write_invalid_recalibrated_data(self):
        data = Hipparcos2Recalibrated()
        data.parse(27321, 'htof/test/data_for_tests/Hip21/')
        data.recalibrated_data = None
        out = data.write_as_javatool_format('')
        assert out is None

    def test_invalid_recalibration(self):
        data = Hipparcos2Recalibrated()
        header, raw_data = data.parse(27321, 'htof/test/data_for_tests/Hip21/', reject_known=False)
        assert header is not None  # if header and raw data (both not None), are returned, then
        # the recalibration was NOT done.
        # testing that a solution type VIM (soltype integer 3) is NOT recalibrated.
        header, raw_data = data.parse(999999, 'htof/test/data_for_tests/Hip21/')
        assert header is not None

    @pytest.mark.e2e
    def test_write_and_read_recalibrated_data(self):
        data = Hipparcos2Recalibrated()
        data.parse('27321', 'htof/test/data_for_tests/Hip21/')
        with tempfile.TemporaryDirectory() as tmp_dir:
            outpath = os.path.join(tmp_dir, '27321_recalibrated.d')
            data.write_as_javatool_format(outpath)
            reloaded_data = HipparcosRereductionJavaTool()
            reloaded_data.parse('27321', tmp_dir, error_inflate=False, attempt_adhoc_rejection=False, reject_known=False)
            assert np.allclose(reloaded_data.residuals, data.residuals, atol=0.001)
            assert np.allclose(reloaded_data.along_scan_errs, data.along_scan_errs, atol=0.001)
            assert np.allclose(reloaded_data._iorb, data._iorb)


class TestHipparcos2ParserFactory:
    factory = Hipparcos2ParserFactory

    def test_get_datatype(self):
        dtype = get_datatype('htof/test/data_for_tests/Hip21/IntermediateData/H027321.d')
        assert dtype == 'hip2javatool'
        dtype = get_datatype('htof/test/data_for_tests/Hip2/IntermediateData/HIP027321.d')
        assert dtype == 'hip2dvd'

    def test_get_appropriate_parser(self):
        parser = self.factory.get_appropriate_parser('htof/test/data_for_tests/Hip21/IntermediateData/H027321.d')
        assert parser is HipparcosRereductionJavaTool
        parser = self.factory.get_appropriate_parser('htof/test/data_for_tests/Hip2/IntermediateData/HIP027321.d')
        assert parser is HipparcosRereductionDVDBook

    @pytest.mark.e2e
    def test_parse_and_instantiate_hip2dvd(self):
        star_id, iad_dir = '27321', 'htof/test/data_for_tests/Hip2/IntermediateData/'
        parser = self.factory.parse_and_instantiate(star_id, iad_dir)
        assert type(parser) is HipparcosRereductionDVDBook
        comparison_parser = HipparcosRereductionDVDBook.parse_and_instantiate(star_id, iad_dir)
        assert np.allclose(comparison_parser.scan_angle, parser.scan_angle)
        assert np.allclose(comparison_parser.residuals, parser.residuals)

    @pytest.mark.e2e
    def test_parse_and_instantiate_hip2javatool(self):
        star_id, iad_dir = '27321', 'htof/test/data_for_tests/Hip21/IntermediateData/'
        parser = self.factory.parse_and_instantiate(star_id, iad_dir)
        assert type(parser) is HipparcosRereductionJavaTool
        comparison_parser = HipparcosRereductionJavaTool.parse_and_instantiate(star_id, iad_dir)
        assert np.allclose(comparison_parser.scan_angle, parser.scan_angle)
        assert np.allclose(comparison_parser.residuals, parser.residuals)


@pytest.mark.e2e
class TestParallaxFactorsGaia:
    """
    Tests the computation of the parallax factors anew, as well as tests the basis transformations from along-scan
    to ra and dec.
    """
    # Hip 27321 parameters from the Gaia EDR3 archive.
    cntr_ra, cntr_dec = Angle(86.82123452009108, 'degree'), Angle(-51.066136257823345, 'degree')
    # generate fitter and parse intermediate data
    astro = Astrometry('Gaiaedr3', '27321', 'htof/test/data_for_tests/GaiaeDR3', central_epoch_ra=2016,
                       central_epoch_dec=2016, format='jyear', fit_degree=1, use_parallax=True,
                       central_ra=cntr_ra, central_dec=cntr_dec)
    ra_motion = astro.fitter.parallactic_pertubations['ra_plx']
    dec_motion = astro.fitter.parallactic_pertubations['dec_plx']
    along_scan_parallax_factors = astro.data.parallax_factors.values
    scan_angle = astro.data.scan_angle.values

    def test_new_computed_parallax_factors_agree_with_scanninglaw(self):
        # test that the newly computed parallax factors agree with
        # parallaxFactorAlongScan from the GOST data.
        assert np.allclose(self.along_scan_parallax_factors,
                           to_along_scan_basis(self.ra_motion, self.dec_motion, self.scan_angle), atol=0.03)


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

    def test_new_computed_parallax_factors_agree_with_catalog(self):
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

    def test_parallax_factors_independent_of_reference_epoch(self):
        astron = Astrometry('Hip1', '27321', 'htof/test/data_for_tests/Hip1', central_epoch_ra=2000,
                           central_epoch_dec=2000, format='jyear', fit_degree=1, use_parallax=True,
                           central_ra=self.cntr_ra, central_dec=self.cntr_dec)
        ra_motionn = astron.fitter.parallactic_pertubations['ra_plx']
        dec_motionn = astron.fitter.parallactic_pertubations['dec_plx']
        assert np.allclose(to_along_scan_basis(self.ra_motion, self.dec_motion, self.scan_angle),
                           to_along_scan_basis(ra_motionn, dec_motionn, self.scan_angle))


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
