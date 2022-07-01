import pytest
import mock
import shutil
import tempfile
import os
import glob
from htof.utils.ftp import download_and_save_hip21_data_to


class TestFTP:
    def test_oninvalid_hipid(self):
        with pytest.raises(RuntimeError):
            download_and_save_hip21_data_to(2222222222, outpath='')

    @mock.patch('htof.utils.ftp.FTP_TLS', autospec=True)
    def test_successful_ftp(self, mock_ftp):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outpath = os.path.join(tmp_dir, 'H000581.d')
            mock_ftp.return_value = MockFTP(login_error=False, outpath=outpath)
            download_and_save_hip21_data_to(581, outpath)
            assert os.path.exists(outpath)

    @mock.patch('htof.utils.ftp.FTP_TLS', autospec=True)
    def test_on_login_error(self, mock_ftp):
        mock_ftp.return_value = MockFTP(login_error=True)
        with pytest.raises(RuntimeError):
            download_and_save_hip21_data_to(581, outpath='')

    @mock.patch('htof.utils.ftp.FTP_TLS', autospec=True)
    def test_on_retrbinary_error(self, mock_ftp):
        with tempfile.TemporaryDirectory() as tmp_dir:
            outpath = os.path.join(tmp_dir, 'H000581.d')
            mock_ftp.return_value = MockFTP(login_error=False, outpath=outpath)
            with pytest.raises(RuntimeError):
                # hip 10 is a file that DOES NOT exist in 'htof/test/data_for_tests/Hip21/IntermediateData'
                # if H000010.d is ever added to 'htof/test/data_for_tests/Hip21/IntermediateData', then this
                # test will need to change to another hip id.
                download_and_save_hip21_data_to(10, outpath=outpath)


class MockFTP(object):
    fake_server_dir = 'htof/test/data_for_tests/Hip21/IntermediateData'

    def __init__(self, login_error=False, outpath=''):
        self.login_error = login_error
        self.outpath = outpath

    def retrbinary(self, filepath, *args, **kwargs):
        basename = os.path.basename(filepath.split('RETR ')[1])
        remote_filepaths = glob.glob(os.path.join(self.fake_server_dir, basename))
        if len(remote_filepaths) == 0:
            raise RuntimeError('File not found.')
        return shutil.copyfile(remote_filepaths[0], self.outpath)

    def login(self, username, password):
        if self.login_error is False:
            pass
        else:
            raise RuntimeError('Error')

    def prot_p(self):
        pass

    def quit(self):
        pass

