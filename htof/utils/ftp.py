import pysftp
from ftplib import FTP_TLS


def download_and_save_hip21_data_to(star_id, outpath):
    # TODO this is a stop gap solution using sftp "anomalously" which is pretty unsecure.
    #  once the ESA fixes the website, we should use just normal FTP in passive mode, which
    #   also won't require the pysftp package.
    file_name = 'H' + str(star_id).zfill(6) + '.d'
    subdir = file_name[:4]
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    # the filepath on the anonymous@ftp.cosmos.esa.int ftp server. I think this path needs to be in the
    # unix / forward slashes, so we are not using os.path here because on windows it will use \ slashes.
    fullpath = f'HIPPARCOS_PUBLIC_DATA/ResRec_JavaTool_2014/{subdir}/{file_name}'

    with pysftp.Connection('ftp.cosmos.esa.int', username='anonymous',
                           password='https://github.com/gmbrandt/HTOF', private_key=".ppk",
                           cnopts=cnopts) as sftp:
        sftp.get(fullpath, localpath=outpath)  # get a remote file
    return None


def ftpdownload_and_save_hip21_data_to(star_id, outpath):
    file_name = 'H' + str(star_id).zfill(6) + '.d'
    subdir = file_name[:4]
    # the filepath on the anonymous@ftp.cosmos.esa.int ftp server. I think this path needs to be in the
    # unix / forward slashes, so we are not using os.path here because on windows it will use \ slashes.
    fullpath = f'HIPPARCOS_PUBLIC_DATA/ResRec_JavaTool_2014/{subdir}/{file_name}'

    HOSTNAME = "ftp.cosmos.esa.int"
    USERNAME = "anonymous"
    PASSWORD = "https://github.com/gmbrandt/HTOF"
    ftps = FTP_TLS(HOSTNAME)
    ftps.login(USERNAME, PASSWORD)
    ftps.prot_p()
    # force UTF-8 encoding
    #ftps.encoding = "utf-8"
    # Write file in binary mode
    with open(outpath, "wb") as file:
        # Command for Downloading the file "RETR filename"
        ftps.retrbinary(f"RETR {fullpath}", file.write)
    return None

