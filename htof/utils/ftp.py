from ftplib import FTP_TLS


def download_and_save_hip21_data_to(star_id, outpath):
    if int(star_id) < 1 or int(star_id) > 120404:
            raise RuntimeError("Can not download data. The Hipparcos star ID is likely invalid.")
    file_name = 'H' + str(star_id).zfill(6) + '.d'
    subdir = file_name[:4]
    # the filepath on the anonymous@ftp.cosmos.esa.int ftp server. I think this path needs to be in the
    # unix / forward slashes, so we are not using os.path here because on windows it will use \ slashes.
    fullpath = f'HIPPARCOS_PUBLIC_DATA/ResRec_JavaTool_2014/{subdir}/{file_name}'

    HOSTNAME = "ftp.cosmos.esa.int"
    USERNAME = "anonymous"
    PASSWORD = "https://github.com/gmbrandt/HTOF" # so that the host knows where this request is coming from.
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

