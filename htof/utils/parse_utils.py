from astropy.time import Time
from html.parser import HTMLParser


def gaia_obmt_to_tcb_julian_year(obmt):
    """
    convert OBMT (on board mission timeline) to TCB julian years via
    https://gea.esac.esa.int/archive/documentation/GDR2/Introduction/chap_cu0int/cu0int_sec_release_framework/cu0int_ssec_time_coverage.html
    Equation 1.1
    :param obmt: on-board mission timeline in units of six-hour revolutions since launch. OBMT (in revolutions)
    :return: astropy.time.Time

    Note that this is the same for DR2 as it is for eDR3, as of 12 23 2020.
    """
    tcbjy = 2015 + (obmt - 1717.6256)/(1461)
    return Time(tcbjy, scale='tcb', format='jyear')


def parse_html(response):
    parser = HipparcosOriginalDataHTMLParser()
    parser.feed(response)
    parser.close()
    data = parser.data
    if data is None or "not found" in data:
        # TODO: add warning
        return None
    return data


class HipparcosOriginalDataHTMLParser(HTMLParser):
    """
    Pull the Hipparcos Original IAD that is hosted online, source-by-source,
    at f"https://hipparcos-tools.cosmos.esa.int/cgi-bin/HIPcatalogueSearch.pl?hipiId={star_id}"
    """
    def __init__(self):
        super(HipparcosOriginalDataHTMLParser, self).__init__()
        self.prev_tag = None
        self.current_tag = None
        self.data = None

    def handle_starttag(self, tag, attrs):
        self.prev_tag = self.current_tag
        self.current_tag = tag
    
    def handle_data(self, data):
        if self.prev_tag == 'pre' and self.current_tag == "b":
            self.data = data.strip()
    
    def close(self):
        self.current_tag = None
        self.prev_tag = None
        super(HipparcosOriginalDataHTMLParser, self).close()
