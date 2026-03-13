from scripts.noaa_loader import load_noaa_dataframe

OKC_ISD = "723530-03927"


def load_okc_2012():
    """
    Load NOAA Global Hourly data for Oklahoma City (Will Rogers)
    for the year 2012.
    """
    return load_noaa_dataframe(OKC_ISD, 2012)
