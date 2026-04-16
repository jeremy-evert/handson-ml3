import pandas as pd

STATION_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def load_station_catalog():
    """
    Download the NOAA ISD station catalog.
    """
    print("Loading NOAA station catalog...")
    df = pd.read_csv(STATION_URL)
    print("Stations loaded:", len(df))
    return df


def find_station_by_city(city_name):
    """
    Search station catalog by city name.
    """
    df = load_station_catalog()

    results = df[df["STATION NAME"].str.contains(city_name, case=False, na=False)]

    return results[[
        "USAF",
        "WBAN",
        "STATION NAME",
        "CTRY",
        "STATE",
        "LAT",
        "LON"
    ]]
