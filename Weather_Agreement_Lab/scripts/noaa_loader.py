import pandas as pd

from scripts.config import DATA_RAW
from scripts.noaa_download import download_noaa_hourly


def load_noaa_dataframe(station_id: str, year: int):

    file_path = DATA_RAW / f"{station_id}_{year}.csv"

    if not file_path.exists():
        file_path = download_noaa_hourly(station_id, year)

    return pd.read_csv(file_path)

