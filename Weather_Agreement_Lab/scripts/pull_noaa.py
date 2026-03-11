import os
from datetime import datetime
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from scripts.config import DATA_RAW, ensure_directories


def download_noaa_hourly(station_id: str, year: int):

    ensure_directories()

    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"

    output_file = DATA_RAW / f"{station_id}_{year}.csv"

    response = requests.get(url)

    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    print("Saved:", output_file)

    return output_file

