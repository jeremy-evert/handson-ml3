import requests
from pathlib import Path

from scripts.config import DATA_RAW, ensure_directories
from scripts.noaa_auth import get_noaa_token
from scripts.noaa_urls import build_noaa_hourly_url


def download_noaa_hourly(station_id: str, year: int, overwrite=False):

    ensure_directories()

    token = get_noaa_token()
    url = build_noaa_hourly_url(station_id, year)

    output_file = DATA_RAW / f"{station_id}_{year}.csv"

    if output_file.exists() and not overwrite:
        print("✓ File already exists:", output_file)
        return output_file

    print("Downloading NOAA data:", station_id, year)

    headers = {"token": token}

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    print("✓ Saved:", output_file)

    return output_file
