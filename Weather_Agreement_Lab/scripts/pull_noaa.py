"""
pull_noaa.py

Engine for retrieving NOAA Global Hourly observations.

This module handles:

• loading the NOAA API token
• building download URLs
• retrieving hourly station CSV files
• storing them in data/raw
"""

import os
from pathlib import Path
import requests
from dotenv import load_dotenv

from scripts.config import DATA_RAW, ensure_directories


# -------------------------------------------------------------------
# Token Management
# -------------------------------------------------------------------

def get_noaa_token() -> str:
    """
    Load NOAA API token from the project .env file.

    Returns
    -------
    str
        NOAA API token
    """

    load_dotenv()

    token = os.getenv("NOAA_API_TOKEN")

    if not token:
        raise RuntimeError(
            "NOAA_API_TOKEN not found.\n"
            "Create a .env file in the project root containing:\n"
            "NOAA_API_TOKEN=your_token_here"
        )

    return token


# -------------------------------------------------------------------
# URL Construction
# -------------------------------------------------------------------

def build_noaa_hourly_url(station_id: str, year: int) -> str:
    """
    Construct NOAA Global Hourly dataset URL.
    """

    return f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"


# -------------------------------------------------------------------
# Download Engine
# -------------------------------------------------------------------

def download_noaa_hourly(station_id: str, year: int, overwrite: bool = False) -> Path:
    """
    Download NOAA Global Hourly observations for a station.

    Parameters
    ----------
    station_id : str
        NOAA station ID (example: 723530-03927)

    year : int
        Year of data

    overwrite : bool
        If True, re-download even if file exists

    Returns
    -------
    Path
        Path to downloaded file
    """

    ensure_directories()

    token = get_noaa_token()

    url = build_noaa_hourly_url(station_id, year)

    output_file = DATA_RAW / f"{station_id}_{year}.csv"

    if output_file.exists() and not overwrite:
        print("✓ File already exists:", output_file)
        return output_file

    print("Downloading NOAA data...")
    print("Station:", station_id)
    print("Year:", year)

    headers = {"token": token}

    response = requests.get(url, headers=headers, timeout=60)

    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    print("✓ Saved:", output_file)

    return output_file


# -------------------------------------------------------------------
# Optional Quick Loader
# -------------------------------------------------------------------

def load_noaa_dataframe(station_id: str, year: int):
    """
    Convenience function for loading NOAA data as a pandas DataFrame.
    """

    import pandas as pd

    file_path = DATA_RAW / f"{station_id}_{year}.csv"

    if not file_path.exists():
        file_path = download_noaa_hourly(station_id, year)

    df = pd.read_csv(file_path)

    return df

