import requests
from pathlib import Path

from scripts.config import DATA_RAW, ensure_directories
from scripts.noaa_auth import get_noaa_token
from scripts.noaa_urls import build_noaa_hourly_url


def download_noaa_hourly(station_id: str, year: int, overwrite: bool = False) -> Path:
    """
    Download NOAA ISD observations for a station.

    Parameters
    ----------
    station_id : str
        NOAA station ID (example: 723530-03927)

    year : int
        Year of data

    overwrite : bool
        If True, force re-download even if file already exists

    Returns
    -------
    Path
        Path to downloaded file
    """

    ensure_directories()

    token = get_noaa_token()

    url = build_noaa_hourly_url(station_id, year)

    output_file = DATA_RAW / f"{station_id}_{year}.gz"

    print("\nDownloading NOAA data")
    print("Station:", station_id)
    print("Year:", year)
    print("URL:", url)
    print("Destination:", output_file)

    # Cache check
    if output_file.exists() and not overwrite:
        print("✓ File already exists:", output_file)
        return output_file

    headers = {"token": token}

    try:
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code == 404:
            raise RuntimeError(
                f"""
NOAA dataset not found.

Station: {station_id}
Year: {year}

Attempted URL:
{url}

Possible causes:
- Station ID incorrect
- Year not available
- NOAA archive structure changed
"""
            )

        response.raise_for_status()

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to retrieve NOAA data:\n{e}")

    # Save file
    with open(output_file, "wb") as f:
        f.write(response.content)

    print("✓ Saved:", output_file)

    return output_file
