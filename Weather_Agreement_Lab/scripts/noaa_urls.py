def build_noaa_hourly_url(station_id: str, year: int) -> str:
    """
    Construct NOAA ISD archive URL.
    """
    return f"https://www.ncei.noaa.gov/pub/data/noaa/{year}/{station_id}-{year}.gz"
