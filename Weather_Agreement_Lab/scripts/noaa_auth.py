import os
from dotenv import load_dotenv


def get_noaa_token():

    load_dotenv()

    token = os.getenv("NOAA_API_TOKEN")

    if not token:
        raise RuntimeError(
            "NOAA_API_TOKEN not found.\n"
            "Create a .env file containing:\n"
            "NOAA_API_TOKEN=your_token_here"
        )

    return token

