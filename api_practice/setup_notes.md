Absolutely. This is a perfect “learn the plumbing while making something pretty” exercise.

## What we’ll build

A **clean Jupyter Notebook** that:

* calls a real API (no API key needed for this demo),
* uses **two endpoints** (geocoding + weather forecast),
* converts JSON into a **pandas DataFrame**,
* and creates a couple of **visualizations**.

I’m going to use **Open-Meteo** because it is well-documented, supports multiple APIs (including geocoding + forecast), and returns rich hourly weather data. The docs show a geocoding endpoint and a `/v1/forecast` endpoint with configurable hourly variables. ([open-meteo.com][1])

Jupyter is a web-based environment specifically built for documents that combine **live code + narrative text + visualizations**, which is exactly what you asked for. ([docs.jupyter.org][2])

---

## PowerShell setup from absolute zero (Windows)

### 0) Install Python (one-time, if needed)

If you do not already have Python installed, install Python 3 from python.org (make sure the installer adds Python to PATH). The PyPA guide also recommends checking Python and pip from the command line first. ([packaging.python.org][3])

---

### 1) Open PowerShell and create a project folder

```powershell
# Make a folder for the project
mkdir C:\Users\$env:USERNAME\api_jupyter_demo
cd C:\Users\$env:USERNAME\api_jupyter_demo
```

---

### 2) Verify Python and pip

(Using `py` is standard on Windows and is documented in the Python Packaging guide.) ([packaging.python.org][3])

```powershell
py --version
py -m pip --version
```

---

### 3) Create and activate a virtual environment

(PyPA documents `py -m venv <DIR>` and `<DIR>\Scripts\activate` on Windows.) ([packaging.python.org][3])

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### If PowerShell blocks activation

This happens a lot. Run this in the current PowerShell session, then try activation again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```


Run this in the same PowerShell window if it complains
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### 4) Upgrade pip and install packages

Jupyter docs show pip-based installation, and the `jupyter` metapackage includes Notebook, JupyterLab, and IPython kernel. ([docs.jupyter.org][4])



```powershell
py -m pip install --upgrade pip
py -m pip install jupyter requests pandas matplotlib
```

Optional but nice (makes your venv show up clearly as a kernel choice in Jupyter):

```powershell
py -m pip install ipykernel
py -m ipykernel install --user --name api-demo --display-name "Python (api-demo)"
```

---

### 5) Launch Jupyter Notebook

I am goin to try:

```
code api_practice.ipynb
```

```powershell
jupyter notebook
```

If `jupyter` is not recognized, use:

```powershell
py -m notebook
```

Then in the browser:

* Click **New** → **Python 3** (or **Python (api-demo)** if you installed the kernel above)
* Save the notebook as: `api_weather_demo.ipynb`

---

# Notebook content (copy this cell-by-cell)

Below is a **ready-to-build notebook layout** with markdown cells and code cells exactly the way you requested.

---

## Cell 1 (Markdown)

```markdown
# API + Python in Jupyter: Weather Data Visualization Demo

## Goal
Learn how to:
1. Call a real API from Python
2. Read JSON data returned by the API
3. Convert the JSON into a pandas DataFrame
4. Visualize the results with matplotlib

## What we will do
We will use the Open-Meteo API to:
- Look up a city (geocoding endpoint)
- Request hourly weather data for that location (forecast endpoint)
- Plot temperature and precipitation over time

## Why this is useful
This is the same workflow used in many machine learning and data science projects:
- fetch data
- clean/reshape data
- inspect it
- visualize it
```

---

## Cell 2 (Code)

```python
# Step 1: Import the libraries we need
# - requests: make HTTP/API calls
# - pandas: table/dataframe handling
# - matplotlib: plotting/visualization

import requests
import pandas as pd
import matplotlib.pyplot as plt

# Make plots a little easier to read in notebooks
plt.rcParams["figure.figsize"] = (12, 5)

# A tiny helper function so our API calls are clean and reusable
def get_json(url, params=None):
    """
    Send a GET request to an API endpoint and return the JSON response.
    
    Why this helper exists:
    - keeps our code short
    - gives us one place to handle request errors
    """
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  # raises an error if the request failed (4xx/5xx)
    return response.json()
```

---

## Cell 3 (Markdown)

```markdown
## Step 2: Use a geocoding API to find latitude/longitude

Most weather APIs need coordinates (latitude/longitude), not just a city name.

So we first call a geocoding endpoint:
- input: a city name (for example, "Weatherford")
- output: matching places with coordinates
```

---

## Cell 4 (Code)

```python
# Step 2: Geocoding API call (city name -> coordinates)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Try your city here. You can change this and rerun the cell.
city_query = "Weatherford, Oklahoma"

geocode_params = {
    "name": city_query,
    "count": 5,        # ask for up to 5 matches
    "language": "en",
    "format": "json"
}

geocode_data = get_json(GEOCODE_URL, geocode_params)

# Inspect the top-level keys to learn the JSON structure
print("Top-level keys:", geocode_data.keys())

# Pull out results safely
results = geocode_data.get("results", [])

if not results:
    raise ValueError(f"No geocoding results found for: {city_query}")

# Show the candidate locations in a small table
locations_df = pd.DataFrame(results)[["name", "country", "admin1", "latitude", "longitude", "timezone"]]
locations_df
```

---

## Cell 5 (Markdown)

```markdown
## Step 3: Choose a location and request hourly forecast data

Now that we have coordinates, we call the weather forecast endpoint.

We will request hourly values for:
- temperature_2m
- precipitation
- relative_humidity_2m
```

---

## Cell 6 (Code)

```python
# Step 3: Pick the first matching location (you can choose another row if needed)
chosen = results[0]

city_name = chosen["name"]
country = chosen.get("country", "")
admin1 = chosen.get("admin1", "")
latitude = chosen["latitude"]
longitude = chosen["longitude"]
timezone = chosen.get("timezone", "auto")

print(f"Using location: {city_name}, {admin1}, {country}")
print(f"Latitude: {latitude}, Longitude: {longitude}, Timezone: {timezone}")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

forecast_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "temperature_2m,precipitation,relative_humidity_2m",
    "forecast_days": 5,
    "timezone": "auto"
}

forecast_data = get_json(FORECAST_URL, forecast_params)

# Quick peek at the response structure
print("Forecast keys:", forecast_data.keys())
print("Hourly keys:", forecast_data["hourly"].keys())
```

---

## Cell 7 (Markdown)

```markdown
## Step 4: Convert JSON into a pandas DataFrame

APIs often return nested JSON.
For analysis and plotting, we usually reshape it into a table.

This is a key data-science move:
JSON -> DataFrame -> analysis/visualization
```

---

## Cell 8 (Code)

```python
# Step 4: Build a DataFrame from the hourly data block

hourly = forecast_data["hourly"]

weather_df = pd.DataFrame({
    "time": hourly["time"],
    "temperature_2m": hourly["temperature_2m"],
    "precipitation": hourly["precipitation"],
    "relative_humidity_2m": hourly["relative_humidity_2m"]
})

# Convert time strings to actual datetime objects
weather_df["time"] = pd.to_datetime(weather_df["time"])

# Set time as the index (makes time-series work easier)
weather_df = weather_df.set_index("time")

# Inspect the first few rows
weather_df.head()
```

---

## Cell 9 (Markdown)

```markdown
## Step 5: Do a couple of simple analyses

Before plotting, let's compute a daily summary.
This shows how the same API data can support multiple kinds of analysis.
```

---

## Cell 10 (Code)

```python
# Step 5: Daily summary from hourly data
daily_summary = weather_df.resample("D").agg({
    "temperature_2m": ["min", "max", "mean"],
    "precipitation": "sum",
    "relative_humidity_2m": "mean"
})

daily_summary
```

---

## Cell 11 (Markdown)

```markdown
## Step 6: Visualize the results

We will make two plots:
1. Hourly temperature over time
2. Hourly precipitation over time (bar chart)

This is the "see the shape of the data" moment.
```

---

## Cell 12 (Code)

```python
# Step 6A: Plot hourly temperature
plt.figure()
weather_df["temperature_2m"].plot()
plt.title(f"Hourly Temperature Forecast for {city_name}")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()
```

---

## Cell 13 (Code)

```python
# Step 6B: Plot hourly precipitation
plt.figure()
weather_df["precipitation"].plot(kind="bar")
plt.title(f"Hourly Precipitation Forecast for {city_name}")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## Cell 14 (Markdown)

```markdown
## Why we did the key steps

### 1) Geocoding first
Weather APIs often require coordinates, so we used a geocoding endpoint to translate a city name into latitude/longitude.

### 2) Parameterized API calls
We passed parameters like `hourly`, `forecast_days`, and `timezone`.
This makes the request reproducible and easy to change.

### 3) JSON to DataFrame
API responses are usually JSON, but plotting and analysis are easier in pandas.
Converting to a DataFrame is a standard workflow in data science and ML preprocessing.

### 4) Datetime conversion
Turning timestamps into real datetime values lets us:
- resample by day
- sort naturally
- create better time-series plots

### 5) Visualization
Plots help us quickly inspect trends, spikes, and possible data issues before doing any modeling.

## Next ideas (great practice)
- Compare two cities on the same chart
- Request more variables (wind speed, cloud cover, apparent temperature)
- Save the DataFrame to CSV
- Build a tiny model using the weather data later
```

---

## Why this is a strong API example for ML/data science

This notebook teaches the full pipeline without a lot of chaos:

* **API request mechanics** (`requests.get`, params, JSON)
* **Data wrangling** (nested JSON to DataFrame)
* **Time series handling** (`pd.to_datetime`, indexing, resampling)
* **Visualization** (line + bar plots)
* **Reusability** (helper function + parameter dictionaries)

It is the same skeleton you will use later for:

* finance APIs
* sports APIs
* public health APIs
* geospatial APIs
* model-serving APIs

Basically, this is a small rocket that uses training wheels but still leaves the launchpad.

---

## Optional next notebook (if you want to level it up)

Next step I’d build for you is an **authenticated API** example (API keys + headers), such as:

* NASA APIs
* Alpha Vantage / market data
* OpenWeather (API key flow)
* a simple REST API you build locally with Flask/FastAPI

That version teaches:

* headers,
* auth tokens,
* rate limiting,
* and pagination.

If you want, I can also give you this same notebook as a `.ipynb` file next.

[1]: https://open-meteo.com/en/docs/geocoding-api "Geocoding API | Open-Meteo.com"
[2]: https://docs.jupyter.org/en/stable/install.html "Install and Use — Jupyter Documentation 4.1.1 alpha documentation"
[3]: https://packaging.python.org/tutorials/installing-packages/ "Installing Packages - Python Packaging User Guide"
[4]: https://docs.jupyter.org/en/latest/install/notebook-classic.html "Installing the classic Jupyter Notebook interface — Jupyter Documentation 4.1.1 alpha documentation"
