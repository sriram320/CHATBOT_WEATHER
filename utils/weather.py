"""Open-Meteo forecast client.

Requests a FIXED BROAD variable set rather than only the metrics today's
policies happen to use. That's deliberate: it's what makes "add a new SOP
without touching code" true for any policy written over a metric already in
this list. A policy over a brand-new metric still needs one line added here --
stated honestly in the README rather than pretended away.
"""

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "weather_code",
    "uv_index",
    "relative_humidity_2m",
    "visibility",
]

DAILY_VARS = [
    "precipitation_sum",
    "snowfall_sum",
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
]


def fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetch current + today's daily weather.

    Returns {"current": {...}, "daily": {...}} with daily's day-0 values
    flattened from lists to scalars, or {"error": "<plain sentence>"}.
    """
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(CURRENT_VARS),
                "daily": ",".join(DAILY_VARS),
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"the weather service was unreachable ({type(e).__name__})"}

    current = data.get("current")
    if not current:
        return {"error": "the weather service returned no current conditions"}

    # Flatten daily lists to day-0 scalars so the evaluator sees plain numbers.
    daily_raw = data.get("daily") or {}
    daily = {}
    for key, value in daily_raw.items():
        if isinstance(value, list):
            daily[key] = value[0] if value else None
        else:
            daily[key] = value

    return {
        "current": current,
        "daily": daily,
        "units": {
            "current": data.get("current_units", {}),
            "daily": data.get("daily_units", {}),
        },
    }
