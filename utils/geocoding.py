"""Open-Meteo geocoding: city name -> coordinates. Free, no API key."""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_location(location: str) -> dict:
    """Resolve a city name to coordinates.

    Returns {"latitude", "longitude", "resolved_location", "candidate_count"}
    on success, or {"error": "<plain sentence>"} on failure.

    Ambiguous names (Bhopal, Springfield) resolve to the first result, which
    Open-Meteo orders by population/relevance -- a documented default, not a
    silent one. An empty result set and a network failure both return the same
    error shape, because both are the same thing from the user's point of view:
    we cannot produce a forecast for a place we couldn't pin down.
    """
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": location, "count": 5, "language": "en"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"the geocoding service was unreachable ({type(e).__name__})"}

    results = data.get("results") or []
    if not results:
        return {"error": f"no location matching '{location}' was found"}

    first = results[0]
    name_parts = [first.get("name"), first.get("admin1"), first.get("country")]
    return {
        "latitude": first["latitude"],
        "longitude": first["longitude"],
        "resolved_location": ", ".join(p for p in name_parts if p),
        "candidate_count": len(results),
    }
