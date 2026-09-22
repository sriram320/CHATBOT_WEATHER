"""Open-Meteo geocoding: city name -> coordinates. Free, no API key.

Resolving the wrong place is the worst failure this system can have, because
it fails *silently*: every downstream number is real, grounded and verifiable,
and all of it describes somewhere the user never asked about. Nothing in the
grounding layer can catch it. So the resolution step is defensive on purpose.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Open-Meteo's gazetteer stores the official post-renaming forms. The older
# names are still what most people type, and searching them does NOT fall back
# to the modern city -- it returns whatever unrelated place happens to share
# the string. "Bangalore" is the case that caught us: the only result is
# "Bangalore Town", a neighbourhood in Karachi, Pakistan, so a question about
# Bangalore was silently answered with Pakistani weather.
_ALIASES = {
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "poona": "Pune",
    "gurgaon": "Gurugram",
    "mysore": "Mysuru",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi",
    "baroda": "Vadodara",
    "pondicherry": "Puducherry",
    "simla": "Shimla",
    "benares": "Varanasi",
    "banaras": "Varanasi",
    "cawnpore": "Kanpur",
    "vizag": "Visakhapatnam",
    "saigon": "Ho Chi Minh City",
    "peking": "Beijing",
}

# Populated-place classes, best first. PPLX ("section of populated place") is
# the one that beat a city of 8.5 million, so class is weighted before size.
_FEATURE_RANK = {
    "PPLC": 0,   # national capital
    "PPLA": 1,   # first-order admin capital
    "PPLA2": 2,
    "PPLA3": 3,
    "PPLA4": 4,
    "PPL": 5,    # ordinary populated place
    "PPLX": 8,   # section of a populated place -- a neighbourhood, not a city
}


def _score(result: dict) -> tuple:
    """Rank key for one candidate: lower sorts first.

    Feature class leads, population breaks ties. Sorting by population alone
    is not enough, because the entries that cause wrong answers (neighbourhoods,
    hamlets) usually carry no population at all and would tie at zero.
    """
    rank = _FEATURE_RANK.get(result.get("feature_code"), 6)
    population = result.get("population") or 0
    return (rank, -population)


def geocode_location(location: str, country_hint: str | None = None) -> dict:
    """Resolve a city name to coordinates.

    Returns {"latitude", "longitude", "resolved_location", "candidate_count",
    "alternatives", "alias_applied"} on success, or {"error": "<sentence>"}.

    An empty result set and a network failure return the same error shape,
    because from the user's point of view they are the same thing: we cannot
    produce a forecast for a place we could not pin down.
    """
    if not location or not location.strip():
        return {"error": "no location was given"}

    query = location.strip()
    alias_applied = None
    canonical = _ALIASES.get(query.lower())
    if canonical:
        alias_applied = f"{query} -> {canonical}"
        query = canonical

    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": query, "count": 10, "language": "en"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"the geocoding service was unreachable ({type(e).__name__})"}

    results = data.get("results") or []
    if not results:
        return {"error": f"no location matching '{location}' was found"}

    ranked = sorted(results, key=_score)

    # A country hint only promotes a candidate that is already a valid match;
    # it never invents or overrides a place the user explicitly named.
    if country_hint:
        hint = country_hint.strip().lower()
        preferred = [r for r in ranked if (r.get("country") or "").lower() == hint
                     or (r.get("country_code") or "").lower() == hint]
        if preferred:
            ranked = preferred + [r for r in ranked if r not in preferred]

    best = ranked[0]
    name_parts = [best.get("name"), best.get("admin1"), best.get("country")]

    return {
        "latitude": best["latitude"],
        "longitude": best["longitude"],
        "resolved_location": ", ".join(p for p in name_parts if p),
        "candidate_count": len(ranked),
        "alias_applied": alias_applied,
        "alternatives": [
            ", ".join(p for p in [r.get("name"), r.get("admin1"), r.get("country")] if p)
            for r in ranked[1:4]
        ],
    }
