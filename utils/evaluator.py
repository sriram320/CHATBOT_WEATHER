"""The generic SOP condition evaluator -- written once, never edited per policy.

This is the heart of "adding a policy requires no code change". Every SOP's
condition is expressed in one small grammar:

    leaf     : {"metric": str, "op": ">"|">="|"<"|"<="|"=="|"in", "value": number|list}
    compound : {"all_of": [cond, ...]}   # AND
               {"any_of": [cond, ...]}   # OR
               {"not": cond}
               {"always": bool}
               {"fuzzy": true}           # deferred to the LLM node, never decided here

Nothing in this file knows what any individual SOP means. Add SOP-026 with a
new combination of metrics and operators and this file is untouched.
"""

import json
from pathlib import Path

SOPS_PATH = Path(__file__).parent.parent / "sops_final_25.json"

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "in": lambda a, b: a in b,
}

# Fields copied onto a match record so downstream nodes never need the raw SOP.
_MATCH_FIELDS = (
    "id", "name", "severity", "severity_name", "category",
    "description", "advice_template", "traceable_to_api",
    # Who the policy is written for, and whether it is a fallback policy.
    # Both are read downstream (utils/audience.py, node_resolve_and_rank);
    # this file still never interprets them.
    "applies_to", "baseline",
)


def load_sops() -> list[dict]:
    """Read policies fresh from disk on every call.

    Deliberately NOT cached: editing sops_final_25.json takes effect on the
    very next question with no restart, which is what makes the "add an Nth
    SOP live" demo work.
    """
    with open(SOPS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["sops"]


def flatten_weather(weather_data: dict) -> dict:
    """Merge daily + current into one lookup table.

    Order matters: daily first, then current, so current WINS on key clashes.
    Both carry weather_code, and "what the sky is doing right now" must beat
    "what the day as a whole is coded as".
    """
    return {**(weather_data.get("daily") or {}), **(weather_data.get("current") or {})}


def evaluate_condition(cond, weather_data: dict):
    """Evaluate one condition node. Returns True/False, or None for fuzzy.

    Fails CLOSED: a missing metric means the policy does not fire. Inventing a
    hazard from absent data would be worse than staying quiet.
    """
    if cond is None:
        return False
    if "fuzzy" in cond:
        return None  # not ours to decide -- fuzzy_check_node handles it
    if "always" in cond:
        return bool(cond["always"])
    if "all_of" in cond:
        return all(evaluate_condition(c, weather_data) is True for c in cond["all_of"])
    if "any_of" in cond:
        return any(evaluate_condition(c, weather_data) is True for c in cond["any_of"])
    if "not" in cond:
        return evaluate_condition(cond["not"], weather_data) is not True

    w = flatten_weather(weather_data)
    value = w.get(cond["metric"])
    if value is None:
        return False  # fail closed

    op = _OPS.get(cond["op"])
    if op is None:
        return False
    try:
        return bool(op(value, cond["value"]))
    except TypeError:
        return False


def evaluate_all_sops(sops: list[dict], weather_data: dict) -> list[dict]:
    """Run every non-fuzzy SOP against the weather. Pure Python, no LLM.

    Returns one match record per firing policy, carrying the exact metric
    values that made it fire so the audit trail can show its work.
    """
    matches = []
    for sop in sops:
        if sop.get("match_type") == "fuzzy":
            continue
        if evaluate_condition(sop.get("condition"), weather_data) is not True:
            continue

        record = {k: sop.get(k) for k in _MATCH_FIELDS}
        record["specificity"] = sop.get("specificity", 1)
        record["values_used"] = _collect_values(sop.get("condition"), weather_data)
        matches.append(record)
    return matches


def _collect_values(cond, weather_data: dict) -> dict:
    """Pull out the metric values a condition actually looked at, for the audit trail."""
    w = flatten_weather(weather_data)
    found = {}

    def walk(c):
        if not isinstance(c, dict):
            return
        if "metric" in c:
            metric = c["metric"]
            if metric in w:
                found[metric] = w[metric]
                found[f"threshold_{metric}"] = c.get("value")
        for key in ("all_of", "any_of"):
            for sub in c.get(key, []):
                walk(sub)
        if "not" in c:
            walk(c["not"])

    walk(cond)
    return found


def rank_sops(matches: list[dict]) -> list[dict]:
    """(severity DESC, specificity DESC).

    Severity leads because it is a safety signal; specificity only breaks ties
    between policies of equal seriousness, so a narrowly-scoped medium rule can
    never outrank a broad severe one.
    """
    return sorted(
        matches,
        key=lambda m: (m.get("severity", 0), m.get("specificity", 0)),
        reverse=True,
    )
