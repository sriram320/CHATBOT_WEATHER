"""Audience gating: does this policy apply to THIS person?

Every SOP in sops.json already declares who it is written for, in its
`applies_to` field (['pregnant_women'], ['pet_owners'], ['cyclists'], ...).
Until this module existed nothing read that field, so policies matched purely
on weather and fired at whoever was asking. A generic cyclist in Aberdeen was
told about pregnancy balance risk, because SOP-022's temperature trigger was
satisfied and nothing checked that the reader wasn't pregnant.

Weather decides whether a hazard exists. Audience decides whether a given
policy is addressed to this reader. Both are pure Python; neither is a model
judgment.

Two tiers, because the two kinds of tag fail in opposite directions:

  IDENTITY tags (pregnant_women, pet_owners, elderly_60_plus, kids_5_15 ...)
      describe who the reader *is*. Firing one without evidence produces advice
      that is absurd at best and offensive at worst. These require positive
      evidence: no evidence means the policy stays silent. Fails closed, the
      same principle the weather evaluator already uses for missing metrics.

  ACTIVITY tags (cyclists, runners, drivers, swimmers ...)
      describe what the reader is *doing*. Here silence is the dangerous
      failure -- withholding a wind warning from someone who never specified
      that they were cycling is worse than showing it. So an unknown activity
      is permissive, and only a positive mismatch filters the policy out.
"""

import re

UNIVERSAL_TAGS = {"all", "all_ages", "all_activities"}

# Identity tags -> the profile/question signals that license them.
IDENTITY_TAGS = {
    "pregnant_women": {"pregnant"},
    "women_of_reproductive_age": {"female"},
    "elderly_60_plus": {"elderly"},
    "elderly": {"elderly"},
    "families_with_elderly": {"elderly", "family"},
    "kids_5_15": {"kids"},
    "kids": {"kids"},
    "parents": {"kids", "family"},
    "caregivers": {"elderly", "kids"},
    "teachers": {"kids"},
    "pet_owners": {"pets"},
    "dog_owners": {"pets"},
    "cat_owners": {"pets"},
    "homeless": {"homeless"},
    "outdoor_workers": {"outdoor_worker"},
    "hill_station_residents": {"hill_station"},
    "farmers": {"farmer"},
    # Occupational identity: a job is something you are, not something you
    # happen to be doing this afternoon, so these fail closed like the rest.
    "construction_workers": {"outdoor_worker", "construction_worker"},
    "street_vendors": {"outdoor_worker", "street_vendor"},
    "delivery_workers": {"outdoor_worker", "delivery_worker"},
    "gig_workers": {"delivery_worker", "gig_worker"},
    "schools": {"kids", "school_staff"},
}

# Activity tags -> keywords that indicate the reader is doing that thing.
ACTIVITY_TAGS = {
    # A "two-wheeler" in Indian usage is a motorcycle or scooter, NOT a bicycle.
    # Conflating them made the gig-delivery-rider policy fire at every ordinary
    # cyclist, and at severity 3 it pushed genuinely relevant policies out of
    # the top 3. Pedal words belong to `cyclists` only.
    "cyclists": {"cycl", "bike", "biking", "bicycle"},
    "motorcyclists": {"motorcycl", "motorbike", "scooter", "two wheeler", "two-wheeler"},
    "two_wheeler_riders": {"two wheeler", "two-wheeler", "scooter", "motorcycl", "motorbike"},
    "runners": {"run", "jog", "marathon"},
    "athletes": {"athlet", "sport", "training", "practice", "run", "cycl", "cricket", "football"},
    "fitness_enthusiasts": {"workout", "exercise", "gym", "fitness", "run", "cycl"},
    "outdoor_sports": {"sport", "cricket", "football", "tennis", "match", "practice"},
    "cricket_players": {"cricket"},
    "students_in_sports": {"sport", "practice", "school", "cricket", "football", "training"},
    "sports_teams": {"team", "practice", "match", "tournament"},
    "swimmers": {"swim", "beach", "sea", "ocean"},
    "surfers": {"surf", "beach", "sea", "ocean"},
    "water_sports_enthusiasts": {"swim", "surf", "kayak", "boat", "sail", "beach", "water sport"},
    "drivers": {"driv", "car", "commute", "road trip", "travel"},
    "office_commuters": {"commute", "work", "office"},
    "students": {"school", "college", "class", "student"},
    "air_travelers": {"flight", "fly", "airport", "plane"},
    "frequent_flyers": {"flight", "fly", "airport", "plane"},
    "rail_travelers": {"train", "rail"},
    "commuters": {"commute", "work", "office", "travel"},
    "travelers": {"travel", "trip", "journey", "visit"},
    "pedestrians": {"walk", "stroll", "pedestrian"},
    "families": {"family", "kids", "picnic", "park"},
    "social_groups": {"picnic", "friends", "gathering", "outing", "party"},
}

# Free-text signals -> identity evidence, for facts stated in the question
# itself ("taking my toddler to the park") rather than in a stored profile.
_QUESTION_IDENTITY_PATTERNS = {
    "pregnant": r"\bpregnan|\bexpecting\b",
    "kids": r"\bkid|\bchild|\btoddler|\bbaby|\binfant|\bson\b|\bdaughter\b|\bschool\b",
    "elderly": r"\belder|\bsenior|\bgrandmother\b|\bgrandfather\b|\bgranny\b|\bold(er)? (parent|father|mother)",
    "pets": r"\bdog\b|\bpuppy\b|\bcat\b|\bpet\b",
    "family": r"\bfamily\b|\bfamilies\b",
    "outdoor_worker": r"\boutdoor work|\bconstruction\b|\bfield work|\blabour\b|\blabor\b|\bsite work",
    "construction_worker": r"\bconstruction\b|\bbuilding site\b|\bmason\b",
    "street_vendor": r"\bstreet vendor\b|\bhawker\b|\broadside stall\b",
    "delivery_worker": r"\bdeliver\w*\b|\bswiggy\b|\bzomato\b|\bzepto\b|\bblinkit\b|\bcourier\b|\bparcel\b",
    "gig_worker": r"\bgig work|\brider\b|\bgig worker\b",
    "school_staff": r"\bschool\b|\bassembly\b|\bsports day\b|\bPE class\b|\bteacher\b|\bprincipal\b",
    "farmer": r"\bfarm|\bcrop\b|\bharvest\b",
}


def derive_signals(profile: dict | None, question: str, activity: str) -> set[str]:
    """Collect identity signals from the stored profile and from this question.

    The question is additive, never subtractive: asking about your toddler adds
    a 'kids' signal for this turn without implying you are not also an adult
    cyclist. That is why a profile is a default and a question is an override
    only for the turn it appears in.
    """
    signals: set[str] = set()
    profile = profile or {}

    gender = str(profile.get("gender") or "").strip().lower()
    if gender in ("female", "woman", "f"):
        signals.add("female")

    if profile.get("pregnant"):
        signals.update({"pregnant", "female"})
    if profile.get("has_pets"):
        signals.add("pets")

    age_group = str(profile.get("age_group") or "").strip().lower()
    if age_group in ("elderly", "60+", "senior"):
        signals.add("elderly")
    if age_group in ("child", "kid", "under_18", "5-15"):
        signals.add("kids")

    for tag in profile.get("extra_tags") or []:
        signals.add(str(tag).strip().lower())

    haystack = f"{question} {activity}".lower()
    for signal, pattern in _QUESTION_IDENTITY_PATTERNS.items():
        if re.search(pattern, haystack):
            signals.add(signal)

    return signals


def _activity_matches(tag: str, question: str, activity: str) -> bool:
    keywords = ACTIVITY_TAGS.get(tag)
    if not keywords:
        return False
    haystack = f"{question} {activity}".lower()
    return any(k in haystack for k in keywords)


def applies_to_user(sop: dict, signals: set[str], question: str, activity: str,
                    activity_known: bool) -> tuple[bool, str]:
    """Decide whether one SOP is addressed to this reader.

    Returns (eligible, reason). The reason is carried into the audit trail so a
    reviewer can see why a policy that matched the weather was still withheld.
    """
    tags = sop.get("applies_to") or []
    if not tags:
        return True, "policy declares no audience -- treated as universal"

    tags = [str(t).strip().lower() for t in tags]

    if any(t in UNIVERSAL_TAGS or t.startswith("esp_") for t in tags):
        return True, "universal policy"

    for tag in tags:
        if tag in IDENTITY_TAGS and IDENTITY_TAGS[tag] & signals:
            return True, f"identity match: {tag}"

    for tag in tags:
        if tag in ACTIVITY_TAGS and _activity_matches(tag, question, activity):
            return True, f"activity match: {tag}"

    has_identity = any(t in IDENTITY_TAGS for t in tags)
    has_activity = any(t in ACTIVITY_TAGS for t in tags)

    # A policy written only for a specific group, with nothing indicating the
    # reader is in it, stays silent.
    if has_identity and not has_activity:
        return False, f"written for {', '.join(tags)}; nothing indicates that applies here"

    # Activity policies are permissive when we do not know the activity:
    # withholding a hazard warning is the more dangerous error.
    if has_activity and not activity_known:
        return True, "activity not stated -- hazard policy shown anyway"

    if has_activity:
        return False, f"written for {', '.join(tags)}; this activity is {activity!r}"

    return True, "no recognised audience tags -- not filtered"


def gate_matches(matches: list[dict], profile: dict | None, question: str,
                 activity: str) -> tuple[list[dict], list[dict]]:
    """Split weather-matched policies into (addressed to this reader, withheld)."""
    activity_known = bool(activity) and activity.strip().lower() not in ("unknown", "none", "null", "")
    signals = derive_signals(profile, question, activity)

    eligible, withheld = [], []
    for match in matches:
        ok, reason = applies_to_user(match, signals, question, activity, activity_known)
        record = {**match, "audience_reason": reason}
        (eligible if ok else withheld).append(record)

    return eligible, withheld


def describe_profile(profile: dict | None, signals: set[str]) -> str:
    """One human-readable line for the compose prompt and the audit trail."""
    profile = profile or {}
    bits = []
    if profile.get("name"):
        bits.append(f"name: {profile['name']}")
    if profile.get("age_group"):
        bits.append(f"age group: {profile['age_group']}")
    if profile.get("gender"):
        bits.append(f"gender: {profile['gender']}")
    if signals:
        bits.append("signals: " + ", ".join(sorted(signals)))
    return "; ".join(bits) if bits else "no profile given"
