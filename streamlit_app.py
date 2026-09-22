"""Streamlit front end for the Weather Advisory Bot.

This file is a view layer and nothing else. Every decision -- which policy
fires, who it is addressed to, how the top 3 rank, whether each number in the
answer is traceable -- still happens in graph.py and utils/, unchanged. Swapping
FastAPI for Streamlit therefore cannot alter a single answer, which is the point
of keeping the determinism boundary out of the transport layer.

Two things differ from main.py, both environmental rather than behavioural:

  * Streamlit Community Cloud supplies secrets through st.secrets, not the
    process environment, so the key is copied into os.environ before graph.py
    is imported. utils/llm.py keeps reading os.environ and needs no edit.

  * Streamlit re-runs this script top to bottom on every interaction, so the
    compiled graph is cached (@st.cache_resource) and conversation state lives
    in st.session_state. Session memory still carries the same structured facts
    main.py carried -- activity, location, chosen SOP -- never raw transcript.
"""

import base64
import os
from pathlib import Path

import streamlit as st

# st.dialog was added in Streamlit 1.39. Cloud may run older versions, so
# detect and skip the gate if unavailable -- the advisory is still functional,
# just without the explicit terms upfront.
HAS_DIALOG = hasattr(st, "dialog") and callable(st.dialog)

st.set_page_config(
    page_title="Weather Advisory Bot",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bridge Streamlit Cloud secrets -> process env, before graph.py imports
# utils/llm.py. Locally there are no secrets and python-dotenv reads .env
# instead, so the same code path works in both places.
try:
    if "NVIDIA_API_KEY" in st.secrets:
        os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass  # no secrets.toml locally -- .env handles it

from graph import build_graph  # noqa: E402  (must follow the secrets bridge)


@st.cache_resource(show_spinner=False)
def get_graph():
    """Compile the LangGraph once per process, not once per rerun."""
    return build_graph()


CONTACT_EMAIL = "sriramkundapur777@gmail.com"

# Drop a logo at any of these paths and it appears in the masthead; with none
# present the masthead simply renders without one.
LOGO_CANDIDATES = (
    "assets/logo.png", "assets/logo.svg", "assets/logo.jpg",
    "static/logo.png", "static/logo.svg",
)
_MIME = {".png": "image/png", ".svg": "image/svg+xml",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str | None:
    """Inline the logo so the masthead stays a single HTML block."""
    here = Path(__file__).parent
    for candidate in LOGO_CANDIDATES:
        path = here / candidate
        if path.is_file():
            mime = _MIME.get(path.suffix.lower())
            if mime:
                encoded = base64.b64encode(path.read_bytes()).decode()
                return f"data:{mime};base64,{encoded}"
    return None


TERMS_HTML = f"""
<div class="terms">
  <h4>What this is</h4>
  <p>Weather Advisory Bot gives general outdoor-safety guidance by combining live
  readings from Open-Meteo with a library of 32 written safety policies. Every
  answer names the policy it applied, and every number it quotes is traceable to
  a live reading, a policy threshold, or the policy text.</p>

  <h4>What this is not</h4>
  <ul>
    <li><b>Not an official weather warning service.</b> Always defer to your
    national meteorological agency and to local emergency services.</li>
    <li><b>Not medical advice.</b> Guidance relating to pregnancy, age, or any
    health circumstance is general in nature. Consult a qualified clinician for
    decisions about your health.</li>
    <li><b>Not a substitute for your own judgement.</b> Conditions change, and
    local factors — terrain, traffic, air quality, your own fitness — are
    outside what this tool can see.</li>
  </ul>

  <h4>Accuracy</h4>
  <p>Weather data comes from Open-Meteo and may be delayed, interpolated, or
  briefly unavailable. When a reading cannot be retrieved the tool says so
  rather than guessing. Forecasts are inherently uncertain.</p>

  <h4>Your data</h4>
  <p>The optional profile is held only for your current browser session and is
  used solely to decide which policies are addressed to you. It is not written
  to any database, is never shared, and is discarded when the session ends.</p>

  <h4>Liability</h4>
  <p>This tool is provided as-is, without warranty. You remain responsible for
  the decisions you make about going outdoors.</p>

  <h4>Questions, complaints or suggestions</h4>
  <p>Reach out any time — feedback genuinely shapes the policy library:<br>
  <span class="mail">{CONTACT_EMAIL}</span></p>
</div>
"""


if HAS_DIALOG:
    @st.dialog("Terms of Use", width="large", dismissible=False)
    def terms_gate():
        """Shown once per session before the advisory is usable.

        A safety tool that touches pregnancy, children and the elderly should state
        its limits before it gives a first answer, not in a footnote afterwards.
        Not dismissible: the script calls st.stop() behind this, so an X would
        leave the reader on a dead page with no way back.
        """
        st.markdown(TERMS_HTML, unsafe_allow_html=True)
        if st.button("I understand and agree", type="primary", use_container_width=True):
            st.session_state.terms_accepted = True
            st.rerun()

    @st.dialog("Terms of Use", width="large")
    def terms_view():
        """The same text, re-openable from the sidebar after acceptance."""
        st.markdown(TERMS_HTML, unsafe_allow_html=True)
else:
    # Fallback for older Streamlit: show terms inline, once per session.
    def terms_gate():
        if not st.session_state.get("terms_shown"):
            st.warning(
                "⚠️ **Please read before using**\n\n" +
                TERMS_HTML.replace("<div class='terms'>", "").replace("</div>", "")
            )
            if st.button("I understand and agree", type="primary", use_container_width=True):
                st.session_state.terms_accepted = True
                st.session_state.terms_shown = True
                st.rerun()
            st.stop()
        st.session_state.setdefault("terms_accepted", True)

    def terms_view():
        with st.expander("📋 Terms of Use"):
            st.markdown(TERMS_HTML, unsafe_allow_html=True)


# --------------------------------------------------------------- presentation

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink:       #0A0E14;
  --surface:   #111922;
  --surface-2: #16212C;
  --line:      #22303D;
  --text:      #E3E9F0;
  --muted:     #8593A4;
  --amber:     #E8A33D;
  --teal:      #45B8AC;
  --red:       #D9544D;
  --blue:      #5B9BD5;
}

.stApp {
  background:
    radial-gradient(1100px 500px at 15% -8%, #16222E 0%, transparent 58%),
    radial-gradient(900px 420px at 88% 4%, #1A1726 0%, transparent 55%),
    var(--ink);
  font-family: 'Inter', system-ui, sans-serif;
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1080px; }

/* ---------- masthead ---------- */
.masthead {
  border-bottom: 1px solid var(--line);
  padding-bottom: 18px;
  margin-bottom: 26px;
}
.masthead .eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 10px;
}
.masthead h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 700;
  font-size: 44px;
  line-height: 1.02;
  letter-spacing: -.02em;
  color: var(--text);
  margin: 0 0 12px 0;
}
.masthead p {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
  max-width: 62ch;
  margin: 0 0 16px 0;
}
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 4px 9px;
  background: rgba(255,255,255,.015);
}
.chip .dot {
  display: inline-block; width: 5px; height: 5px; border-radius: 50%;
  background: var(--teal); margin-right: 6px; vertical-align: middle;
}

/* ---------- chat messages as briefing cards ---------- */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] > img:first-child {
  display: none !important;
}
[data-testid="stChatMessage"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 16px 20px;
  margin-bottom: 14px;
  gap: 0;
}
[data-testid="stChatMessage"] > div { width: 100%; }

.msg-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .2em;
  text-transform: uppercase;
  margin-bottom: 9px;
  padding-bottom: 8px;
  border-bottom: 1px dashed var(--line);
}
.msg-label.enquiry  { color: var(--blue); }
.msg-label.advisory { color: var(--amber); }

[data-testid="stChatMessage"] p {
  font-size: 14.5px;
  line-height: 1.72;
  color: var(--text);
}

/* numerals in the advisory body read as instrument data */
[data-testid="stChatMessage"] strong { color: var(--amber); font-weight: 600; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: #080C11;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.side-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--amber);
  padding-bottom: 9px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 14px;
}
[data-testid="stSidebar"] label p {
  font-size: 11.5px !important;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 500;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #10161D !important;
  border-color: var(--line) !important;
  border-radius: 3px !important;
  font-size: 13px !important;
}

/* ---------- audit appendix ---------- */
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  background: var(--surface-2) !important;
  margin-top: 4px;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  letter-spacing: .16em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
  font-size: 13px;
  line-height: 1.6;
}
[data-testid="stExpander"] [data-testid="stCaptionContainer"] p {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  line-height: 1.65;
}

.policy {
  border-left: 2px solid var(--line);
  padding: 2px 0 2px 13px;
  margin: 9px 0;
}
.policy.sev-critical, .policy.sev-severe { border-left-color: var(--red); }
.policy.sev-high     { border-left-color: var(--amber); }
.policy.sev-medium   { border-left-color: var(--blue); }
.policy.sev-low, .policy.sev-info { border-left-color: var(--teal); }

.sop-id {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
}
.sev-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px;
  letter-spacing: .12em;
  border: 1px solid;
  border-radius: 2px;
  padding: 1px 6px;
  margin-left: 7px;
}
.spec { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--muted); margin-left: 6px; }

.section-head {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 16px 0 7px 0;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}
.section-head:first-child { border-top: none; padding-top: 0; margin-top: 0; }

.verdict-ok   { color: var(--teal); font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.verdict-bad  { color: var(--red);  font-family: 'JetBrains Mono', monospace; font-size: 12px; }

.claim {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  padding: 2px 0 2px 11px;
  border-left: 1px solid var(--line);
  margin: 2px 0;
}
.claim b { color: var(--text); font-weight: 500; }

/* ---------- opening hint ---------- */
.hint {
  border: 1px dashed var(--line);
  border-radius: 4px;
  padding: 22px 24px;
  color: var(--muted);
  font-size: 13.5px;
  line-height: 1.75;
}
.hint .q {
  display: block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--amber);
  margin-top: 9px;
}

/* ---------- input ---------- */
[data-testid="stChatInput"] {
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  background: var(--surface) !important;
}
[data-testid="stChatInput"] textarea { font-size: 14px !important; }

[data-testid="stSidebar"] button {
  border-radius: 3px;
  border: 1px solid var(--line);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
}

/* ---------- masthead: logo + slogan ---------- */
.brand { display: flex; align-items: center; gap: 15px; margin-bottom: 6px; }
/* The mark is a light-background badge, so it sits on its own soft plate
   rather than floating as a bright rectangle against the ink background. */
.brand img {
  height: 58px; width: 58px; object-fit: cover; flex-shrink: 0;
  border-radius: 50%;
  background: #F2F6FA;
  border: 1px solid var(--line);
  box-shadow: 0 0 0 3px rgba(232,163,61,.10);
}
.brand h1 { margin: 0 !important; }

.slogan {
  font-family: 'Fraunces', Georgia, serif;
  font-style: italic;
  font-size: 17px;
  color: var(--amber);
  margin: 0 0 14px 0;
  letter-spacing: .01em;
}
.slogan .sub {
  display: block;
  font-family: 'Inter', sans-serif;
  font-style: normal;
  font-size: 13px;
  color: var(--muted);
  margin-top: 5px;
}

/* ---------- terms dialog ---------- */
.terms h4 {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--amber);
  margin: 18px 0 7px 0;
}
.terms h4:first-child { margin-top: 0; }
.terms p, .terms li {
  font-size: 13.5px;
  line-height: 1.68;
  color: var(--text);
}
.terms li { margin-bottom: 5px; }
.terms .mail { color: var(--amber); font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }

/* ---------- footer ---------- */
.footer {
  border-top: 1px solid var(--line);
  margin-top: 34px;
  padding-top: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  line-height: 1.85;
  color: var(--muted);
  letter-spacing: .04em;
}
.footer a { color: var(--amber); text-decoration: none; }
.footer a:hover { text-decoration: underline; }
</style>
"""

SEVERITY_COLOURS = {
    "critical": "#D9544D", "severe": "#D9544D",
    "high": "#E8A33D", "medium": "#5B9BD5",
    "low": "#45B8AC", "info": "#45B8AC",
}


def severity_tag(severity: str) -> str:
    if not severity:
        return ""
    colour = SEVERITY_COLOURS.get(str(severity).lower(), "#8593A4")
    return (
        f"<span class='sev-tag' style='color:{colour};border-color:{colour}'>"
        f"{str(severity).upper()}</span>"
    )


def render_audit(audit: dict) -> None:
    """The 'Why this answer' appendix: the same trail the FastAPI page showed.

    This is the part a reviewer actually reads, so it shows the policies that
    fired, the policies that matched the weather but were withheld as not
    addressed to this reader, and the provenance of every number quoted.
    """
    with st.expander("Why this answer  ·  decision trace", expanded=False):
        if audit.get("error"):
            st.markdown("<div class='section-head'>Failure path</div>", unsafe_allow_html=True)
            st.markdown(
                "No policy was evaluated, because we never got data.",
                unsafe_allow_html=True,
            )
            st.caption(f"reason: {audit.get('reason', 'unknown')}")
            st.caption("No forecast is reported when we don't have one.")
            return

        top_3 = audit.get("top_3") or []

        if not top_3:
            st.markdown("<div class='section-head'>No policy matched</div>", unsafe_allow_html=True)
            if audit.get("reason"):
                st.caption(f"reason: {audit['reason']}")
            st.caption(
                "The honest \"no guidance\" answer is returned without calling the "
                "model at all, so no advice can be invented here."
            )
        else:
            st.markdown(
                "<div class='section-head'>Policies applied · ranked</div>",
                unsafe_allow_html=True,
            )
            for s in top_3:
                sev = str(s.get("severity") or "").lower()
                st.markdown(
                    f"<div class='policy sev-{sev}'>"
                    f"<span class='sop-id'>#{s.get('rank')} · {s.get('sop_id')} — "
                    f"{s.get('name')}</span>{severity_tag(s.get('severity'))}"
                    f"<span class='spec'>spec {s.get('specificity')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if s.get("fuzzy_verdict"):
                    st.caption(f"    fuzzy verdict: {s['fuzzy_verdict']}")
                triggers = {
                    k: v for k, v in (s.get("values_used") or {}).items()
                    if not k.startswith("threshold_")
                }
                if triggers:
                    st.caption(
                        "    triggered by "
                        + ", ".join(f"{k}={v}" for k, v in triggers.items())
                    )

            if audit.get("other_candidates"):
                st.caption(
                    "also matched, ranked below the top 3: "
                    + ", ".join(audit["other_candidates"])
                )
            if audit.get("ranking_rule"):
                st.caption(audit["ranking_rule"])

            grounding = audit.get("grounding") or {}
            violations = grounding.get("violations") or []
            st.markdown(
                "<div class='section-head'>Grounding</div>", unsafe_allow_html=True
            )
            if violations:
                st.markdown(
                    f"<span class='verdict-bad'>✕ violations: "
                    f"{', '.join(violations)}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='verdict-ok'>✓ every weather number traced to "
                    "source</span>",
                    unsafe_allow_html=True,
                )

            notes = []
            if grounding.get("retries"):
                notes.append(f"retried {grounding['retries']}×")
            if grounding.get("fell_back_to_template"):
                notes.append("fell back to deterministic template")
            if notes:
                st.caption(" · ".join(notes))

            claims = grounding.get("claims") or []
            if claims:
                st.markdown(
                    "<div class='section-head'>Per-claim provenance</div>",
                    unsafe_allow_html=True,
                )
                for c in claims:
                    st.markdown(
                        f"<div class='claim'><b>{c.get('value')}"
                        f"{c.get('unit', '')}</b> ← {c.get('source') or 'UNVERIFIED'}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            current = (audit.get("weather_used") or {}).get("current") or {}
            live = ", ".join(
                f"{k}={v}" for k, v in current.items() if isinstance(v, (int, float))
            )
            if live:
                st.markdown(
                    "<div class='section-head'>Live API values</div>",
                    unsafe_allow_html=True,
                )
                st.caption(live)

        location = audit.get("location_resolution") or {}
        if audit.get("resolved_location") or location.get("alias_applied"):
            st.markdown(
                "<div class='section-head'>Resolution</div>", unsafe_allow_html=True
            )
        if audit.get("resolved_location"):
            st.caption(
                f"location: {audit['resolved_location']}  ·  "
                f"activity: {audit.get('activity') or 'unknown'}"
            )
        if location.get("alias_applied"):
            st.caption(f"name mapped: {location['alias_applied']}")
        if location.get("other_candidates"):
            st.caption(
                "other places with this name: "
                + " · ".join(location["other_candidates"])
            )

        audience = audit.get("audience") or {}
        if audience.get("signals") or audience.get("withheld"):
            st.markdown(
                "<div class='section-head'>Audience gating</div>",
                unsafe_allow_html=True,
            )
        if audience.get("signals"):
            st.caption("understood you as: " + ", ".join(audience["signals"]))
        if audience.get("withheld"):
            st.caption("matched the weather but withheld as not addressed to you:")
            for w in audience["withheld"]:
                st.markdown(
                    f"<div class='claim'><b>{w.get('sop_id')}</b> — "
                    f"{w.get('name')}<br>{w.get('reason')}</div>",
                    unsafe_allow_html=True,
                )


def build_profile() -> dict:
    """Sidebar 'About you'. Entirely optional.

    A profile only lets audience-scoped policies (pregnancy, pets, elderly,
    children) resolve; with no profile those policies stay silent rather than
    firing at the wrong reader, which is the fail-closed behaviour in
    utils/audience.py.
    """
    st.sidebar.markdown(
        "<div class='side-title'>Reader profile</div>", unsafe_allow_html=True
    )
    st.sidebar.caption(
        "Optional. Supplying this lets policies written for specific groups "
        "resolve instead of staying silent."
    )

    name = st.sidebar.text_input("Name", key="profile_name")
    home_city = st.sidebar.text_input(
        "Default city", key="profile_city",
        placeholder="used when you don't name one",
    )

    age_label = st.sidebar.selectbox(
        "Age group", ["Not specified", "Child (5-15)", "Adult", "Elderly (60+)"],
        key="profile_age",
    )
    age_group = {
        "Not specified": "", "Child (5-15)": "child",
        "Adult": "adult", "Elderly (60+)": "elderly",
    }[age_label]

    gender_label = st.sidebar.selectbox(
        "Gender", ["Not specified", "Female", "Male", "Other"], key="profile_gender",
    )
    gender = {
        "Not specified": "", "Female": "female", "Male": "male", "Other": "other",
    }[gender_label]

    pregnant = False
    if gender == "female":
        pregnant = st.sidebar.checkbox("Currently pregnant", key="profile_pregnant")

    has_pets = st.sidebar.checkbox("I have pets", key="profile_pets")

    profile = {
        "name": name.strip(),
        "home_city": home_city.strip(),
        "age_group": age_group,
        "gender": gender,
        "pregnant": pregnant,
        "has_pets": has_pets,
    }
    return {k: v for k, v in profile.items() if v}


# --------------------------------------------------------------------- render

st.markdown(STYLES, unsafe_allow_html=True)

_logo = logo_data_uri()
_logo_img = f'<img src="{_logo}" alt="logo">' if _logo else ""

st.markdown(
    f"""
    <div class="masthead">
      <div class="eyebrow">Outdoor Safety · Policy-Grounded Advisory</div>
      <div class="brand">{_logo_img}<h1>Weather Advisory Bot</h1></div>
      <div class="slogan">Know before you go.
        <span class="sub">Check the conditions, get the policy behind the answer,
        and decide with the numbers in front of you.</span>
      </div>
      <p>Answers assembled from live Open-Meteo readings and a library of 32 written
      policies. Which policy applies is decided in Python against the numbers the
      API returned — never by the language model.</p>
      <div class="chips">
        <span class="chip"><span class="dot"></span>Live Open-Meteo</span>
        <span class="chip">32 Policies</span>
        <span class="chip">Deterministic Selection</span>
        <span class="chip">Per-Claim Provenance</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Terms gate: the advisory is not reachable until its limits have been read.
if not st.session_state.get("terms_accepted"):
    terms_gate()
    if HAS_DIALOG:
        st.stop()

profile = build_profile()

st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_history = []
    st.rerun()

if st.sidebar.button("Terms of use", use_container_width=True):
    terms_view()

st.sidebar.markdown(
    f"<div class='footer' style='margin-top:22px'>Issues · suggestions<br>"
    f"<a href='mailto:{CONTACT_EMAIL}'>{CONTACT_EMAIL}</a></div>",
    unsafe_allow_html=True,
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_history", [])

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hint">
          Ask about an outdoor activity and a place. The advisory names the policy
          it applied, and every number it quotes is traceable to the live reading,
          a policy threshold, or the policy text.
          <span class="q">→ Is it safe to cycle in Bhopal today?</span>
          <span class="q">→ Can I take my dog for a walk this evening?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        label = "enquiry" if message["role"] == "user" else "advisory"
        st.markdown(
            f"<div class='msg-label {label}'>{label}</div>", unsafe_allow_html=True
        )
        st.markdown(message["content"])
        if message.get("audit"):
            render_audit(message["audit"])

if prompt := st.chat_input("Ask about outdoor safety…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown("<div class='msg-label enquiry'>enquiry</div>", unsafe_allow_html=True)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.markdown(
            "<div class='msg-label advisory'>advisory</div>", unsafe_allow_html=True
        )
        with st.spinner("Reading live conditions · evaluating policy library…"):
            try:
                result = get_graph().invoke({
                    "user_question": prompt,
                    "session_history": st.session_state.session_history,
                    "profile": profile,
                })
                answer = result.get("final_response", "I couldn't come up with a response.")
                audit = result.get("audit_trail", {})
                st.session_state.session_history = result.get(
                    "session_history", st.session_state.session_history
                )
            except Exception as e:
                # Same last-resort guard main.py carried: an unhandled failure
                # still returns an honest message rather than a stack trace.
                answer = (
                    "Something went wrong on my end and I couldn't process that. "
                    "Please try again."
                )
                audit = {"error": True, "reason": f"{type(e).__name__}: {e}"}

        st.markdown(answer)
        render_audit(audit)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "audit": audit}
    )

st.markdown(
    f"""
    <div class="footer">
      General outdoor-safety guidance · not an official weather warning and not
      medical advice · defer to your national meteorological agency and to
      emergency services.<br>
      Issues, complaints or suggestions —
      <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
    </div>
    """,
    unsafe_allow_html=True,
)
