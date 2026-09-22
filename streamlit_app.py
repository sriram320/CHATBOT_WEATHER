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

import os

import streamlit as st

st.set_page_config(
    page_title="Weather Advisory Bot",
    page_icon="⛅",
    layout="centered",
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


SEVERITY_COLOURS = {
    "critical": "#e5484d",
    "high": "#f5a524",
    "severe": "#e5484d",
    "medium": "#3b82f6",
    "low": "#4caf82",
    "info": "#4caf82",
}


def severity_badge(severity: str) -> str:
    if not severity:
        return ""
    colour = SEVERITY_COLOURS.get(str(severity).lower(), "#6b7280")
    return (
        f"<span style='background:{colour};color:#fff;padding:1px 7px;"
        f"border-radius:10px;font-size:11px;font-weight:600;"
        f"letter-spacing:.3px'>{str(severity).upper()}</span>"
    )


def render_audit(audit: dict) -> None:
    """The 'Why this answer' panel: the same trail the FastAPI page showed.

    This is the part a reviewer actually reads, so it shows the policies that
    fired, the policies that matched the weather but were withheld as not
    addressed to this reader, and the provenance of every number quoted.
    """
    with st.expander("Why this answer", expanded=False):
        if audit.get("error"):
            st.markdown("**Failure path** — no policy was evaluated, because we never got data.")
            st.markdown(f"Reason: `{audit.get('reason', 'unknown')}`")
            st.caption("No forecast is reported when we don't have one.")
            return

        top_3 = audit.get("top_3") or []

        if not top_3:
            st.markdown(f"**No policy matched.** {audit.get('reason', '')}")
            st.caption(
                "The honest \"no guidance\" answer is returned without calling the "
                "model at all, so no advice can be invented here."
            )
        else:
            st.markdown("**Policies applied (ranked)**")
            for s in top_3:
                st.markdown(
                    f"**#{s.get('rank')} {s.get('sop_id')}** — {s.get('name')} "
                    f"{severity_badge(s.get('severity'))} "
                    f"<span style='opacity:.65;font-size:12px'>spec {s.get('specificity')}</span>",
                    unsafe_allow_html=True,
                )
                if s.get("fuzzy_verdict"):
                    st.caption(f"fuzzy verdict: {s['fuzzy_verdict']}")
                triggers = {
                    k: v for k, v in (s.get("values_used") or {}).items()
                    if not k.startswith("threshold_")
                }
                if triggers:
                    st.caption(
                        "triggered by "
                        + ", ".join(f"{k}={v}" for k, v in triggers.items())
                    )

            if audit.get("other_candidates"):
                st.caption(
                    "Also matched, ranked below the top 3: "
                    + ", ".join(audit["other_candidates"])
                )
            if audit.get("ranking_rule"):
                st.caption(audit["ranking_rule"])

            grounding = audit.get("grounding") or {}
            violations = grounding.get("violations") or []
            st.markdown("---")
            if violations:
                st.markdown(
                    f"**Grounding:** :red[violations: {', '.join(violations)}]"
                )
            else:
                st.markdown("**Grounding:** :green[all weather numbers traced]")

            notes = []
            if grounding.get("retries"):
                notes.append(f"retried {grounding['retries']}×")
            if grounding.get("fell_back_to_template"):
                notes.append("fell back to deterministic template")
            if notes:
                st.caption(" · ".join(notes))

            for claim in grounding.get("claims") or []:
                st.caption(
                    f"{claim.get('value')}{claim.get('unit', '')} "
                    f"← {claim.get('source') or 'UNVERIFIED'}"
                )

            current = (audit.get("weather_used") or {}).get("current") or {}
            live = ", ".join(
                f"{k}={v}" for k, v in current.items() if isinstance(v, (int, float))
            )
            if live:
                st.markdown("**Live API values used**")
                st.caption(live)

        if audit.get("resolved_location"):
            st.caption(
                f"Location resolved to: {audit['resolved_location']} · "
                f"activity: {audit.get('activity') or 'unknown'}"
            )

        location = audit.get("location_resolution") or {}
        if location.get("alias_applied"):
            st.caption(f"Name mapped: {location['alias_applied']}")
        if location.get("other_candidates"):
            st.caption(
                "Other places with this name: " + " · ".join(location["other_candidates"])
            )

        audience = audit.get("audience") or {}
        if audience.get("signals"):
            st.caption("Who we understood you to be: " + ", ".join(audience["signals"]))
        if audience.get("withheld"):
            st.markdown("**Policies that matched the weather but were not for you**")
            for w in audience["withheld"]:
                st.caption(f"{w.get('sop_id')} — {w.get('name')}: {w.get('reason')}")


def build_profile() -> dict:
    """Sidebar 'About you'. Entirely optional.

    A profile only lets audience-scoped policies (pregnancy, pets, elderly,
    children) resolve; with no profile those policies stay silent rather than
    firing at the wrong reader, which is the fail-closed behaviour in
    utils/audience.py.
    """
    st.sidebar.header("About you")
    st.sidebar.caption(
        "Optional. Supplying this lets policies written for specific groups "
        "resolve correctly instead of staying silent."
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


st.title("⛅ Weather Advisory Bot")
st.caption(
    "Answers grounded in live Open-Meteo data and a library of 32 written "
    "policies. Which policy applies is decided in Python, never by the model."
)

profile = build_profile()

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_history = []
    st.rerun()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_history", [])

if not st.session_state.messages:
    st.info(
        "Try: *Is it safe to cycle in Bhopal today?* — or set a profile in the "
        "sidebar and ask *Can I take my dog for a walk?*"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audit"):
            render_audit(message["audit"])

if prompt := st.chat_input("Ask about outdoor safety…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking live weather and policy library…"):
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
