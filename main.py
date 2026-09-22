"""FastAPI app: serves the chat frontend and the /chat endpoint.

Session state is a plain in-memory dict keyed by session_id, holding each
session's structured turn history (see graph.py's return_response). This
resets on process restart -- intentional, per the brief: memory persists
within a session, not across app restarts or across users.
"""

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import build_graph

app = FastAPI(title="Weather Advisory Bot")

_graph = build_graph()
_sessions: dict[str, list[dict]] = {}
# Last profile seen for a session, so a profile set once keeps applying to
# later turns without the page resending it.
_profiles: dict[str, dict] = {}

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    # Optional. Everything still works with no profile at all; supplying one
    # only lets audience-scoped policies (pregnancy, pets, elderly, children)
    # resolve correctly instead of staying silent.
    profile: dict | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    audit_trail: dict


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    if req.profile:
        _profiles[session_id] = {**_profiles.get(session_id, {}), **req.profile}
    profile = _profiles.get(session_id, {})

    try:
        result = _graph.invoke({
            "user_question": req.message,
            "session_history": history,
            "profile": profile,
        })
    except Exception as e:
        # Last-resort guard: any unhandled exception still gets an honest
        # message back to the user instead of a raw 500.
        return ChatResponse(
            session_id=session_id,
            response=f"Something went wrong on my end and I couldn't process that. Please try again.",
            audit_trail={"error": True, "exception": f"{type(e).__name__}: {e}"},
        )

    _sessions[session_id] = result.get("session_history", history)

    return ChatResponse(
        session_id=session_id,
        response=result.get("final_response", "I couldn't come up with a response."),
        audit_trail=result.get("audit_trail", {}),
    )


@app.post("/reset")
def reset(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    _profiles.pop(session_id, None)
    return {"ok": True}
