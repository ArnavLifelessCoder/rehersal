"""FastAPI app: REST for session setup/reports, WebSocket for the live interview.

WebSocket protocol
------------------
Binary frames  : raw int16 PCM at settings.sample_rate, mono. Chosen over JSON
                 base64 because it halves bandwidth and removes a decode step
                 from the hot path.
Text frames    : JSON control messages.

  client -> server
    {"type": "cv", "ts_ms": int, "eye_contact": bool, "gaze_offset": float,
     "posture": float, "blink": bool, "head_movement": float, "face_present": bool}
    {"type": "playback_done"}    -- the interviewer's audio finished playing
    {"type": "end"}              -- candidate ended the session
    {"type": "ping"}

  server -> client
    {"type": "ready", "session_id": str, "backends": {...}}
    {"type": "state", "state": "listening"|"speaking"|"thinking"|"ended", ...}
    {"type": "transcript", "text": str, "final": bool}
    {"type": "speech", "chunk": {"text": str, "audio": b64|null, "mime": str}}
    {"type": "speech_end", "text": str, "move": str}
    {"type": "turn_scored", "turn": {...}}
    {"type": "latency", "timings": {...}}
    {"type": "report", "report": {...}}
    {"type": "error", "message": str}
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .deps import build_deps
from .ratelimit import SlidingWindowLimiter
from .session.manager import SessionRegistry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("rehearsal")

VALID_TRACKS = {"dsa", "system_design", "behavioral", "jd"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.deps = build_deps(settings)
    app.state.registry = SessionRegistry()
    app.state.limiter = SlidingWindowLimiter(limit=settings.rate_limit_sessions_per_hour)
    yield
    for sid in list(app.state.registry._sessions):
        await app.state.registry.drop(sid)


app = FastAPI(title="Rehearsal", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    track: str = Field(default="behavioral")
    jd_text: str | None = None


@app.get("/api/health")
async def health() -> dict:
    s = app.state.settings
    return {
        "status": "ok",
        "backends": app.state.deps.describe(),
        "active_sessions": app.state.registry.active_count(),
        "latency_budget_ms": s.total_response_budget_ms,
    }


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest, request: Request) -> dict:
    if req.track not in VALID_TRACKS:
        raise HTTPException(400, f"track must be one of {sorted(VALID_TRACKS)}")
    if req.track == "jd" and not (req.jd_text or "").strip():
        raise HTTPException(400, "track 'jd' requires jd_text")

    s = app.state.settings
    client = request.client.host if request.client else "unknown"
    allowed, retry = app.state.limiter.hit(client)
    if not allowed:
        raise HTTPException(
            429, f"too many interviews started. Try again in {retry // 60 + 1} minutes.",
            headers={"Retry-After": str(retry)},
        )
    # Each live session pins a WebSocket plus its own decode buffers; past this
    # the box degrades for everyone rather than refusing one person.
    if app.state.registry.active_count() >= s.max_concurrent_sessions:
        raise HTTPException(503, "the server is at capacity right now. Try again shortly.")

    import uuid

    sid = uuid.uuid4().hex[:12]
    app.state.deps.store.create_session(sid, req.track, req.jd_text)
    return {"session_id": sid, "track": req.track, "ws_url": f"/ws/{sid}"}


@app.get("/api/sessions/{session_id}/events")
async def session_events(session_id: str, limit: int = 200) -> dict:
    """The session's event trace: live if running, from the report if finished.

    curl-able mid-interview, which matters -- turn-taking bugs are easiest to
    see while they're happening, and attaching a debugger to a WebSocket
    actor is nobody's idea of easiest.
    """
    live = app.state.registry.get(session_id)
    if live is not None:
        return {
            "session_id": session_id,
            "live": True,
            "state": live.state.state.value,
            "counts": live.trace.counts(),
            "events": live.trace.events(limit=min(max(limit, 1), 600)),
        }
    report = app.state.deps.store.get_report(session_id)
    if report is None:
        raise HTTPException(404, "no such session")
    return {
        "session_id": session_id,
        "live": False,
        "counts": report.get("event_counts", {}),
        "events": (report.get("debug_events") or [])[-limit:],
    }


@app.get("/api/progress")
async def progress(limit: int = 40) -> dict:
    """Session-over-session trend: am I actually getting better?"""
    return app.state.deps.store.progress(min(max(limit, 1), 200))


@app.get("/api/sessions/{session_id}/turns/{turn_idx}/audio")
async def turn_audio(session_id: str, turn_idx: int) -> Response:
    """The recording behind a worst-moment citation.

    Without this the report's timestamps are unfalsifiable -- it tells you that
    you said "um" nine times at 12.4s and gives you no way to check.
    """
    blobs = app.state.deps.blobs
    if blobs is None:
        raise HTTPException(404, "recordings are disabled on this server")
    data = blobs.read(session_id, turn_idx)
    if data is None:
        raise HTTPException(404, "no recording for that turn (it may have been pruned)")
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{session_id}-turn{turn_idx}.wav"',
        },
    )


@app.post("/api/sessions/{session_id}/star")
async def star_session(session_id: str, starred: bool = True) -> dict:
    """Starring keeps a session's recordings past the retention window."""
    if not app.state.deps.store.set_starred(session_id, starred):
        raise HTTPException(404, "no such session")
    return {"session_id": session_id, "starred": starred}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session and its recordings.

    The product records someone's face and voice while judging them. Being able
    to remove that is not a feature, it's the baseline.
    """
    if app.state.deps.blobs:
        app.state.deps.blobs.delete_session(session_id)
    removed = app.state.deps.store.delete_session(session_id)
    if not removed:
        raise HTTPException(404, "no such session")
    return {"deleted": session_id}


@app.get("/api/sessions")
async def list_sessions(limit: int = 50) -> dict:
    return {"sessions": app.state.deps.store.list_sessions(limit)}


@app.get("/api/sessions/{session_id}/report")
async def get_report(session_id: str) -> dict:
    report = app.state.deps.store.get_report(session_id)
    if not report:
        raise HTTPException(404, "no report for that session (is the interview finished?)")
    return report


@app.get("/api/training-data")
async def training_data() -> dict:
    """Export stored feature vectors for retraining. See training/README.md."""
    rows = app.state.deps.store.export_training_rows()
    return {"count": len(rows), "rows": rows}


@app.websocket("/ws/{session_id}")
async def interview_ws(ws: WebSocket, session_id: str, track: str = "behavioral",
                       jd_text: str | None = None) -> None:
    await ws.accept()
    settings, deps, registry = app.state.settings, app.state.deps, app.state.registry

    if track not in VALID_TRACKS:
        await ws.close(code=4400, reason="invalid track")
        return

    async def emit(message: dict) -> None:
        try:
            await ws.send_text(json.dumps(message))
        except (WebSocketDisconnect, RuntimeError):
            pass  # client vanished; the actor gets torn down in finally

    session = registry.create(
        session_id=session_id, track=track, jd_text=jd_text,
        settings=settings, deps=deps, emit=emit,
    )
    await emit({
        "type": "ready",
        "session_id": session.id,
        "backends": deps.describe(),
        "question_budget": settings.max_questions_per_session,
    })
    await session.start()

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                # int16 LE -> float32 in [-1, 1], the format every model wants.
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                await session.push_audio(pcm)
                continue

            if (text := message.get("text")) is None:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await emit({"type": "error", "message": "malformed JSON"})
                continue

            kind = payload.get("type")
            if kind == "cv":
                await session.push_cv(payload)
            elif kind == "debug":
                await session.set_debug(bool(payload.get("enabled", False)))
            elif kind == "playback_done":
                await session.notify_playback_done()
            elif kind == "end":
                await session.request_end()
            elif kind == "ping":
                await emit({"type": "pong"})
            else:
                await emit({"type": "error", "message": f"unknown message type: {kind}"})

    except WebSocketDisconnect:
        log.info("session %s: client disconnected", session_id)
    except Exception:  # noqa: BLE001
        log.exception("session %s: websocket error", session_id)
    finally:
        await registry.drop(session_id)
