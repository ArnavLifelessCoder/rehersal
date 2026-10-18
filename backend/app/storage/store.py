"""Persistence facade.

Writes are best-effort by design: a database hiccup must never take down a live
interview. The session is the source of truth while it runs; the DB is where it
goes to be read later. So every method here swallows and logs rather than
raising into the actor loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import Session as SessionRow
from .models import SessionReport, Turn, make_engine, make_session_factory

if TYPE_CHECKING:  # pragma: no cover
    from ..session.manager import InterviewSession

log = logging.getLogger(__name__)

# Below this many sessions, any "trend" is noise dressed up as a finding.
# Six, not four: the comparison window is a third of the sessions, so at four
# or five it collapses to one-vs-one -- a first-versus-last delta, which is the
# exact thing the thirds-based comparison exists to avoid. Six is the smallest
# count that gives at least two sessions on each side.
MIN_SESSIONS_FOR_TREND = 6
MIN_TREND_WINDOW = 2

# Smallest change worth calling a change, per metric. Without this, a delta of
# 0.005 on a 0-1 score renders as "+1 better" -- noise presented as progress,
# from a product whose whole posture is refusing to overclaim.
MEANINGFUL_DELTA = {
    "delivery": 0.02,          # 2 points on a 0-100 scale
    "content": 0.02,
    "filler_per_min": 0.5,     # half a filler per minute
}


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _trend(points: list[dict]) -> dict:
    """Compare the first third of sessions to the last third.

    Refuses to report anything below MIN_SESSIONS_FOR_TREND. Telling someone
    their filler rate "improved 40%" off two sessions is exactly the kind of
    confident nonsense this product is supposed to not do.
    """
    if len(points) < MIN_SESSIONS_FOR_TREND:
        return {
            "available": False,
            "sessions": len(points),
            "needed": MIN_SESSIONS_FOR_TREND,
        }

    third = max(MIN_TREND_WINDOW, len(points) // 3)
    early, late = points[:third], points[-third:]
    out: dict = {"available": True, "sessions": len(points), "window": third}

    for metric, lower_is_better in (
        ("delivery", False), ("content", False), ("filler_per_min", True),
    ):
        a = _mean([p[metric] for p in early])
        b = _mean([p[metric] for p in late])
        if a is None or b is None:
            continue
        delta = b - a
        threshold = MEANINGFUL_DELTA.get(metric, 0.0)
        if abs(delta) < threshold:
            improved = None            # real, but too small to call either way
        else:
            improved = delta < 0 if lower_is_better else delta > 0
        out[metric] = {
            "early": round(a, 3),
            "late": round(b, 3),
            "delta": round(delta, 3),
            "improved": improved,
        }
    return out


class Store:
    def __init__(self, database_url: str) -> None:
        self._engine = make_engine(database_url)
        self._factory = make_session_factory(self._engine)
        self.url = database_url

    def create_session(self, session_id: str, track: str, jd_text: str | None) -> None:
        try:
            with self._factory() as db:
                db.add(SessionRow(id=session_id, track=track, jd_text=jd_text, status="active"))
                db.commit()
        except Exception:  # noqa: BLE001
            log.exception("store: failed to create session row %s", session_id)

    def save_session(self, session: "InterviewSession", report: dict) -> None:
        try:
            with self._factory() as db:
                row = db.get(SessionRow, session.id)
                if row is None:
                    row = SessionRow(id=session.id, track=session.track, jd_text=session.jd_text)
                    db.add(row)
                row.status = "completed"
                row.ended_at = datetime.now(timezone.utc)

                db.query(Turn).filter(Turn.session_id == session.id).delete()
                for t in session.turns:
                    db.add(
                        Turn(
                            session_id=session.id, idx=t.idx, role=t.role,
                            question_text=t.question_text, transcript=t.transcript,
                            duration_s=t.duration_s, features=t.features,
                            delivery=t.delivery, content=t.content, fillers=t.fillers,
                            timings=t.timings, audio_url=t.audio_path,
                        )
                    )
                existing = db.get(SessionReport, session.id)
                if existing:
                    db.delete(existing)
                    db.flush()
                db.add(
                    SessionReport(
                        session_id=session.id,
                        overall_delivery=report["overall_delivery"],
                        overall_content=report["overall_content"],
                        confidence=report["confidence"],
                        payload=report,
                    )
                )
                db.commit()
                log.info("store: persisted session %s (%d turns)", session.id, len(session.turns))
        except Exception:  # noqa: BLE001
            log.exception("store: failed to persist session %s", session.id)

    def delete_session(self, session_id: str) -> bool:
        try:
            with self._factory() as db:
                row = db.get(SessionRow, session_id)
                if not row:
                    return False
                db.query(Turn).filter(Turn.session_id == session_id).delete()
                if (rep := db.get(SessionReport, session_id)) is not None:
                    db.delete(rep)
                db.delete(row)
                db.commit()
                return True
        except Exception:  # noqa: BLE001
            log.exception("store: failed to delete %s", session_id)
            return False

    def get_report(self, session_id: str) -> dict | None:
        try:
            with self._factory() as db:
                row = db.get(SessionReport, session_id)
                return row.payload if row else None
        except Exception:  # noqa: BLE001
            log.exception("store: failed to read report %s", session_id)
            return None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        try:
            with self._factory() as db:
                rows = (
                    db.query(SessionRow)
                    .order_by(SessionRow.started_at.desc())
                    .limit(limit)
                    .all()
                )
                out = []
                for r in rows:
                    rep = db.get(SessionReport, r.id)
                    out.append(
                        {
                            "id": r.id,
                            "track": r.track,
                            "status": r.status,
                            "started_at": r.started_at.isoformat() if r.started_at else None,
                            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                            "overall_delivery": rep.overall_delivery if rep else None,
                            "overall_content": rep.overall_content if rep else None,
                        }
                    )
                return out
        except Exception:  # noqa: BLE001
            log.exception("store: failed to list sessions")
            return []

    def set_starred(self, session_id: str, starred: bool) -> bool:
        """Starred sessions keep their recordings past the retention window."""
        try:
            with self._factory() as db:
                row = db.get(SessionRow, session_id)
                if not row:
                    return False
                row.starred = 1 if starred else 0
                db.commit()
                return True
        except Exception:  # noqa: BLE001
            log.exception("store: failed to star %s", session_id)
            return False

    def starred_ids(self) -> set[str]:
        try:
            with self._factory() as db:
                return {r.id for r in db.query(SessionRow).filter(SessionRow.starred == 1).all()}
        except Exception:  # noqa: BLE001
            return set()

    def progress(self, limit: int = 40) -> dict:
        """Session-over-session trend -- the longitudinal view.

        This is both a product feature ("am I actually getting better?") and
        the raw material for the spec's optional user study. Returned oldest
        first so a chart reads left to right without the client re-sorting.

        The `trend` numbers compare the first and last thirds rather than
        first-vs-last session: single sessions are noisy enough that a
        first-to-last delta mostly measures which two days you happened to pick.
        """
        try:
            with self._factory() as db:
                rows = (
                    db.query(SessionRow)
                    .filter(SessionRow.status == "completed")
                    .order_by(SessionRow.started_at.desc())
                    .limit(limit)
                    .all()
                )
                points = []
                for r in reversed(rows):
                    rep = db.get(SessionReport, r.id)
                    if not rep or not rep.payload:
                        continue
                    p = rep.payload
                    points.append({
                        "session_id": r.id,
                        "track": r.track,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                        "delivery": p.get("overall_delivery"),
                        "content": p.get("overall_content"),
                        "filler_per_min": p.get("filler_per_min"),
                        "answers": p.get("answers_given"),
                        "speaking_time_s": p.get("speaking_time_s"),
                        "confidence": p.get("confidence"),
                        "starred": bool(r.starred),
                    })
        except Exception:  # noqa: BLE001
            log.exception("store: failed to build progress")
            return {"sessions": [], "trend": {}}

        return {"sessions": points, "trend": _trend(points)}

    def export_training_rows(self) -> list[dict]:
        """Feature vectors + labels for retraining the delivery scorer.

        This is the loop that closes the product back into the research: every
        real session becomes a candidate training row, using the stored feature
        vector rather than re-deriving anything.
        """
        try:
            with self._factory() as db:
                rows = db.query(Turn).filter(Turn.role == "candidate").all()
                return [
                    {
                        "session_id": t.session_id, "turn_idx": t.idx,
                        "features": t.features, "delivery": t.delivery,
                        "content": t.content, "transcript": t.transcript,
                    }
                    for t in rows if t.features
                ]
        except Exception:  # noqa: BLE001
            log.exception("store: failed to export training rows")
            return []
