"""Per-session event trace.

Debugging a turn-taking system from ordinary logs is miserable: the question is
never "did an error happen" but "why did the interviewer respond *there*" --
which is a sequence question, answerable only by seeing VAD events, state
transitions, decisions and their inputs in order, on one timeline.

So every session keeps a ring buffer of structured events. It costs a few KB,
is shipped to the client live when debug mode is on, is queryable over REST
while the session runs, and the tail is persisted into the report so a weird
interview can be diagnosed *after* the fact -- which is when users actually
report it.

Ring buffer, not a list: a 40-minute rambling session must not grow memory
without bound just because tracing is always on.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

MAX_EVENTS = 600           # ~a long session's worth of decisions and states
REPORT_TAIL = 200          # how much survives into the persisted report


@dataclass
class SessionTrace:
    started_at: float = field(default_factory=time.time)
    _events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    _seq: int = 0

    def add(self, kind: str, **detail: Any) -> dict:
        """Record one event. Returns it so callers can also emit it live."""
        self._seq += 1
        event = {
            "seq": self._seq,
            "t_ms": round((time.time() - self.started_at) * 1000),
            "kind": kind,
            **{k: v for k, v in detail.items() if v is not None},
        }
        self._events.append(event)
        return event

    def events(self, limit: int | None = None) -> list[dict]:
        items = list(self._events)
        return items[-limit:] if limit else items

    def tail_for_report(self) -> list[dict]:
        return self.events(REPORT_TAIL)

    def counts(self) -> dict[str, int]:
        """Event-kind histogram -- the 10-second overview of a session."""
        out: dict[str, int] = {}
        for e in self._events:
            out[e["kind"]] = out.get(e["kind"], 0) + 1
        return out
