"""In-process rate limiting.

Deliberately not Redis. A session pins a WebSocket and ~350MB of shared models
to one worker, so the deployment is single-worker with sticky routing anyway --
a distributed limiter would add an operational dependency to enforce a limit
that only ever needs to be true for one process.

Sliding window rather than a fixed bucket: fixed windows let someone burn the
full hour's allowance in the last second of one window and again in the first
second of the next.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class SlidingWindowLimiter:
    limit: int
    window_s: float = 3600.0
    max_keys: int = 10_000

    _hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def _evict(self, key: str, now: float) -> deque[float]:
        q = self._hits[key]
        cutoff = now - self.window_s
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, seconds_until_retry). Does not consume."""
        now = time.time()
        q = self._evict(key, now)
        if len(q) < self.limit:
            return True, 0
        return False, max(1, int(q[0] + self.window_s - now))

    def hit(self, key: str) -> tuple[bool, int]:
        """Consume one unit if allowed."""
        allowed, retry = self.check(key)
        if allowed:
            self._hits[key].append(time.time())
            self._sweep()
        return allowed, retry

    def _sweep(self) -> None:
        """Drop empty keys so a long-running process can't leak memory.

        Without this, every unique client IP leaves a permanent dict entry --
        slow, but unbounded, which is the kind of leak that only shows up in
        the one environment you can't restart.
        """
        if len(self._hits) <= self.max_keys:
            return
        now = time.time()
        stale = [k for k, q in self._hits.items() if not q or q[-1] < now - self.window_s]
        for k in stale:
            del self._hits[k]
