"""Aggregates the browser's per-frame CV samples into per-turn features.

The client streams ~5Hz samples of {eye_contact, posture, blink, head_pose}.
The server never sees pixels -- this module is the entire server-side vision
surface, which is what makes the privacy claim in the architecture real rather
than aspirational.

Two things it must get right:
  * Freeze detection. "Longest freeze" means the candidate went unnaturally
    still (a stress tell), which is NOT the same as the camera dropping out. We
    distinguish them: missing samples are a gap, not a freeze. Conflating the
    two would penalise people for wifi.
  * Windowing. Features are accumulated per candidate turn and reset on turn
    boundaries, so a great answer isn't dragged down by the previous one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CVSample:
    ts_ms: int
    eye_contact: bool
    gaze_offset: float      # 0 = centred, 1 = fully averted
    posture: float          # 0..1, higher = more upright
    blink: bool
    head_movement: float    # normalised inter-frame head displacement
    face_present: bool = True
    # How well framed the face was. Iris landmarks degrade badly when the face
    # occupies little of the frame, and confidently scoring someone's gaze from
    # a 40-pixel face is worse than admitting we couldn't see them.
    quality: float = 1.0


# Below this mean framing quality the landmarks aren't worth scoring.
MIN_USABLE_QUALITY = 0.25


@dataclass
class CVAggregator:
    """Streaming accumulator; O(1) memory per feature except the ts list."""

    samples: list[CVSample] = field(default_factory=list)
    _gap_threshold_ms: int = 700   # beyond this, treat as camera dropout

    def add(self, sample: CVSample) -> None:
        self.samples.append(sample)

    def reset(self) -> None:
        self.samples.clear()

    @property
    def has_data(self) -> bool:
        return any(s.face_present for s in self.samples)

    def mean_quality(self) -> float:
        present = [s for s in self.samples if s.face_present]
        return float(np.mean([s.quality for s in present])) if present else 0.0

    def features(self) -> dict[str, float]:
        present = [s for s in self.samples if s.face_present]
        if not present:
            return {}
        # Too poorly framed to score. Returning {} routes this through the same
        # path as "camera off" -- presence abstains to its neutral prior and
        # confidence drops -- rather than emitting numbers we don't believe.
        if float(np.mean([s.quality for s in present])) < MIN_USABLE_QUALITY:
            return {}

        ts = np.array([s.ts_ms for s in present], dtype=np.float64)
        span_s = max((ts[-1] - ts[0]) / 1000.0, 1e-3)

        eye = np.array([s.eye_contact for s in present], dtype=np.float64)
        gaze = np.array([s.gaze_offset for s in present], dtype=np.float64)
        posture = np.array([s.posture for s in present], dtype=np.float64)
        head = np.array([s.head_movement for s in present], dtype=np.float64)
        blinks = sum(1 for s in present if s.blink)

        # Freeze = a run of near-zero head movement across *contiguous* samples.
        # A gap in timestamps breaks the run rather than extending it.
        longest_freeze = 0.0
        run_start: float | None = None
        prev_ts: float | None = None
        for s in present:
            gap = (s.ts_ms - prev_ts) if prev_ts is not None else 0.0
            if prev_ts is not None and gap > self._gap_threshold_ms:
                run_start = None          # dropout, not stillness
            if s.head_movement < 0.015:
                if run_start is None:
                    run_start = s.ts_ms
                longest_freeze = max(longest_freeze, s.ts_ms - run_start)
            else:
                run_start = None
            prev_ts = s.ts_ms

        # Gaze stability: low variance in gaze offset = a steady, settled gaze.
        gaze_stability = float(np.clip(1.0 - np.std(gaze) * 2.5, 0.0, 1.0))
        posture_drift = float(np.clip(np.std(posture) * 2.0, 0.0, 1.0))

        return {
            "eye_contact_pct": float(np.mean(eye)),
            "gaze_stability": gaze_stability,
            "posture_score": float(np.mean(posture)),
            "posture_drift": posture_drift,
            "blink_per_min": float(blinks / (span_s / 60.0)),
            "head_movement": float(np.mean(head)),
            "longest_freeze_ms": float(longest_freeze),
        }

    def timeline(self, bucket_ms: int = 2000) -> list[dict]:
        """Down-sampled series for the report chart."""
        present = [s for s in self.samples if s.face_present]
        if not present:
            return []
        out: list[dict] = []
        start = present[0].ts_ms
        bucket: list[CVSample] = []
        bucket_start = start
        for s in present:
            if s.ts_ms - bucket_start >= bucket_ms and bucket:
                out.append(
                    {
                        "ts_ms": bucket_start - start,
                        "eye_contact": round(float(np.mean([b.eye_contact for b in bucket])), 3),
                        "posture": round(float(np.mean([b.posture for b in bucket])), 3),
                    }
                )
                bucket, bucket_start = [], s.ts_ms
            bucket.append(s)
        if bucket:
            out.append(
                {
                    "ts_ms": bucket_start - start,
                    "eye_contact": round(float(np.mean([b.eye_contact for b in bucket])), 3),
                    "posture": round(float(np.mean([b.posture for b in bucket])), 3),
                }
            )
        return out
