"""Dataset assembly for the delivery scorer.

Imports FEATURE_ORDER from the serving code rather than redefining it. That
single import is what prevents the train/serve skew that would otherwise make
the §7 ablation meaningless -- if the training script and the product disagree
about column 7, the ablation measures the bug, not the hypothesis.

Three sources, all producing the same schema:
  * `synthetic`  -- generated from a known ground-truth model. Not for the
                    paper: it's for validating that the whole pipeline (fit,
                    ablate, export, load, serve) is correct before a single
                    real label is collected. Debugging a training bug and a
                    data-collection bug at the same time is how weeks vanish.
  * `sessions`   -- feature rows exported from real product sessions via
                    /api/training-data.
  * `csv`        -- an external corpus already reduced to the feature schema.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.features import (  # noqa: E402
    AUDIO_FEATURES, FEATURE_DEFAULTS, FEATURE_ORDER, SCHEMA_VERSION,
    TEXT_FEATURES, VISION_FEATURES,
)

LABEL_NAMES = ["confidence", "clarity", "pace", "presence"]


@dataclass
class Dataset:
    X: np.ndarray                 # (n, n_features)
    y: dict[str, np.ndarray]      # label name -> (n,)
    groups: np.ndarray            # speaker/session id, for grouped splitting
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, feature_names: list[str]) -> "Dataset":
        idx = [self.feature_names.index(f) for f in feature_names]
        return Dataset(self.X[:, idx], self.y, self.groups, list(feature_names))


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    X = np.array(
        [[r["features"].get(f, FEATURE_DEFAULTS[f]) for f in FEATURE_ORDER] for r in rows],
        dtype=np.float64,
    )
    y = {
        name: np.array([r["labels"][name] for r in rows], dtype=np.float64)
        for name in LABEL_NAMES
    }
    groups = np.array([r.get("group", i) for i, r in enumerate(rows)])
    return Dataset(X, y, groups, list(FEATURE_ORDER))


def load_synthetic(n: int = 4000, seed: int = 0, noise: float = 0.12) -> Dataset:
    """Sample features from realistic marginals, then label with a known model.

    The generating function is intentionally non-linear and includes
    interactions (fillers hurt more when you're also rushing), so a linear
    baseline should underperform the trees -- if it doesn't, the training code
    is broken, not the hypothesis.
    """
    rng = np.random.default_rng(seed)

    wpm = rng.normal(145, 35, n).clip(50, 260)
    filler_pm = rng.gamma(2.0, 3.0, n).clip(0, 40)
    restarts = rng.poisson(2.0, n).astype(float)
    pitch_var = rng.lognormal(6.2, 0.8, n).clip(5, 15000)
    energy = rng.normal(0.08, 0.035, n).clip(0.002, 0.5)
    long_pause = rng.gamma(2.0, 500, n).clip(0, 9000)
    mean_pause = rng.gamma(2.0, 220, n).clip(0, 3000)
    eye = rng.beta(5, 3, n)
    posture = rng.beta(6, 3, n)
    gaze_stab = rng.beta(5, 3, n)
    freeze = rng.gamma(1.3, 900, n).clip(0, 12000)
    ttr = rng.beta(6, 5, n)
    duration = rng.gamma(3.0, 25, n).clip(4, 400)

    def band(x, lo_bad, lo_ok, hi_ok, hi_bad):
        out = np.zeros_like(x)
        rising = (x > lo_bad) & (x < lo_ok)
        falling = (x > hi_ok) & (x < hi_bad)
        out[rising] = (x[rising] - lo_bad) / (lo_ok - lo_bad)
        out[falling] = (hi_bad - x[falling]) / (hi_bad - hi_ok)
        out[(x >= lo_ok) & (x <= hi_ok)] = 1.0
        return out

    def decay(x, good, bad):
        return np.clip(1.0 - (x - good) / (bad - good), 0, 1)

    pace = 0.7 * band(wpm, 60, 110, 165, 230) + 0.3 * decay(long_pause, 900, 4000)
    clarity = (
        0.55 * decay(filler_pm, 3, 20)
        + 0.25 * decay(restarts, 1, 10)
        + 0.20 * band(ttr, 0.2, 0.45, 0.85, 1.01)
    )
    # Interaction: fillers are more damaging when the speaker is also rushing.
    clarity -= 0.15 * (filler_pm / 40.0) * np.clip((wpm - 165) / 65.0, 0, 1)
    confidence = (
        0.4 * band(pitch_var, 20, 250, 3000, 12000)
        + 0.3 * band(energy, 0.005, 0.02, 0.30, 0.60)
        + 0.3 * decay(mean_pause, 400, 2000)
    )
    presence = (
        0.4 * band(eye, 0.10, 0.55, 0.92, 1.01)
        + 0.25 * posture + 0.2 * gaze_stab + 0.15 * decay(freeze, 1500, 8000)
    )

    rows = []
    for i in range(n):
        feats = dict(FEATURE_DEFAULTS)
        feats.update({
            "words_per_min": wpm[i], "filler_per_min": filler_pm[i],
            "filler_ratio": filler_pm[i] / max(wpm[i], 1),
            "restart_count": restarts[i], "pitch_var": pitch_var[i],
            "pitch_mean_hz": rng.normal(155, 30),
            "energy_mean": energy[i], "energy_var": energy[i] * 0.02,
            "longest_pause_ms": long_pause[i], "mean_pause_ms": mean_pause[i],
            "pause_count_per_min": rng.gamma(2, 3),
            "speech_ratio": rng.beta(7, 3),
            "eye_contact_pct": eye[i], "posture_score": posture[i],
            "gaze_stability": gaze_stab[i], "posture_drift": rng.beta(2, 6),
            "blink_per_min": rng.normal(18, 6), "head_movement": rng.beta(2, 5),
            "longest_freeze_ms": freeze[i],
            "type_token_ratio": ttr[i], "answer_duration_s": duration[i],
            "word_count": wpm[i] * duration[i] / 60.0,
        })
        labels = {
            "confidence": float(np.clip(confidence[i] + rng.normal(0, noise), 0, 1)),
            "clarity": float(np.clip(clarity[i] + rng.normal(0, noise), 0, 1)),
            "pace": float(np.clip(pace[i] + rng.normal(0, noise), 0, 1)),
            "presence": float(np.clip(presence[i] + rng.normal(0, noise), 0, 1)),
        }
        # Grouped by pseudo-speaker so the split can be speaker-disjoint.
        rows.append({"features": feats, "labels": labels, "group": i % 200})

    return _rows_to_dataset(rows)


def load_from_sessions(path: Path) -> Dataset:
    """Rows exported from /api/training-data, labelled by human ratings.

    Expects each row to carry a `labels` dict. Rows without one are skipped --
    the product's own heuristic scores must NEVER be used as labels, or the
    model just learns to imitate the heuristic and the ablation is circular.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("rows", payload) if isinstance(payload, dict) else payload
    rows = []
    skipped = 0
    for r in raw:
        labels = r.get("labels")
        if not labels or not all(k in labels for k in LABEL_NAMES):
            skipped += 1
            continue
        rows.append({
            "features": r["features"],
            "labels": labels,
            "group": r.get("session_id", r.get("group", len(rows))),
        })
    if skipped:
        print(f"  skipped {skipped} unlabelled rows (human ratings required)")
    if not rows:
        raise ValueError(
            "no labelled rows found. Export with /api/training-data, then attach "
            "human ratings as a `labels` object on each row."
        )
    return _rows_to_dataset(rows)


# Naming matters here. `text_meta_only` is length/lexical-variety features --
# it is a proxy for "what you can know without listening", NOT the content
# rubric. A true content-only arm needs the ContentScorer's relevance/structure/
# completeness outputs, which only exist for rows exported from real sessions
# (they're in each row's `content` dict). With synthetic data there is no
# transcript, so that arm cannot be run and claiming otherwise would overstate
# the ablation.
ABLATIONS = {
    "text_meta_only": TEXT_FEATURES,
    "audio_only": AUDIO_FEATURES,
    "vision_only": VISION_FEATURES,
    "audio_plus_vision": AUDIO_FEATURES + VISION_FEATURES,
    "fused": FEATURE_ORDER,
}

__all__ = [
    "Dataset", "load_synthetic", "load_from_sessions", "ABLATIONS",
    "LABEL_NAMES", "FEATURE_ORDER", "SCHEMA_VERSION",
]
