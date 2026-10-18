"""Delivery scoring: feature vector -> {confidence, clarity, pace, presence}.

Two paths:
  * TrainedDeliveryScorer -- LightGBM heads produced by training/train_delivery.py.
    Refuses to load against a mismatched feature SCHEMA_VERSION.
  * HeuristicDeliveryScorer -- piecewise-linear scoring from published speech
    norms. Not a placeholder returning 0.5: it produces defensible, monotone,
    explainable scores, and it is the ablation's honest baseline.

Every score carries `evidence` (which features moved it, and how far) and a
`confidence` derived from modality coverage and answer length. The product
promise in the spec -- "let users see exactly which evidence drove each score"
-- is only real if the scorer emits it, so it does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..features import SCHEMA_VERSION, DeliverySubscores, FeatureVector

log = logging.getLogger(__name__)


def _band(x: float, lo_bad: float, lo_ok: float, hi_ok: float, hi_bad: float) -> float:
    """Trapezoid: 1.0 inside [lo_ok, hi_ok], ramping to 0.0 outside the bad edges.

    Models "there is a healthy range and both extremes are worse", which is true
    of nearly every delivery feature and false of the linear scaling most naive
    scorers use (they reward talking infinitely fast).
    """
    if x <= lo_bad or x >= hi_bad:
        return 0.0
    if lo_ok <= x <= hi_ok:
        return 1.0
    if x < lo_ok:
        return (x - lo_bad) / max(lo_ok - lo_bad, 1e-6)
    return (hi_bad - x) / max(hi_bad - hi_ok, 1e-6)


def _decay(x: float, good: float, bad: float) -> float:
    """1.0 at or below `good`, 0.0 at or above `bad`, linear between."""
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / max(bad - good, 1e-6)


@dataclass
class DeliveryResult:
    subscores: DeliverySubscores
    overall: float
    confidence: float
    evidence: list[dict] = field(default_factory=list)
    backend: str = "heuristic"

    def to_dict(self) -> dict:
        return {
            "subscores": self.subscores.to_dict(),
            "overall": self.overall,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "backend": self.backend,
        }


class HeuristicDeliveryScorer:
    name = "heuristic"

    def score(self, fv: FeatureVector) -> DeliveryResult:
        v = fv.values
        ev: list[dict] = []

        def note(feature: str, value: float, contribution: float, msg: str) -> None:
            ev.append(
                {
                    "feature": feature,
                    "value": round(value, 3),
                    "contribution": round(contribution, 3),
                    "note": msg,
                }
            )

        # --- pace: wpm band + pause discipline -------------------------------
        wpm = v["words_per_min"]
        pace_wpm = _band(wpm, 60, 110, 165, 230)
        pause_pen = _decay(v["longest_pause_ms"], 900, 4000)
        pace = 0.7 * pace_wpm + 0.3 * pause_pen
        note("words_per_min", wpm, pace_wpm - 0.5,
             "comfortable range" if pace_wpm > 0.8
             else ("rushing" if wpm > 165 else "dragging"))
        if v["longest_pause_ms"] > 1500:
            note("longest_pause_ms", v["longest_pause_ms"], pause_pen - 1.0, "long dead air")

        # --- clarity: fillers, restarts, lexical variety ----------------------
        fpm = v["filler_per_min"]
        filler_s = _decay(fpm, 3.0, 20.0)
        restart_s = _decay(v["restart_count"], 1.0, 10.0)
        variety = _band(v["type_token_ratio"], 0.2, 0.45, 0.85, 1.01)
        clarity = 0.55 * filler_s + 0.25 * restart_s + 0.20 * variety
        note("filler_per_min", fpm, filler_s - 0.5,
             "clean delivery" if fpm < 4 else f"{fpm:.0f} fillers/min is audible")
        if v["restart_count"] >= 2:
            note("restart_count", v["restart_count"], restart_s - 1.0, "false starts break flow")

        # --- confidence: pitch variance, energy, hesitation -------------------
        # Monotone pitch reads as flat/nervous; wild pitch reads as unsteady.
        pv = _band(v["pitch_var"], 20, 250, 3000, 12000)
        energy = _band(v["energy_mean"], 0.005, 0.02, 0.30, 0.60)
        steadiness = _decay(v["mean_pause_ms"], 400, 2000)
        conf = 0.4 * pv + 0.3 * energy + 0.3 * steadiness
        note("pitch_var", v["pitch_var"], pv - 0.5,
             "expressive range" if pv > 0.8 else "monotone")
        note("energy_mean", v["energy_mean"], energy - 0.5,
             "projecting well" if energy > 0.8 else "quiet / low projection")

        # --- presence: gaze + posture (vision only) ---------------------------
        if fv.has_vision:
            eye = _band(v["eye_contact_pct"], 0.10, 0.55, 0.92, 1.01)
            posture = v["posture_score"]
            stability = v["gaze_stability"]
            freeze = _decay(v["longest_freeze_ms"], 1500, 8000)
            presence = 0.4 * eye + 0.25 * posture + 0.2 * stability + 0.15 * freeze
            note("eye_contact_pct", v["eye_contact_pct"], eye - 0.5,
                 "steady eye contact" if eye > 0.8 else "looking away often")
            if v["longest_freeze_ms"] > 2000:
                note("longest_freeze_ms", v["longest_freeze_ms"], freeze - 1.0,
                     "froze / went still")
        else:
            # No camera: abstain rather than invent. Presence is pinned to the
            # neutral prior and confidence drops accordingly.
            presence = 0.65
            note("eye_contact_pct", 0.0, 0.0, "camera off -- presence not scored")

        subs = DeliverySubscores(
            confidence=round(min(max(conf, 0.0), 1.0), 3),
            clarity=round(min(max(clarity, 0.0), 1.0), 3),
            pace=round(min(max(pace, 0.0), 1.0), 3),
            presence=round(min(max(presence, 0.0), 1.0), 3),
        )

        # Short answers carry little signal -- say so instead of over-claiming.
        length_conf = _band(v["answer_duration_s"], 3, 15, 240, 600)
        confidence = round(0.6 * fv.coverage() + 0.4 * length_conf, 3)
        ev.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        return DeliveryResult(subs, subs.overall(), confidence, ev[:6], self.name)


class TrainedDeliveryScorer:
    """LightGBM heads, one per sub-score, sharing the FEATURE_ORDER vector."""

    name = "lightgbm"

    def __init__(self, model_path: Path) -> None:
        import lightgbm as lgb

        meta_path = model_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"delivery model was trained on feature schema v{meta.get('schema_version')}, "
                f"runtime is v{SCHEMA_VERSION} -- retrain before serving"
            )
        self._boosters = {
            name: lgb.Booster(model_file=str(model_path.with_suffix(f".{name}.txt")))
            for name in DeliverySubscores.names()
        }
        self._fallback = HeuristicDeliveryScorer()

    # Sub-scores that cannot be computed without a given modality. A tree will
    # happily predict from the neutral defaults and return a confident-looking
    # number, which is precisely the "confident wrong answer where abstention
    # was correct" that features.py warns about.
    _REQUIRES_VISION = ("presence",)

    def score(self, fv: FeatureVector) -> DeliveryResult:
        import numpy as np

        x = np.asarray([fv.as_list()], dtype=np.float64)
        preds = {
            name: float(np.clip(b.predict(x)[0], 0.0, 1.0)) for name, b in self._boosters.items()
        }

        # Reuse the heuristic's evidence: the trees give us numbers, the bands
        # give us the human-readable "why", and they read the same features.
        base = self._fallback.score(fv)

        # Honour modality abstention exactly as the heuristic does. Without
        # this the report contradicts itself -- stating "camera off, presence
        # not scored" while displaying a presence score of 81.
        if not fv.has_vision:
            for name in self._REQUIRES_VISION:
                preds[name] = getattr(base.subscores, name)

        subs = DeliverySubscores(**{k: round(v, 3) for k, v in preds.items()})
        return DeliveryResult(subs, subs.overall(), base.confidence, base.evidence, self.name)


def build_delivery_scorer(model_path: Path):
    if model_path.exists():
        try:
            scorer = TrainedDeliveryScorer(model_path)
            log.info("Delivery scorer: lightgbm (%s)", model_path)
            return scorer
        except Exception as exc:  # noqa: BLE001
            log.warning("Delivery scorer: heuristic (trained model unusable: %s)", exc)
    else:
        log.info("Delivery scorer: heuristic (no trained artefact at %s)", model_path)
    return HeuristicDeliveryScorer()
