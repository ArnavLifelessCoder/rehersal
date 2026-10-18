"""The feature contract.

This module is the *single* definition of the delivery feature vector. The live
pipeline and the offline training script both import FEATURE_ORDER from here, so
a feature added in one place cannot silently drift out of order in the other --
the classic train/serve skew that quietly poisons a fused-model ablation.

Anything appended must go at the END of FEATURE_ORDER, and SCHEMA_VERSION must
bump. Trained artefacts record the version they were fitted on and refuse to
load against a mismatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

SCHEMA_VERSION = 1

# (name, neutral_default) -- the default is what we feed the model when a
# modality is missing (e.g. camera off), NOT zero. Zero would read as "stared at
# nothing for the whole answer", which is a confident wrong answer rather than
# an abstention.
FEATURE_SPEC: list[tuple[str, float]] = [
    # --- audio / prosody ---
    ("words_per_min", 140.0),
    ("filler_per_min", 0.0),
    ("filler_ratio", 0.0),
    ("pitch_mean_hz", 150.0),
    ("pitch_var", 400.0),
    ("energy_mean", 0.05),
    ("energy_var", 0.001),
    ("pause_count_per_min", 6.0),
    ("mean_pause_ms", 350.0),
    ("longest_pause_ms", 800.0),
    ("speech_ratio", 0.75),
    ("restart_count", 0.0),
    # --- vision (browser-computed) ---
    ("eye_contact_pct", 0.65),
    ("gaze_stability", 0.6),
    ("posture_score", 0.7),
    ("posture_drift", 0.2),
    ("blink_per_min", 17.0),
    ("head_movement", 0.3),
    ("longest_freeze_ms", 0.0),
    # --- text-derived, still "delivery" not "content" ---
    ("answer_duration_s", 60.0),
    ("word_count", 140.0),
    ("type_token_ratio", 0.55),
]

FEATURE_ORDER: list[str] = [name for name, _ in FEATURE_SPEC]
FEATURE_DEFAULTS: dict[str, float] = dict(FEATURE_SPEC)

AUDIO_FEATURES = FEATURE_ORDER[:12]
VISION_FEATURES = FEATURE_ORDER[12:19]
TEXT_FEATURES = FEATURE_ORDER[19:]


@dataclass
class FeatureVector:
    """Typed view over the feature dict, with per-modality coverage tracking."""

    values: dict[str, float]
    has_audio: bool = True
    has_vision: bool = True

    @classmethod
    def from_partial(
        cls, partial: dict[str, float], *, has_audio: bool = True, has_vision: bool = True
    ) -> "FeatureVector":
        values = dict(FEATURE_DEFAULTS)
        for k, v in partial.items():
            if k in values and v is not None:
                values[k] = float(v)
        return cls(values=values, has_audio=has_audio, has_vision=has_vision)

    def as_list(self) -> list[float]:
        return [self.values[name] for name in FEATURE_ORDER]

    def coverage(self) -> float:
        """Fraction of modalities actually observed -- drives score confidence."""
        return (0.6 if self.has_audio else 0.0) + (0.4 if self.has_vision else 0.0)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass
class DeliverySubscores:
    confidence: float
    clarity: float
    pace: float
    presence: float

    def overall(self) -> float:
        return round(
            0.3 * self.confidence + 0.3 * self.clarity + 0.2 * self.pace + 0.2 * self.presence, 2
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]
