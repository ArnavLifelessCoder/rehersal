"""Voice activity detection + end-of-utterance detection.

An energy gate cannot do this job. A door slam, a keyboard clack, a cough and a
chair creak are all *louder* than conversational speech, so any threshold that
hears you also hears them -- and in a turn-taking system a false positive isn't
a cosmetic glitch, it hands the floor over and makes the interviewer respond to
a noise. That is the single most immersion-breaking failure this product has.

So the detector asks a different question: not "is this loud?" but "is this a
human voice?". Four cheap signals, computed per frame:

  harmonicity      Voiced speech is periodic -- the autocorrelation has a sharp
                   peak at the pitch lag. Slams, clacks and rustles have none.
                   This is the single most discriminative feature here.
  spectral flatness Speech energy is concentrated in formants (peaky, low
                   flatness). Broadband transients are flat. Cheap and orthogonal
                   to harmonicity, which catches unvoiced fricatives too.
  zero-crossing rate Separates voiced speech (low), fricatives (moderate) and
                   click/hiss noise (very high).
  band energy ratio Speech lives in 80-4000Hz. Rumble and hiss don't.

Decisions use hysteresis (a Schmitt trigger): it's harder to *start* speech than
to *continue* it. Without that, natural intra-word dips get chopped into pieces
and the endpointer fires mid-sentence.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0


class BaseVAD:
    name = "base"

    def is_speech(self, frame: np.ndarray) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def telemetry(self) -> dict:
        """Live internals for the debug channel. Backends without useful
        internals return {} and the UI simply shows less."""
        return {}


class EnergyVAD(BaseVAD):
    """Pure energy gate. Kept for benchmarking against SpectralVAD only.

    Retained deliberately: the tests compare the two, and being able to
    demonstrate *why* the naive approach fails is worth more than deleting it.
    """

    name = "energy"

    _WARMUP_FRAMES = 25
    _WARMUP_FLOOR = 3e-4
    _ABS_GATE = 1e-3

    def __init__(self, *, window_frames: int = 500, margin_db: float = 9.0) -> None:
        self.margin = 10 ** (margin_db / 20.0)
        self._window: deque[float] = deque(maxlen=window_frames)

    def reset(self) -> None:
        pass

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)
        self._window.append(rms)
        if len(self._window) < self._WARMUP_FRAMES:
            floor = self._WARMUP_FLOOR
        else:
            floor = float(np.percentile(np.fromiter(self._window, dtype=np.float64), 5))
        return rms > floor * self.margin and rms > self._ABS_GATE


@dataclass
class FrameFeatures:
    rms: float
    harmonicity: float
    flatness: float
    zcr: float
    band_ratio: float


class SpectralVAD(BaseVAD):
    """Voice detector using harmonicity + spectral shape. The default.

    Analysis runs over a 3-frame (60ms) sliding context rather than a single
    20ms frame: a 20ms window is too short to see a full pitch period at 70Hz,
    and harmonicity estimated from less than one period is noise.
    """

    name = "spectral"

    _WARMUP_FRAMES = 20
    _WARMUP_FLOOR = 3e-4
    _ABS_GATE = 6e-4

    # Hysteresis: the bar to open speech is higher than the bar to sustain it.
    _ONSET_SCORE = 0.55
    _SUSTAIN_SCORE = 0.32
    # You cannot START speech without periodicity. Hysteresis is what makes a
    # detector usable mid-utterance, but it cuts both ways: a resonant-but-
    # inharmonic noise (a chair creak) that clips the onset threshold on one
    # loud frame would then sustain for its whole duration on the lower bar.
    # Requiring real periodicity to open -- not merely to continue -- closes
    # that door, and costs nothing because utterances effectively never begin
    # with a sustained unvoiced fricative.
    _ONSET_MIN_HARMONICITY = 0.35
    # Onset must hold for consecutive frames. A single fluke frame -- and shaped
    # noise does produce them -- would otherwise latch the detector and then
    # coast for the noise's full duration on the lower sustain bar. Real speech
    # clears onset on essentially every frame, so this costs 20ms and nothing else.
    _ONSET_CONSECUTIVE = 2
    # Once we've heard the candidate clearly, anything more than this far below
    # them is someone or something else. A crude proximity gate: it's the only
    # lever available without speaker identification, and it's what stops a
    # conversation in the next room from taking the floor.
    _RELATIVE_FLOOR_DB = 22.0
    # How long we keep believing speech after the score drops. Covers stop
    # consonants ("t", "k"), which are genuinely silent for ~30-60ms mid-word.
    _HANGOVER_FRAMES = 6

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        window_frames: int = 500,
        context_frames: int = 3,
        energy_margin_db: float = 7.0,
    ) -> None:
        self.sr = sample_rate
        self.energy_margin = 10 ** (energy_margin_db / 20.0)
        self._floor_window: deque[float] = deque(maxlen=window_frames)
        self._context: deque[np.ndarray] = deque(maxlen=context_frames)
        self._in_speech = False
        self._hangover = 0
        self._onset_run = 0
        self._speech_level: float | None = None
        self.last_features: FrameFeatures | None = None
        # Telemetry for the debug channel: the score and gates behind the most
        # recent decision. "Why didn't it hear me" is unanswerable without them.
        self.last_score: float = 0.0
        self.last_gate: float = 0.0
        self.last_rms: float = 0.0

    def reset(self) -> None:
        # The noise floor and the learned speech level persist across turns --
        # neither the room nor the candidate's distance from the mic changes
        # between question 3 and question 4, and re-learning them each turn
        # throws away exactly the evidence that makes the gates reliable.
        # The latches must NOT persist, or a new turn inherits the old state.
        self._in_speech = False
        self._hangover = 0
        self._onset_run = 0
        self._context.clear()

    # ------------------------------------------------------------------ features

    def telemetry(self) -> dict:
        out = {
            "score": round(self.last_score, 3),
            "rms": round(self.last_rms, 5),
            "gate": round(self.last_gate, 5),
            "in_speech": self._in_speech,
            "speech_level": round(self._speech_level, 5) if self._speech_level else None,
        }
        if self.last_features is not None:
            f = self.last_features
            out.update(
                harmonicity=round(f.harmonicity, 3),
                flatness=round(f.flatness, 3),
                zcr=round(f.zcr, 3),
                band_ratio=round(f.band_ratio, 3),
            )
        return out

    def _harmonicity(self, x: np.ndarray) -> float:
        """Normalised autocorrelation peak within the plausible pitch range."""
        n = x.size
        if n < 256:
            return 0.0
        x = x - x.mean()
        energy = float(np.dot(x, x))
        if energy < 1e-10:
            return 0.0

        size = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(x * np.hanning(n), n=size)
        acf = np.fft.irfft(spec * np.conj(spec), n=size)[:n]
        if acf[0] <= 1e-12:
            return 0.0

        lag_min = max(int(self.sr / F0_MAX_HZ), 2)
        lag_max = min(int(self.sr / F0_MIN_HZ), n - 1)
        if lag_max <= lag_min:
            return 0.0
        return float(np.clip(acf[lag_min:lag_max].max() / acf[0], 0.0, 1.0))

    def _spectral(self, x: np.ndarray) -> tuple[float, float]:
        """(flatness, band_ratio) from one power spectrum."""
        size = 512
        win = x[-size:] if x.size >= size else np.pad(x, (0, size - x.size))
        power = np.abs(np.fft.rfft(win * np.hanning(size))) ** 2 + 1e-12
        # Wiener entropy: geometric/arithmetic mean. 1.0 = white noise, 0 = tone.
        flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))

        freqs = np.fft.rfftfreq(size, 1.0 / self.sr)
        speech_band = (freqs >= 80) & (freqs <= 4000)
        band_ratio = float(power[speech_band].sum() / power.sum())
        return flatness, band_ratio

    @staticmethod
    def _zcr(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    # ------------------------------------------------------------------ decision

    def _voice_score(self, f: FrameFeatures) -> float:
        """0..1 confidence that this context contains a human voice.

        Structure matters as much as the weights. Periodicity and peakiness
        combine ADDITIVELY -- either can carry a frame, since voiced vowels are
        strongly periodic while unvoiced fricatives are merely peaky. Band ratio
        and ZCR then apply MULTIPLICATIVELY, as gates.

        The additive-only version of this failed on a fan: a 50Hz hum is
        perfectly periodic and perfectly peaky, so it collected most of the
        score and squeaked over the threshold. But essentially none of its
        energy lies in the 80-4000Hz speech band, and nothing that fails that
        test is a voice regardless of how tonal it is. A gate expresses that;
        a 13% additive term does not.
        """
        # Periodicity -- what a slam, clack or rustle can never fake.
        h = float(np.clip((f.harmonicity - 0.30) / 0.35, 0.0, 1.0))
        # Peakiness. Flatness ~0.05 for vowels, ~0.5+ for broadband noise.
        p = float(np.clip((0.45 - f.flatness) / 0.35, 0.0, 1.0))
        voice = 0.60 * h + 0.40 * p

        # Gate 1: is the energy even in the vocal range? Rejects rumble and hiss.
        band_gate = float(np.clip((f.band_ratio - 0.30) / 0.30, 0.0, 1.0))
        # Gate 2: zero-crossing plausibility. Voiced speech ~0.02-0.35;
        # fricatives at the top of that; clicks and hiss far above; mains hum
        # and rumble far below.
        if f.zcr <= 0.02:
            z = f.zcr / 0.02 * 0.5
        elif f.zcr <= 0.35:
            z = 1.0
        else:
            z = float(np.clip(1.0 - (f.zcr - 0.35) / 0.25, 0.0, 1.0))

        # Gates soften rather than zero the score, so a single odd frame in the
        # middle of real speech can't punch a hole in an utterance.
        return voice * (0.35 + 0.65 * band_gate) * (0.5 + 0.5 * z)

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            return False

        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)
        self._floor_window.append(rms)
        self._context.append(np.asarray(frame, dtype=np.float64))

        if len(self._floor_window) < self._WARMUP_FRAMES:
            floor = self._WARMUP_FLOOR
        else:
            floor = float(np.percentile(np.fromiter(self._floor_window, dtype=np.float64), 5))

        # Energy is a necessary condition, never a sufficient one. Two floors
        # apply: the absolute/adaptive noise floor, and -- once we've actually
        # heard this speaker -- a relative one tied to their established level.
        gate = max(floor * self.energy_margin, self._ABS_GATE)
        if self._speech_level is not None:
            gate = max(gate, self._speech_level / (10 ** (self._RELATIVE_FLOOR_DB / 20.0)))
        self.last_rms = rms
        self.last_gate = gate

        if rms < gate:
            self.last_score = 0.0
            if self._hangover > 0:
                self._hangover -= 1
                return True
            self._in_speech = False
            self._onset_run = 0
            return False

        context = np.concatenate(self._context)
        flatness, band_ratio = self._spectral(context)
        feats = FrameFeatures(
            rms=rms,
            harmonicity=self._harmonicity(context),
            flatness=flatness,
            zcr=self._zcr(frame),
            band_ratio=band_ratio,
        )
        self.last_features = feats
        score = self._voice_score(feats)
        self.last_score = score

        if self._in_speech:
            active = score >= self._SUSTAIN_SCORE
        else:
            if (
                score >= self._ONSET_SCORE
                and feats.harmonicity >= self._ONSET_MIN_HARMONICITY
            ):
                self._onset_run += 1
            else:
                self._onset_run = 0
            active = self._onset_run >= self._ONSET_CONSECUTIVE

        if active:
            self._in_speech = True
            self._hangover = self._HANGOVER_FRAMES
            # Learn this speaker's level from frames we're confident about, so
            # the relative gate calibrates to them rather than to a guess.
            if score > 0.8:
                self._speech_level = (
                    rms if self._speech_level is None
                    else 0.97 * self._speech_level + 0.03 * rms
                )
            return True

        if self._hangover > 0:
            self._hangover -= 1
            return True

        self._in_speech = False
        return False


def build_vad(backend: str, sample_rate: int) -> BaseVAD:
    if backend == "energy":
        log.info("VAD backend: energy (naive, for comparison only)")
        return EnergyVAD()
    if backend in ("auto", "silero"):
        try:
            vad = SileroVAD(sample_rate)
            log.info("VAD backend: silero")
            return vad
        except Exception as exc:  # noqa: BLE001
            if backend == "silero":
                raise
            log.info("VAD backend: spectral (silero unavailable: %s)", exc)
    return SpectralVAD(sample_rate)


class SileroVAD(BaseVAD):
    """Silero VAD via torch hub. Strictly better than SpectralVAD when present."""

    name = "silero"

    def __init__(self, sample_rate: int = 16_000) -> None:
        import torch  # noqa: F401

        model, _ = torch.hub.load(
            repo_or_url="snakers4/silero-vad", model="silero_vad", onnx=True, trust_repo=True
        )
        self._model = model
        self._sr = sample_rate
        self._torch = torch
        self._buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._buffer = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        # Silero requires exactly 512 samples at 16k. Buffer rather than pad:
        # zero-padding every 320-sample frame to 512 feeds the model 40%
        # silence and drags its probability down across the board.
        self._buffer = np.concatenate([self._buffer, frame.astype(np.float32)])
        if self._buffer.size < 512:
            return False
        chunk, self._buffer = self._buffer[:512], self._buffer[512:]
        tensor = self._torch.from_numpy(chunk)
        return float(self._model(tensor, self._sr).item()) > 0.5


@dataclass
class EndpointEvent:
    kind: str  # "speech_start" | "endpoint" | "stuck" | "ramble"
    at_ms: int
    speech_ms: int = 0


@dataclass
class Endpointer:
    """Turns per-frame speech flags into turn-taking events.

    Two independent guards against reacting to noise:

      onset_ms          speech must be sustained this long to open a turn, so a
                        single cough or clack never starts one.
      min_utterance_ms  total accumulated speech must reach this before a
                        trailing silence counts as an endpoint. A 200ms bark
                        does not end anyone's turn.

    Both are necessary. Onset alone lets a long noise (a passing truck) open a
    turn; minimum-utterance alone lets a click open one that then never closes.
    """

    frame_ms: int
    silence_ms: int = 700
    stuck_ms: int = 8_000
    ramble_ms: int = 90_000
    min_utterance_ms: int = 600
    onset_ms: int = 200

    _in_speech: bool = field(default=False, init=False)
    _speech_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _speech_total: int = field(default=0, init=False)
    _elapsed: int = field(default=0, init=False)
    _turn_started: bool = field(default=False, init=False)
    _fired_stuck: bool = field(default=False, init=False)
    _fired_ramble: bool = field(default=False, init=False)
    # Disarmed after an endpoint: the turn belongs to the session actor now.
    # Without this the trailing silence (the interviewer thinking, then talking)
    # would later fire a spurious "stuck", accusing the candidate of freezing on
    # a question they had already answered.
    _armed: bool = field(default=True, init=False)

    def reset(self) -> None:
        self._reset_counters()
        self._armed = True

    def _reset_counters(self) -> None:
        self._in_speech = False
        self._speech_run = self._silence_run = self._speech_total = self._elapsed = 0
        self._turn_started = self._fired_stuck = self._fired_ramble = False

    @property
    def speech_ms(self) -> int:
        return self._speech_total

    def snapshot(self) -> dict:
        """Live counters for the debug channel -- what the endpointer believes
        right now, which is the thing you can never reconstruct from logs."""
        return {
            "in_speech": self._in_speech,
            "speech_ms": self._speech_total,
            "speech_run_ms": self._speech_run,
            "silence_run_ms": self._silence_run,
            "armed": self._armed,
            "silence_needed_ms": self.silence_ms,
            "min_utterance_ms": self.min_utterance_ms,
        }

    def push(self, is_speech: bool) -> EndpointEvent | None:
        if not self._armed:
            return None
        self._elapsed += self.frame_ms

        if is_speech:
            self._speech_run += self.frame_ms
            self._silence_run = 0
            self._speech_total += self.frame_ms
            if not self._in_speech and self._speech_run >= self.onset_ms:
                self._in_speech = True
                self._turn_started = True
                return EndpointEvent("speech_start", self._elapsed)
        else:
            self._silence_run += self.frame_ms
            self._speech_run = 0

        if self._in_speech and self._silence_run >= self.silence_ms:
            if self._speech_total >= self.min_utterance_ms:
                ev = EndpointEvent("endpoint", self._elapsed, self._speech_total)
                self._reset_counters()
                self._armed = False
                return ev
            # Too little speech to be an answer. Discard and keep listening --
            # crucially WITHOUT resetting _elapsed, so a room full of
            # intermittent noise still eventually reaches the 'stuck' timeout.
            self._in_speech = False
            self._speech_total = 0
            self._turn_started = False

        if self._in_speech and not self._fired_ramble and self._speech_total >= self.ramble_ms:
            self._fired_ramble = True
            return EndpointEvent("ramble", self._elapsed, self._speech_total)

        if (
            not self._turn_started
            and not self._fired_stuck
            and self._silence_run >= self.stuck_ms
        ):
            self._fired_stuck = True
            return EndpointEvent("stuck", self._elapsed)
        return None
