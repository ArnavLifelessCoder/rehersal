"""Noise rejection: the detector must not mistake a room for a person.

Each generator below approximates a real interruption that reaches a laptop
mic. The bar is strict on purpose -- in a turn-taking system a false positive
hands the floor over, and the interviewer answering a door slam is the single
most immersion-breaking failure this product can have.

The EnergyVAD comparison tests exist to document *why* the naive detector was
replaced, and to fail loudly if SpectralVAD ever regresses to being one.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.vad import EnergyVAD, Endpointer, SpectralVAD

SR = 16_000
FRAME = int(SR * 0.02)


# --------------------------------------------------------------------- signals

def room_tone(seconds: float, amp: float = 0.0006) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(0, amp, int(seconds * SR)).astype(np.float32)


def _bandpass_noise(n: int, low: float, high: float, seed: int) -> np.ndarray:
    """Noise shaped to a frequency band, with phase left random.

    Random phase is the point: it's what makes shaped noise *inharmonic*, which
    is precisely how a real creak or rustle differs from a voice at the same
    loudness and in the same frequency band.
    """
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec *= ((freqs >= low) & (freqs <= high)).astype(float)
    out = np.fft.irfft(spec, n=n)
    return out / (np.std(out) + 1e-9)


def speech(seconds: float, f0: float = 150.0, amp: float = 0.2) -> np.ndarray:
    """Voiced speech: harmonic stack, formant shaping, vibrato, syllable rate.

    Phase is built by integrating instantaneous frequency. Multiplying the
    phase ARGUMENT by a jitter term instead (the obvious-looking shortcut)
    scrambles phase by tens of cycles per sample at high harmonics and yields
    white noise wearing a harmonic costume -- which is exactly what it produced
    the first time, and it made a working detector look broken.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(2)

    # Instantaneous f0: slow vibrato + slow drift. No per-sample randomness.
    inst_f0 = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 4.5 * t) + 0.01 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(inst_f0) / SR

    sig = np.zeros(n)
    for k in range(1, 40):
        f = k * f0
        if f > 5200:
            break
        # Two formants, roughly a mid vowel.
        gain = (
            np.exp(-((f - 650) ** 2) / (2 * 450**2))
            + 0.6 * np.exp(-((f - 1700) ** 2) / (2 * 550**2))
            + 0.25 * np.exp(-((f - 2900) ** 2) / (2 * 700**2))
        )
        sig += (gain / (1 + 0.35 * k)) * np.sin(k * phase)

    sig *= 0.65 + 0.35 * np.sin(2 * np.pi * 4.0 * t)          # syllables
    sig /= np.max(np.abs(sig)) + 1e-9
    # A little breath noise, as any real voice has.
    sig = 0.92 * sig + 0.08 * _bandpass_noise(n, 1500, 6000, 22)
    return (amp * sig).astype(np.float32)


def door_slam(amp: float = 0.85) -> np.ndarray:
    """Broadband transient with a fast exponential decay."""
    n = int(0.18 * SR)
    rng = np.random.default_rng(3)
    env = np.exp(-np.arange(n) / (0.022 * SR))
    return (amp * rng.normal(0, 1, n) * env).astype(np.float32)


def keyboard_clack(amp: float = 0.5) -> np.ndarray:
    """Very short, very broadband, high zero-crossing rate."""
    n = int(0.03 * SR)
    rng = np.random.default_rng(4)
    env = np.exp(-np.arange(n) / (0.004 * SR))
    return (amp * rng.normal(0, 1, n) * env).astype(np.float32)


def chair_creak(amp: float = 0.25) -> np.ndarray:
    """Resonant but inharmonic: shaped noise, not a tone.

    A creak has spectral peaks, so flatness alone won't reject it. What it does
    NOT have is periodicity -- which is why harmonicity carries most of the
    weight in the score.
    """
    n = int(0.5 * SR)
    body = _bandpass_noise(n, 300, 1200, 5) + 0.6 * _bandpass_noise(n, 1800, 2600, 7)
    env = np.hanning(n) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * np.arange(n) / SR))
    out = body * env
    return (amp * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def paper_rustle(amp: float = 0.2) -> np.ndarray:
    """High-frequency broadband hiss."""
    n = int(0.7 * SR)
    rng = np.random.default_rng(6)
    white = rng.normal(0, 1, n)
    highpassed = np.diff(np.concatenate([[0], white]))  # crude 6dB/oct highpass
    return (amp * highpassed / (np.std(highpassed) + 1e-9) * 0.3).astype(np.float32)


def fan_hum(seconds: float, amp: float = 0.03) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * (np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 100 * t))).astype(
        np.float32
    )


def distant_speech(seconds: float) -> np.ndarray:
    """Someone talking in the next room -- real speech, well below the floor.

    Energy is the ONLY lever available here. Telling "my quiet voice" from
    "someone else's quiet voice" requires speaker identification, which this
    system deliberately doesn't do. So the guarantee is bounded: clearly
    distant speech is rejected; a soft-spoken candidate close to the mic is
    not. See ARCHITECTURE.md §11.
    """
    return speech(seconds, f0=190, amp=0.002)


# --------------------------------------------------------------------- helpers

def run_vad(vad, signal: np.ndarray) -> list[bool]:
    return [
        vad.is_speech(signal[i : i + FRAME])
        for i in range(0, len(signal) - FRAME + 1, FRAME)
    ]


def calibrated(vad_cls=SpectralVAD, quiet_s: float = 1.2, *, heard_speaker: bool = False):
    """A VAD that has heard the room, as it always has in practice.

    `heard_speaker` additionally plays one real answer through it first. That
    mirrors the actual session: the candidate answers the opening question
    before any of this matters, and it's what arms the relative-level gate.
    """
    vad = vad_cls(SR) if vad_cls is SpectralVAD else vad_cls()
    run_vad(vad, room_tone(quiet_s))
    if heard_speaker:
        run_vad(vad, speech(2.0))
        run_vad(vad, room_tone(0.6))
    return vad


def speech_ratio(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


# --------------------------------------------------------------------- tests

class TestRejectsNoise:
    @pytest.mark.parametrize(
        "name,make",
        [
            ("door slam", door_slam),
            ("keyboard clack", keyboard_clack),
            ("chair creak", chair_creak),
            ("paper rustle", paper_rustle),
        ],
    )
    def test_transients_are_not_speech(self, name, make):
        vad = calibrated()
        flags = run_vad(vad, make())
        assert speech_ratio(flags) < 0.25, (
            f"{name}: {speech_ratio(flags):.0%} of frames read as speech"
        )

    def test_fan_hum_is_not_speech(self):
        vad = calibrated()
        flags = run_vad(vad, fan_hum(2.0))
        assert speech_ratio(flags[25:]) < 0.1

    def test_distant_speech_is_rejected_once_the_speaker_is_known(self):
        vad = calibrated(heard_speaker=True)
        flags = run_vad(vad, distant_speech(1.5))
        assert speech_ratio(flags) < 0.15, (
            "background conversation took the floor despite the relative-level gate"
        )

    def test_the_real_speaker_is_still_heard_after_going_quiet(self):
        """The relative gate must not lock out the candidate's own soft voice."""
        vad = calibrated(heard_speaker=True)
        flags = run_vad(vad, speech(1.5, amp=0.04))   # ~14dB below their normal
        assert speech_ratio(flags) > 0.7

    def test_silence_is_not_speech(self):
        vad = calibrated()
        assert speech_ratio(run_vad(vad, room_tone(2.0))) < 0.05


class TestAcceptsSpeech:
    @pytest.mark.parametrize("f0", [95.0, 150.0, 210.0])
    def test_voiced_speech_is_detected(self, f0):
        vad = calibrated()
        flags = run_vad(vad, speech(1.5, f0=f0))
        assert speech_ratio(flags) > 0.75, (
            f"f0={f0}Hz: only {speech_ratio(flags):.0%} detected"
        )

    def test_quiet_speech_is_still_detected(self):
        vad = calibrated()
        flags = run_vad(vad, speech(1.5, amp=0.05))
        assert speech_ratio(flags) > 0.6

    def test_speech_immediately_after_calibration(self):
        vad = calibrated()
        flags = run_vad(vad, speech(1.0))
        assert all(flags[3:20]), "opening of an immediate answer was dropped"

    def test_hangover_bridges_stop_consonants(self):
        """A 40ms gap mid-word must not read as end-of-speech."""
        vad = calibrated()
        signal = np.concatenate([
            speech(0.5), np.zeros(int(0.04 * SR), dtype=np.float32), speech(0.5)
        ])
        flags = run_vad(vad, signal)
        # Find the gap region and confirm it stayed latched.
        gap_start = int(0.5 / 0.02)
        assert any(flags[gap_start : gap_start + 3]), "stop consonant broke the run"


class TestBeatsEnergyGate:
    """Documents why the energy gate was replaced."""

    @pytest.mark.parametrize("make", [door_slam, keyboard_clack, chair_creak])
    def test_spectral_rejects_what_energy_accepts(self, make):
        noise = make()
        energy_flags = run_vad(calibrated(EnergyVAD), noise)
        spectral_flags = run_vad(calibrated(SpectralVAD), noise)
        assert speech_ratio(spectral_flags) < speech_ratio(energy_flags) or (
            speech_ratio(energy_flags) == 0.0
        ), "spectral VAD should reject at least as much noise as the energy gate"

    def test_both_still_accept_real_speech(self):
        sig = speech(1.5)
        assert speech_ratio(run_vad(calibrated(EnergyVAD), sig)) > 0.7
        assert speech_ratio(run_vad(calibrated(SpectralVAD), sig)) > 0.7


class TestEndToEndNoiseDoesNotEndTurns:
    """The property that actually matters: noise must not take the floor."""

    def _drive(self, signal: np.ndarray, *, silence_ms=700, min_utt=600, onset=200):
        vad = calibrated()
        ep = Endpointer(
            frame_ms=20, silence_ms=silence_ms, min_utterance_ms=min_utt, onset_ms=onset
        )
        events = []
        for flag in run_vad(vad, signal):
            if (e := ep.push(flag)):
                events.append(e.kind)
        return events

    @pytest.mark.parametrize(
        "name,make",
        [
            ("door slam", door_slam),
            ("keyboard clack", keyboard_clack),
            ("chair creak", chair_creak),
            ("paper rustle", paper_rustle),
        ],
    )
    def test_single_noise_never_produces_an_endpoint(self, name, make):
        signal = np.concatenate([room_tone(0.5), make(), room_tone(1.5)])
        assert "endpoint" not in self._drive(signal), f"{name} ended the turn"

    def test_repeated_clacks_never_produce_an_endpoint(self):
        parts = [room_tone(0.4)]
        for _ in range(6):
            parts += [keyboard_clack(), room_tone(0.25)]
        parts.append(room_tone(1.2))
        assert "endpoint" not in self._drive(np.concatenate(parts))

    def test_real_answer_does_produce_an_endpoint(self):
        signal = np.concatenate([room_tone(0.5), speech(3.0), room_tone(1.2)])
        assert "endpoint" in self._drive(signal)

    def test_answer_survives_a_cough_in_the_middle(self):
        signal = np.concatenate([
            room_tone(0.5), speech(1.5), door_slam(0.4), speech(1.5), room_tone(1.2)
        ])
        events = self._drive(signal)
        assert events.count("endpoint") == 1, "the interruption split the answer in two"
