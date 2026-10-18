"""VAD, endpointing and prosody."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.prosody import extract_prosody
from app.audio.vad import EnergyVAD, Endpointer

SR = 16_000
FRAME = int(SR * 0.02)


def _tone(freq: float, seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _noise(seconds: float, amp: float = 0.0005, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, amp, int(seconds * sr))).astype(np.float32)


class TestEnergyVAD:
    def test_detects_speech_over_silence(self):
        vad = EnergyVAD()
        for _ in range(20):  # let the noise floor settle
            vad.is_speech(_silence(0.02))
        loud = _tone(200, 0.02)
        assert vad.is_speech(loud)

    def test_silence_is_not_speech(self):
        vad = EnergyVAD()
        results = [vad.is_speech(_noise(0.02)) for _ in range(50)]
        assert sum(results) <= 2, "quiet noise should not read as sustained speech"

    def test_noise_floor_adapts_to_loud_room(self):
        """Sustained background hum must be absorbed into the floor, not
        reported as permanent speech. Takes a warmup window to learn."""
        vad = EnergyVAD()
        hum = _tone(60, 0.02, amp=0.02)
        flags = [vad.is_speech(hum) for _ in range(300)]
        assert sum(flags[100:]) == 0, "sustained background must be absorbed into the floor"

    def test_sustained_speech_does_not_gate_itself_out(self):
        vad = EnergyVAD()
        for _ in range(20):
            vad.is_speech(_noise(0.02))
        speech = _tone(180, 0.02, amp=0.3)
        flags = [vad.is_speech(speech) for _ in range(100)]
        assert all(flags), "the floor must not creep up during continuous speech"

    def test_speech_from_the_very_first_frame_is_detected(self):
        """A candidate who starts talking the instant the question ends must be
        heard. Regression: the old one-pole floor calibrated to frame zero."""
        vad = EnergyVAD()
        speech = _tone(180, 0.02, amp=0.3)
        flags = [vad.is_speech(speech) for _ in range(30)]
        assert all(flags[:20]), "opening frames of an immediate answer were dropped"

    def test_floor_survives_a_turn_boundary(self):
        vad = EnergyVAD()
        for _ in range(60):
            vad.is_speech(_noise(0.02))
        vad.reset()
        assert vad.is_speech(_tone(180, 0.02, amp=0.3))
        assert not vad.is_speech(_noise(0.02))


class TestEndpointer:
    def test_fires_endpoint_after_configured_silence(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, min_utterance_ms=400)
        events = []
        for _ in range(50):        # 1000ms of speech
            if (e := ep.push(True)):
                events.append(e)
        fired_at = None
        for i in range(60):        # up to 1200ms of silence
            if (e := ep.push(False)):
                events.append(e)
                if e.kind == "endpoint":
                    fired_at = (i + 1) * 20
                    break
        assert fired_at == 700, f"expected endpoint at exactly 700ms, got {fired_at}"
        assert [e.kind for e in events] == ["speech_start", "endpoint"]

    def test_short_noise_burst_does_not_open_a_turn(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, min_utterance_ms=400)
        for _ in range(3):   # 60ms -- below the 120ms onset window
            assert ep.push(True) is None
        for _ in range(60):
            ev = ep.push(False)
            assert ev is None or ev.kind != "endpoint"

    def test_utterance_below_minimum_is_discarded(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, min_utterance_ms=400)
        for _ in range(10):  # 200ms of speech: opens a turn, too short to score
            ep.push(True)
        kinds = []
        for _ in range(60):
            if (e := ep.push(False)):
                kinds.append(e.kind)
        assert "endpoint" not in kinds

    def test_stuck_fires_only_when_nothing_was_said(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, stuck_ms=8000)
        kinds = []
        for _ in range(450):  # 9s of silence
            if (e := ep.push(False)):
                kinds.append(e.kind)
        assert "stuck" in kinds

    def test_stuck_does_not_fire_after_the_user_spoke(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, stuck_ms=8000, min_utterance_ms=400)
        for _ in range(50):
            ep.push(True)
        kinds = []
        for _ in range(500):
            if (e := ep.push(False)):
                kinds.append(e.kind)
        assert "stuck" not in kinds, "a completed answer must not later fire 'stuck'"

    def test_ramble_fires_once(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, ramble_ms=2000)
        kinds = [e.kind for _ in range(300) if (e := ep.push(True))]
        assert kinds.count("ramble") == 1


class TestProsody:
    @pytest.mark.parametrize("freq", [120.0, 180.0, 240.0])
    def test_f0_tracks_a_known_tone(self, freq):
        feats = extract_prosody(_tone(freq, 1.5), SR)
        assert abs(feats.pitch_mean_hz - freq) < freq * 0.12, (
            f"F0 estimate {feats.pitch_mean_hz:.0f}Hz too far from true {freq}Hz"
        )

    def test_monotone_has_lower_pitch_variance_than_varied(self):
        mono = extract_prosody(_tone(150, 2.0), SR)
        varied = extract_prosody(
            np.concatenate([_tone(120, 0.5), _tone(200, 0.5), _tone(160, 0.5), _tone(240, 0.5)]),
            SR,
        )
        assert varied.pitch_var > mono.pitch_var * 3

    def test_pauses_are_counted(self):
        audio = np.concatenate([
            _tone(150, 1.0), _silence(0.8), _tone(150, 1.0), _silence(0.6), _tone(150, 1.0)
        ])
        feats = extract_prosody(audio, SR)
        assert feats.pause_count_per_min > 0
        assert 600 < feats.longest_pause_ms < 1000

    def test_speech_ratio_reflects_silence_share(self):
        audio = np.concatenate([_tone(150, 1.0), _silence(1.0)])
        feats = extract_prosody(audio, SR)
        assert 0.35 < feats.speech_ratio < 0.65

    def test_empty_audio_returns_neutral_not_crash(self):
        feats = extract_prosody(np.zeros(10, dtype=np.float32), SR)
        assert feats.pitch_mean_hz == 150.0
