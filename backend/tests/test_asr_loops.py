"""ASR decoder loops and the turn-latency path, pinned.

Found by running the smoke script against real faster-whisper:

  1. Whisper locked into "Whoa, whoa, whoa..." and the report scored it as an
     answer delivered at 536 words/min.
  2. Every reply took ~3.5s: partial decodes were awaited inside push_audio,
     and the loop above ran each decode to the 224-token limit.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from app.audio.asr import MockASR, Transcript, _token_cap, collapse_repetition
from tests.test_e2e import Recorder, deps, settings, wait_for  # noqa: F401
from tests.test_phantom_and_patience import silence, speech


class TestCollapseRepetition:
    def test_single_word_loop_collapses(self):
        loop = "Oh, " + ", ".join(["whoa"] * 40)
        assert len(collapse_repetition(loop).split()) <= 2

    def test_phrase_loop_collapses(self):
        loop = " ".join(["thank you for watching"] * 6)
        assert collapse_repetition(loop) == "thank you for watching"

    def test_loop_after_real_speech_keeps_the_speech(self):
        text = "I would use a hash map " + " ".join(["map"] * 10)
        assert collapse_repetition(text) == "I would use a hash map"

    @pytest.mark.parametrize(
        "text",
        [
            "no, no, no, that is not what I meant",
            "um um I think the answer is O of n",
            "we sharded by user id and then by region and then by tenant",
        ],
    )
    def test_ordinary_speech_is_untouched(self, text):
        assert collapse_repetition(text) == text

    def test_empty(self):
        assert collapse_repetition("") == ""


class TestTokenCap:
    def test_scales_with_duration(self):
        assert _token_cap(2.0) < _token_cap(8.0)

    def test_leaves_room_for_fast_speech(self):
        # 4 words/s is fast; ~1.5 tokens/word with punctuation.
        assert _token_cap(10.0) >= 10 * 4 * 1.5

    def test_bounded_per_whisper_window(self):
        assert _token_cap(120.0) == _token_cap(30.0) <= 256


class _SlowLoopingASR(MockASR):
    """Behaves like Whisper on non-speech: slow, and stuck in a loop."""

    name = "slow-loop"

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def transcribe(self, audio, *, is_final):
        await asyncio.sleep(self.delay_s)
        return Transcript(collapse_repetition(" ".join(["whoa"] * 40)), is_final, 0.5)


async def _open(settings, deps, sid):  # noqa: F811
    from app.session.manager import InterviewSession

    rec = Recorder()
    s = InterviewSession(
        session_id=sid, track="dsa", jd_text=None, settings=settings, deps=deps, emit=rec,
    )
    await s.start()
    await wait_for(rec, "speech_end")
    await s.notify_playback_done()
    await wait_for(rec, "state")
    return s, rec


class TestPartialsDoNotBlockIngest:
    async def test_push_audio_returns_while_a_partial_decodes(self, settings, deps):  # noqa: F811
        deps.asr = _SlowLoopingASR(delay_s=1.0)
        s, rec = await _open(settings, deps, "lat1")
        t0 = time.perf_counter()
        for chunk in [speech(0.1)] * 20:
            await s.push_audio(chunk)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"push_audio stalled {elapsed:.2f}s behind ASR partials"
        await s.close()


class TestLoopingTranscriptIsNotAnAnswer:
    async def test_decoder_loop_is_discarded_not_scored(self, settings, deps):  # noqa: F811
        deps.asr = _SlowLoopingASR(delay_s=0.0)
        s, rec = await _open(settings, deps, "loop1")
        for chunk in [silence(0.1)] * 4 + [speech(0.1)] * 30 + [silence(0.1)] * 15:
            await s.push_audio(chunk)
            await asyncio.sleep(0)
        await wait_for(rec, "noise_discarded", timeout=6.0)
        assert not rec.of("turn_scored")
        await s.close()
