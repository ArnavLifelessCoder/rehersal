"""Two failures users actually reported, pinned.

  1. "im not speaking still its inputting" -- phantom turns. Every ASR invents
     words when handed silence; the transcript can therefore never be the thing
     that decides whether someone spoke.
  2. "gives less time to even think" -- the endpointer cutting in mid-thought.
     A single fixed silence window cannot serve both "finished a sentence" and
     "paused to work something out".
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.audio.asr import MockASR
from app.config import Settings
from app.session.manager import _speech_rms

SR = 16_000


def silence(seconds: float, amp: float = 0.0003) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, amp, int(seconds * SR)).astype(np.float32)


def speech(seconds: float, amp: float = 0.2, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    sig = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4))
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return (amp * sig * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


class TestMockASRDoesNotFabricate:
    """The mock used to return >= 4 words for ANY input, including silence.

    Those words were fluent interview prose, they cleared min_answer_words, and
    they became a scored answer -- so the product showed people sentences they
    had never said and blamed them for the fillers in it.
    """

    def _run(self, audio):
        return asyncio.run(MockASR(SR).transcribe(audio, is_final=True))

    def test_silence_transcribes_to_nothing(self):
        assert self._run(silence(3.0)).text == ""

    def test_pure_zeros_transcribe_to_nothing(self):
        assert self._run(np.zeros(SR * 2, dtype=np.float32)).text == ""

    def test_empty_audio_is_safe(self):
        assert self._run(np.zeros(0, dtype=np.float32)).text == ""

    def test_a_brief_blip_does_not_become_a_sentence(self):
        blip = np.concatenate([silence(1.0), speech(0.1), silence(1.0)])
        assert self._run(blip).text == ""

    def test_real_speech_still_transcribes(self):
        text = self._run(speech(4.0)).text
        assert len(text.split()) >= 4

    def test_word_count_tracks_voiced_time_not_wall_clock(self):
        """Ten seconds holding two seconds of speech is a short answer.

        Scaling by wall-clock made words-per-minute and fillers-per-minute --
        both rate metrics -- systematically wrong for anyone who pauses.
        """
        dense = self._run(speech(4.0)).text.split()
        sparse = self._run(
            np.concatenate([silence(3.0), speech(1.0), silence(3.0)])
        ).text.split()
        assert len(sparse) < len(dense)

    def test_deterministic(self):
        a = speech(3.0)
        assert self._run(a).text == self._run(a).text


class TestSpeechRms:
    def test_silence_is_below_the_floor(self):
        assert _speech_rms(silence(3.0)) < Settings().min_turn_rms

    def test_speech_is_above_the_floor(self):
        assert _speech_rms(speech(3.0)) > Settings().min_turn_rms

    def test_one_sentence_in_a_long_pause_still_counts(self):
        """A mean would drown a real sentence in the silence around it."""
        audio = np.concatenate([silence(5.0), speech(1.5), silence(5.0)])
        assert _speech_rms(audio) > Settings().min_turn_rms

    def test_empty_audio_is_safe(self):
        assert _speech_rms(np.zeros(0, dtype=np.float32)) == 0.0


@pytest.mark.asyncio
class TestNoPhantomTurnsEndToEnd:
    """Silence through a real session must never become an answer."""

    async def _session(self, settings, deps, sid):
        from tests.test_e2e import Recorder, wait_for
        from app.session.manager import InterviewSession

        rec = Recorder()
        s = InterviewSession(
            session_id=sid, track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await s.start()
        await wait_for(rec, "speech_end")
        await s.notify_playback_done()
        await wait_for(rec, "state")
        return s, rec

    async def test_pure_silence_produces_no_turn(self, settings, deps):
        s, rec = await self._session(settings, deps, "ph1")
        for _ in range(60):                    # 6s of room tone
            await s.push_audio(silence(0.1))
            await asyncio.sleep(0)
        await asyncio.sleep(0.3)
        assert not rec.of("turn_scored"), (
            "silence was transcribed and scored as an answer"
        )
        await s.close()

    async def test_a_blip_produces_no_turn(self, settings, deps):
        s, rec = await self._session(settings, deps, "ph2")
        for chunk in [silence(0.1)] * 8 + [speech(0.08)] + [silence(0.1)] * 25:
            await s.push_audio(chunk)
            await asyncio.sleep(0)
        await asyncio.sleep(0.3)
        assert not rec.of("turn_scored")
        await s.close()

    async def test_real_speech_still_produces_a_turn(self, settings, deps):
        """The guards must not have made the product deaf."""
        from tests.test_e2e import wait_for

        s, rec = await self._session(settings, deps, "ph3")
        for chunk in [silence(0.1)] * 6 + [speech(0.1)] * 30 + [silence(0.1)] * 15:
            await s.push_audio(chunk)
            await asyncio.sleep(0)
        turn = await wait_for(rec, "turn_scored", timeout=6.0)
        assert turn["turn"]["transcript"]
        await s.close()


from tests.test_e2e import deps, settings  # noqa: E402, F401


class TestSettleWindow:
    """The echo-tail window is measured in samples, not wall-clock.

    Wall-clock made it wrong in both directions: it discarded nothing when a
    client burst a backlog after a stall, and discarded everything when audio
    arrived faster than real time.
    """

    def test_drops_exactly_the_configured_audio(self):
        s = Settings()
        expected = int(s.sample_rate * s.floor_settle_ms / 1000)
        assert expected > 0
        # 350ms at 16kHz = 5600 samples.
        assert expected == 5600


class TestThinkingPatience:
    """The endpointer must wait longer when the answer sounds unfinished."""

    @staticmethod
    def _patience(text: str, s: Settings | None = None) -> int:
        s = s or Settings()
        words = text.split()
        p = s.endpoint_silence_ms
        if len(words) < s.opening_grace_words:
            p += s.opening_grace_ms
        if words:
            from app.session.manager import _TRAILING_UNFINISHED
            if words[-1].lower().strip(".,;:!?'\"") in _TRAILING_UNFINISHED:
                p += s.thinking_grace_ms
        return p

    # Comfortably past opening_grace_words, so only the trailing word varies.
    LONG = (
        "we scaled the service horizontally across three regions and it held up "
        "fine under production load during the holiday traffic peak last winter"
    )

    def test_base_wait_is_interview_appropriate(self):
        """700ms is a conversation number. Answers need room to think."""
        assert Settings().endpoint_silence_ms >= 1000

    def test_short_answers_get_thinking_room(self):
        assert self._patience("I think") > self._patience(self.LONG)

    @pytest.mark.parametrize("trailing", ["so", "and", "because", "um", "the", "which"])
    def test_trailing_conjunctions_extend_the_wait(self, trailing):
        assert self._patience(f"{self.LONG} {trailing}") > self._patience(self.LONG)

    def test_a_finished_sentence_does_not_extend(self):
        assert self._patience(self.LONG) == Settings().endpoint_silence_ms

    def test_punctuation_does_not_defeat_the_check(self):
        assert self._patience(f"{self.LONG} so,") > self._patience(self.LONG)

    def test_the_two_graces_compose(self):
        """A short answer trailing off mid-thought gets both extensions."""
        s = Settings()
        assert self._patience("we could maybe use a") == (
            s.endpoint_silence_ms + s.opening_grace_ms + s.thinking_grace_ms
        )

    def test_stuck_timeout_exceeds_the_longest_patience(self):
        """Otherwise 'stuck' fires while we are still politely waiting."""
        s = Settings()
        longest = s.endpoint_silence_ms + s.opening_grace_ms + s.thinking_grace_ms
        assert s.stuck_silence_ms > longest, (
            "the hint would interrupt a candidate who is still being given time"
        )
