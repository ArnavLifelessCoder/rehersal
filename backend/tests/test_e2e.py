"""End-to-end: a full interview through the real actor, scorers and report.

This is the test that matters. It drives synthesised audio through the actual
InterviewSession -- real VAD, real endpointing, real feature extraction, real
state machine, real report builder -- with only the ASR and brain swapped for
their deterministic offline backends. Nothing is stubbed inside the pipeline.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.config import Settings
from app.deps import Deps
from app.interviewer.brain import BrainFactory
from app.interviewer.tts import BrowserTTS
from app.audio.asr import MockASR
from app.scoring.content import ContentScorer, LexicalEmbedder
from app.scoring.delivery import HeuristicDeliveryScorer
from app.session.manager import InterviewSession, SessionRegistry
from app.storage.store import Store

SR = 16_000


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        blob_dir=tmp_path / "blobs",
        # Scaled-down turn-taking so the suite stays fast. The *ratios* still
        # mirror production -- graces are a fraction of the base wait, and
        # stuck exceeds their sum -- so these tests keep exercising the real
        # patience logic rather than switching it off.
        endpoint_silence_ms=300,
        thinking_grace_ms=100,
        opening_grace_ms=100,
        floor_settle_ms=50,
        stuck_silence_ms=2500,
        max_questions_per_session=3,
        asr_backend="mock",
        vad_backend="energy",
        brain_backend="scripted",
    )


@pytest.fixture
def deps(settings) -> Deps:
    return Deps(
        asr=MockASR(SR),
        brain=BrainFactory("semantic", "concept-graph"),
        tts=BrowserTTS(),
        delivery_scorer=HeuristicDeliveryScorer(),
        content_scorer=ContentScorer(LexicalEmbedder()),
        store=Store(settings.database_url),
    )


class Recorder:
    """Captures the outbound message stream, exactly as a client would see it.

    Consumption is cursor-based per message kind. A count-snapshot approach
    races: the actor is fast enough that the message often lands *before* the
    test starts waiting for it, and the test then waits for a second one that
    never comes.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self._cursor: dict[str, int] = {}

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    def of(self, kind: str) -> list[dict]:
        return [m for m in self.messages if m["type"] == kind]

    def take(self, kind: str) -> dict | None:
        """Return the next unconsumed message of `kind`, or None."""
        pending = self.of(kind)
        idx = self._cursor.get(kind, 0)
        if idx < len(pending):
            self._cursor[kind] = idx + 1
            return pending[idx]
        return None

    @property
    def kinds(self) -> list[str]:
        return [m["type"] for m in self.messages]


def speech(seconds: float, freq: float = 160.0, amp: float = 0.25) -> np.ndarray:
    """Voiced-ish audio: a tone with harmonics and jitter, so prosody has signal."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(7)
    wobble = 1.0 + 0.06 * np.sin(2 * np.pi * 3.0 * t)
    sig = (
        np.sin(2 * np.pi * freq * wobble * t)
        + 0.4 * np.sin(2 * np.pi * 2 * freq * wobble * t)
        + 0.2 * np.sin(2 * np.pi * 3 * freq * wobble * t)
    )
    sig = sig * (0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t))  # envelope
    return (amp * sig + rng.normal(0, 0.004, t.size)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.normal(0, 0.0008, int(seconds * SR)).astype(np.float32)


async def wait_for(rec: Recorder, kind: str, timeout: float = 5.0) -> dict:
    """Consume the next unseen message of `kind`, waiting for it if needed."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if (msg := rec.take(kind)) is not None:
            return msg
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {kind!r}; got {rec.kinds}")
        await asyncio.sleep(0.01)


async def answer_once(session: InterviewSession, rec: Recorder, seconds: float = 3.0) -> None:
    """Simulate one candidate answer: a beat of room tone, talk, then go quiet
    past the endpoint. The leading room tone is realistic (nobody answers in
    zero milliseconds) and it lets the VAD see the room before the voice."""
    await session.notify_playback_done()
    await wait_for(rec, "state")
    for _ in range(6):
        await session.push_audio(quiet(0.1))
        await asyncio.sleep(0)
    for _ in range(int(seconds * 10)):
        await session.push_audio(speech(0.1))
        await asyncio.sleep(0)
    for _ in range(12):  # 1.2s of silence, past the 300ms endpoint
        await session.push_audio(quiet(0.1))
        await asyncio.sleep(0)


@pytest.mark.asyncio
class TestFullInterview:
    async def test_session_opens_with_a_question(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t1", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        end = await wait_for(rec, "speech_end")
        assert end["move"] == "opening"
        assert len(end["text"]) > 20
        assert rec.of("speech"), "the opening must be streamed as speech chunks"
        await session.close()

    async def test_answer_produces_scores(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t2", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await answer_once(session, rec)

        scored = await wait_for(rec, "turn_scored")
        turn = scored["turn"]
        assert turn["transcript"], "ASR should have produced text"
        assert 0.0 <= turn["delivery"]["overall"] <= 1.0
        assert 0.0 <= turn["content"]["overall"] <= 1.0
        assert turn["delivery"]["evidence"], "scores must cite their evidence"
        assert turn["fillers"] is not None
        await session.close()

    async def test_interviewer_responds_after_each_answer(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t3", track="behavioral", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        before = len(rec.of("speech_end"))
        await answer_once(session, rec)
        await wait_for(rec, "speech_end")
        assert len(rec.of("speech_end")) > before

        moves = [m["move"] for m in rec.of("speech_end")]
        assert moves[0] == "opening"
        assert moves[1] in {
            "next_question", "probe", "challenge", "hint", "move_on", "closing"
        }
        await session.close()

    async def test_cv_features_reach_the_score(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t4", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await session.notify_playback_done()
        await wait_for(rec, "state")
        for _ in range(6):
            await session.push_audio(quiet(0.1))
            await asyncio.sleep(0)

        for i in range(40):
            await session.push_cv({
                "ts_ms": i * 100, "eye_contact": i % 4 != 0, "gaze_offset": 0.2,
                "posture": 0.85, "blink": i % 12 == 0, "head_movement": 0.1,
                "face_present": True,
            })
        for _ in range(30):
            await session.push_audio(speech(0.1))
            await asyncio.sleep(0)
        for _ in range(12):
            await session.push_audio(quiet(0.1))
            await asyncio.sleep(0)

        scored = await wait_for(rec, "turn_scored")
        subs = scored["turn"]["delivery"]["subscores"]
        assert subs["presence"] != 0.65, "presence should be computed, not the no-camera prior"
        assert any(e["feature"] == "eye_contact_pct" for e in scored["turn"]["delivery"]["evidence"])
        await session.close()

    async def test_latency_is_measured_and_reported(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t5", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await answer_once(session, rec)

        lat = await wait_for(rec, "latency")
        t = lat["timings"]
        assert "asr_flush" in t["stages"]
        assert "scoring" in t["stages"]
        assert t["total_ms"] > 0
        assert "within_budget" in t
        await session.close()

    async def test_full_session_produces_a_report(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t6", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        for _ in range(2):
            await answer_once(session, rec)
            await wait_for(rec, "speech_end")

        await session.request_end()
        msg = await wait_for(rec, "report")
        report = msg["report"]

        assert report["answers_given"] >= 2
        assert 0.0 <= report["overall_delivery"] <= 1.0
        assert 0.0 <= report["overall_content"] <= 1.0
        assert 0.0 <= report["confidence"] <= 1.0
        assert report["summary_text"]
        assert report["delivery_timeline"]
        assert report["transcript"]
        assert set(report["delivery_breakdown"]) == {
            "confidence", "clarity", "pace", "presence"
        }
        assert report["backends"]["asr"] == "mock"
        await session.close()

    async def test_report_persists_and_reloads(self, settings, deps):
        rec = Recorder()
        session = InterviewSession(
            session_id="t7", track="behavioral", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await answer_once(session, rec)
        await wait_for(rec, "speech_end")
        await session.request_end()
        await wait_for(rec, "report")

        stored = deps.store.get_report("t7")
        assert stored is not None
        assert stored["session_id"] == "t7"

        rows = deps.store.export_training_rows()
        assert rows, "candidate turns must be exportable as training data"
        assert rows[0]["features"], "the stored feature vector is what makes retraining possible"
        await session.close()

    async def test_session_never_wedges_in_speaking(self, settings, deps):
        """If the client never acks playback, the floor must reopen on its own."""
        rec = Recorder()
        session = InterviewSession(
            session_id="t8", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        end = await wait_for(rec, "speech_end")
        # Deliberately never call notify_playback_done. The net is intentionally
        # generous (it must not fire during normal playback -- see
        # TestPlaybackTimeout), so derive the wait from the same formula rather
        # than hardcoding a duration that silently rots when the bound changes.
        words = len(end["text"].split())
        bound = min(max((words / 2.0) * 1.5 + 3.0, 5.0), 45.0)

        deadline = asyncio.get_event_loop().time() + bound + 3.0
        while asyncio.get_event_loop().time() < deadline:
            if any(m.get("state") == "listening" for m in rec.of("state")):
                break
            await asyncio.sleep(0.05)
        assert any(m.get("state") == "listening" for m in rec.of("state")), (
            f"playback-ack timeout failed to reopen the floor within {bound:.0f}s"
        )
        await session.close()

    async def test_audio_during_interviewer_speech_is_ignored(self, settings, deps):
        """The interviewer's own voice, echoed by speakers, must not be transcribed."""
        rec = Recorder()
        session = InterviewSession(
            session_id="t9", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        for _ in range(20):
            await session.push_audio(speech(0.1))
            await asyncio.sleep(0)
        assert not rec.of("turn_scored"), "echo during SPEAKING must not create a turn"
        await session.close()


@pytest.mark.asyncio
class TestRegistry:
    async def test_create_get_drop(self, settings, deps):
        reg = SessionRegistry()
        rec = Recorder()
        s = reg.create(track="dsa", jd_text=None, settings=settings, deps=deps, emit=rec)
        assert reg.get(s.id) is s
        assert reg.active_count() == 1
        await reg.drop(s.id)
        assert reg.get(s.id) is None
        assert reg.active_count() == 0
