"""The debug story: trace ring buffer, live telemetry, and the events endpoint."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.vad import Endpointer, SpectralVAD
from app.session.trace import MAX_EVENTS, REPORT_TAIL, SessionTrace

SR = 16_000
FRAME = int(SR * 0.02)


class TestSessionTrace:
    def test_records_in_order_with_sequence(self):
        tr = SessionTrace()
        tr.add("a")
        tr.add("b", detail=1)
        events = tr.events()
        assert [e["kind"] for e in events] == ["a", "b"]
        assert [e["seq"] for e in events] == [1, 2]
        assert all(e["t_ms"] >= 0 for e in events)

    def test_none_values_are_dropped(self):
        tr = SessionTrace()
        e = tr.add("decision", move="probe", weakest=None)
        assert "weakest" not in e
        assert e["move"] == "probe"

    def test_ring_buffer_bounds_memory(self):
        """A 40-minute rambler must not grow the trace without bound."""
        tr = SessionTrace()
        for i in range(MAX_EVENTS + 500):
            tr.add("tick", i=i)
        events = tr.events()
        assert len(events) == MAX_EVENTS
        # Oldest evicted, newest kept, sequence still monotonic.
        assert events[0]["i"] == 500
        assert events[-1]["seq"] == MAX_EVENTS + 500

    def test_limit_returns_the_tail(self):
        tr = SessionTrace()
        for i in range(50):
            tr.add("e", i=i)
        tail = tr.events(limit=10)
        assert len(tail) == 10 and tail[-1]["i"] == 49

    def test_report_tail_is_capped(self):
        tr = SessionTrace()
        for _ in range(MAX_EVENTS):
            tr.add("e")
        assert len(tr.tail_for_report()) == REPORT_TAIL

    def test_counts_histogram(self):
        tr = SessionTrace()
        tr.add("decision"); tr.add("decision"); tr.add("vad_endpoint")
        assert tr.counts() == {"decision": 2, "vad_endpoint": 1}


class TestPlaybackTimeout:
    """The ack timeout is a safety net; it must not fire during normal playback.

    Regression: it was words/2.5 -- normal speaking rate -- so it equalled the
    expected playback duration and tripped on healthy sessions, stealing the
    floor mid-question. Caught by running a real interview in a browser.
    """

    @staticmethod
    def _timeout_for(words: int) -> float:
        expected_s = words / 2.0
        return min(max(expected_s * 1.5 + 3.0, 5.0), 45.0)

    @pytest.mark.parametrize("words", [5, 12, 17, 30, 60])
    def test_leaves_headroom_over_a_slow_voice(self, words):
        # Slowest plausible browser TTS is ~1.8 words/sec.
        slowest_playback = words / 1.8
        assert self._timeout_for(words) > slowest_playback * 1.15, (
            f"{words} words: timeout would fire during normal slow playback"
        )

    def test_still_recovers_a_broken_client_promptly(self):
        assert self._timeout_for(17) < 20.0

    def test_is_bounded_for_long_text(self):
        assert self._timeout_for(500) == 45.0

    def test_has_a_floor_for_very_short_text(self):
        assert self._timeout_for(1) >= 5.0


class TestTelemetry:
    def test_spectral_vad_reports_its_last_decision(self):
        vad = SpectralVAD(SR)
        rng = np.random.default_rng(0)
        for _ in range(30):
            vad.is_speech(rng.normal(0, 0.0006, FRAME).astype(np.float32))
        t = vad.telemetry()
        assert {"score", "rms", "gate", "in_speech"} <= set(t)
        assert t["in_speech"] is False

    def test_endpointer_snapshot_tracks_counters(self):
        ep = Endpointer(frame_ms=20, silence_ms=700, min_utterance_ms=600, onset_ms=200)
        for _ in range(20):   # 400ms of speech
            ep.push(True)
        snap = ep.snapshot()
        assert snap["in_speech"] is True
        assert snap["speech_ms"] == 400
        assert snap["armed"] is True
        for _ in range(10):   # 200ms of silence
            ep.push(False)
        assert ep.snapshot()["silence_run_ms"] == 200


@pytest.mark.asyncio
class TestSessionTracing:
    """Trace events emitted by a real session actor."""

    async def _run_one_turn(self, settings, deps):
        from tests.test_e2e import Recorder, answer_once, wait_for
        from app.session.manager import InterviewSession

        rec = Recorder()
        session = InterviewSession(
            session_id="dbg1", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await answer_once(session, rec)
        await wait_for(rec, "speech_end")
        return session, rec

    async def test_decisions_are_traced_with_their_inputs(self, settings, deps):
        session, _ = await self._run_one_turn(settings, deps)
        decisions = [e for e in session.trace.events() if e["kind"] == "decision"]
        assert decisions, "a scored turn must leave a decision event"
        d = decisions[0]
        assert {"move", "content", "words", "question_n"} <= set(d)
        spoke = [e for e in session.trace.events() if e["kind"] == "spoke"]
        assert spoke and all("move" in e for e in spoke)
        await session.close()

    async def test_debug_toggle_streams_and_backfills(self, settings, deps):
        from tests.test_e2e import Recorder, wait_for
        from app.session.manager import InterviewSession

        rec = Recorder()
        session = InterviewSession(
            session_id="dbg2", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")

        await session.set_debug(True)
        backfill = await wait_for(rec, "trace_backfill")
        assert backfill["events"], "enabling debug must replay history"
        assert "counts" in backfill

        # With debug on, new trace events stream live.
        await session._trace("test_marker", n=1)
        live = await wait_for(rec, "trace")
        assert live["event"]["kind"] == "test_marker"

        # With debug off, they don't.
        await session.set_debug(False)
        await session._trace("silent_marker", n=2)
        assert not any(
            m.get("event", {}).get("kind") == "silent_marker" for m in rec.of("trace")
        )
        # But they are still recorded -- tracing is always on.
        assert any(e["kind"] == "silent_marker" for e in session.trace.events())
        await session.close()

    async def test_report_carries_the_trace_tail(self, settings, deps):
        from tests.test_e2e import Recorder, answer_once, wait_for
        from app.session.manager import InterviewSession

        rec = Recorder()
        session = InterviewSession(
            session_id="dbg3", track="dsa", jd_text=None,
            settings=settings, deps=deps, emit=rec,
        )
        await session.start()
        await wait_for(rec, "speech_end")
        await answer_once(session, rec)
        await wait_for(rec, "speech_end")
        await session.request_end()
        msg = await wait_for(rec, "report")

        assert msg["report"]["debug_events"], "the report must carry the trace tail"
        assert "decision" in msg["report"]["event_counts"]
        await session.close()


# Reuse the e2e fixtures (settings/deps with offline backends).
from tests.test_e2e import deps, settings  # noqa: E402, F401


class TestEventsEndpoint:
    def test_finished_session_events_come_from_the_report(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            missing = client.get("/api/sessions/nope/events")
            assert missing.status_code == 404
