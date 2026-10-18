"""The turn-taking policy -- the interview's actual behaviour."""

from __future__ import annotations

import pytest

from app.session.cv_features import CVAggregator, CVSample
from app.session.state import AnswerSignals, Move, SessionState


def signals(**kw) -> AnswerSignals:
    base = dict(
        content_overall=0.7, relevance=0.7, completeness=0.7, structure=0.7,
        correctness_signal=0.7, word_count=80, duration_s=45.0,
    )
    base.update(kw)
    return AnswerSignals(**base)


class TestDecisions:
    def test_good_answer_advances(self):
        s = SessionState(track="dsa")
        assert s.decide(signals(content_overall=0.8)) is Move.NEXT_QUESTION

    def test_weak_answer_probes(self):
        s = SessionState(track="dsa")
        assert s.decide(signals(content_overall=0.45)) is Move.PROBE

    def test_confident_but_shallow_gets_challenged(self):
        s = SessionState(track="system_design")
        move = s.decide(signals(correctness_signal=0.85, completeness=0.3, content_overall=0.55))
        assert move is Move.CHALLENGE

    def test_silence_hints_then_moves_on(self):
        s = SessionState(track="dsa")
        assert s.decide(signals(was_silent=True, word_count=0)) is Move.HINT
        s.apply(Move.HINT)
        assert s.decide(signals(was_silent=True, word_count=0)) is Move.MOVE_ON

    def test_rambling_interrupts(self):
        s = SessionState(track="behavioral")
        assert s.decide(signals(was_rambling=True)) is Move.INTERRUPT

    def test_probe_cap_forces_advance(self):
        """Two probes is the limit -- an interviewer that never moves on is a bug."""
        s = SessionState(track="dsa")
        weak = signals(content_overall=0.4)
        assert s.decide(weak) is Move.PROBE
        s.apply(Move.PROBE)
        assert s.decide(weak) is Move.PROBE
        s.apply(Move.PROBE)
        assert s.decide(weak) is Move.NEXT_QUESTION

    def test_budget_exhaustion_closes(self):
        s = SessionState(track="dsa", question_budget=2)
        s.questions_asked = 2
        assert s.decide(signals()) is Move.CLOSING

    def test_terse_answer_is_probed_not_scored(self):
        s = SessionState(track="dsa")
        assert s.decide(signals(word_count=5, content_overall=0.9)) is Move.PROBE


class TestBookkeeping:
    def test_new_question_resets_probe_counters(self):
        s = SessionState(track="dsa")
        s.apply(Move.PROBE)
        assert s.probes_on_current == 1
        s.apply(Move.NEXT_QUESTION)
        assert s.probes_on_current == 0
        assert s.questions_asked == 1

    def test_probes_do_not_consume_question_budget(self):
        s = SessionState(track="dsa")
        for _ in range(2):
            s.apply(Move.PROBE)
        assert s.questions_asked == 0

    def test_closing_ends_the_session(self):
        s = SessionState(track="dsa")
        s.apply(Move.CLOSING)
        assert s.state.value == "ended"

    def test_session_terminates_within_budget(self):
        """Property: no signal sequence can make the interview run forever."""
        s = SessionState(track="dsa", question_budget=5)
        for i in range(200):
            move = s.decide(signals(content_overall=0.1 if i % 3 else 0.9))
            s.apply(move)
            if move is Move.CLOSING:
                break
        else:
            pytest.fail("interview never reached CLOSING within 200 turns")
        assert s.questions_asked <= 6


class TestWeakestAxis:
    def test_identifies_the_lowest_axis(self):
        s = signals(relevance=0.2, completeness=0.8, structure=0.8, correctness_signal=0.8)
        assert "relevance" in s.weakest_axis()

    def test_returns_none_when_everything_is_fine(self):
        assert signals(relevance=0.8, completeness=0.8, structure=0.8,
                       correctness_signal=0.8).weakest_axis() is None


class TestCVAggregator:
    def test_stillness_counts_as_a_freeze(self):
        agg = CVAggregator()
        for i in range(30):
            agg.add(CVSample(ts_ms=i * 200, eye_contact=True, gaze_offset=0.1,
                             posture=0.8, blink=False, head_movement=0.001))
        assert agg.features()["longest_freeze_ms"] >= 5000

    def test_camera_dropout_is_not_a_freeze(self):
        """Wifi problems must not be scored as a stress response."""
        agg = CVAggregator()
        for i in range(10):
            agg.add(CVSample(ts_ms=i * 200, eye_contact=True, gaze_offset=0.1,
                             posture=0.8, blink=False, head_movement=0.001))
        for i in range(10):  # 5s gap, then resume
            agg.add(CVSample(ts_ms=7000 + i * 200, eye_contact=True, gaze_offset=0.1,
                             posture=0.8, blink=False, head_movement=0.001))
        assert agg.features()["longest_freeze_ms"] < 3000

    def test_absent_face_yields_no_features(self):
        agg = CVAggregator()
        for i in range(10):
            agg.add(CVSample(ts_ms=i * 200, eye_contact=False, gaze_offset=0.0,
                             posture=0.0, blink=False, head_movement=0.0, face_present=False))
        assert agg.features() == {}
        assert not agg.has_data

    def test_eye_contact_percentage(self):
        agg = CVAggregator()
        for i in range(10):
            agg.add(CVSample(ts_ms=i * 200, eye_contact=i < 7, gaze_offset=0.2,
                             posture=0.8, blink=False, head_movement=0.1))
        assert abs(agg.features()["eye_contact_pct"] - 0.7) < 0.01

    def test_timeline_buckets(self):
        agg = CVAggregator()
        for i in range(50):
            agg.add(CVSample(ts_ms=i * 200, eye_contact=True, gaze_offset=0.1,
                             posture=0.8, blink=False, head_movement=0.1))
        tl = agg.timeline(bucket_ms=2000)
        assert 4 <= len(tl) <= 7
        assert all("eye_contact" in b for b in tl)
