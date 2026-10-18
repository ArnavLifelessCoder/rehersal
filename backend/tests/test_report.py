"""Report building -- especially the honesty guarantees.

This product assigns numbers to people. The rules it must not break:
  * a caveat has to name its actual cause, or the user can't act on it;
  * every "worst moment" must cite the measurement behind it;
  * a low-confidence score must pull less weight than a high-confidence one.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from app.scoring.report import build_report
from app.session.cv_features import CVAggregator, CVSample
from app.session.latency import LatencyLog
from app.session.manager import TurnRecord
from app.session.state import SessionState
from app.session.trace import SessionTrace


def make_turn(idx, *, features=None, delivery_overall=0.7, content_overall=0.7,
              conf=0.8, fillers=None, transcript="a fairly normal answer here",
              duration=45.0):
    return TurnRecord(
        idx=idx, role="candidate", transcript=transcript,
        question_text="Tell me about a hard bug.", duration_s=duration,
        features=features or {"words_per_min": 140, "word_count": 100,
                              "filler_per_min": 2.0, "longest_pause_ms": 500,
                              "longest_freeze_ms": 0, "eye_contact_pct": 0.75},
        delivery={
            "overall": delivery_overall, "confidence": conf,
            "subscores": {"confidence": 0.7, "clarity": 0.7, "pace": 0.7, "presence": 0.7},
            "evidence": [], "backend": "heuristic",
        },
        content={
            "overall": content_overall, "confidence": conf, "relevance": 0.7,
            "structure": 0.7, "completeness": 0.7, "correctness_signal": 0.7,
            "notes": {"structure_elements": ["opening"]}, "backend": "lexical",
        },
        fillers=fillers or {"filler_count": 3, "breakdown": {"um": 3}, "spans": [],
                            "restart_count": 0},
    )


def make_session(turns, *, cv_samples=None, track="behavioral"):
    cv = CVAggregator()
    for s in cv_samples or []:
        cv.add(s)
    return SimpleNamespace(
        id="s1", track=track, turns=turns,
        started_at=time.time() - 300, ended_at=time.time(),
        state=SessionState(track=track, question_budget=6),
        latency=LatencyLog(),
        trace=SessionTrace(),
        _session_cv=cv,
        _asr=SimpleNamespace(backend="mock"),
        brain=SimpleNamespace(name="semantic"),
        deps=SimpleNamespace(
            delivery_scorer=SimpleNamespace(name="heuristic"),
            content_scorer=SimpleNamespace(_embedder=SimpleNamespace(name="lexical")),
        ),
    )


class TestConfidenceCaveat:
    def test_no_camera_is_named_explicitly(self):
        turns = [make_turn(i, conf=0.3, duration=20.0) for i in range(2)]
        report = build_report(make_session(turns))
        assert report["had_vision"] is False
        assert "camera was off" in report["summary_text"]

    def test_short_session_is_named_explicitly(self):
        turns = [make_turn(0, conf=0.3, duration=12.0)]
        cv = [CVSample(ts_ms=i * 200, eye_contact=True, gaze_offset=0.1, posture=0.8,
                       blink=False, head_movement=0.1) for i in range(20)]
        report = build_report(make_session(turns, cv_samples=cv))
        assert "camera was off" not in report["summary_text"]
        assert "seconds in total" in report["summary_text"]

    def test_confident_session_gets_no_caveat(self):
        turns = [make_turn(i, conf=0.9, duration=90.0) for i in range(4)]
        cv = [CVSample(ts_ms=i * 200, eye_contact=True, gaze_offset=0.1, posture=0.8,
                       blink=False, head_movement=0.1) for i in range(50)]
        report = build_report(make_session(turns, cv_samples=cv))
        assert "directional" not in report["summary_text"]


class TestWeighting:
    def test_low_confidence_turns_pull_less(self):
        """A score we barely trust must not drag the headline as hard."""
        confident_good = make_turn(0, delivery_overall=0.9, conf=0.95)
        unsure_bad = make_turn(1, delivery_overall=0.1, conf=0.05)
        report = build_report(make_session([confident_good, unsure_bad]))
        # A plain mean would give 0.50; weighting must favour the trusted score.
        assert report["overall_delivery"] > 0.7


class TestWorstMoments:
    def test_high_filler_rate_is_flagged_with_the_words(self):
        turns = [make_turn(0,
                           features={"words_per_min": 140, "word_count": 100,
                                     "filler_per_min": 18.0, "longest_pause_ms": 400,
                                     "longest_freeze_ms": 0},
                           fillers={"filler_count": 15, "restart_count": 1,
                                    "breakdown": {"um": 9, "like": 6},
                                    "spans": [{"token_index": 3, "text": "um"}]})]
        report = build_report(make_session(turns))
        reasons = [m["reason"] for m in report["worst_moments"]]
        assert any("fillers per minute" in r for r in reasons)
        assert any('"um" x9' in r for r in reasons)

    def test_every_moment_cites_a_metric_and_value(self):
        turns = [make_turn(0,
                           features={"words_per_min": 215, "word_count": 200,
                                     "filler_per_min": 22.0, "longest_pause_ms": 5200,
                                     "longest_freeze_ms": 4000, "eye_contact_pct": 0.2})]
        report = build_report(make_session(turns))
        assert report["worst_moments"]
        for m in report["worst_moments"]:
            assert m["metric"] and m["value"] is not None
            assert 0.0 <= m["severity"] <= 1.0

    def test_moments_are_ranked_by_severity(self):
        turns = [make_turn(0,
                           features={"words_per_min": 250, "word_count": 200,
                                     "filler_per_min": 30.0, "longest_pause_ms": 6000,
                                     "longest_freeze_ms": 9000, "eye_contact_pct": 0.1})]
        report = build_report(make_session(turns))
        sev = [m["severity"] for m in report["worst_moments"]]
        assert sev == sorted(sev, reverse=True)

    def test_clean_answer_produces_no_worst_moments(self):
        turns = [make_turn(i) for i in range(3)]
        report = build_report(make_session(turns))
        assert report["worst_moments"] == []

    def test_eye_contact_not_flagged_when_camera_absent(self):
        """Without vision data, presence must not generate accusations."""
        turns = [make_turn(0, features={"words_per_min": 140, "word_count": 100,
                                        "filler_per_min": 1.0, "longest_pause_ms": 300,
                                        "longest_freeze_ms": 0})]
        report = build_report(make_session(turns))
        assert not any(m["metric"] == "eye_contact_pct" for m in report["worst_moments"])


class TestStructure:
    def test_report_has_the_fields_the_ui_needs(self):
        turns = [make_turn(i) for i in range(3)]
        report = build_report(make_session(turns))
        for key in ("overall_delivery", "overall_content", "confidence",
                    "delivery_breakdown", "content_breakdown", "delivery_timeline",
                    "worst_moments", "strengths", "summary_text", "transcript",
                    "backends", "filler_breakdown", "had_vision"):
            assert key in report, f"missing {key}"

    def test_empty_session_does_not_crash(self):
        report = build_report(make_session([]))
        assert report["answers_given"] == 0
        assert report["summary_text"]
