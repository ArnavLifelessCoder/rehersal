"""Fillers, delivery scoring, content scoring."""

from __future__ import annotations

import pytest

from app.features import FeatureVector
from app.scoring.content import ContentScorer, LexicalEmbedder
from app.scoring.delivery import HeuristicDeliveryScorer
from app.scoring.fillers import detect_fillers, type_token_ratio, words_per_minute


class TestFillers:
    def test_counts_basic_hesitations(self):
        r = detect_fillers("um so uh I think um that works", 10.0)
        assert r.breakdown.get("um") == 2
        assert r.breakdown.get("uh") == 1

    def test_detects_code_mixed_fillers(self):
        r = detect_fillers("matlab the approach is yaar quite simple na", 10.0)
        assert "matlab" in r.breakdown
        assert "yaar" in r.breakdown

    def test_detects_multiword_phrases(self):
        r = detect_fillers("it was you know sort of the right call", 10.0)
        assert r.breakdown.get("you know") == 1
        assert r.breakdown.get("sort of") == 1

    def test_like_as_comparison_is_not_a_filler(self):
        """The whole point of context gating: engineers speak in analogies."""
        r = detect_fillers("it works like a heap and looks like a tree", 10.0)
        assert "like" not in r.breakdown

    def test_like_as_hesitation_is_a_filler(self):
        r = detect_fillers("it was like really hard to debug", 10.0)
        assert r.breakdown.get("like") == 1

    def test_consequential_so_is_not_a_filler(self):
        r = detect_fillers("we cached it so we could serve reads faster", 10.0)
        assert "so" not in r.breakdown

    def test_detects_restarts(self):
        r = detect_fillers("I I think we we should refactor", 10.0)
        assert r.restart_count == 2

    def test_rate_scales_with_duration(self):
        text = "um " * 10 + "words here"
        fast = detect_fillers(text, 30.0)
        slow = detect_fillers(text, 120.0)
        assert fast.per_min > slow.per_min
        assert fast.count == slow.count

    def test_clean_speech_scores_zero(self):
        r = detect_fillers("The algorithm runs in linear time and constant space.", 10.0)
        assert r.count == 0

    def test_empty_text_is_safe(self):
        r = detect_fillers("", 10.0)
        assert r.count == 0 and r.per_min == 0.0

    def test_wpm_and_ttr(self):
        assert words_per_minute("one two three four five six", 60.0) == 6.0
        assert type_token_ratio("a a b b") == 0.5


class TestDeliveryScorer:
    scorer = HeuristicDeliveryScorer()

    def test_good_delivery_beats_poor_delivery(self):
        good = self.scorer.score(FeatureVector.from_partial({
            "words_per_min": 140, "filler_per_min": 1.0, "restart_count": 0,
            "pitch_var": 900, "energy_mean": 0.08, "eye_contact_pct": 0.8,
            "posture_score": 0.85, "gaze_stability": 0.8, "longest_freeze_ms": 0,
            "longest_pause_ms": 500, "mean_pause_ms": 300, "type_token_ratio": 0.6,
            "answer_duration_s": 90,
        }))
        poor = self.scorer.score(FeatureVector.from_partial({
            "words_per_min": 215, "filler_per_min": 22.0, "restart_count": 9,
            "pitch_var": 15, "energy_mean": 0.006, "eye_contact_pct": 0.15,
            "posture_score": 0.3, "gaze_stability": 0.2, "longest_freeze_ms": 6000,
            "longest_pause_ms": 5000, "mean_pause_ms": 2200, "type_token_ratio": 0.3,
            "answer_duration_s": 90,
        }))
        assert good.overall > poor.overall + 0.3
        for axis in ("confidence", "clarity", "pace", "presence"):
            assert getattr(good.subscores, axis) > getattr(poor.subscores, axis), axis

    def test_extremes_are_both_penalised_on_pace(self):
        mid = self.scorer.score(FeatureVector.from_partial({"words_per_min": 140}))
        fast = self.scorer.score(FeatureVector.from_partial({"words_per_min": 240}))
        slow = self.scorer.score(FeatureVector.from_partial({"words_per_min": 45}))
        assert mid.subscores.pace > fast.subscores.pace
        assert mid.subscores.pace > slow.subscores.pace

    def test_missing_camera_lowers_confidence_not_presence(self):
        with_cam = self.scorer.score(
            FeatureVector.from_partial({"answer_duration_s": 90}, has_vision=True)
        )
        without = self.scorer.score(
            FeatureVector.from_partial({"answer_duration_s": 90}, has_vision=False)
        )
        assert without.confidence < with_cam.confidence
        assert without.subscores.presence > 0.5, "must abstain to neutral, not score zero"

    def test_short_answer_lowers_confidence(self):
        short = self.scorer.score(FeatureVector.from_partial({"answer_duration_s": 4}))
        long = self.scorer.score(FeatureVector.from_partial({"answer_duration_s": 90}))
        assert short.confidence < long.confidence

    def test_evidence_is_populated_and_ranked(self):
        r = self.scorer.score(FeatureVector.from_partial({
            "filler_per_min": 25.0, "words_per_min": 210, "answer_duration_s": 60
        }))
        assert r.evidence
        contribs = [abs(e["contribution"]) for e in r.evidence]
        assert contribs == sorted(contribs, reverse=True)

    def test_scores_stay_in_range(self):
        for wpm in (0, 50, 140, 300, 1000):
            r = self.scorer.score(FeatureVector.from_partial({"words_per_min": wpm}))
            assert 0.0 <= r.overall <= 1.0


def _available_scorers():
    """Every delivery backend present on this machine.

    The abstention contract is a property of the *product*, not of one
    implementation. Testing it against the heuristic alone is what let the
    trained scorer ship contradicting it.
    """
    from pathlib import Path

    from app.scoring.delivery import build_delivery_scorer

    scorers = [HeuristicDeliveryScorer()]
    trained = build_delivery_scorer(
        Path(__file__).resolve().parents[2] / "models" / "delivery_scorer.txt"
    )
    if trained.name != "heuristic":
        scorers.append(trained)
    return scorers


class TestAbstentionContract:
    """No backend may emit a confident sub-score for a modality it never saw.

    Regression: the LightGBM scorer predicted `presence` from neutral defaults
    when the camera was off, so the report said "camera off -- presence not
    scored" while displaying presence 81. Found by running a real interview in
    a browser with no detectable face.
    """

    @pytest.mark.parametrize("scorer", _available_scorers(), ids=lambda s: s.name)
    def test_presence_abstains_without_vision(self, scorer):
        fv = FeatureVector.from_partial(
            {
                "words_per_min": 150, "filler_per_min": 2.0, "pitch_var": 800,
                "energy_mean": 0.07, "answer_duration_s": 90,
            },
            has_vision=False,
        )
        result = scorer.score(fv)
        # The neutral prior, not a prediction dressed up as a measurement.
        assert result.subscores.presence == pytest.approx(0.65, abs=0.01), (
            f"{scorer.name} invented a presence score with no camera data"
        )

    @pytest.mark.parametrize("scorer", _available_scorers(), ids=lambda s: s.name)
    def test_presence_is_computed_when_vision_exists(self, scorer):
        good = scorer.score(FeatureVector.from_partial(
            {"eye_contact_pct": 0.85, "posture_score": 0.9, "gaze_stability": 0.85,
             "longest_freeze_ms": 0, "answer_duration_s": 90},
            has_vision=True,
        ))
        poor = scorer.score(FeatureVector.from_partial(
            {"eye_contact_pct": 0.1, "posture_score": 0.2, "gaze_stability": 0.15,
             "longest_freeze_ms": 7000, "answer_duration_s": 90},
            has_vision=True,
        ))
        assert good.subscores.presence > poor.subscores.presence, (
            f"{scorer.name} does not discriminate presence when vision IS present"
        )

    @pytest.mark.parametrize("scorer", _available_scorers(), ids=lambda s: s.name)
    def test_evidence_says_so_when_abstaining(self, scorer):
        fv = FeatureVector.from_partial({"answer_duration_s": 60}, has_vision=False)
        notes = " ".join(e["note"] for e in scorer.score(fv).evidence)
        assert "camera off" in notes


class TestContentScorer:
    scorer = ContentScorer(LexicalEmbedder())

    def test_on_topic_beats_off_topic(self):
        q = "How would you find a cycle in a linked list?"
        on = self.scorer.score(
            q,
            "First I would use two pointers on the linked list, a slow and a fast pointer. "
            "Then if they meet there is a cycle. The complexity is O(n) time and O(1) space.",
            "dsa",
        )
        off = self.scorer.score(
            q, "I really enjoy hiking on weekends and my favourite food is pasta.", "dsa"
        )
        assert on.relevance > off.relevance
        assert on.overall > off.overall

    def test_structured_answer_scores_higher_structure(self):
        q = "Tell me about a conflict on your team."
        structured = self.scorer.score(
            q,
            "The situation was that we had a deadline slip. My task was to get us back on track. "
            "So I decided to cut scope and I reached out to the PM. As a result we shipped on time.",
            "behavioral",
        )
        rambling = self.scorer.score(
            q, "There was some disagreement and it got resolved eventually somehow.", "behavioral"
        )
        assert structured.structure > rambling.structure

    def test_hedging_lowers_correctness_signal(self):
        q = "What is the complexity?"
        committed = self.scorer.score(
            q, "The complexity is O(n log n) specifically because we sort first. "
               "For example with a million items that is concretely twenty million operations.",
            "dsa")
        hedged = self.scorer.score(
            q, "I guess maybe it is probably something like that, not sure honestly, "
               "kind of depends I think maybe.", "dsa")
        assert committed.correctness_signal > hedged.correctness_signal

    def test_too_short_answer_returns_low_confidence(self):
        r = self.scorer.score("Explain sharding.", "yes", "system_design")
        assert r.confidence < 0.3
        assert r.overall == 0.0

    def test_lexical_backend_reports_lower_confidence(self):
        r = self.scorer.score(
            "Explain caching.",
            "We put a cache in front of the database to reduce latency and load. " * 4,
            "system_design",
        )
        assert r.confidence <= 0.55, "the fallback must not claim neural-grade certainty"

    def test_jd_track_uses_jd_text(self):
        r = self.scorer.score(
            "What is your experience?",
            "I built distributed streaming pipelines in Kubernetes with Kafka for three years.",
            "jd",
            jd_text="Looking for experience with Kubernetes, Kafka and distributed streaming.",
        )
        assert r.relevance > 0.2
