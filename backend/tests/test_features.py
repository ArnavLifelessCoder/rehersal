"""Blob storage, progress trends, rate limiting, and Ollama output sanitising."""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.interviewer.ollama import sanitise
from app.ratelimit import SlidingWindowLimiter
from app.storage.blobs import BlobStore
from app.storage.store import MIN_SESSIONS_FOR_TREND, _trend

SR = 16_000


def tone(seconds: float, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestBlobStore:
    def test_round_trips_audio(self, tmp_path):
        store = BlobStore(root=tmp_path, sample_rate=SR)
        key = store.save_turn("abc123", 0, tone(0.5))
        assert key
        data = store.read("abc123", 0)
        assert data and data[:4] == b"RIFF"
        # 0.5s of 16kHz mono int16 = 16000 bytes of PCM, plus a 44-byte header.
        assert 15_000 < len(data) < 17_000

    def test_missing_turn_returns_none(self, tmp_path):
        assert BlobStore(root=tmp_path).read("nope", 0) is None

    def test_empty_audio_is_not_written(self, tmp_path):
        store = BlobStore(root=tmp_path)
        assert store.save_turn("s", 0, np.zeros(0, dtype=np.float32)) is None

    def test_rejects_path_traversal(self, tmp_path):
        """Session ids arrive over the wire and become filesystem paths."""
        store = BlobStore(root=tmp_path)
        store.save_turn("../../evil", 0, tone(0.1))
        assert not (tmp_path.parent / "evil").exists()
        assert not list(tmp_path.parent.glob("evil*"))

    def test_delete_removes_the_session(self, tmp_path):
        store = BlobStore(root=tmp_path)
        store.save_turn("gone", 0, tone(0.2))
        store.delete_session("gone")
        assert store.read("gone", 0) is None

    def test_prune_drops_old_sessions(self, tmp_path):
        store = BlobStore(root=tmp_path, retention_days=1)
        store.save_turn("old", 0, tone(0.2))
        store.save_turn("new", 0, tone(0.2))
        old_dir = tmp_path / "old"
        stale = time.time() - 5 * 86_400
        for f in old_dir.glob("*.wav"):
            import os
            os.utime(f, (stale, stale))

        store.prune()
        assert store.read("old", 0) is None
        assert store.read("new", 0) is not None

    def test_prune_keeps_starred_sessions(self, tmp_path):
        store = BlobStore(root=tmp_path, retention_days=1)
        store.save_turn("kept", 0, tone(0.2))
        stale = time.time() - 5 * 86_400
        import os
        for f in (tmp_path / "kept").glob("*.wav"):
            os.utime(f, (stale, stale))

        store.prune(keep_sessions={"kept"})
        assert store.read("kept", 0) is not None

    def test_prune_enforces_the_size_quota(self, tmp_path):
        # 1MB cap; each 2s turn is ~64KB, so 40 sessions blow well past it.
        store = BlobStore(root=tmp_path, retention_days=999, max_total_mb=1)
        for i in range(40):
            store.save_turn(f"s{i:02d}", 0, tone(2.0))
        assert store.total_bytes() > 1024 * 1024
        store.prune()
        assert store.total_bytes() <= 1024 * 1024


class TestTrend:
    @staticmethod
    def _points(deliveries, fillers=None):
        fillers = fillers or [5.0] * len(deliveries)
        return [
            {"delivery": d, "content": d, "filler_per_min": f}
            for d, f in zip(deliveries, fillers)
        ]

    def test_refuses_to_report_below_the_minimum(self):
        out = _trend(self._points([0.5, 0.9]))
        assert out["available"] is False
        assert out["needed"] == MIN_SESSIONS_FOR_TREND

    def test_never_compares_a_single_session_to_a_single_session(self):
        """The whole point of thirds is to avoid a first-vs-last delta."""
        for n in range(MIN_SESSIONS_FOR_TREND, 12):
            out = _trend(self._points([0.5] * n))
            assert out["window"] >= 2, f"{n} sessions collapsed to a 1v1 comparison"

    def test_detects_improvement(self):
        out = _trend(self._points([0.4, 0.45, 0.6, 0.7, 0.8, 0.85]))
        assert out["available"] is True
        assert out["delivery"]["improved"] is True
        assert out["delivery"]["delta"] > 0

    def test_lower_filler_rate_counts_as_improvement(self):
        """The metric where down is good -- the direction must be inverted."""
        out = _trend(self._points([0.6] * 6, fillers=[14, 13, 10, 7, 5, 4]))
        assert out["filler_per_min"]["improved"] is True
        assert out["filler_per_min"]["delta"] < 0

    def test_detects_regression(self):
        out = _trend(self._points([0.9, 0.85, 0.7, 0.5, 0.45, 0.4]))
        assert out["delivery"]["improved"] is False

    def test_flat_is_neither(self):
        out = _trend(self._points([0.6] * 6))
        assert out["delivery"]["improved"] is None

    def test_change_below_the_noise_floor_is_not_progress(self):
        """A 0.005 delta rendered as "+1 better" is noise sold as progress.

        Found in the browser: a card read "+1 better" beside "12 → 2".
        """
        out = _trend(self._points([0.600, 0.601, 0.602, 0.603, 0.604, 0.605]))
        assert out["delivery"]["improved"] is None
        assert out["delivery"]["delta"] != 0    # the change is real, just tiny

    def test_meaningful_change_still_registers(self):
        out = _trend(self._points([0.50, 0.51, 0.52, 0.70, 0.71, 0.72]))
        assert out["delivery"]["improved"] is True

    def test_filler_noise_floor(self):
        out = _trend(self._points([0.6] * 6, fillers=[5.0, 5.1, 5.0, 5.2, 5.1, 5.2]))
        assert out["filler_per_min"]["improved"] is None

    def test_handles_missing_metrics(self):
        points = [{"delivery": None, "content": None, "filler_per_min": None}] * 8
        out = _trend(points)
        assert out["available"] is True   # no crash, just no metric entries
        assert "delivery" not in out


class TestRateLimiter:
    def test_allows_up_to_the_limit(self):
        lim = SlidingWindowLimiter(limit=3)
        assert all(lim.hit("a")[0] for _ in range(3))
        assert lim.hit("a")[0] is False

    def test_reports_retry_seconds(self):
        lim = SlidingWindowLimiter(limit=1, window_s=60)
        lim.hit("a")
        allowed, retry = lim.hit("a")
        assert not allowed and 0 < retry <= 60

    def test_keys_are_independent(self):
        lim = SlidingWindowLimiter(limit=1)
        assert lim.hit("a")[0] and lim.hit("b")[0]

    def test_window_slides(self):
        lim = SlidingWindowLimiter(limit=2, window_s=0.25)
        lim.hit("a"); lim.hit("a")
        assert lim.hit("a")[0] is False
        time.sleep(0.3)
        assert lim.hit("a")[0] is True

    def test_check_does_not_consume(self):
        lim = SlidingWindowLimiter(limit=1)
        assert lim.check("a")[0] and lim.check("a")[0]
        assert lim.hit("a")[0]

    def test_sweeps_stale_keys(self):
        """Otherwise every unique IP leaks a dict entry forever."""
        lim = SlidingWindowLimiter(limit=5, window_s=0.1, max_keys=10)
        for i in range(30):
            lim.hit(f"ip{i}")
        time.sleep(0.15)
        lim.hit("trigger")
        assert len(lim._hits) < 30


class TestOllamaModelSubstitution:
    """A running Ollama with the "wrong" models should still be used.

    Found by running the app: the user had nine models pulled, none matching
    the configured tag, and silently got the concept graph while a perfectly
    good 4B model sat idle.
    """

    @staticmethod
    def _pick(names):
        from app.interviewer.ollama import OllamaBrain
        return OllamaBrain._pick_substitute(names)

    def test_deprioritises_reasoning_models(self):
        """qwen3 emits a <think> block that ate the whole token budget and
        returned an empty message. Measured: 9s to produce nothing."""
        assert self._pick(["llama3:latest", "qwen3:4b", "mistral:latest"]) != "qwen3:4b"

    def test_still_uses_a_reasoning_model_if_it_is_all_there_is(self):
        assert self._pick(["qwen3:4b"]) == "qwen3:4b"

    def test_prefers_smaller_within_a_family(self):
        assert self._pick(["llama3.2:70b", "llama3.2:3b"]) == "llama3.2:3b"

    def test_skips_embedding_models(self):
        """They cannot chat at all -- picking one would break every turn."""
        assert self._pick(["nomic-embed-text:latest", "llama3.2:3b"]) == "llama3.2:3b"

    def test_skips_cloud_models(self):
        """Defeats the point: this path exists to be local and free."""
        assert self._pick(["deepseek-v4-pro:cloud", "phi:latest"]) == "phi:latest"

    def test_returns_none_when_nothing_is_usable(self):
        assert self._pick(["nomic-embed-text:latest", "bge-m3:latest"]) is None

    def test_empty_list_is_safe(self):
        assert self._pick([]) is None

    def test_the_real_world_case(self):
        """Exactly what was installed on the machine that surfaced this.

        Must pick a direct-answering model, not the cloud one (defeats the
        point), not the embedder (can't chat), not qwen3 (reasoning).
        """
        installed = [
            "deepseek-v4-pro:cloud", "llama2:latest", "llama3.2:3b", "llama3:latest",
            "mistral:latest", "nomic-embed-text:latest", "phi:latest",
            "qwen2:latest", "qwen3:4b",
        ]
        picked = self._pick(installed)
        assert picked == "llama3.2:3b", picked


class TestBrainTimeout:
    """The fallback timeout must not be shorter than local generation.

    4s was tuned for hosted APIs. A warm 3B model on CPU measured 5.5-7s per
    question, so ResilientBrain fell back on *every single turn* while the
    startup log cheerfully reported 'Interviewer brain: ollama'. The LLM was
    never once used, and nothing said so.
    """

    def test_default_exceeds_measured_local_generation(self):
        from app.config import Settings
        assert Settings().brain_first_sentence_timeout_s >= 10.0

    def test_factory_passes_the_timeout_through(self):
        from app.interviewer.brain import BrainFactory, ResilientBrain
        from app.interviewer.semantic import SemanticBrain

        class Dummy:
            name = "dummy"

        factory = BrainFactory("dummy", "m", lambda: Dummy(), first_sentence_timeout=9.5)
        brain = factory.build("dsa")
        assert isinstance(brain, ResilientBrain)
        assert brain._timeout == 9.5

    def test_repeated_fallbacks_escalate_to_error(self, caplog):
        """A brain that always falls back is not installed -- say so loudly."""
        import asyncio, logging

        from app.interviewer.brain import ResilientBrain
        from app.interviewer.semantic import SemanticBrain

        class Broken:
            name = "broken"

            async def ask(self, **kw):
                raise RuntimeError("nothing usable")
                yield ""  # pragma: no cover

        brain = ResilientBrain(Broken(), SemanticBrain("dsa", seed=1), 1.0)

        async def drain():
            async for _ in brain.ask(move="opening", track="dsa"):
                pass

        with caplog.at_level(logging.WARNING):
            asyncio.run(drain())
            asyncio.run(drain())

        assert any(r.levelno == logging.ERROR for r in caplog.records), (
            "a persistently failing brain should escalate beyond WARNING"
        )


class TestEnvFileResolution:
    """backend/.env must load regardless of the process working directory.

    A bare relative `env_file` silently did nothing whenever the server was
    started from the repo root -- which is how the launch config, most editor
    run-configs and typical Docker entrypoints start it. The README tells people
    to create `backend/.env`, so their configuration was being ignored while the
    app looked like it was working.
    """

    def test_env_paths_are_absolute(self):
        from app.config import ENV_FILES
        assert all(p.is_absolute() for p in ENV_FILES)

    def test_backend_env_is_searched(self):
        from app.config import BACKEND_ROOT, ENV_FILES
        assert BACKEND_ROOT / ".env" in ENV_FILES

    def test_repo_root_env_is_searched_too(self):
        from app.config import ENV_FILES, REPO_ROOT
        assert REPO_ROOT / ".env" in ENV_FILES


class TestUsableTurn:
    """Reject model output that isn't an interview turn.

    Both failure modes here were produced by real 3B models on this machine,
    not imagined: an empty message (reasoning model burned its budget on
    <think>), and the candidate's own sentence echoed back with no question.
    """

    @staticmethod
    def _ok(text, move="probe", answer=None):
        from app.interviewer.ollama import is_usable_turn
        return is_usable_turn(text, move, answer)

    def test_accepts_a_real_question(self):
        assert self._ok("What happens when two keys collide in that map?")

    def test_rejects_empty(self):
        assert not self._ok("")

    def test_rejects_a_statement_where_a_question_is_required(self):
        assert not self._ok("That is an interesting approach to the problem.")

    def test_allows_statements_for_hint_and_closing(self):
        assert self._ok("Try the smallest version of the problem first.", move="hint")
        assert self._ok("Thanks for your time today.", move="closing")

    def test_rejects_an_echo_of_the_candidate(self):
        """Observed: probe returned the answer verbatim, technically 'quoting'."""
        answer = "I would just use a hash map to store the counts and then look them up"
        assert not self._ok("Just use a hash map to store the counts and then look them up?",
                            answer=answer)

    def test_accepts_a_probe_that_quotes_but_adds_a_question(self):
        answer = "I would just use a hash map to store the counts and then look them up"
        assert self._ok(
            "You reached for a hash map. What happens when two inputs collide "
            "in the same bucket, and how does that change your worst case?",
            answer=answer,
        )

    def test_short_answers_do_not_trigger_the_echo_check(self):
        """Too little text to judge overlap meaningfully."""
        assert self._ok("Can you say more about that?", answer="yes ok")


class TestOllamaSanitiser:
    @pytest.mark.parametrize("raw", [
        "Great question! What's your eviction policy?",
        "Interesting. What's your eviction policy?",
        "Thanks for that answer. What's your eviction policy?",
    ])
    def test_strips_sycophancy(self, raw):
        out = sanitise(raw)
        assert out.startswith("What's your eviction policy")

    def test_strips_role_labels(self):
        assert sanitise("Interviewer: What happens at scale?") == "What happens at scale?"

    def test_strips_markdown(self):
        assert "*" not in sanitise("What about **latency** here?")

    def test_keeps_only_the_first_question(self):
        """Small instruct models stack three questions into one turn."""
        out = sanitise("What's the TTL? How do you invalidate? What about cold starts?")
        assert out == "What's the TTL?"

    def test_keeps_setup_before_the_question(self):
        out = sanitise("You mentioned caching. What's the TTL? And the hit rate?")
        assert out == "You mentioned caching. What's the TTL?"

    def test_collapses_whitespace_and_quotes(self):
        assert sanitise('  "What   happens\n\nnext?"  ') == "What happens next?"

    def test_capitalises(self):
        assert sanitise("what happens next?").startswith("What")

    def test_passes_clean_text_through(self):
        text = "What's the complexity of that approach?"
        assert sanitise(text) == text

    def test_empty_input_is_safe(self):
        assert sanitise("") == ""
