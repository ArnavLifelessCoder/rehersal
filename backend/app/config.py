"""Runtime configuration.

Every knob is env-overridable with the REHEARSAL_ prefix so the same image runs
locally on SQLite with no models and in prod on Postgres with hosted APIs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

# Absolute, and checked in both plausible locations. A bare ".env" resolves
# against the process working directory, so it silently did nothing whenever
# the server was launched from the repo root rather than backend/ -- which is
# how every editor run-config and most Docker entrypoints start it. Config that
# is quietly ignored is worse than config that errors: the app looks like it
# works and behaves as though you never set anything.
ENV_FILES = (BACKEND_ROOT / ".env", REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REHEARSAL_",
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    database_url: str = f"sqlite:///{(DATA_DIR / 'rehearsal.db').as_posix()}"
    blob_dir: Path = DATA_DIR / "blobs"

    # --- audio ---
    sample_rate: int = 16_000
    frame_ms: int = 20

    # --- turn-taking ---
    # Fast, natural turn-taking with responsive endpointing and thinking grace.
    endpoint_silence_ms: int = 1_000
    # Extra silence allowed when the utterance trails off mid-thought.
    thinking_grace_ms: int = 450
    # Extra silence allowed while the answer is still very short.
    opening_grace_ms: int = 500
    opening_grace_words: int = 8

    stuck_silence_ms: int = 10_000
    ramble_limit_ms: int = 120_000
    # Noise rejection.
    min_utterance_ms: int = 400
    speech_onset_ms: int = 140
    min_answer_words: int = 3
    # Turn-level energy floor.
    min_turn_rms: float = 0.004
    # Settle window for echo-cancellation.
    floor_settle_ms: int = 350
    max_questions_per_session: int = 12

    # --- adapters: "auto" picks the real one if importable, else the fallback ---
    asr_backend: str = "auto"        # auto | faster-whisper | mock
    # tiny.en by default: it decodes at ~0.04x realtime on CPU, which is what
    # makes the transcript appear live while you talk. base.en is ~7x slower
    # (a 30s answer flushes in ~9s -- the interviewer reads as broken) and
    # small.en needs a GPU. Choose accuracy over speed only on a fast machine.
    asr_model: str = "tiny.en"
    vad_backend: str = "auto"        # auto | silero | spectral | energy
    brain_backend: str = "auto"      # auto | anthropic | openai | ollama | semantic
    brain_model: str = "claude-haiku-4-5-20251001"
    # Free local LLM. `ollama pull qwen2.5:3b-instruct` then set backend=ollama,
    # or leave on auto -- if Ollama is already running it's preferred over the
    # concept graph automatically.
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_host: str = "http://127.0.0.1:11434"
    # How long to wait for a brain's first sentence before using the concept
    # graph instead. 4s was tuned for hosted APIs and far too tight for CPU
    # inference: a warm 3B model measures 3-10s per turn (median ~7s), so it
    # fell back on *every* turn while the logs claimed the LLM was in use.
    # Generous, because the alternative is never using the model you installed
    # -- and the graph is still there when this expires.
    brain_first_sentence_timeout_s: float = 15.0
    tts_backend: str = "auto"        # auto | piper | browser
    embed_backend: str = "auto"      # auto | sentence-transformers | lexical

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- trained artefacts (absent => calibrated heuristic) ---
    delivery_model_path: Path = MODELS_DIR / "delivery_scorer.txt"
    rubric_model_path: Path = MODELS_DIR / "rubric_head.txt"

    # --- latency budget, ms; breaches are logged and surfaced to the client ---
    budget_asr_flush_ms: int = 100
    budget_content_check_ms: int = 50
    budget_llm_first_token_ms: int = 400
    budget_tts_first_chunk_ms: int = 250

    # --- recordings: what makes worst-moment playback possible ---
    store_audio: bool = True
    blob_retention_days: int = 7
    blob_max_total_mb: int = 512

    # --- abuse limits (needed before any public URL) ---
    max_concurrent_sessions: int = 8
    rate_limit_sessions_per_hour: int = 20

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def post_endpoint_budget_ms(self) -> int:
        """Budget for the work we actually control and measure.

        Timing starts when the endpointer fires, i.e. AFTER the deliberate
        `endpoint_silence_ms` wait. Measuring against the full user-facing
        budget would flatter every number by 700ms of work we never did.
        """
        return (
            self.budget_asr_flush_ms
            + self.budget_content_check_ms
            + self.budget_llm_first_token_ms
            + self.budget_tts_first_chunk_ms
        )

    @property
    def total_response_budget_ms(self) -> int:
        """What the candidate actually experiences: silence wait + our work."""
        return self.endpoint_silence_ms + self.post_endpoint_budget_ms


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    s.blob_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return s
