"""Shared, process-wide model instances.

Everything here is loaded exactly once at startup and shared across sessions.
Loading per-session would multiply the ~350MB resident footprint by the
concurrency count and blow any free tier instantly. All the shared components
are stateless per call (or internally thread-safe), which is what makes the
sharing safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .audio.asr import BaseASR, build_asr
from .config import Settings
from .interviewer.brain import build_brain
from .interviewer.tts import build_tts
from .scoring.content import ContentScorer, build_embedder
from .scoring.delivery import build_delivery_scorer
from .storage.blobs import BlobStore
from .storage.store import Store

log = logging.getLogger(__name__)


@dataclass
class Deps:
    asr: BaseASR
    brain: object
    tts: object
    delivery_scorer: object
    content_scorer: ContentScorer
    store: Store
    blobs: BlobStore | None = None

    def describe(self) -> dict:
        return {
            "asr": self.asr.name,
            "brain": self.brain.name,
            "tts": self.tts.name,
            "delivery_scorer": self.delivery_scorer.name,
            "embedder": getattr(self.content_scorer._embedder, "name", "lexical"),
            "database": self.store.url.split("://")[0],
            "recordings": "on" if self.blobs else "off",
        }


def build_deps(settings: Settings) -> Deps:
    log.info("Loading models -- this is the slow part of cold start")
    blobs = None
    if settings.store_audio:
        blobs = BlobStore(
            root=settings.blob_dir,
            sample_rate=settings.sample_rate,
            retention_days=settings.blob_retention_days,
            max_total_mb=settings.blob_max_total_mb,
        )

    store = Store(settings.database_url)
    deps = Deps(
        asr=build_asr(settings.asr_backend, settings.asr_model, settings.sample_rate),
        brain=build_brain(
            settings.brain_backend, settings.brain_model,
            settings.anthropic_api_key, settings.openai_api_key,
            ollama_model=settings.ollama_model, ollama_host=settings.ollama_host,
            first_sentence_timeout=settings.brain_first_sentence_timeout_s,
        ),
        tts=build_tts(settings.tts_backend),
        delivery_scorer=build_delivery_scorer(settings.delivery_model_path),
        content_scorer=ContentScorer(build_embedder(settings.embed_backend)),
        store=store,
        blobs=blobs,
    )

    # Prune on boot rather than on a timer: this process sleeps on free tiers,
    # so a background scheduler would mostly not run. Startup is the one moment
    # we're reliably alive.
    if blobs:
        blobs.prune(keep_sessions=store.starred_ids())

    log.info("Ready: %s", deps.describe())
    return deps
