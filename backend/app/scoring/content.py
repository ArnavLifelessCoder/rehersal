"""Content scoring: did they actually answer the question, and how well.

Four rubric axes, each computed from signals that are cheap enough to run
inside the 50ms content-check budget in the turn loop:

  relevance          -- semantic similarity to the question (embeddings, or a
                        BM25-ish lexical fallback with stopword removal).
  structure          -- discourse markers, ordering, STAR-ness for behavioural.
  completeness       -- coverage of the track's expected concept anchors.
  correctness_signal -- named explicitly as a *signal*, not a verdict. We do not
                        claim to grade DSA correctness from a transcript; we
                        detect whether the answer commits to complexity claims,
                        tradeoffs and concrete mechanisms. Overclaiming here is
                        the fastest way to make the whole product untrustworthy.

The embedding backend is optional: without sentence-transformers the lexical
path still discriminates, it just blurs paraphrase.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9+#']+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "for", "with", "as", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "would", "could", "should", "can", "will",
    "do", "does", "did", "have", "has", "had", "at", "by", "from", "not", "no",
    "what", "how", "why", "when", "where", "which", "who", "about", "there",
}

STRUCTURE_MARKERS = {
    "opening": {"first", "firstly", "to start", "the problem", "let me", "so the", "my approach"},
    "sequencing": {"then", "next", "after that", "second", "secondly", "finally", "lastly"},
    "reasoning": {"because", "since", "the reason", "which means", "therefore", "that way"},
    "tradeoff": {"tradeoff", "trade off", "however", "on the other hand", "downside", "but the"},
    "closing": {"in summary", "to summarise", "to summarize", "overall", "so in the end", "that's"},
}

STAR_MARKERS = {
    "situation": {"we had", "the situation", "at the time", "the context", "i was working"},
    "task": {"my job", "i was responsible", "the goal", "i needed to", "my task"},
    "action": {"i did", "i built", "i decided", "i reached out", "i implemented", "so i"},
    "result": {"as a result", "the outcome", "we shipped", "it reduced", "we improved", "ended up"},
}

TRACK_ANCHORS: dict[str, set[str]] = {
    "dsa": {
        "complexity", "time", "space", "o(n", "big o", "edge case", "hash", "array",
        "tree", "graph", "sort", "binary", "pointer", "recursion", "iterate", "optimal",
    },
    "system_design": {
        "scale", "latency", "throughput", "cache", "database", "shard", "partition",
        "replica", "queue", "consistency", "availability", "load balancer", "api",
        "bottleneck", "failure", "tradeoff",
    },
    "behavioral": {
        "team", "conflict", "deadline", "stakeholder", "feedback", "learned",
        "outcome", "impact", "responsibility", "decision", "communicate",
    },
    "jd": set(),
}

COMMITMENT_MARKERS = {
    "o(n", "o(1", "o(log", "complexity", "milliseconds", "percent", "%", "throughput",
    "specifically", "for example", "in practice", "the reason is", "concretely",
}

HEDGE_MARKERS = {
    "i guess", "not sure", "maybe", "probably", "i think maybe", "something like that",
    "i don't really know", "kind of", "or whatever",
}


@dataclass
class ContentResult:
    relevance: float
    structure: float
    completeness: float
    correctness_signal: float
    overall: float
    confidence: float
    notes: dict = field(default_factory=dict)
    backend: str = "lexical"

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "structure": self.structure,
            "completeness": self.completeness,
            "correctness_signal": self.correctness_signal,
            "overall": self.overall,
            "confidence": self.confidence,
            "notes": self.notes,
            "backend": self.backend,
        }


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


class LexicalEmbedder:
    """IDF-free cosine over token sets, with light stemming.

    Deliberately simple: it is a *fallback*, and a fallback that pretends to
    semantic nuance would mask the fact that the real embedder failed to load.
    """

    name = "lexical"

    @staticmethod
    def _stem(t: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suf):
                return t[: -len(suf)]
        return t

    def similarity(self, a: str, b: str) -> float:
        ta = {self._stem(t) for t in _tokens(a)}
        tb = {self._stem(t) for t in _tokens(b)}
        if not ta or not tb:
            return 0.0
        overlap = len(ta & tb)
        return overlap / math.sqrt(len(ta) * len(tb))


class SentenceTransformerEmbedder:
    name = "bge-small-en-v1.5"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def similarity(self, a: str, b: str) -> float:
        import numpy as np

        vecs = self._model.encode([a, b], normalize_embeddings=True)
        return float(np.clip(np.dot(vecs[0], vecs[1]), 0.0, 1.0))

    def encode(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_embedder(backend: str):
    if backend in ("auto", "sentence-transformers"):
        try:
            emb = SentenceTransformerEmbedder()
            log.info("Embedder: bge-small-en-v1.5")
            return emb
        except Exception as exc:  # noqa: BLE001
            if backend == "sentence-transformers":
                raise
            log.info("Embedder: lexical fallback (%s)", exc)
    return LexicalEmbedder()


def _marker_hits(text: str, groups: dict[str, set[str]]) -> tuple[float, list[str]]:
    lowered = text.lower()
    hit = [name for name, phrases in groups.items() if any(p in lowered for p in phrases)]
    return len(hit) / len(groups), hit


class ContentScorer:
    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def score(
        self, question: str, answer: str, track: str, jd_text: str | None = None
    ) -> ContentResult:
        words = _WORD_RE.findall(answer.lower())
        if len(words) < 5:
            return ContentResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.2,
                                 {"reason": "answer too short to score"},
                                 getattr(self._embedder, "name", "lexical"))

        # --- relevance -------------------------------------------------------
        sim = self._embedder.similarity(question, answer)
        # Raw cosine sits high for any same-domain text; rescale so the usable
        # band (0.25-0.75 lexical, 0.5-0.9 neural) spans the full 0-1 output.
        lo, hi = (0.5, 0.9) if getattr(self._embedder, "name", "") != "lexical" else (0.12, 0.55)
        relevance = min(max((sim - lo) / (hi - lo), 0.0), 1.0)
        if jd_text:
            jd_sim = self._embedder.similarity(jd_text, answer)
            relevance = 0.75 * relevance + 0.25 * min(max((jd_sim - lo) / (hi - lo), 0.0), 1.0)

        # --- structure -------------------------------------------------------
        groups = STAR_MARKERS if track == "behavioral" else STRUCTURE_MARKERS
        struct_frac, struct_hits = _marker_hits(answer, groups)
        # Very long unsegmented answers lose structure points regardless.
        sentences = max(len(re.findall(r"[.!?]", answer)), 1)
        wps = len(words) / sentences
        structure = min(struct_frac * 1.15, 1.0) * (0.75 if wps > 60 else 1.0)

        # --- completeness ----------------------------------------------------
        anchors = TRACK_ANCHORS.get(track, set())
        if jd_text and not anchors:
            anchors = {t for t in _tokens(jd_text) if len(t) > 4}
        lowered = answer.lower()
        if anchors:
            hits = {a for a in anchors if a in lowered}
            # Saturating: covering 6 of 16 anchors is a solid answer, not 37%.
            completeness = min(len(hits) / max(min(len(anchors), 6), 1), 1.0)
            covered = sorted(hits)[:8]
        else:
            completeness = min(len(words) / 120.0, 1.0)
            covered = []

        # --- correctness signal ----------------------------------------------
        commits = sum(1 for m in COMMITMENT_MARKERS if m in lowered)
        hedges = sum(1 for m in HEDGE_MARKERS if m in lowered)
        correctness = min(commits / 3.0, 1.0) * max(1.0 - hedges * 0.2, 0.3)

        overall = round(
            0.4 * relevance + 0.2 * structure + 0.25 * completeness + 0.15 * correctness, 3
        )
        # Confidence is low for the lexical backend by construction -- it should
        # be, and hiding that would be the dishonest choice.
        base_conf = 0.85 if getattr(self._embedder, "name", "") != "lexical" else 0.55
        confidence = round(base_conf * min(len(words) / 60.0, 1.0) ** 0.5, 3)

        return ContentResult(
            relevance=round(relevance, 3),
            structure=round(structure, 3),
            completeness=round(completeness, 3),
            correctness_signal=round(correctness, 3),
            overall=overall,
            confidence=confidence,
            notes={
                "structure_elements": struct_hits,
                "missing_structure": [g for g in groups if g not in struct_hits],
                "concepts_covered": covered,
                "raw_similarity": round(sim, 3),
                "hedges": hedges,
                "words": len(words),
            },
            backend=getattr(self._embedder, "name", "lexical"),
        )
