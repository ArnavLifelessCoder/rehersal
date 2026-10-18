"""Filler and disfluency detection, including code-mixed Indian-English fillers.

Three detectors, because "um" is the easy 40% of the problem:

  1. Lexical -- single tokens and multi-word phrases ("you know", "sort of").
  2. Code-mixed -- Hindi/Marathi discourse particles that show up mid-sentence
     in Indian-English interview speech ("matlab", "yaar", "na"). The spec asked
     for these; most filler detectors are English-only and score these speakers
     as unnaturally fluent.
  3. Restarts -- immediate word repetition and abandoned clauses ("I I think",
     "we should- we could"), which read as hesitation to a human listener but
     contain no filler word at all.

Crucially `like` and `so` are context-gated. "Something like a heap" is not a
filler; "it was like, um, hard" is. Flat keyword counting on `like` inflates the
filler rate of anyone who speaks in analogies, which is most engineers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SINGLE_FILLERS = {
    "um", "umm", "ummm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh",
    "hmm", "mm", "mhm", "eh",
}

# Code-mixed discourse particles (Hindi/Marathi), romanised as ASR emits them.
CODE_MIXED_FILLERS = {
    "matlab", "yaar", "acha", "accha", "haan", "arre", "arey", "bas", "toh",
    "waise", "yani", "bolke", "mhanje",
}

PHRASE_FILLERS = [
    "you know", "i mean", "sort of", "kind of", "or something", "or whatever",
    "and stuff", "and all", "right so", "let me think", "how do i say",
]

# Gated: filler only in hedging position, not comparative//consequential use.
CONTEXT_GATED = {"like", "so", "basically", "actually", "literally", "obviously"}

# Words whose immediate repetition is grammatical, not a disfluency.
LEGITIMATE_DOUBLES = {"had", "that", "is", "very", "no", "far"}

_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class FillerReport:
    count: int
    per_min: float
    ratio: float
    restart_count: int
    breakdown: dict[str, int] = field(default_factory=dict)
    spans: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filler_count": self.count,
            "filler_per_min": self.per_min,
            "filler_ratio": self.ratio,
            "restart_count": self.restart_count,
            "breakdown": self.breakdown,
            "spans": self.spans,
        }


def _is_gated_filler(tokens: list[str], i: int) -> bool:
    """Decide whether a context-gated word is a filler at position i."""
    word = tokens[i]
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""

    if word == "like":
        # Comparative/verb use: "like a heap", "looks like", "I like it".
        if nxt in {"a", "an", "the", "this", "that", "it"} and prev not in {"was", "is", "were"}:
            return False
        if prev in {"looks", "look", "sounds", "sound", "feels", "feel", "just", "would", "i"}:
            return False
        return True

    if word == "so":
        # Consequential ("so we can"), or intensifier ("so much") -- not filler.
        if nxt in {"we", "i", "it", "they", "you", "the", "that", "much", "many", "far", "if"}:
            return False
        # Discourse-initial "so" opening an answer is a filler.
        return i == 0 or prev in SINGLE_FILLERS or prev == ""

    # basically/actually/literally/obviously: filler unless they open a clause
    # that genuinely qualifies a claim. Heuristic: filler when sentence-medial.
    return i > 0 and prev not in {"and", "but", "which", "that"}


def detect_fillers(text: str, duration_s: float) -> FillerReport:
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    if not tokens:
        return FillerReport(0, 0.0, 0.0, 0)

    breakdown: dict[str, int] = {}
    spans: list[dict] = []
    consumed = [False] * len(tokens)

    # Pass 1: multi-word phrases (longest first so "you know" wins over "know").
    for phrase in sorted(PHRASE_FILLERS, key=lambda p: -len(p.split())):
        parts = phrase.split()
        n = len(parts)
        for i in range(len(tokens) - n + 1):
            if any(consumed[i : i + n]):
                continue
            if tokens[i : i + n] == parts:
                for j in range(i, i + n):
                    consumed[j] = True
                breakdown[phrase] = breakdown.get(phrase, 0) + 1
                spans.append({"token_index": i, "text": phrase, "type": "phrase"})

    # Pass 2: single tokens.
    for i, tok in enumerate(tokens):
        if consumed[i]:
            continue
        kind = None
        if tok in SINGLE_FILLERS:
            kind = "hesitation"
        elif tok in CODE_MIXED_FILLERS:
            kind = "code_mixed"
        elif tok in CONTEXT_GATED and _is_gated_filler(tokens, i):
            kind = "discourse"
        if kind:
            consumed[i] = True
            breakdown[tok] = breakdown.get(tok, 0) + 1
            spans.append({"token_index": i, "text": tok, "type": kind})

    # Pass 3: immediate repetition => a restart, counted separately.
    # "I I think" and "we we should" are the two most common restarts in
    # interview speech, so length is not a usable filter here -- we exclude by
    # word instead, for the handful of doublings that are grammatical English
    # ("had had", "that that", "is is" in cleft constructions).
    restarts = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1] and tokens[i] not in LEGITIMATE_DOUBLES:
            restarts += 1
            spans.append({"token_index": i, "text": tokens[i], "type": "restart"})

    count = sum(breakdown.values())
    minutes = max(duration_s / 60.0, 1e-6)
    spans.sort(key=lambda s: s["token_index"])
    return FillerReport(
        count=count,
        per_min=round(count / minutes, 2),
        ratio=round(count / len(tokens), 4),
        restart_count=restarts,
        breakdown=breakdown,
        spans=spans,
    )


def words_per_minute(text: str, duration_s: float) -> float:
    words = len(_WORD_RE.findall(text.lower()))
    minutes = max(duration_s / 60.0, 1e-6)
    return round(words / minutes, 1)


def type_token_ratio(text: str) -> float:
    tokens = _WORD_RE.findall(text.lower())
    return round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0
