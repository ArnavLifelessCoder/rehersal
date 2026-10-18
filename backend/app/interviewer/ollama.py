"""Ollama brain — a real language model, locally, for free.

This is the answer to "I want adaptive questions but I'm not paying per
session". Ollama runs a quantised 3B model on CPU; `qwen2.5:3b-instruct` is
about 2GB on disk and produces a usable interviewer question in well under a
second on a modern laptop.

    ollama pull qwen2.5:3b-instruct
    REHEARSAL_BRAIN_BACKEND=ollama python -m uvicorn app.main:app

Quality sits between the concept graph and a hosted frontier model: it handles
novel answers the graph can't, but it needs firmer constraints to stay in
character. Hence the tighter generation settings and the output sanitiser
below -- small instruct models love to emit "Great question!", markdown bullets,
and three questions where one was asked.

Falls back to the concept graph if Ollama isn't running, which is the common
case, so enabling this can never make the product worse.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import AsyncIterator

from .prompts import build_compact_prompt

log = logging.getLogger(__name__)

# 127.0.0.1, not "localhost". On dual-stack hosts (notably Windows) "localhost"
# resolves to ::1 first; with nothing listening there the connection waits for
# an IPv6 timeout before retrying IPv4, which cost ~3.5s on every single boot.
# That lands directly on free-tier cold start, where 30s is already the budget.
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
# Ollama is either local (answers in milliseconds) or absent (refused
# immediately). A long timeout buys nothing and delays startup.
PROBE_TIMEOUT_S = 1.0

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Praise a real interviewer never gives. Matched against a whole leading
# SENTENCE, not a prefix: "Thanks for that answer." stripped as a prefix leaves
# "For that answer.", which is worse than what we started with.
_SYCOPHANCY_SENTENCE = re.compile(
    r"^(great|good|excellent|interesting|nice|perfect|awesome|fantastic|makes sense|"
    r"thank you|thanks|got it|i see|sure|okay|ok)\b",
    re.I,
)
_LABEL = re.compile(r"^\s*(interviewer|question|q)\s*[:\-]\s*", re.I)
_MARKDOWN = re.compile(r"[*_`#>]|^\s*[-•]\s+", re.M)

# Reasoning models (qwen3, deepseek-r1, ...) emit a chain-of-thought block
# before the answer. Two problems in this application: it is not speech, and it
# consumes the token budget -- a 90-token cap on qwen3:4b produced an UNCLOSED
# think block and therefore an entirely empty message. Strip closed blocks, and
# treat an unclosed one as "the model never got to the answer".
_THINK_BLOCK = re.compile(r"<(think|thought|reasoning)>.*?</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(think|thought|reasoning)>.*$", re.S | re.I)


def sanitise(text: str) -> str:
    """Make model output speakable and in-character.

    Kept as a pure function so it can be tested directly -- the failure modes
    here are the difference between an interviewer and a chatbot, and they're
    much easier to pin down as string transforms than as prompt tweaks.
    """
    text = _THINK_BLOCK.sub(" ", text)
    text = _THINK_OPEN.sub(" ", text)
    text = _LABEL.sub("", text.strip())
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"“”')

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]

    # Drop leading pleasantry sentences. Only short, non-question ones: "Sure,
    # but what happens at scale?" is a legitimate interviewer turn, and a long
    # sentence starting with "Good" is probably doing real work.
    while sentences and not sentences[0].endswith("?"):
        first = sentences[0]
        if _SYCOPHANCY_SENTENCE.match(first) and len(first.split()) <= 6:
            sentences.pop(0)
        else:
            break

    # One question per turn. Small models stack them; keep the first question
    # plus any setup before it ("You mentioned caching. What's the TTL?").
    kept: list[str] = []
    for s in sentences:
        kept.append(s)
        if s.endswith("?"):
            break
    text = " ".join(kept) if kept else " ".join(sentences)

    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


# Moves whose output is not required to be a question.
_STATEMENT_MOVES = {"hint", "closing"}


def is_usable_turn(text: str, move: str, answer: str | None = None) -> bool:
    """Is this a plausible interviewer turn, or model noise?

    Kept separate and pure so the failure modes are testable directly. Both of
    these were observed from real 3B models, not hypothesised.
    """
    if not text or len(text) < 8:
        return False

    if move not in _STATEMENT_MOVES and "?" not in text:
        return False

    # Echo check: a "probe" that is just the candidate's sentence played back is
    # not a probe. Compare content words, since the model may re-case or clip.
    if answer:
        words = re.findall(r"[a-z']{4,}", text.lower())
        said = set(re.findall(r"[a-z']{4,}", answer.lower()))
        if words and len(said) >= 4:
            overlap = sum(1 for w in words if w in said) / len(words)
            if overlap > 0.85:
                return False
    return True


class OllamaBrain:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: float = 20.0) -> None:
        import httpx

        self._httpx = httpx
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    @property
    def model(self) -> str:
        """The model actually in use -- may differ from the one requested if
        `probe()` substituted an installed one."""
        return self._model

    def probe(self) -> bool:
        """Check Ollama is up and a usable model is present. Called at startup.

        If the configured model is missing but the daemon is running with other
        instruct-capable models, we adopt one rather than falling back to the
        concept graph. A running Ollama is a strong signal the user wants a real
        LLM; refusing over an exact tag mismatch means they silently get the
        weaker interviewer while a perfectly good 4B model sits idle.
        """
        try:
            with self._httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
                res = client.get(f"{self._host}/api/tags")
                res.raise_for_status()
                names = [m.get("name", "") for m in res.json().get("models", []) if m.get("name")]
            if not names:
                log.warning("ollama: reachable but no models pulled")
                return False

            # Ollama reports "qwen2.5:3b-instruct"; accept a bare-name match too.
            base = self._model.split(":")[0]
            if self._model in names or any(n.split(":")[0] == base for n in names):
                return True

            if (substitute := self._pick_substitute(names)) is not None:
                log.warning(
                    "ollama: %r not installed; using %r instead "
                    "(run `ollama pull %s` for the configured model)",
                    self._model, substitute, self._model,
                )
                self._model = substitute
                return True

            log.warning(
                "ollama: no usable chat model. Installed: %s. Run: ollama pull %s",
                ", ".join(sorted(names)), self._model,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            log.info("ollama: not available at %s (%s)", self._host, exc)
            return False

    def warm(self) -> None:
        """Load the model into memory so the first real turn isn't the slow one.

        Cold load measured ~14s against ~4-5s warm, and it lands on the opening
        question -- the one moment the candidate is definitely paying attention.

        Runs on a daemon thread: even a 2s client timeout is 2s added to every
        cold start, which on a free tier is charged against a 30s budget that
        already has models to load. A failure costs nothing; the model just
        loads on first use as before.
        """
        def _warm() -> None:
            try:
                with self._httpx.Client(timeout=120.0) as client:
                    client.post(
                        f"{self._host}/api/chat",
                        json={
                            "model": self._model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": False,
                            "options": {"num_predict": 1},
                            # Keep it resident well past a typical session.
                            "keep_alive": "30m",
                        },
                    )
                log.info("ollama: %s warmed and resident", self._model)
            except Exception as exc:  # noqa: BLE001
                log.info("ollama: warm-up skipped (%s)", exc)

        threading.Thread(target=_warm, name="ollama-warm", daemon=True).start()
        log.info("ollama: warming %s in the background", self._model)

    @staticmethod
    def _pick_substitute(names: list[str]) -> str | None:
        """Best locally-installed stand-in, or None.

        Prefers small instruct-tuned models: this is a latency-sensitive path
        where a 4B model answering in under a second beats a 70B one that makes
        the candidate wait, and embedding-only models can't chat at all.
        """
        unusable = ("embed", "nomic", "bge", "minilm", "vision", "clip", "coder")
        cloud = ":cloud"
        candidates = [
            n for n in names
            if not any(u in n.lower() for u in unusable) and cloud not in n.lower()
        ]
        if not candidates:
            return None

        def rank(name: str) -> tuple[int, int, float]:
            lowered = name.lower()
            # Reasoning models emit a <think> block before answering. That is
            # dead weight on a latency-sensitive path where we cap output at ~90
            # tokens: qwen3:4b spent the entire budget reasoning and returned an
            # empty message. Usable if nothing else exists, but never preferred.
            reasoning = 1 if any(r in lowered for r in ("qwen3", "r1", "reason", "qwq")) else 0
            # Families known to follow instructions well at small sizes.
            family = 0 if any(f in lowered for f in ("qwen", "llama3", "mistral", "phi")) else 1
            # Parameter count, if the tag advertises one ("3b", "4b", "70b").
            size = 999.0
            if (m := re.search(r"(\d+(?:\.\d+)?)\s*b\b", lowered)):
                size = float(m.group(1))
            return (reasoning, family, size)

        return min(candidates, key=rank)

    async def ask(self, *, move: str, track: str, **kw) -> AsyncIterator[str]:
        # Compact prompt: on CPU the cost is prompt evaluation, not generation.
        # A full-history prompt measured 413 prompt tokens -> 8 output tokens in
        # 12.4s; the compact form is roughly half that, which is the difference
        # between the model being used and timing out on every turn.
        system, prompt = build_compact_prompt(
            move=move,
            track=track,
            question=kw.get("question"),
            answer=kw.get("answer"),
            weakest_axis=kw.get("weakest_axis"),
        )
        body = {
            "model": self._model,
            "stream": True,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.5,
                "top_p": 0.85,
                "repeat_penalty": 1.15,
                "num_predict": 120,
                "stop": ["Candidate:", "You:", "Interviewer:", "\n\n"],
            },
        }

        buffer = ""
        emitted_any = False
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._host}/api/chat", json=body) as res:
                res.raise_for_status()
                async for line in res.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"ollama: {chunk['error']}")
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        buffer += piece
                        # If a complete sentence is ready and not inside an open think tag, yield it early
                        if not _THINK_OPEN.search(buffer):
                            parts = _SENTENCE_END.split(buffer, maxsplit=1)
                            if len(parts) >= 2 and len(parts[0].strip()) >= 10:
                                candidate_sentence = sanitise(parts[0])
                                if candidate_sentence and is_usable_turn(candidate_sentence, move, answer=kw.get("answer")):
                                    emitted_any = True
                                    yield candidate_sentence
                                    buffer = parts[1]
                                    if move not in _STATEMENT_MOVES and candidate_sentence.endswith("?"):
                                        return
                    if chunk.get("done"):
                        break

        # Flush any remaining sentence in buffer
        if buffer.strip():
            cleaned = sanitise(buffer)
            if cleaned:
                if not emitted_any and not is_usable_turn(cleaned, move, answer=kw.get("answer")):
                    raise RuntimeError(f"unusable output: {cleaned[:80]!r}")
                for sentence in _SENTENCE_END.split(cleaned):
                    if sentence.strip():
                        emitted_any = True
                        yield sentence.strip()
                        if move not in _STATEMENT_MOVES and sentence.strip().endswith("?"):
                            break

        if not emitted_any:
            raise RuntimeError("ollama returned nothing usable")


def build_ollama(model: str, host: str):
    """Return a probed OllamaBrain, or None if unavailable."""
    try:
        brain = OllamaBrain(model, host)
    except ImportError:
        log.info("ollama: httpx not installed")
        return None
    if not brain.probe():
        return None
    brain.warm()
    return brain
