"""The interviewer brain: decides nothing, phrases everything.

The *decision* (probe vs advance vs interrupt) is made by the turn-taking
controller from measured signals -- deterministic, testable, cheap. The brain
only phrases the chosen move. Splitting it this way means an LLM outage
degrades the interview's wording, never its logic, and it makes the state
machine unit-testable without mocking a network call.

All brains stream sentence-by-sentence so TTS can start on sentence one. That
pipelining is what buys the sub-1.5s turn latency.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator

from .prompts import build_system_prompt, build_turn_prompt
from .semantic import SemanticBrain

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


async def _stream_sentences(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup token chunks into whole sentences.

    TTS on a half-sentence produces the wrong prosody, so we buffer to a
    boundary -- but only to the *first* boundary, then emit immediately. That is
    the whole pipelining trick: the candidate hears sentence one while the model
    is still generating sentence two.
    """
    buf = ""
    async for chunk in chunks:
        buf += chunk
        while True:
            parts = _SENTENCE_END.split(buf, maxsplit=1)
            if len(parts) < 2:
                break
            sentence, buf = parts[0].strip(), parts[1]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


class BaseBrain:
    name = "base"

    async def ask(self, **kwargs) -> AsyncIterator[str]:  # pragma: no cover - interface
        raise NotImplementedError
        yield ""  # type: ignore[unreachable]


class AnthropicBrain:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def ask(self, *, move: str, track: str, **kw) -> AsyncIterator[str]:
        prompt = build_turn_prompt(
            move=move,
            question=kw.get("question"),
            answer=kw.get("answer"),
            history=kw.get("history") or [],
            jd_text=kw.get("jd_text"),
            weakest_axis=kw.get("weakest_axis"),
            questions_asked=kw.get("questions_asked", 0),
            question_budget=kw.get("question_budget", 12),
        )

        async def raw() -> AsyncIterator[str]:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=150,
                system=build_system_prompt(track),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        async for sentence in _stream_sentences(raw()):
            yield sentence


class OpenAIBrain:
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def ask(self, *, move: str, track: str, **kw) -> AsyncIterator[str]:
        prompt = build_turn_prompt(
            move=move,
            question=kw.get("question"),
            answer=kw.get("answer"),
            history=kw.get("history") or [],
            jd_text=kw.get("jd_text"),
            weakest_axis=kw.get("weakest_axis"),
            questions_asked=kw.get("questions_asked", 0),
            question_budget=kw.get("question_budget", 12),
        )

        async def raw() -> AsyncIterator[str]:
            stream = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=150,
                stream=True,
                messages=[
                    {"role": "system", "content": build_system_prompt(track)},
                    {"role": "user", "content": prompt},
                ],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        async for sentence in _stream_sentences(raw()):
            yield sentence


class ResilientBrain:
    """Wraps a remote brain with a timeout and a local safety net.

    A hung API call must not hang the interview. If the first sentence hasn't
    arrived within `first_token_timeout`, we abandon the remote call and fall
    through to the local semantic brain -- the candidate gets a slightly duller
    question instead of ten seconds of silence.
    """

    def __init__(self, primary, fallback, first_token_timeout: float = 4.0) -> None:
        self._primary = primary
        self._fallback = fallback
        self._timeout = first_token_timeout
        self._fallbacks = 0

    @property
    def name(self) -> str:
        return self._primary.name

    async def ask(self, **kwargs) -> AsyncIterator[str]:
        try:
            agen = self._primary.ask(**kwargs)
            first = await asyncio.wait_for(agen.__anext__(), timeout=self._timeout)
            yield first
            async for sentence in agen:
                yield sentence
            return
        except StopAsyncIteration:
            return
        except Exception as exc:  # noqa: BLE001 - network, auth, rate limit, timeout
            self._fallbacks += 1
            # Escalate on a repeat: one slow turn is a hiccup, but a brain that
            # falls back every time is effectively not installed, and the logs
            # should say so rather than reading as routine. This silently masked
            # an Ollama model that never produced a usable token.
            level = log.warning if self._fallbacks < 2 else log.error
            level(
                "Brain %s failed (%s); using semantic instead [failure %d]%s",
                self._primary.name, exc, self._fallbacks,
                " -- the configured model is not usable; check its output and speed"
                if self._fallbacks == 2 else "",
            )
        async for sentence in self._fallback.ask(**kwargs):
            yield sentence


class BrainFactory:
    """Builds a per-session brain.

    Brains carry session state -- which concepts the candidate has covered,
    which questions have been asked, which templates are spent. Sharing one
    instance across concurrent interviews makes session B skip the questions
    session A already used, and lets A's covered-concept set steer B's probes.
    So the expensive part (the HTTP client) is shared and the stateful part is
    built per session.
    """

    def __init__(self, kind: str, model: str, client_factory=None,
                 first_sentence_timeout: float = 12.0) -> None:
        self.kind = kind
        self.model = model
        self._client_factory = client_factory
        self._timeout = first_sentence_timeout

    @property
    def name(self) -> str:
        return self.kind

    def build(self, track: str, jd_text: str | None = None):
        local = SemanticBrain(track=track, jd_text=jd_text)
        if self._client_factory is None:
            return local
        try:
            return ResilientBrain(self._client_factory(), local, self._timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("Remote brain unavailable (%s); using semantic", exc)
            return local


def build_brain(
    backend: str,
    model: str,
    anthropic_key: str | None,
    openai_key: str | None,
    *,
    ollama_model: str = "qwen2.5:3b-instruct",
    ollama_host: str = "http://127.0.0.1:11434",
    first_sentence_timeout: float = 12.0,
):
    # Ollama is checked before the paid providers when explicitly requested,
    # and after them on "auto" -- a configured API key is a clear signal of
    # intent, but a local model that's already running is free and should be
    # preferred over the concept graph without the user asking twice.
    if backend == "ollama":
        from .ollama import build_ollama

        if (probe := build_ollama(ollama_model, ollama_host)) is not None:
            # Bind the model the probe SETTLED on, not the one that was asked
            # for. The probe may substitute an installed model, and closing over
            # the requested name would throw that away on every session -- every
            # request would then 404 and silently fall back.
            resolved = probe.model
            log.info("Interviewer brain: ollama (%s)", resolved)
            return BrainFactory(
                "ollama", resolved, lambda: type(probe)(resolved, ollama_host),
                first_sentence_timeout,
            )
        log.warning("Ollama requested but unavailable; using semantic")
        return BrainFactory("semantic", "concept-graph")

    if backend in ("auto", "anthropic") and anthropic_key:
        try:
            AnthropicBrain(anthropic_key, model)  # fail fast on bad config
            log.info("Interviewer brain: anthropic (%s)", model)
            return BrainFactory("anthropic", model,
                                lambda: AnthropicBrain(anthropic_key, model),
                                first_sentence_timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("Anthropic brain unavailable: %s", exc)
    if backend in ("auto", "openai") and openai_key:
        try:
            om = model if model.startswith("gpt") else "gpt-4o-mini"
            OpenAIBrain(openai_key, om)
            log.info("Interviewer brain: openai (%s)", om)
            return BrainFactory("openai", om,
                                lambda: OpenAIBrain(openai_key, om),
                                first_sentence_timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAI brain unavailable: %s", exc)
    if backend == "auto":
        from .ollama import build_ollama

        if (probe := build_ollama(ollama_model, ollama_host)) is not None:
            resolved = probe.model
            log.info("Interviewer brain: ollama (%s) -- local and free", resolved)
            return BrainFactory(
                "ollama", resolved, lambda: type(probe)(resolved, ollama_host),
                first_sentence_timeout,
            )

    log.info("Interviewer brain: semantic (local concept graph)")
    return BrainFactory("semantic", "concept-graph")
