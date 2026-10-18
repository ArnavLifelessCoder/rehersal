"""Streaming ASR.

FasterWhisperASR is the production path. MockASR keeps the entire turn-taking
loop, scorer stack and report generator exercisable with no model download --
which matters more than it sounds: it means CI runs the real state machine, and
a demo works on a plane.

Both are wrapped in an async worker so transcription never blocks the event
loop that is also servicing the WebSocket.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    is_final: bool
    confidence: float = 0.0
    words: list[dict] | None = None


class BaseASR:
    name = "base"

    async def transcribe(self, audio: np.ndarray, *, is_final: bool) -> Transcript:
        raise NotImplementedError  # pragma: no cover


class MockASR(BaseASR):
    """Deterministic stand-in.

    Emits plausible interview prose whose length scales with the audio duration
    and whose content is seeded by an audio hash, so the same utterance always
    yields the same transcript. That determinism is what makes the scoring tests
    meaningful rather than flaky.

    It must, however, stay silent on silence. The original version returned at
    least four words for *any* input, so a room-noise blip produced a fluent
    sentence the candidate never said -- and, being >= min_answer_words, it
    sailed past the noise guard and became a scored answer. A stand-in that
    invents speech isn't a stand-in, it's a hallucination generator, and it made
    the product look like it was mis-hearing people rather than making them up.
    """

    name = "mock"

    # Below this RMS there is nothing to transcribe, and saying so is the whole
    # difference between a mock and a fabricator.
    SILENCE_RMS = 0.004

    _CORPUS = [
        "so um basically the approach I would take here is to start with a hash map",
        "and then you know iterate over the array once keeping a running sum",
        "uh let me think about the edge cases for a second",
        "I think the time complexity would be O of n and space is also O of n",
        "in my previous role I actually ran into a very similar problem",
        "we had a service that was timing out under load so I profiled it",
        "and it turned out the bottleneck was an N plus one query in the ORM",
        "like the main tradeoff is between consistency and availability here",
        "matlab we could shard by user id but that makes cross user joins painful",
        "so I would probably go with eventual consistency and a reconciliation job",
    ]

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    def _voiced_seconds(self, audio: np.ndarray) -> float:
        """Seconds of audio actually above the noise floor."""
        frame = max(int(self.sample_rate * 0.02), 1)
        n = audio.size // frame
        if n == 0:
            return 0.0
        frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        return float(np.count_nonzero(rms > self.SILENCE_RMS) * frame / self.sample_rate)

    async def transcribe(self, audio: np.ndarray, *, is_final: bool) -> Transcript:
        await asyncio.sleep(0.005)  # fast decode stand-in
        if audio.size == 0:
            return Transcript("", is_final, 0.0)

        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms < self.SILENCE_RMS:
            return Transcript("", is_final, 0.0)

        # Scale words to the *voiced* portion, not wall-clock duration.
        voiced_s = self._voiced_seconds(audio)
        if voiced_s < 0.25:
            return Transcript("", is_final, 0.0)

        seed = int(hashlib.sha256(np.ascontiguousarray(audio[:4096]).tobytes()).hexdigest()[:8], 16)
        # ~2.3 words/sec of speech is a normal interview pace.
        target_words = max(1, int(voiced_s * 2.3))
        rng = np.random.default_rng(seed)
        picked: list[str] = []
        count = 0
        while count < target_words:
            line = self._CORPUS[int(rng.integers(len(self._CORPUS)))]
            picked.append(line)
            count += len(line.split())
        text = " ".join(picked)
        words = text.split()[:target_words]
        return Transcript(" ".join(words), is_final, confidence=0.55)


class FasterWhisperASR(BaseASR):
    name = "faster-whisper"

    def __init__(self, model: str = "base.en", sample_rate: int = 16_000) -> None:
        import os
        from faster_whisper import WhisperModel

        threads = max(1, min(os.cpu_count() or 4, 8))
        self._model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        self.sample_rate = sample_rate

    async def transcribe(self, audio: np.ndarray, *, is_final: bool) -> Transcript:
        return await asyncio.to_thread(self._transcribe_sync, audio, is_final)

    def _transcribe_sync(self, audio: np.ndarray, is_final: bool) -> Transcript:
        if audio.size == 0:
            return Transcript("", is_final, 0.0)

        beam_size = 1 if not is_final else 2
        vad_filter = False if not is_final else True
        vad_params = dict(min_silence_duration_ms=300) if is_final else None

        # No word timestamps: nothing downstream reads them, and the alignment
        # pass cost ~0.7s on every final flush -- squarely in the turn latency.
        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=beam_size,
            best_of=1 if not is_final else 2,
            temperature=0.0,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            without_timestamps=True,
            condition_on_previous_text=False,
            max_new_tokens=_token_cap(audio.size / self.sample_rate),
        )
        parts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            parts.append(seg.text.strip())
            logprobs.append(seg.avg_logprob)
        conf = float(np.exp(np.mean(logprobs))) if logprobs else 0.0
        text = collapse_repetition(" ".join(parts).strip())
        return Transcript(text, is_final, conf)


def _token_cap(duration_s: float) -> int:
    """Most tokens a human could have produced in this much audio.

    Fast speech is ~4 words/s, ~6 tokens/s with punctuation. Whisper decodes
    in 30s windows, so the cap is per window. Without it, a repetition loop
    ("whoa, whoa, whoa...") runs to the 224-token limit -- seconds of CPU
    spent generating words nobody said, on the turn-latency path.
    """
    return int(min(duration_s, 30.0) * 8) + 16


_LOOP_MIN_REPEATS = 4


def collapse_repetition(text: str, min_repeats: int = _LOOP_MIN_REPEATS) -> str:
    """Collapse decoder loops: an n-gram (n<=4) repeated back-to-back
    `min_repeats`+ times is reduced to one occurrence.

    Whisper's classic failure on non-speech or degraded audio is to lock into
    a loop and emit the same word or phrase dozens of times. Left alone, that
    becomes a scored "answer" at 500+ words/min. People do repeat themselves
    ("no, no, no"), so the bar is four consecutive repeats -- rare in speech,
    universal in decoder loops. Collapsed, a pure loop is a word or two and
    falls under the minimum-answer-words guard like any other noise.
    """
    words = text.split()
    if len(words) < min_repeats:
        return text
    norm = [w.lower().strip(".,;:!?'\"-") for w in words]
    out: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for n in range(1, 5):
            gram = norm[i : i + n]
            if len(gram) < n:
                break
            reps = 1
            while norm[i + reps * n : i + (reps + 1) * n] == gram:
                reps += 1
            if reps >= min_repeats:
                out.extend(words[i : i + n])
                i += reps * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def build_asr(backend: str, model: str, sample_rate: int) -> BaseASR:
    if backend in ("auto", "faster-whisper"):
        try:
            asr = FasterWhisperASR(model, sample_rate)
            log.info("ASR backend: faster-whisper (%s)", model)
            return asr
        except Exception as exc:  # noqa: BLE001
            if backend == "faster-whisper":
                raise
            log.warning("ASR backend: mock (faster-whisper unavailable: %s)", exc)
    return MockASR(sample_rate)


class StreamingASR:
    """Chunked partial decoding with a single-flight guard.

    Partials fire on a fast cadence while the user talks. `_busy` drops
    overlapping partial requests instead of queueing them.

    The final flush reuses the latest partial when that partial already heard
    everything the candidate said. By the time the endpointer fires, a full
    second or more of silence has passed -- several partial cycles -- so a
    partial covering the end of speech almost always exists, and decoding the
    same audio again just to say the same words was the single largest cost
    in the turn: ~1s of Whisper on CPU against a 100ms budget. Only when no
    partial covers the end of speech does the flush decode from scratch.
    """

    # Audio past the last voiced frame a partial must have heard before its
    # text can stand in for the final: soft word endings fall below the VAD.
    COVER_MARGIN_S = 0.3

    def __init__(
        self,
        asr: BaseASR,
        sample_rate: int,
        partial_interval_s: float = 0.35,
        reuse_partial: bool = True,
    ) -> None:
        self._asr = asr
        self._sr = sample_rate
        self._partial_interval = partial_interval_s
        self._reuse_partial = reuse_partial
        self._buf: list[np.ndarray] = []
        self._samples = 0
        self._last_partial_at = 0.0
        self._busy = False
        self._gen = 0                       # bumped on reset; stale partials check it
        self._speech_end = 0                # samples fed up to the last voiced frame
        self._partial: Transcript | None = None
        self._partial_covers = 0            # samples the stored partial decoded
        self._inflight_covers = 0           # samples the running partial is decoding
        self._partial_done = asyncio.Event()
        self._partial_done.set()
        self.reused_partial = False         # whether the last flush skipped decoding

    @property
    def backend(self) -> str:
        return self._asr.name

    def feed(self, frame: np.ndarray, *, is_speech: bool = False) -> None:
        self._buf.append(frame)
        self._samples += frame.size
        if is_speech:
            self._speech_end = self._samples

    def duration_s(self) -> float:
        return self._samples / self._sr

    def audio(self) -> np.ndarray:
        return np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buf.clear()
        self._samples = 0
        self._last_partial_at = 0.0
        self._gen += 1
        self._speech_end = 0
        self._partial = None
        self._partial_covers = 0

    async def maybe_partial(self) -> Transcript | None:
        now = self.duration_s()
        if self._busy or now - self._last_partial_at < self._partial_interval or self._samples == 0:
            return None
        self._last_partial_at = now
        self._busy = True
        gen, audio = self._gen, self.audio()
        self._inflight_covers = audio.size
        self._partial_done.clear()
        try:
            result = await self._asr.transcribe(audio, is_final=False)
            if gen == self._gen:
                self._partial, self._partial_covers = result, audio.size
            return result
        finally:
            self._busy = False
            self._partial_done.set()

    def _covers_speech(self, n_samples: int) -> bool:
        return self._speech_end > 0 and n_samples >= self._speech_end + int(
            self.COVER_MARGIN_S * self._sr
        )

    async def flush(self) -> Transcript:
        self.reused_partial = False
        if self._samples == 0:
            return Transcript("", True, 0.0)
        if self._reuse_partial:
            # A finished partial that heard the end of speech wins outright --
            # one still running would only add more trailing silence. Failing
            # that, a running partial that will cover it finishes sooner than
            # a fresh decode would, and says the same thing.
            if not (self._partial is not None and self._covers_speech(self._partial_covers)):
                if self._busy and self._covers_speech(self._inflight_covers):
                    await self._partial_done.wait()
            if self._partial is not None and self._covers_speech(self._partial_covers):
                self.reused_partial = True
                p = self._partial
                return Transcript(p.text, True, p.confidence)
        return await self._asr.transcribe(self.audio(), is_final=True)
