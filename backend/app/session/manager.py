"""The session actor: one per live interview.

Concurrency model -- this is the part worth reading twice. Three independent
streams touch one session: inbound audio frames (~50/s), inbound CV samples
(~5/s), and the outbound interviewer pipeline (LLM -> TTS). Guarding shared
state with locks across await points invites deadlock and, worse, interleaved
state transitions where the machine both advances the question and probes it.

Instead: a single-writer actor. Every input becomes a message on a queue, and
exactly one coroutine mutates session state. Audio frames bypass the queue for
VAD (pure, non-mutating, and must not fall behind), and only the resulting
*events* are queued. The audio queue is bounded and drops oldest-first under
pressure, because a late frame is worthless -- far better to lose 20ms of audio
than to accumulate a backlog that pushes endpointing seconds behind reality.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from ..audio.asr import StreamingASR
from ..audio.prosody import extract_prosody
from ..audio.vad import Endpointer, build_vad
from ..config import Settings
from ..features import FeatureVector
from ..scoring.content import ContentResult
from ..scoring.delivery import DeliveryResult
from ..scoring.fillers import detect_fillers, type_token_ratio, words_per_minute
from .cv_features import CVAggregator, CVSample
from .latency import LatencyLog, TurnTimings
from .state import AnswerSignals, Move, SessionState, State
from .trace import SessionTrace

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]

# Words that mean "I haven't finished this thought". Someone who pauses after
# "so" or "because" is mid-sentence, and cutting in there is the rudest thing
# an interviewer can do.
_TRAILING_UNFINISHED = {
    "so", "and", "but", "or", "because", "since", "which", "that", "the", "a",
    "an", "to", "of", "in", "for", "with", "then", "if", "when", "like",
    "um", "uh", "erm", "hmm", "well", "maybe", "basically", "actually",
    "i", "we", "it", "there", "my", "its", "is", "was", "would", "could",
}


def _speech_rms(audio: np.ndarray, sample_rate: int = 16_000, floor: float = 0.004) -> float:
    """RMS of the audio's *loudest* portion, not its mean.

    A mean over a 12-second turn containing one real sentence is dominated by
    the silence around it and reads as noise. Taking a high percentile of the
    per-frame energy asks the question that matters: was there ever actually
    speech in here?
    """
    if audio.size == 0:
        return 0.0
    frame = max(int(sample_rate * 0.02), 1)
    n = audio.size // frame
    if n == 0:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return float(np.percentile(rms, 90))


@dataclass
class TurnRecord:
    idx: int
    role: str
    question_text: str | None = None
    transcript: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    duration_s: float = 0.0
    delivery: dict | None = None
    content: dict | None = None
    features: dict | None = None
    fillers: dict | None = None
    timings: dict | None = None
    audio_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "role": self.role,
            "question_text": self.question_text,
            "transcript": self.transcript,
            "duration_s": round(self.duration_s, 2),
            "delivery": self.delivery,
            "content": self.content,
            "fillers": self.fillers,
            "timings": self.timings,
        }


class InterviewSession:
    def __init__(
        self,
        *,
        session_id: str,
        track: str,
        jd_text: str | None,
        settings: Settings,
        deps,
        emit: Emit,
    ) -> None:
        self.id = session_id
        self.track = track
        self.jd_text = jd_text
        self.settings = settings
        self.deps = deps
        self.emit = emit

        self.state = SessionState(track=track, question_budget=settings.max_questions_per_session)
        self.turns: list[TurnRecord] = []
        self.latency = LatencyLog()
        self.started_at = time.time()
        self.ended_at: float | None = None

        self._vad = build_vad(settings.vad_backend, settings.sample_rate)
        self._endpointer = Endpointer(
            frame_ms=settings.frame_ms,
            silence_ms=settings.endpoint_silence_ms,
            stuck_ms=settings.stuck_silence_ms,
            ramble_ms=settings.ramble_limit_ms,
            min_utterance_ms=settings.min_utterance_ms,
            onset_ms=settings.speech_onset_ms,
        )
        self._asr = StreamingASR(deps.asr, settings.sample_rate)
        self._cv = CVAggregator()
        self._session_cv = CVAggregator()   # never reset: powers the report timeline

        # The brain is per-session: it tracks which concepts the candidate has
        # covered and which questions have been asked. A shared instance would
        # let concurrent interviews consume each other's question space.
        self.brain = deps.brain.build(track, jd_text)

        # Always-on event trace (cheap ring buffer) + an opt-in live debug
        # channel. Tracing unconditionally is deliberate: the session you need
        # to diagnose is the one where nobody thought to turn debugging on.
        self.trace = SessionTrace()
        self.debug_live = False
        self._last_vad_emit = 0.0

        self._events: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._runner: asyncio.Task | None = None
        self._speaking_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        self._turn_gen = 0   # bumped per listening turn; stale partials check it
        self._turn_start_ts: float | None = None
        self._settle_samples: int = 0
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._runner = asyncio.create_task(self._run(), name=f"session-{self.id}")
        await self._events.put(("open", {}))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (self._partial_task, self._speaking_task, self._runner):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _trace(self, kind: str, **detail) -> None:
        """Record a trace event; stream it to the client when debug is live."""
        event = self.trace.add(kind, **detail)
        if self.debug_live:
            await self.emit({"type": "trace", "event": event})

    async def set_debug(self, enabled: bool) -> None:
        self.debug_live = enabled
        if enabled:
            # Backfill so the drawer opens with history, not an empty pane.
            await self.emit({
                "type": "trace_backfill",
                "events": self.trace.events(limit=60),
                "counts": self.trace.counts(),
            })

    # ------------------------------------------------------------------ ingest

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Called per inbound audio frame. Must stay cheap and non-blocking."""
        if self._closed or self.state.state == State.ENDED:
            return

        # Live VAD telemetry at ~4Hz while debugging. This is the answer to
        # "why doesn't it hear me": the voice score, the energy gate and the
        # endpointer's counters, as they move. Emitted BEFORE the LISTENING
        # gate so the instruments drawer stays alive while the interviewer is
        # speaking too (it shows the last-known snapshot during that stretch).
        if self.debug_live:
            now = time.time()
            if now - self._last_vad_emit >= 0.25:
                self._last_vad_emit = now
                await self.emit({
                    "type": "vad",
                    "vad": self._vad.telemetry(),
                    "endpointer": self._endpointer.snapshot(),
                    "state": self.state.state.value,
                })

        # Only the candidate's floor matters here. While the interviewer is
        # speaking (or the next move is being decided) the mic is hearing the
        # interviewer's own voice through the candidate's speakers, and that
        # echo must reach NEITHER the VAD nor the ASR buffer.
        #
        # The VAD's noise floor and learned speech level deliberately persist
        # across turns -- the room and the speaker don't change between
        # questions -- which is exactly why feeding it the interviewer's voice
        # is dangerous: a loud echo would push the energy gate up and silently
        # deafen the next answer. Barge-in is not implemented, so running VAD
        # while speaking has no upside, only that risk.
        if self.state.state != State.LISTENING:
            return

        # Drop the settling window -- the tail of the interviewer's voice, and
        # the echo canceller re-converging, must not latch the VAD or land in
        # the transcript as the start of the answer.
        if self._settle_samples > 0:
            drop = min(self._settle_samples, pcm.size)
            self._settle_samples -= drop
            pcm = pcm[drop:]
            if pcm.size == 0:
                return
        frame_len = int(self.settings.sample_rate * self.settings.frame_ms / 1000)

        for i in range(0, len(pcm), frame_len):
            frame = pcm[i : i + frame_len]
            if frame.size < frame_len // 2:
                break
            self._asr.feed(frame)
            event = self._endpointer.push(self._vad.is_speech(frame))
            if event:
                await self._events.put((event.kind, {"speech_ms": event.speech_ms}))

        # Partials run off to the side. Awaiting one here stalled this whole
        # path for a full decode of the answer-so-far (~1s with Whisper on CPU)
        # -- the WebSocket reader waits on us, so frames backed up, and the
        # endpointer only saw the candidate stop talking once the backlog
        # drained. Single-flight: one partial at a time, the rest are skipped.
        if self._partial_task is None or self._partial_task.done():
            self._partial_task = asyncio.create_task(self._run_partial(self._turn_gen))

    async def _run_partial(self, gen: int) -> None:
        partial = await self._asr.maybe_partial()
        # The turn may have ended while we decoded; a late partial must not
        # overwrite the final transcript or retune the next turn's patience.
        if not partial or not partial.text:
            return
        if gen != self._turn_gen or self.state.state != State.LISTENING:
            return
        self._adjust_patience(partial.text)
        await self.emit({"type": "transcript", "text": partial.text, "final": False})

    def _adjust_patience(self, partial_text: str) -> None:
        """Wait longer when the answer sounds unfinished.

        A fixed silence window forces one number to serve two very different
        moments: the beat after a full stop, and the beat in the middle of a
        thought. They are not the same, and treating them the same is what makes
        an interviewer feel like it's interrupting.

        Two signals, both cheap and both from the partial transcript we already
        have: the answer is still very short (the pause right after a question
        is thinking time), or it trails off on a conjunction or filler.
        """
        s = self.settings
        words = partial_text.split()
        patience = s.endpoint_silence_ms

        if len(words) < s.opening_grace_words:
            patience += s.opening_grace_ms

        if words:
            last = words[-1].lower().strip(".,;:!?'\"")
            if last in _TRAILING_UNFINISHED:
                patience += s.thinking_grace_ms

        if patience != self._endpointer.silence_ms:
            self._endpointer.silence_ms = patience

    async def push_cv(self, payload: dict) -> None:
        if self._closed:
            return
        sample = CVSample(
            ts_ms=int(payload.get("ts_ms", 0)),
            eye_contact=bool(payload.get("eye_contact", False)),
            gaze_offset=float(payload.get("gaze_offset", 0.0)),
            posture=float(payload.get("posture", 0.7)),
            blink=bool(payload.get("blink", False)),
            head_movement=float(payload.get("head_movement", 0.0)),
            face_present=bool(payload.get("face_present", True)),
            quality=float(payload.get("quality", 1.0)),
        )
        self._cv.add(sample)
        self._session_cv.add(sample)

    async def request_end(self) -> None:
        await self._events.put(("end", {}))

    # ------------------------------------------------------------------ actor loop

    async def _run(self) -> None:
        try:
            while not self._closed:
                kind, payload = await self._events.get()
                try:
                    await self._handle(kind, payload)
                except Exception:  # noqa: BLE001
                    log.exception("session %s: error handling %s", self.id, kind)
                    await self.emit(
                        {"type": "error", "message": f"internal error handling {kind}"}
                    )
        except asyncio.CancelledError:
            raise

    async def _handle(self, kind: str, payload: dict) -> None:
        if kind in ("speech_start", "endpoint", "stuck", "ramble"):
            await self._trace(f"vad_{kind}", **payload)
        if kind == "open":
            await self._speak(Move.OPENING)
        elif kind == "speech_start":
            await self.emit({"type": "state", "state": "listening", "speaking": True})
        elif kind == "endpoint":
            await self._on_endpoint()
        elif kind == "stuck":
            await self._on_stuck()
        elif kind == "ramble":
            await self._on_ramble()
        elif kind == "speech_done":
            await self._begin_listening()
        elif kind == "end":
            await self._on_end()

    # ------------------------------------------------------------------ turn logic

    async def _begin_listening(self) -> None:
        self.state.state = State.LISTENING
        self._turn_gen += 1
        self._endpointer.reset()
        self._endpointer.silence_ms = self.settings.endpoint_silence_ms
        self._vad.reset()
        self._asr.reset()
        self._cv.reset()
        self._turn_start_ts = time.time()
        # Ignore the first few hundred ms of the candidate's floor. The
        # interviewer's audio is still ringing out of the speakers, and the
        # browser's echo canceller needs a moment to re-converge after the
        # far-end signal stops -- without this, the tail of the question gets
        # transcribed as the start of the answer.
        #
        # Counted in SAMPLES, not wall-clock. Audio does not always arrive in
        # real time: a client draining a backlog after a network stall delivers
        # seconds of audio in milliseconds, and a wall-clock window would throw
        # away real speech instead of the intended echo tail.
        self._settle_samples = int(
            self.settings.sample_rate * self.settings.floor_settle_ms / 1000
        )
        await self.emit({"type": "state", "state": "listening"})

    async def _on_endpoint(self) -> None:
        timings = TurnTimings()
        self.state.state = State.THINKING
        await self.emit({"type": "state", "state": "thinking"})

        s = self.settings
        with timings.stage("asr_flush", s.budget_asr_flush_ms):
            transcript = await self._asr.flush()
        audio = self._asr.audio()
        duration_s = max(len(audio) / s.sample_rate, 0.1)

        # Second line of defence against phantom turns.
        #
        # The transcript can never prove someone spoke: every ASR invents words
        # when handed silence (Whisper famously emits "Thank you." on a silent
        # clip, and the mock backend used to emit a fluent sentence). So the
        # audio itself is checked first -- if the turn never reached speech
        # level, no transcript is trustworthy, however confident it sounds.
        speech_rms = _speech_rms(audio)
        if speech_rms < self.settings.min_turn_rms:
            log.info(
                "session %s: discarding %.1fs turn -- no speech energy (rms %.5f < %.5f), "
                "ASR claimed %r",
                self.id, duration_s, speech_rms, self.settings.min_turn_rms,
                transcript.text[:60],
            )
            await self._trace(
                "phantom_discarded",
                reason="no_speech_energy",
                rms=round(speech_rms, 5),
                claimed=transcript.text[:60],
            )
            await self.emit({"type": "noise_discarded", "duration_s": round(duration_s, 2)})
            await self._begin_listening()
            return

        words = transcript.text.split()
        if len(words) < self.settings.min_answer_words:
            log.info(
                "session %s: discarding %.1fs turn as noise (%d words: %r)",
                self.id, duration_s, len(words), transcript.text[:60],
            )
            await self._trace(
                "noise_discarded",
                duration_s=round(duration_s, 2),
                words=len(words),
                text=transcript.text[:60],
            )
            await self.emit({"type": "noise_discarded", "duration_s": round(duration_s, 2)})
            await self._begin_listening()
            return

        await self.emit({"type": "transcript", "text": transcript.text, "final": True})

        with timings.stage("scoring", s.budget_content_check_ms * 4):
            record = self._score_turn(transcript.text, audio, duration_s)
        record.timings = timings.stages

        # Persist the recording so the report's worst-moment timestamps have
        # something to play. Deliberately AFTER scoring and outside the timed
        # section: disk I/O must not sit in the turn-latency path.
        if s.store_audio and self.deps.blobs is not None:
            record.audio_path = await asyncio.to_thread(
                self.deps.blobs.save_turn, self.id, record.idx, audio
            )

        await self.emit(
            {
                "type": "turn_scored",
                "turn": record.to_dict(),
            }
        )

        signals = AnswerSignals(
            content_overall=record.content["overall"],
            relevance=record.content["relevance"],
            completeness=record.content["completeness"],
            structure=record.content["structure"],
            correctness_signal=record.content["correctness_signal"],
            word_count=int(record.features.get("word_count", 0)),
            duration_s=duration_s,
            confidence=record.content["confidence"],
        )
        move = self.state.decide(signals)
        # The single most valuable trace event: the decision AND its inputs.
        # Every "why did it probe/advance there" question resolves to this row.
        await self._trace(
            "decision",
            move=move.value,
            content=round(signals.content_overall, 3),
            relevance=round(signals.relevance, 3),
            completeness=round(signals.completeness, 3),
            structure=round(signals.structure, 3),
            specificity=round(signals.correctness_signal, 3),
            words=signals.word_count,
            weakest=signals.weakest_axis(),
            probes_used=self.state.probes_on_current,
            question_n=self.state.questions_asked,
        )
        await self._speak(move, signals=signals, timings=timings, answer=transcript.text)

    async def _on_stuck(self) -> None:
        if self.state.state != State.LISTENING:
            return
        signals = AnswerSignals(0, 0, 0, 0, 0, 0, 0.0, was_silent=True)
        move = self.state.decide(signals)
        await self._speak(move, signals=signals)

    async def _on_ramble(self) -> None:
        if self.state.state != State.LISTENING:
            return
        transcript = await self._asr.flush()
        signals = AnswerSignals(0.5, 0.5, 0.4, 0.3, 0.4, 200, 90.0, was_rambling=True)
        await self._speak(Move.INTERRUPT, signals=signals, answer=transcript.text)

    def _score_turn(self, text: str, audio: np.ndarray, duration_s: float) -> TurnRecord:
        prosody = extract_prosody(audio, self.settings.sample_rate)
        fillers = detect_fillers(text, duration_s)
        cv_feats = self._cv.features()

        partial: dict[str, float] = {
            **prosody.to_dict(),
            "filler_per_min": fillers.per_min,
            "filler_ratio": fillers.ratio,
            "restart_count": float(fillers.restart_count),
            "words_per_min": words_per_minute(text, duration_s),
            "word_count": float(len(text.split())),
            "type_token_ratio": type_token_ratio(text),
            "answer_duration_s": duration_s,
            **cv_feats,
        }
        fv = FeatureVector.from_partial(
            partial, has_audio=audio.size > 0, has_vision=bool(cv_feats)
        )

        delivery: DeliveryResult = self.deps.delivery_scorer.score(fv)
        content: ContentResult = self.deps.content_scorer.score(
            self.state.current_question or "", text, self.track, self.jd_text
        )

        record = TurnRecord(
            idx=self.state.turn_index,
            role="candidate",
            transcript=text,
            question_text=self.state.current_question,
            duration_s=duration_s,
            delivery=delivery.to_dict(),
            content=content.to_dict(),
            features=fv.to_dict(),
            fillers=fillers.to_dict(),
            ended_at=time.time(),
        )
        self.state.turn_index += 1
        self.turns.append(record)
        return record

    # ------------------------------------------------------------------ speaking

    async def _speak(
        self,
        move: Move,
        *,
        signals: AnswerSignals | None = None,
        timings: TurnTimings | None = None,
        answer: str | None = None,
    ) -> None:
        self.state.apply(move)
        await self.emit({"type": "state", "state": "speaking", "move": move.value})

        history = [t.to_dict() for t in self.turns]
        stream = self.brain.ask(
            move=move.value,
            track=self.track,
            question=self.state.current_question,
            answer=answer,
            history=history,
            jd_text=self.jd_text,
            weakest_axis=signals.weakest_axis() if signals else None,
            questions_asked=self.state.questions_asked,
            question_budget=self.state.question_budget,
        )

        spoken: list[str] = []
        first = True
        s = self.settings
        async for sentence in stream:
            if first and timings is not None:
                timings.mark("llm_first_sentence")
                if timings.stages["llm_first_sentence"] > s.budget_llm_first_token_ms:
                    timings.breaches.append(
                        f"llm_first_sentence: {timings.stages['llm_first_sentence']:.0f}ms "
                        f"> {s.budget_llm_first_token_ms}ms budget"
                    )
            # Synthesise and ship each sentence as it lands: the candidate hears
            # sentence one while the model is still writing sentence two.
            chunk = await self.deps.tts.synthesise(sentence)
            if first and timings is not None:
                timings.mark("first_audio_out")
                first = False
            spoken.append(sentence)
            await self.emit({"type": "speech", "chunk": chunk.to_dict()})

        text = " ".join(spoken)
        if self.state.is_new_question(move):
            self.state.current_question = text

        self.turns.append(
            TurnRecord(
                idx=self.state.turn_index, role="interviewer", question_text=text,
                ended_at=time.time(),
            )
        )
        self.state.turn_index += 1

        if timings is not None:
            # Measured against the post-endpoint budget, since the clock starts
            # after the silence wait. `perceived_ms` adds it back so the client
            # can show the number the candidate actually felt.
            payload = timings.to_dict(s.post_endpoint_budget_ms)
            payload["perceived_ms"] = round(
                payload["total_ms"] + s.endpoint_silence_ms, 1
            )
            payload["perceived_budget_ms"] = s.total_response_budget_ms
            if payload["breaches"]:
                await self._trace("latency_breach", breaches=payload["breaches"])
            self.latency.add(payload)
            await self.emit({"type": "latency", "timings": payload})

        await self._trace("spoke", move=move.value, text=text[:90], sentences=len(spoken))
        await self.emit({"type": "speech_end", "text": text, "move": move.value})

        if move == Move.CLOSING:
            await self._on_end()
        else:
            # The client tells us when playback finished; if it never does
            # (tab backgrounded, TTS voice missing), this timeout reopens the
            # floor so the session cannot wedge in SPEAKING forever.
            self._speaking_task = asyncio.create_task(self._await_playback(text))

    async def _await_playback(self, text: str) -> None:
        # Safety net for a client that never acks playback -- NOT a schedule.
        #
        # This was originally words/2.5, which is normal *speaking* rate, so the
        # timeout equalled the expected playback duration with zero headroom and
        # fired on perfectly healthy sessions (browser TTS voices vary from ~1.8
        # to ~3 wps). A safety net that trips during normal operation is worse
        # than none: it silently steals the floor mid-question.
        #
        # So: assume a slow voice, then add a multiplicative margin and fixed
        # slack. A genuinely broken client still recovers in ~15s.
        expected_s = len(text.split()) / 2.0
        timeout = min(max(expected_s * 1.5 + 3.0, 5.0), 45.0)
        try:
            await asyncio.sleep(timeout)
            if self.state.state == State.SPEAKING:
                log.info("session %s: playback ack timed out, opening floor", self.id)
                await self._trace("playback_timeout", waited_s=round(timeout, 1))
                await self._events.put(("speech_done", {}))
        except asyncio.CancelledError:
            pass

    async def notify_playback_done(self) -> None:
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()
        if self.state.state == State.SPEAKING:
            await self._events.put(("speech_done", {}))

    # ------------------------------------------------------------------ end

    async def _on_end(self) -> None:
        if self.ended_at:
            return
        self.ended_at = time.time()
        self.state.state = State.ENDED
        from ..scoring.report import build_report

        report = build_report(self)
        await self.emit({"type": "report", "report": report})
        await self.emit({"type": "state", "state": "ended"})
        self.deps.store.save_session(self, report)


class SessionRegistry:
    """In-process registry. Swap for Redis-backed lookup when running >1 worker.

    Single-worker is not a limitation to apologise for here: a session holds a
    live WebSocket and ~350MB of shared models, so sticky routing is required
    regardless of where the registry lives.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, InterviewSession] = {}

    def create(self, **kwargs) -> InterviewSession:
        sid = kwargs.pop("session_id", None) or uuid.uuid4().hex[:12]
        session = InterviewSession(session_id=sid, **kwargs)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> InterviewSession | None:
        return self._sessions.get(session_id)

    async def drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    def active_count(self) -> int:
        return len(self._sessions)
