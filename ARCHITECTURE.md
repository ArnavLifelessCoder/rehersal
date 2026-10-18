# Architecture

How Rehearsal is built, why each load-bearing decision went the way it did, and where it deliberately departs from [`rehearsal_build_spec.md`](rehearsal_build_spec.md).

---

## 1. System shape

```
┌──────────────────────── BROWSER (React + Vite) ────────────────────────┐
│  Preflight device check ── must actually hear you before starting      │
│  getUserMedia (mic + cam)                                              │
│  ├─ Audio: AudioWorklet → 16kHz int16 PCM ──┐                          │
│  ├─ Video: MediaPipe Face+Pose in WASM      │  raw pixels stop here    │
│  │         head-pose-compensated gaze,      │                          │
│  │         per-person calibration           │                          │
│  │         └─ {eye_contact, gaze_offset,    │                          │
│  │             posture, blink, head_move,   │                          │
│  │             quality} @5Hz, ~950 B/s ─────┤                          │
│  ├─ Web Speech API playback (interviewer)   │                          │
│  ├─ Live readouts: fillers, pace, gaze, posture                        │
│  └─ Debug drawer (opt-in): VAD telemetry, event trace, timings         │
└─────────────────────────────────────────────┼──────────────────────────┘
                                    WebSocket │ binary = PCM, text = JSON
┌─────────────────────────────────────────────▼──────────────────────────┐
│                       BACKEND (FastAPI, asyncio)                       │
│                                                                        │
│  InterviewSession — ONE actor coroutine per interview                  │
│    ├─ SpectralVAD           harmonicity + spectral shape, not energy   │
│    ├─ Endpointer            adaptive silence: 1.2s + thinking graces   │
│    ├─ StreamingASR          partials while talking, beam-3 on flush    │
│    ├─ Phantom guard         turn-level energy floor before trusting    │
│    │                        any transcript                             │
│    ├─ Prosody (NumPy DSP)   F0 via FFT autocorrelation, pauses, energy │
│    ├─ CVAggregator          per-turn windowing of browser samples      │
│    ├─ Scoring               fillers · delivery · content               │
│    ├─ SessionState          PURE turn-taking policy (no I/O)           │
│    ├─ Brain (per session)   phrases the chosen move, streams sentences │
│    ├─ TTS                   per-sentence, pipelined                    │
│    └─ SessionTrace          always-on event ring buffer                │
│                                                                        │
│  Shared once per process: ASR · embedder · scorers · brain factory ·   │
│                           store · blob store                           │
│  Persistence: SQLAlchemy → SQLite (dev) or Postgres (prod)             │
│               WAV recordings on disk, age + size bounded               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The decisions that shape everything

### 2.1 CV runs in the browser

Inherited from the spec and the single most important choice in the system. MediaPipe's face-mesh and pose run in WASM at 30fps on a normal laptop, so the server never touches a video frame and never needs a GPU.

What makes the privacy claim *structural* rather than a promise: there is no server-side code path that accepts video. [`push_cv`](backend/app/session/manager.py) takes a dict of scalars. The claim holds by construction, not by policy.

### 2.2 The session is a single-writer actor

Three independent streams touch one session: audio frames (~50/s), CV samples (~5/s), and the outbound LLM→TTS pipeline. Guarding shared state with locks across `await` points invites deadlock and — worse — interleaved state transitions where the machine both advances the question *and* probes it.

So every input becomes a message on a queue, and exactly one coroutine mutates state.

VAD is the deliberate exception: it runs inline on the audio path because it's pure, non-mutating, and must not fall behind. Only the resulting *events* are queued.

Under pressure, audio frames are dropped oldest-first rather than buffered. A late frame is worthless — better to lose 20ms of audio than accumulate a backlog that pushes endpointing seconds behind reality. The same reasoning drives the client-side WebSocket backpressure in [`session.js`](frontend/src/lib/session.js).

### 2.3 Policy and phrasing are separate

The *decision* — probe, advance, challenge, interrupt, hint, move on — is made by [`SessionState.decide()`](backend/app/session/state.py) from measured signals. It's pure logic: no I/O, no asyncio, no model calls. The LLM only *phrases* the chosen move.

Three things fall out of this:

- The interview's behaviour is fully unit-testable without audio, a network, or a model.
- An LLM outage degrades the *wording*, never the *logic*. `ResilientBrain` times out the first sentence at 4s and falls through to the local brain — the candidate gets a duller question instead of ten seconds of silence.
- The offline mode is a real interviewer, not a canned script. See below.

### 2.3.1 The offline brain composes; it does not select

A question bank can only ask what it was written to ask, in an order fixed before the candidate spoke. That is exactly what it feels like. `SemanticBrain` reasons over a **concept graph** instead ([`concepts.py`](backend/app/interviewer/concepts.py)): every concept knows what a good answer pairs it with (`companions`), how to deepen it, and how to stress it.

Given an answer it (1) detects the concepts actually raised, (2) lifts a salient phrase from the candidate's own wording, (3) finds the **gap** — a companion a good answer would have included but they never mentioned — and (4) composes a question aimed at that gap, quoting them.

```
A: "I'd put a redis cache in front of the database so reads don't hit disk"
Q: "You'd put redis cache in front. What's your eviction policy, and what's
    the hit rate you're betting on?"

A: "we shard by user id across maybe ten nodes"
Q: "One shard is taking sixty percent of the traffic. How did that happen
    and what do you do?"

A: "we had a tight deadline and the whole team was under pressure"
Q: "You've told me what happened, but not what YOU did. What was your
    specific contribution?"          ← situation raised, action never was
```

None of those exist in advance. The last one is the clearest demonstration: it requires knowing that the candidate described a *situation*, that a complete behavioural answer pairs situation with *action*, and that they never gave one.

Details that make the difference between this and mad-libs:

- **Salient phrases are quoted only when they're worth quoting.** Known compounds first ("hash map", not "hash"), then modifier-noun bigrams within a single clause, then repeated content words. When nothing clears the bar, templates with no `{term}` slot are used — quoting the *wrong* phrase is far more damaging than quoting none. Bigrams formed across a comma produced "weeks migration"; requiring the technical word second killed "migration took".
- **Aliases match at word boundaries, and fillers are stripped first.** Unanchored substring search read `"it sort of depends"` as the *sorting* concept, so a content-free answer got a complexity probe.
- **Nothing repeats.** Templates and openers are consumed per session; when openers are exhausted it deepens a covered concept rather than re-asking. A repeat is the most obvious possible tell that nobody is listening.
- **The brain is per-session.** It carries covered-concept and asked-question state, so a process-wide instance would let concurrent interviews consume each other's question space. `BrainFactory` shares the expensive HTTP client and builds the stateful part per session.

It remains weaker than a good LLM — it can't understand a genuinely novel answer, and its phrasing space is bounded. It is the floor, not the ceiling. But it is a floor that reasons about content.

---

## 2.4 Speech detection is not energy detection

An energy gate cannot do this job, and shipping one was the original mistake. A door slam, a keyboard clack, a cough and a chair creak are all *louder* than conversational speech, so any threshold that hears you also hears them. In a turn-taking system a false positive isn't cosmetic — it hands the floor over and makes the interviewer respond to a noise.

`SpectralVAD` asks a different question: not "is this loud?" but "is this a human voice?"

| Signal | What it separates |
|---|---|
| **Harmonicity** (autocorrelation peak in the pitch range) | Voiced speech is periodic. Slams, clacks and rustles have no periodicity at all. Carries most of the weight. |
| **Spectral flatness** (Wiener entropy) | Speech energy sits in formants; transients are broadband. Catches unvoiced fricatives that harmonicity misses. |
| **Zero-crossing rate** | Separates voiced speech (low), fricatives (moderate), clicks and hiss (very high), mains hum (near zero). |
| **Band energy ratio** (80–4000Hz) | Rumble and hiss live outside the vocal range. |

Structure matters as much as weights. Harmonicity and flatness combine **additively** — either can carry a frame, since vowels are periodic while fricatives are merely peaky. Band ratio and ZCR then apply **multiplicatively, as gates**. The additive-only version failed on a fan: a 50Hz hum is perfectly periodic *and* perfectly peaky, so it collected most of the score and cleared the threshold. Almost none of its energy is in the speech band, and nothing that fails that test is a voice however tonal it is. A gate expresses that; a 13% additive term does not.

Measured separation (median voice score, synthetic but calibrated):

```
speech (f0 95-210Hz)  0.95 - 1.00     chair creak     0.40
                                      fan hum         0.17
                                      paper rustle    0.03
                                      door slam       0.00
                                      keyboard clack  0.00
```

Four further guards, each of which fixed an observed failure:

- **Onset needs periodicity, not just score.** Hysteresis makes a detector usable mid-utterance, but it cuts both ways: shaped noise that clipped the onset threshold on one loud frame would then coast for its whole duration on the lower sustain bar. A creak cleared onset on exactly 1 frame in 25 and was then "speech" for the other 24.
- **Onset must hold for 2 consecutive frames**, killing single-frame flukes. Real speech clears onset on essentially every frame, so this costs 20ms.
- **A relative level gate.** Once the candidate has been heard clearly, anything more than 22dB below them is something else. This is a crude proximity gate — the only lever available without speaker identification — and it's what stops a conversation in the next room taking the floor.
- **Hangover of 6 frames**, so stop consonants ("t", "k") — genuinely silent for 30–60ms mid-word — don't split one answer into two.

Then guards above the detector: the endpointer requires `speech_onset_ms` of sustained speech to open a turn and `min_utterance_ms` of accumulated speech before trailing silence can close one.

**The transcript can never prove someone spoke.** This is the important one, and getting it wrong produced the worst bug in the project. *Every* ASR invents words when handed silence — Whisper is notorious for emitting "Thank you." on a silent clip — and the `mock` backend was worse: it returned at least four words of fluent interview prose for *any* input, including pure silence. Those words cleared `min_answer_words`, became a scored answer, and the product showed people sentences they had never said, then criticised the filler words in them. It read as catastrophically bad speech recognition; it was fabrication.

So the audio is checked before the transcript is trusted:

- `_speech_rms` takes the **90th percentile** of per-frame energy, not the mean. A mean over a 12-second turn holding one real sentence is dominated by the surrounding silence and reads as noise; the percentile asks the question that matters — *was there ever speech in here?*
- Below `min_turn_rms` the turn is discarded whatever the ASR claims, and the claim is recorded in the trace as `phantom_discarded` so the fabrication is visible rather than silent.
- `MockASR` now returns empty on silence and scales its word count to **voiced** seconds rather than wall-clock. Scaling by wall-clock also quietly corrupted every rate metric (words-per-minute, fillers-per-minute) for anyone who pauses.

**Echo settling.** The first `floor_settle_ms` of the candidate's floor is dropped, so the tail of the interviewer's own voice — still leaving the speakers while the browser's echo canceller re-converges — cannot be transcribed as the start of the answer. It is counted in **samples, not wall-clock**: audio does not always arrive in real time, and a client draining a backlog after a network stall would otherwise have seconds of genuine speech discarded.

`tests/test_vad_noise.py` fires synthesised slams, clacks, creaks, rustles, hum and next-room speech at the detector and asserts none of them produce an endpoint, while real speech at three pitches does. The end-to-end smoke script does the same through a live WebSocket.

---

## 3. The turn-taking loop

```
LISTENING ──(VAD: speech end + 700ms silence)──► THINKING
THINKING  ──(score the answer, then decide)──►
    ├─ content ≥ 0.62                    ──► NEXT_QUESTION
    ├─ confident but incomplete          ──► CHALLENGE
    ├─ 0.30 ≤ content < 0.62, probes < 2 ──► PROBE
    ├─ rambling > 90s                    ──► INTERRUPT
    ├─ silent > 8s, first time           ──► HINT
    ├─ silent > 8s, already hinted       ──► MOVE_ON
    └─ question budget exhausted         ──► CLOSING
NEXT_QUESTION / PROBE / … ──(LLM → sentence stream → TTS)──► SPEAKING
SPEAKING ──(playback ack, or timeout)──► LISTENING
```

Details that matter more than they look:

**The silence window is adaptive, and 700ms was wrong.** 700ms is the right number for *conversation* and a bad one for an interview: people think mid-answer. They pause after "so…", they stop to work out a complexity bound, they restart a sentence. A fixed window forces one number to serve two completely different moments — the beat after a full stop, and the beat in the middle of a thought — and users experience the mismatch as being interrupted while thinking.

Base is now 1200ms, extended by two graces read off the partial transcript we already have:

| Signal | Extra wait | Why |
|---|---|---|
| Answer still under ~12 words | +1400ms | The pause right after a question is thinking time, not a finished answer. |
| Trails off on a conjunction or filler (`so`, `and`, `because`, `um`, `the`) | +1100ms | Someone who stops after "because" is mid-sentence. Cutting in there is the rudest thing an interviewer can do. |

They compose, so a short answer trailing mid-thought gets 3.7s. `stuck_silence_ms` was raised to 12s specifically so the "need a hint?" prompt can't fire while we're still deliberately being patient — a test asserts that ordering, because the two timers silently fighting is exactly the kind of bug that looks like random behaviour.

**Two probes per question, hard cap.** An interviewer that probes every answer is exhausting and stops carrying information. After two, it advances regardless.

**Confident-but-shallow is branched on explicitly.** High `correctness_signal` with low `completeness` averages out to "good enough", so a plain threshold advances past exactly the answer that most deserves a challenge. That case is checked *before* the average is taken.

**A short noise burst can't open a turn.** The endpointer needs 120ms of sustained speech onset, and an utterance below `min_utterance_ms` is discarded as noise rather than scored as an answer.

**The endpointer disarms after firing.** Otherwise the trailing silence — which is just the interviewer thinking and then talking — would later fire a spurious `stuck`, accusing the candidate of freezing on a question they had already answered. (This was a real bug; [`test_stuck_does_not_fire_after_the_user_spoke`](backend/tests/test_audio.py) pins it.)

**Playback acks have a timeout.** If the client never sends `playback_done` — backgrounded tab, missing TTS voice — a word-count-derived timeout reopens the floor. Without it the session wedges in SPEAKING forever.

---

## 4. Latency

Target: you stop talking → the interviewer starts speaking in **under 1.5s**.

| Segment | Budget | Measured as |
|---|---:|---|
| VAD endpoint confirmation | 1200ms+ | **deliberate wait**, not measured |
| Final ASR flush | 100ms | `asr_flush` |
| Scoring (prosody + fillers + content) | 200ms | `scoring` |
| LLM first sentence | 400ms | `llm_first_sentence` |
| TTS first audio out | 250ms | `first_audio_out` |
| **Server work** | **800ms** | the part we control |
| **Perceived** | **2000ms+** | what the candidate feels |

The spec set a 1.5s budget but gave no way to know whether it holds. A budget you don't measure is a wish, so [`TurnTimings`](backend/app/session/latency.py) times every stage of every turn, flags breaches, ships the breakdown to the debug drawer, and rolls up to p50/p95 in the report.

**Two honesty notes.**

Timing starts when the endpointer fires — *after* the silence wait. Measuring that against a budget that includes the wait would flatter every number by over a second of work we never did. So measurement runs against `post_endpoint_budget_ms` (800ms), and `perceived_ms` adds the silence back for the number the candidate actually feels. Both are reported.

And the perceived total deliberately **exceeded** the spec's 1.5s when the silence window was made adaptive (§3). That's a considered trade, not a regression: the spec's 700ms produced a system that interrupts people mid-thought, and 1.5s of never being cut off beats 1.5s of being cut off. The 800ms of *actual work* — the only part engineering effort can move — is unchanged and comfortably met.

Everything streams. The brain yields whole sentences (`_stream_sentences` regroups token chunks at sentence boundaries — TTS on a half-sentence produces the wrong prosody), each is synthesised and shipped immediately, and the candidate hears sentence one while the model is still writing sentence two.

Observed on the reference machine (i5 12th gen, no GPU, offline backends): **p50 60ms, p95 75ms** of server work against the 800ms budget. The LLM path is what consumes the rest in production.

---

## 5. Feature contract

[`backend/app/features.py`](backend/app/features.py) is the single definition of the delivery feature vector. Both the live pipeline and `training/dataset.py` import `FEATURE_ORDER` from it.

This exists because train/serve skew is the quiet killer of a fused-model ablation: if the training script and the product disagree about what column 7 means, the ablation measures the bug rather than the hypothesis, and nothing about the result is interpretable. Appending is allowed; reordering isn't; `SCHEMA_VERSION` bumps on change, and trained artefacts **refuse to load** against a mismatch rather than silently serving nonsense.

Missing modalities use **neutral defaults, not zeros**. Camera off should mean "I didn't measure your eye contact", not "you stared at nothing for the whole answer" — the latter is a confident wrong answer where an abstention was correct. `FeatureVector.coverage()` then feeds score confidence.

22 features across three modalities:

- **Audio (12)** — wpm, filler rate/ratio, pitch mean/variance, energy mean/variance, pause count/mean/max, speech ratio, restarts
- **Vision (7)** — eye contact %, gaze stability, posture score/drift, blink rate, head movement, longest freeze
- **Text meta (3)** — duration, word count, type-token ratio

---

## 6. Scoring

### Delivery

`HeuristicDeliveryScorer` is not a placeholder returning 0.5 — it's piecewise-linear scoring from published speech norms, and it's the ablation's honest baseline. The core primitive is a **trapezoid band**: 1.0 inside a healthy range, ramping to 0.0 at both extremes. That models "both extremes are worse", which is true of nearly every delivery feature and false of the linear scaling a naive scorer uses (which rewards talking infinitely fast).

`TrainedDeliveryScorer` swaps in LightGBM heads and reuses the heuristic's evidence generation — the trees supply the numbers, the bands supply the human-readable "why", and both read the same features.

Every score carries `evidence`: which features moved it, by how much, and a plain-English note, ranked by absolute contribution. The spec promised users could "see exactly which evidence drove each score"; that's only real if the scorer emits it, so it does.

### Fillers

Three detectors, because "um" is the easy 40%:

1. **Lexical** — single tokens and phrases (`you know`, `sort of`).
2. **Code-mixed** — `matlab`, `yaar`, `arre`, `bas`. The spec asked for these; most detectors are English-only and score Indian-English speakers as unnaturally fluent.
3. **Restarts** — immediate repetition and abandoned clauses. These read as hesitation to a listener but contain no filler word at all.

`like` and `so` are **context-gated**. "Something like a heap" isn't a filler; "it was like, um, hard" is. Flat keyword counting on `like` inflates the filler rate of anyone who speaks in analogies, which is most engineers.

### Content

Four axes: `relevance` (embedding or lexical similarity to the question), `structure` (discourse markers; STAR markers on the behavioural track), `completeness` (saturating coverage of track concept anchors — 6 of 16 anchors is a solid answer, not 37%), and `correctness_signal`.

That last name is load-bearing. We do not claim to grade DSA correctness from a transcript. It detects whether an answer *commits* — complexity claims, concrete mechanisms, worked examples — versus hedging. Overclaiming there is the fastest way to make the whole product untrustworthy.

The lexical fallback reports lower confidence **by construction** (0.55 vs 0.85). Hiding the difference would mask a failed model load behind numbers that look just as authoritative.

### Report

Three rules, tested in [`test_report.py`](backend/tests/test_report.py):

1. **Headline scores are confidence-weighted** across turns. A score we barely trust pulls less.
2. **Low-confidence caveats name their cause** — "you only spoke for 16 seconds in total", not a caveat you can't act on.
3. **Every worst moment cites its measurement**, with severity and a timestamp. No unfalsifiable vibes.

Strengths are surfaced alongside weaknesses — a failures-only report isn't more rigorous, it just makes people over-correct on whatever was listed first.

---

## 7. Deviations from the spec

Everything here is a departure, with the reasoning.

| # | Spec | Built | Why |
|---|---|---|---|
| 1 | librosa / torchaudio for prosody | Pure NumPy DSP | librosa pulls numba, torchaudio pulls torch — ~700MB for what is four DSP operations on 16kHz mono. F0 via FFT autocorrelation with a voicing gate is accurate enough for *variance* over an answer, which is all the model consumes. This is the difference between fitting the free tier and not. |
| 2 | Piper TTS on the server (60MB) | Browser Web Speech API by default | Saves 250ms/turn (no encode + network hop), 60MB of RAM, and downstream bandwidth — the thing free tiers throttle first. Piper stays available for deployments that need a consistent voice across devices, which is the one thing browser TTS is genuinely worse at. |
| 3 | pgvector for JD embeddings | JSON float array | The session flow only compares a JD against turns *within* one session. There's no ANN search to accelerate, so pgvector would be a hard dependency bought for nothing. Add it when cross-session JD search actually ships. |
| 4 | Models required to run | Every model is an adapter with an offline fallback | Not a convenience. It means **CI runs the real state machine**, a demo works on a plane, and a failed model load degrades one component instead of the product. It also makes `/api/health` a meaningful diagnostic. |
| 5 | Latency budget stated | Budget instrumented, breaches surfaced, p50/p95 in the report | See §4. Also fixed the accounting so it can't flatter itself by 700ms. |
| 6 | Scores | Scores **+ confidence + per-feature evidence** | The spec's §11 promises users can see the evidence behind each score. That's a code requirement, not a UI one. |
| 7 | Features → scores → DB | Feature **vectors** stored alongside scores | ~2KB/turn buys the ability to re-run a retrained scorer over all historical sessions instead of collecting fresh data. Cheapest possible insurance for the §7 ablation. |
| 8 | Feature vector defined per component | One `features.py`, imported by product *and* training | Prevents the train/serve skew that would make the ablation uninterpretable. See §5. |
| 9 | "Train the delivery scorer" | Grouped-by-speaker splits + ridge baseline + synthetic pipeline validation | Random row splits leak speaker identity and are how these papers overstate. A constant-predictor baseline can't tell you whether the model class earned its complexity. |
| 10 | — | Frontend code-split, and Recharts removed entirely | MediaPipe (~130KB) loads on demand. Recharts was ~400KB — more than twice the rest of the app — to draw two bar charts, and its defaults fought the typography; replaced with CSS bars and an inline SVG sparkline. Initial bundle 693KB → 173KB; the report chunk went 400KB → 9KB. |
| 11 | 700ms endpoint silence | **Adaptive** silence: 1.2s base + thinking graces | 700ms is a *conversation* number. Interview answers contain thinking pauses, and one fixed value cannot serve both "finished a sentence" and "paused mid-thought". Users experience the mismatch as being interrupted while thinking — the single most infuriating thing this product can do. See §3. |
| 12 | ASR output is the transcript | Transcript is **not trusted without audio energy** | Every ASR fabricates on silence (Whisper emits "Thank you."; the mock backend emitted fluent prose). Without an audio-level check the product shows people sentences they never said, then criticises the fillers in them. See §2.4. |
| 13 | Hosted LLM, or a weaker local one | Ollama as a first-class free path | The spec framed the interviewer as "pay cents or accept worse". A quantised 3B model on CPU is a real third option, auto-detected, costing nothing. See §10.1. |
| 14 | — | Always-on session trace + live telemetry channel | Turn-taking bugs are sequence bugs; ordinary logs can't answer "why did it respond *there*". The trace is always recorded because the session you need to diagnose is the one where nobody enabled debugging. See §10.1. |
| 15 | — | Preflight device check | Permissions, device selection and levels all fail silently. Discovering a dead microphone at the report, three answers in, is the worst failure mode available. |

---

## 8. Data model

Mirrors the spec's schema. `turns` additionally stores `features`, `delivery`, `content`, `fillers` and `timings` as JSON — scores are a lossy function of features, and keeping the inputs is what makes deviation #7 possible.

```
users(id, email, created_at)
sessions(id, user_id, track, jd_text, jd_embedding, started_at, ended_at, status, starred)
turns(id, session_id, idx, role, question_text, transcript, audio_url, duration_s,
      features, delivery, content, fillers, timings)          -- JSON columns
session_reports(session_id, overall_delivery, overall_content, confidence, payload)
```

SQLite by default, Postgres via the same SQLAlchemy layer — no schema divergence between dev and prod.

Writes are **best-effort by design**: a database hiccup must never take down a live interview. `Store` swallows and logs rather than raising into the actor loop. The session is the source of truth while it runs; the DB is where it goes to be read later.

Free-tier hygiene: feature rows are tiny, keep them forever. Prune raw audio blobs after 7 days unless the session is starred.

---

## 9. Frontend

**Audio** ([`audio.js`](frontend/src/lib/audio.js)) — AudioWorklet, not the deprecated `ScriptProcessorNode`. The worklet runs on the audio render thread, so React re-renders and MediaPipe inference can't cause dropouts in the frames endpointing depends on. Main-thread audio would be a permanent source of phantom latency.

We request a 16kHz `AudioContext` directly so the browser's own high-quality resampler usually does the work. When it won't, the worklet decimates with a **box filter** — averaging every input sample in each output bin. Nearest-neighbour picking would fold everything above 8kHz back into the speech band and put artefacts straight into the ASR's mel spectrogram.

`autoGainControl` is **off**: AGC would flatten exactly the energy variance the confidence sub-score reads. Echo cancellation is on, to stop the interviewer's own voice re-entering the mic.

**Vision** ([`vision.js`](frontend/src/lib/vision.js)) — two rates, deliberately. Face at 10Hz (gaze and blink are fast; under-sampling misses blinks entirely), pose at 3Hz (posture changes over seconds, and pose inference costs ~3× face). Running both at 30fps pegs the CPU and starves the audio thread, which surfaces as turn-taking latency.

Three things it gets right that a naive landmark reader does not:

**Gaze is head pose + eye-in-socket, not eye-in-socket alone.** Iris displacement measured against the eye corners tells you where the eyes point *within the skull*. Someone who turns their head 20° and keeps looking at you has strongly off-centre irises — and a naive detector marks them avoidant at the moment they're most engaged. We recover yaw/pitch/roll from MediaPipe's facial transformation matrix and combine them, so head rotation and eye rotation cancel to "on camera" when they should.

**Everything is calibrated per person.** Eye aspect ratio at rest varies enormously with eye shape, glasses and camera angle; a fixed blink threshold either misses every blink or counts three per blink. Neutral gaze varies with where the webcam sits — a camera below the display makes *everyone* look downward forever. We spend ~3s learning both (using the **median**, so a blink during calibration can't drag the baseline), and report `face_present: false` until calibrated so the first answer isn't scored against an uncalibrated origin. That's the answer people are most nervous on.

**"Not measured" is distinguished from "measured as zero".** A dropped frame, a face out of shot, a face held still, and a face too small for reliable iris landmarks are four different things. A `quality` signal rides along with each sample; below `MIN_USABLE_QUALITY` the server routes the turn through the same path as "camera off" — presence abstains to its neutral prior — rather than emitting numbers nobody believes.

Blink detection is **edge-triggered** with hysteresis on the eye-aspect-ratio crossing. Level-triggering counts one slow blink as three and inflates the rate ~3×.

Pose landmarks are checked for `visibility` before use: MediaPipe happily invents off-screen shoulders, and a hallucinated shoulder yields a confident, meaningless posture score.

**Freeze vs dropout** ([`cv_features.py`](backend/app/session/cv_features.py)) — "longest freeze" means the candidate went unnaturally still, a stress tell. A gap in sample timestamps is a *camera dropout* and breaks the run instead of extending it. Conflating them would penalise people for their wifi.

### 9.1 Visual design

The product is an interview. The screen should feel like a quiet room with one person in it, not a telemetry dashboard. Three commitments follow, and they're enforced in [`styles.css`](frontend/src/styles.css):

**Warm paper, not black glass.** A warm neutral ground (`#f6f4f0` / `#14130f`) reads as calm rather than clinical and keeps a single accent — a terracotta, not the default tech-blue — doing real work. Both themes are first-class: the palette is defined once and inverted, not bolted on.

**The question is the only hero.** It gets a serif face at display size and the full centre column, capped at 24 characters per line so it scans in one movement. Everything else — camera, metrics, timing — is peripheral by construction. The camera is a small rounded tile in the rail, roughly where a video-call PiP sits.

**Numbers earn their weight.** Tabular figures so a changing value doesn't reflow its neighbours. No gradients, no glow, no drop shadows on data. Colour appears only when a value is genuinely out of range — a metric that is always coloured communicates nothing.

The live overlay is deliberately restrained: feedback that moves sharply during an answer is feedback you have to consciously suppress, and that costs exactly the working memory you need to answer. Readouts smooth over ~3s with long CSS transitions, and only the filler counter carries real visual weight, because it's the one signal a person can act on mid-sentence. Input level is a genuine 22-bar waveform from the analyser rather than an abstract meter — it makes "the mic is working" legible at a glance without needing a label.

The report is laid out as an **article**, not a dashboard. Reading order is the argument: summary, headline numbers, what to fix, what worked, then the evidence. A grid of widgets would let someone skim to a number without the context that makes it fair.

Charts are hand-built CSS and a small inline SVG sparkline. Recharts was ~400KB — more than twice the rest of the application — to draw two bar charts, and its defaults fight the typography. Removing it took the report chunk from 400KB to 9KB.

`prefers-reduced-motion` disables every animation.

---

## 10. Testing

236 tests, ~27s, no network and no model downloads.

| File | Covers |
|---|---|
| `test_audio.py` | VAD floor adaptation, endpointing timing to the millisecond, F0 accuracy against known tones |
| `test_vad_noise.py` | Slams, clacks, creaks, rustle, hum and next-room speech vs. real speech at three pitches |
| `test_phantom_and_patience.py` | ASR fabrication on silence, turn-level energy floor, adaptive thinking graces, settle window |
| `test_semantic_brain.py` | Concept detection, gap-targeted composition, phrase extraction, no-repeat guarantees |
| `test_scoring.py` | Filler context-gating, delivery monotonicity, the abstention contract **across every backend**, content discrimination |
| `test_state_machine.py` | Every turn-taking branch, probe caps, CV aggregation, **termination as a property** |
| `test_report.py` | Confidence weighting, caveat causes, evidence citation |
| `test_features.py` | Blob round-trip/pruning/traversal, trend honesty, rate limiting, Ollama sanitising |
| `test_debug.py` | Trace ring buffer, VAD/endpointer telemetry, debug toggle + backfill, playback-timeout margin |
| `test_e2e.py` | Complete interviews through the real actor |
| `test_api.py` | REST surface, WebSocket handshake and protocol |

Three worth calling out:

- `test_session_terminates_within_budget` is a property test: no sequence of answer signals can make the interview run forever.
- `test_audio_during_interviewer_speech_is_ignored` pins the echo guard — the interviewer's own voice, played through the candidate's speakers, must never land in the transcript as an answer.
- `test_camera_dropout_is_not_a_freeze` pins the fairness property in §9.

Bugs caught by this suite during the build, all now regression-pinned:

- The spurious-`stuck` endpointer bug (§3).
- Single-character restarts (`"I I think"`) skipped by a length guard.
- The VAD calibrating its noise floor from the *first frame*, so a candidate who started talking immediately was never heard.
- A creak clearing onset on 1 frame in 25 and then coasting on the sustain threshold (§2.4).
- `"sort of"` matching the *sorting* concept via unanchored substring search (§2.3.1).
- Salient phrases formed across clause boundaries (`"weeks migration"`) and in verb order (`"migration took"`).
- The broken ridge baseline in the ablation — zeroing the intercept's diagonal entry instead of un-penalising it, which showed as an identical −35 R² across every feature set.
- The brain being shared process-wide while carrying per-session state.

Two reported by a user actually trying to use it, which no amount of synthetic testing had surfaced:

- **"it's inputting shit when I'm not speaking."** Not mis-hearing — fabricating. `MockASR` returned at least four words of fluent interview prose for *any* input including silence, those words cleared `min_answer_words`, and the product presented sentences the user had never said and then criticised the fillers in them. Fixed at three levels (§2.4), and the UI now labels simulated transcription as simulated instead of printing it under a "You" heading.
- **"gives less time to even think."** The 700ms window, inherited from the spec, is a conversation number applied to interview answers. Now adaptive (§3).

Both are cases where the synthetic tests passed happily because the test signals never *paused mid-thought* and never contained *silence the ASR would invent over*. Test fixtures inherit their author's assumptions; a user does not.

Three more found only by driving the real app in a browser, which is the point of doing that:

- **The playback-ack timeout fired on healthy sessions.** It was `words / 2.5` — but 2.5 words/sec *is* normal speaking rate, so the "generous upper bound" equalled the expected playback duration with zero headroom. Browser TTS voices run 1.8–3 wps, so slower ones tripped the safety net and it silently stole the floor mid-question. A safety net that fires during normal operation is worse than none. Now assumes a slow voice, then adds a multiplicative margin plus fixed slack.
- **The trained scorer ignored modality abstention.** `HeuristicDeliveryScorer` carefully abstains to a neutral prior when the camera is off; `TrainedDeliveryScorer` fed the neutral *defaults* into the trees and emitted a confident-looking `presence: 0.81`. The report then contradicted itself — "camera off, presence not scored" printed beside a presence score of 81. The abstention contract is a property of the product, not of one implementation, and testing it against only the heuristic is what let this ship. `TestAbstentionContract` now runs against every backend present on the machine.
- **A progress card contradicted itself.** It read `+1 better` beside `12 → 2`: the delta came from thirds-means, the footer from first-vs-last raw sessions. Two different bases inside one card. Both now share the thirds basis, and deltas below a per-metric noise floor report "no change" rather than dressing a 0.005 change as progress.

And one introduced *while* fixing the above, caught by the suite immediately: the echo-settle window first used wall-clock time, which is wrong in both directions — it discards nothing when a client bursts a backlog after a network stall, and everything when audio arrives faster than real time. It counts samples now.

One worth singling out because the *test* was wrong, not the code: the first noise-rejection run showed synthetic "speech" with a zero-crossing rate of 0.48 and harmonicity of 0.11 — indistinguishable from white noise, which is exactly what it was. The generator multiplied the phase *argument* by per-sample jitter, scrambling phase by tens of cycles per sample at high harmonics. A working detector looked broken. The fix was to integrate instantaneous frequency properly; measuring the features before touching a threshold is what surfaced it.

---

## 10.1 Later additions

**Recordings and playback.** Each candidate turn is written as a WAV via the stdlib `wave` module (no soundfile, no ffmpeg) and served at `/api/sessions/{id}/turns/{n}/audio`. The report's play button seeks to 1.5s *before* the cited moment — landing exactly on the filler gives no run-up and the moment is unrecognisable out of context. Retention is dual-bounded: age (7 days) *and* a total-size quota (512MB, oldest first), because age alone doesn't bound disk on a busy day and size alone would delete something recorded an hour ago. Starred sessions survive both. Writes happen after scoring and outside the timed section — disk I/O must not sit in the turn-latency path. Session ids arrive over the wire and become filesystem paths, so they're sanitised against traversal.

**Progress tracking.** `GET /api/progress` returns per-session delivery, content and filler rate, oldest first. The trend compares the first third to the last third and refuses below `MIN_SESSIONS_FOR_TREND`.

That constant is 6, not 4, and the correction is worth recording: with the window computed as `len // 3`, four or five sessions collapse to `max(1, 1)` — a one-versus-one comparison, which is exactly the first-versus-last delta the thirds split exists to avoid. Shipping it at four meant the honesty rule was defeated by its own arithmetic. Live data surfaced it ("Comparing your first 1 session to your most recent 1"), and `test_never_compares_a_single_session_to_a_single_session` now pins it across every count.

**Ollama brain.** A free local LLM path ([`ollama.py`](backend/app/interviewer/ollama.py)), preferred over the concept graph automatically when detected. Two things it needs that the hosted brains don't:

- An **output sanitiser**, kept as a pure function so its failure modes are testable as string transforms rather than prompt tweaks. Small instruct models emit "Great question!", markdown, role labels, and three stacked questions. Pleasantries are stripped as whole *sentences*: removing "Thanks for that answer." as a prefix leaves "For that answer.", which is worse than the input.
- **Whole-utterance buffering** before emission, because sanitising may drop a trailing stacked question. It still yields sentence-by-sentence so TTS pipelines normally; the cost is first-audio latency, which local generation dominates anyway.

The probe defaults to `127.0.0.1`, not `localhost`. On dual-stack hosts `localhost` resolves to `::1` first, and with nothing listening the connection waits for an IPv6 timeout before retrying IPv4 — ~3.5s on *every boot*, landing directly on free-tier cold start where 30s is already the budget. Fixing that plus a 1s probe timeout took test-suite startup from 3.7s to 1.2s.

**Preflight device check.** The worst failure this product has is a candidate answering three questions into a dead microphone and learning about it from the report. Permissions, device selection and levels all fail silently. The check requires *sustained* level (~0.6s, mirroring `speech_onset_ms`) rather than a single peak, so a desk bump doesn't pass the test that exists to catch a broken mic. The session is created only after it passes, so a failed check never leaves an abandoned session counting against the rate limit.

**Rate limiting.** A sliding window, in-process. Deliberately not Redis: a session pins a WebSocket and ~350MB of shared models to one worker, so the deployment is single-worker with sticky routing regardless — a distributed limiter would add an operational dependency to enforce a limit that only needs to hold for one process. Fixed windows would let someone burn a full hour's allowance in the last second of one window and again in the first second of the next. Empty keys are swept so unique client IPs can't leak memory indefinitely.

**Debug observability.** Debugging a turn-taking system from logs is miserable because the question is never "did an error happen" but "why did it respond *there*" — a sequence question. Three layers answer it:

- **An always-on per-session trace** ([`trace.py`](backend/app/session/trace.py)): a ring buffer of structured events — VAD events, decisions *with their inputs* (content score, word count, weakest axis, probes used), noise discards, latency breaches, playback timeouts. Always on because the session you need to diagnose is the one where nobody thought to enable debugging; bounded (600 events) so a 40-minute rambler can't grow memory. The tail (200 events) is persisted into the report, so "it cut me off yesterday" is diagnosable today.
- **An opt-in live channel**: the client sends `{"type": "debug", "enabled": true}` and gets an immediate backfill plus streamed trace events and ~4Hz VAD telemetry — voice score vs threshold, energy vs gate, harmonicity, and the endpointer's live countdown to the endpoint. Opt-in on the wire because the stream is pure overhead for the sessions where nobody opens the drawer.
- **`GET /api/sessions/{id}/events`**: the same trace over REST, live mid-session (curl-able while the bug is happening) or from the stored report afterwards.

The UI's "instruments" drawer renders the three panes that map to the three real questions: *why doesn't it hear me* (VAD strip), *why did it respond there* (event log), *why did it feel slow* (per-stage timings). Everything renders from server truth, not a client-side reconstruction — the point is to see disagreement between what you did and what the server perceived.

**Markdown export**, not JSON. The plausible things someone does with a report are paste it into notes, send it to a mentor, or diff it against next week's; Markdown survives all three and JSON survives none. The raw data stays available at the report endpoint.

---

## 11. Known limits

- **Single worker.** A session holds a live WebSocket plus ~350MB of shared models, so sticky routing is required regardless. Horizontal scaling needs a Redis-backed `SessionRegistry` and sticky sessions at the load balancer.
- **No speaker identification.** `SpectralVAD` rejects clearly distant speech via the relative-level gate, but it cannot tell *your* quiet voice from *someone else's* quiet voice at the same level. A person talking loudly next to you will be transcribed as you. Silero (`requirements-asr.txt`) improves detection but not this — genuine speaker separation needs diarisation, which is out of scope.
- **The concept graph is hand-authored and finite.** `SemanticBrain` handles the concepts it knows; a genuinely novel answer falls through to axis-level probes, which are noticeably more generic. This is the reason to configure an API key.
- **The lexical embedding fallback blurs paraphrase.** It reports lower confidence accordingly, but it will under-score a correct answer worded unusually.
- **Gaze calibration assumes the candidate looks at the screen during the first ~3 seconds.** Someone reading notes off to the side throughout calibration will have their neutral origin set there, and eye-contact scores for that session will be systematically generous.
- **No auth.** Rate limiting and a concurrency cap are in place, but there are no user accounts — `/api/progress` and `/api/sessions` show every session on the instance. Fine for single-user or local use; a shared deployment needs auth and a `user_id` filter before it's honest to call the progress screen "yours".
- **Recordings are local disk.** Fine for a single instance; a multi-instance deployment needs S3/R2 behind the same `BlobStore` interface.
- **The "hear it" timestamp is approximate for fillers.** It's derived from token position as a fraction of the answer, not from word-level ASR timings. Accurate enough to find the moment, not to land on the syllable. `faster-whisper` already returns word timestamps on the final flush — wiring those through is the fix.
- **`asr: mock` makes content scores meaningless.** The placeholder transcript is unrelated to the question, so relevance and completeness score near zero and the report reads as though the candidate rambled. Delivery scores are real (they come from the audio); content scores need real transcription. The UI labels the transcript as simulated, but the content half of the report is still noise until `requirements-asr.txt` is installed — arguably it should be suppressed outright rather than shown with a caveat.
- **Patience defaults are tuned for a deliberate speaker.** Someone naturally fast and fluent may find 1.2s + graces a beat slow. Every value is env-tunable (`REHEARSAL_ENDPOINT_SILENCE_MS`, `REHEARSAL_THINKING_GRACE_MS`, `REHEARSAL_OPENING_GRACE_MS`), but there is no per-user adaptation — learning an individual's pause distribution over a session is the obvious next step.
