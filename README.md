# Rehearsal

A multimodal mock-interview simulator. It runs a live interview, listens and watches, adapts its follow-ups to your answers, and afterwards scores both **what you said** and **how you said it** — filler rate, pace, pitch variance, eye contact, posture, and whether you actually answered the question.

The demo line it's built to earn:

> *"You said 'um' 34 times, your pace hit 190 wpm when nervous, and you broke eye contact for the entire hard question."*

Built from [`rehearsal_build_spec.md`](rehearsal_build_spec.md). For the design rationale and where this deviates from that spec, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

Nothing below needs an API key, a GPU, or a model download. Every heavy dependency is an adapter with a working offline fallback, so the full turn-taking loop, scoring stack and report run out of the box.

**Backend**

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

**Frontend** (separate terminal)

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, pick a track, allow mic (and camera, if you want delivery feedback), and talk.

Check what's actually loaded at any time:

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "backends": {
    "asr": "mock", "brain": "semantic", "tts": "browser",
    "delivery_scorer": "heuristic", "embedder": "lexical",
    "database": "sqlite", "recordings": "on"
  },
  "active_sessions": 0,
  "latency_budget_ms": 1500
}
```

That means you're running fully offline. The full interview loop, scoring stack and report all work — but note `"asr": "mock"`: **transcription is simulated placeholder text, not your words.** The UI labels it as such. Delivery scoring (pace, pauses, volume, fillers-in-the-placeholder) still runs on your real audio. Upgrade whichever pieces you care about:

---

## Turning on the real models

Each install is independent — take them in any order. The app detects what's importable at startup and logs which backend it chose.

| Want | Install | Effect |
|---|---|---|
| **Real transcription** | `pip install -r requirements-asr.txt` | `mock` → `faster-whisper`. Downloads `tiny.en` (~75MB) on first use — fast enough on CPU that the transcript appears live while you talk. Prefer accuracy over speed (e.g. Indian-accented English)? Set `REHEARSAL_ASR_MODEL=base.en`; it costs ~7x the decode latency. |
| **A real LLM, free** | [install Ollama](https://ollama.com), then `ollama pull qwen2.5:3b-instruct` | `semantic` → `ollama`. Runs on CPU, costs nothing, detected automatically. |
| **Sharper questions still** | `pip install -r requirements-llm.txt` + set a key | → `anthropic`/`openai`. Best quality, costs cents per session. |
| **Semantic content scoring** | `pip install -r requirements-scoring.txt` | `lexical` → `bge-small-en-v1.5`. Catches paraphrase; raises score confidence. |
| **Trained delivery scorer** | see [Training](#training-the-delivery-scorer) | `heuristic` → `lightgbm`. |
| **Postgres** | `pip install -r requirements-pg.txt` | Set `REHEARSAL_DATABASE_URL`. |

Configuration is env vars with a `REHEARSAL_` prefix, or a `backend/.env` file:

```bash
REHEARSAL_ANTHROPIC_API_KEY=sk-ant-...
REHEARSAL_BRAIN_MODEL=claude-haiku-4-5-20251001
REHEARSAL_ENDPOINT_SILENCE_MS=1200      # base pause before it replies (adapts upward)
REHEARSAL_MAX_QUESTIONS_PER_SESSION=12
REHEARSAL_DATABASE_URL=postgresql+psycopg://user:pass@host/rehearsal
```

Every setting lives in [`backend/app/config.py`](backend/app/config.py).

---

### Free is a first-class configuration

Nothing in the default path costs money. If you want a genuine language model without paying per session, Ollama covers it:

```bash
ollama pull qwen2.5:3b-instruct      # ~2GB, runs on CPU
```

That's it — the backend probes `127.0.0.1:11434` at startup and prefers it over the concept graph automatically. Quality sits between the concept graph and a hosted frontier model: it handles novel answers the graph can't, at the cost of needing firmer constraints to stay in character (there's an output sanitiser for exactly that — small instruct models love "Great question!" and stacking three questions into one turn).

If Ollama isn't running, nothing breaks; you get the concept graph.

---

## What it does, turn by turn

0. **A device check runs first.** You have to actually be heard before the interview starts — the worst failure this product can have is answering three questions into a dead microphone and finding out at the report. It uses the same sustained-level logic the server does, so "we can hear you" means the same thing it will mean during the interview.
1. Browser captures mic + camera. Audio streams up as 16kHz int16 PCM; **video is analysed locally** and only ~1KB/sec of numeric features goes to the server — about 0.3% of what the raw video would cost.
2. The server transcribes continuously and watches for end-of-utterance (VAD + an adaptive trailing-silence window that lengthens when you're mid-thought).
3. When you stop, it scores the answer — prosody, fillers, gaze, posture, relevance, structure, completeness — and a state machine decides its next move: **advance, probe, challenge, interrupt, hint,** or **move on**.
4. The interviewer's reply streams sentence-by-sentence straight into speech, so you hear sentence one while sentence two is still being written.
5. On end, you get a report: scores with confidence, the worst moments with the measurement behind each one **and a play button to hear the exact moment**, a per-answer timeline, and the full transcript. Download it as Markdown, or track it against past sessions.

### It waits while you think

The silence window adapts instead of being a fixed 700ms — that's a conversation number, and answers aren't conversation. It waits 1.2s by default, longer when your answer is still short (the pause after a question is thinking time) and longer again when you trail off mid-thought on "so…", "because…", "um…". Pause to work out a complexity bound and it will let you.

### It won't type words you didn't say

If speech recognition isn't installed the transcript is **simulated placeholder text**, and the UI now says so plainly rather than presenting invented sentences under a "You" heading.

Separately, no backend's transcript is trusted on its own: a turn whose audio never reached speech level is discarded whatever the ASR claims it heard. This matters for real Whisper too — it hallucinates "Thank you." on silent clips.

### It won't answer a door slam

Speech detection is spectral, not energy-based: it asks whether a sound is *periodic and shaped like a voice*, not whether it's loud. A cough, a keyboard clack, a chair creak and a fan are all louder than conversational speech, and an energy gate hears all of them — which in a turn-taking system means the interviewer responds to a noise.

Four independent layers, all of which have to be fooled before a noise becomes a turn:

1. **Harmonicity and spectral gating** in the VAD — is this periodic and voice-shaped?
2. **Sustained-onset and minimum-utterance** requirements in the endpointer — a single burst can't open or close a turn.
3. **A turn-level energy check** — did the audio ever actually reach speech level?
4. **A word-count check** on the transcript.

`tests/test_vad_noise.py` fires the whole catalogue at it — door slam, keyboard clack, chair creak, paper rustle, fan hum, next-room conversation — and asserts none of them take the floor while real speech at three pitches still does. Details in [ARCHITECTURE.md §2.4](ARCHITECTURE.md).

### Questions are composed, not selected

Without an API key the interviewer runs on a local concept graph. It detects what you actually raised, finds what a good answer would have paired with it and you didn't, and builds a question around the gap — quoting you:

```
You:  "I'd put a redis cache in front of the database so reads don't hit disk"
Them: "You'd put redis cache in front. What's your eviction policy, and
       what's the hit rate you're betting on?"

You:  "we had a tight deadline and the whole team was under pressure"
Them: "You've told me what happened, but not what YOU did. What was your
       specific contribution?"
```

None of those are stored questions. With an API key configured a hosted model takes over and handles genuinely novel answers better; the local graph stays as the fallback when the API is slow or down.

### Hear the moment, not just the number

Each answer is recorded to disk and served back at the cited offset, so "you said 'um' nine times around 12s in" comes with a **play** button that starts 1.5s before the moment. A citation you can't check is just an assertion, and this product judges people — the evidence has to be reachable.

Recordings are pruned after 7 days (and past a 512MB quota, oldest first) unless you star the session. `DELETE /api/sessions/{id}` removes a session and its audio outright.

### Am I actually getting better?

`GET /api/progress` and the **See past sessions** screen chart delivery, content and filler rate across sessions.

The trend refuses to report below **6 sessions**, and compares the first third to the last third rather than first-to-last. Six is deliberate: the window is a third of your sessions, so at four or five it collapses to a one-vs-one comparison — precisely the noise the thirds-based split exists to avoid. (I shipped it as four first; live data showed the collapse.)

### When it does something odd, you can see why

Hit **instruments** during an interview for three live panes, all rendered from server truth rather than a client-side guess:

- **Hearing** — the voice score against its threshold, energy against the gate, harmonicity, and the endpointer's live countdown. This is the answer to "why isn't it hearing me".
- **Events** — decisions *with the inputs behind them* (`probe — content 0.42 · words 31 · weakest: completeness`), VAD events, noise discards.
- **Last turn** — per-stage timings against budget.

Every session keeps this trace whether or not you're looking, because the session you need to diagnose is the one where nobody thought to turn debugging on. The tail is saved into the report, and `GET /api/sessions/{id}/events` serves it — live mid-interview (curl-able while the bug is happening) or from the report afterwards.

### The privacy claim, precisely

Raw video never reaches the server. MediaPipe runs in WASM in your browser ([`frontend/src/lib/vision.js`](frontend/src/lib/vision.js)), and the only thing transmitted is a 5Hz JSON sample:

```json
{"type":"cv","ts_ms":12345,"face_present":true,"eye_contact":true,
 "gaze_offset":0.18,"posture":0.81,"blink":false,"head_movement":0.06,"quality":0.92}
```

That's it — 189 bytes, five times a second. There is no server-side code path that can receive a video frame; [`push_cv`](backend/app/session/manager.py) takes a dict of scalars, so the claim holds by construction rather than by policy. Audio *is* sent, because transcription needs it.

Gaze is computed from head pose **and** eye-in-socket, not eye position alone — turning your head while still looking at the camera should not read as avoidance. Blink threshold and neutral gaze are calibrated to you over the first few seconds, which is why the first moments show "Calibrating"; a fixed threshold either misses every blink or triple-counts them, and a webcam below your screen otherwise makes everyone look permanently downward.

---

## Testing

```bash
cd backend
pip install pytest pytest-asyncio httpx
python -m pytest -q            # 275 tests, ~50s
```

The suite is not just unit tests. [`tests/test_e2e.py`](backend/tests/test_e2e.py) drives complete interviews through the real session actor — real VAD, real endpointing, real feature extraction, real state machine, real report builder — with only ASR and the LLM swapped for deterministic offline backends. That's what makes the offline fallbacks load-bearing rather than decorative: CI exercises the actual interview logic on every run.

Against a **running** server, over the real wire protocol:

```bash
python scripts/smoke_interview.py --answers 3
```

```
  backends: {'asr': 'mock', 'brain': 'semantic', 'delivery_scorer': 'lightgbm', ...}
  [opening] Thanks for making the time. How would you find the first non-repeating…
noise check (slam + clacks)…
  ok: noise ignored, floor retained
answer 1…
  you said: I think the time complexity would be O of…
  scored -> delivery 0.72 content 0.15 fillers 0
  latency OK  70ms / 800ms
  [probe] Let's stress it: what does your approach do on an empty input…
  ...
  PASS
```

It fires a door slam and two keyboard clacks at the live server before the first answer and fails if either takes the floor.

The default answers are synthetic tones, which only the mock ASR turns into words. With faster-whisper installed, Whisper correctly hears no speech in them, so give it a real recording instead (16kHz mono 16-bit WAV):

```bash
python scripts/smoke_interview.py --answer-wav answer.wav
```

Both suites run with no network and no model downloads, so they work on a plane.

---

## Training the delivery scorer

The trainable model is the research artefact. It maps the fused feature vector to four delivery sub-scores (confidence, clarity, pace, presence).

Validate the whole pipeline on synthetic data first — this proves fit → ablate → export → load → serve is correct before you spend a week collecting labels:

```bash
pip install lightgbm
python training/train_delivery.py --source synthetic --n 3000 --export
```

```
ABLATION -- mean R^2 across the four delivery sub-scores (held-out speakers)
feature set            n     ridge   lightgbm    delta
text_meta_only         3     0.003      0.037   +0.034
audio_only            12     0.189      0.348   +0.159
vision_only            7     0.054     -0.075   -0.130
audio_plus_vision     19     0.246      0.448   +0.202
fused                 22     0.250      0.467   +0.217
```

**Synthetic numbers are not a result.** They confirm the machinery works and that the trees beat a linear baseline. The real run uses your labelled data:

```bash
curl http://localhost:8000/api/training-data > rows.json
# attach human ratings as a `labels` object on each row, then:
python training/train_delivery.py --source sessions --data rows.json --export
```

Restart the backend and it logs `Delivery scorer: lightgbm`.

Two methodological choices worth defending in a review:

- **Splits are grouped by speaker.** Two answers from the same person share accent, room acoustics, camera angle and baseline pitch. A random row split leaks all of that and produces a model that scores beautifully and generalises to nobody. This is the most common way delivery papers overstate their results.
- **The baseline is ridge regression on the same features**, not a constant predictor. If gradient boosting can't beat a linear model, "we used gradient boosting" isn't a finding.

The ablation is built so that **either outcome is publishable**. `fused − audio_only` is the contested line: whether gaze and posture add anything *on top of* prosody. A clean "no" there is a real negative result that pushes back on a lot of assumed wisdom, and the script is designed to be able to say it.

---

## Project layout

```
backend/
  app/
    main.py              FastAPI: REST + the WebSocket protocol
    config.py            every tunable, env-overridable
    features.py          THE feature contract — imported by product and training
    deps.py              process-wide model instances (loaded once)
    ratelimit.py         sliding-window limiter
    audio/
      vad.py             spectral VAD + adaptive endpointer
      asr.py             streaming ASR (faster-whisper | mock)
      prosody.py         F0, energy, pauses — pure NumPy DSP
    scoring/
      fillers.py         incl. code-mixed + context-gated detection
      delivery.py        heuristic and LightGBM heads
      content.py         relevance · structure · completeness · specificity
      report.py          the end screen, and its honesty rules
    interviewer/
      semantic.py        composes questions from the concept graph
      concepts.py        the graph itself
      brain.py           backend selection + per-session factory
      ollama.py          free local LLM
      prompts.py · tts.py
    session/
      state.py           PURE turn-taking policy (no I/O)
      manager.py         the actor
      cv_features.py     per-turn windowing of browser samples
      latency.py         per-stage turn timing
      trace.py           the event trace behind the debug drawer
    storage/
      models.py · store.py
      blobs.py           recordings + retention
  tests/                 275 tests incl. full end-to-end interviews
  scripts/               smoke_interview.py — real WebSocket, real PCM, real noise
frontend/
  src/lib/               audio.js (AudioWorklet) · vision.js (MediaPipe)
                         tts.js · session.js
  src/components/        Setup · Preflight · Interview · Readouts
                         DebugDrawer · Report · Progress
  src/styles.css         the design system
training/
  dataset.py             imports FEATURE_ORDER from the serving code
  train_delivery.py      ablation + export
```

---

## Deployment

| Piece | Where | Note |
|---|---|---|
| Frontend | Vercel / Render Static | Static build, always on |
| Backend | Render Web Service / Fly.io | Needs sticky sessions — a live WebSocket plus ~350MB of shared models |
| Interviewer LLM | Hosted API | Pennies per session; the best value in the stack |
| Database | Supabase free | Preferred over Render Postgres — no 90-day expiry |

Two honest warnings:

- Render's free tier **sleeps after 15 minutes** and cold-starts in ~30s while loading ASR. Fine for a portfolio demo, not for real users. The first $7/mo removes it.
- Keeping the interviewer local to stay entirely free makes follow-ups blunter on unusual answers. The state machine still probes and challenges correctly — the *decisions* are ours, not the model's — and the concept graph still aims at real gaps, but its phrasing space is finite.

---

## What this is careful about

It judges people, so it tries to be honest about what it knows:

- **Every score ships with a confidence**, derived from modality coverage and answer length. No camera means presence abstains to a neutral prior rather than scoring zero — the difference between "I didn't measure this" and "you failed at this".
- **Low-confidence caveats name their cause.** "You only spoke for 16 seconds in total", not a vague shrug you can't act on.
- **Every worst moment cites its measurement**, with a timestamp, so you can check whether you agree.
- **`correctness_signal` is named a signal, not a verdict.** It detects whether an answer commits to complexity claims, tradeoffs and concrete mechanisms. It does not claim to grade whether your algorithm is right.
- **Strengths are surfaced alongside weaknesses.** A report that only lists failures isn't more rigorous, it's just worse feedback.
- **Every cited moment is playable.** You can check whether you agree with the machine.
- **Trends refuse to appear before there's enough data** to mean anything, and changes below a noise floor report "no change" rather than being dressed up as progress.
- **It never shows you words you didn't say.** No transcript is trusted without the audio energy to back it, and simulated transcription is labelled as simulated.
- **It waits while you think.** Being cut off mid-sentence by something that then scores your "incomplete" answer is the worst thing this product could do.
- **You can delete a session and its recordings.** The product records your face and voice while judging you; removing that is a baseline, not a feature.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Which backends are live, latency budget |
| `POST /api/sessions` | Start one (rate limited, capacity capped) |
| `GET /api/sessions` | List |
| `GET /api/sessions/{id}/report` | Full report JSON |
| `GET /api/sessions/{id}/turns/{n}/audio` | The recording behind a citation |
| `GET /api/sessions/{id}/events` | The session's event trace (live or from the report) |
| `POST /api/sessions/{id}/star` | Keep recordings past retention |
| `DELETE /api/sessions/{id}` | Remove session + audio |
| `GET /api/progress` | Cross-session trend |
| `GET /api/training-data` | Feature vectors for retraining |
| `WS /ws/{id}` | The live interview |
