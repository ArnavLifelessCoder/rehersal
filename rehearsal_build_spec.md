# Rehearsal — Production Build Spec

A multimodal mock-interview simulator. It runs a live interview, listens and watches, adapts its follow-up questions to your answers, and afterward scores both content and delivery: what you said, and how you said it (filler rate, pace, eye contact, confidence, whether you actually answered).

This document is the complete build: architecture, per-stage models with sizes, data flow, schema, latency budget, the deployment topology on free/cheap tiers, the paper design, and a week-by-week roadmap.

Target hardware and tiers assumed throughout: local dev on i5 12th gen (no GPU), training on free Colab/Kaggle T4, deploy on Render free tier + a browser that does the heavy CV client-side. Where the free tier genuinely can't hold, the doc says so and gives the paid fallback, because pretending 512MB runs everything is how these projects die in week 8.

---

## 1. What it does, end to end

1. User picks a track (DSA, system design, behavioral, or a pasted JD) and starts a session in the browser.
2. Browser captures webcam + mic. Audio streams to the backend; video is analyzed **in the browser** (privacy + zero server GPU) with only lightweight numeric features sent up.
3. The backend transcribes speech in near-real-time, and when the user pauses, decides: probe deeper, move on, or challenge.
4. An interviewer persona asks the next question (text → TTS, played in the browser).
5. Throughout, a delivery scorer accumulates metrics (fillers/min, words/min, pitch variance, eye-contact %, posture, longest freeze).
6. On "end", a report is generated: content scores per answer, delivery timeline, and clip timestamps of the worst moments so the user can rewatch exactly where they said "um" nine times.

The uncomfortable-but-useful demo line: *"You said 'um' 34 times, your pace hit 190 wpm when nervous, and you broke eye contact for the entire hard question."*

---

## 2. Architecture

```
┌────────────────────────── BROWSER (React) ──────────────────────────┐
│  WebRTC capture (mic + cam)                                          │
│  ├─ Audio: 16kHz PCM chunks ──► WebSocket ──► backend ASR            │
│  ├─ Video: analyzed LOCALLY (MediaPipe Face/Pose in WASM)            │
│  │        └─ emits {eye_contact, posture, blink, headpose} @ 5Hz    │
│  │           ──► WebSocket (tiny JSON, ~200 B/s) ──► backend         │
│  ├─ TTS audio playback (interviewer voice)                          │
│  └─ Live overlay: current filler count, pace meter, timer           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ WebSocket (bidirectional)
┌──────────────────────────────▼──────────────────────────────────────┐
│                        BACKEND (FastAPI, async)                      │
│  Session Manager (one state machine per interview)                   │
│  ├─ ASR worker: streaming whisper.cpp (tiny/base, ONNX/GGUF)         │
│  ├─ Turn-taking controller: VAD + end-of-utterance detection        │
│  ├─ Interviewer brain: LLM (hosted API or local 3B) — question gen   │
│  ├─ Delivery scorer: fuses audio prosody + browser CV features       │
│  ├─ Content scorer: answer-vs-question relevance + rubric            │
│  └─ Report builder (on session end)                                 │
│                                                                      │
│  Persistence: Postgres (sessions, turns, scores, JD embeddings)     │
│  Cache/pubsub: Redis (live session state, or in-proc fallback)      │
│  Object store: S3-compatible (recorded audio/video for review)      │
└──────────────────────────────────────────────────────────────────────┘
```

Key decision: **CV runs in the browser**, not the server. MediaPipe's face-mesh and pose run in WASM at 30fps on a normal laptop, so the server never touches raw video and never needs a GPU. This is the single most important architectural choice for making it deployable and private.

---

## 3. Models, per stage, with sizes and why

| Stage | Model | Size / cost | Where it runs | Why this one |
|---|---|---|---|---|
| **ASR (streaming)** | whisper.cpp `base.en` (or `tiny.en` for speed) | ~74MB / ~140MB | Backend, CPU | GGUF quantized, streams on CPU with AVX2; `base.en` handles Indian-accented English far better than `tiny`. Swap to `small` only if you get a GPU. |
| **Voice Activity + endpointing** | Silero VAD | ~2MB ONNX | Backend, CPU | Detects when the user stopped talking so the interviewer can respond. Sub-ms. |
| **Face / eye-contact / posture** | MediaPipe FaceMesh + Pose | WASM, no download to server | **Browser** | Gaze direction from iris landmarks, posture from pose. Zero server load. |
| **Prosody features** | librosa / torchaudio DSP (no model) | n/a | Backend, CPU | Pitch (F0), energy, speaking rate, pause stats. Pure signal processing. |
| **Filler detection** | rule + phonetic match on ASR tokens, backed by a tiny classifier | <5MB | Backend, CPU | "um/uh/like/you know/matlab/basically" incl. code-mixed fillers; classifier catches disfluent restarts. |
| **Interviewer brain (question gen)** | Hosted API (GPT-4o-mini / Claude Haiku) *or* local Qwen2.5-3B-Instruct GGUF | API cost ~cents/session, or ~2GB local | API in prod, local optional | Needs strong instruction-following for adaptive follow-ups. Local 3B is the free-tier fallback but weaker at probing. Budget the API; it's the one place paying pennies is worth it. |
| **Content scorer** | `bge-small-en-v1.5` embeddings + a LightGBM rubric head | ~130MB + <1MB | Backend, CPU | Answer-vs-question relevance via embedding similarity; LightGBM maps features → rubric scores (structure, completeness, correctness signal). |
| **Delivery scorer** | Your trained model (see §7) — gradient-boosted on fused features | <5MB | Backend, CPU | The paper artifact. Takes prosody + CV feature vector → delivery sub-scores. |
| **TTS (interviewer voice)** | Piper (local) or a cheap hosted TTS | ~60MB local | Backend or browser | Piper is fast on CPU and offline; hosted TTS sounds better for the demo. Streamable. |

Total resident backend footprint with local models: ASR base (~150MB) + VAD (2MB) + embeddings (130MB) + scorers (10MB) + Piper (60MB) ≈ **~350MB**, before the interviewer LLM. That's why the interviewer LLM should be a hosted API in production: it's the one component that blows the 512MB budget if kept local.

---

## 4. The real-time turn-taking loop

This is the hard part; most "AI interview" projects skip it and just play canned questions. The state machine per session:

```
LISTENING ──(VAD: speech end + 700ms silence)──► THINKING
THINKING  ──(content check: did they answer?)──►
    ├─ answered fully      ──► NEXT_QUESTION
    ├─ answered partially  ──► PROBE (ask for the missing piece)
    ├─ rambling >90s       ──► INTERRUPT (politely cut, redirect)
    └─ stuck/silent >8s    ──► HINT or MOVE_ON
NEXT_QUESTION / PROBE ──(LLM gen → TTS)──► SPEAKING
SPEAKING ──(playback done)──► LISTENING
```

Endpointing tuning matters: 700ms of trailing silence before responding feels natural; less and you cut people off, more and it feels laggy. Make it configurable and measure it.

The interviewer prompt carries: the track rubric, the question asked, the running transcript of this answer, and an instruction to either probe the weakest part or advance. Keep a per-session question budget so it doesn't run forever.

---

## 5. Latency budget (the thing that makes or breaks "real-time")

Target: user stops talking → interviewer starts speaking in **under 1.5s**.

| Segment | Budget |
|---|---|
| VAD endpoint confirmation | 700ms (the deliberate silence wait) |
| Final ASR flush | 100ms |
| Content check (embedding + rule) | 50ms |
| Interviewer LLM first token (hosted, streamed) | 400ms |
| TTS first audio chunk (streamed) | 250ms |
| **Total to first interviewer sound** | **~1.5s** |

Everything streams. You do not wait for the full LLM answer before starting TTS; you sentence-chunk and pipeline. This pipelining is itself a legitimate systems talking point.

---

## 6. Data model (Postgres)

```sql
users(id, email, created_at)

sessions(
  id, user_id, track,            -- 'dsa' | 'system_design' | 'behavioral' | 'jd'
  jd_text, jd_embedding,          -- pgvector, nullable
  started_at, ended_at, status
)

turns(
  id, session_id, idx,
  role,                           -- 'interviewer' | 'candidate'
  question_text,                  -- for interviewer turns
  transcript,                     -- for candidate turns
  audio_url,                      -- S3 key, candidate turns
  started_at, ended_at
)

turn_delivery(                    -- one row per candidate turn
  turn_id,
  fillers_count, words_per_min, pitch_var,
  eye_contact_pct, posture_score, longest_freeze_ms,
  delivery_subscores jsonb        -- {confidence, clarity, pace, presence}
)

turn_content(                     -- one row per candidate turn
  turn_id,
  relevance, structure, completeness, correctness_signal,
  rubric_notes jsonb
)

session_report(
  session_id,
  overall_delivery, overall_content,
  worst_moments jsonb,            -- [{turn_id, ts_ms, reason}]
  summary_text
)
```

Free-tier data hygiene (Render Postgres 1GB, 90-day expiry): store feature rows forever (tiny), but **prune raw audio/video after 7 days** unless the user stars a session. Keep a `pg_dump` in the repo for the 90-day recreate.

---

## 7. The trainable model + the paper

The delivery scorer is where the research lives, and it's outcome-symmetric.

**Question:** do multimodal delivery features (audio prosody + visual gaze/posture) predict human-perceived interview performance better than transcript content alone?

**Data:** public mock-interview and public-speaking video (YouTube mock interviews, TED-style talks with audience ratings, the MIT Interview dataset if accessible). Extract the same feature vector your product computes. Labels: use available performance ratings, or collect lightweight human ratings on clips (a small annotation effort = a cleaner dataset).

**Model:** gradient-boosted trees on the fused feature vector → delivery score. Ablate: content-only vs delivery-only vs fused.

**Either outcome is a paper:**
- Fused beats content-only → "how you say it measurably matters, here's how much" (the interesting positive).
- Fused doesn't help → "delivery signals are weak predictors of rated performance" (a real negative that pushes back on a lot of assumed wisdom).

**Optional user study** (stronger, HCI/education track): does real-time feedback actually improve users across sessions? Measure filler rate over N sessions. That's a small but genuine longitudinal result.

**Venue ladder:** multimodal / affective-computing workshops (ICMI, ACII) → the dataset as a resource paper → a main workshop if the user study lands.

---

## 8. Deployment topology

```
Static frontend (React)        → Render Static Site or Vercel free (always on)
Backend (FastAPI + ASR + scorers) → Render Web Service
Interviewer LLM                → hosted API (OpenAI/Anthropic), pay-per-session
Postgres (+ pgvector)          → Render Postgres, or Supabase free (no 90-day expiry — prefer this)
Redis (live session state)     → Render Redis, or in-process dict fallback
Audio/video blobs              → Cloudflare R2 / Supabase Storage free tier
```

Honest free-tier warnings:
- Render free web service **sleeps after 15 min** and cold-starts ~30s while loading ASR. Fine for a portfolio demo, not for real users. First paid upgrade ($7/mo Render, or Fly.io) removes this.
- Keeping the interviewer LLM **local** to stay fully free makes the product noticeably dumber at follow-ups. Recommendation: spend the ~cents/session on a hosted mini model; it's the highest-leverage dollar in the whole project.
- **Supabase over Render Postgres** specifically to dodge the 90-day expiry pain.

---

## 9. Tech stack

- **Frontend:** React + Vite, WebRTC, MediaPipe Tasks (Vision) in WASM, Web Audio API, Recharts for the report.
- **Backend:** FastAPI, `websockets`, `asyncio`; whisper.cpp via `pywhispercpp` or ONNX Runtime; Silero VAD; librosa; sentence-transformers; LightGBM; Piper.
- **Infra:** Postgres + pgvector, Redis, S3-compatible store, Docker.
- **Training (Colab):** feature extraction + LightGBM; export scorer to a pickle/ONNX.

---

## 10. Week-by-week roadmap (~10 weeks, one person)

**Wk 1 — Skeleton.** FastAPI + React, WebSocket echo, capture mic+cam in browser, stream audio to server, log it. No models yet. Deliverable: audio reaches the server, you see the waveform.

**Wk 2 — ASR + VAD.** Streaming whisper.cpp, Silero VAD, live transcript in the UI, endpoint detection. Deliverable: you talk, transcript appears, server knows when you stopped.

**Wk 3 — Turn-taking + interviewer.** State machine, LLM question gen (hosted), TTS playback, one full canned→adaptive loop. Deliverable: a real back-and-forth interview, even if scoring is fake.

**Wk 4 — Browser CV.** MediaPipe face/pose, compute eye-contact + posture client-side, stream features up, show a live overlay. Deliverable: the live meters move as you look away / slouch.

**Wk 5 — Delivery features.** Prosody (F0, wpm, pauses), filler detection incl. code-mixed, fuse with CV features per turn. Deliverable: real delivery numbers per answer.

**Wk 6 — Content scoring.** Embedding relevance + LightGBM rubric head, JD parsing + pgvector. Deliverable: "did you answer the question" scores.

**Wk 7 — Report.** Session report page, worst-moment clip timestamps that jump to the video, delivery timeline chart. Deliverable: the shareable end screen that sells the whole thing.

**Wk 8 — The dataset + model.** Collect/label the delivery dataset, train the scorer on Colab, run the content-vs-delivery-vs-fused ablation. Deliverable: the paper's core table.

**Wk 9 — Harden + deploy.** Auth, session limits, cold-start mitigation, error handling, deploy the full stack. Deliverable: a public URL a stranger can use.

**Wk 10 — Paper + polish.** Write up the ablation (and user study if you ran it), record the demo video, README. Deliverable: submission draft + a portfolio-ready repo.

---

## 11. What makes this hard enough to be impressive (so you can defend it in interviews)
- Real-time bidirectional streaming with a sub-1.5s turn latency budget, pipelined LLM→TTS.
- Privacy-preserving architecture: raw video never leaves the browser, only numeric features do.
- Genuine multimodal fusion (audio + vision + text) with a defensible ablation, not three models bolted together.
- A safety-adjacent framing (it judges people) handled honestly: report confidence, never gatekeep, let users see exactly which evidence drove each score.

That combination — real-time systems + on-device CV + multimodal ML + a real paper — is the thing that makes a reviewer or an interviewer stop and ask "wait, you built the whole pipeline yourself?"
