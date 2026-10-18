"""Drive a full interview against a RUNNING server over a real WebSocket.

Complements the in-process tests: this one exercises the actual wire protocol
(binary int16 PCM frames, JSON control messages, the HTTP session handshake),
which is exactly the layer where a client and server quietly disagree about
sample format and everyone blames the model.

    python scripts/smoke_interview.py --answers 3

The default answers are synthetic voiced tones, which only the mock ASR turns
into words -- real Whisper correctly hears no speech in them and the turn is
discarded. With faster-whisper loaded, pass a recording of real speech
(16kHz mono 16-bit WAV) instead:

    python scripts/smoke_interview.py --answer-wav answer.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets  # needed only for this smoke script")

import urllib.request

SR = 16_000
FRAME_MS = 100


def voiced(seconds: float, freq: float = 165.0, amp: float = 0.25) -> np.ndarray:
    """Synthetic voiced speech: harmonics + vibrato + a syllable envelope."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(11)
    wobble = 1.0 + 0.05 * np.sin(2 * np.pi * 4.0 * t)
    sig = sum(
        (1.0 / k) * np.sin(2 * np.pi * k * freq * wobble * t) for k in (1, 2, 3, 4)
    )
    sig *= 0.75 + 0.25 * np.sin(2 * np.pi * 3.0 * t)   # syllable rate
    return (amp * sig + rng.normal(0, 0.003, t.size)).astype(np.float32)


def room_tone(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.normal(0, 0.0007, int(seconds * SR)).astype(np.float32)


def noise_burst() -> np.ndarray:
    """A door slam and two keyboard clacks — the classic false-trigger set."""
    rng = np.random.default_rng(31)
    slam_n = int(0.18 * SR)
    slam = 0.8 * rng.normal(0, 1, slam_n) * np.exp(-np.arange(slam_n) / (0.022 * SR))
    clack_n = int(0.03 * SR)
    clack = 0.5 * rng.normal(0, 1, clack_n) * np.exp(-np.arange(clack_n) / (0.004 * SR))
    return np.concatenate([
        room_tone(0.3), slam, room_tone(0.4), clack, room_tone(0.2), clack, room_tone(1.4)
    ]).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            sys.exit(f"{path}: need 16kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


async def send_audio(ws, signal: np.ndarray) -> None:
    """Stream in real time so VAD and endpointing see realistic pacing."""
    step = int(SR * FRAME_MS / 1000)
    for i in range(0, len(signal), step):
        await ws.send(to_pcm16(signal[i : i + step]))
        await asyncio.sleep(FRAME_MS / 1000)


async def send_cv(ws, seconds: float, stop: asyncio.Event) -> None:
    """Emit plausible CV samples at 5Hz, as the browser would."""
    t0 = time.time()
    i = 0
    while not stop.is_set() and time.time() - t0 < seconds:
        await ws.send(json.dumps({
            "type": "cv",
            "ts_ms": int((time.time() - t0) * 1000),
            "face_present": True,
            "eye_contact": i % 5 != 0,
            "gaze_offset": 0.18,
            "posture": 0.8,
            "blink": i % 15 == 0,
            "head_movement": 0.08,
        }))
        i += 1
        await asyncio.sleep(0.2)


async def main(answers: int, host: str, answer_wav: str | None = None) -> int:
    recorded = load_wav(answer_wav) if answer_wav else None
    req = urllib.request.Request(
        f"http://{host}/api/sessions",
        data=json.dumps({"track": "dsa"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        session = json.loads(res.read())
    sid = session["session_id"]
    print(f"session {sid}")

    report: dict | None = None
    stop_cv = asyncio.Event()

    async with websockets.connect(f"ws://{host}/ws/{sid}?track=dsa", max_size=None) as ws:
        cv_task = asyncio.create_task(send_cv(ws, 600, stop_cv))
        answered = 0
        speaking_done = asyncio.Event()

        async def reader():
            nonlocal report, answered
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg["type"]
                if kind == "ready":
                    print(f"  backends: {msg['backends']}")
                elif kind == "speech_end":
                    print(f"  [{msg['move']}] {msg['text'][:100]}")
                    speaking_done.set()
                elif kind == "transcript" and msg["final"]:
                    print(f"  you said: {msg['text'][:90]}…")
                elif kind == "turn_scored":
                    t = msg["turn"]
                    answered += 1
                    print(
                        f"  scored -> delivery {t['delivery']['overall']:.2f} "
                        f"content {t['content']['overall']:.2f} "
                        f"fillers {t['fillers']['filler_count']}"
                    )
                elif kind == "latency":
                    tm = msg["timings"]
                    flag = "OK " if tm["within_budget"] else "OVER"
                    # total_ms starts when the endpointer fires: the deliberate
                    # end-of-turn silence is excluded, and the budget with it.
                    print(f"  latency {flag} {tm['total_ms']:.0f}ms / {tm['budget_ms']}ms "
                          f"after end-of-turn silence (felt: {tm.get('perceived_ms', 0):.0f}ms "
                          f"/ {tm.get('perceived_budget_ms', '?')}ms)")
                elif kind == "report":
                    report = msg["report"]
                    return
                elif kind == "error":
                    print(f"  ERROR {msg['message']}")

        reader_task = asyncio.create_task(reader())

        for n in range(answers):
            try:
                await asyncio.wait_for(speaking_done.wait(), timeout=30)
            except asyncio.TimeoutError:
                print("FAIL: interviewer never replied (real ASR + synthetic tones? "
                      "pass --answer-wav)")
                return 1
            speaking_done.clear()
            await ws.send(json.dumps({"type": "playback_done"}))

            # Before the first answer, fire the noises that used to take the
            # floor. Nothing should be transcribed and the interviewer must
            # stay silent.
            if n == 0:
                print("noise check (slam + clacks)…")
                turns_before = answered
                await send_audio(ws, noise_burst())
                await asyncio.sleep(0.6)
                if answered > turns_before or speaking_done.is_set():
                    print("  FAIL: noise was treated as an answer")
                    return 1
                print("  ok: noise ignored, floor retained")

            print(f"answer {n + 1}…")
            # Trailing silence must exceed the endpoint wait plus both patience
            # graces (the interviewer deliberately waits longer when an answer
            # looks unfinished), or the turn never closes and this hangs.
            await send_audio(ws, room_tone(0.8))
            answer = recorded if recorded is not None else voiced(4.0, freq=160 + n * 12)
            await send_audio(ws, answer)
            await send_audio(ws, room_tone(4.5))

        await asyncio.sleep(1.5)
        await ws.send(json.dumps({"type": "end"}))
        try:
            await asyncio.wait_for(reader_task, timeout=20)
        except asyncio.TimeoutError:
            print("FAIL: no report received")
            return 1
        finally:
            stop_cv.set()
            cv_task.cancel()

    if not report:
        print("FAIL: no report")
        return 1

    print("\n--- REPORT ---")
    print(f"summary       : {report['summary_text']}")
    print(f"delivery      : {report['overall_delivery']:.2f}  content: {report['overall_content']:.2f}")
    print(f"confidence    : {report['confidence']:.2f}")
    print(f"answers       : {report['answers_given']}   fillers: {report['filler_total']}")
    print(f"latency       : {report['latency']}")
    print(f"worst moments : {len(report['worst_moments'])}")
    for m in report["worst_moments"][:3]:
        print(f"   - {m['reason']}")
    print(f"strengths     : {[s['reason'] for s in report['strengths']]}")

    ok = report["answers_given"] >= 1 and report["summary_text"]
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--answers", type=int, default=3)
    p.add_argument("--host", default="localhost:8000")
    p.add_argument("--answer-wav", help="16kHz mono WAV of real speech, sent as every answer")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args.answers, args.host, args.answer_wav)))
