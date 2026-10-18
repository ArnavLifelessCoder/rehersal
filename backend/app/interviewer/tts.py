"""Text-to-speech for the interviewer voice.

Default backend is `browser`: we send the sentence to the client and let the Web
Speech API speak it. This is a deliberate change from the spec's server-side
Piper default, for three reasons:

  1. Latency. Server TTS costs an encode + a network hop per sentence (~250ms
     budgeted). Browser synthesis starts in ~30ms, so the turn budget stops
     being tight and starts being comfortable.
  2. Footprint. Piper's 60MB and its ONNX runtime come straight off the 512MB
     free-tier budget.
  3. Bandwidth. Audio downstream on a free tier is the thing that gets throttled.

Piper stays available for deployments that want a consistent voice across
devices (browser voices vary wildly), which is the one thing browser TTS is
genuinely worse at.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class SpeechChunk:
    text: str
    audio_b64: str | None = None   # None => client should synthesise locally
    mime: str = "audio/wav"

    def to_dict(self) -> dict:
        return {"text": self.text, "audio": self.audio_b64, "mime": self.mime}


class BrowserTTS:
    """Pass-through: the client speaks the text with the Web Speech API."""

    name = "browser"
    client_synthesises = True

    async def synthesise(self, text: str) -> SpeechChunk:
        return SpeechChunk(text=text, audio_b64=None)


class PiperTTS:
    name = "piper"
    client_synthesises = False

    def __init__(self, voice_model: Path | None = None) -> None:
        self._exe = shutil.which("piper")
        if not self._exe:
            raise RuntimeError("piper binary not on PATH")
        self._voice = voice_model

    async def synthesise(self, text: str) -> SpeechChunk:
        return await asyncio.to_thread(self._synthesise_sync, text)

    def _synthesise_sync(self, text: str) -> SpeechChunk:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            cmd = [self._exe, "--output_file", str(out)]
            if self._voice:
                cmd += ["--model", str(self._voice)]
            subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
            return SpeechChunk(text, base64.b64encode(out.read_bytes()).decode("ascii"))
        finally:
            out.unlink(missing_ok=True)


def build_tts(backend: str):
    if backend == "piper":
        try:
            log.info("TTS backend: piper")
            return PiperTTS()
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS backend: browser (piper unavailable: %s)", exc)
    log.info("TTS backend: browser (Web Speech API)")
    return BrowserTTS()
