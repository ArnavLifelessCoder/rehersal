"""Audio blob storage.

The report promises "here's the moment you said 'um' nine times" and gives a
timestamp. Without the audio behind it that's a claim the user can't check,
which is exactly the kind of unfalsifiable feedback this product is supposed to
avoid. So each candidate turn is written to disk as a WAV and served back for
playback at the cited offset.

Written with the stdlib `wave` module -- no soundfile, no ffmpeg. 16kHz mono
int16 is ~32KB/second, so a 90-second answer is ~2.9MB. That's why retention is
short by default and why starring a session is what keeps it.
"""

from __future__ import annotations

import logging
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BlobStore:
    root: Path
    sample_rate: int = 16_000
    retention_days: int = 7
    max_total_mb: int = 512

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        # Basic traversal guard: session ids are generated hex, but this file
        # writes to disk from a value that reaches us over the wire.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        if not safe:
            raise ValueError("invalid session id")
        return self.root / safe

    def path_for(self, session_id: str, turn_idx: int) -> Path:
        return self._session_dir(session_id) / f"turn-{int(turn_idx):03d}.wav"

    def save_turn(self, session_id: str, turn_idx: int, audio: np.ndarray) -> str | None:
        """Write one candidate turn. Returns a relative key, or None on failure.

        Best-effort like the rest of persistence: losing a recording must never
        interrupt a live interview.
        """
        if audio.size == 0:
            return None
        try:
            path = self.path_for(session_id, turn_idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            pcm = np.clip(audio, -1.0, 1.0)
            pcm = (pcm * 32767).astype("<i2")
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(pcm.tobytes())
            return f"{path.parent.name}/{path.name}"
        except Exception:  # noqa: BLE001
            log.exception("blobs: failed to save %s turn %s", session_id, turn_idx)
            return None

    def read(self, session_id: str, turn_idx: int) -> bytes | None:
        try:
            path = self.path_for(session_id, turn_idx)
            return path.read_bytes() if path.is_file() else None
        except Exception:  # noqa: BLE001
            return None

    def delete_session(self, session_id: str) -> None:
        try:
            shutil.rmtree(self._session_dir(session_id), ignore_errors=True)
        except Exception:  # noqa: BLE001
            log.exception("blobs: failed to delete %s", session_id)

    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*.wav") if f.is_file())

    def prune(self, keep_sessions: set[str] | None = None) -> dict:
        """Delete recordings past retention, then oldest-first if over quota.

        Two independent limits. Age alone doesn't bound disk on a busy day, and
        size alone would delete a session someone recorded an hour ago. Starred
        sessions (`keep_sessions`) survive both.
        """
        keep = keep_sessions or set()
        cutoff = time.time() - self.retention_days * 86_400
        removed, freed = 0, 0

        dirs = [d for d in self.root.iterdir() if d.is_dir()] if self.root.exists() else []
        for d in dirs:
            if d.name in keep:
                continue
            try:
                mtime = max((f.stat().st_mtime for f in d.glob("*.wav")), default=0)
            except Exception:  # noqa: BLE001
                continue
            if mtime and mtime < cutoff:
                size = sum(f.stat().st_size for f in d.glob("*.wav"))
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                freed += size

        limit = self.max_total_mb * 1024 * 1024
        if self.total_bytes() > limit:
            survivors = [
                d for d in self.root.iterdir()
                if d.is_dir() and d.name not in keep and any(d.glob("*.wav"))
            ]
            survivors.sort(key=lambda d: max(f.stat().st_mtime for f in d.glob("*.wav")))
            for d in survivors:
                if self.total_bytes() <= limit:
                    break
                size = sum(f.stat().st_size for f in d.glob("*.wav"))
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                freed += size

        if removed:
            log.info("blobs: pruned %d sessions, freed %.1fMB", removed, freed / 1e6)
        return {"sessions_removed": removed, "bytes_freed": freed}
