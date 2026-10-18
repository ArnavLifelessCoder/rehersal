"""Prosody feature extraction -- pure NumPy, no librosa.

The spec called for librosa/torchaudio. Both are heavy (librosa pulls numba,
torchaudio pulls torch) for what is ultimately four DSP operations on 16kHz
mono. Doing it directly keeps the container ~700MB smaller, which is the
difference between fitting the free tier and not.

F0 is autocorrelation-based with a voicing gate, which is a YIN-lite: accurate
enough for *variance* over an answer, which is all the delivery model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0


@dataclass
class ProsodyFeatures:
    pitch_mean_hz: float
    pitch_var: float
    energy_mean: float
    energy_var: float
    pause_count_per_min: float
    mean_pause_ms: float
    longest_pause_ms: float
    speech_ratio: float
    duration_s: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pitch_mean_hz": self.pitch_mean_hz,
            "pitch_var": self.pitch_var,
            "energy_mean": self.energy_mean,
            "energy_var": self.energy_var,
            "pause_count_per_min": self.pause_count_per_min,
            "mean_pause_ms": self.mean_pause_ms,
            "longest_pause_ms": self.longest_pause_ms,
            "speech_ratio": self.speech_ratio,
        }


def _frame(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if signal.size < frame_len:
        signal = np.pad(signal, (0, frame_len - signal.size))
    n = 1 + (signal.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def _f0_autocorr(frames: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame F0 in Hz; NaN where the frame is unvoiced."""
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(frames.shape[1])
    frames = frames * window

    # Autocorrelation via FFT (Wiener-Khinchin) -- O(n log n) not O(n^2).
    size = 1 << int(np.ceil(np.log2(2 * frames.shape[1])))
    spec = np.fft.rfft(frames, n=size, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n=size, axis=1)[:, : frames.shape[1]]

    lag_min = max(int(sr / F0_MAX_HZ), 2)
    lag_max = min(int(sr / F0_MIN_HZ), frames.shape[1] - 1)
    if lag_max <= lag_min:
        return np.full(frames.shape[0], np.nan)

    window_acf = acf[:, lag_min:lag_max]
    peak_lag = np.argmax(window_acf, axis=1) + lag_min
    peak_val = window_acf.max(axis=1)
    energy = acf[:, 0]

    # Voicing gate: the periodic peak must carry a real share of frame energy.
    voiced = (energy > 1e-8) & (peak_val > 0.35 * energy)
    f0 = np.where(voiced, sr / np.maximum(peak_lag, 1), np.nan)
    return f0


def extract_prosody(audio: np.ndarray, sr: int = 16_000, *, silence_ms: int = 250) -> ProsodyFeatures:
    duration_s = max(audio.size / sr, 1e-3)
    if audio.size < sr // 20:
        return ProsodyFeatures(150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, duration_s)

    frame_len = int(0.032 * sr)   # 32ms
    hop = int(0.010 * sr)         # 10ms
    frames = _frame(audio.astype(np.float64), frame_len, hop)

    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    f0 = _f0_autocorr(frames, sr)

    # Voiced frames only -- averaging silence into pitch flattens the variance
    # that the confidence sub-score is actually reading.
    voiced_f0 = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 150.0
    pitch_var = float(np.var(voiced_f0)) if voiced_f0.size > 1 else 0.0

    # Speech/silence segmentation off an adaptive percentile threshold.
    thresh = max(float(np.percentile(rms, 20)) * 2.2, float(np.max(rms)) * 0.06)
    is_speech = rms > thresh
    speech_ratio = float(np.mean(is_speech))

    pauses_ms: list[float] = []
    run = 0
    for flag in is_speech:
        if flag:
            if run:
                pauses_ms.append(run * 10.0)
                run = 0
        else:
            run += 1
    if run:
        pauses_ms.append(run * 10.0)

    real_pauses = [p for p in pauses_ms if p >= silence_ms]
    minutes = duration_s / 60.0
    return ProsodyFeatures(
        pitch_mean_hz=pitch_mean,
        pitch_var=pitch_var,
        energy_mean=float(np.mean(rms)),
        energy_var=float(np.var(rms)),
        pause_count_per_min=float(len(real_pauses) / minutes) if minutes else 0.0,
        mean_pause_ms=float(np.mean(real_pauses)) if real_pauses else 0.0,
        longest_pause_ms=float(np.max(real_pauses)) if real_pauses else 0.0,
        speech_ratio=speech_ratio,
        duration_s=duration_s,
    )
