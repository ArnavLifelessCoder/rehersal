import { useEffect, useRef, useState } from 'react';
import { AudioCapture, requestMedia } from '../lib/audio';

const BARS = 36;
const SPEECH_RMS = 0.012;     // clearly above a quiet room
const SPEECH_FRAMES = 6;      // ~0.6s of sustained level, mirrors speech_onset_ms

export default function Preflight({ config, onReady, onBack }) {
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const videoRef = useRef(null);
  const runRef = useRef(0);

  const [bars, setBars] = useState(() => new Array(BARS).fill(0.05));
  const [micState, setMicState] = useState('pending'); // pending|listening|heard|error
  const [camState, setCamState] = useState(config.useCamera ? 'pending' : 'off');
  const [peak, setPeak] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      let stream;
      try {
        stream = await requestMedia({ video: config.useCamera });
      } catch (err) {
        if (cancelled) return;
        setMicState('error');
        setError(
          err?.name === 'NotAllowedError'
            ? 'Hardware permission denied. Please allow microphone access in your browser settings.'
            : 'No active audio input device detected. Please verify your hardware connection.',
        );
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;

      if (config.useCamera) {
        const track = stream.getVideoTracks()[0];
        setCamState(track && track.readyState === 'live' ? 'ok' : 'error');
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      }

      const capture = new AudioCapture({
        onLevel: (v) => {
          setBars((b) => {
            const n = b.slice(1);
            n.push(Math.max(0.04, v));
            return n;
          });
          setPeak((p) => Math.max(p * 0.985, v));
          if (v > SPEECH_RMS) {
            runRef.current += 1;
            if (runRef.current >= SPEECH_FRAMES) setMicState('heard');
          } else {
            runRef.current = Math.max(0, runRef.current - 1);
          }
        },
      });
      captureRef.current = capture;
      await capture.start(stream);
      if (!cancelled) setMicState((s) => (s === 'heard' ? s : 'listening'));
    }

    boot();
    return () => {
      cancelled = true;
      captureRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [config.useCamera]);

  const micCopy = {
    pending: 'Initializing audio stream...',
    listening: 'Speak normally: "Testing audio pipeline, one two three..."',
    heard: 'Crystal clear input signal calibrated & locked.',
    error: 'Audio input stream unavailable.',
  }[micState];

  const canStart = micState === 'heard';

  return (
    <div className="setup preflight-view">
      <div className="setup-hero">
        <div className="brand-badge">
          <span className="brand-sparkle">●</span>
          <span className="brand-badge-text">STAGE 01 — PREFLIGHT CALIBRATION</span>
        </div>
        <h1 className="hero-title">
          Hardware & Sensor <span className="gradient-text">Diagnostics</span>
        </h1>
        <p className="lede">
          Ensuring zero-latency transcription and accurate composure metrics before initiating the session.
        </p>
      </div>

      <div className="preflight-grid">
        <div className="field">
          <div className="field-header">
            <span className="field-label">Microphone Spectrum</span>
            <span className={`signal-tag signal-${micState}`}>
              {micState === 'heard' ? 'SIGNAL LOCKED' : micState === 'listening' ? 'LISTENING' : micState.toUpperCase()}
            </span>
          </div>

          <div className={`check-card ${micState === 'heard' ? 'card-success' : ''}`}>
            <div className="check-head">
              <div className={`status-pill pill-${micState}`}>
                {micState === 'heard' ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                ) : micState === 'error' ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                ) : (
                  <span className="pulse-ping" />
                )}
              </div>
              <div className="check-copy">
                <span className="check-title">Acoustic Input Check</span>
                <span className="check-subtitle">{micCopy}</span>
              </div>
            </div>

            <div className={`levels-container ${micState === 'heard' ? 'levels-active' : ''}`}>
              <div className="levels">
                {bars.map((v, i) => (
                  <div
                    key={i}
                    className="level-bar"
                    style={{ height: `${Math.max(4, Math.round(v * 56))}px` }}
                  />
                ))}
              </div>
            </div>

            {micState === 'listening' && peak > 0 && peak < SPEECH_RMS && (
              <div className="subtle-warning">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                  <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                <span>Input volume is low. Please speak louder or adjust microphone sensitivity.</span>
              </div>
            )}
          </div>
        </div>

        {config.useCamera && (
          <div className="field">
            <div className="field-header">
              <span className="field-label">Visual Composure Viewfinder</span>
              <span className={`signal-tag ${camState === 'ok' ? 'signal-heard' : 'signal-pending'}`}>
                {camState === 'ok' ? 'CAMERA ACTIVE' : 'INITIALIZING'}
              </span>
            </div>

            <div className="check-card viewfinder-card">
              <div className="viewfinder-wrap">
                <div className="hud-corner top-left" />
                <div className="hud-corner top-right" />
                <div className="hud-corner bottom-left" />
                <div className="hud-corner bottom-right" />
                <div className="hud-reticle" />
                <video ref={videoRef} muted playsInline className="preflight-cam-video" />
                <div className="viewfinder-badge">
                  <span className="cam-rec-dot" />
                  <span>ON-DEVICE CV FEED</span>
                </div>
              </div>
              <div className="viewfinder-footer">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
                </svg>
                <span>Position yourself centrally with good lighting facing the display.</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="notice notice-error" style={{ marginBottom: 24 }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      <div className="setup-actions">
        <button
          type="button"
          className="btn btn-primary btn-glow"
          disabled={!canStart}
          onClick={onReady}
        >
          <span>{canStart ? 'Launch Live Interview' : 'Calibrating Audio...'}</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
          </svg>
        </button>
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          Back to Setup
        </button>
      </div>

      {!canStart && micState !== 'error' && (
        <div className="rail-note tip-badge" style={{ marginTop: 18 }}>
          <span className="sparkle-icon">✦</span>
          <span>Speak clearly for a couple seconds to unlock the simulation trigger.</span>
        </div>
      )}
    </div>
  );
}
