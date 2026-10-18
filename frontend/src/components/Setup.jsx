import { useEffect, useState } from 'react';
import { fetchHealth } from '../lib/session';

const TRACKS = [
  {
    id: 'behavioral',
    name: 'Behavioural',
    note: 'Specific leadership stories, real impact & STAR method.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" x2="12" y1="19" y2="22"/>
      </svg>
    ),
    tag: 'Culture & EQ',
  },
  {
    id: 'dsa',
    name: 'Algorithms',
    note: 'Optimal approach, space/time complexity & edge cases.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6"/>
        <polyline points="8 6 2 12 8 18"/>
        <line x1="14" x2="10" y1="4" y2="20"/>
      </svg>
    ),
    tag: 'CS Fundamentals',
  },
  {
    id: 'system_design',
    name: 'System Design',
    note: 'Distributed scale, bottlenecks, tradeoffs & resilience.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect width="7" height="7" x="14" y="3" rx="1"/>
        <path d="M10 21V8a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5a1 1 0 0 0-1-1H3"/>
      </svg>
    ),
    tag: 'Large Scale',
  },
  {
    id: 'jd',
    name: 'Targeted Job Spec',
    note: 'Paste any job description to generate tailored questions.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="22" x2="18" y1="12" y2="12"/>
        <line x1="6" x2="2" y1="12" y2="12"/>
        <line x1="12" x2="12" y1="6" y2="2"/>
        <line x1="12" x2="12" y1="22" y2="18"/>
      </svg>
    ),
    tag: 'Custom Role',
  },
];

const RUNTIME_LABELS = {
  asr: 'Transcription Engine',
  brain: 'Adaptive Interviewer',
  tts: 'Neural Voice Synthesis',
  delivery_scorer: 'Delivery Dynamics',
  embedder: 'Semantic Concept Matrix',
};

const FALLBACKS = new Set(['mock', 'semantic', 'lexical', 'heuristic']);

export default function Setup({ onStart, onViewProgress, error }) {
  const [track, setTrack] = useState('behavioral');
  const [jdText, setJdText] = useState('');
  const [useCamera, setUseCamera] = useState(true);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch((e) => setHealthError(e.message));
  }, []);

  const ready = track !== 'jd' || jdText.trim().length > 30;

  return (
    <div className="setup">
      <div className="setup-hero">
        <div className="brand-badge">
          <span className="brand-sparkle">✦</span>
          <span className="brand-badge-text">NEURAL INTERVIEW LAB</span>
          <span className="brand-version">v2.4</span>
        </div>

        <h1 className="hero-title">
          Master the art of <span className="gradient-text">high-stakes communication</span>.
        </h1>
        <p className="lede">
          Real-time conversational AI that evaluates your verbal depth, cadence, filler words,
          and non-verbal composure in high-pressure interview settings.
        </p>
      </div>

      <div className="field">
        <div className="field-header">
          <span className="field-label">Select Simulation Track</span>
          <span className="field-sublabel">Tailored questioning engine</span>
        </div>
        <div className="tracks">
          {TRACKS.map((t) => {
            const isSelected = track === t.id;
            return (
              <button
                key={t.id}
                type="button"
                className={`track ${isSelected ? 'track-selected' : ''}`}
                aria-pressed={isSelected}
                onClick={() => setTrack(t.id)}
              >
                <div className="track-icon-wrapper">
                  <span className="track-icon">{t.icon}</span>
                  <span className="track-tag">{t.tag}</span>
                </div>
                <div className="track-content">
                  <span className="track-name">{t.name}</span>
                  <span className="track-note">{t.note}</span>
                </div>
                <div className="track-radio-indicator">
                  <div className="track-radio-dot" />
                </div>
              </button>
            );
          })}
        </div>

        {track === 'jd' && (
          <div className="jd-wrapper">
            <div className="jd-header">
              <label htmlFor="jd-textarea" className="jd-label">
                <span>Job Description / Role Requirements</span>
                <span className="char-counter">{jdText.trim().length} chars</span>
              </label>
            </div>
            <textarea
              id="jd-textarea"
              className="jd-input"
              rows={6}
              placeholder="Paste job posting here (e.g. Senior Frontend Engineer with React, WebRTC, High-Scale Systems requirements)..."
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              autoFocus
            />
          </div>
        )}
      </div>

      <div className="field">
        <div className="field-header">
          <span className="field-label">Sensor Telemetry</span>
          <span className="field-sublabel">Private edge processing</span>
        </div>
        <label className="switch-row">
          <div className="switch-left">
            <div className="privacy-shield-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
              </svg>
            </div>
            <div>
              <div className="switch-title-wrap">
                <span className="switch-title">Computer Vision Composure Tracking</span>
                <span className="privacy-badge">100% Client-Side</span>
              </div>
              <span className="switch-note">
                Live eye-contact and posture analytics are processed on-device via MediaPipe.
                Zero video frames or raw facial data leave your browser.
              </span>
            </div>
          </div>
          <div className="toggle-switch">
            <input
              type="checkbox"
              id="camera-toggle"
              checked={useCamera}
              onChange={(e) => setUseCamera(e.target.checked)}
            />
            <span className="toggle-slider" />
          </div>
        </label>
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
          disabled={!ready}
          onClick={() => onStart({ track, jdText: track === 'jd' ? jdText : null, useCamera })}
        >
          <span>Enter Simulator</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
          </svg>
        </button>
        {onViewProgress && (
          <button type="button" className="btn btn-secondary" onClick={onViewProgress}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
            </svg>
            <span>Past Sessions</span>
          </button>
        )}
        {track === 'jd' && !ready && (
          <span className="rail-note tip-badge">Paste at least 30 characters of the job description.</span>
        )}
      </div>

      <div className="runtime">
        <div className="runtime-header">
          <div className="runtime-title-wrap">
            <span className="runtime-dot" />
            <h2>System Architecture Status</h2>
          </div>
          <span className="runtime-badge">Live Microservices</span>
        </div>

        {healthError && <div className="notice notice-error">Backend unreachable — {healthError}</div>}
        {health && (
          <>
            <div className="runtime-grid">
              {Object.entries(RUNTIME_LABELS).map(([key, label]) => {
                const isFallback = FALLBACKS.has(health.backends[key]);
                return (
                  <div className="runtime-item" key={key}>
                    <div className="runtime-item-top">
                      <span className={`status-indicator ${isFallback ? 'status-fallback' : 'status-live'}`} />
                      <span className="runtime-key">{label}</span>
                    </div>
                    <span
                      className="runtime-val"
                      data-fallback={isFallback ? 'true' : 'false'}
                    >
                      {health.backends[key]}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="runtime-footer">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
              </svg>
              <span>
                {Object.entries(RUNTIME_LABELS).some(([k]) => FALLBACKS.has(health.backends[k]))
                  ? 'Active offline intelligence engine enabled with adaptive concept graphs.'
                  : 'All neural pipelines executing at ultra-low latency.'}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
