function tone(value, { good, caution }) {
  if (value === null || value === undefined) return undefined;
  if (good(value)) return 'good';
  if (caution(value)) return 'caution';
  return 'bad';
}

function MetricCard({ label, value, unit, fill, toneName, icon }) {
  return (
    <div className="metric-hud-card" data-tone={toneName}>
      <div className="metric-hud-header">
        <div className="metric-hud-title-wrap">
          {icon && <span className="metric-hud-icon">{icon}</span>}
          <span className="metric-hud-label">{label}</span>
        </div>
        <span className="metric-hud-val tabular" data-tone={toneName}>
          {value}
          {unit && <span className="metric-hud-unit">{unit}</span>}
        </span>
      </div>
      <div className="metric-hud-bar-track">
        <div
          className="metric-hud-bar-fill"
          data-tone={toneName}
          style={{ width: `${Math.round(Math.max(0, Math.min(1, fill)) * 100)}%` }}
        />
      </div>
    </div>
  );
}

function Levels({ bars, active }) {
  return (
    <div className={`levels-box ${active ? 'levels-active' : ''}`}>
      <div className="levels-box-header">
        <span className="levels-dot" />
        <span>LIVE AUDIO ENVELOPE</span>
      </div>
      <div className="levels">
        {bars.map((v, i) => (
          <div
            key={i}
            className="level-bar"
            style={{ height: `${Math.max(3, Math.round(v * 32))}px` }}
          />
        ))}
      </div>
    </div>
  );
}

export default function Readouts({
  wpm, fillers, eyeContact, posture, levelBars, listening, hasVision, calibrating,
}) {
  const paceTone = tone(wpm, {
    good: (v) => v >= 110 && v <= 170,
    caution: (v) => v >= 90 && v <= 195,
  });
  const fillerTone = fillers === 0 ? 'good' : fillers <= 3 ? 'good' : fillers <= 7 ? 'caution' : 'bad';

  return (
    <div className="readouts-panel">
      <div className="readouts-panel-header">
        <span className="panel-title">TELEMETRY HUD</span>
        <span className="live-pill">
          <span className="live-dot" /> LIVE
        </span>
      </div>

      <div className="metrics-stack">
        <MetricCard
          label="Filler Frequency"
          value={fillers}
          unit="count"
          fill={Math.min(fillers / 10, 1)}
          toneName={fillerTone}
          icon={
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          }
        />
        <MetricCard
          label="Speech Pace"
          value={wpm === null ? '—' : Math.round(wpm)}
          unit={wpm === null ? '' : 'wpm'}
          fill={wpm === null ? 0 : Math.min(wpm / 220, 1)}
          toneName={paceTone}
          icon={
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
          }
        />

        {hasVision && !calibrating && (
          <>
            <MetricCard
              label="Visual Gaze & Eye Contact"
              value={Math.round(eyeContact * 100)}
              unit="%"
              fill={eyeContact}
              toneName={tone(eyeContact, { good: (v) => v >= 0.55, caution: (v) => v >= 0.35 })}
              icon={
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>
                </svg>
              }
            />
            <MetricCard
              label="Composure & Posture"
              value={posture >= 0.7 ? 'Centered' : posture >= 0.5 ? 'Fair' : 'Slouched'}
              fill={posture}
              toneName={tone(posture, { good: (v) => v >= 0.7, caution: (v) => v >= 0.5 })}
              icon={
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
                </svg>
              }
            />
          </>
        )}
      </div>

      <div style={{ marginTop: 14 }}>
        <Levels bars={levelBars} active={listening} />
      </div>

      {calibrating && (
        <div className="calibrating-pill">
          <span className="pulse-ping" />
          <span>Calibrating facial landmarks...</span>
        </div>
      )}
      {!hasVision && (
        <div className="rail-note" style={{ marginTop: 8 }}>
          <span>Camera inactive · Acoustic metrics only</span>
        </div>
      )}
    </div>
  );
}
