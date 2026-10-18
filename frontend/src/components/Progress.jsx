import { useEffect, useState } from 'react';

const METRICS = [
  { key: 'delivery', label: 'Delivery Composure', fmt: (v) => `${Math.round(v * 100)}%`, lowerBetter: false },
  { key: 'content', label: 'Content Depth', fmt: (v) => `${Math.round(v * 100)}%`, lowerBetter: false },
  { key: 'filler_per_min', label: 'Fillers / Min', fmt: (v) => v.toFixed(1), lowerBetter: true },
];

function Sparkline({ points, accessor, lowerBetter }) {
  const values = points.map(accessor).filter((v) => v !== null && v !== undefined);
  if (values.length < 2) return null;

  const W = 100, H = 32;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * W;
      const y = H - ((v - min) / span) * (H - 6) - 3;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');

  const first = values[0], last = values[values.length - 1];
  const better = lowerBetter ? last < first : last > first;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="spark-svg">
      <path
        d={path}
        fill="none"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
        stroke={better ? 'var(--positive)' : 'var(--accent)'}
      />
    </svg>
  );
}

function Delta({ metric, trend }) {
  const t = trend?.[metric.key];
  if (!t) return <span className="delta-pill delta-none">—</span>;
  if (t.improved === null) return <span className="delta-pill delta-none">Neutral</span>;
  const sign = t.delta > 0 ? '+' : '';
  return (
    <span className={`delta-pill ${t.improved ? 'delta-up' : 'delta-down'}`}>
      {t.improved ? '▲' : '▼'} {sign}{metric.fmt(t.delta)}
    </span>
  );
}

export default function Progress({ onBack, onOpenReport }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/progress')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('could not load progress'))))
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="report">
        <div className="brand-badge">
          <span className="brand-sparkle">✦</span>
          <span className="brand-badge-text">PROGRESS INTELLIGENCE</span>
        </div>
        <div className="notice notice-error">{error}</div>
        <div className="report-actions" style={{ marginTop: 28 }}>
          <button className="btn btn-secondary" onClick={onBack}>Back to Simulator</button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="setup" style={{ textAlign: 'center', padding: '120px 24px' }}>
        <div className="pulse-ping" style={{ margin: '0 auto 18px', width: 24, height: 24 }} />
        <span className="rail-note">Retrieving longitudinal session history...</span>
      </div>
    );
  }

  const { sessions, trend } = data;

  return (
    <div className="report">
      <div className="report-hero">
        <div className="brand-badge">
          <span className="brand-sparkle">✦</span>
          <span className="brand-badge-text">LONGITUDINAL PERFORMANCE MATRIX</span>
        </div>
        <h1 className="hero-title">
          Skill Trajectory & <span className="gradient-text">Evolution</span>
        </h1>

        {sessions.length === 0 ? (
          <p className="report-summary">
            No completed sessions recorded yet. Run your first simulation to initialize baseline metrics.
          </p>
        ) : (
          <p className="report-summary">
            {trend.available
              ? `Comparing baseline (${trend.window} initial sessions) against recent (${trend.window} latest sessions).`
              : `${sessions.length} session${sessions.length === 1 ? '' : 's'} recorded. ` +
                `Comprehensive statistical trends unlock after ${trend.needed} sessions.`}
          </p>
        )}
      </div>

      {sessions.length > 0 && (
        <>
          <section className="report-section">
            <div className="section-header">
              <span className="section-tag">BENCHMARK TRENDS</span>
              <h2>Performance Indicators</h2>
            </div>
            <div className="trend-cards-grid">
              {METRICS.map((m) => (
                <div className="trend-hud-card" key={m.key}>
                  <div className="trend-hud-head">
                    <span className="trend-hud-label">{m.label}</span>
                    {trend.available && <Delta metric={m} trend={trend} />}
                  </div>
                  <div className="trend-spark-box">
                    <Sparkline
                      points={sessions}
                      accessor={(s) => s[m.key]}
                      lowerBetter={m.lowerBetter}
                    />
                  </div>
                  <div className="trend-hud-foot tabular">
                    {(() => {
                      const t = trend.available ? trend[m.key] : null;
                      if (t) return `${m.fmt(t.early)} → ${m.fmt(t.late)} avg`;
                      const vals = sessions.map((s) => s[m.key]).filter((v) => v != null);
                      if (!vals.length) return '—';
                      return `${m.fmt(vals[0])} → ${m.fmt(vals[vals.length - 1])}`;
                    })()}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="report-section">
            <div className="section-header">
              <span className="section-tag">SIMULATION ARCHIVE</span>
              <h2>Recorded Sessions</h2>
            </div>
            <div className="session-cards-list">
              {[...sessions].reverse().map((s) => (
                <button
                  type="button"
                  className="session-row-card"
                  key={s.session_id}
                  onClick={() => onOpenReport(s.session_id)}
                >
                  <div className="session-row-left">
                    <span className="session-date-pill">
                      {s.started_at ? new Date(s.started_at).toLocaleDateString(undefined,
                        { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Recent'}
                    </span>
                    <span className="session-track-name">{s.track.replace('_', ' ')}</span>
                  </div>

                  <div className="session-stats-group">
                    <div className="session-stat-pill">
                      <span className="stat-pill-num tabular">
                        {s.delivery != null ? `${Math.round(s.delivery * 100)}%` : '—'}
                      </span>
                      <span className="stat-pill-label">Delivery</span>
                    </div>
                    <div className="session-stat-pill">
                      <span className="stat-pill-num tabular">
                        {s.content != null ? `${Math.round(s.content * 100)}%` : '—'}
                      </span>
                      <span className="stat-pill-label">Content</span>
                    </div>
                    <div className="session-stat-pill">
                      <span className="stat-pill-num tabular">
                        {s.filler_per_min != null ? s.filler_per_min.toFixed(1) : '—'}
                      </span>
                      <span className="stat-pill-label">Fillers/m</span>
                    </div>
                    <div className="session-arrow">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <polyline points="9 18 15 12 9 6"/>
                      </svg>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </section>
        </>
      )}

      <div className="report-actions">
        <button type="button" className="btn btn-primary btn-glow" onClick={onBack}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          <span>Launch New Simulation</span>
        </button>
      </div>
    </div>
  );
}
