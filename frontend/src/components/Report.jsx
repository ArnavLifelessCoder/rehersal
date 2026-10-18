import { useCallback, useRef, useState } from 'react';

const AXIS_COPY = {
  relevance: 'Answer Relevance & Directness',
  structure: 'Logical Structure & Framing',
  completeness: 'Depth & Solution Completeness',
  correctness_signal: 'Technical Specificity & Accuracy',
  confidence: 'Vocal Presence & Authority',
  clarity: 'Verbal Articulation',
  pace: 'Pacing & Cadence',
  presence: 'Physical Composure & Gaze',
};

const pct = (v) => Math.round((v ?? 0) * 100);
const toneOf = (v) => (v >= 0.7 ? 'good' : v >= 0.5 ? 'mid' : 'bad');

function ScoreCard({ name, value, meta, icon }) {
  const tone = toneOf(value);
  return (
    <div className="report-score-card" data-tone={tone}>
      <div className="report-score-head">
        <span className="report-score-icon">{icon}</span>
        <span className="report-score-label">{name}</span>
      </div>
      <div className="report-score-main">
        <span className="report-score-num tabular" data-tone={tone}>
          {pct(value)}
          <span className="report-score-pct">%</span>
        </span>
      </div>
      {meta && <span className="report-score-meta">{meta}</span>}
      <div className="report-score-bar-bg">
        <div
          className="report-score-bar-fill"
          data-tone={tone}
          style={{ width: `${pct(value)}%` }}
        />
      </div>
    </div>
  );
}

function StatCard({ name, value, unit, meta, icon }) {
  return (
    <div className="report-score-card">
      <div className="report-score-head">
        <span className="report-score-icon">{icon}</span>
        <span className="report-score-label">{name}</span>
      </div>
      <div className="report-score-main">
        <span className="report-score-num tabular">
          {value}
          {unit && <span className="report-score-pct">{unit}</span>}
        </span>
      </div>
      {meta && <span className="report-score-meta">{meta}</span>}
      <div className="report-score-bar-bg">
        <div className="report-score-bar-fill" style={{ width: '100%', opacity: 0.4 }} />
      </div>
    </div>
  );
}

function Bars({ data, group }) {
  return (
    <div className="bars-matrix">
      {data.map(({ label, value }, i) => (
        <div className="bar-row-modern" key={label}>
          <div className="bar-row-info">
            <span className="bar-name-modern">{label}</span>
            <span className="bar-num-modern tabular">{pct(value)}%</span>
          </div>
          <div className="bar-track-modern">
            <div
              className="bar-fill-modern"
              data-group={group}
              style={{ width: `${pct(value)}%`, animationDelay: `${i * 60}ms` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function Trend({ timeline }) {
  const points = timeline.filter((t) => t.delivery !== null);
  if (points.length < 2) return null;

  const W = 100, H = 32;
  const path = (key) =>
    points
      .map((t, i) => {
        const x = (i / (points.length - 1)) * W;
        const y = H - (t[key] ?? 0) * (H - 4) - 2;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');

  return (
    <div className="trend-card-wrap">
      <div className="trend-svg-container">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="trend-svg">
          <defs>
            <linearGradient id="deliveryGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path
            d={path('delivery')}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          <path
            d={path('content')}
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeDasharray="4 4"
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      </div>
      <div className="trend-legend">
        <div className="legend-item">
          <span className="legend-line line-delivery" />
          <span>Delivery Composure</span>
        </div>
        <div className="legend-item">
          <span className="legend-line line-content" />
          <span>Content Accuracy</span>
        </div>
        <span className="legend-sample-count">{points.length} consecutive turn evaluations</span>
      </div>
    </div>
  );
}

function useMomentPlayer(sessionId) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(null);
  const [failed, setFailed] = useState(false);

  const play = useCallback((turnIdx, tsMs) => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.addEventListener('ended', () => setPlaying(null));
      audioRef.current.addEventListener('error', () => { setFailed(true); setPlaying(null); });
    }
    const el = audioRef.current;
    const key = `${turnIdx}:${tsMs}`;

    if (playing === key) { el.pause(); setPlaying(null); return; }

    const src = `/api/sessions/${sessionId}/turns/${turnIdx}/audio`;
    const seek = () => {
      el.currentTime = Math.max(0, (tsMs || 0) / 1000 - 1.5);
      el.play().then(() => setPlaying(key)).catch(() => setFailed(true));
    };

    if (!el.src.endsWith(src)) {
      el.src = src;
      el.addEventListener('loadedmetadata', seek, { once: true });
      el.load();
    } else {
      seek();
    }
  }, [sessionId, playing]);

  return { play, playing, failed };
}

export default function Report({ report, onRestart, onViewProgress }) {
  const { play, playing, failed } = useMomentPlayer(report.session_id);
  const delivery = Object.entries(report.delivery_breakdown)
    .map(([k, v]) => ({ label: AXIS_COPY[k] || k, value: v }));
  const content = Object.entries(report.content_breakdown)
    .map(([k, v]) => ({ label: AXIS_COPY[k] || k, value: v }));

  const confLabel =
    report.confidence >= 0.7 ? 'High Statistical Confidence'
      : report.confidence >= 0.45 ? 'Moderate Confidence'
        : 'Preliminary Estimate';

  return (
    <div className="report">
      <div className="report-hero">
        <div className="brand-badge">
          <span className="brand-sparkle">✦</span>
          <span className="brand-badge-text">POST-SIMULATION PERFORMANCE AUDIT</span>
        </div>

        <h1 className="hero-title">
          Session Diagnostic <span className="gradient-text">Report</span>
        </h1>
        <p className="report-summary">{report.summary_text}</p>
      </div>

      <section>
        <div className="scores-grid">
          <ScoreCard
            name="Delivery & Cadence"
            value={report.overall_delivery}
            meta={confLabel}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              </svg>
            }
          />
          <ScoreCard
            name="Content & Logic"
            value={report.overall_content}
            meta={confLabel}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
              </svg>
            }
          />
          <StatCard
            name="Filler Word Count"
            value={report.filler_total}
            unit="tokens"
            meta={`${report.filler_per_min} fillers / min`}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            }
          />
          <StatCard
            name="Answer Turns"
            value={report.answers_given}
            unit="turns"
            meta={`${Math.round(report.speaking_time_s)}s spoken total`}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
            }
          />
        </div>
      </section>

      {report.worst_moments.length > 0 && (
        <section className="report-section">
          <div className="section-header">
            <span className="section-tag tag-warning">CRITICAL FEEDBACK</span>
            <h2>Actionable Weak Spots & Opportunities</h2>
          </div>
          <div className="moments-list">
            {report.worst_moments.map((m, i) => {
              const key = `${m.turn_idx}:${m.ts_ms}`;
              const isCurrentPlaying = playing === key;
              return (
                <div className="moment-card" key={i}>
                  <div className="moment-badge">#{i + 1}</div>
                  <div className="moment-content">
                    <div className="moment-what">{m.reason}</div>
                    <div className="moment-where">
                      <span className="turn-pill">Turn {m.turn_idx}</span>
                      {m.question && <span className="question-quote">"{m.question.slice(0, 95)}..."</span>}
                      {m.ts_ms > 0 && <span className="timestamp-pill">@{(m.ts_ms / 1000).toFixed(1)}s</span>}
                    </div>
                  </div>
                  {!failed && (
                    <button
                      type="button"
                      className={`btn-listen ${isCurrentPlaying ? 'btn-playing' : ''}`}
                      onClick={() => play(m.turn_idx, m.ts_ms)}
                      aria-label={`Play recording of answer ${m.turn_idx}`}
                    >
                      {isCurrentPlaying ? (
                        <>
                          <span className="play-wave" />
                          <span>Stop</span>
                        </>
                      ) : (
                        <>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"/>
                          </svg>
                          <span>Hear Audio</span>
                        </>
                      )}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {report.strengths.length > 0 && (
        <section className="report-section">
          <div className="section-header">
            <span className="section-tag tag-success">COMPETITIVE EDGE</span>
            <h2>Demonstrated Strengths</h2>
          </div>
          <div className="strengths-grid">
            {report.strengths.map((s, i) => (
              <div className="strength-card" key={i}>
                <div className="strength-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                </div>
                <span>{s.reason}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="breakdowns-split">
        <section className="report-section">
          <div className="section-header">
            <span className="section-tag">ACOUSTICS & CADENCE</span>
            <h2>Delivery Dimensions</h2>
          </div>
          <Bars data={delivery} group="delivery" />
        </section>

        <section className="report-section">
          <div className="section-header">
            <span className="section-tag">SUBSTANCE & STRUCTURE</span>
            <h2>Content Matrix</h2>
          </div>
          <Bars data={content} group="content" />
        </section>
      </div>

      {report.delivery_timeline.length > 1 && (
        <section className="report-section">
          <div className="section-header">
            <span className="section-tag">CHRONOLOGICAL DYNAMICS</span>
            <h2>Performance Over Time</h2>
          </div>
          <Trend timeline={report.delivery_timeline} />
        </section>
      )}

      {Object.keys(report.filler_breakdown).length > 0 && (
        <section className="report-section">
          <div className="section-header">
            <span className="section-tag">DISCOURSE PARTICLES</span>
            <h2>Filler Word Heatmap</h2>
          </div>
          <div className="filler-chips">
            {Object.entries(report.filler_breakdown).map(([word, n]) => (
              <div className="filler-chip-card" key={word}>
                <span className="filler-word">"{word}"</span>
                <span className="filler-count">{n}x</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="report-section">
        <div className="section-header">
          <span className="section-tag">RAW SESSION LOG</span>
          <h2>Verbatim Transcript & Turn Scores</h2>
        </div>
        <div className="transcript-chat">
          {report.transcript.map((t, i) => (
            <div className={`chat-bubble-row ${t.role}`} key={i}>
              <div className="chat-avatar">
                {t.role === 'interviewer' ? 'AI' : 'YOU'}
              </div>
              <div className="chat-content">
                <div className="chat-author">
                  <span>{t.role === 'interviewer' ? 'AI Interviewer' : 'Candidate'}</span>
                  {t.role === 'candidate' && t.delivery && (
                    <div className="chat-scores">
                      <span className="chat-score-pill">Delivery: {pct(t.delivery.overall)}%</span>
                      <span className="chat-score-pill">Content: {pct(t.content.overall)}%</span>
                      {t.fillers?.filler_count > 0 && (
                        <span className="chat-score-pill pill-fillers">{t.fillers.filler_count} fillers</span>
                      )}
                    </div>
                  )}
                </div>
                <p className="chat-text">{t.question_text || t.transcript}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="report-section report-meta-card">
        <div className="section-header">
          <span className="section-tag">AUDIT METADATA</span>
          <h2>Telemetry & Pipeline Provenance</h2>
        </div>
        <p className="method-text">
          Transcribed with <code>{report.backends.asr}</code>, questions synthesized via{' '}
          <code>{report.backends.brain}</code>, acoustic delivery scored by{' '}
          <code>{report.backends.delivery_scorer}</code>, conceptual matrix scored by{' '}
          <code>{report.backends.content_scorer}</code>.
          {report.had_vision
            ? ' Computer vision posture & gaze were processed locally via client-side MediaPipe.'
            : ' Camera disabled — presence dimension omitted.'}
          {report.latency?.turns_measured > 0 && (
            <> Median turn turnaround latency: {report.latency.p50_ms}ms across {report.latency.turns_measured} conversational turns.</>
          )}
        </p>
      </section>

      <div className="report-actions">
        <button type="button" className="btn btn-primary btn-glow" onClick={onRestart}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
          </svg>
          <span>Run Another Simulation</span>
        </button>
        {onViewProgress && (
          <button type="button" className="btn btn-secondary" onClick={onViewProgress}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
            </svg>
            <span>Progress Analytics</span>
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={() => downloadReport(report)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          <span>Export Markdown</span>
        </button>
      </div>
    </div>
  );
}

function downloadReport(report) {
  const line = [];
  const p = (v) => Math.round((v ?? 0) * 100);

  line.push(`# Rehearsal Diagnostic Report — ${report.track.replace('_', ' ').toUpperCase()}`, '');
  line.push(report.summary_text, '');
  line.push(`- **Overall Delivery:** ${p(report.overall_delivery)}/100`);
  line.push(`- **Overall Content:** ${p(report.overall_content)}/100`);
  line.push(`- **Filler Words:** ${report.filler_total} (${report.filler_per_min}/min)`);
  line.push(`- **Speaking Turns:** ${report.answers_given} over ${Math.round(report.speaking_time_s)}s of active speech`);
  line.push(`- **Confidence Level:** ${p(report.confidence)}%`, '');

  if (report.worst_moments.length) {
    line.push('## Actionable Weak Spots', '');
    report.worst_moments.forEach((m, i) => {
      line.push(`${i + 1}. ${m.reason} _(Answer Turn ${m.turn_idx})_`);
    });
    line.push('');
  }
  if (report.strengths.length) {
    line.push('## Demonstrated Strengths', '');
    report.strengths.forEach((s) => line.push(`- ${s.reason}`));
    line.push('');
  }

  line.push('## Metric Breakdown', '');
  line.push('| Dimension | Score |', '| --- | --- |');
  Object.entries({ ...report.delivery_breakdown, ...report.content_breakdown })
    .forEach(([k, v]) => line.push(`| ${AXIS_COPY[k] || k} | ${p(v)}% |`));
  line.push('');

  line.push('## Conversational Transcript', '');
  report.transcript.forEach((t) => {
    const who = t.role === 'interviewer' ? '**AI Interviewer:**' : '**Candidate:**';
    line.push(`${who} ${t.question_text || t.transcript}`, '');
  });

  const blob = new Blob([line.join('\n')], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rehearsal-${report.track}-${report.session_id.slice(0, 6)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}
