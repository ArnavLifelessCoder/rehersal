const KIND_TONE = {
  decision: 'accent',
  vad_endpoint: 'good',
  vad_speech_start: 'good',
  vad_stuck: 'warn',
  vad_ramble: 'warn',
  noise_discarded: 'warn',
  latency_breach: 'bad',
  playback_timeout: 'bad',
  spoke: 'dim',
};

function fmtEvent(e) {
  switch (e.kind) {
    case 'decision':
      return `${e.move} — content ${e.content} · words ${e.words}` +
        (e.weakest ? ` · weakest: ${String(e.weakest).split('--')[0].trim()}` : '');
    case 'vad_endpoint':
      return `endpoint after ${((e.speech_ms ?? 0) / 1000).toFixed(1)}s of speech`;
    case 'vad_speech_start':
      return 'speech opened';
    case 'vad_stuck':
      return 'stuck — no speech before timeout';
    case 'vad_ramble':
      return `ramble limit at ${((e.speech_ms ?? 0) / 1000).toFixed(0)}s`;
    case 'noise_discarded':
      return `noise discarded (${e.words ?? 0} words, ${e.duration_s}s)`;
    case 'latency_breach':
      return (e.breaches || []).join('; ');
    case 'playback_timeout':
      return `playback ack never came — floor reopened after ${e.waited_s}s`;
    case 'spoke':
      return `[${e.move}] "${e.text ?? ''}"`;
    default:
      return e.kind;
  }
}

function MiniBar({ label, value, threshold, max, unit = '' }) {
  const pct = Math.min((value / max) * 100, 100);
  const thrPct = threshold != null ? Math.min((threshold / max) * 100, 100) : null;
  const over = threshold != null && value >= threshold;
  return (
    <div className="dbg-bar">
      <span className="dbg-bar-label">{label}</span>
      <div className="dbg-bar-track">
        <div className={`dbg-bar-fill ${over ? 'over' : ''}`} style={{ width: `${pct}%` }} />
        {thrPct != null && <div className="dbg-bar-thr" style={{ left: `${thrPct}%` }} />}
      </div>
      <span className="dbg-bar-val tabular">{value?.toFixed?.(2) ?? value}{unit}</span>
    </div>
  );
}

export default function DebugDrawer({ vad, events, latency, state }) {
  const v = vad?.vad || {};
  const ep = vad?.endpointer || {};

  return (
    <div className="debug-drawer">
      <div className="dbg-col">
        <div className="dbg-col-header">
          <span className="dbg-dot dot-green" />
          <h4>Acoustic VAD Engine</h4>
        </div>
        {v.score !== undefined ? (
          <>
            <MiniBar label="Voice Probability" value={v.score} threshold={0.55} max={1} />
            <MiniBar label="RMS Energy" value={(v.rms ?? 0) * 100} threshold={(v.gate ?? 0) * 100} max={8} />
            {v.harmonicity !== undefined && (
              <MiniBar label="Harmonicity" value={v.harmonicity} threshold={0.35} max={1} />
            )}
            <div className="dbg-kv">
              <span>Endpoint Trigger</span>
              <span className="tabular dbg-highlight">
                {ep.in_speech
                  ? `${Math.max(0, (ep.silence_needed_ms ?? 700) - (ep.silence_run_ms ?? 0))}ms silence`
                  : state === 'listening' ? 'Awaiting onset' : '—'}
              </span>
            </div>
            <div className="dbg-kv">
              <span>Speech Duration</span>
              <span className="tabular">{((ep.speech_ms ?? 0) / 1000).toFixed(1)}s
                {' / '}{((ep.min_utterance_ms ?? 600) / 1000).toFixed(1)}s min</span>
            </div>
          </>
        ) : (
          <p className="rail-note">Waiting for live audio frames...</p>
        )}
      </div>

      <div className="dbg-col dbg-events">
        <div className="dbg-col-header">
          <span className="dbg-dot dot-purple" />
          <h4>Decision Log & State Bus</h4>
        </div>
        <div className="dbg-log">
          {events.length === 0 && <p className="rail-note">Telemetry stream standby.</p>}
          {events.slice(-16).reverse().map((e) => (
            <div className="dbg-event" key={e.seq} data-tone={KIND_TONE[e.kind] || 'dim'}>
              <span className="dbg-time tabular">{(e.t_ms / 1000).toFixed(1)}s</span>
              <span className="dbg-text">{fmtEvent(e)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="dbg-col">
        <div className="dbg-col-header">
          <span className="dbg-dot dot-blue" />
          <h4>Latency Waterfall</h4>
        </div>
        {latency ? (
          <>
            <div className="dbg-kv">
              <span>Turn Budget</span>
              <span className={`tabular ${latency.within_budget ? 'good-text' : 'over-text'}`}>
                {latency.total_ms}ms / {latency.budget_ms}ms
              </span>
            </div>
            <div className="dbg-kv">
              <span>Perceived End-to-End</span>
              <span className="tabular dbg-highlight">{latency.perceived_ms}ms</span>
            </div>
            {Object.entries(latency.stages).map(([k, val]) => (
              <div className="dbg-kv" key={k}>
                <span>{k.replaceAll('_', ' ')}</span>
                <span className="tabular">{val}ms</span>
              </div>
            ))}
          </>
        ) : (
          <p className="rail-note">No turn evaluated yet.</p>
        )}
      </div>
    </div>
  );
}
