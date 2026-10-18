import { Suspense, lazy, useCallback, useEffect, useState } from 'react';
import Interview from './components/Interview';
import Preflight from './components/Preflight';
import Setup from './components/Setup';
import { createSession } from './lib/session';

// Both are only reachable after an interview (or from the setup screen), by
// which point we've had plenty of idle network time to fetch them.
const Report = lazy(() => import('./components/Report'));
const Progress = lazy(() => import('./components/Progress'));

export default function App() {
  const [stage, setStage] = useState('setup');
  const [config, setConfig] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  // Warn before a refresh or tab close mid-interview: the session is in
  // memory on the server and a reload loses the answers already given.
  useEffect(() => {
    if (stage !== 'interview') return undefined;
    const guard = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [stage]);

  const chooseConfig = useCallback((cfg) => {
    setError(null);
    setConfig(cfg);
    setStage('preflight');
  }, []);

  // The session is created only after the device check passes, so a failed
  // preflight never leaves an abandoned session against the rate limit.
  const beginInterview = useCallback(async () => {
    try {
      const { session_id } = await createSession(config.track, config.jdText);
      setSessionId(session_id);
      setStage('interview');
    } catch (err) {
      setError(err.message);
      setStage('setup');
    }
  }, [config]);

  const onReport = useCallback((r) => { setReport(r); setStage('report'); }, []);

  const onFatal = useCallback((message) => {
    setError(message);
    setStage('setup');
  }, []);

  const restart = useCallback(() => {
    setReport(null);
    setSessionId(null);
    setStage('setup');
  }, []);

  const openReport = useCallback(async (id) => {
    try {
      const res = await fetch(`/api/sessions/${id}/report`);
      if (!res.ok) throw new Error('that report is no longer available');
      setReport(await res.json());
      setStage('report');
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const fallback = <p className="rail-note" style={{ padding: 72 }}>Loading…</p>;

  return (
    <div className="app">
      {stage === 'setup' && (
        <Setup
          onStart={chooseConfig}
          onViewProgress={() => setStage('progress')}
          error={error}
        />
      )}

      {stage === 'preflight' && (
        <Preflight config={config} onReady={beginInterview} onBack={() => setStage('setup')} />
      )}

      {stage === 'interview' && (
        <Interview
          config={config}
          sessionId={sessionId}
          onReport={onReport}
          onFatal={onFatal}
        />
      )}

      {stage === 'report' && (
        <Suspense fallback={fallback}>
          <Report
            report={report}
            onRestart={restart}
            onViewProgress={() => setStage('progress')}
          />
        </Suspense>
      )}

      {stage === 'progress' && (
        <Suspense fallback={fallback}>
          <Progress onBack={restart} onOpenReport={openReport} />
        </Suspense>
      )}
    </div>
  );
}
