import { useCallback, useEffect, useRef, useState } from 'react';
import { AudioCapture, requestMedia } from '../lib/audio';
import { InterviewSocket } from '../lib/session';
import { SpeechPlayer } from '../lib/tts';
import DebugDrawer from './DebugDrawer';
import Readouts from './Readouts';

const PHASE_COPY = {
  connecting: 'CONNECTING SYSTEM',
  speaking: 'INTERVIEWER SPEAKING',
  thinking: 'PROCESSING RESPONSE',
  listening: 'YOUR TURN TO ANSWER',
  ended: 'SIMULATION COMPLETE',
};

const TRACK_NAMES = {
  dsa: 'Algorithms & Structures',
  system_design: 'System Architecture',
  behavioral: 'Behavioural & Culture',
  jd: 'Targeted Job Spec',
};

const LEVEL_BARS = 24;

export default function Interview({ config, sessionId, onReport, onFatal }) {
  const videoRef = useRef(null);
  const captureRef = useRef(null);
  const visionRef = useRef(null);
  const socketRef = useRef(null);
  const playerRef = useRef(null);
  const streamRef = useRef(null);
  const answerStartRef = useRef(null);
  const cvWindowRef = useRef([]);

  const [phase, setPhase] = useState('connecting');
  const [question, setQuestion] = useState('');
  const [questionKey, setQuestionKey] = useState(0);
  const [partial, setPartial] = useState('');
  const [levelBars, setLevelBars] = useState(() => new Array(LEVEL_BARS).fill(0.05));
  const [meters, setMeters] = useState({ wpm: null, fillers: 0, eye: 0.7, posture: 0.7 });
  const [hasVision, setHasVision] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [latency, setLatency] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [vadLive, setVadLive] = useState(null);
  const [traceEvents, setTraceEvents] = useState([]);
  const [ending, setEnding] = useState(false);
  const [simulatedAsr, setSimulatedAsr] = useState(false);
  const [answered, setAnswered] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [budget, setBudget] = useState(12);

  const recognizerRef = useRef(null);

  useEffect(() => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      try {
        const rec = new SpeechRec();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-US';
        rec.onresult = (event) => {
          let str = '';
          for (let i = 0; i < event.results.length; ++i) {
            str += event.results[i][0].transcript;
          }
          if (str.trim()) {
            setPartial(str.trim());
            const words = str.trim().split(/\s+/).filter(Boolean).length;
            const secs = answerStartRef.current
              ? (performance.now() - answerStartRef.current) / 1000
              : 0;
            setMeters((m) => ({
              ...m,
              wpm: secs > 2.0 ? (words / secs) * 60 : null,
              fillers: countFillers(str),
            }));
          }
        };
        rec.onerror = () => {};
        recognizerRef.current = rec;
      } catch (err) {
        console.warn('[speech-rec] client recognizer unavailable:', err);
      }
    }
    return () => {
      try {
        recognizerRef.current?.abort();
      } catch {}
    };
  }, []);

  useEffect(() => {
    if (phase === 'listening') {
      try {
        recognizerRef.current?.start();
      } catch {}
    } else {
      try {
        recognizerRef.current?.stop();
      } catch {}
    }
  }, [phase]);

  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    socketRef.current?.sendJSON({ type: 'debug', enabled: showDebug });
  }, [showDebug]);

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'ready':
        if (msg.question_budget) setBudget(msg.question_budget);
        setSimulatedAsr(msg.backends?.asr === 'mock');
        break;

      case 'state':
        if (msg.state === 'listening') {
          answerStartRef.current = performance.now();
          setPartial('');
          setMeters((m) => ({ ...m, wpm: null, fillers: 0 }));
          visionRef.current?.markTurn?.();
        }
        setPhase(msg.state);
        break;

      case 'speech':
        playerRef.current?.enqueue(msg.chunk);
        setQuestion((q) => (q ? `${q} ${msg.chunk.text}` : msg.chunk.text));
        break;

      case 'speech_end':
        setQuestion(msg.text);
        playerRef.current?.endTurn();
        break;

      case 'transcript': {
        setPartial(msg.text);
        const words = msg.text.trim().split(/\s+/).filter(Boolean).length;
        const secs = answerStartRef.current
          ? (performance.now() - answerStartRef.current) / 1000
          : 0;
        setMeters((m) => ({
          ...m,
          wpm: secs > 2.5 ? (words / secs) * 60 : null,
          fillers: countFillers(msg.text),
        }));
        break;
      }

      case 'turn_scored':
        setAnswered((n) => n + 1);
        setQuestionKey((k) => k + 1);
        setQuestion('');
        break;

      case 'noise_discarded':
        setPartial('');
        break;

      case 'latency':
        setLatency(msg.timings);
        break;

      case 'vad':
        setVadLive(msg);
        break;

      case 'trace':
        setTraceEvents((prev) => [...prev.slice(-59), msg.event]);
        break;

      case 'trace_backfill':
        setTraceEvents(msg.events || []);
        break;

      case 'report':
        onReport(msg.report);
        break;

      default:
        break;
    }
  }, [onReport]);

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      let stream;
      try {
        stream = await requestMedia({ video: config.useCamera });
      } catch {
        onFatal('Rehearsal requires audio device permissions to conduct an interview.');
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current && config.useCamera) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }

      playerRef.current = new SpeechPlayer({
        onDone: () => socketRef.current?.playbackDone(),
      });

      const socket = new InterviewSocket({
        sessionId,
        track: config.track,
        jdText: config.jdText,
        handlers: {
          onMessage: handleMessage,
          onError: () => onFatal('Lost the connection to the interview server.'),
        },
      });
      socketRef.current = socket;
      socket.connect();

      const capture = new AudioCapture({
        onChunk: (buf) => socket.sendAudio(buf),
        onLevel: (v) =>
          setLevelBars((bars) => {
            const next = bars.slice(1);
            next.push(Math.max(0.04, v));
            return next;
          }),
      });
      captureRef.current = capture;
      await capture.start(stream);

      if (config.useCamera) {
        const { VisionAnalyzer } = await import('../lib/vision');
        if (cancelled) return;
        const vision = new VisionAnalyzer({
          onCalibrated: () => setCalibrating(false),
          onSample: (sample) => {
            socket.sendJSON(sample);
            if (sample.calibrating) return;
            const w = cvWindowRef.current;
            w.push(sample);
            if (w.length > 15) w.shift();
            const seen = w.filter((s) => s.face_present);
            if (seen.length) {
              setMeters((m) => ({
                ...m,
                eye: seen.filter((s) => s.eye_contact).length / seen.length,
                posture: seen.reduce((a, s) => a + s.posture, 0) / seen.length,
              }));
            }
          },
        });
        visionRef.current = vision;
        if (await vision.init()) {
          if (cancelled) return;
          setHasVision(true);
          setCalibrating(true);
          vision.start(videoRef.current);
        }
      }
    }

    boot();
    return () => {
      cancelled = true;
      playerRef.current?.cancel();
      visionRef.current?.close();
      captureRef.current?.stop();
      socketRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [config, sessionId, handleMessage, onFatal]);

  const end = () => {
    setEnding(true);
    playerRef.current?.cancel();
    socketRef.current?.end();
  };

  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;

  return (
    <div className="interview">
      <header className="topbar">
        <div className="topbar-left">
          <div className="brand-pill">
            <span className="brand-dot" />
            <span className="brand-name">REHEARSAL LAB</span>
          </div>
          <span className="track-chip">{TRACK_NAMES[config.track] || config.track}</span>
          <div className="progress-dots" aria-label={`${answered} answers given`}>
            {Array.from({ length: budget }, (_, i) => (
              <span
                key={i}
                className="dot"
                data-done={i < answered}
                data-current={i === answered}
              />
            ))}
          </div>
        </div>

        <div className="topbar-right">
          <div className="timer-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            <span className="elapsed tabular">{mmss}</span>
          </div>
        </div>
      </header>

      <main className="stage">
        <div className="center-stage">
          <div className={`speaker-card ${phase === 'speaking' ? 'speaking-active' : ''}`}>
            <div className="speaker-header">
              <div className="speaker-avatar-wrap">
                <div className={`ai-orb ${phase === 'speaking' ? 'ai-orb-pulsing' : ''}`}>
                  <div className="ai-orb-core" />
                  <div className="ai-orb-ring" />
                </div>
                <span className="speaker-title">AI INTERVIEWER</span>
              </div>
              <div className={`phase-tag phase-${phase}`}>
                <span className="phase-indicator-dot" />
                <span>{PHASE_COPY[phase] || phase}</span>
              </div>
            </div>

            <div className="question-wrapper">
              <p className="question question-enter" key={questionKey}>
                {question || (phase === 'connecting' ? 'Calibrating neural prompt & question context...' : 'Waiting for prompt...')}
              </p>
            </div>
          </div>

          <div className={`answer-card ${phase === 'listening' ? 'answer-active' : ''}`}>
            <div className="answer-header">
              <div className="candidate-badge">
                <div className="candidate-dot" />
                <span>CANDIDATE AUDIO FEED</span>
              </div>
              {simulatedAsr && <span className="sim-tag">TRANSCRIPTION EMULATED</span>}
            </div>

            <div className="answer-body">
              <p
                className={`answer-text ${partial ? 'live' : 'waiting'}`}
                aria-live="polite"
              >
                {partial || (phase === 'listening'
                  ? 'Listening to your microphone...'
                  : phase === 'thinking'
                    ? 'Synthesizing response & evaluating delivery metrics...'
                    : 'Awaiting interviewer question...')}
                {phase === 'listening' && <span className="caret" />}
              </p>
            </div>

            {simulatedAsr && (partial || phase === 'listening') && (
              <div className="sim-note">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                <span>Offline ASR emulator active. Words shown are generated mock tokens.</span>
              </div>
            )}
          </div>
        </div>

        <aside className="rail">
          <div className="camera-viewfinder">
            {config.useCamera ? (
              <div className="camera-wrap">
                <div className="hud-corner top-left" />
                <div className="hud-corner top-right" />
                <div className="hud-corner bottom-left" />
                <div className="hud-corner bottom-right" />
                <video ref={videoRef} muted playsInline />
                <span className="camera-badge">
                  <span className="cam-live-indicator" /> CLIENT CV
                </span>
              </div>
            ) : (
              <div className="camera-off">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="2" y1="2" x2="22" y2="22"/>
                  <path d="M10.41 10.41a2 2 0 1 0 2.83 2.83"/><path d="M13.5 17.5 18 21V7.5"/>
                  <path d="M6.75 6.75H4a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h10a2 2 0 0 0 1.25-.44"/>
                </svg>
                <span>Camera Offline</span>
              </div>
            )}
          </div>

          <Readouts
            wpm={meters.wpm}
            fillers={meters.fillers}
            eyeContact={meters.eye}
            posture={meters.posture}
            levelBars={levelBars}
            listening={phase === 'listening'}
            hasVision={hasVision}
            calibrating={calibrating}
          />
        </aside>
      </main>

      <footer className="bottombar-wrap">
        {showDebug && (
          <DebugDrawer vad={vadLive} events={traceEvents} latency={latency} state={phase} />
        )}
        <div className="bottombar">
          <button
            type="button"
            className="btn btn-danger-glow"
            onClick={end}
            disabled={ending}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 9 6 6"/><path d="m15 9-6 6"/>
            </svg>
            <span>{ending ? 'Generating Comprehensive Report...' : 'Conclude Session'}</span>
          </button>

          <button
            type="button"
            className={`btn-instruments ${showDebug ? 'active' : ''}`}
            onClick={() => setShowDebug((s) => !s)}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            <span>{showDebug ? 'Hide Diagnostics' : 'Telemetry Cockpit'}</span>
          </button>
        </div>
      </footer>
    </div>
  );
}

const LIVE_FILLERS =
  /\b(um+|uh+|erm?|hmm+|you know|i mean|sort of|kind of|matlab|yaar|basically)\b/gi;
function countFillers(text) {
  return (text.match(LIVE_FILLERS) || []).length;
}
