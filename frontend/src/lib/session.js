/**
 * WebSocket transport for a live interview.
 *
 * Applies backpressure on audio: if the socket's send buffer builds up (weak
 * uplink, or the server stalled), we drop the oldest frames rather than queue
 * them. Buffered audio is worthless -- by the time it arrives the turn has
 * moved on, and a growing buffer turns a 200ms hiccup into a permanently
 * lagging interview.
 */

const MAX_BUFFERED_BYTES = 256 * 1024;

export class InterviewSocket {
  constructor({ sessionId, track, jdText, handlers = {} }) {
    this.sessionId = sessionId;
    this.track = track;
    this.jdText = jdText;
    this.handlers = handlers;
    this.ws = null;
    this.closed = false;
    this.droppedFrames = 0;
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const params = new URLSearchParams({ track: this.track });
    if (this.jdText) params.set('jd_text', this.jdText);
    const url = `${proto}//${location.host}/ws/${this.sessionId}?${params}`;

    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => this.handlers.onOpen?.();
    this.ws.onclose = (e) => { if (!this.closed) this.handlers.onClose?.(e); };
    this.ws.onerror = (e) => this.handlers.onError?.(e);
    this.ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      this.handlers.onMessage?.(msg);
    };
  }

  get ready() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  sendAudio(buffer) {
    if (!this.ready) return;
    if (this.ws.bufferedAmount > MAX_BUFFERED_BYTES) {
      this.droppedFrames += 1;
      return;
    }
    this.ws.send(buffer);
  }

  sendJSON(obj) {
    if (!this.ready) return;
    this.ws.send(JSON.stringify(obj));
  }

  playbackDone() {
    this.sendJSON({ type: 'playback_done' });
  }

  end() {
    this.sendJSON({ type: 'end' });
  }

  close() {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
  }
}

export async function createSession(track, jdText) {
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ track, jd_text: jdText || null }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `could not start session (${res.status})`);
  }
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('backend unreachable');
  return res.json();
}
