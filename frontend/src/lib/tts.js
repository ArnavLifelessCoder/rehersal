/**
 * Interviewer voice playback.
 *
 * Handles both server modes: a base64 audio chunk (Piper) or a bare sentence to
 * synthesise locally (Web Speech API). Either way the queue is the important
 * part -- sentences arrive faster than they can be spoken, so they must be
 * serialised, and `playback_done` must fire exactly once when the LAST sentence
 * of a turn finishes. Firing it per-sentence would hand the floor back to the
 * candidate mid-question.
 */

export class SpeechPlayer {
  constructor({ onStart, onDone } = {}) {
    this.onStart = onStart;
    this.onDone = onDone;
    this.queue = [];
    this.speaking = false;
    this.turnOpen = false;
    this.voice = null;
    this.audio = null;
    this._pickVoice();
  }

  _pickVoice() {
    if (typeof speechSynthesis === 'undefined') return;
    const choose = () => {
      const voices = speechSynthesis.getVoices();
      if (!voices.length) return;
      // Prefer a natural-sounding English voice; fall back to any English one.
      this.voice =
        voices.find((v) => /en-(US|GB)/.test(v.lang) && /natural|neural|google/i.test(v.name)) ||
        voices.find((v) => v.lang?.startsWith('en')) ||
        voices[0];
    };
    choose();
    speechSynthesis.onvoiceschanged = choose;
  }

  get supported() {
    return typeof speechSynthesis !== 'undefined';
  }

  /** Called for each streamed sentence from the server. */
  enqueue(chunk) {
    this.turnOpen = true;
    this.queue.push(chunk);
    if (!this.speaking) this._drain();
  }

  /** Called when the server signals the turn's text is complete. */
  endTurn() {
    this.turnOpen = false;
    if (!this.speaking && this.queue.length === 0) this._finish();
  }

  async _drain() {
    if (this.speaking) return;
    this.speaking = true;
    if (this.onStart) this.onStart();

    while (this.queue.length > 0) {
      const chunk = this.queue.shift();
      try {
        await this._speak(chunk);
      } catch (err) {
        console.warn('[tts] chunk failed, continuing:', err);
      }
    }

    this.speaking = false;
    // Only hand the floor back if the server has finished sending this turn.
    if (!this.turnOpen) this._finish();
  }

  _finish() {
    if (this.onDone) this.onDone();
  }

  _speak(chunk) {
    if (chunk.audio) return this._playAudio(chunk.audio, chunk.mime);
    return this._synthesise(chunk.text);
  }

  _playAudio(b64, mime) {
    return new Promise((resolve, reject) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], { type: mime || 'audio/wav' }));
      const audio = new Audio(url);
      this.audio = audio;
      const cleanup = () => {
        URL.revokeObjectURL(url);
        this.audio = null;
      };
      audio.onended = () => { cleanup(); resolve(); };
      audio.onerror = (e) => { cleanup(); reject(e); };
      audio.play().catch(reject);
    });
  }

  _synthesise(text) {
    return new Promise((resolve) => {
      if (!this.supported) {
        // No speech synthesis: pace the text visually so the transcript still
        // reads at a human rate instead of the whole question flashing at once.
        setTimeout(resolve, Math.min(text.split(/\s+/).length * 350, 8000));
        return;
      }
      const utter = new SpeechSynthesisUtterance(text);
      if (this.voice) utter.voice = this.voice;
      utter.rate = 1.02;
      utter.pitch = 1.0;
      let settled = false;
      const done = () => { if (!settled) { settled = true; resolve(); } };
      utter.onend = done;
      utter.onerror = done;
      // Chrome drops long utterances silently; this guarantees the queue moves.
      setTimeout(done, Math.min(text.split(/\s+/).length * 600 + 3000, 20000));
      speechSynthesis.speak(utter);
    });
  }

  cancel() {
    this.queue = [];
    this.turnOpen = false;
    this.speaking = false;
    if (this.supported) speechSynthesis.cancel();
    if (this.audio) { this.audio.pause(); this.audio = null; }
  }
}
