/**
 * Microphone capture -> 16kHz mono int16 PCM.
 *
 * Uses an AudioWorklet rather than the deprecated ScriptProcessorNode: the
 * worklet runs on the audio render thread, so React re-renders and MediaPipe
 * inference on the main thread cannot cause dropouts or jitter in the frames
 * that endpointing depends on. Given the turn-taking budget is measured in
 * hundreds of milliseconds, main-thread audio would be a constant source of
 * phantom latency.
 *
 * Resampling is done here rather than server-side so we ship 32KB/s instead of
 * ~190KB/s of 48kHz float.
 */

const WORKLET_SOURCE = `
class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetRate;
    this.ratio = sampleRate / this.targetRate;
    this.buffer = [];
    // Emit ~40ms at a time: delivers audio swiftly to VAD and ASR
    // without paying excessive WebSocket framing overhead.
    this.chunkSize = Math.round(this.targetRate * 0.04);
    // Box-filter decimation state. Averaging every input sample that falls in
    // an output bin is a crude but real anti-alias filter -- nearest-neighbour
    // picking would fold everything above 8kHz back down into the speech band
    // and put artefacts straight into the ASR's mel spectrogram.
    this.acc = 0;
    this.sum = 0;
    this.count = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];

    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count += 1;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        this.buffer.push(this.count > 0 ? this.sum / this.count : 0);
        this.sum = 0;
        this.count = 0;
      }
    }

    while (this.buffer.length >= this.chunkSize) {
      const slice = this.buffer.splice(0, this.chunkSize);
      const pcm = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        const s = Math.max(-1, Math.min(1, slice[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-downsampler', PCMDownsampler);
`;

export class AudioCapture {
  constructor({ targetRate = 16000, onChunk, onLevel } = {}) {
    this.targetRate = targetRate;
    this.onChunk = onChunk;
    this.onLevel = onLevel;
    this.context = null;
    this.stream = null;
    this.node = null;
    this.analyser = null;
    this._levelTimer = null;
    this.muted = false;
  }

  async start(stream) {
    this.stream = stream;
    // Ask for the target rate directly -- when the browser honours it, its own
    // high-quality resampler does the work and our worklet ratio is 1.0. When
    // it doesn't (Safari historically), the worklet decimates as a fallback.
    try {
      this.context = new AudioContext({ sampleRate: this.targetRate });
    } catch {
      this.context = new AudioContext();
    }
    // Some browsers start the context suspended until a user gesture.
    if (this.context.state === 'suspended') await this.context.resume();

    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await this.context.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    const source = this.context.createMediaStreamSource(stream);
    this.node = new AudioWorkletNode(this.context, 'pcm-downsampler', {
      processorOptions: { targetRate: this.targetRate },
      numberOfInputs: 1,
      numberOfOutputs: 0,
    });
    this.node.port.onmessage = (event) => {
      if (!this.muted && this.onChunk) this.onChunk(event.data);
    };
    source.connect(this.node);

    // A separate analyser drives the UI level meter. Deriving it from the PCM
    // chunks instead would couple the visual refresh rate to the network path.
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 512;
    source.connect(this.analyser);
    const buf = new Float32Array(this.analyser.fftSize);
    this._levelTimer = setInterval(() => {
      if (!this.analyser || !this.onLevel) return;
      this.analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      this.onLevel(Math.min(Math.sqrt(sum / buf.length) * 4, 1));
    }, 60);
  }

  setMuted(muted) {
    this.muted = muted;
  }

  async stop() {
    if (this._levelTimer) clearInterval(this._levelTimer);
    this._levelTimer = null;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    this.analyser = null;
    if (this.context && this.context.state !== 'closed') await this.context.close();
    this.context = null;
  }
}

export async function requestMedia({ video = true } = {}) {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,   // stops the interviewer's TTS re-entering the mic
      noiseSuppression: true,
      autoGainControl: false,   // AGC would flatten the energy variance the
                                // confidence sub-score reads from
    },
    video: video ? { width: 640, height: 480, frameRate: 30 } : false,
  });
}
