/**
 * In-browser CV: MediaPipe FaceLandmarker + PoseLandmarker.
 *
 * This module is the entire vision system. Raw pixels never leave this file --
 * what goes to the server is a ~200 byte JSON sample at 5Hz. That is the whole
 * privacy argument, and it holds only because the inference happens here.
 *
 * Three things this gets right that a naive landmark reader does not:
 *
 * 1. GAZE IS HEAD POSE + EYE-IN-SOCKET, NOT EYE-IN-SOCKET ALONE.
 *    Iris displacement measured against the eye corners tells you where the
 *    eyes point *within the skull*. Someone who turns their head 20 degrees
 *    and keeps looking at you has strongly off-centre irises -- and a naive
 *    detector marks them as avoidant at the exact moment they're most engaged.
 *    We recover head yaw/pitch from the facial transformation matrix and
 *    combine the two into a real gaze direction.
 *
 * 2. EVERYTHING IS CALIBRATED PER PERSON.
 *    Eye aspect ratio at rest varies enormously with eye shape, glasses and
 *    camera angle; a fixed blink threshold either misses every blink or counts
 *    three per blink. Neutral gaze varies with where the webcam sits relative
 *    to the screen -- a camera below the display makes everyone look downward
 *    forever. We spend the first few seconds learning both, and report low
 *    confidence until we have.
 *
 * 3. IT DISTINGUISHES "NOT MEASURED" FROM "MEASURED AS ZERO".
 *    A dropped frame, a face out of shot, and a face held still are three
 *    different things. Conflating them means penalising people for their wifi.
 */

import { FilesetResolver, FaceLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// FaceMesh indices (478-point model with iris refinement).
const L_IRIS = 468, R_IRIS = 473;
const L_EYE_OUT = 33, L_EYE_IN = 133;
const R_EYE_IN = 362, R_EYE_OUT = 263;
const L_LID_UP = 159, L_LID_LO = 145;
const R_LID_UP = 386, R_LID_LO = 374;

// Pose indices.
const P_NOSE = 0, P_L_SHOULDER = 11, P_R_SHOULDER = 12, P_L_EAR = 7, P_R_EAR = 8;

const CALIBRATION_SAMPLES = 30;   // ~3s of face frames at 10Hz
const GAZE_TOLERANCE = 0.30;      // combined gaze offset counted as "at camera"

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp01 = (v) => Math.max(0, Math.min(1, v));

/** Rolling median — robust to the occasional wild landmark frame. */
function median(values) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Euler angles from MediaPipe's 4x4 facial transformation matrix.
 * The matrix arrives column-major, so element (row, col) is data[row + col*4].
 */
function headPoseFromMatrix(data) {
  if (!data || data.length < 16) return null;
  const m = (r, c) => data[r + c * 4];
  const sy = Math.hypot(m(0, 0), m(1, 0));
  if (sy < 1e-6) return null;
  return {
    pitch: Math.atan2(m(2, 1), m(2, 2)),   // nod
    yaw: Math.atan2(-m(2, 0), sy),          // shake
    roll: Math.atan2(m(1, 0), m(0, 0)),     // tilt
  };
}

/** Fallback head yaw from landmark geometry, when no matrix is available. */
function headYawFromLandmarks(lm) {
  const l = lm[L_EYE_OUT], r = lm[R_EYE_OUT], nose = lm[1];
  if (!l || !r || !nose) return 0;
  const span = Math.abs(r.x - l.x);
  if (span < 1e-6) return 0;
  // Nose sits midway between the outer eye corners when facing forward; it
  // shifts toward whichever side the head turns away from.
  const centre = (l.x + r.x) / 2;
  return clamp01(Math.abs(nose.x - centre) / span) * Math.sign(nose.x - centre) * 1.4;
}

/** Iris displacement within the eye socket, normalised by eye width. */
function irisOffset(lm, iris, cornerA, cornerB) {
  const a = lm[cornerA], b = lm[cornerB], i = lm[iris];
  if (!a || !b || !i) return { x: 0, y: 0 };
  const width = dist(a, b);
  if (width < 1e-6) return { x: 0, y: 0 };
  const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
  // Vertical travel is roughly half of horizontal in a human eye; without
  // rescaling, the vertical term dominates and everyone reads as averted.
  return { x: (i.x - cx) / width, y: ((i.y - cy) / width) * 0.5 };
}

function eyeAspectRatio(lm, up, lo, cornerA, cornerB) {
  const w = dist(lm[cornerA], lm[cornerB]);
  if (w < 1e-6) return 1;
  return dist(lm[up], lm[lo]) / w;
}

export class VisionAnalyzer {
  constructor({ onSample, onCalibrated, sampleRateHz = 5 } = {}) {
    this.onSample = onSample;
    this.onCalibrated = onCalibrated;
    this.samplePeriodMs = 1000 / sampleRateHz;

    this.faceLandmarker = null;
    this.poseLandmarker = null;
    this.running = false;
    this.available = false;
    this.error = null;
    this.calibrated = false;

    this._raf = null;
    this._t0 = 0;
    this._lastFaceMs = 0;
    this._lastPoseMs = 0;
    this._lastEmitMs = 0;

    // Calibration buffers.
    this._calEar = [];
    this._calGazeX = [];
    this._calGazeY = [];
    this._earOpen = null;
    this._gazeOrigin = { x: 0, y: 0 };

    this._acc = this._freshAcc();
    this._posture = 0.7;
    this._blinkArmed = true;
    this._lastPose = null;
  }

  _freshAcc() {
    return {
      frames: 0, faceSeen: 0, eyeContact: 0, gazeSum: 0,
      blinks: 0, moveSum: 0, quality: 0,
    };
  }

  async init() {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        // Required for head-pose compensation. Without it we fall back to a
        // landmark-geometry estimate, which is noisier but still far better
        // than ignoring head pose entirely.
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false,
      });
      this.poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
      this.available = true;
    } catch (err) {
      // Offline, blocked CDN, or no WebGL. The interview must still run --
      // delivery scoring degrades to audio-only, and the server is told there
      // is no vision data rather than being fed zeros.
      this.error = err?.message || String(err);
      this.available = false;
      console.warn('[vision] unavailable, continuing audio-only:', this.error);
    }
    return this.available;
  }

  start(videoEl) {
    if (!this.available || this.running) return;
    this.running = true;
    this._t0 = performance.now();
    const loop = () => {
      if (!this.running) return;
      this._tick(videoEl);
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  close() {
    this.stop();
    this.faceLandmarker?.close?.();
    this.poseLandmarker?.close?.();
    this.faceLandmarker = null;
    this.poseLandmarker = null;
  }

  /** Called when a new answer begins, so posture drift is per-turn. */
  markTurn() {
    this._acc = this._freshAcc();
  }

  _tick(videoEl) {
    if (!videoEl || videoEl.readyState < 2) return;
    const now = performance.now();
    // Face at ~10Hz: gaze and blink are fast signals and under-sampling misses
    // blinks entirely. Pose at ~3Hz: posture moves over seconds and pose
    // inference costs roughly 3x face. Running both at 30fps pegs the CPU and
    // starves the audio thread, which surfaces as turn-taking latency.
    if (now - this._lastFaceMs >= 100) {
      this._lastFaceMs = now;
      this._processFace(videoEl, now);
    }
    if (now - this._lastPoseMs >= 333) {
      this._lastPoseMs = now;
      this._processPose(videoEl, now);
    }
    if (now - this._lastEmitMs >= this.samplePeriodMs) {
      this._lastEmitMs = now;
      this._emit(now);
    }
  }

  _processFace(videoEl, now) {
    let result;
    try {
      result = this.faceLandmarker.detectForVideo(videoEl, now);
    } catch {
      return;
    }
    this._acc.frames += 1;

    const faces = result?.faceLandmarks;
    if (!faces?.length || !faces[0][R_IRIS]) return;
    const lm = faces[0];
    this._acc.faceSeen += 1;

    // --- head pose -------------------------------------------------------
    const matrix = result.facialTransformationMatrixes?.[0]?.data;
    const pose = headPoseFromMatrix(matrix) || {
      yaw: headYawFromLandmarks(lm), pitch: 0, roll: 0,
    };

    // --- gaze: eye-in-socket combined with where the head points ----------
    const left = irisOffset(lm, L_IRIS, L_EYE_OUT, L_EYE_IN);
    const right = irisOffset(lm, R_IRIS, R_EYE_IN, R_EYE_OUT);
    const eyeX = (left.x + right.x) / 2;
    const eyeY = (left.y + right.y) / 2;

    // Head rotation contributes to where they're actually looking. The signs
    // oppose: turning the head right while the eyes stay on target pushes the
    // irises left in their sockets, and the two should cancel to "on camera".
    const HEAD_GAIN = 0.55;   // radians -> normalised offset units
    const gazeX = eyeX + pose.yaw * HEAD_GAIN;
    const gazeY = eyeY + pose.pitch * HEAD_GAIN * 0.6;

    // --- blink, against a per-person open-eye baseline ---------------------
    const ear =
      (eyeAspectRatio(lm, L_LID_UP, L_LID_LO, L_EYE_OUT, L_EYE_IN) +
        eyeAspectRatio(lm, R_LID_UP, R_LID_LO, R_EYE_IN, R_EYE_OUT)) / 2;

    if (!this.calibrated) {
      this._calEar.push(ear);
      this._calGazeX.push(gazeX);
      this._calGazeY.push(gazeY);
      if (this._calEar.length >= CALIBRATION_SAMPLES) {
        // Median, not mean: a blink during calibration would drag a mean down
        // and permanently raise the blink threshold.
        this._earOpen = median(this._calEar);
        this._gazeOrigin = { x: median(this._calGazeX), y: median(this._calGazeY) };
        this.calibrated = true;
        this.onCalibrated?.();
      }
      return;   // don't score during calibration
    }

    const blinkThreshold = this._earOpen * 0.68;
    // Edge-triggered: a blink is the transition into closure, not every frame
    // the eye happens to be shut. Level-triggering counts one slow blink as
    // three and inflates the rate ~3x.
    if (ear < blinkThreshold) {
      if (this._blinkArmed) {
        this._acc.blinks += 1;
        this._blinkArmed = false;
      }
    } else if (ear > blinkThreshold * 1.15) {
      this._blinkArmed = true;   // hysteresis, so a flicker isn't two blinks
    }

    // Offset from THIS person's neutral, which absorbs webcam placement.
    const dx = gazeX - this._gazeOrigin.x;
    const dy = gazeY - this._gazeOrigin.y;
    const offset = Math.hypot(dx, dy);
    this._acc.gazeSum += Math.min(offset, 1);
    if (offset < GAZE_TOLERANCE) this._acc.eyeContact += 1;

    // --- movement, from head rotation rather than a single landmark --------
    if (this._lastPose) {
      this._acc.moveSum +=
        Math.abs(pose.yaw - this._lastPose.yaw) +
        Math.abs(pose.pitch - this._lastPose.pitch) +
        Math.abs(pose.roll - this._lastPose.roll) * 0.5;
    }
    this._lastPose = pose;

    // Framing quality: a face filling too little of the frame gives unreliable
    // iris landmarks, and we'd rather say so than emit confident noise.
    const faceWidth = dist(lm[L_EYE_OUT], lm[R_EYE_OUT]);
    this._acc.quality += clamp01((faceWidth - 0.06) / 0.10);
  }

  _processPose(videoEl, now) {
    let result;
    try {
      result = this.poseLandmarker.detectForVideo(videoEl, now);
    } catch {
      return;
    }
    const lm = result?.landmarks?.[0];
    if (!lm) return;
    const ls = lm[P_L_SHOULDER], rs = lm[P_R_SHOULDER], nose = lm[P_NOSE];
    if (!ls || !rs || !nose) return;

    // Visibility guard: MediaPipe happily invents off-screen shoulders, and a
    // hallucinated shoulder produces a confident, meaningless posture score.
    const visible = (p) => p.visibility === undefined || p.visibility > 0.5;
    if (!visible(ls) || !visible(rs)) return;

    const tilt = Math.abs(ls.y - rs.y);
    const shoulderY = (ls.y + rs.y) / 2;
    const headroom = shoulderY - nose.y;           // larger = sitting taller

    // Forward lean (slouching toward the screen) shows up as the ears moving
    // ahead of the shoulder line.
    const ears = [lm[P_L_EAR], lm[P_R_EAR]].filter(Boolean);
    const earY = ears.length ? ears.reduce((a, e) => a + e.y, 0) / ears.length : nose.y;
    const neckExtension = clamp01((shoulderY - earY - 0.10) / 0.18);

    const level = Math.max(0, 1 - tilt * 6);
    const upright = clamp01((headroom - 0.14) / 0.16);
    const instant = 0.35 * level + 0.40 * upright + 0.25 * neckExtension;

    // Posture is a slow signal; raw per-frame pose is jittery.
    this._posture = this._posture * 0.7 + instant * 0.3;
  }

  _emit(now) {
    const a = this._acc;
    const seen = a.faceSeen;
    const facePresent = seen > 0;
    // During calibration we emit face_present:false so the server treats the
    // window as "no vision data" rather than as a measured result. Reporting
    // uncalibrated gaze would put a systematic bias into the first answer's
    // presence score, which is exactly the answer people are most nervous on.
    const usable = facePresent && this.calibrated;

    const sample = {
      type: 'cv',
      ts_ms: Math.round(now - this._t0),
      face_present: usable,
      calibrating: facePresent && !this.calibrated,
      eye_contact: usable ? a.eyeContact / seen >= 0.5 : false,
      gaze_offset: usable ? Math.min(a.gazeSum / seen, 1) : 0,
      posture: usable ? Number(this._posture.toFixed(3)) : 0,
      blink: a.blinks > 0,
      // Scaled so a normal amount of head movement lands mid-range.
      head_movement: usable ? Math.min((a.moveSum / seen) * 3.0, 1) : 0,
      quality: usable ? Number((a.quality / seen).toFixed(2)) : 0,
    };
    this._acc = this._freshAcc();
    this.onSample?.(sample);
  }
}
