// Nobility2 client: captures mic at 16kHz, streams PCM to /ws/session,
// plays back PersonaPlex's speech gaplessly, and draws AVTR-1's video.
// Protocol matches app.py's byte-tag scheme:
//   kind=1: audio PCM (float32 LE @24kHz)   kind=2: text token
//   kind=3: video frame (fmt 0=raw I420, 2=JPEG)   kind=4: error message

const statusEl = document.getElementById("status");
const canvas = document.getElementById("video");
const ctx = canvas.getContext("2d");
const transcriptEl = document.getElementById("transcript");
const startBtn = document.getElementById("start");

let audioCtx;
let ws;
let wantConnected = false;
let reconnectDelayMs = 1000;

// ---- Audio out: gapless scheduled playback ---------------------------------
// Naively calling source.start() on arrival overlaps/gaps chunks (crackle).
// Instead keep a running play cursor on the AudioContext clock and schedule
// each chunk back-to-back, with a tiny lead-in for jitter.
let playCursor = 0;
const AUDIO_LEAD_S = 0.06;

function playPcm(pcmFloat32) {
  const buffer = audioCtx.createBuffer(1, pcmFloat32.length, 24000);
  buffer.copyToChannel(pcmFloat32, 0);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (playCursor < now + 0.01) playCursor = now + AUDIO_LEAD_S;
  source.start(playCursor);
  playCursor += buffer.duration;
}

// ---- Video: decode off-main-thread, present on a rAF-paced queue -----------
// AVTR-1 yields frames in ~0.4s bursts; drawing on arrival looks stuttery.
// Queue decoded bitmaps and present at a steady 25fps, dropping the oldest
// frames if the queue backs up (latency guard beats slideshow).
const frameQueue = [];
const TARGET_FRAME_MS = 40; // 25fps
const MAX_QUEUE = 15;
let lastPresent = 0;

function presentLoop(ts) {
  if (frameQueue.length > 0 && ts - lastPresent >= TARGET_FRAME_MS - 1) {
    const bmp = frameQueue.shift();
    if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
      canvas.width = bmp.width;
      canvas.height = bmp.height;
    }
    ctx.drawImage(bmp, 0, 0);
    if (bmp.close) bmp.close();
    lastPresent = ts;
  }
  requestAnimationFrame(presentLoop);
}
requestAnimationFrame(presentLoop);

function enqueueBitmap(bmp) {
  frameQueue.push(bmp);
  while (frameQueue.length > MAX_QUEUE) {
    const dropped = frameQueue.shift();
    if (dropped.close) dropped.close();
  }
}

async function handleJpegFrame(bytes) {
  // Browser-native JPEG decode, off the main thread.
  const blob = new Blob([bytes], { type: "image/jpeg" });
  try {
    enqueueBitmap(await createImageBitmap(blob));
  } catch (e) {
    /* drop bad frame */
  }
}

function handleRawI420Frame(data, height, width) {
  // Fallback path (NOBILITY2_FRAME_CODEC=raw). ImageData is Uint8ClampedArray,
  // so out-of-range writes clamp for free -- no Math.min/max per channel.
  const ySize = width * height;
  const uSize = (width >> 1) * (height >> 1);
  const yPlane = data.subarray(0, ySize);
  const uPlane = data.subarray(ySize, ySize + uSize);
  const vPlane = data.subarray(ySize + uSize, ySize + 2 * uSize);
  const imageData = new ImageData(width, height);
  const out = imageData.data;
  const halfW = width >> 1;
  for (let row = 0; row < height; row++) {
    const yRow = row * width;
    const uvRow = (row >> 1) * halfW;
    for (let col = 0; col < width; col++) {
      const Y = yPlane[yRow + col];
      const uvIdx = uvRow + (col >> 1);
      const U = uPlane[uvIdx] - 128;
      const V = vPlane[uvIdx] - 128;
      const o = (yRow + col) * 4;
      out[o] = Y + 1.402 * V;
      out[o + 1] = Y - 0.344136 * U - 0.714136 * V;
      out[o + 2] = Y + 1.772 * U;
      out[o + 3] = 255;
    }
  }
  createImageBitmap(imageData).then(enqueueBitmap);
}

// ---- Session -----------------------------------------------------------------
function setStatus(text) {
  statusEl.textContent = text;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/session`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    reconnectDelayMs = 1000;
    setStatus("Connected — say hello!");
  };
  ws.onclose = () => {
    if (!wantConnected) return setStatus("Disconnected");
    setStatus("Connection lost — reconnecting...");
    setTimeout(connectWs, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 10000);
  };
  ws.onerror = () => {};

  ws.onmessage = (event) => {
    const bytes = new Uint8Array(event.data);
    const kind = bytes[0];
    if (kind === 1) {
      const pcm = new Float32Array(bytes.buffer, bytes.byteOffset + 1);
      playPcm(pcm);
    } else if (kind === 2) {
      transcriptEl.textContent += new TextDecoder().decode(bytes.subarray(1));
    } else if (kind === 3) {
      const fmt = bytes[1];
      const view = new DataView(bytes.buffer, bytes.byteOffset + 2, 8);
      const height = view.getUint32(0, true);
      const width = view.getUint32(4, true);
      const frameData = bytes.subarray(10);
      if (fmt === 2) handleJpegFrame(frameData);
      else handleRawI420Frame(frameData, height, width);
    } else if (kind === 4) {
      setStatus("Error: " + new TextDecoder().decode(bytes.subarray(1)));
    }
  };
}

async function start() {
  // Force the context clock to 16kHz: getUserMedia's sampleRate constraint is
  // only a hint most browsers ignore, and ScriptProcessor delivers PCM at the
  // CONTEXT rate. Without this the mic ships 48kHz mislabeled as 16kHz and
  // the model hears 3x-slowed audio. The context resamples both capture and
  // our 24kHz playback buffers internally.
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
  });

  wantConnected = true;
  connectWs();

  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  // Route through a muted gain node (not audioCtx.destination) so the
  // ScriptProcessorNode fires without echoing the mic back to speakers.
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioCtx.destination);

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    ws.send(input.slice().buffer);
  };
}

startBtn.addEventListener("click", () => {
  startBtn.disabled = true;
  setStatus("Connecting...");
  start().catch((err) => {
    startBtn.disabled = false;
    setStatus("Failed to start: " + err.message);
  });
});
