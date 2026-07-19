// Nobility2 client: captures mic at 16kHz, streams PCM to /ws/session,
// plays back PersonaPlex's speech, and decodes+draws AVTR-1's YUV I420
// video frames onto <canvas>. Protocol matches app.py's byte-tag scheme:
//   kind=1: audio PCM (float32 LE)   kind=2: text token   kind=3: video frame
//   kind=4: error message

const statusEl = document.getElementById("status");
const canvas = document.getElementById("video");
const ctx = canvas.getContext("2d");
const transcriptEl = document.getElementById("transcript");
const startBtn = document.getElementById("start");

let audioCtx;
let ws;

function setStatus(text) {
  statusEl.textContent = text;
}

function drawYuv420Frame(data, height, width) {
  const ySize = width * height;
  const uSize = (width / 2) * (height / 2);
  const yPlane = data.subarray(0, ySize);
  const uPlane = data.subarray(ySize, ySize + uSize);
  const vPlane = data.subarray(ySize + uSize, ySize + 2 * uSize);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const imageData = ctx.createImageData(width, height);
  const out = imageData.data;

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const yIdx = row * width + col;
      const uvRow = row >> 1, uvCol = col >> 1;
      const uvIdx = uvRow * (width >> 1) + uvCol;

      const Y = yPlane[yIdx];
      const U = uPlane[uvIdx] - 128;
      const V = vPlane[uvIdx] - 128;

      const r = Y + 1.402 * V;
      const g = Y - 0.344136 * U - 0.714136 * V;
      const b = Y + 1.772 * U;

      const outIdx = yIdx * 4;
      out[outIdx] = Math.max(0, Math.min(255, r));
      out[outIdx + 1] = Math.max(0, Math.min(255, g));
      out[outIdx + 2] = Math.max(0, Math.min(255, b));
      out[outIdx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function playPcm(pcmFloat32) {
  const buffer = audioCtx.createBuffer(1, pcmFloat32.length, 24000);
  buffer.copyToChannel(pcmFloat32, 0);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
}

async function start() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
  });

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/session`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("Connected — say hello!");
  ws.onclose = () => setStatus("Disconnected");
  ws.onerror = () => setStatus("Connection error");

  ws.onmessage = (event) => {
    const bytes = new Uint8Array(event.data);
    const kind = bytes[0];
    if (kind === 1) {
      const pcm = new Float32Array(bytes.buffer, bytes.byteOffset + 1);
      playPcm(pcm);
    } else if (kind === 2) {
      const text = new TextDecoder().decode(bytes.subarray(1));
      transcriptEl.textContent += text;
    } else if (kind === 3) {
      const fmt = bytes[1];
      const view = new DataView(bytes.buffer, bytes.byteOffset + 2, 8);
      const height = view.getUint32(0, true);
      const width = view.getUint32(4, true);
      const frameData = bytes.subarray(10);
      drawYuv420Frame(frameData, height, width);
    } else if (kind === 4) {
      setStatus("Error: " + new TextDecoder().decode(bytes.subarray(1)));
    }
  };

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
    if (ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    ws.send(input.slice().buffer);
  };
}

startBtn.addEventListener("click", () => {
  startBtn.disabled = true;
  setStatus("Connecting...");
  start().catch((err) => setStatus("Failed to start: " + err.message));
});
