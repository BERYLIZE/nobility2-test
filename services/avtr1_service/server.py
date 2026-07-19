"""AVTR-1 rendering microservice.

Runs in its own venv (torch>=2.5,<2.8 + TensorRT), isolated from PersonaPlex's
venv (torch<2.5) since the two have incompatible torch requirements -- see
BUILD_STATUS.md Step 12. Exposes a local WebSocket that the main app.py
process (a third, lightweight environment) connects to over localhost.

Wire protocol (binary WS messages), mirroring PersonaPlex's own byte-tag
convention for consistency:
  kind=1: agent speech PCM chunk (float32 LE, 16kHz mono) -- drives lip-sync
  kind=2: user listen PCM chunk (float32 LE, 16kHz mono) -- drives idle motion
  kind=3 (server->client): one rendered video frame: 1 byte format tag,
          4 bytes height (uint32 LE), 4 bytes width (uint32 LE), then the
          frame payload. Format tags: 0 = raw yuv_i420 planes, 1 = raw
          yuv_i420_stacked_alpha planes, 2 = JPEG (BGR-encoded server-side).

JPEG is the default (NOBILITY2_FRAME_CODEC=jpeg). A raw 720x1280 I420 frame
is ~1.4MB; at 25fps that's ~35MB/s -- fine on localhost, unusable from a
Space to a real browser. JPEG at quality ~82 is ~40-80KB/frame (~1-2MB/s),
and also removes the client's slow per-pixel JS YUV->RGB loop (the browser
decodes JPEG natively off the main thread via createImageBitmap).
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import sys

import cv2
import numpy as np
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from avatar import build_avatar_from_config, ReactionClip
from avtr1_renderer.types import Chunk

# From Chunk's docstring: (5+5)*640 + 80 = 6480 samples per call, both tracks.
CHUNK_SAMPLES = 6480


class AudioAccumulator:
    def __init__(self):
        self.speech = np.zeros(0, dtype=np.float32)
        self.listen = np.zeros(0, dtype=np.float32)

    def add_speech(self, pcm: np.ndarray) -> None:
        self.speech = np.concatenate([self.speech, pcm])

    def add_listen(self, pcm: np.ndarray) -> None:
        self.listen = np.concatenate([self.listen, pcm])

    def pop_ready_chunk(self) -> Chunk | None:
        if len(self.speech) < CHUNK_SAMPLES or len(self.listen) < CHUNK_SAMPLES:
            return None
        chunk = Chunk(
            audio_speech=self.speech[:CHUNK_SAMPLES].copy(),
            audio_listen=self.listen[:CHUNK_SAMPLES].copy(),
        )
        self.speech = self.speech[CHUNK_SAMPLES:]
        self.listen = self.listen[CHUNK_SAMPLES:]
        return chunk


PIXEL_FORMAT_CODES = {"yuv_i420": 0, "yuv_i420_stacked_alpha": 1}
FMT_JPEG = 2

FRAME_CODEC = os.environ.get("NOBILITY2_FRAME_CODEC", "jpeg").lower()
JPEG_QUALITY = int(os.environ.get("NOBILITY2_JPEG_QUALITY", "82"))


def encode_frame(frame) -> bytes:
    """Pack one Frame into a kind=3 wire message (JPEG by default)."""
    if FRAME_CODEC == "jpeg" and frame.format == "yuv_i420":
        # Frame.data is (3H//2, W) uint8 planar I420 on host (upstream
        # types.py docstring) -- exactly cv2's I420 layout.
        bgr = cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR_I420)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            header = bytes([3, FMT_JPEG]) + struct.pack("<II", frame.height, frame.width)
            return header + jpg.tobytes()
        # fall through to raw on encoder failure
    fmt_code = PIXEL_FORMAT_CODES.get(frame.format, 0)
    header = bytes([3, fmt_code]) + struct.pack("<II", frame.height, frame.width)
    return header + frame.data.tobytes()


async def handle_connection(ws, avatar):
    state = avatar.initial_state()
    acc = AudioAccumulator()

    async for message in ws:
        if not isinstance(message, bytes) or len(message) == 0:
            continue
        kind = message[0]
        payload = message[1:]
        if kind == 1:
            acc.add_speech(np.frombuffer(payload, dtype=np.float32))
        elif kind == 2:
            acc.add_listen(np.frombuffer(payload, dtype=np.float32))
        else:
            continue

        chunk = acc.pop_ready_chunk()
        if chunk is None:
            continue

        state, frame_iter = avatar.process_chunk(chunk=chunk, state=state, trigger=None)
        for frame in frame_iter:
            await ws.send(encode_frame(frame))


async def main():
    config_path = os.environ.get("NOBILITY2_REFERENCE_CONFIG", "config/reference.json")
    avatar = build_avatar_from_config(config_path=config_path, bg_id=os.environ.get("NOBILITY2_BG_ID", "plain_white"))
    port = int(os.environ.get("AVTR1_SERVICE_PORT", "9001"))

    async def handler(ws):
        await handle_connection(ws, avatar)

    async with websockets.serve(handler, "0.0.0.0", port, max_size=None):
        print(f"AVTR-1 rendering service listening on 0.0.0.0:{port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
