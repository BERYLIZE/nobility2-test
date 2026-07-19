"""Nobility2 main app: serves the web UI and bridges the browser to the two
isolated model services (PersonaPlex on :8998, AVTR-1 on :9001) that run in
their own venvs due to their incompatible torch requirements (see
BUILD_STATUS.md Step 12).

Audio rates: PersonaPlex/Mimi runs at 24kHz; AVTR-1/HuBERT expects 16kHz.
This process resamples between them with soxr so neither service needs to
change its native rate.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import time

import numpy as np
import soxr
import sphn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from context_weaver import build_from_env
from director import Director
from pipeline import Pipeline, PipelineConfig, SessionState

app = FastAPI(title="Nobility2")
app.mount("/static", StaticFiles(directory="static"), name="static")

PERSONAPLEX_HOST = os.environ.get("PERSONAPLEX_HOST", "localhost")
PERSONAPLEX_PORT = int(os.environ.get("PERSONAPLEX_PORT", "8998"))
AVTR1_HOST = os.environ.get("AVTR1_HOST", "localhost")
AVTR1_PORT = int(os.environ.get("AVTR1_PORT", "9001"))

BROWSER_SAMPLE_RATE = 16000  # requested from getUserMedia; see static/app.js
PERSONAPLEX_SAMPLE_RATE = 24000
AVTR1_SAMPLE_RATE = 16000


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.websocket("/ws/session")
async def session_ws(browser_ws: WebSocket):
    await browser_ws.accept()

    config = PipelineConfig(personaplex_host=PERSONAPLEX_HOST, personaplex_port=PERSONAPLEX_PORT)
    context_weaver = build_from_env()
    director = Director()
    pipeline = Pipeline(config, context_weaver, director)
    state = SessionState()

    try:
        await pipeline.start_session(state)
    except Exception as exc:
        await browser_ws.send_bytes(b"\x04" + str(exc).encode("utf-8"))
        await browser_ws.close()
        return

    avtr1_ws = await websockets.connect(f"ws://{AVTR1_HOST}:{AVTR1_PORT}", max_size=None)

    opus_reader = sphn.OpusStreamReader(PERSONAPLEX_SAMPLE_RATE)
    opus_writer = sphn.OpusStreamWriter(PERSONAPLEX_SAMPLE_RATE)
    close = False

    async def browser_to_services():
        """Mic PCM from the browser -> PersonaPlex (opus, 24k) + AVTR-1 listen track (16k, native)."""
        nonlocal close
        try:
            while True:
                data = await browser_ws.receive_bytes()
                pcm16k = np.frombuffer(data, dtype=np.float32)
                pcm24k = soxr.resample(pcm16k, BROWSER_SAMPLE_RATE, PERSONAPLEX_SAMPLE_RATE, quality="HQ").astype(np.float32)
                opus_writer.append_pcm(pcm24k)
                opus_bytes = opus_writer.read_bytes()
                if opus_bytes:
                    await pipeline._ws.send(b"\x01" + opus_bytes)
                await avtr1_ws.send(b"\x02" + pcm16k.astype(np.float32).tobytes())
        except WebSocketDisconnect:
            pass
        finally:
            close = True

    async def personaplex_to_services():
        """PersonaPlex's generated speech -> browser (playback, resampled) + AVTR-1 speech track (16k)."""
        try:
            async for msg in pipeline._ws:
                if close:
                    break
                if not isinstance(msg, bytes) or len(msg) == 0:
                    continue
                kind = msg[0]
                if kind == 1:
                    opus_reader.append_bytes(msg[1:])
                    pcm24k = opus_reader.read_pcm()
                    if pcm24k is None or pcm24k.shape[-1] == 0:
                        continue
                    await browser_ws.send_bytes(b"\x01" + pcm24k.astype(np.float32).tobytes())
                    pcm16k = soxr.resample(pcm24k, PERSONAPLEX_SAMPLE_RATE, AVTR1_SAMPLE_RATE, quality="HQ").astype(np.float32)
                    await avtr1_ws.send(b"\x01" + pcm16k.tobytes())
                elif kind == 2:
                    text = msg[1:].decode("utf-8", errors="replace")
                    context_weaver.add_transcript_line("Agent", text)
                    await browser_ws.send_bytes(b"\x02" + text.encode("utf-8"))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def avtr1_to_browser():
        """Rendered video frames -> browser, unchanged."""
        try:
            async for msg in avtr1_ws:
                if close:
                    break
                if isinstance(msg, bytes) and len(msg) > 0 and msg[0] == 3:
                    await browser_ws.send_bytes(msg)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def refresh_loop():
        """Context Weaver's scheduled reconnect, per Step 2/10's verified constraint."""
        while not close:
            await asyncio.sleep(5)
            if context_weaver.due_for_refresh():
                await pipeline.maybe_refresh(state)

    tasks = [
        asyncio.create_task(browser_to_services()),
        asyncio.create_task(personaplex_to_services()),
        asyncio.create_task(avtr1_to_browser()),
        asyncio.create_task(refresh_loop()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        close = True
        for t in tasks:
            t.cancel()
        await pipeline.close()
        await avtr1_ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
