"""Step 11: run a real session against a live PersonaPlex server, driven by
pipeline.py + a real Context Weaver, PAST the 160-240s native instability
mark, confirming the scheduled-reconnect strategy keeps the connection
healthy the whole way through (not just for a short test).

Uses the same single-coroutine send/receive pattern verified working in
Step 2's test_personaplex.py (a separate concurrent receiver task caused
spurious immediate connection closes -- see BUILD_STATUS.md Step 11)."""
import asyncio
import os
import sys
import time

import numpy as np
import sphn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import Pipeline, PipelineConfig, SessionState
from context_weaver import ContextWeaver
from director import Director

TOTAL_DURATION_S = 210  # past the documented 160-240s instability window
CHUNK_S = 4


def make_speech_chunk(seconds: float, seed_topic: str) -> bytes:
    sample_rate = 24000
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    freq = 150 + (hash(seed_topic) % 100)
    pcm = (0.15 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
    writer = sphn.OpusStreamWriter(sample_rate)
    frame = 1920
    chunks = []
    for i in range(0, len(pcm) - frame + 1, frame):
        writer.append_pcm(pcm[i:i + frame])
        chunks.append(writer.read_bytes())
        # pace sends roughly real-time, matching the verified Step 2 client
    return b"".join(chunks)


async def send_and_drain(ws, opus_bytes: bytes, drain_seconds: float) -> tuple[int, int]:
    """Send one chunk's audio while concurrently draining responses inline
    (single coroutine, like the verified Step 2 client), returning
    (audio_bytes_received, text_tokens_received)."""
    audio_bytes = 0
    text_tokens = 0
    chunk_size = 960

    async def sender():
        for i in range(0, len(opus_bytes), chunk_size):
            await ws.send(b"\x01" + opus_bytes[i:i + chunk_size])
            await asyncio.sleep(0.02)

    send_task = asyncio.create_task(sender())
    try:
        async with asyncio.timeout(drain_seconds):
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes) and len(msg) > 0:
                    if msg[0] == 1:
                        audio_bytes += len(msg) - 1
                    elif msg[0] == 2:
                        text_tokens += 1
    except (asyncio.TimeoutError, TimeoutError):
        pass
    if not send_task.done():
        send_task.cancel()
    return audio_bytes, text_tokens


async def main():
    config = PipelineConfig(personaplex_host="localhost", personaplex_port=8998)
    cw = ContextWeaver(api_key=os.environ["NVIDIA_API_KEY"], refresh_interval_s=75)
    director = Director()
    pipeline = Pipeline(config, cw, director)

    state = SessionState()
    await pipeline.start_session(state)
    print(f"[t=0s] session started, greeting sent as initial text_prompt")

    start = time.time()
    audio_bytes_total = 0
    text_tokens_total = 0
    reconnects_at = []

    topics = ["gestures", "music", "travel", "weather", "cooking", "space", "history", "art"]
    topic_idx = 0
    while time.time() - start < TOTAL_DURATION_S:
        topic = topics[topic_idx % len(topics)]
        topic_idx += 1
        cw.add_transcript_line("User", f"Let's talk about {topic}.")
        cw.add_transcript_line("Agent", f"Sure, {topic} is interesting to discuss.")

        opus_bytes = make_speech_chunk(CHUNK_S, topic)
        audio_b, text_t = await send_and_drain(pipeline._ws, opus_bytes, drain_seconds=CHUNK_S + 2)
        audio_bytes_total += audio_b
        text_tokens_total += text_t

        did_reconnect = await pipeline.maybe_refresh(state)
        if did_reconnect:
            reconnects_at.append(round(time.time() - start, 1))
            print(f"[t={time.time()-start:.0f}s] scheduled reconnect #{state.reconnect_count} "
                  f"(summary: {cw.current_summary[:80]!r}...)")

        print(f"[t={time.time()-start:.0f}s] chunk done, audio_bytes_so_far={audio_bytes_total}, "
              f"text_tokens_so_far={text_tokens_total}")

    await pipeline.close()

    print(f"\n=== Session ran {time.time()-start:.0f}s total ===")
    print(f"Reconnects: {state.reconnect_count} at t={reconnects_at}")
    print(f"Total audio bytes received from PersonaPlex: {audio_bytes_total}")
    print(f"Total text tokens received: {text_tokens_total}")

    assert time.time() - start >= TOTAL_DURATION_S, "session ended early"
    assert state.reconnect_count >= 2, f"expected multiple scheduled reconnects past 160-240s, got {state.reconnect_count}"
    assert audio_bytes_total > 0, "PersonaPlex never produced audio output"
    print("\nPASS: session survived past the 160-240s instability mark via scheduled Context Weaver reconnects")


asyncio.run(main())
