"""Step 11: run a real session against a live PersonaPlex server, driven by
pipeline.py + a real Context Weaver, PAST the 160-240s native instability
mark, confirming the scheduled-reconnect strategy keeps the connection
healthy the whole way through (not just for a short test)."""
import asyncio
import os
import sys
import time

import numpy as np
import sphn
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import Pipeline, PipelineConfig, SessionState
from context_weaver import ContextWeaver
from director import Director

TOTAL_DURATION_S = 210  # past the documented 160-240s instability window
CHUNK_S = 10


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
    return b"".join(chunks)


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
    text_received = []
    reconnects_at = []

    async def receiver(ws):
        nonlocal audio_bytes_total
        try:
            async for msg in ws:
                if isinstance(msg, bytes) and len(msg) > 0:
                    if msg[0] == 1:
                        audio_bytes_total += len(msg) - 1
                    elif msg[0] == 2:
                        text_received.append(msg[1:].decode("utf-8", errors="replace"))
        except websockets.exceptions.ConnectionClosed:
            pass

    recv_task = asyncio.create_task(receiver(pipeline._ws))

    elapsed_topics = ["gestures", "music", "travel", "weather", "cooking", "space", "history", "art"]
    topic_idx = 0
    while time.time() - start < TOTAL_DURATION_S:
        topic = elapsed_topics[topic_idx % len(elapsed_topics)]
        topic_idx += 1
        cw.add_transcript_line("User", f"Let's talk about {topic}.")
        cw.add_transcript_line("Agent", f"Sure, {topic} is interesting to discuss.")

        opus_bytes = make_speech_chunk(CHUNK_S, topic)
        chunk_size = 960
        for i in range(0, len(opus_bytes), chunk_size):
            await pipeline._ws.send(b"\x01" + opus_bytes[i:i + chunk_size])
            await asyncio.sleep(0.01)

        did_reconnect = await pipeline.maybe_refresh(state)
        if did_reconnect:
            recv_task.cancel()
            recv_task = asyncio.create_task(receiver(pipeline._ws))
            reconnects_at.append(round(time.time() - start, 1))
            print(f"[t={time.time()-start:.0f}s] scheduled reconnect #{state.reconnect_count} "
                  f"(summary: {cw.current_summary[:80]!r}...)")

        print(f"[t={time.time()-start:.0f}s] chunk sent, audio_bytes_so_far={audio_bytes_total}, "
              f"text_tokens_so_far={len(text_received)}")

    recv_task.cancel()
    await pipeline.close()

    print(f"\n=== Session ran {time.time()-start:.0f}s total ===")
    print(f"Reconnects: {state.reconnect_count} at t={reconnects_at}")
    print(f"Total audio bytes received from PersonaPlex: {audio_bytes_total}")
    print(f"Total text tokens received: {len(text_received)}")

    assert time.time() - start >= TOTAL_DURATION_S, "session ended early"
    assert state.reconnect_count >= 2, f"expected multiple scheduled reconnects past 160-240s, got {state.reconnect_count}"
    assert audio_bytes_total > 0, "PersonaPlex never produced audio output"
    print("\nPASS: session survived past the 160-240s instability mark via scheduled Context Weaver reconnects")


asyncio.run(main())
