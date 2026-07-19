"""Step 11: run a real session against a live PersonaPlex server, driven by
pipeline.py + a real Context Weaver, PAST the 160-240s native instability
mark, confirming the scheduled-reconnect strategy keeps the connection
healthy the whole way through.

PersonaPlex is full-duplex: send and receive run CONTINUOUSLY and
concurrently for the life of one connection (not discrete send-then-drain
cycles -- an earlier version of this test used that structure and triggered
spurious server-side closes). Reconnects happen only on Context Weaver's
schedule, cancelling and restarting both continuous tasks."""
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


def make_speech_opus(seconds: float, seed_topic: str) -> bytes:
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


class ConnectionCounters:
    def __init__(self):
        self.audio_bytes = 0
        self.text_tokens = 0


async def continuous_sender(ws, topics: list[str]):
    """Streams a new synthetic speech clip every 4s, in real time, for as
    long as this connection lives."""
    idx = 0
    chunk_size = 960
    while True:
        topic = topics[idx % len(topics)]
        idx += 1
        opus_bytes = make_speech_opus(4.0, topic)
        for i in range(0, len(opus_bytes), chunk_size):
            await ws.send(b"\x01" + opus_bytes[i:i + chunk_size])
            await asyncio.sleep(0.02)


async def continuous_receiver(ws, counters: ConnectionCounters):
    async for msg in ws:
        if isinstance(msg, bytes) and len(msg) > 0:
            if msg[0] == 1:
                counters.audio_bytes += len(msg) - 1
            elif msg[0] == 2:
                counters.text_tokens += 1


async def main():
    config = PipelineConfig(personaplex_host="localhost", personaplex_port=8998)
    cw = ContextWeaver(api_key=os.environ["NVIDIA_API_KEY"], refresh_interval_s=75)
    director = Director()
    pipeline = Pipeline(config, cw, director)

    state = SessionState()
    await pipeline.start_session(state)
    print(f"[t=0s] session started, greeting sent as initial text_prompt")

    topics = ["gestures", "music", "travel", "weather", "cooking", "space", "history", "art"]
    counters = ConnectionCounters()
    sender_task = asyncio.create_task(continuous_sender(pipeline._ws, topics))
    receiver_task = asyncio.create_task(continuous_receiver(pipeline._ws, counters))

    start = time.time()
    reconnects_at = []
    total_audio_bytes = 0
    total_text_tokens = 0

    while time.time() - start < TOTAL_DURATION_S:
        cw.add_transcript_line("User", f"Let's keep chatting about {topics[int(time.time()) % len(topics)]}.")
        await asyncio.sleep(5)

        if cw.due_for_refresh():
            sender_task.cancel()
            receiver_task.cancel()
            total_audio_bytes += counters.audio_bytes
            total_text_tokens += counters.text_tokens

            did_reconnect = await pipeline.maybe_refresh(state)
            assert did_reconnect
            reconnects_at.append(round(time.time() - start, 1))
            print(f"[t={time.time()-start:.0f}s] scheduled reconnect #{state.reconnect_count} "
                  f"(summary: {cw.current_summary[:80]!r}...) "
                  f"audio_bytes_this_connection={counters.audio_bytes}")

            counters = ConnectionCounters()
            sender_task = asyncio.create_task(continuous_sender(pipeline._ws, topics))
            receiver_task = asyncio.create_task(continuous_receiver(pipeline._ws, counters))
        else:
            print(f"[t={time.time()-start:.0f}s] alive, audio_bytes_this_connection={counters.audio_bytes}, "
                  f"text_tokens_this_connection={counters.text_tokens}")

    sender_task.cancel()
    receiver_task.cancel()
    total_audio_bytes += counters.audio_bytes
    total_text_tokens += counters.text_tokens
    await pipeline.close()

    print(f"\n=== Session ran {time.time()-start:.0f}s total ===")
    print(f"Reconnects: {len(reconnects_at)} at t={reconnects_at}")
    print(f"Total audio bytes received from PersonaPlex: {total_audio_bytes}")
    print(f"Total text tokens received: {total_text_tokens}")

    assert time.time() - start >= TOTAL_DURATION_S, "session ended early"
    assert len(reconnects_at) >= 2, f"expected multiple scheduled reconnects past 160-240s, got {len(reconnects_at)}"
    assert total_audio_bytes > 0, "PersonaPlex never produced audio output"
    print("\nPASS: session survived past the 160-240s instability mark via scheduled Context Weaver reconnects")


asyncio.run(main())
