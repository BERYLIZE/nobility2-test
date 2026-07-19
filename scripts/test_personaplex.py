"""Step 2 verification: send real audio through PersonaPlex's live WebSocket
and confirm it produces real output audio + text tokens (not just that the
server boots)."""
import asyncio
import sys
import numpy as np
import sphn
import websockets


async def main():
    sample_rate = 24000
    duration_s = 4
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # speech-like modulated tone, not silence/noise, to give the model real signal to react to
    pcm = (0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t))).astype(np.float32)

    writer = sphn.OpusStreamWriter(sample_rate)
    frame = 1920
    opus_chunks = []
    for i in range(0, len(pcm) - frame + 1, frame):
        writer.append_pcm(pcm[i:i + frame])
        await asyncio.sleep(0.01)
        chunk = writer.read_bytes()
        if chunk:
            opus_chunks.append(chunk)
    opus_bytes = b"".join(opus_chunks)
    print(f"Encoded {len(pcm)} PCM samples -> {len(opus_bytes)} opus bytes", file=sys.stderr)

    import urllib.parse
    text_prompt = urllib.parse.quote("You are Nobility, a friendly and curious conversational AI assistant.")
    uri = f"ws://localhost:8998/api/chat?voice_prompt=NATF0.pt&text_prompt={text_prompt}&seed=-1"
    audio_bytes_received = 0
    text_tokens = []

    async with websockets.connect(uri, max_size=None) as ws:
        handshake = await ws.recv()
        print(f"Handshake: {handshake!r}", file=sys.stderr)

        async def sender():
            chunk_size = 960
            for i in range(0, len(opus_bytes), chunk_size):
                await ws.send(b"\x01" + opus_bytes[i:i + chunk_size])
                await asyncio.sleep(0.02)

        send_task = asyncio.create_task(sender())

        try:
            async with asyncio.timeout(20):
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes) and len(msg) > 0:
                        kind = msg[0]
                        if kind == 1:
                            audio_bytes_received += len(msg) - 1
                        elif kind == 2:
                            text_tokens.append(msg[1:].decode("utf-8", errors="replace"))
        except (asyncio.TimeoutError, TimeoutError):
            pass
        send_task.cancel()

    print(f"RESULT audio_bytes_received={audio_bytes_received} text_tokens={''.join(text_tokens)!r}")
    assert audio_bytes_received > 0, "No audio came back from PersonaPlex"


if __name__ == "__main__":
    asyncio.run(main())
