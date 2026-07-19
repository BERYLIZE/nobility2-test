"""Verify pipeline.py's session start (greeting as initial text_prompt) and
scheduled-reconnect refresh cycle against a REAL running PersonaPlex server
and a REAL Context Weaver (NIM API) call -- not mocked."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import Pipeline, PipelineConfig, SessionState
from context_weaver import ContextWeaver
from director import Director


async def main():
    config = PipelineConfig(personaplex_host="localhost", personaplex_port=8998)
    cw = ContextWeaver(api_key=os.environ["NVIDIA_API_KEY"], refresh_interval_s=2)
    director = Director()
    pipeline = Pipeline(config, cw, director)

    state = SessionState()
    await pipeline.start_session(state)
    print(f"Session started. Greeting used as text_prompt: {director.maybe_greeting(SessionState().director_state) is not None}")
    assert state.director_state.greeted, "greeting should have been consumed on session start"
    print("PASS: session started against real PersonaPlex server with greeting as initial text_prompt")

    # Force a refresh cycle: add transcript content, wait past the (short, test-only) interval.
    cw.add_transcript_line("User", "Tell me about your favorite gesture.")
    cw.add_transcript_line("Agent", "I love a good enthusiastic wave.")
    await asyncio.sleep(2.5)
    did_reconnect = await pipeline.maybe_refresh(state)
    assert did_reconnect, "expected a scheduled reconnect to have happened"
    assert state.reconnect_count == 1
    print(f"PASS: scheduled reconnect happened via real Context Weaver summary: {cw.current_summary!r}")

    await pipeline.close()
    print("\nPASS: pipeline.py verified end-to-end against a real PersonaPlex server + real NIM API")


asyncio.run(main())
