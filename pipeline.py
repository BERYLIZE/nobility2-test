"""pipeline.py -- top-level orchestrator wiring PersonaPlex's full-duplex
loop, Context Weaver's scheduled refresh, and the Director/Avatar event
chain into one running session.

Per Step 2's verified finding, PersonaPlex's text_prompt can only be set at
WebSocket connection time -- there is no mid-session hot-swap. So Context
Weaver's "push" is implemented here as a scheduled reconnect: every
`refresh_interval_s`, the current PersonaPlex connection is closed and
reopened with the updated text_prompt, well ahead of PersonaPlex's native
~160-240s instability window.
"""
from __future__ import annotations

import asyncio
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

from context_weaver import ContextWeaver
from director import Director, DirectorState


@dataclass
class PipelineConfig:
    personaplex_host: str = "localhost"
    personaplex_port: int = 8998
    voice_prompt: str = "NATF0.pt"
    seed: int = -1


@dataclass
class SessionState:
    director_state: DirectorState = field(default_factory=lambda: Director().initial_state())
    connection_start_ts: float = field(default_factory=time.time)
    reconnect_count: int = 0


class Pipeline:
    """Orchestrates one conversational session end to end.

    Owns:
    - the PersonaPlex WebSocket lifecycle (connect / scheduled reconnect)
    - Context Weaver's background transcript summarization
    - Director's greeting + CAD + Reaction Library trigger checks
    - handing rendered frames from avatar.py to the caller (WebRTC out)
    """

    def __init__(self, config: PipelineConfig, context_weaver: ContextWeaver, director: Director):
        self._config = config
        self._context_weaver = context_weaver
        self._director = director
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None

    def _build_uri(self, text_prompt: str) -> str:
        encoded = urllib.parse.quote(text_prompt)
        return (
            f"ws://{self._config.personaplex_host}:{self._config.personaplex_port}"
            f"/api/chat?voice_prompt={self._config.voice_prompt}"
            f"&text_prompt={encoded}&seed={self._config.seed}"
        )

    async def start_session(self, state: SessionState) -> None:
        """Cold-start: speak the greeting as the initial text_prompt, then
        connect to PersonaPlex."""
        import websockets

        greeting = self._director.maybe_greeting(state.director_state)
        text_prompt = greeting or self._context_weaver.text_prompt()
        self._ws = await websockets.connect(self._build_uri(text_prompt), max_size=None)
        await self._ws.recv()  # handshake byte, per Step 2's verified protocol
        state.connection_start_ts = time.time()

    async def maybe_refresh(self, state: SessionState) -> bool:
        """Check if Context Weaver is due for a refresh; if so, run it and
        do the scheduled reconnect. Returns True if a reconnect happened."""
        if not self._context_weaver.due_for_refresh():
            return False
        new_prompt = self._context_weaver.refresh()
        await self._ws.close()
        import websockets
        self._ws = await websockets.connect(self._build_uri(new_prompt), max_size=None)
        await self._ws.recv()
        state.connection_start_ts = time.time()
        state.reconnect_count += 1
        return True

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
