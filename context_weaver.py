"""Context Weaver.

Background LLM giving PersonaPlex conversational memory beyond its ~160-240s
native instability window. Not on the critical response-latency path.

Verified constraint (Step 2): PersonaPlex's text-prompt channel is only
settable at WebSocket connection time -- there is no mid-session control
message to hot-swap it. So the push mechanism here is NOT an invisible
in-place update; it's a scheduled reconnect: every `refresh_interval_s`,
compress the rolling transcript into a new summary and hand it to the
caller, which is responsible for closing and reopening the PersonaPlex
WebSocket with the new `text_prompt` query param (a brief handoff window,
per the handoff doc's own documented fallback for this exact case).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

NIM_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
DEFAULT_REFRESH_INTERVAL_S = 90

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a rolling summary of an ongoing spoken conversation for a "
    "voice AI persona. Given the previous summary (if any) and new transcript "
    "lines, produce an updated summary under 80 words: who the speakers are, "
    "what's been discussed, the current topic, and the persona's established "
    "role/tone. This summary will be re-injected as the persona's system "
    "prompt, so write it as an instruction to the persona, not as narration."
)


@dataclass
class ContextWeaver:
    api_key: str
    model: str = DEFAULT_MODEL
    refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S
    base_persona: str = "You are Nobility, a friendly and curious conversational AI assistant."
    transcript: list[str] = field(default_factory=list)
    current_summary: str = ""
    last_refresh_ts: float = field(default_factory=time.time)

    def add_transcript_line(self, speaker: str, text: str) -> None:
        self.transcript.append(f"{speaker}: {text}")

    def due_for_refresh(self, now: float | None = None) -> bool:
        if not self.api_key:
            # Degraded mode (no NIM key): never refresh; base persona only.
            return False
        now = now if now is not None else time.time()
        return (now - self.last_refresh_ts) >= self.refresh_interval_s

    def refresh(self) -> str:
        """Call the NIM-hosted LLM to compress the transcript into an updated
        summary. Returns the new text_prompt to use on PersonaPlex's next
        reconnect."""
        new_lines = "\n".join(self.transcript)
        user_content = (
            f"Previous summary: {self.current_summary or '(none yet)'}\n\n"
            f"New transcript since last summary:\n{new_lines}"
        )
        response = requests.post(
            NIM_CHAT_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()

        self.current_summary = summary
        self.transcript = []  # compressed into the summary; drop raw lines
        self.last_refresh_ts = time.time()
        return self.text_prompt()

    def text_prompt(self) -> str:
        """The full text_prompt to hand to PersonaPlex's server.py on
        (re)connect: base persona + rolling summary, if any."""
        if not self.current_summary:
            return self.base_persona
        return f"{self.base_persona} Conversation so far: {self.current_summary}"


def build_from_env() -> ContextWeaver:
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_CLI_API_KEY") or ""
    if not api_key or api_key == "placeholder":
        # Don't kill live sessions over a missing summarizer key -- run
        # degraded (base persona, no rolling summary) and say so once.
        print("context_weaver: no NVIDIA_API_KEY -- running without rolling summaries", flush=True)
        api_key = ""
    return ContextWeaver(api_key=api_key)
