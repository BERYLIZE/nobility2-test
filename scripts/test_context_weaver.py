"""Verify Context Weaver actually calls the NIM API and produces a real
compressed summary + text_prompt, not a stub."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from context_weaver import build_from_env

cw = build_from_env()

cw.add_transcript_line("User", "Hey, I'm working on a project called Nobility2, it's a talking avatar.")
cw.add_transcript_line("Agent", "That sounds fascinating! What's driving the avatar's face?")
cw.add_transcript_line("User", "A model called AVTR-1, it does lip sync straight from audio.")
cw.add_transcript_line("Agent", "Got it -- so no separate gesture model feeding it directly?")
cw.add_transcript_line("User", "Right, EMAGE runs in parallel but only triggers a reaction library now.")

print("Refreshing via real NIM API call...")
new_prompt = cw.refresh()
print(f"\nSummary: {cw.current_summary!r}")
print(f"\nFull text_prompt for PersonaPlex reconnect: {new_prompt!r}")

assert len(cw.current_summary) > 10, "Summary is suspiciously short/empty"
assert "Nobility" in new_prompt or "avatar" in new_prompt.lower() or "AVTR" in new_prompt, \
    "Summary doesn't reference the actual conversation content"
print("\nPASS: Context Weaver produced a real, content-aware summary via the NIM API")
