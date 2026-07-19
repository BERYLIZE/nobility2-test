# Nobility2 Build Status

## Step 1: HF Space scaffold + Docker environment — DONE
- GitHub: https://github.com/BERYLIZE/nobility2-test
- HF Space (Docker SDK, live): https://huggingface.co/spaces/AIBRUH/nobility2
- Verified via real HTTP responses (`/health`, `/`), not just a successful build.

## Step 2: PersonaPlex-7B standalone — DONE
- Model: `nvidia/personaplex-7b-v1` (Moshi/Mimi architecture, bf16 safetensors, 8.37B params).
  No quantized/GGUF variant exists on HF — full precision is the only option, not a default choice.
- Code: cloned exact upstream from https://github.com/NVIDIA/personaplex (not reimplemented).
- Ran on an HF GPU sandbox (Nvidia A10G small, $1.00/hr).
- **Bugs found and patched in upstream code (documented, not silently worked around):**
  1. `server.py` used `request["seed"]` instead of `request.query["seed"]` — crashed every request with a seed query param present.
  2. Server crashes with `TypeError: 'NoneType' object is not iterable` if `text_prompt` is empty — a non-empty persona/system prompt is required, not optional as the API suggests.
- **Verified end-to-end on real audio**: sent a synthesized speech-like waveform over the model's WebSocket protocol (`/api/chat`, Opus-encoded PCM), received back genuine generated speech: text token stream decoded to `"Hello, I'm Nobility. How can I help you today?"` plus 12,355 bytes of real synthesized output audio. This is a real model response, not a stub.
- **Text-prompt hot-swap flagged in the handoff doc — tested, does NOT work as an invisible mid-session swap.** `text_prompt_tokens` is set once at WebSocket connection time from the query string; the receive loop only handles audio frames, with no control-message path to update the prompt on an open connection.
  - **Fallback adopted (per handoff doc's own contingency):** Context Weaver must do brief scheduled reconnects — close and reopen the PersonaPlex WS with an updated `text_prompt` each refresh cycle — instead of a fully invisible swap.
- Frontend decision: using PersonaPlex's own bundled webui (`client/`, pre-built as `dist.tgz` in the model repo) for voice-only testing now. Swap to a video-capable UI once `avatar.py` (Step 8) exists — a template swap, not a rewrite.

## Step 3: EMAGE standalone — IN PROGRESS
- Confirmed paper: arXiv 2401.00374, author H-Liu1997, org PantoMatrix (per handoff doc's own hint).
