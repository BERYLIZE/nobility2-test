---
title: Nobility2
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Nobility2 — Face-to-Face Bi-Directional Conversational Avatar

Real-time, full-duplex, face-to-face conversational avatar.

- **PersonaPlex-7B** — full-duplex brain, voice, primary driver of facial expression
- **Context Weaver** — background LLM giving PersonaPlex conversational memory beyond its native window
- **EMAGE** — gesture and expression choreography generator
- **AVTR-1** — sole live renderer, driven by EMAGE conditioning
- **Reaction Library** — pre-rendered high-intensity reaction clips (generated offline via EchoMimicV3), swapped in via crossfade for big laughs/gestures instead of live diffusion

See `HANDOFF.md` for the full build spec and build order.
