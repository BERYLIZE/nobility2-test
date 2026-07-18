# Single source of truth for "why is this package here."
# Populated as each model's dependency chain lands (build order step 1).
#
# aiortc / av        -> WebRTC serving + audio/video muxing for live output (step 11)
# huggingface_hub    -> gated weight downloads (PersonaPlex, AVTR-1) after license accept
# fastapi / uvicorn  -> Space HTTP + WebSocket/WebRTC signaling server
#
# Not yet pinned (added when their build-order step lands):
# - PersonaPlex Mimi audio codec deps (step 2)
# - EMAGE torch/smplx deps (step 3) -- GitHub-sourced, not pip-installable as a package
# - Face detector: InsightFace (Phase 1, non-commercial internal-only) vs MediaPipe
#   Face Mesh (Phase 2 public-release gate) -- see AVTR-1 row in HANDOFF.md. Pin ONE,
#   not both.
# - ffmpeg (system package, not pip) -- Reaction Library playback/crossfade muxing
