# Nobility2 — Face-to-Face Bi-Directional Conversational Avatar
## Claude Code Build Doc — v3 — Hugging Face Spaces Target

## What Nobility2 is
A real-time, full-duplex, face-to-face conversational avatar, built from
scratch, deployed as a Hugging Face Space. PersonaPlex is the brain, voice,
and primary driver of facial expression, backed by a Context Weaver giving
it conversational memory beyond its native window. EMAGE generates gesture
and expression choreography. AVTR-1, fully driven by EMAGE's conditioning,
is the sole live renderer. High-intensity reactions (big laughs, large
gestures) are handled by a pre-rendered Reaction Library, not live
diffusion — this is the deliberate fix for a real latency problem diffusion
models have at conversational speed; see the Reaction Library section.

This is a new project. No code, file structure, or architecture from any
other build should be assumed, referenced, or reused. Build everything here
from a clean repository, deployed as a Docker-based HF Space.

## You are connected and authorized
GitHub, Hugging Face, and NGC (NVIDIA GPU Cloud) accounts are already
connected. You have the authority to clone repos, push code, download
gated model weights (after license acceptance where required), pull NGC
containers/assets, and create/configure the HF Space directly. Do not ask
for permission to use these connections — using them is the job. If any
one of the three connections fails or lacks a required scope, say so
explicitly in your report; do not silently work around a missing
connection by skipping the component that needed it.

## Standing orders (non-negotiable)
- No agent proliferation. One agent, one task queue. Do not spawn sub-agents.
- No premature completion claims. A component is "done" only after it has
  run end-to-end on real audio and produced real output frames — not after
  the code compiles, imports cleanly, or the Space builds without erroring.
- Copy exact original code from upstream repos when integrating third-party
  models. Wrap and adapt, don't rewrite or "simplify" working reference code.
- Full autonomy on stack decisions within this doc's scope. Log a one-line
  rationale in commit messages, keep moving. No permission cycles.
- **Debugging discipline:** if the same error persists after 4 fix attempts
  on that segment of code, stop iterating blindly. Pivot to researching
  Hugging Face and GitHub for existing working code covering that specific
  segment (a reference implementation, an issue thread with a fix, a
  working fork) before attempting a 5th fix from scratch. Note in the
  report which components needed this pivot and what was found.
- This is a v1 baseline meant to be fine-tuned further from model files and
  custom instructions after this build lands. Build it clean and modular —
  every component should be swappable without a pipeline rewrite.

## Reference image
Not provided yet — the person will supply it separately. Do not block the
build on it. Every component that depends on it (the face detector/
appearance extractor, AVTR-1's identity lock) should be built and tested
against a placeholder/sample portrait first, with the actual reference
image swapped in as the final integration step once supplied. Structure
the image-loading path so that swap is a config change, not a code change.

## Core architecture

```
Mic in ──────────────┐
                      ▼
              PersonaPlex-7B (full-duplex brain + voice)
              ├─ outgoing speech token stream
              ├─ incoming/listening audio stream
              ├─ turn-state (speaking / listening / interrupted / backchannel)
              └─ text-prompt channel (behavioral control, separate from
                 the audio stream — this is the seam Context Weaver uses)
                      │
        ┌─────────────┼──────────────────────┐
        ▼             ▼                       ▼
   CAD           Context Weaver          (turn-state + audio
   (conversation   — background LLM,       continue downstream)
   activity        rolling transcript
   classifier)     summary pushed into
                    PersonaPlex's text-
                    prompt channel on a
                    fixed schedule, well
                    ahead of PersonaPlex's
                    ~160s instability point
        │
        ▼
   EMAGE (gesture/expression curve generator)
   — audio + CAD conditioned, generative, no per-scenario authoring
   — output: full-body gesture curves, facial expression targets,
     and a gesture-intensity signal used for Reaction Library triggering
        │
        ├──────────────────────────────┐
        ▼                              ▼
   AVTR-1 (sole live renderer)    Reaction Library trigger
   — dual-stream: speech-track     (on high gesture-intensity signal)
     + listen-track from                    │
     PersonaPlex, full EMAGE                ▼
     conditioning at all times      Pre-rendered clip playback
        │                           + short crossfade blend in/out
        └──────────────┬────────────────────┘
                        ▼
                   WebRTC out
```

EchoMimicV3 is not in the live path in this build. It is used offline,
ahead of time, to generate the Reaction Library assets — see below.

## New modules

### Conversational Activity Detector (CAD)
Classifies each moment into active listening / backchannel-ready /
interruption / idle-ambient, feeding EMAGE as a conditioning input
alongside raw audio. Build as its own module (`cad.py`).

### Context Weaver
Background LLM, not on the critical response-latency path. Consumes the
full session transcript continuously, maintains a rolling compressed
summary, and pushes it into PersonaPlex's text-prompt channel on a fixed
schedule (target: every ~90 seconds) — well ahead of, not reactive to,
PersonaPlex's native ~160-240 second instability point.

**Flag for verification, not assumed:** confirm PersonaPlex's text-prompt
channel can be updated mid-session without an audio artifact. Test this in
isolation (build order step 2) before anything else depends on it. If it
causes any glitch, fall back to a brief scheduled handoff window instead of
a fully invisible swap.

### Reaction Library (replaces live EchoMimicV3)
An offline-generated set of short, high-intensity reaction clips —
laughing (several variants), surprise, emphatic gestures — rendered once
using EchoMimicV3 with as much compute and denoising quality as needed,
since there's no real-time deadline for this generation. These become
finished video assets stored with the Space.

At runtime, when EMAGE's gesture-intensity signal crosses a threshold, the
system doesn't generate anything live — it selects and plays back the
best-matching pre-rendered clip with a short crossfade in and out of
AVTR-1's live output, so the transition doesn't read as a hard cut. This
avoids EchoMimicV3's real-time latency problem entirely: diffusion
generation at conversational speed doesn't keep pace with playback, but
generating the same clips offline with no deadline produces the same
visual quality with zero live-inference cost.

Build enough variety in the library (multiple takes per reaction type,
selected semi-randomly or by rough intensity match at trigger time) to
avoid visible repetition — this is a content-authoring task, not a model
constraint, and should be scoped as its own build step, not squeezed in
as an afterthought.

## Model format optimization — check before defaulting to full-precision
For every model below, before pulling the standard full-precision
checkpoint, check Hugging Face for a smaller/faster variant that preserves
acceptable quality — GGUF, quantized (int8/int4), ONNX, TensorRT-compiled,
or a vLLM-servable version. Where one exists and quality holds up in
testing, use it over the full-precision default. Note in the report which
components ended up on an optimized format vs. full precision, and why for
any that stayed full precision (no viable smaller version found, or
quality dropped unacceptably in testing — either is a valid reason, but it
should be a tested decision, not a default).

## Weight sources and per-model construction notes
| Component | Source | Extra construction steps |
|---|---|---|
| PersonaPlex-7B | Hugging Face (gated) via NGC/HF | Accept license before download. Verify text-prompt hot-swap in isolation before Context Weaver depends on it. Check for a quantized/optimized variant per the format section above. |
| Context Weaver | No fixed model — any capable LLM with a large context window | New integration work, not a plain download: transcript ingestion, rolling summarization, and the push mechanism into PersonaPlex's text-prompt channel all need to be built. |
| EMAGE | **GitHub only, not Hugging Face** | Weights and code are on GitHub (PantoMatrix org — verify exact repo via the paper's arXiv page if it 404s). The `H-Liu1997/EMAGE` HF Space is a demo front-end only, not a weights source. Requires a format-conversion adapter to translate its native output (55 joints Rot6D + FLAME parameters) into whatever conditioning format AVTR-1 and the Reaction Library trigger consume — build and test this adapter against a canned example before wiring live audio through it. |
| AVTR-1 | Hugging Face (weights) + GitHub (runtime) | Weights via `huggingface-cli download avaturn-live/avtr-1`; separate GitHub repo for inference/streaming code. **Face detection dependency — phased approach:** AVTR-1's LivePortrait-derived pipeline needs a face detector/landmark model for the appearance extraction step. **Phase 1 (current — pre-funding, internal evaluation only):** use InsightFace's pretrained detection models (buffalo_l/antelopev2) to evaluate quality. This is acceptable now because there is no commercial deployment, no paying customers, and no revenue — this matches the non-commercial research license InsightFace's models are actually released under. **Boundary to respect even in Phase 1:** keep this to internal pipeline testing. Once anything built with InsightFace is shown externally in a way that functions as promotion for the product — investor demos, a public YC-style demo video, anything distributed outside internal testing — that starts to blur into the kind of use the non-commercial license doesn't cover, so treat "internal only" as the real boundary, not just "pre-revenue." **Phase 2 gate (required before any public-facing release, paid launch, or external demo):** swap to a permissively-licensed alternative (MediaPipe Face Mesh, Apache 2.0) or obtain a commercial license from InsightFace directly. Build the face-detection step as a swappable module now specifically so this later swap is a config change, not a rebuild. Flag the phase decision explicitly in the report. |
| EchoMimicV3 | Hugging Face | Used offline only, for Reaction Library generation — not part of the live path, so real-time speed is not a requirement for this component. Standard download is sufficient; no TensorRT/Flash urgency since it doesn't run during live conversation. |

## Utilities and dependency setup — build this first, in the Docker layer
Before building any model-specific logic, set up a `utils/` module and the
Space's Dockerfile so every model's full dependency chain — including
things easy to miss — is accounted for in one place, not discovered
piecemeal as errors surface:
- Standard Python/CUDA/PyTorch base matched to what AVTR-1 and EMAGE
  actually require (verify versions against each repo's own requirements,
  don't assume they match each other — reconcile conflicts explicitly if
  they exist rather than picking one arbitrarily).
- Face detection/landmark dependency (see AVTR-1 row above) — install and
  pin whichever detector is chosen, not both.
- Audio codec and processing dependencies for PersonaPlex's Mimi-based
  audio tokenization.
- ONNX runtime / TensorRT if any component ends up on an accelerated
  format per the optimization section above.
- ffmpeg and any video muxing tools needed for the Reaction Library
  playback and crossfade blending.
- WebRTC serving dependencies for the Space's live output.
Document every non-obvious dependency directly in `utils/` with a short
comment on which model needs it and why — this file is the single source
of truth for "why is this package here" for future maintenance.

## Build order
1. **HF Space scaffold + Docker environment first.** Get the utils/
   dependency layer building cleanly on the Space before any model logic —
   confirm the Docker image builds and the Space boots on an empty
   placeholder app before adding real components.
2. **PersonaPlex standalone.** Full-duplex mic-in/speech-out working.
   Verify the dual-stream signal is genuinely continuous. Test text-prompt
   hot-swapping in isolation.
3. **EMAGE standalone**, against a canned audio file, output format
   confirmed.
4. **Build the EMAGE output adapter** feeding both AVTR-1's conditioning
   input and the Reaction Library's intensity-trigger signal.
5. **Build `cad.py`.**
6. **Build the Context Weaver**, tested against a recorded long session
   before connecting to a live one.
7. **Generate the Reaction Library offline**, using EchoMimicV3 with no
   real-time constraint. Build enough variety to avoid visible repetition.
8. **Build `avatar.py`**, wrapping AVTR-1 as the sole live renderer with
   full EMAGE conditioning, plus the Reaction Library playback/crossfade
   trigger logic.
9. **Build `director.py`** orchestrating CAD, EMAGE, and the Reaction
   Library trigger threshold.
10. **Build `pipeline.py`**, orchestrating PersonaPlex's full-duplex loop,
    the Context Weaver's background refresh, and the full event chain,
    end to end, on the placeholder reference image.
11. **End-to-end test on the Space itself**, not just locally — confirm
    the deployed Docker environment actually runs the full pipeline, not
    only that it runs in a dev environment. Test past the 160-240s mark to
    confirm the Context Weaver prevents PersonaPlex's native instability.
    Confirm the Reaction Library triggers only on genuinely high-intensity
    moments and crossfades cleanly, not on every gesture.

## Report back
Per-component: which ran clean vs. needed the 4-error research pivot and
what was found; final model formats used (optimized vs. full precision,
and why); the InsightFace/MediaPipe decision and rationale; confirmation
of the EMAGE weight source; whether PersonaPlex's text-prompt hot-swap
worked cleanly; Reaction Library clip count and variety; and confirmation
the Space builds and runs end-to-end in its actual Docker environment, not
only in local testing.
