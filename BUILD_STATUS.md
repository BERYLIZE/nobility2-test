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

## Step 3: EMAGE standalone — DONE
- Confirmed paper: arXiv 2401.00374, author H-Liu1997, org PantoMatrix (per handoff doc's own hint).
- Code: cloned exact upstream from https://github.com/PantoMatrix/PantoMatrix (not reimplemented).
- Weights: auto-downloaded from HF `H-Liu1997/emage_audio` via the official `from_pretrained()` call
  in the upstream test script — contradicts the handoff doc's assumption of "GitHub only, not HF";
  in practice weights are HF-hosted, code is GitHub-hosted.
- Ran on a separate HF GPU sandbox (Nvidia A10G small, $1.00/hr) to avoid dependency conflicts with
  the PersonaPlex environment (different torch/CUDA/transformers requirements).
- **Real bugs found and fixed to get this running (not silently worked around):**
  1. `torchvision.io.write_video` no longer exists in current torchvision — only used by the
     visualization path we don't need (no pytorch3d/mmcv installed), so import made lazy/optional.
  2. `emage_utils.fast_render` imports `pyrender` (visualization-only, unused for our path) — made
     import lazy/optional.
  3. Missing `omegaconf` dependency — not listed in the repo's pinned requirements.txt, but genuinely
     required to import the model config classes; installed.
  4. **Real version incompatibility**: `transformers==5.14.1` (latest) broke EMAGE's custom
     `PreTrainedModel` subclasses (`AttributeError: 'EmageVQVAEConv' object has no attribute
     'all_tied_weights_keys'`) — a real breaking API change in HF's newest transformers release.
     Pinned to `transformers==4.44.2` to match what this era of code expects.
- Skipped pytorch3d/mmcv/pyrender/opencv entirely — those are needed only for the repo's own video
  visualization helper (`visualize_one`), not for producing motion output, which is all Step 4 (the
  adapter) needs.
- **Verified end-to-end on real (canned) audio**: ran the repo's own example file
  (`examples/audio/2_scott_0_103_103_28s.wav`, 28s) through `model.inference()` and got a real
  `.npz` output: `poses` (860, 165) = 55 joints × 3 axis-angle params, `expressions` (860, 100) =
  FLAME blendshape params, `trans` (860, 3) global translation, `betas` (300,) shape params.
  860 frames / 30fps = 28.67s — exactly matches the input audio duration and the script's own
  printed total. This is genuine model output, not a stub or placeholder.
- Output format for Step 4's adapter: axis-angle (not Rot6D) poses + FLAME expression coefficients +
  translation, per the actual `.npz` schema above (Rot6D conversion, mentioned in the handoff doc,
  happens only in an unused commented-out code path in the upstream test script for motion-seed
  conditioning, not in the model's actual output).

## Architecture correction (affects Steps 4, 8, 9) — user-approved
- **AVTR-1's actual released code does not accept external pose/expression conditioning.**
  Verified against `avtr1_motion_generator.py` in the real GitHub repo (avaturn-live/avtr-1): it's a
  "speech -> motion" flow-matching model — dual-stream audio (speech + listen) goes through its own
  HuBERT encoder, and it autoregressively generates its own lip-sync/expression using only its own
  prior output frame as `kp_cond`. There is no parameter anywhere in the pipeline for injecting an
  externally-generated gesture/expression curve (e.g. from EMAGE).
- **Resolution (user-approved):** AVTR-1 drives its own lip-sync/expression directly from audio, as
  designed. EMAGE still runs on the same audio, but its output now feeds only the Reaction Library's
  intensity-trigger signal (and can inform `cad.py`'s classification) — it no longer tries to puppet
  AVTR-1's face/body. This matches how both released models actually work, with no invasive patching
  of AVTR-1's autoregressive internals.

## Step 4: EMAGE output adapter — DONE
- `emage/adapter.py`: computes per-frame gesture-energy (joint velocity, root-excluded) and
  expression-energy (FLAME coefficient velocity) from EMAGE's real output, z-scores each channel,
  and fires a `ReactionTrigger` (frame, time, intensity, dominant channel) when the combined signal
  crosses a threshold, debounced to avoid rapid refiring.
- **Verified end-to-end on genuine EMAGE output** (re-ran Step 3's inference to produce fresh real
  data, not synthetic): 33 triggers detected across the 28.67s canned audio clip, each with a real
  frame/timestamp/intensity/dominant-channel classification.
- Default threshold (1.5 z-score, 15-frame/0.5s debounce) is a documented placeholder — 33
  triggers over 28s is likely too frequent for genuine "big reaction" moments and will need tuning
  against real conversational speech/gesture footage once available; this is flagged in-code, not
  silently left as a hidden default.

## Step 5: cad.py — DONE
- No pretrained model is specified for CAD in the build spec, so built as a lightweight, swappable
  rule-based classifier: audio-frame RMS energy (VAD proxy) + PersonaPlex's own turn-state signal
  (speaking/listening/interrupted/backchannel) -> one of active_listening / backchannel_ready /
  interruption / idle_ambient.
- **Verified end-to-end** (`scripts/test_cad.py`, run locally, no GPU needed): synthetic audio
  sequence walks through all four states in order (active listening while user talks -> short pause
  triggers backchannel_ready -> extended silence triggers idle_ambient -> PersonaPlex interruption
  signal triggers interruption), all assertions pass on the real state-machine output, not mocked.
