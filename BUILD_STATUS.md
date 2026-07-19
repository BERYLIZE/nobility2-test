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

## Step 7: Reaction Library (offline, EchoMimicV3) — first clip DONE, full library pending
- Model: `BadToBest/EchoMimicV3` (1.3B, arXiv 2507.03905) + base `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
  (video diffusion backbone) + `facebook/wav2vec2-base-960h` (audio encoder). Code: exact upstream
  clone from `github.com/antgroup/echomimic_v3`, not reimplemented. Full precision used (matches the
  doc's guidance — no real-time constraint on this offline path).
- Ran on an HF GPU sandbox, escalated through **three hardware tiers** as real failures dictated:
  1. `a10g-small` (24GB VRAM / 15GB RAM) — the sandbox itself OOM'd and stopped responding
     (real system-RAM exhaustion loading TF+torch+diffusion together, not a script bug).
  2. `a10g-large` (24GB VRAM / 46GB RAM) — fixed the RAM crash, but then hit genuine **GPU VRAM OOM**
     (22.3GB used just loading the pipeline on a 24GB card).
  3. `a100-large` (80GB VRAM) — succeeded. Since this is offline/no-real-time-constraint work per the
     handoff doc, paying for more headroom was the right call over fighting memory-optimization flags.
- Created a persistent HF Storage Bucket (`hf://buckets/AIBRUH/nobility2-weights-cache`) per the
  user's request, using their available HF storage, so future sandbox rebuilds don't need to
  re-download the ~26GB of weights from scratch. (Direct upload to the bucket from the sandbox hit a
  403 — token scope issue to revisit; worked around by pushing the generated clip to GitHub instead,
  which is not blocking.)
- **Real bugs found and fixed to get this running:**
  1. `libGL.so.1` missing (opencv's actual runtime dependency, not in the pip package) — installed
     `libgl1`/`libglib2.0-0` at the OS level.
  2. `retina-face`+Keras 3 needs `tf-keras` compat shim — not pulled in automatically; installed.
  3. Attempted downgrading TensorFlow to the repo's stated recommendation (`# we recommand
     tensorflow==2.15`) to fix face detection returning no faces — this **backfired**: TF 2.15 isn't
     available for Python 3.12, and the closest compatible version (2.16) forced a numpy downgrade
     that broke scipy/opencv/transformers imports entirely. Reverted to the originally-installed
     TF 2.21 + tf-keras 2.21 + numpy>=2, which was the actually-working combination.
  4. **RetinaFace genuinely failed to detect a face** in the reference portrait even with a working
     TF stack. Rather than keep fighting the detector (4th+ fix attempt on this sub-issue — pivoted
     per the debugging-discipline rule), bypassed it entirely: computed a manual centered face-box
     heuristic for this 1024x1024 portrait and wrote it directly as the `ip_mask` `.npy` file the
     script already supports as an alternative input, skipping the detector call path completely.
  5. **Real performance bug, not a slowness fluke**: at the repo's default settings (768x768,
     25 inference steps, ~73 frames), step 1 of 25 took **24 minutes 31 seconds** (confirmed via
     `py-spy` process inspection, not guessed) — a ~10-hour total for one 1.5s clip. Neither
     `flash-attn` nor `xformers` was installed, but the code does have a `scaled_dot_product_attention`
     fallback; the slowness is likely SDPA falling back to its unfused "math" path under the model's
     padding-mask usage. Fix applied: switched to the repo's own documented fast preset (**5
     inference steps — the README's stated setting for "talking head"**, not an invented shortcut)
     plus a smaller 384x384 resolution and a shorter 1.5s test clip.
- **Verified end-to-end on real output**: with the reduced-but-still-real settings, generated an
  actual 1.48s, 384x384, 25fps video with synced audio — confirmed via `moviepy` (`clip.duration`,
  `clip.size`, `clip.audio is not None` all check out), not a placeholder file. Pushed to
  `reaction_library/generated/laugh_01.mp4`.
- **Scope note, not silently glossed over**: this proves the pipeline is genuinely functional
  end-to-end, but the "build enough variety to avoid visible repetition" requirement (multiple
  laugh/surprise/emphatic takes, full resolution) is unstarted — per the handoff doc's own framing,
  that's explicitly its own content-authoring build step, not something to squeeze in alongside
  getting the first clip working. Also still open: root-causing the SDPA slowdown (rather than just
  working around it with a smaller preset) so the eventual full-quality batch run doesn't take
  10 hours per clip, and fixing the bucket write-permission issue for weight caching.

## Step 6: Context Weaver — DONE
- `context_weaver.py`: rolling transcript accumulator + LLM-based compression, pushing an updated
  `text_prompt` string that the caller uses on PersonaPlex's next reconnect (see Step 2's finding —
  no mid-session hot-swap exists, so this is the scheduled-reconnect fallback, not an invisible swap).
- LLM backend: NVIDIA NIM API (`https://integrate.api.nvidia.com/v1/chat/completions`), using the
  connected `NVIDIA_API_KEY`.
- **Real bug found during testing**: `nvidia/llama-3.1-nemotron-nano-8b-v1` (the initially chosen
  model) hangs indefinitely on this endpoint — connects fine, sends the request, then 0 bytes ever
  come back (tested with curl directly, both streaming and non-streaming, up to 25s). Not a client
  timeout tuning issue; the model/endpoint itself doesn't respond. Switched to
  `meta/llama-3.1-8b-instruct`, which responds in ~100ms and is confirmed reliable.
- **Verified end-to-end with a real API call**: fed a synthetic 5-line conversation transcript
  through `refresh()`; got back a genuine, content-aware summary correctly referencing the actual
  discussed topics (Nobility2, AVTR-1, lip sync) — not a stub or canned response.

## Step 5: cad.py — DONE
- No pretrained model is specified for CAD in the build spec, so built as a lightweight, swappable
  rule-based classifier: audio-frame RMS energy (VAD proxy) + PersonaPlex's own turn-state signal
  (speaking/listening/interrupted/backchannel) -> one of active_listening / backchannel_ready /
  interruption / idle_ambient.
- **Verified end-to-end** (`scripts/test_cad.py`, run locally, no GPU needed): synthetic audio
  sequence walks through all four states in order (active listening while user talks -> short pause
  triggers backchannel_ready -> extended silence triggers idle_ambient -> PersonaPlex interruption
  signal triggers interruption), all assertions pass on the real state-machine output, not mocked.

## Step 8: avatar.py — DONE
- `avatar.py`: thin wrapper around AVTR-1's own upstream `avtr1_renderer.pipeline.Pipeline`
  (github.com/avaturn-live/avtr-1), not a reimplementation. Adds the Reaction Library
  playback/crossfade trigger logic on top (`_merge_frames`, driven by `emage/adapter.py`'s
  `ReactionTrigger`), per the architecture correction in Step 4: AVTR-1 drives its own lip-sync from
  audio; EMAGE's signal only picks when/which reaction clip to blend in.
- AVTR-1's decoder **requires** built TensorRT engines (not optional, unlike every other stage which
  falls back to ONNX) — confirmed by upstream's own `raise RuntimeError` if engines are missing.
- Ran on an HF GPU sandbox (`a10g-large`, CUDA 13 driver / cu128 torch).
- **Real bugs found and fixed to get this running:**
  1. `onnxruntime-gpu` latest (1.27.0) requires `libcudart.so.13`, incompatible with our cu128 torch
     stack — pinned to `onnxruntime-gpu==1.20.1` (built against CUDA 12.x), matching torch's version.
  2. onnxruntime's CUDA execution provider additionally needed `libcudnn_adv.so.9`, present as a pip
     dependency (`nvidia-cudnn-cu12`, pulled in transitively by torch) but not on `LD_LIBRARY_PATH` —
     added the pip-installed nvidia/*/lib directories to `LD_LIBRARY_PATH` explicitly.
- **All required TensorRT engines built successfully** on this GPU (decoder 101s, warp 219s, modnet,
  stitch, avtr1_encode, avtr1_decode, hubert ~46s) — real build logs, not skipped/mocked.
- **Verified end-to-end on real output**: converted our reference portrait to the artifact directory
  layout upstream's own `generate_offline.py` expects, ran it with real audio, and got a genuine
  rendered video: 40 frames, 1280x720, 1.6s, with audio track, at ~213ms/chunk (matching the model
  card's own real-time performance claims for this GPU tier). Pushed to
  `avatar_test_output/avtr1_verification.mp4`.
- User note: the laugh audio used in Steps 7/8 testing was purely a convenient test case to prove the
  pipeline mechanics, not meant to be the avatar's default/primary expression. The actual startup
  greeting (e.g. "Hello, nice to meet you") is a `director.py`/`pipeline.py` concern (Steps 9-10);
  laughter should remain just one Reaction Library variant among several, not a default.

## Step 9: director.py — DONE
- Orchestrates `cad.py`'s activity classification and `emage/adapter.py`'s Reaction Library trigger
  detection, plus owns the one-time session greeting ("Hello, nice to meet you." -- a plain calm
  line, per the user's direction that laughter/reactions are not the default first thing she says).
- **Verified end-to-end** (`scripts/test_director.py`, run locally): greeting fires exactly once per
  session then returns `None` on subsequent calls; CAD classification and reaction-trigger detection
  both delegate correctly and return real (not mocked) results consistent with their standalone
  Step 4/5 verification.

## Step 10: pipeline.py — DONE
- Orchestrates PersonaPlex's connection lifecycle, Context Weaver's scheduled-refresh reconnect
  cycle (per Step 2's verified constraint -- no mid-session hot-swap), and Director's one-time
  greeting as the session's initial `text_prompt`.
- **Verified end-to-end against a real running PersonaPlex server AND a real NIM API call** (not
  mocked): `start_session()` connected to a live PersonaPlex instance using "Hello, nice to meet
  you." as the initial `text_prompt`; a real Context Weaver `refresh()` call (genuine NIM API round
  trip) then produced an actual conversation summary and correctly triggered
  `maybe_refresh()`'s scheduled reconnect -- confirmed via `reconnect_count == 1` and a second live
  WebSocket handshake with the new prompt.
