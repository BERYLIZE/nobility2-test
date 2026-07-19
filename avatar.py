"""avatar.py -- AVTR-1 as the sole live renderer, with Reaction Library
playback/crossfade triggered by EMAGE's gesture-intensity adapter.

Thin wrapper around AVTR-1's own upstream Pipeline (avtr1_renderer.pipeline),
not a reimplementation, per the build spec's standing order to wrap and adapt
third-party model code rather than rewrite it.

Architecture note (see BUILD_STATUS.md, Step 4/8): AVTR-1 drives its own
lip-sync and expression autoregressively from its own dual-stream audio
encoder -- it has no input hook for externally supplied pose/expression
curves. EMAGE's output does not condition AVTR-1 here; it only feeds the
Reaction Library trigger via emage/adapter.py's ReactionTrigger.

Dependency note: AVTR-1's decoder REQUIRES built TensorRT engines (not
optional, unlike every other stage which falls back to ONNX). Building those
engines (scripts/build_avtr1_engines.py in the avtr1-code upstream repo)
must happen once per target GPU before Pipeline.from_artifacts() will
succeed -- see the AVTR-1 row in HANDOFF.md and the Step 8 section of
BUILD_STATUS.md for status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from emage.adapter import ReactionTrigger

CROSSFADE_FRAMES = 8  # short blend in/out, per the handoff doc's "not a hard cut" requirement


@dataclass
class ReactionClip:
    """A pre-rendered Reaction Library clip (see Step 7): frames as a list of
    np.ndarray in the same pixel format AVTR-1 yields, plus its own audio."""
    name: str
    frames: list  # list[Frame]-compatible; kept generic to avoid importing avtr1_renderer at module load
    audio: np.ndarray


@dataclass
class AvatarState:
    """Wraps the upstream pipeline's opaque per-session state plus our own
    crossfade bookkeeping."""
    pipeline_state: object = None
    active_clip: Optional[ReactionClip] = None
    active_clip_frame_idx: int = 0
    crossfade_remaining: int = 0
    fade_reference_frame: object = None  # last clip frame, held for the fade-back-to-live blend


def _env_float(name: str, default: float) -> float:
    import os
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Avatar:
    """Wraps AVTR-1's Pipeline + a Reaction Library, exposing one
    process_chunk-like call that transparently swaps in reaction clips on
    high-intensity triggers instead of live-rendering them.

    Realism tuning (all env-overridable, no engine rebuild needed -- these
    are per-chunk RenderOptions inputs per upstream types.py):
    - cfg_self_audio: how strongly her OWN speech drives articulation.
      Slightly above upstream default (2.0 -> 2.3) for crisper lip-sync.
    - cfg_other_audio: how strongly the USER's speech drives listening
      reactions (nods, micro-expressions). Upstream default 2.0.
    - cfg_kp: identity/pose stability. Upstream default 3.0 -- kept; raising
      it flattens expressiveness, lowering it drifts identity.
    - noise_trunc_z: caps the AR(1) idle-motion noise. Slightly below
      upstream default (1.2 -> 1.1) to trim occasional jerky outliers while
      keeping natural micro-motion alive.
    """

    def __init__(self, pipeline, avatar_handle, reaction_library: dict[str, ReactionClip],
                 bg_id: str, intensity_threshold: float = 1.5):
        self._pipeline = pipeline
        self._avatar = avatar_handle
        self._reaction_library = reaction_library
        self._bg_id = bg_id
        self._intensity_threshold = intensity_threshold
        self._render_options = None  # built lazily; needs avtr1_renderer importable

    def _build_render_options(self):
        from avtr1_renderer.types import RenderOptions
        return RenderOptions(
            bg_id=self._bg_id,
            cfg_self_audio=_env_float("NOBILITY2_CFG_SELF_AUDIO", 2.3),
            cfg_other_audio=_env_float("NOBILITY2_CFG_OTHER_AUDIO", 2.0),
            cfg_kp=_env_float("NOBILITY2_CFG_KP", 3.0),
            noise_alpha=_env_float("NOBILITY2_NOISE_ALPHA", 2.0),
            noise_trunc_z=_env_float("NOBILITY2_NOISE_TRUNC_Z", 1.1),
        )

    def initial_state(self) -> AvatarState:
        return AvatarState(pipeline_state=self._pipeline.initial_state(self._avatar))

    def process_chunk(self, chunk, state: AvatarState, trigger: Optional[ReactionTrigger] = None):
        """Render one chunk's worth of frames.

        If `trigger` is set (from emage/adapter.py, driven by the same
        audio) and no reaction is already playing, start blending from live
        AVTR-1 output into the matching reaction clip. Playback of the clip
        itself doesn't consume live pipeline compute -- but the live
        pipeline keeps running underneath so we can crossfade back out.
        """
        if self._render_options is None:
            self._render_options = self._build_render_options()

        pipeline_state, live_frames = self._pipeline.process_chunk(
            self._avatar, chunk, state.pipeline_state, self._render_options,
        )
        state.pipeline_state = pipeline_state

        if trigger is not None and state.active_clip is None and self._reaction_library:
            clip = self._reaction_library.get(self._pick_clip_name(trigger))
            if clip is not None:
                state.active_clip = clip
                state.active_clip_frame_idx = 0
                state.crossfade_remaining = CROSSFADE_FRAMES

        return state, self._merge_frames(live_frames, state)

    def _pick_clip_name(self, trigger: ReactionTrigger) -> str:
        # Gesture-dominant spikes -> emphatic-gesture clips; expression-dominant -> laugh/surprise.
        # Simple v1 mapping; refine once the library has real variety (Step 7 follow-on).
        candidates = [name for name in self._reaction_library if trigger.dominant in name] or list(self._reaction_library)
        return candidates[0]

    def _merge_frames(self, live_frames, state: AvatarState):
        """Yield either live frames, clip frames, or a crossfaded blend,
        advancing state.active_clip playback and the fade counter."""
        for live_frame in live_frames:
            if state.active_clip is None:
                if state.crossfade_remaining > 0 and state.fade_reference_frame is not None:
                    # Fading back out of the just-finished clip toward live.
                    alpha = state.crossfade_remaining / CROSSFADE_FRAMES
                    yield _blend_frames(state.fade_reference_frame, live_frame, alpha)
                    state.crossfade_remaining -= 1
                    if state.crossfade_remaining == 0:
                        state.fade_reference_frame = None
                else:
                    yield live_frame
                continue

            clip = state.active_clip
            if state.active_clip_frame_idx >= len(clip.frames):
                # Clip finished -- start blending back to live using its last frame as reference.
                state.fade_reference_frame = clip.frames[-1]
                state.active_clip = None
                state.crossfade_remaining = CROSSFADE_FRAMES
                alpha = state.crossfade_remaining / CROSSFADE_FRAMES
                yield _blend_frames(state.fade_reference_frame, live_frame, alpha)
                state.crossfade_remaining -= 1
                continue

            clip_frame = clip.frames[state.active_clip_frame_idx]
            state.active_clip_frame_idx += 1

            if state.crossfade_remaining > 0:
                alpha = state.crossfade_remaining / CROSSFADE_FRAMES
                yield _blend_frames(live_frame, clip_frame, alpha)
                state.crossfade_remaining -= 1
            else:
                yield clip_frame


def _blend_frames(a, b, alpha: float):
    """Linear crossfade between two Frame-like objects with a `.data` array."""
    blended_data = (alpha * a.data.astype(np.float32) + (1 - alpha) * b.data.astype(np.float32)).astype(a.data.dtype)
    return type(a)(data=blended_data, format=a.format, height=a.height, width=a.width)


def build_avatar_from_config(config_path: str = "config/reference.json", bg_id: str = "transparent") -> Avatar:
    """Load the Pipeline via AVTR-1's own from_artifacts(), using our
    configured reference identity image as the sole avatar portrait."""
    import json
    from avtr1_renderer.pipeline import Pipeline

    with open(config_path) as f:
        cfg = json.load(f)
    identity_path = Path(cfg["identity_image_path"])

    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=[identity_path.stem],
        portraits_dir=identity_path.parent,
    )
    avatar_handle = registry[identity_path.stem]
    return Avatar(pipeline=pipeline, avatar_handle=avatar_handle, reaction_library={}, bg_id=bg_id)
