"""Conversational Activity Detector (CAD).

Classifies each moment into one of: active_listening, backchannel_ready,
interruption, idle_ambient. Feeds EMAGE as a conditioning input alongside
raw audio (per the architecture: CAD -> EMAGE, not the reverse).

No pretrained model is specified for this component in the build spec --
built here as a lightweight, swappable rule-based classifier over audio
energy (VAD) and PersonaPlex's own turn-state signal (speaking / listening /
interrupted / backchannel), which PersonaPlex's server already exposes via
its dual-stream design (see moshi/server.py: separate self/other Mimi
streams). A model-based classifier can replace `classify()` later without
touching callers -- the state enum and frame-based interface are the
contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ActivityState(str, Enum):
    ACTIVE_LISTENING = "active_listening"
    BACKCHANNEL_READY = "backchannel_ready"
    INTERRUPTION = "interruption"
    IDLE_AMBIENT = "idle_ambient"


class TurnState(str, Enum):
    SPEAKING = "speaking"
    LISTENING = "listening"
    INTERRUPTED = "interrupted"
    BACKCHANNEL = "backchannel"


@dataclass
class CADConfig:
    energy_silence_threshold: float = 0.01  # RMS below this = silence
    pause_frames_for_idle: int = 45  # ~1.5s at 30fps-equivalent frame rate
    backchannel_pause_frames: int = 10  # short pause while listening -> backchannel-ready window


@dataclass
class CADState:
    """Rolling state carried across calls to `classify` (streaming use)."""
    silence_run: int = 0
    speech_run: int = 0


def frame_energy(pcm_frame: np.ndarray) -> float:
    """RMS energy of a raw PCM frame, used as a lightweight VAD proxy."""
    if pcm_frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm_frame.astype(np.float32)))))


def classify(
    pcm_frame: np.ndarray,
    turn_state: TurnState,
    state: CADState,
    config: CADConfig = CADConfig(),
) -> ActivityState:
    """Classify one frame's conversational activity.

    `turn_state` comes from PersonaPlex's own dual-stream signal (which
    stream is currently producing speech tokens). `pcm_frame` is the raw
    audio for this frame, used to detect pauses/energy the turn-state alone
    doesn't capture (e.g. a listening user going silent vs. still talking).
    """
    energy = frame_energy(pcm_frame)
    is_silent = energy < config.energy_silence_threshold

    if is_silent:
        state.silence_run += 1
        state.speech_run = 0
    else:
        state.speech_run += 1
        state.silence_run = 0

    if turn_state == TurnState.INTERRUPTED:
        return ActivityState.INTERRUPTION

    if turn_state == TurnState.LISTENING:
        if state.silence_run >= config.pause_frames_for_idle:
            return ActivityState.IDLE_AMBIENT
        if 0 < state.silence_run < config.backchannel_pause_frames:
            return ActivityState.BACKCHANNEL_READY
        return ActivityState.ACTIVE_LISTENING

    if turn_state == TurnState.BACKCHANNEL:
        return ActivityState.BACKCHANNEL_READY

    # SPEAKING: the agent itself is talking, not the user -- ambient by definition
    # for the listener-facing classification this module targets.
    return ActivityState.ACTIVE_LISTENING
