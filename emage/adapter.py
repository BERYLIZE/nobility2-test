"""EMAGE output adapter.

Feeds the Reaction Library's intensity-trigger signal only. AVTR-1's own
released code (avtr1_motion_generator.py) turns out to generate lip-sync and
expression autoregressively from its own dual-stream audio encoder, with no
input hook for externally supplied pose/expression curves -- so EMAGE's
motion output does NOT drive AVTR-1 (verified against upstream source, not
assumed). EMAGE still runs on the same audio and its output is the input to
this adapter, which produces the gesture-intensity signal that triggers
Reaction Library clip playback.

Input: the exact .npz schema EMAGE's own test_emage_audio.py produces
(confirmed on real audio in Step 3):
  poses       (T, 165)  -- 55 joints x 3 axis-angle params
  expressions (T, 100)  -- FLAME expression coefficients
  trans       (T, 3)    -- global translation
  mocap_frame_rate       -- scalar, frames per second (30 in the reference run)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_JOINTS = 55
ROOT_JOINT_DIMS = 3  # first joint = global root orientation, excluded from gesture energy

# Empirically reasonable defaults; tune once real speech/gesture footage is available.
DEFAULT_THRESHOLD = 1.5
DEFAULT_MIN_GAP_FRAMES = 15  # debounce: don't refire within half a second at 30fps


@dataclass
class ReactionTrigger:
    frame: int
    time_s: float
    intensity: float
    dominant: str  # "gesture" or "expression" -- which channel drove the spike


def compute_intensity_signal(poses: np.ndarray, expressions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (gesture_energy, expression_energy), each L2 velocity magnitude."""
    joint_poses = poses.reshape(poses.shape[0], N_JOINTS, 3)[:, 1:]  # drop root joint
    gesture_velocity = np.diff(joint_poses, axis=0, prepend=joint_poses[:1])
    gesture_energy = np.linalg.norm(gesture_velocity.reshape(gesture_velocity.shape[0], -1), axis=1)

    expr_velocity = np.diff(expressions, axis=0, prepend=expressions[:1])
    expression_energy = np.linalg.norm(expr_velocity, axis=1)

    return gesture_energy, expression_energy


def detect_triggers(
    poses: np.ndarray,
    expressions: np.ndarray,
    frame_rate: float,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES,
) -> list[ReactionTrigger]:
    gesture_energy, expression_energy = compute_intensity_signal(poses, expressions)

    # Normalize each channel independently (z-score against its own running stats)
    # so gesture and expression energy are comparable before combining.
    def zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / std if std > 1e-8 else np.zeros_like(x)

    g_z = zscore(gesture_energy)
    e_z = zscore(expression_energy)
    combined = np.maximum(g_z, e_z)

    triggers: list[ReactionTrigger] = []
    last_trigger_frame = -min_gap_frames
    for frame, score in enumerate(combined):
        if score < threshold:
            continue
        if frame - last_trigger_frame < min_gap_frames:
            continue
        dominant = "gesture" if g_z[frame] >= e_z[frame] else "expression"
        triggers.append(ReactionTrigger(frame=frame, time_s=frame / frame_rate, intensity=float(score), dominant=dominant))
        last_trigger_frame = frame

    return triggers


def load_and_detect(npz_path: str, **kwargs) -> list[ReactionTrigger]:
    data = np.load(npz_path, allow_pickle=True)
    frame_rate = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data.files else 30.0
    return detect_triggers(data["poses"], data["expressions"], frame_rate, **kwargs)
