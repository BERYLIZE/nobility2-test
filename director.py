"""director.py -- orchestrates CAD, EMAGE's Reaction Library trigger, and the
avatar's startup greeting.

Sits between the raw audio/frame stream and avatar.py: classifies each frame
via cad.py, runs EMAGE's adapter to decide if a reaction should fire, and
owns the one-time greeting behavior when a session starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cad import ActivityState, CADConfig, CADState, TurnState, classify
from emage.adapter import ReactionTrigger, detect_triggers

# Per user direction: the startup greeting is a simple, calm line -- not a
# reaction-library clip. Laughter and other high-intensity reactions are
# triggered later, during conversation, by real gesture/expression spikes;
# they are not the default first thing she says.
GREETING_TEXT = "Hello, nice to meet you."


@dataclass
class DirectorState:
    cad_state: CADState = field(default_factory=CADState)
    greeted: bool = False


class Director:
    def __init__(self, reaction_threshold: float = 1.5, reaction_min_gap_frames: int = 15):
        self._reaction_threshold = reaction_threshold
        self._reaction_min_gap_frames = reaction_min_gap_frames

    def initial_state(self) -> DirectorState:
        return DirectorState()

    def maybe_greeting(self, state: DirectorState) -> Optional[str]:
        """Returns the greeting text exactly once per session, on the first
        call, then None afterward. Caller is responsible for actually
        speaking/rendering it (e.g. via PersonaPlex's text_prompt or a
        pre-rendered greeting clip)."""
        if state.greeted:
            return None
        state.greeted = True
        return GREETING_TEXT

    def classify_activity(
        self,
        pcm_frame: np.ndarray,
        turn_state: TurnState,
        state: DirectorState,
        config: CADConfig = CADConfig(),
    ) -> ActivityState:
        return classify(pcm_frame, turn_state, state.cad_state, config)

    def check_reaction_triggers(self, poses: np.ndarray, expressions: np.ndarray, frame_rate: float) -> list[ReactionTrigger]:
        """Run EMAGE's adapter over a window of motion output to decide if
        a Reaction Library clip should fire. Does not consider the greeting
        -- that's handled separately via maybe_greeting()."""
        return detect_triggers(
            poses, expressions, frame_rate,
            threshold=self._reaction_threshold,
            min_gap_frames=self._reaction_min_gap_frames,
        )
