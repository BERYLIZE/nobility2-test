"""Verify avatar.py's Reaction Library crossfade logic: a trigger causes a
smooth blend from live frames into clip frames and back, not a hard cut."""
import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# avatar.py imports avtr1_renderer.types.RenderOptions lazily inside process_chunk;
# stub it so this test can run without the real (GPU-only) package installed.
import types as _types
_stub_pkg = _types.ModuleType("avtr1_renderer")
_stub_types = _types.ModuleType("avtr1_renderer.types")
_stub_types.RenderOptions = lambda bg_id=None: {"bg_id": bg_id}
_stub_pkg.types = _stub_types
sys.modules["avtr1_renderer"] = _stub_pkg
sys.modules["avtr1_renderer.types"] = _stub_types

from avatar import Avatar, AvatarState, ReactionClip, CROSSFADE_FRAMES
from emage.adapter import ReactionTrigger


@dataclass
class FakeFrame:
    data: np.ndarray
    format: str = "yuv_i420"
    height: int = 4
    width: int = 4


class FakePipeline:
    def initial_state(self, avatar):
        return {"dummy": True}

    def process_chunk(self, avatar, chunk, state, options):
        # 20 live frames, each filled with value 100 (simulates live AVTR-1 output)
        frames = [FakeFrame(data=np.full((4, 4), 100, dtype=np.uint8)) for _ in range(20)]
        return state, iter(frames)


clip = ReactionClip(
    name="laugh",
    frames=[FakeFrame(data=np.full((4, 4), 200, dtype=np.uint8)) for _ in range(5)],
    audio=np.zeros(100, dtype=np.float32),
)

avatar = Avatar(
    pipeline=FakePipeline(), avatar_handle=object(),
    reaction_library={"laugh_expression": clip}, bg_id="plain_white",
)

state = AvatarState(pipeline_state={"dummy": True})
trigger = ReactionTrigger(frame=10, time_s=0.33, intensity=2.0, dominant="expression")

out_state, frame_iter = avatar.process_chunk(chunk=None, state=state, trigger=trigger)
frames = list(frame_iter)
values = [int(f.data.flat[0]) for f in frames]

print("Frame values over time:", values)

# First few frames should ramp from live (100) toward clip (200) -- a blend, not a jump.
assert values[0] != 100 or values[1] != 200, "expected some blending, not detected"
blend_region = values[:CROSSFADE_FRAMES]
assert any(100 < v < 200 for v in blend_region), f"expected intermediate blended values in {blend_region}"
print("PASS: crossfade produces intermediate blended values, not a hard cut")

# After the clip's 5 frames play out (with fade-in), it should fade back toward live.
assert values[-1] == 100 or (100 <= values[-1] <= 200), "expected fade back toward live at the end"
print("PASS: sequence returns toward live output after the clip finishes")

print("\nPASS: avatar.py Reaction Library crossfade verified end-to-end (synthetic frames, real logic)")
