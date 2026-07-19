"""Verify director.py's greeting-once behavior and that it correctly
delegates to cad.py and emage/adapter.py (both already verified standalone)."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from director import Director
from cad import ActivityState, TurnState

director = Director()
state = director.initial_state()

# Greeting fires exactly once
g1 = director.maybe_greeting(state)
g2 = director.maybe_greeting(state)
assert g1 == "Hello, nice to meet you.", f"expected greeting text, got {g1!r}"
assert g2 is None, f"greeting should not repeat, got {g2!r}"
print(f"PASS: greeting fired once: {g1!r}, then None")

# CAD delegation still works through the director
loud_frame = (0.3 * np.sin(np.linspace(0, 6.28, 480))).astype(np.float32)
result = director.classify_activity(loud_frame, TurnState.LISTENING, state)
assert result == ActivityState.ACTIVE_LISTENING, f"expected active_listening, got {result}"
print(f"PASS: CAD delegation works: {result.value}")

# Reaction trigger delegation still works through the director (synthetic motion data)
T = 60
poses = np.zeros((T, 165), dtype=np.float32)
poses[30:35] += np.random.randn(5, 165).astype(np.float32) * 5  # spike mid-sequence
expressions = np.zeros((T, 100), dtype=np.float32)
triggers = director.check_reaction_triggers(poses, expressions, frame_rate=30.0)
assert len(triggers) > 0, "expected at least one trigger from the injected spike"
print(f"PASS: reaction trigger delegation works: {len(triggers)} trigger(s) found")

print("\nPASS: director.py verified end-to-end")
