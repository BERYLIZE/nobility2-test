"""Exercise cad.py's classify() through all four states with synthetic audio,
confirming the state machine actually transitions correctly, not just imports."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cad import ActivityState, TurnState, CADConfig, CADState, classify

config = CADConfig(energy_silence_threshold=0.01, pause_frames_for_idle=5, backchannel_pause_frames=3)
state = CADState()

loud_frame = (0.3 * np.sin(np.linspace(0, 6.28, 480))).astype(np.float32)
silent_frame = np.zeros(480, dtype=np.float32)

results = []

# 1. User actively talking while agent listens -> active_listening
for _ in range(3):
    results.append(classify(loud_frame, TurnState.LISTENING, state, config))

# 2. Short pause while listening -> backchannel_ready
for _ in range(2):
    results.append(classify(silent_frame, TurnState.LISTENING, state, config))

# 3. Extended silence -> idle_ambient
for _ in range(6):
    results.append(classify(silent_frame, TurnState.LISTENING, state, config))

# 4. Interruption signal from PersonaPlex turn-state -> interruption
results.append(classify(loud_frame, TurnState.INTERRUPTED, state, config))

print([r.value for r in results])

assert results[0] == ActivityState.ACTIVE_LISTENING, f"expected active_listening, got {results[0]}"
assert results[3] == ActivityState.BACKCHANNEL_READY, f"expected backchannel_ready, got {results[3]}"
assert results[-2] == ActivityState.IDLE_AMBIENT, f"expected idle_ambient, got {results[-2]}"
assert results[-1] == ActivityState.INTERRUPTION, f"expected interruption, got {results[-1]}"
print("PASS: all four CAD states reached correctly")
