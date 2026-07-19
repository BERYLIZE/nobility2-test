# EMAGE upstream patches (PantoMatrix/PantoMatrix)

Applied to `test_emage_audio.py` to run inference without the (unneeded, for
our purposes) video visualization stack:

```python
# torchvision.io.write_video no longer exists in current torchvision;
# only used by visualize_one(), which we don't call.
try:
    from torchvision.io import write_video
except ImportError:
    write_video = None

# emage_utils.fast_render imports pyrender; also only used by visualize_one().
try:
    from emage_utils import fast_render
except ImportError:
    fast_render = None
```

Dependency notes:
- `omegaconf` is required but missing from the repo's requirements.txt.
- `transformers` must be pinned to `4.44.2` — the latest (5.14.1 at time of
  testing) breaks EMAGE's custom `PreTrainedModel` subclasses with
  `AttributeError: 'EmageVQVAEConv' object has no attribute
  'all_tied_weights_keys'`.
- pytorch3d / mmcv / pyrender / opencv are NOT needed for motion generation,
  only for the repo's own video rendering helper. Skipped entirely.

Verified command: `python3 test_emage_audio.py` (no `--visualization` flag)
against the repo's own `examples/audio/2_scott_0_103_103_28s.wav`, producing
`examples/motion/2_scott_0_103_103_28s_output.npz` with real motion data
(poses/expressions/trans/betas — see BUILD_STATUS.md for shapes).
