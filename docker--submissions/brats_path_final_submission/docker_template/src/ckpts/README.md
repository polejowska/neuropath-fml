# Checkpoints

Place model checkpoint files here if your inference implementation loads
local weights from `src/ckpts`.

This directory contains your trained ensemble, copied verbatim from
`official_models/vhg_aug_ivy14950_source_soft_full_exact_v1/`:

```
manifest.json          schema "official-vhg-separate-heads-exact-internal-semantics-v1"
                        foundations, chunking, rare_classes/rare_boost,
                        thresholds, and the list of all 24 head records
                        (foundation/classifier/start/stop/path/classes)
heads/virchow2/...      8 heads  (4 chunks x {sgd_lr_log_a3e-5, ridge_lsqr_a10})
heads/hoptimus1/...     4 heads  (2 chunks x {sgd_lr_log_a3e-5, ridge_lsqr_a10})
heads/genbiopathfm/...  12 heads (6 chunks x {sgd_lr_log_a3e-5, ridge_lsqr_a10})
```

`src/inference_dependencies.py` reads `manifest.json` directly (foundations,
per-foundation chunk boundaries, rare_classes, rare_boost, thresholds) rather
than hardcoding any of this, so re-running `prepare_submission.py` against a
newer/retrained model directory with the same schema requires no code changes
here.

Keep checkpoint paths relative to `src/inference.py` so the Docker image is
self-contained.

Not copied (not needed for inference): `calibration/`, `reports/`, `split/`
from the training output. Those hold calibration-set diagnostics from
training and aren't read by the packaged container.
