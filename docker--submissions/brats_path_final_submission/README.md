# BraTS-Path 2026 submission package -- Virchow2 + H-optimus-1 + GenBio-PathFM

## MAJOR REWRITE: the Docker's scoring engine now uses your real ensemble class directly

Everything below this notice describes the package as it stood before this
change. Since then, you provided the actual final training/eval recipe
(`genbiopathfm_ivy_end_to_end.py`, `BratsPath2025ChunkedSGDEnsemble` --
chunk-SGD + Ridge probe heads on all 3 foundations, `source_mean`
aggregation, chunked feature-masking TTA), and comparing it against this
package's Docker-side scoring code (`inference_dependencies.py`) surfaced a
real, consequential bug:

- **The old Docker scoring code was NOT mathematically equivalent to your
  real ensemble**, despite a docstring claiming "byte-identical." It masked
  and TTA-averaged each foundation independently, combining foundations only
  at the very end; your real class draws one shared mask across all 3
  foundations per pass and combines them (`source_mean`) INSIDE each pass,
  before averaging across passes. It also re-seeded a fresh RNG per
  mini-batch, where your real class uses one continuous
  `np.random.default_rng(tta_seed)` advanced sequentially over the whole
  TTA loop.

**Fix:** rather than patch that reimplementation a second time, the whole
approach changed:

1. `genbiopathfm_ivy_end_to_end.py` is now saved in this package **verbatim**
   as the canonical source of truth, with one additive, opt-in
   `--export-docker-ckpts DIR` flag that serializes every fitted head
   (joblib) plus a `manifest.json`, using the SAME already-fitted `model`
   object `main()` produces -- no retraining, no separate code path.
2. `docker_template/src/brats_path_ensemble.py` is new: it vendors
   `BratsPath2025ChunkedSGDEnsemble`'s scoring methods (`_chunk_head_proba`,
   `_raw_avg_proba`, `_apply_rare_boost`, `_apply_decision_rule`,
   `_mask_bundle`, `predict_proba`, `predict_proba_tta`, `predict_tta`)
   **verbatim** -- same expressions, same order of operations -- stripped
   only of `.fit()` and everything under it, which inference never needs.
3. `inference.py` now extracts all 3 foundations' full embeddings, then
   calls this vendored class's `predict_proba_tta(X_bundle, tta_aug=16,
   tta_keep=0.9, tta_seed=42)` directly (optionally row-chunked purely for
   memory safety -- see `_score_ensemble_in_row_chunks`, which does not
   change the math, only bounds peak RAM for a large hidden test set). The
   old `inference_dependencies.py` aggregation/TTA code (manifest loading,
   head loading, `predict_foundation`, `exact_mask_for_foundation`) has been
   removed rather than left around as dead/confusing code.

**Verified, not just written:** a direct numerical test built a small
`BratsPath2025ChunkedSGDEnsemble`, fit it on synthetic data, and compared its
`predict_proba()` / `predict_proba_tta(tta_aug=16, tta_keep=0.9,
tta_seed=42)` output against the vendored `FittedChunkEnsemble` built from
the SAME fitted heads -- **exact bit-identical match (max abs diff = 0.0)**
for both. A second test ran the full `export_docker_checkpoints()` ->
joblib/manifest write -> `load_fitted_ensemble()` round-trip and got the
same exact match. A third test ran the actual `run_inference()` end-to-end
(shard discovery, per-foundation extraction loop, ensemble loading, TTA
scoring with row-chunking deliberately forced on, CSV/JSON writing) against
a synthetic tar shard and small synthetic heads, with only the foundation
model forward pass mocked (that part was separately verified earlier via
the `local-dir:`/vendored-GenBio-PathFM fixes) -- produced a correctly
formatted, complete `predictions.csv` and `inference_summary.json`.

**What you need to do:** re-run your real training recipe with the new flag
to (re-)generate the checkpoints this Docker image will actually use:

```bash
python genbiopathfm_ivy_end_to_end.py \
  --export-docker-ckpts docker_template/src/ckpts \
  [... your other real training args ...]
```

This is independent of `--tta-aug`/`--tta-keep`/`--tta-seed` on that
command -- those only affect that script's own printed val CSV/metrics, not
what gets exported (head fitting happens before TTA is ever applied). The
Docker container's own TTA settings default to `--tta-aug 16 --tta-keep 0.9
--tta-seed 42` regardless (see `BRATS_PATH_TTA_AUG` etc. below), matching
what you specified.

**One thing to double-check yourself:** whatever `scikit-learn` version is
installed on the machine that runs `--export-docker-ckpts` is what your
heads get pickled under. `requirements.txt` is currently pinned to
`scikit-learn==1.7.2` (chosen for the previous checkpoints, which were
pickled under 1.9.0 and needed a compatible-enough downgrade -- see that
file's own comment). If your real training run uses a different sklearn
version, confirm `requirements.txt`'s pin is actually compatible with
unpickling heads produced by it, the same way the 1.7.2 choice was
originally verified.

**New file in this package:** `docker_template/src/brats_path_ensemble.py`.

---

Built against the **actual** `BraTS-Path-2026-Docker-Submission-Kit` repo
(not just its README) -- `run.py`, `.dockerignore`, and all three
`scripts/*.sh` are copied verbatim; `src/webdataset_loader.py` is the kit's
own file with one additive helper; `src/inference.py` /
`src/inference_dependencies.py` follow `example_base_line/`'s worked
reference structure (Step-numbered flow, bracketed `[section]` logging,
`SubjectID,Prediction` CSV via the same `discover_tar_shards` /
`build_inference_dataloader` / `to_filename` utilities).

## What's in here

```
docker_template/                  the actual submission (build this)
  Dockerfile                      kit structure/comments; base image intentionally
                                   pinned to 2.3.1 instead of the kit's 2.2.0 default
                                   (2.3.1 is proven to build+run; 2.2.0 was never tested)
  requirements.txt                 kit's own pins (pillow, webdataset) + this ensemble's
                                   actual deps, incl. scikit-learn==1.7.2 (see below)
  .dockerignore                   kit's base excludes + defensive excludes added later
                                   for stray HF Hub cache dirs (hf_cache/, blobs/,
                                   snapshots/, models--*/) -- see below
  run.py, scripts/*.sh            copied verbatim from the real kit
  src/
    __init__.py
    brats_path_extract.py         3-foundation extraction; flat state-dict loading
                                   for ALL THREE foundations now, including
                                   genbiopathfm (kit's documented pattern, no
                                   AutoModel/trust_remote_code anywhere)
    genbio_pathfm_model.py         vendored copy of genbio-ai/genbio-pathfm's model
                                   class -- replaces the old trust_remote_code path
    GENBIO_PATHFM_LICENSE.txt, GENBIO_PATHFM_NOTICE.md   required attribution for
                                   the vendored GenBio-PathFM code (Non-Commercial
                                   license -- see below)
    webdataset_loader.py           the kit's own file + one additive progress-count helper
    inference_dependencies.py      manifest/heads loading, aggregation math, TTA RNG
    inference.py                   entrypoint: run_inference(input_dir, output_dir)
    ckpts/
      manifest.json                copied verbatim from your training run
      heads/                       all 24 joblib heads, copied verbatim
    foundation_model_weights/
      README.md                    what to populate (all 3 as flat weight files now)
prepare_submission.py              re-run this if you retrain, or to populate weights
```

## What changed after reading the real kit (not just its README)

- **Foundation weight loading was rebuilt.** The kit's actual worked example
  (`example_base_line/src/inference_dependencies.py`) shows loading a flat
  `model.safetensors`/`pytorch_model.bin` state dict directly into a
  hand-built model skeleton -- no HF Hub cache tree, no `hf-hub:` resolution.
  Virchow2 and H-optimus-1 now do exactly that: a `pretrained=False` timm
  skeleton + `model.load_state_dict(...)` from a local file. This is a real
  fix, not a stylistic one -- the earlier hf_cache-tree approach almost
  certainly didn't match what the kit (and likely the evaluator) expects.
- **GenBio-PathFM no longer uses `trust_remote_code` at all.** Its model
  class is now vendored directly as plain local Python
  (`src/genbio_pathfm_model.py`, an unmodified copy of the official
  `genbio-ai/genbio-pathfm` repo's `genbio_pathfm/model.py` -- the repo's own
  "Option 2: pip package" code path, rather than "Option 1: HuggingFace
  AutoModel"). `src/brats_path_extract.py` builds that class directly and
  loads a flat `model.pth` state dict into it (`torch.load(...,
  weights_only=True)` + `load_state_dict(..., strict=True)`), exactly the
  same pattern as virchow2/hoptimus1. No config.json, no HF snapshot
  directory, no repo-provided code executes at container runtime beyond what
  ships as reviewable source in this package. Verified end-to-end with a
  round-trip test (build skeleton -> save state dict -> reload -> forward
  pass -> shape check against what `extract_features()` expects).
  **Licensing note:** GenBio-PathFM's weights/code are under the GenBio AI
  Community License (Non-Commercial use, with required attribution) --
  `src/GENBIO_PATHFM_LICENSE.txt` and `src/GENBIO_PATHFM_NOTICE.md` are
  included to satisfy that; confirm with the organizers that a
  non-commercially-licensed component is acceptable for your submission
  track.
- **virchow2 and hoptimus1 had a second, separate offline-loading bug,
  caught only after the organizers confirmed containers run with `docker run
  --network none`.** Both used `timm.create_model("hf-hub:<repo>",
  pretrained=False, ...)`. It turns out timm's `"hf-hub:"` prefix
  *unconditionally* fetches `config.json` from the Hub to resolve the
  architecture, even when `pretrained=False` -- confirmed directly (raises
  `LocalEntryNotFoundError` with no network). Fix: `prepare_submission.py`
  now also caches each foundation's `config.json` locally (a small metadata
  file, fetched once alongside the weights, on a machine with network), and
  `load_foundation()` builds the architecture via timm's `"local-dir:<path>"`
  scheme instead -- which runs through the exact same config-parsing code,
  just reading from disk. Verified end-to-end offline (build -> save state
  dict -> reload via `"local-dir:"` -> `load_state_dict` -> match) with
  `HF_HUB_OFFLINE=1` and no network path available at all.
- **A third bug -- found only by actually running the built container**:
  `"local-dir:"` support doesn't exist in every timm version. The
  `requirements.txt` pin at the time (`timm==1.0.9`) predates it entirely --
  its `parse_model_name()` only accepts schemes `('', 'timm', 'hf-hub')`, so
  the fix above raised a bare `AssertionError` at container runtime despite
  working fine in local testing (which had happened to use a newer timm).
  Checked every point release between 1.0.9 and 1.0.28 directly (downloaded
  each wheel, grepped its `_factory.py`): `"local-dir:"` support was added in
  1.0.16. `requirements.txt` is now pinned to `timm==1.0.28` -- the exact
  version the fix was actually tested against -- and the real
  `load_foundation()` code path was re-verified end-to-end under that pin.
  **Lesson for anything else in this package**: local testing against
  whatever's already installed on a dev machine is not the same as testing
  against what `requirements.txt` actually pins inside the container: only
  the real `docker run` (with `--network none`, exactly as the organizers
  specified) caught this one.
- **A fourth issue, this time about build hygiene rather than correctness**:
  `prepare_submission.py`'s downloads go through whatever `HF_HOME` /
  `HF_HUB_CACHE` happens to be set to in the calling shell, which on at
  least one dev machine pointed *inside* `docker_template/src/`. That leaves
  a raw HF Hub download cache (`blobs/`, `snapshots/`, `models--<org>--<repo>/`
  -- easily multiple extra GB per foundation) sitting inside the Docker build
  context, which `COPY src/ /workspace/src/` would then bundle into the image
  wholesale, on top of the flat files that are actually needed.
  **An earlier draft of this fix forced every download through a brand-new
  scratch `cache_dir=`** -- this stopped the leak, but it also meant
  already-downloaded weights on your machine could never be found/reused,
  so every run re-downloaded everything from scratch (multiple GB, every
  time). That's now reverted: downloads use whatever HF cache your machine
  would normally resolve to (so anything you already have cached gets
  reused), and `prepare_submission.py` instead just **detects and warns**
  if that resolved location happens to sit inside the project
  (`_warn_if_cache_inside_project()`), telling you to point `HF_HOME`
  elsewhere rather than silently working around it. `.dockerignore` still
  excludes `hf_cache/`, `blobs/`, `snapshots/`, `models--*/` defensively
  either way, as a second line of defense regardless of what the warning
  catches. See `src/foundation_model_weights/README.md` for what should
  (and only should) be in that directory.
- **Logging now matches the kit's own convention**: bracketed `[inference]`,
  `[data]`, `[model]`, `[ensemble]` tags, `Step N: ... / Step N output: ...`
  framing, first-batch previews, end-of-run label-count summaries -- plus
  periodic progress lines with real totals/percentages/ETA (an addition,
  since a 16-pass TTA run over a full val set would otherwise be silent for
  long stretches under the kit's own minimal example).
- **`webdataset_loader.py`, `run.py`, and all three `scripts/*.sh` are the
  kit's actual files**, not reconstructions from its README -- copied
  verbatim (plus one additive helper in `webdataset_loader.py` for progress
  counts). `.dockerignore` started as the kit's own file too, with the
  `hf_cache`/etc. excludes above added on top.

## What's verified, concretely

- manifest.json + all 24 heads load and score correctly against real random
  input (checked again after this rewrite, including the new bracketed
  logging output, with `tta_aug=16` matching your official run).
- `scikit-learn==1.7.2` is pinned because `1.9.0` (what your heads were
  actually pickled with) requires Python>=3.11, which this Dockerfile's base
  image doesn't have -- confirmed by an actual failed build, then verified in
  a real end-to-end container run that predictions still matched a local
  reference.

## What I still could NOT verify here (no GPU access in my own environment)

1. **Foundation model weights aren't included in this package.** Run, on a
   machine with network access:

   ```
   python prepare_submission.py --populate-weights
   ```

   This extracts flat `model.safetensors` + `config.json` files for
   virchow2/hoptimus1 (via `timm.create_model(..., pretrained=True)` +
   `state_dict()`, saved with `safetensors.torch.save_file`, plus
   `config.json` fetched separately for the `"local-dir:"` runtime load
   path) and downloads genbiopathfm's flat `model.pth` file directly via
   `huggingface_hub.hf_hub_download`. Downloads reuse your existing local HF
   cache if you already have these models cached (no forced re-download);
   if your resolved HF cache happens to sit inside this project, the script
   warns you about it (see the fourth bullet above) instead of silently
   redirecting downloads elsewhere. Re-run your build/diff cycle after this
   changes.

2. **This package has now actually been run against the organizers' real
   `BraTS-Path2026-Docker-Sainity-Check-No-Label-shard-000000.tar`**
   (3,200 patches) -- on a Mac, so CPU-only/emulated rather than real GPU,
   but with `--network none` exactly as specified. That run is what
   surfaced and got fixes for: the missing `timm==1.0.28` pin (the earlier
   `1.0.9` pin predates `"local-dir:"` support entirely) and the stray
   `hf_cache/` build-context leak, both described above. **Still
   outstanding**: a real run on actual NVIDIA GPU hardware, for genuine
   timing/GPU-correctness signal -- CPU+emulation numbers aren't a valid
   stand-in for that.

3. **CSV output schema** (`SubjectID,Prediction`) now matches
   `example_base_line/`'s worked reference exactly, which is a stronger
   signal than before -- but still worth confirming against the actual
   BraTS-Path 2026 evaluator spec if one exists separately from this kit.

4. **TTA is on by default** (16 passes), exactly matching your original
   `official__inference_FINAL_clean_and_exact_tta.py`. Set
   `BRATS_PATH_TTA_AUG=0` to run clean-only instead (~17x faster, ~0.16
   percentage points of macro-F1 difference per your own validation numbers).
   The task's allocated compute is a fixed 24h wall-clock budget on a single
   A6000 (48GB VRAM) with 128GB RAM -- more generous on RAM/VRAM than the
   local `--memory=48G` test command, but 24h is a hard ceiling and hidden
   test set size isn't known ahead of time. **An earlier draft of this
   package added an automatic fallback that measured elapsed time and
   disabled TTA if projected total time looked likely to breach 24h -- this
   was removed on request: TTA now always runs exactly as configured via
   `BRATS_PATH_TTA_AUG`, full stop, no automatic override, matching the
   original solution literally.** This means the burden of confirming TTA
   finishes within budget is entirely on testing beforehand (see "Testing
   locally" below) -- there is no safety net if it doesn't.

## Testing locally against the organizers' exact run command

The organizers run submissions with **zero network access**
(`--network none`) and specific resource limits. Test with the same
constraints before submitting -- don't rely on the organizers to catch
run-time failures, since containers that fail to run or produce invalid
output are not evaluated:

```
docker run \
  --rm \
  --network none \
  --gpus=all \
  --volume /PATH/TO/INPUT:/input:ro \
  --volume /PATH/TO/OUTPUT:/output:rw \
  --memory=48G --shm-size=16G \
  <your-image>:<tag>
```

(use `--runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=0` instead of `--gpus=all`
if that's your local Docker setup)

**If you're on macOS (including Apple Silicon):** Docker Desktop for Mac has
no NVIDIA GPU/CDI support at all, regardless of chip -- `--gpus=all` will
fail outright with `no known GPU vendor found`. Drop the `--gpus` flag
entirely for local testing there (the code's `_auto_device()` already falls
back to CPU on its own); this only gets you a *functional* smoke test
(pipeline runs, output is well-formed), not a real GPU/timing test -- the
image is `linux/amd64`-only (the CUDA base image has no arm64 build), so a
Mac run is also CPU + emulated, likely 10-50x slower than the real hardware.
For genuine GPU/timing validation you need an actual Linux + NVIDIA machine
-- `docker save`/`docker load` the image over if needed.

After any run, validate the output with `check_predictions.sh input_dir
output_dir` (provided alongside this package) -- it checks the CSV header,
that every subject in the input tar(s) got exactly one prediction, that all
prediction values are in `[0, 9]`, and prints the key fields from
`inference_summary.json`.

Things this checks that a plain build+run without `--network none` would
NOT catch:
- Any accidental Hub/network resolution at runtime (this is exactly how the
  `hf-hub:` config-fetch bug above was caught -- it doesn't fail until you
  actually run with no network).
- `/input:ro` really is read-only -- confirm nothing in `src/` ever opens it
  for writing (checked: `webdataset_loader.py` only opens shards with
  `tarfile.open(shard, "r")`).
- `/output/predictions.csv` is written directly and flatly, matching the
  organizers' spec exactly (`src/inference.py` writes `output_dir /
  "predictions.csv"` with no subdirectory; it also writes an
  `inference_summary.json` alongside it, which is an extra diagnostic file,
  not a replacement for `predictions.csv`).
- Memory fits in 48G / 16G shm: the ensemble loads one foundation model at a
  time (`extract_foundation_embeddings` deletes each model + calls
  `torch.cuda.empty_cache()` before moving to the next foundation), so peak
  memory is roughly one foundation's weights + one batch of activations, not
  all three simultaneously.

## Next steps, in order

```
# 1. Re-export checkpoints from your REAL training run (see the notice at
#    the top of this file for why this replaces the old ckpts/):
python genbiopathfm_ivy_end_to_end.py \
  --export-docker-ckpts docker_template/src/ckpts \
  [... your other real training args ...]

# 2. Populate foundation model weights (unrelated to the heads above --
#    virchow2/hoptimus1/genbiopathfm feature extractors themselves):
python prepare_submission.py --populate-weights
python prepare_submission.py --make-sample-tar --val-glob "data/val-shard-*.tar" --n-samples 10

# 3. Build under a new name -- distinct from any image built against the
#    old, mathematically-incorrect scoring code:
cd docker_template
IMAGE_NAME=brats_path_ivy_stainaug_ridge IMAGE_TAG=latest ./scripts/01_build_image.sh
INPUT_DIR=../test_artifacts OUTPUT_DIR=../docker_output \
  IMAGE_NAME=brats_path_ivy_stainaug_ridge IMAGE_TAG=latest ./scripts/02_run_docker_image.sh
diff <(sort ../test_artifacts/local_reference_predictions.csv) \
     <(sort ../docker_output/predictions.csv)

# 4. Validate against the organizers' real sanity-check tar, with the
#    organizers' EXACT command (see "Testing locally" above -- `--network
#    none`, `--gpus=all`, `--memory=48G --shm-size=16G`), on real GPU
#    hardware if at all possible. Run check_predictions.sh against the
#    output.

# 5. Submission target referenced a Synapse Docker registry
#    (docker.synapse.org/PROJECT_ID/IMAGE_NAME:TAG) rather than an
#    Apptainer/SIF path -- confirm which your track actually wants before
#    using scripts/03_convert_tar_to_sif.sh; if it's Synapse, the path is:
docker login docker.synapse.org
docker tag brats_path_ivy_stainaug_ridge:latest docker.synapse.org/<project-id>/brats_path_ivy_stainaug_ridge:latest
docker push docker.synapse.org/<project-id>/brats_path_ivy_stainaug_ridge:latest
```

(`brats_path_ivy_stainaug_ridge` is a suggested name reflecting this recipe
-- Ivy GAP + stain augmentation + all-foundation Ridge probe -- swap in
whatever you prefer via `IMAGE_NAME=...`; nothing in the build/run scripts
hardcodes it.)
