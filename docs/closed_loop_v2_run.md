# Closed-loop v2: fixes and running

This revision fixes training and validation semantics on top of `78192f2`.
The controller supplies internally supervised lesion feedback; it is not an
independent downstream segmentation model, and no-harm remains a soft penalty.

## Changes

- Weighted Dice/Tversky now average across samples. Previously their magnitudes
  grew with batch size. New runs record `segmentation_loss_version=2`; the old
  behavior is available explicitly as version 1 for legacy reproduction.
- Validation version 2 measures each slice separately before aggregation.
  Batch size no longer changes lesion thresholds or metric weighting. It is
  still slice-level validation; full-volume evaluation remains patient-level.
- `closed_loop_noharm_composite` includes regression against the adapter-free
  reference and reports ET/TC/WT reference errors and harm rates. The reference
  is the current base, or a frozen base when explicitly requested. It does not
  include the optional stage-2 refinement.
- Removed the unsupervised, unused failure head. Inference accepts its obsolete
  checkpoint keys but rejects missing active transport weights. Old closed-loop
  optimizer checkpoints containing this removed head require a warm start.
- Detached feedback training is rejected because its projection has no direct
  supervision. Historical inference configuration remains loadable.
- Freezing the base requires a checkpoint and rejects missing/incompatible base
  weights. Training must still backpropagate through the frozen base to earlier
  adapter steps; it is deliberately not wrapped in `no_grad()`.
- Full-volume generation skips the independent reference trajectory without
  changing generated images or uncertainty. Training retains it when no-harm
  losses need it, and validation retains it for reference metrics.
- Slice loading memory-maps volumes and converts only the selected image slices.
  Only the last controller lesion head is computed, and baseline velocity
  outputs reuse existing storage when the adapter is disabled.

## Existing data and environment: joint training

The following uses the existing processed BraTS manifest and current Python
environment, without downloading data or creating another environment. Wait
until the GPU has enough free memory; the existing H200 job was still using
about 136 GB during verification. Do not run a second full job beside it.

```bash
cd /home/yuey21/neuroState3D

SKIP_ENV_SETUP=1 SKIP_DOWNLOAD=1 SKIP_PREPARE=1 \
DATA_ROOT=/home/yuey21/NeuroState3D_data \
RUN_NAME=h200_closedloop_v2_t1c_seed46 \
TARGET_MODALITY=t1c SEED=46 SPLIT_SEED=4601 \
CLOSED_LOOP_TOKEN_DRIFT=1 TRAIN_STAGE2_HARD=0 \
BATCH_SIZE=8 NUM_WORKERS=4 CPU_THREADS=4 \
RUN_FULL_VOLUME_EVAL=1 RUN_FUSION_EVAL=0 \
bash scripts/run_h200_best_t1c_pipeline.sh
```

Batch size 8 is a conservative starting point, not a measured maximum. The
architecture retains the H200 preset's 96 hidden channels, 10 transport steps
and 128-pixel slices. New loss normalization removes the previous implicit
batch-size scaling of auxiliary objectives, but weights still need empirical
validation. No image-quality improvement is claimed from smoke tests.

Outputs:

- `outputs/h200_closedloop_v2_t1c_seed46_transport/`
- `reports/h200_pipeline/h200_closedloop_v2_t1c_seed46_transport.json`
- Logs under `$DATA_ROOT/logs/h200_pipeline/h200_closedloop_v2_t1c_seed46/`.

Rerun the same command to resume an unfinished run automatically. A completed
run is skipped. Use a new run name when changing objectives or model structure.
Add `PROFILE_STEPS=3` to time the first three executed steps of each epoch;
profiling synchronizes the device and should be disabled for production timing.

## Frozen-base adapter training

For an adapter-only experiment, add the following environment settings to the
command above and use a distinct run name:

```bash
FREEZE_TRANSPORT_BASE=1 \
TRANSPORT_RESUME_CHECKPOINT=/absolute/path/to/compatible_transport_checkpoint.pt
```

The base architecture/conditioning settings must match the checkpoint. This is
a warm start when adding adapters or changing loss versions, not an exact
continuation of the old optimizer. The optional refinement branch also remains
trainable if enabled; the recommended command keeps stage 2 off.

To exactly continue a pre-closed-loop legacy experiment, keep its original
training configuration, disable the adapter, and explicitly set
`SEGMENTATION_LOSS_VERSION=1 EVALUATION_VERSION=1`. Absent closed-loop fields
are compatible only when the feature is disabled. Version-2 objectives
intentionally prevent exact restoration of old optimization history.

## Validation

CPU checks use the current environment; pytest is absent, so test functions were
invoked directly with temporary-directory fixtures.

- 22 tests covering existing transport/runtime behavior and new loss scaling,
  gradients, frozen-base training, checkpoint validation, memory-mapped data,
  reference skipping and batch-independent validation.
- Real BraTS smoke training: 4 subjects, 16-pixel slices, 8 hidden channels,
  2 transport steps, 2 batches per epoch, medical drift bank and closed-loop
  feedback/no-harm losses enabled.
- Epoch and within-epoch resume produce bitwise-identical model weights to
  uninterrupted CPU training in that smoke configuration.
- Frozen-base training and one-patient full-volume inference complete.
- The H200 shell entry point completes on CPU with a small configuration,
  including training, full-volume evaluation, visualization and pipeline report.
- Python compilation, Bash syntax and whitespace checks pass.

The full H200 training preset and its peak memory have not been benchmarked.
Validation reports are kept under `outputs/code_review_validation_20260930/`.
