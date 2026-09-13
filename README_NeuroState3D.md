# NeuroState-3D First-Round Extension

NeuroState-3D extends BrainMVP as a subject-specific 3D latent state research
framework while preserving official BrainMVP behavior. BrainMVP remains the
pretrained visual prior; new modules live outside the official UniFormer and
U-Net source files so official baselines can still be reproduced.

This first round intentionally implements only:

- `BrainMVPEncoder` wrapper for official multi-scale UniFormer features.
- Lightweight modality adapters: `identity` and `residual_conv`.
- Modality masks for incomplete subject-level observations.
- Mask-aware `mean` fusion baseline.
- Fixed-slot `concat` + Conv3D fusion baseline with mask conditioning.
- `NeuroState3D` forward returning intermediate research artifacts.

It does not implement diffusion, teacher learning, uncertainty estimation,
BraTS clinical fine-tuning, or complex spatial evidence fusion.

## Shape Convention

NeuroState-3D external modules use:

```text
[B, C, D, H, W]
```

Multi-modal fusion uses fixed modality slots:

```text
[B, M, C, D, H, W]
```

The official BrainMVP UniFormer path historically expects MONAI-style
`[B, C, H, W, D]` at its boundary and permutes internally. `BrainMVPEncoder`
isolates that convention and returns stage features in `[B, C, D, H, W]`.

## Implemented Forward Contract

```python
output = model(
    modalities={
        "t1": x_t1,
        "t2": x_t2,
        "fa": x_fa,
        "alff": x_alff,
    },
    modality_mask=mask,
)
```

Returned keys:

- `modality_features`: all multi-scale features per observed modality.
- `selected_features`: the configured feature stage per observed modality.
- `fused_feature`: mask-aware fused 3D representation.
- `brain_state`: currently identical to `fused_feature` in round one.
- `fusion_weights`: mean-fusion weights, or `None` for concat fusion.
- `modality_mask`: observed modality mask in configured order.
- `modalities`: configured modality order.

The current `brain_state` is a subject-specific 3D latent representation for
experimentation. It is not a biological ground truth brain state.

## Smoke Tests

Run the NeuroState-3D module smoke test without requiring MONAI:

```bash
python scripts/test_neurostate_forward.py --fusion mean --encoder tiny
python scripts/test_neurostate_forward.py --fusion concat --encoder tiny
```

Run the local engineering smoke suite with the official pretrained BrainMVP
UniFormer and synthetic inputs only:

```bash
python scripts/smoke_test_local.py \
  --checkpoint pretrained/BrainMVP_uniformer.pt \
  --device cuda \
  --tiny-size 16 \
  --real-size 96
```

This writes `outputs/local_smoke_report.json`, which is intentionally ignored
by git. The report records checkpoint coverage, synthetic missing-modality
cases, mean/concat fusion behavior, NaN/Inf checks, gradient flow, one optimizer
step, and real-size 96^3 forward status.

Run the same forward contract with the official pretrained BrainMVP UniFormer:

```bash
python scripts/test_neurostate_forward.py \
  --fusion mean \
  --encoder brainmvp \
  --checkpoint pretrained/BrainMVP_uniformer.pt \
  --size 16 \
  --batch-size 1

python scripts/test_neurostate_forward.py \
  --fusion concat \
  --encoder brainmvp \
  --checkpoint pretrained/BrainMVP_uniformer.pt \
  --size 16 \
  --batch-size 1
```

## Official BrainMVP Backbone Validation

The official BrainMVP repository recommends:

```bash
conda create -n brainmvp python=3.9
conda activate brainmvp
pip install -r requirements.txt
```

For a clean validation-only environment:

```bash
conda create -n brainmvp-py39 python=3.9 -y
conda activate brainmvp-py39
pip install torch==2.6.0 torchvision==0.21.0 monai==1.4.0 timm==1.0.15 numpy==1.26.4 PyYAML==6.0.1
```

If CUDA wheels are required, install PyTorch from the matching PyTorch CUDA
index before running GPU memory validation, for example:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Inspect checkpoint structure, robust key mapping, parameter coverage, and
representative loaded-vs-random tensors:

```bash
python scripts/inspect_brainmvp.py \
  --checkpoint pretrained/BrainMVP_uniformer.pt \
  --device cpu \
  --skip-forward
```

Run real-size `[B,1,96,96,96]` forward, GPU memory logging when CUDA is
available, and wrapper equivalence against the official `SSLEncoder`:

```bash
python scripts/inspect_brainmvp.py \
  --checkpoint pretrained/BrainMVP_uniformer.pt \
  --device cuda \
  --size 96 \
  --batch-size 1
```

Checkpoint loading is considered valid only when encoder parameter coverage is
at least 95%, no core encoder keys are missing or shape-mismatched, selected
loaded tensors differ from random initialization, selected loaded tensors match
the checkpoint tensors, the real 96^3 forward succeeds, and wrapper outputs are
allclose with the official `SSLEncoder`.

## Local Engineering Defaults

Current local smoke modality order is centrally defined as:

```text
t1, t2, fa, md, alff
```

Default adapters are `identity` for `t1`/`t2` and `residual_conv` for
`fa`/`md`/`alff`. The current config uses `stage4` as the selected fusion
feature for this first engineering smoke. This is not a final research
conclusion; future controlled experiments should compare `stage2`, `stage3`,
and `stage4`.

## HCP Real-Data Smoke Preparation

Real HCP images must stay outside this repository. The local target layout is:

```text
D:/NeuroStateData/HCP_YA_2025/
  raw/
  manifests/
  processed/
  cache/
```

Use `manifests/hcp_local_smoke.template.json` as the repo-side template. After
authorized download, place the real local manifest at
`D:/NeuroStateData/HCP_YA_2025/manifests/hcp_local_smoke.json`. It should store
only `data_root` and relative T1/T2 paths, such as `raw/.../T1w.nii.gz`.
HCP-YA 2025 data access requires the user to register/log in to BALSA, accept
the HCP-YA 2025 Data Use Terms, choose 3 subjects, and download only the
official structural T1w/T2w package/files.

Inspect a populated subject manifest:

```bash
python scripts/inspect_hcp_subject.py \
  --manifest D:/NeuroStateData/HCP_YA_2025/manifests/hcp_local_smoke.json \
  --preprocess \
  --roi-size 96 \
  --output outputs/hcp_subject_inspection.json
```

Run real HCP T1/T2 smoke forward after official data have been downloaded:

```bash
python scripts/test_real_hcp.py \
  --config configs/hcp.yaml \
  --device cuda
```

The HCP preprocessing path follows the audited BrainMVP training intent:
load NIfTI, channel-first, RAS orientation, 1 mm spacing, foreground crop,
5-95 percentile intensity scaling to `[0,1]`, foreground crop again, then a
96^3 patch plus padding. Official pretraining used random spatial crop samples;
the real-data smoke uses a deterministic center crop so the engineering check
is reproducible. It never resizes the whole brain directly to 96^3.

## HCP S1200 Full-Cohort Data Strategy

The full-data phase now targets all HCP S1200 subjects that can support the
five model-ready maps:

```text
T1, T2, FA, MD, ALFF
```

The corresponding acquisition dependencies are preserved:

```text
T1 -> T1w
T2 -> T2w
FA + MD -> diffusion MRI + bvals/bvecs
ALFF -> resting-state fMRI BOLD
```

Data must be written under `E:/NeuroState3D_Data`, never to `C:` and never
inside this git repository. The first phase only audits S3 object existence and
sizes; it does not download data:

```bash
aws configure --profile hcp
python scripts/hcp_s1200_audit.py --config configs/hcp_s1200.yaml --profile hcp
```

This writes:

```text
E:/NeuroState3D_Data/manifests/hcp_s1200_all_subjects.csv
E:/NeuroState3D_Data/manifests/hcp_s1200_eligible_subjects.txt
E:/NeuroState3D_Data/manifests/hcp_s1200_eligible_strict.txt
E:/NeuroState3D_Data/manifests/hcp_s1200_eligible_relaxed.txt
E:/NeuroState3D_Data/manifests/hcp_s1200_audit_summary.json
```

`strict` requires T1, T2, DWI, bvals, bvecs, and all four HCP resting runs
(`REST1_LR`, `REST1_RL`, `REST2_LR`, `REST2_RL`). `relaxed` requires T1, T2,
DWI, bvals, bvecs, and at least one usable resting run. The main dataset
defaults to strict unless it is clearly too small.

Build the strict download plan without downloading:

```bash
python scripts/hcp_s1200_download.py --config configs/hcp_s1200.yaml --profile hcp
```

After checking the manifest and E: drive space, execute the resumable download:

```bash
python scripts/hcp_s1200_download.py --config configs/hcp_s1200.yaml --profile hcp --execute
```

The downloader is idempotent, skips completed files by size, resumes `.part`
files with S3 range requests, records per-subject failures, and does not let one
failed subject terminate the whole batch. It downloads only structural T1/T2,
preprocessed DWI with bvals/bvecs, and resting fMRI runs needed for ALFF. Task
fMRI, MEG, 7T, behavioral data, task contrasts, unrelated FreeSurfer outputs,
and unrelated surface files are not part of the plan.

This stage still does not implement Fusion. After download, FA/MD derivation,
ALFF derivation, registration, QC, model-ready conversion, and split generation
must complete before any Fusion benchmark starts.

## First Go/No-Go Boundary

Before implementing teacher learning or diffusion, compare:

- official BrainMVP features
- BrainMVP + adapter
- BrainMVP + mask-aware mean fusion
- BrainMVP + fixed-slot concat fusion

Only proceed to spatial evidence fusion and state learning when these baselines
are reproducible and logged.

## H20 One-Command BraTS T1c Pipeline

The H20 entry point for the current missing-T1c generation line is:

```bash
bash scripts/run_h20_brats_t1c_pipeline.sh
```

By default it runs inside the current project folder, creates or reuses a
`neurostate3d` conda environment, installs missing dependencies only, downloads
BraTS GLI/MEN/PED from Hugging Face, incrementally prepares model-ready arrays,
trains the drift-transport generator, mines hard lesion slices, runs stage-2
no-harm pixel-fidelity fine-tuning, generates visual cases, and writes a JSON
pipeline summary.

Useful H20 overrides:

```bash
USE_HF_MIRROR=1 DATA_ROOT=/data/NeuroState3D RUN_NAME=h20_full_t1c \
  BATCH_SIZE=16 NUM_WORKERS=8 EPOCHS=12 STAGE2_EPOCHS=6 \
  bash scripts/run_h20_brats_t1c_pipeline.sh
```

The script is resumable:

- Existing Python dependencies are reused when import checks pass.
- Existing raw NIfTI data are reused unless `FORCE_DOWNLOAD=1`.
- Existing model-ready subjects are skipped unless `FORCE_PREPARE=1`.
- Existing checkpoints/reports are skipped unless `FORCE_FINETUNE=1`.
- Runtime artifacts stay under ignored folders such as `data/`, `outputs/`,
  and `reports/`; they should not be committed.

For the current best-effect experiment preset, use the stronger wrapper:

```bash
bash scripts/run_h20_best_t1c_experiment.sh
```

This wrapper keeps the same end-to-end pipeline but raises the training budget
and capacity for H20: full subject usage, no max-step cap, 64 hidden channels,
8 transport steps, 2.5D slice context by default, 48 slices per subject, 16 base
epochs, 10 no-harm stage-2 epochs over mined hard slices, lesion-gated no-harm
refinement, background-preservation losses, anti-overfill losses, and
representative/best/worst visual reports. The stage-2 refinement is no longer a
tiny correction head: by default it uses a 2x wider, 4-block convolutional
context refiner over the observed modalities, the stage-1 synthetic T1c, and
uncertainty. It predicts a lesion-region gate, an acceptance gate, and a bounded
residual so the final image can fall back to stage-1 where refinement is not
expected to help. Pixel fidelity is explicitly trained with micro-window
3/5/7-pixel medical feature alignment, micro-window drifting over the same
local feature tokens, multi-scale SSIM, and Laplacian pyramid detail losses,
while the no-harm losses penalize refined pixels that are worse than the frozen
stage-1 transport output. Stage-2 best-checkpoint selection defaults to a lesion
composite score instead of whole-image MAE, so the selected model favors
high-tumor, ET/TC, under-enhancement, structural fidelity, and no-harm behavior.
It is the intended command when the goal is to push the current drifting method
line as hard as the H20 job budget allows.

## Reliable Target-Aware Training

The current H200 entry point is:

```bash
DATA_ROOT=$HOME/NeuroState3D_data \
RUN_NAME=h200_targetaware_t1c_seed46 \
TARGET_MODALITY=t1c TRAIN_STAGE2_HARD=0 \
RUN_FULL_VOLUME_EVAL=1 RUN_FUSION_EVAL=0 \
bash scripts/run_h200_best_t1c_pipeline.sh
```

Training slices and crop locations now change deterministically with the epoch;
the validation set remains fixed. Step checkpoints are versioned, retain the
optimizer, RNG state, global step, within-epoch step, medical drift memory and
best/history state, and the newest checkpoint is resumed automatically. Set
`STEP_CHECKPOINT_EVERY=500` and `CHECKPOINT_KEEP_LAST=3` to control their
frequency and retention.

The one-stage model defaults to deterministic learned initialization. A
stochastic conditional-initial-state experiment can be enabled separately:

```bash
STOCHASTIC_INITIAL_STATE=1 STOCHASTIC_NOISE_SCALE=0.05 POSTERIOR_SAMPLES=8 \
RUN_NAME=h200_targetaware_t1c_stochastic_seed46 \
bash scripts/run_h200_best_t1c_pipeline.sh
```

Patient-level whole-volume validation is enabled by default and is written to
`reports/h200_pipeline/<run-name>_full_volume.json`. Downstream fusion
evaluation is optional because it requires a separately trained fusion model
using the same `SPLIT_SEED`:

```bash
RUN_FUSION_EVAL=1 \
FUSION_CHECKPOINT=/path/to/evidence_fusion_small_last.pt \
BRAINMVP_CHECKPOINT=/path/to/BrainMVP_uniformer.pt \
bash scripts/run_h200_best_t1c_pipeline.sh
```

Use a distinct `RUN_NAME` for every target modality or ablation. Stage-2 hard
slice refinement is retained for ablation (`TRAIN_STAGE2_HARD=1`) but is off by
default; when it is off, neither hard-slice mining nor stale stage-2 checkpoint
selection runs.
