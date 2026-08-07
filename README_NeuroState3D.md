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

## First Go/No-Go Boundary

Before implementing teacher learning or diffusion, compare:

- official BrainMVP features
- BrainMVP + adapter
- BrainMVP + mask-aware mean fusion
- BrainMVP + fixed-slot concat fusion

Only proceed to spatial evidence fusion and state learning when these baselines
are reproducible and logged.
