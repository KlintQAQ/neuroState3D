# CURRENT CODE AUDIT

Audit date: 2026-08-21  
Project root: `D:\BrainMVP-neurostate3d`  
Scope: Phase 0 only. This file documents the current NeuroState-3D / BrainMVP / BraTS missing-modality generation codebase before implementing the new target-aware lesion-guided local medical drifting method.

## 1. Repository State

The repository is a real git working tree, but the current research code is in a dirty state with many untracked experiment files. Do not clean, reset, or delete these files without an explicit backup/commit plan.

Tracked files currently modified include:

- `README_NeuroState3D.md`
- `datasets/brats_fusion_dataset.py`
- `models/evidence_fusion.py`
- `requirements.txt`
- `scripts/smoke_evidence_fusion_brats.py`
- `utils/brats_metrics.py`

Important untracked code includes:

- `models/virtual_modality_generator.py`
- `models/slice_virtual_modality_generator.py`
- `models/prompted_slice_virtual_modality_generator.py`
- `models/high_fidelity_virtual_modality_generator.py`
- `scripts/train_virtual_modality_drifting.py`
- `scripts/train_slice_virtual_modality_drifting.py`
- `scripts/train_high_fidelity_virtual_modality_drifting.py`
- `scripts/precompute_virtual_t1c_cache.py`
- `scripts/precompute_slice_virtual_t1c_cache.py`
- `scripts/evaluate_synthetic_modality_fusion.py`
- `scripts/visualize_slice_virtual_modality_generation.py`
- many `reports/*.json`, `reports/visuals/*`, and `outputs/*`

`.gitignore` already excludes `pretrained/`, `outputs/`, `data/`, `raw/`, `cache/`, `tmp/`, checkpoints, and medical image formats. This is correct; do not commit raw data, model-ready BraTS volumes, or pretrained weights.

## 2. Current Data Assets

Current BraTS model-ready manifest:

`data/manifests/BraTS2023_HF/brats_model_ready_processed.csv`

Manifest columns observed:

- `dataset`
- `subject_id`
- `status`
- `original_shape`
- `target_shape`
- `multimodal_path`
- `seg_path`

Current subject counts from the manifest:

- GLI: 1251
- MEN: 1000
- PED: 99
- Total: 2350

The model-ready files point to `data/model_ready/BraTS2023_HF_128/...` and use:

- `multimodal_4ch.npy`
- `seg.npy`

The model-ready target shape recorded in the manifest is `128x128x128`.

## 3. BraTS Modality Protocol

The current canonical BraTS modality order is already defined in `datasets/brats_fusion_dataset.py`:

```python
BRATS_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
BRATS_REGIONS = ("ET", "TC", "WT")
```

Segmentation labels are converted as:

- ET: `seg == 3`
- TC: `(seg == 1) | (seg == 3)`
- WT: `seg > 0`

This is reusable and should become the single source of truth for future target-conditioned generation. Avoid introducing parallel hard-coded modality orders.

Current risk:

- Several scripts still assume `target_modality="t1c"` by default.
- Some downstream scripts contain T1c-specific names such as `virtual_t1c`.
- There is no formal `modality_to_idx` / `idx_to_modality` utility module yet.

## 4. Current Dataset Loaders

### 4.1 3D BraTS Fusion Dataset

File: `datasets/brats_fusion_dataset.py`

`BraTSFusionDataset` loads:

- image tensor: `[4, D, H, W]`
- segmentation: `[D, H, W]`
- multilabel target: `[3, D, H, W]`
- modality mask: `[4]`
- modality state: `[4]`

It supports:

- fixed or random modality availability masks
- `spatial_size` trilinear resizing for image and nearest resizing for segmentation
- lesion/foreground-biased crop via `foreground_crop_prob`
- region-balanced crop candidates with ET/TC/WT/foreground weights
- optional degradation of observed channels

This loader is directly reusable for downstream segmentation/fusion and future 3D patch generation.

Current limitations:

- It is not yet a missing-modality generation dataset returning `observed_images`, `target_image`, `target_idx`, `target_name`, and target-balanced sampling.
- It does not expose a formal subject-level train/val/test split; many scripts use first rows for train and tail rows for val.
- Normalization is assumed to have been done during preprocessing; the loader does not currently implement selectable `legacy` vs `zscore_brain`.

### 4.2 2D / 2.5D Slice Dataset

File: `scripts/train_slice_virtual_modality_drifting.py`

`BraTSSliceDataset` is embedded inside the training script. It loads axial slices from `multimodal_4ch.npy` and `seg.npy`.

It supports:

- fixed `--target-modality`, default `t1c`
- 2D input when `--slice-context-radius 0`
- 2.5D context input when `--slice-context-radius > 0`
- region-balanced 2D crop via `--slice-crop-mode region_balanced`
- target channel masking, including context channels

Current limitations:

- The dataset is script-local, not reusable as a formal dataset module.
- It samples one fixed target for the whole run; it does not support `target=random`.
- It clamps slice values to `[-3, 3]`, which is a normalization/scale assumption that must be documented and preserved for legacy checkpoints.
- It uses axial slices only; no sagittal/coronal or true 3D patch sampling.

## 5. Current Generators

### 5.1 3D VirtualModalityGenerator

File: `models/virtual_modality_generator.py`

This is already a lightweight 3D target-conditioned generator:

```text
images:        [B, M, D, H, W]
modality_mask: [B, M]
target_index:  scalar or [B]
output:
  synthetic:   [B, 1, D, H, W]
  uncertainty: [B, 1, D, H, W]
  confidence:  [B, 1, D, H, W]
```

It uses a target embedding and concatenates:

- masked images
- mask channels
- target embedding broadcast as volume channels

This is useful as an existing 3D baseline and as a target-conditioned reference, but it is small and not yet lesion-guided or target-specific drifting.

### 5.2 2D SliceVirtualModalityGenerator

File: `models/slice_virtual_modality_generator.py`

This is a small 2D U-Net-like axial slice generator. It predicts:

- `synthetic`
- `uncertainty`
- `confidence`

Input shape:

```text
slices:        [B, M, H, W]
modality_mask: [B, M]
```

It is stable and good for fast local iteration, but it has no target embedding. The target is implied by which channel is masked.

### 5.3 PromptedSliceVirtualModalityGenerator

File: `models/prompted_slice_virtual_modality_generator.py`

This is the current strongest experimental T1c path. It adds:

- ET/TC/WT prompt head
- base output head
- refinement residual
- lesion residual
- detail residual
- optional class conditioning for GLI/MEN/PED
- optional hardtanh/no activation output modes
- optional positive lesion residual

Output includes:

- `synthetic`
- `uncertainty`
- `confidence`
- `prompt_logits`
- `prompt_probs`
- `residual`
- `lesion_residual`
- `detail_residual`
- `base_logits`
- `synthetic_logits`

This should be preserved. It is the main local experimental generator path, but it is still T1c-centric and 2D/2.5D rather than unified four-target 3D.

## 6. Current Drifting / Local Feature Code

File: `utils/torch_drift_loss.py`

There is a reusable PyTorch implementation of the drifting loss:

- positive anchors attract generated tokens
- optional negative anchors repel
- multiple radii are supported
- positive/negative anchors are stop-gradient

File: `scripts/train_slice_virtual_modality_drifting.py`

The script contains several local/slice drifting helpers:

- `patch_tokens_2d`
- `slice_drift_loss`
- `conv_medical_features_2d`
- `window_tokens_2d`
- `window_drift_loss`
- `window_feature_alignment_loss`
- `lesion_texture_loss`
- `gradient_difference_loss`
- region moment / tumor edge / enhancement underestimation / top-intensity losses

Important finding:

- The code already contains 2D local/window feature experiments.
- It does not yet implement formal target-specific global medical feature banks.
- It does not yet implement true 3D lesion-guided local window drifting.
- Current local/window code is inside one training script, not modularized under `models/drifting/`.

## 7. Current Training Pipeline

### 7.1 Slice T1c Generation

Primary script:

`scripts/train_slice_virtual_modality_drifting.py`

Typical current T1c command pattern:

```powershell
python scripts/train_slice_virtual_modality_drifting.py `
  --device cuda `
  --target-modality t1c `
  --spatial-size 128 `
  --max-subjects 1024 `
  --val-subjects 128 `
  --slices-per-subject 8 `
  --slice-crop-size 64 `
  --slice-crop-mode region_balanced `
  --model-kind prompted `
  --output-dir outputs/<experiment_name> `
  --report-path reports/<experiment_name>.json
```

Checkpoint saved:

`<output-dir>/slice_virtual_modality_generator_last.pt`

Checkpoint contains:

- `model`
- `config`
- `target_modality`
- `modalities`

Current resume behavior:

- Can migrate older `SliceVirtualModalityGenerator` weights into prompted model by remapping `out.*` to `base_out.*`.
- Can expand first conv weights for 2.5D context channels.
- Uses `strict=False` and records missing/unexpected keys in report.

Current limitations:

- No optimizer/scheduler state in slice generator checkpoints.
- No formal resume of optimizer/epoch.
- No checkpoint metadata for normalization mode beyond CLI config.
- No four-target random sampling or balanced target accounting.
- No CSV evaluator output yet.

### 7.2 Evidence Fusion / Downstream Segmentation

Primary scripts:

- `scripts/train_evidence_fusion_small.py`
- `scripts/evaluate_synthetic_modality_fusion.py`
- `scripts/visualize_evidence_fusion_predictions.py`

The fusion path trains/evaluates an `EvidenceReliableFusion` segmentation model using BrainMVP stage features.

It supports:

- full real input
- remove target modality
- synthetic filled input
- optional virtual fifth slot `virtual_t1c`
- uncertainty/confidence availability weighting
- region-query fusion and feature completion variants

Current limitation:

- This is not yet the final required frozen downstream evaluation protocol. Some scripts train local segmentation/fusion models and then evaluate them.
- The final CVPR protocol must compare Full Real / Missing / Baseline Generated Fill / Ours Generated Fill using the same frozen segmentation model.

## 8. Current Generated Modality Paths

### 8.1 Generated T1c Cache

Files:

- `scripts/precompute_virtual_t1c_cache.py`
- `scripts/precompute_slice_virtual_t1c_cache.py`

The slice cache script produces per-subject:

- `virtual_t1c.npy`
- `virtual_t1c_uncertainty.npy`
- `virtual_t1c_confidence.npy`
- `virtual_t1c_stats.json`

Current limitation:

- It hardcodes `target_index = BRATS_MODALITIES.index("t1c")`.
- It writes `virtual_t1c`, not generic `generated_<target>`.
- It should be preserved as a historical T1c cache path, but not used as the formal default for four-target generation.

### 8.2 Virtual Fifth Modality

Files:

- `models/evidence_fusion.py`
- `scripts/train_evidence_fusion_small.py`
- `scripts/evaluate_synthetic_modality_fusion.py`

There is explicit support for appending `virtual_t1c` as an additional evidence slot.

This is useful as a historical ablation, but the new default pipeline should not treat generated MRI as a fifth modality. The formal method should fill the original missing slot:

```text
[t1n, generated_t1c, t2w, t2f]
```

when T1c is missing, and analogously for other targets.

### 8.3 Original-Slot Filling

`scripts/evaluate_synthetic_modality_fusion.py` already has an original-slot path when `--enable-virtual-t1c-slot` is not set:

```python
synthetic_image = image.clone()
synthetic_image[:, target_index : target_index + 1] = synthetic
synthetic_mask = full_mask.clone()
```

This is important and should be extracted into a tested utility such as `fill_generated_modality()`.

Current limitation:

- The implementation is inline inside an evaluation script.
- The default confidence/availability behavior can still mark generated evidence as degraded/availability-weighted; this needs a formal protocol.

## 9. Current Evaluation and Reports

Current slice generator evaluation reports:

- global MAE/MSE/PSNR
- uncertainty mean/confidence mean
- uncertainty-error correlation
- prompt dice for ET/TC/WT when prompted model is used
- lesion-region MAE for ET/TC/WT

Important current results:

| Report | MAE | PSNR | MAE_ET | MAE_TC | MAE_WT | Prompt ET/TC/WT |
|---|---:|---:|---:|---:|---:|---|
| `prompted_local_decoder_hardtanh_t1c_128_1000diag.json` | 0.130 | 21.26 | 0.18 | 0.20 | 0.25 | 0.32 / 0.45 / 0.76 |
| `prompted_detail_only_t1c_128_300diag.json` | 0.131 | 21.23 | 0.18 | 0.20 | 0.25 | 0.32 / 0.45 / 0.76 |
| `prompted_class_detail_t1c_128_500diag.json` | 0.142 | 20.79 | 0.18 | 0.20 | 0.26 | 0.32 / 0.45 / 0.76 |
| `prompted_25d_class_detail_t1c_128_500diag.json` | 0.144 | 20.57 | 0.18 | 0.21 | 0.28 | 0.32 / 0.45 / 0.76 |

Qualitative outputs exist under:

- `reports/visuals/prompted_local_decoder_hardtanh_t1c_128_1000diag/`
- `reports/visuals/prompted_detail_only_t1c_128_300diag/`
- `reports/visuals/prompted_25d_class_detail_t1c_128_500diag/`
- other historical visual folders

Current missing evaluation pieces:

- formal `results_global.csv`
- formal `results_lesion.csv`
- boundary fidelity metrics
- small/medium/large lesion analysis
- lesion fidelity gap analysis as a standalone script
- four-target reporting by `t1n/t1c/t2w/t2f`
- SSIM/lesion SSIM with clear small-ROI handling
- HD95 downstream metric
- recovery ratio CSV

## 10. BrainMVP / Medical Encoder Status

Official BrainMVP files are present:

- `models/Uniformer.py`
- `models/uniformer_blocks.py`
- `models/brainmvp_encoder.py`
- `pretrained/BrainMVP_uniformer.pt`

Checkpoint file:

- `pretrained/BrainMVP_uniformer.pt`
- size: about 1.18 GB

Existing validation report:

`reports/brainmvp_checkpoint_full_report.json`

Observed checkpoint structure:

- type: `dict`
- top-level keys: `current_epoch`, `state_dict`, `optimizer`, `scheduler`, `val_best`
- tensor count: 352
- checkpoint parameter count: 105,744,016

Existing BrainMVP wrapper mapping report:

- total encoder parameters: 22,040,000
- matched parameters: 22,037,952
- matched parameter ratio: 0.999907
- matched tensor ratio: 0.996960
- missing key: `uniformer.patch_embed1.proj.weight`
- shape mismatch: first patch embedding weight, because the checkpoint first conv is 1-channel while some validation used 4-channel input
- unexpected keys are mainly decoder and `rep_template`, which are outside the encoder wrapper

Interpretation:

- BrainMVP can be reused as a frozen medical encoder for feature extraction.
- For single-channel encoder use, pretrained coverage is effectively validated.
- For multi-channel direct input, first conv mismatch must be handled deliberately; do not silently rely on `strict=False`.
- The future medical feature encoder interface should keep BrainMVP optional/frozen and record parameter coverage.

## 11. Stage Semantics from BrainMVP

`BrainMVPEncoder` wraps official `SSLEncoder`, which returns:

```text
stage0 = x_0
stage1 = x_enc1
stage2 = x_enc2
stage3 = x_enc3
stage4 = x_enc4
```

Official `uniformer_blocks.py` internally permutes input from `[B,C,H,W,D]` to `[B,C,D,H,W]`, then applies:

- `patch_embed1` + `blocks1`
- `patch_embed2` + `blocks2`
- `patch_embed3` + `blocks3`
- `patch_embed4` + `blocks4`
- final norm on stage4

Stage0 is the input after layout permutation, not a learned feature. Stage1-4 are learned features. For future feature drifting/fusion:

- stage1/stage2 are better for local spatial detail
- stage3/stage4 are better for semantic/global medical features
- stage0 should not be treated as a learned medical feature

## 12. Config Status

Existing configs:

- `configs/neurostate.yaml`: early NeuroState/HCP-style five-modality config.
- `configs/hcp.yaml`: local HCP smoke config.
- `configs/hcp_s1200.yaml`: HCP S1200 download/preprocess plan.

There is no formal BraTS missing-modality drifting config yet. Needed future config:

- `configs/cvpr_missing_modality_drift.yaml`

It should include:

- modality order
- target sampling mode
- normalization mode
- patch size
- prompt config
- global/local drift config
- feature bank config
- residual refinement config
- loss weights
- ablation mode
- checkpoint metadata

## 13. Historical Code to Preserve

Do not delete the following. They are useful baselines/ablations:

- `models/virtual_modality_generator.py`
- `models/slice_virtual_modality_generator.py`
- `models/prompted_slice_virtual_modality_generator.py`
- `models/high_fidelity_virtual_modality_generator.py`
- `scripts/train_slice_virtual_modality_drifting.py`
- `scripts/train_virtual_modality_drifting.py`
- `scripts/train_high_fidelity_virtual_modality_drifting.py`
- `scripts/precompute_slice_virtual_t1c_cache.py`
- `scripts/evaluate_synthetic_modality_fusion.py`
- virtual fifth-modality logic in `models/evidence_fusion.py`
- all current `reports/visuals/*` qualitative outputs
- all `reports/*.json` experiment logs

The virtual fifth-modality code should eventually be labeled as legacy/ablation, not removed.

## 14. Main Shape / Naming / Normalization Risks

### Shape Risks

- 2D slice generator uses `[B, M, H, W]`.
- 2.5D slice generator expands channels to `M * context_depth`.
- 3D generator and fusion use `[B, M, D, H, W]`.
- BrainMVP wrapper accepts `[B, C, D, H, W]` but official UniFormer internally expects/permutates `[B, C, H, W, D]`.
- Slice target index with context is not equal to base modality index; code uses `base_target_index * context_depth + context_radius`.

### Naming Risks

- BraTS uses `t1n`, `t1c`, `t2w`, `t2f`.
- Some documentation and prior HCP configs use `t1`, `t2`, `fa`, `md`, `alff`.
- Historical cache/eval code hardcodes `virtual_t1c`.
- New work must not use `T2F`, `t2f`, and `flair` interchangeably without a mapping.

### Normalization Risks

- Model-ready BraTS `.npy` files are presumed already normalized.
- Slice dataset clamps values to `[-3, 3]`.
- Existing generator outputs often use `tanh` or `hardtanh` to `[-1, 1]`.
- Switching to `zscore_brain` without preserving legacy mode would invalidate existing checkpoints.

## 15. What Can Be Reused Directly

Reusable now:

- `BRATS_MODALITIES`, `BRATS_REGIONS`, `brats_region_targets`
- `BraTSFusionDataset` for 3D fusion/downstream
- `VirtualModalityGenerator` as a simple 3D target-conditioned baseline
- `PromptedSliceVirtualModalityGenerator` for fast lesion-prompted T1c experiments
- `utils/torch_drift_loss.py`
- `utils/brats_metrics.py` for dice and prompt/segmentation losses
- BrainMVP wrapper and checkpoint mapping code
- inline original-slot filling logic from `evaluate_synthetic_modality_fusion.py`
- existing visualization scripts for qualitative case review

## 16. What Needs to Be Added Next

For Phase 1 and later, add incrementally:

1. A reusable four-target missing-modality dataset wrapper.
2. A central modality utility module with `MODALITIES`, `modality_to_idx`, `idx_to_modality`, and `fill_generated_modality()`.
3. Subject-level split handling; avoid slice-level leakage.
4. Standalone lesion fidelity gap analysis script for current T1c checkpoint.
5. Formal global/lesion/boundary/small-lesion CSV evaluators.
6. Unified target-conditioned generator support for all four targets.
7. Target-specific medical feature banks.
8. Modular global drift and local 3D window drift code.
9. Pathology prompt module with `none`, `predicted`, and `gt_upper_bound` modes.
10. Residual refinement module with bounded residual scale.
11. Formal original-slot downstream evaluation with frozen segmentation model.
12. Smoke tests for each new module.

## 17. Recommended Immediate Next Step

Before implementing the full method, run the requested highest-priority experiment:

Lesion Fidelity Gap Analysis using the current best T1c generator.

Best current checkpoint candidate:

`outputs/prompted_local_decoder_hardtanh_t1c_128_1000diag/slice_virtual_modality_generator_last.pt`

Suggested analysis:

- compare `MAE_normal`, `MAE_WT`, `MAE_TC`, `MAE_ET`
- compute lesion volume per subject/slice
- check whether lesion error is higher than normal-brain error
- check whether small lesion error is worse than large lesion error
- output CSV + JSON + plots

This directly tests the paper's core premise before adding new modules.

## 18. Phase 0 Verdict

The current codebase is usable and contains substantial prior work, but it is not yet the requested CVPR-ready four-target missing-modality drifting framework.

Current status:

- BraTS four-modality naming exists.
- Real BraTS model-ready data exists.
- BrainMVP pretrained checkpoint exists and is mostly validated for encoder use.
- T1c generation experiments exist.
- Prompt/local/window/texture ideas exist in 2D/2.5D form.
- Original-slot filling exists inline.
- Virtual fifth-modality experiments exist and should be preserved as ablations.

Not yet implemented:

- unified four-target target-conditioned training
- formal target-balanced dataset
- target-specific feature banks
- global medical drift module
- 3D lesion-guided local window drifting module
- formal residual refinement module
- formal lesion fidelity gap analysis script
- full four-target global/lesion/boundary/downstream evaluation CSV pipeline

Therefore, Phase 0 is complete, and the next safe step is Phase 1 or the standalone Lesion Fidelity Gap Analysis, depending on the user's instruction.

## 19. Direction Update: 2D / 2.5D Main Track

Updated user decision: the main project should not prioritize full 3D generation now. The working assumption is:

```text
2D / 2.5D missing-modality generation + lesion-aware fusion is the main CVPR track.
```

This changes the implementation priority:

- Keep the existing 2D slice generator path as the main engineering base.
- Keep 2.5D context via `slice_context_radius` as a lightweight anatomical continuity upgrade.
- Do not implement heavy full-volume 3D generator, 3D diffusion, or 3D local window drifting as the first paper path.
- Use volume-level/subject-level evaluation by stacking generated 2D slices back into a 3D volume, then filling the original missing modality slot.
- Preserve 3D BrainMVP/evidence-fusion code as downstream evaluation and optional ablation infrastructure, not as the core generator.
- Reinterpret "local window drifting" as 2D/2.5D lesion-guided local medical drifting unless a later experiment proves 3D is necessary.

The revised core problem should be:

```text
For a missing BraTS modality, generate a high-fidelity 2D/2.5D target slice from the observed modalities, with special emphasis on lesion boundary, enhancement, and small lesion fidelity, then verify that filling the original missing slot improves downstream multimodal segmentation.
```

Under this plan, the most important next implementation steps are:

1. Formalize four-target 2D/2.5D dataset sampling.
2. Convert the current T1c-specific slice generator into a unified target-conditioned 2D/2.5D generator.
3. Add lesion fidelity gap analysis for the existing T1c checkpoint.
4. Add original-slot filling for generated 2D slices reconstructed into volumes.
5. Compare downstream segmentation/fusion with Full Real, Missing, Baseline Fill, and Ours Fill.

This preserves the current working code and avoids overcommitting to expensive 3D generation before the 2D lesion-fidelity thesis is validated.
