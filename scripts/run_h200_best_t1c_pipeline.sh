#!/usr/bin/env bash
set -Eeuo pipefail

# H200/H200 NVL end-to-end preset for NeuroState-3D missing-T1c generation.
#
# It runs the whole current best pipeline:
#   environment setup -> BraTS2023 HF download -> model-ready preprocessing
#   -> drift-transport base training -> hard-slice mining
#   -> ROI detail/no-harm stage-2 fine-tune -> visual cases -> JSON report.
#
# Recommended server usage:
#   cd /mnt/neuroState3D
#   bash scripts/run_h200_best_t1c_pipeline.sh
#
# Useful overrides:
#   DATA_ROOT=/mnt/NeuroState3D_data bash scripts/run_h200_best_t1c_pipeline.sh
#   FORCE_FINETUNE=1 bash scripts/run_h200_best_t1c_pipeline.sh
#   BATCH_SIZE=48 HIDDEN_CHANNELS=96 bash scripts/run_h200_best_t1c_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export RUN_NAME="${RUN_NAME:-h200_medical_drift_t1c}"

# Keep large assets outside the git worktree by default.
export DATA_ROOT="${DATA_ROOT:-/mnt/NeuroState3D_data}"
export RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw/BraTS2023_HF}"
export MODEL_READY_ROOT="${MODEL_READY_ROOT:-${DATA_ROOT}/model_ready/BraTS2023_HF_128}"
export MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests/BraTS2023_HF}"
export MANIFEST="${MANIFEST:-${MANIFEST_DIR}/brats_model_ready_processed.csv}"
export CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/cache}"
export LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/h200_pipeline/${RUN_NAME}}"

# Keep model outputs/reports with the repo so they are easy to collect.
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
export REPORT_DIR="${REPORT_DIR:-${ROOT}/reports/h200_pipeline}"
export VIS_ROOT="${VIS_ROOT:-${ROOT}/reports/visuals}"

# Stable clean environment. CUDA 12.4 wheels run on the shown CUDA 12.8 driver.
export ENV_NAME="${ENV_NAME:-neurostate3d}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export INSTALL_MODE="${INSTALL_MODE:-minimal}"
export FORCE_INSTALL="${FORCE_INSTALL:-0}"
export SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# Network/data defaults. HF mirror is enabled because mainland routes are often unstable.
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-gli men ped}"
export DOWNLOAD_MAX_WORKERS="${DOWNLOAD_MAX_WORKERS:-10}"
export DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-50}"
export DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-120}"
export STREAM_PREPARE_BY_DATASET="${STREAM_PREPARE_BY_DATASET:-1}"
export PREPARE_WORKERS="${PREPARE_WORKERS:-16}"
export TARGET_SHAPE="${TARGET_SHAPE:-128 128 128}"
export CROP_MARGIN="${CROP_MARGIN:-8}"
export MIN_PROCESSED_SUBJECTS="${MIN_PROCESSED_SUBJECTS:-2000}"
export ALLOW_PREPARE_FAILURES="${ALLOW_PREPARE_FAILURES:-0}"

# H200 NVL 143 GB preset. If the GPU is shared/busy, lower BATCH_SIZE first.
export DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TARGET_MODALITY="${TARGET_MODALITY:-t1c}"
export SPATIAL_SIZE="${SPATIAL_SIZE:-128}"
export MAX_SUBJECTS="${MAX_SUBJECTS:-0}"
export VAL_SUBJECTS="${VAL_SUBJECTS:-235}"
export SPLIT_SEED="${SPLIT_SEED:-4601}"
export SEED="${SEED:-46}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-96}"
export TRANSPORT_STEPS="${TRANSPORT_STEPS:-10}"
export SLICES_PER_SUBJECT="${SLICES_PER_SUBJECT:-64}"
export SLICE_CONTEXT_RADIUS="${SLICE_CONTEXT_RADIUS:-1}"
export SLICE_CROP_SIZE="${SLICE_CROP_SIZE:-0}"
export VAL_SLICE_CROP_SIZE="${VAL_SLICE_CROP_SIZE:--1}"
export VAL_SLICE_CROP_MODE="${VAL_SLICE_CROP_MODE:-}"
export LOG_EVERY="${LOG_EVERY:-100}"
export TARGET_AWARE_MEDICAL_DEFAULTS="${TARGET_AWARE_MEDICAL_DEFAULTS:-1}"
export CLASS_CONDITIONED="${CLASS_CONDITIONED:-1}"
export MEDICAL_ROLE_CONDITIONING="${MEDICAL_ROLE_CONDITIONING:-1}"
export LEARNED_INITIAL_STATE="${LEARNED_INITIAL_STATE:-1}"
export MEDICAL_PROMPT_CONDITIONING="${MEDICAL_PROMPT_CONDITIONING:-1}"
export ROLE_HIDDEN_CHANNELS="${ROLE_HIDDEN_CHANNELS:-48}"
export INITIAL_RESIDUAL_SCALE="${INITIAL_RESIDUAL_SCALE:-0.30}"
export MEDICAL_PROMPT_CHANNELS="${MEDICAL_PROMPT_CHANNELS:-6}"
export MEDICAL_PROMPT_AUX_WEIGHT="${MEDICAL_PROMPT_AUX_WEIGHT:-0.03}"
export SLICE_SAMPLING_MODE="${SLICE_SAMPLING_MODE:-medical_mixed}"
export MIXED_BACKGROUND_PROB="${MIXED_BACKGROUND_PROB:-0.06}"
export MIXED_ET_PROB="${MIXED_ET_PROB:-0.34}"
export MIXED_TC_PROB="${MIXED_TC_PROB:-0.25}"
export MIXED_WT_PROB="${MIXED_WT_PROB:-0.25}"

# Base drift-transport stage.
export EPOCHS="${EPOCHS:-20}"
export LR="${LR:-5e-5}"
export TRANSPORT_VELOCITY_WEIGHT="${TRANSPORT_VELOCITY_WEIGHT:-0.55}"
export TRANSPORT_PATH_WEIGHT="${TRANSPORT_PATH_WEIGHT:-0.25}"
export TRANSPORT_MONOTONIC_WEIGHT="${TRANSPORT_MONOTONIC_WEIGHT:-0.08}"

# The new main method is one-stage medical drifting. Stage-2 remains available
# for ablations, but it is no longer the default because earlier H200 runs
# showed hard-slice fine-tuning overfits after the first epoch.
export TRAIN_STAGE2_HARD="${TRAIN_STAGE2_HARD:-0}"
export HARD_SLICE_PROB="${HARD_SLICE_PROB:-0.86}"
export HARD_SLICE_TOP_K="${HARD_SLICE_TOP_K:-5}"
export MINE_CANDIDATE_SLICES_PER_SUBJECT="${MINE_CANDIDATE_SLICES_PER_SUBJECT:-10}"
export MINE_TOP_SLICES_PER_SUBJECT="${MINE_TOP_SLICES_PER_SUBJECT:-5}"
export MINE_MAX_SUBJECTS="${MINE_MAX_SUBJECTS:-0}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-18}"
export STAGE2_LR="${STAGE2_LR:-4e-5}"
export STAGE2_SLICE_CROP_SIZE="${STAGE2_SLICE_CROP_SIZE:-96}"
export STAGE2_VAL_SLICE_CROP_SIZE="${STAGE2_VAL_SLICE_CROP_SIZE:-0}"
export STAGE2_VAL_SLICE_CROP_MODE="${STAGE2_VAL_SLICE_CROP_MODE:-none}"
export STAGE2_BEST_METRIC="${STAGE2_BEST_METRIC:-lesion_noharm_composite}"
export STAGE2_GATED_REFINEMENT="${STAGE2_GATED_REFINEMENT:-1}"
export STAGE2_FREEZE_TRANSPORT_BASE="${STAGE2_FREEZE_TRANSPORT_BASE:-1}"
export STAGE2_REFINEMENT_ACCEPTANCE_GATE="${STAGE2_REFINEMENT_ACCEPTANCE_GATE:-1}"
export STAGE2_REFINEMENT_DETAIL_FEATURES="${STAGE2_REFINEMENT_DETAIL_FEATURES:-1}"
export STAGE2_REFINEMENT_CHANNELS_MULTIPLIER="${STAGE2_REFINEMENT_CHANNELS_MULTIPLIER:-3}"
export STAGE2_REFINEMENT_BLOCKS="${STAGE2_REFINEMENT_BLOCKS:-5}"
export REFINEMENT_RESIDUAL_SCALE="${REFINEMENT_RESIDUAL_SCALE:-0.28}"
export GATE_BIAS_INIT="${GATE_BIAS_INIT:--3.0}"
export ACCEPT_BIAS_INIT="${ACCEPT_BIAS_INIT:--2.0}"
export GATE_SUPERVISION_WEIGHT="${GATE_SUPERVISION_WEIGHT:-0.45}"
export RESIDUAL_NEED_GATE_WEIGHT="${RESIDUAL_NEED_GATE_WEIGHT:-0.55}"
export RESIDUAL_NEED_GATE_THRESHOLD="${RESIDUAL_NEED_GATE_THRESHOLD:-0.035}"
export GATE_SPARSITY_WEIGHT="${GATE_SPARSITY_WEIGHT:-0.08}"
export BACKGROUND_PRESERVE_WEIGHT="${BACKGROUND_PRESERVE_WEIGHT:-0.70}"
export CORE_OVERFILL_WEIGHT="${CORE_OVERFILL_WEIGHT:-0.25}"
export MULTISCALE_SSIM_WEIGHT="${MULTISCALE_SSIM_WEIGHT:-0.04}"
export LESION_MULTISCALE_SSIM_WEIGHT="${LESION_MULTISCALE_SSIM_WEIGHT:-0.12}"
export LAPLACIAN_PYRAMID_WEIGHT="${LAPLACIAN_PYRAMID_WEIGHT:-0.05}"
export LESION_LAPLACIAN_PYRAMID_WEIGHT="${LESION_LAPLACIAN_PYRAMID_WEIGHT:-0.16}"
export FIDELITY_PYRAMID_LEVELS="${FIDELITY_PYRAMID_LEVELS:-3}"
export MICRO_WINDOW_FEATURE_WEIGHT="${MICRO_WINDOW_FEATURE_WEIGHT:-0.14}"
export MICRO_WINDOW_DRIFT_WEIGHT="${MICRO_WINDOW_DRIFT_WEIGHT:-0.006}"
export MICRO_WINDOW_DRIFT_RADII="${MICRO_WINDOW_DRIFT_RADII:-0.006 0.015 0.04}"
export MICRO_WINDOW_DRIFT_MAX_TOKENS="${MICRO_WINDOW_DRIFT_MAX_TOKENS:-384}"
export MICRO_WINDOW_SIZES="${MICRO_WINDOW_SIZES:-3 5 7}"
export MICRO_WINDOW_STRIDE="${MICRO_WINDOW_STRIDE:-2}"
export ACCEPTANCE_SUPERVISION_WEIGHT="${ACCEPTANCE_SUPERVISION_WEIGHT:-0.25}"
export ACCEPTANCE_ERROR_THRESHOLD="${ACCEPTANCE_ERROR_THRESHOLD:-0.04}"
export REFINEMENT_RESIDUAL_TARGET_WEIGHT="${REFINEMENT_RESIDUAL_TARGET_WEIGHT:-0.45}"
export REFINEMENT_DIRECTION_WEIGHT="${REFINEMENT_DIRECTION_WEIGHT:-0.16}"
export REFINEMENT_RESIDUAL_BUDGET_WEIGHT="${REFINEMENT_RESIDUAL_BUDGET_WEIGHT:-0.25}"
export REFINEMENT_LESION_BUDGET_WEIGHT="${REFINEMENT_LESION_BUDGET_WEIGHT:-0.30}"
export REFINEMENT_NO_HARM_WEIGHT="${REFINEMENT_NO_HARM_WEIGHT:-0.25}"
export REFINEMENT_LESION_NO_HARM_WEIGHT="${REFINEMENT_LESION_NO_HARM_WEIGHT:-0.45}"
export REFINEMENT_BACKGROUND_NO_HARM_WEIGHT="${REFINEMENT_BACKGROUND_NO_HARM_WEIGHT:-0.50}"
export REFINEMENT_NO_HARM_MARGIN="${REFINEMENT_NO_HARM_MARGIN:-0.0}"
export GATE_TARGET_DILATION="${GATE_TARGET_DILATION:-2}"

# Collect enough examples for visual inspection.
export VIS_NUM_CASES="${VIS_NUM_CASES:-16}"
export VIS_CASE_SELECTIONS="${VIS_CASE_SELECTIONS:-representative best worst}"
export VIS_SELECTION_POOL="${VIS_SELECTION_POOL:-800}"
export VIS_SLICE_CROP_SIZE="${VIS_SLICE_CROP_SIZE:-0}"
export PIPELINE_REPORT="${PIPELINE_REPORT:-${REPORT_DIR}/${RUN_NAME}_pipeline_summary.json}"

mkdir -p "${DATA_ROOT}" "${LOG_DIR}" "${REPORT_DIR}" "${OUTPUT_ROOT}" "${VIS_ROOT}"

echo "[H200] NeuroState-3D full missing-T1c generation pipeline"
echo "[H200] ROOT=${ROOT}"
echo "[H200] RUN_NAME=${RUN_NAME}"
echo "[H200] DATA_ROOT=${DATA_ROOT}"
echo "[H200] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[H200] REPORT=${PIPELINE_REPORT}"
echo "[H200] Expected GPU: NVIDIA H200/H200 NVL class, CUDA driver shown by nvidia-smi may be 12.8."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  python - <<'PY'
import subprocess
import sys

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
except Exception as exc:
    print(f"[H200][WARN] Could not query GPU details: {exc}")
    raise SystemExit(0)

if not out:
    print("[H200][WARN] nvidia-smi returned no GPU rows.")
    raise SystemExit(0)

name, total, used, util = [part.strip() for part in out[0].split(",")]
total_mb = int(float(total))
used_mb = int(float(used))
util_pct = int(float(util))
print(f"[H200] Selected GPU: {name}, memory={used_mb}/{total_mb} MiB, util={util_pct}%")
if "H200" not in name.upper():
    print("[H200][WARN] This preset was tuned for H200-class memory; continuing anyway.")
if total_mb < 120000:
    print("[H200][WARN] GPU memory is below 120 GB. Consider BATCH_SIZE=16 or 24.")
if used_mb > 20000 or util_pct > 30:
    print("[H200][WARN] GPU looks busy. If this is not your job, wait or lower BATCH_SIZE.")
PY
else
  echo "[H200][WARN] nvidia-smi not found. The downstream pipeline will fail if DEVICE=cuda."
fi

echo "[H200] Training preset:"
echo "[H200] BATCH_SIZE=${BATCH_SIZE}, HIDDEN_CHANNELS=${HIDDEN_CHANNELS}, TRANSPORT_STEPS=${TRANSPORT_STEPS}"
echo "[H200] EPOCHS=${EPOCHS}, STAGE2_EPOCHS=${STAGE2_EPOCHS}, SLICES_PER_SUBJECT=${SLICES_PER_SUBJECT}"
echo "[H200] SLICE_CONTEXT_RADIUS=${SLICE_CONTEXT_RADIUS}, STAGE2_BEST_METRIC=${STAGE2_BEST_METRIC}"
echo "[H200] Medical drift: target_aware=${TARGET_AWARE_MEDICAL_DEFAULTS}, role=${MEDICAL_ROLE_CONDITIONING}, learned_init=${LEARNED_INITIAL_STATE}, prompt=${MEDICAL_PROMPT_CONDITIONING}, prompt_channels=${MEDICAL_PROMPT_CHANNELS}, sampling=${SLICE_SAMPLING_MODE}"
echo "[H200] HF mirror=${USE_HF_MIRROR}, pip index=${PIP_INDEX_URL}"

bash scripts/run_h20_brats_t1c_pipeline.sh

echo "[H200] DONE"
echo "[H200] Main report: ${PIPELINE_REPORT}"
echo "[H200] Best stage-2 checkpoint: ${OUTPUT_ROOT}/${RUN_NAME}_transport_stage2_noharm/slice_virtual_modality_generator_best.pt"
echo "[H200] Visual cases: ${VIS_ROOT}/${RUN_NAME}/"
echo "[H200] Logs: ${LOG_DIR}"
