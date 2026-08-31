#!/usr/bin/env bash
set -Eeuo pipefail

# Strong H20 preset for the current NeuroState-3D missing-T1c experiment.
#
# This is the "best-effect" entry point for the current method line:
#   full BraTS GLI/MEN/PED -> model-ready 128^3 -> drift-transport base
#   -> hard-slice mining -> hard-case stage-2 fine-tuning -> reports/visuals.
#
# It is intentionally stronger than the generic pipeline defaults. It is not an
# infinite hyperparameter search; it runs the current best configured method.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export RUN_NAME="${RUN_NAME:-h20_best_t1c}"
export DATA_ROOT="${DATA_ROOT:-/data/NeuroState3D}"

# Download and environment defaults.
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export INSTALL_MODE="${INSTALL_MODE:-minimal}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-gli men ped}"
export DOWNLOAD_MAX_WORKERS="${DOWNLOAD_MAX_WORKERS:-8}"
export DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-40}"
export DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-120}"
export STREAM_PREPARE_BY_DATASET="${STREAM_PREPARE_BY_DATASET:-1}"
export PREPARE_WORKERS="${PREPARE_WORKERS:-12}"
export MIN_PROCESSED_SUBJECTS="${MIN_PROCESSED_SUBJECTS:-2000}"

# Full-data training defaults.
export DEVICE="${DEVICE:-cuda}"
export TARGET_MODALITY="${TARGET_MODALITY:-t1c}"
export SPATIAL_SIZE="${SPATIAL_SIZE:-128}"
export MAX_SUBJECTS="${MAX_SUBJECTS:-0}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
export VAL_SUBJECTS="${VAL_SUBJECTS:-235}"
export SPLIT_SEED="${SPLIT_SEED:-4601}"
export SEED="${SEED:-46}"

# Stronger-than-local capacity/coverage. Override these if H20 scheduling
# requires a smaller job.
export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
export TRANSPORT_STEPS="${TRANSPORT_STEPS:-8}"
export SLICES_PER_SUBJECT="${SLICES_PER_SUBJECT:-48}"
export SLICE_CROP_SIZE="${SLICE_CROP_SIZE:-0}"
export SLICE_CROP_JITTER="${SLICE_CROP_JITTER:-6}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export LOG_EVERY="${LOG_EVERY:-100}"

# Base transport stage.
export EPOCHS="${EPOCHS:-16}"
export LR="${LR:-6e-5}"
export TRANSPORT_VELOCITY_WEIGHT="${TRANSPORT_VELOCITY_WEIGHT:-0.55}"
export TRANSPORT_PATH_WEIGHT="${TRANSPORT_PATH_WEIGHT:-0.25}"
export TRANSPORT_MONOTONIC_WEIGHT="${TRANSPORT_MONOTONIC_WEIGHT:-0.08}"

# Hard-case stage-2.
export TRAIN_STAGE2_HARD="${TRAIN_STAGE2_HARD:-1}"
export HARD_SLICE_PROB="${HARD_SLICE_PROB:-0.82}"
export HARD_SLICE_TOP_K="${HARD_SLICE_TOP_K:-4}"
export MINE_CANDIDATE_SLICES_PER_SUBJECT="${MINE_CANDIDATE_SLICES_PER_SUBJECT:-8}"
export MINE_TOP_SLICES_PER_SUBJECT="${MINE_TOP_SLICES_PER_SUBJECT:-4}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
export STAGE2_LR="${STAGE2_LR:-4e-5}"
export STAGE2_SLICE_CROP_SIZE="${STAGE2_SLICE_CROP_SIZE:-96}"
export STAGE2_GATED_REFINEMENT="${STAGE2_GATED_REFINEMENT:-1}"
export STAGE2_FREEZE_TRANSPORT_BASE="${STAGE2_FREEZE_TRANSPORT_BASE:-1}"
export REFINEMENT_RESIDUAL_SCALE="${REFINEMENT_RESIDUAL_SCALE:-0.25}"
export GATE_BIAS_INIT="${GATE_BIAS_INIT:--3.0}"
export GATE_SUPERVISION_WEIGHT="${GATE_SUPERVISION_WEIGHT:-0.60}"
export GATE_SPARSITY_WEIGHT="${GATE_SPARSITY_WEIGHT:-0.08}"
export BACKGROUND_PRESERVE_WEIGHT="${BACKGROUND_PRESERVE_WEIGHT:-0.60}"
export CORE_OVERFILL_WEIGHT="${CORE_OVERFILL_WEIGHT:-0.25}"
export GATE_TARGET_DILATION="${GATE_TARGET_DILATION:-2}"

# Report all important visual regimes.
export VIS_NUM_CASES="${VIS_NUM_CASES:-12}"
export VIS_CASE_SELECTIONS="${VIS_CASE_SELECTIONS:-representative best worst}"
export VIS_SELECTION_POOL="${VIS_SELECTION_POOL:-600}"
export VIS_SLICE_CROP_SIZE="${VIS_SLICE_CROP_SIZE:-0}"

echo "[BEST] NeuroState-3D H20 best-effect preset"
echo "[BEST] RUN_NAME=${RUN_NAME}"
echo "[BEST] DATA_ROOT=${DATA_ROOT}"
echo "[BEST] Full training: MAX_SUBJECTS=${MAX_SUBJECTS}, EPOCHS=${EPOCHS}, STAGE2_EPOCHS=${STAGE2_EPOCHS}"
echo "[BEST] Capacity: HIDDEN_CHANNELS=${HIDDEN_CHANNELS}, TRANSPORT_STEPS=${TRANSPORT_STEPS}, SLICES_PER_SUBJECT=${SLICES_PER_SUBJECT}"
echo "[BEST] Stage-2 gate: STAGE2_GATED_REFINEMENT=${STAGE2_GATED_REFINEMENT}, FREEZE_BASE=${STAGE2_FREEZE_TRANSPORT_BASE}"

bash scripts/run_h20_brats_t1c_pipeline.sh

echo "[BEST] Done."
echo "[BEST] Pipeline report: reports/h20_pipeline/${RUN_NAME}_pipeline_summary.json"
echo "[BEST] Stage-2 checkpoint: outputs/${RUN_NAME}_transport_stage2_hard/slice_virtual_modality_generator_best.pt"
echo "[BEST] Visuals: reports/visuals/${RUN_NAME}/"
