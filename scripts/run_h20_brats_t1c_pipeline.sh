#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end H20 pipeline for NeuroState-3D BraTS T1c missing-modality generation.
#
# What this script does:
#   1. Reuse/create a Python environment.
#   2. Install only missing runtime dependencies by default.
#   3. Download BraTS2023 HF mirrors with resumable Hugging Face snapshots.
#   4. Prepare model-ready arrays and manifests.
#   5. Train the iterative drift-transport missing-modality generator.
#   6. Generate visual cases and a pipeline summary report.
#
# Typical H20 usage:
#   cd /path/to/neuroState3D
#   bash scripts/run_h20_brats_t1c_pipeline.sh
#
# Useful overrides:
#   USE_HF_MIRROR=1 bash scripts/run_h20_brats_t1c_pipeline.sh
#   DATA_ROOT=/data/NeuroState3D MAX_SUBJECTS=0 EPOCHS=12 BATCH_SIZE=16 bash scripts/run_h20_brats_t1c_pipeline.sh

on_error() {
  local line_no="$1"
  echo "[ERROR] Pipeline failed at line ${line_no}." >&2
  echo "[ERROR] Check logs under: ${LOG_DIR:-<not initialized>}" >&2
}
trap 'on_error $LINENO' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-h20_t1c_transport_${STAMP}}"

DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw/BraTS2023_HF}"
MODEL_READY_ROOT="${MODEL_READY_ROOT:-${DATA_ROOT}/model_ready/BraTS2023_HF_128}"
MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/manifests/BraTS2023_HF}"
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/brats_model_ready_processed.csv}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/cache}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs/h20_pipeline/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/reports/h20_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
VIS_ROOT="${VIS_ROOT:-${ROOT}/reports/visuals}"

mkdir -p "${DATA_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}" "${REPORT_DIR}" "${OUTPUT_ROOT}" "${VIS_ROOT}"

ENV_NAME="${ENV_NAME:-neurostate3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv_h20}"
INSTALL_MODE="${INSTALL_MODE:-minimal}"   # minimal or full
FORCE_INSTALL="${FORCE_INSTALL:-0}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-0}"

USE_HF_MIRROR="${USE_HF_MIRROR:-0}"
if [[ "${USE_HF_MIRROR}" == "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-gli men ped}"
DOWNLOAD_MAX_WORKERS="${DOWNLOAD_MAX_WORKERS:-8}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-30}"
DOWNLOAD_RETRY_SLEEP="${DOWNLOAD_RETRY_SLEEP:-120}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
PREPARE_WORKERS="${PREPARE_WORKERS:-8}"
TARGET_SHAPE="${TARGET_SHAPE:-128 128 128}"
CROP_MARGIN="${CROP_MARGIN:-8}"
MIN_PROCESSED_SUBJECTS="${MIN_PROCESSED_SUBJECTS:-1}"
ALLOW_PREPARE_FAILURES="${ALLOW_PREPARE_FAILURES:-0}"

DEVICE="${DEVICE:-cuda}"
TARGET_MODALITY="${TARGET_MODALITY:-t1c}"
SPATIAL_SIZE="${SPATIAL_SIZE:-128}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"          # 0 means all subjects in the training script.
VAL_SUBJECTS="${VAL_SUBJECTS:-128}"
SLICES_PER_SUBJECT="${SLICES_PER_SUBJECT:-32}"
SLICE_CROP_SIZE="${SLICE_CROP_SIZE:-64}"
SLICE_CROP_JITTER="${SLICE_CROP_JITTER:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-100}"
SEED="${SEED:-46}"
SPLIT_SEED="${SPLIT_SEED:-4601}"

BASE_EPOCHS="${BASE_EPOCHS:-8}"
BASE_MAX_TRAIN_STEPS="${BASE_MAX_TRAIN_STEPS:-0}"
BASE_LR="${BASE_LR:-2e-4}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${OUTPUT_ROOT}/h20_prompted_base_t1c_128}"
BASE_REPORT="${BASE_REPORT:-${REPORT_DIR}/${RUN_NAME}_base.json}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${BASE_OUTPUT_DIR}/slice_virtual_modality_generator_last.pt}"
BASE_RESUME_CHECKPOINT="${BASE_RESUME_CHECKPOINT:-}"
FORCE_BASE_TRAIN="${FORCE_BASE_TRAIN:-0}"

EPOCHS="${EPOCHS:-12}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
LR="${LR:-6e-5}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_transport}"
FINAL_REPORT="${FINAL_REPORT:-${REPORT_DIR}/${RUN_NAME}_transport.json}"
FINAL_CHECKPOINT="${FINAL_OUTPUT_DIR}/slice_virtual_modality_generator_last.pt"
FINAL_BEST_CHECKPOINT="${FINAL_OUTPUT_DIR}/slice_virtual_modality_generator_best.pt"
FORCE_FINETUNE="${FORCE_FINETUNE:-0}"
TRAIN_PROMPT_WITH_DETAIL="${TRAIN_PROMPT_WITH_DETAIL:-0}"
TRAIN_PROMPTED_BASELINE="${TRAIN_PROMPTED_BASELINE:-0}"
TRANSPORT_STEPS="${TRANSPORT_STEPS:-6}"
TRANSPORT_STEP_SCALE="${TRANSPORT_STEP_SCALE:-1.0}"
TRANSPORT_VELOCITY_SCALE="${TRANSPORT_VELOCITY_SCALE:-1.0}"
TRANSPORT_INIT_BLUR_KERNEL="${TRANSPORT_INIT_BLUR_KERNEL:-5}"
TRANSPORT_VELOCITY_WEIGHT="${TRANSPORT_VELOCITY_WEIGHT:-0.55}"
TRANSPORT_PATH_WEIGHT="${TRANSPORT_PATH_WEIGHT:-0.25}"
TRANSPORT_MONOTONIC_WEIGHT="${TRANSPORT_MONOTONIC_WEIGHT:-0.08}"

TRAIN_STAGE2_HARD="${TRAIN_STAGE2_HARD:-1}"
HARD_SLICE_JSON="${HARD_SLICE_JSON:-${REPORT_DIR}/${RUN_NAME}_hard_slices.json}"
HARD_SLICE_PROB="${HARD_SLICE_PROB:-0.75}"
HARD_SLICE_TOP_K="${HARD_SLICE_TOP_K:-3}"
MINE_CANDIDATE_SLICES_PER_SUBJECT="${MINE_CANDIDATE_SLICES_PER_SUBJECT:-6}"
MINE_TOP_SLICES_PER_SUBJECT="${MINE_TOP_SLICES_PER_SUBJECT:-3}"
MINE_MAX_SUBJECTS="${MINE_MAX_SUBJECTS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-6}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
STAGE2_SLICE_CROP_SIZE="${STAGE2_SLICE_CROP_SIZE:-96}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_transport_stage2_hard}"
STAGE2_REPORT="${STAGE2_REPORT:-${REPORT_DIR}/${RUN_NAME}_transport_stage2_hard.json}"
STAGE2_CHECKPOINT="${STAGE2_OUTPUT_DIR}/slice_virtual_modality_generator_last.pt"
STAGE2_BEST_CHECKPOINT="${STAGE2_OUTPUT_DIR}/slice_virtual_modality_generator_best.pt"

VIS_NUM_CASES="${VIS_NUM_CASES:-12}"
VIS_DIR="${VIS_DIR:-${VIS_ROOT}/${RUN_NAME}}"
PIPELINE_REPORT="${PIPELINE_REPORT:-${REPORT_DIR}/${RUN_NAME}_pipeline_summary.json}"

stage_log() {
  local msg="$1"
  echo "[$(date '+%F %T')] ${msg}"
}

run_stage() {
  local name="$1"
  shift
  local stdout_log="${LOG_DIR}/${STAMP}_${name}.stdout.log"
  local stderr_log="${LOG_DIR}/${STAMP}_${name}.stderr.log"
  stage_log "START ${name}"
  stage_log "CMD $*"
  "$@" > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)
  stage_log "DONE ${name}"
}

csv_row_count() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print(0)
    raise SystemExit
with path.open(newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
}

file_count() {
  local root="$1"
  local pattern="$2"
  python - "$root" "$pattern" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
pattern = sys.argv[2]
print(sum(1 for _ in root.rglob(pattern)) if root.exists() else 0)
PY
}

ensure_environment() {
  if [[ "${SKIP_ENV_SETUP}" == "1" ]]; then
    stage_log "SKIP environment setup because SKIP_ENV_SETUP=1"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
      stage_log "Using existing conda env: ${ENV_NAME}"
    else
      stage_log "Creating conda env: ${ENV_NAME} python=${PYTHON_VERSION}"
      conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi
    conda activate "${ENV_NAME}"
  else
    if [[ ! -d "${VENV_DIR}" ]]; then
      stage_log "Creating venv: ${VENV_DIR}"
      python3 -m venv "${VENV_DIR}"
    else
      stage_log "Using existing venv: ${VENV_DIR}"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
  fi

  if [[ "${FORCE_INSTALL}" == "1" ]] || ! python - <<'PY' >/dev/null 2>&1
import importlib
required = ["torch", "numpy", "scipy", "nibabel", "huggingface_hub", "matplotlib"]
for name in required:
    importlib.import_module(name)
PY
  then
    stage_log "Installing Python dependencies, mode=${INSTALL_MODE}"
    python -m pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}"
    python -m pip install --index-url "${TORCH_INDEX_URL}" \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
    if [[ "${INSTALL_MODE}" == "full" ]]; then
      python -m pip install -r requirements.txt -i "${PIP_INDEX_URL}"
    else
      python -m pip install -i "${PIP_INDEX_URL}" \
        numpy==1.26.4 scipy==1.13.1 nibabel==5.2.1 \
        huggingface-hub==0.29.3 matplotlib==3.9.4 \
        tqdm==4.67.1 requests==2.32.3 PyYAML==6.0.1 \
        pillow==11.1.0 scikit-image==0.22.0
    fi
  else
    stage_log "Python dependencies already import successfully; no install needed."
  fi

  python - <<'PY'
import json
import platform
import torch
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY
}

download_and_prepare_data() {
  local rows
  rows="$(csv_row_count "${MANIFEST}")"
  if [[ "${FORCE_DOWNLOAD}" != "1" && "${FORCE_PREPARE}" != "1" && "${rows}" -ge "${MIN_PROCESSED_SUBJECTS}" ]]; then
    stage_log "Model-ready manifest exists with ${rows} rows; skipping download and prepare."
    return
  fi

  read -r -a dataset_args <<< "${DOWNLOAD_DATASETS}"
  if [[ "${FORCE_DOWNLOAD}" == "1" || "$(file_count "${RAW_ROOT}" "*.nii.gz")" -eq 0 || "${rows}" -lt "${MIN_PROCESSED_SUBJECTS}" ]]; then
    run_stage download_brats \
      python scripts/download_brats_hf.py \
        --datasets "${dataset_args[@]}" \
        --data-root "${DATA_ROOT}" \
        --max-workers "${DOWNLOAD_MAX_WORKERS}" \
        --retries "${DOWNLOAD_RETRIES}" \
        --retry-sleep "${DOWNLOAD_RETRY_SLEEP}"
  else
    stage_log "Raw NIfTI files already exist; skipping download."
  fi

  read -r -a target_shape_args <<< "${TARGET_SHAPE}"
  run_stage prepare_brats \
    python scripts/prepare_brats_model_ready.py \
      --raw-root "${RAW_ROOT}" \
      --output-root "${MODEL_READY_ROOT}" \
      --manifest-dir "${MANIFEST_DIR}" \
      --target-shape "${target_shape_args[@]}" \
      --crop-margin "${CROP_MARGIN}" \
      --workers "${PREPARE_WORKERS}"

  local prepared_rows
  prepared_rows="$(csv_row_count "${MANIFEST}")"
  if [[ "${prepared_rows}" -lt "${MIN_PROCESSED_SUBJECTS}" ]]; then
    echo "[ERROR] Prepared manifest has only ${prepared_rows} rows: ${MANIFEST}" >&2
    exit 2
  fi
  if [[ "${ALLOW_PREPARE_FAILURES}" != "1" && -s "${MANIFEST_DIR}/brats_model_ready_failed.csv" ]]; then
    echo "[ERROR] Prepare failures found: ${MANIFEST_DIR}/brats_model_ready_failed.csv" >&2
    exit 3
  fi
}

train_base_if_needed() {
  if [[ "${FORCE_BASE_TRAIN}" != "1" && -f "${BASE_CHECKPOINT}" ]]; then
    stage_log "Base checkpoint exists; skipping base training: ${BASE_CHECKPOINT}"
    return
  fi

  local base_args=(
    python scripts/train_slice_virtual_modality_drifting.py
    --manifest "${MANIFEST}"
    --device "${DEVICE}"
    --model-kind prompted
    --target-modality "${TARGET_MODALITY}"
    --spatial-size "${SPATIAL_SIZE}"
    --max-subjects "${MAX_SUBJECTS}"
    --val-subjects "${VAL_SUBJECTS}"
    --slices-per-subject "${SLICES_PER_SUBJECT}"
    --slice-crop-size "${SLICE_CROP_SIZE}"
    --slice-crop-jitter "${SLICE_CROP_JITTER}"
    --slice-crop-mode region_balanced
    --epochs "${BASE_EPOCHS}"
    --max-train-steps "${BASE_MAX_TRAIN_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --output-activation hardtanh
    --residual-scale 0.75
    --lesion-residual-scale 1.5
    --detail-residual-scale 0.20
    --enhancement-residual-scale 0.0
    --lr "${BASE_LR}"
    --recon-weight 1.0
    --nll-weight 0.02
    --gradient-weight 0.12
    --drift-weight 0.0
    --medical-drift-weight 0.0
    --region-moment-weight 0.12
    --edge-weight 0.12
    --enhancement-under-weight 0.40
    --top-intensity-weight 0.45
    --top-intensity-quantile 0.70
    --prompt-weight 0.12
    --prompt-balanced-bce-weight 0.25
    --prompt-max-pos-weight 100.0
    --prompt-et-weight 5.0
    --prompt-tc-weight 2.5
    --prompt-wt-weight 0.8
    --focus-dilation 2
    --focus-base 0.1
    --focus-et 22
    --focus-tc 7
    --focus-wt 0.2
    --background-weight 0.001
    --seed "${SEED}"
    --split-seed "${SPLIT_SEED}"
    --log-every "${LOG_EVERY}"
    --output-dir "${BASE_OUTPUT_DIR}"
    --report-path "${BASE_REPORT}"
  )

  if [[ -n "${BASE_RESUME_CHECKPOINT}" && -f "${BASE_RESUME_CHECKPOINT}" ]]; then
    base_args+=(--resume-checkpoint "${BASE_RESUME_CHECKPOINT}")
  fi

  run_stage train_base "${base_args[@]}"
}

train_enhancement() {
  if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "[ERROR] Base checkpoint missing after base stage: ${BASE_CHECKPOINT}" >&2
    exit 4
  fi
  if [[ "${FORCE_FINETUNE}" != "1" && -f "${FINAL_CHECKPOINT}" && -f "${FINAL_REPORT}" ]]; then
    stage_log "Final checkpoint/report exist; skipping enhancement fine-tune."
    return
  fi

  local finetune_args=(
    python scripts/train_slice_virtual_modality_drifting.py
    --manifest "${MANIFEST}"
    --device "${DEVICE}"
    --model-kind prompted
    --target-modality "${TARGET_MODALITY}"
    --spatial-size "${SPATIAL_SIZE}"
    --max-subjects "${MAX_SUBJECTS}"
    --val-subjects "${VAL_SUBJECTS}"
    --slices-per-subject "${SLICES_PER_SUBJECT}"
    --slice-crop-size "${SLICE_CROP_SIZE}"
    --slice-crop-jitter "${SLICE_CROP_JITTER}"
    --slice-crop-mode region_balanced
    --epochs "${EPOCHS}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --freeze-base-generator
    --detail-only-refinement
    --output-activation hardtanh
    --residual-scale 0.75
    --lesion-residual-scale 1.5
    --detail-residual-scale 0.28
    --enhancement-residual-scale 0.35
    --lr "${LR}"
    --recon-weight 1.0
    --nll-weight 0.02
    --gradient-weight 0.08
    --drift-weight 0.0
    --medical-drift-weight 0.006
    --medical-drift-memory-tokens 64
    --medical-drift-max-current-tokens 192
    --medical-drift-max-add-tokens 768
    --lesion-texture-weight 0.22
    --region-moment-weight 0.04
    --edge-weight 0.10
    --enhancement-under-weight 0.35
    --enhancement-residual-target-weight 0.85
    --enhancement-leak-weight 0.12
    --top-intensity-weight 0.38
    --top-intensity-quantile 0.70
    --focus-dilation 2
    --focus-base 0.1
    --focus-et 24
    --focus-tc 8
    --focus-wt 0.2
    --background-weight 0.001
    --resume-checkpoint "${BASE_CHECKPOINT}"
    --seed "${SEED}"
    --split-seed "${SPLIT_SEED}"
    --log-every "${LOG_EVERY}"
    --output-dir "${FINAL_OUTPUT_DIR}"
    --report-path "${FINAL_REPORT}"
  )

  if [[ "${TRAIN_PROMPT_WITH_DETAIL}" == "1" ]]; then
    finetune_args+=(
      --train-prompt-with-detail
      --prompt-weight 0.20
      --prompt-balanced-bce-weight 0.005
      --prompt-et-weight 5.0
      --prompt-tc-weight 2.0
      --prompt-wt-weight 0.5
    )
  fi

  run_stage train_enhancement "${finetune_args[@]}"
}

train_transport() {
  if [[ "${FORCE_FINETUNE}" != "1" && -f "${FINAL_CHECKPOINT}" && -f "${FINAL_REPORT}" ]]; then
    stage_log "Final checkpoint/report exist; skipping transport training."
    return
  fi

  local transport_args=(
    python scripts/train_slice_virtual_modality_drifting.py
    --manifest "${MANIFEST}"
    --device "${DEVICE}"
    --model-kind transport
    --target-modality "${TARGET_MODALITY}"
    --spatial-size "${SPATIAL_SIZE}"
    --max-subjects "${MAX_SUBJECTS}"
    --val-subjects "${VAL_SUBJECTS}"
    --slices-per-subject "${SLICES_PER_SUBJECT}"
    --slice-crop-size "${SLICE_CROP_SIZE}"
    --slice-crop-jitter "${SLICE_CROP_JITTER}"
    --slice-crop-mode region_balanced
    --epochs "${EPOCHS}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --output-activation hardtanh
    --transport-steps "${TRANSPORT_STEPS}"
    --transport-step-scale "${TRANSPORT_STEP_SCALE}"
    --transport-velocity-scale "${TRANSPORT_VELOCITY_SCALE}"
    --transport-init-blur-kernel "${TRANSPORT_INIT_BLUR_KERNEL}"
    --transport-velocity-weight "${TRANSPORT_VELOCITY_WEIGHT}"
    --transport-path-weight "${TRANSPORT_PATH_WEIGHT}"
    --transport-monotonic-weight "${TRANSPORT_MONOTONIC_WEIGHT}"
    --lr "${LR}"
    --recon-weight 1.0
    --nll-weight 0.02
    --gradient-weight 0.10
    --drift-weight 0.0
    --medical-drift-weight 0.004
    --medical-drift-memory-tokens 64
    --medical-drift-max-current-tokens 192
    --medical-drift-max-add-tokens 768
    --lesion-texture-weight 0.18
    --region-moment-weight 0.05
    --edge-weight 0.10
    --enhancement-under-weight 0.30
    --top-intensity-weight 0.35
    --top-intensity-quantile 0.70
    --focus-dilation 2
    --focus-base 0.1
    --focus-et 24
    --focus-tc 8
    --focus-wt 0.2
    --background-weight 0.001
    --seed "${SEED}"
    --split-seed "${SPLIT_SEED}"
    --log-every "${LOG_EVERY}"
    --output-dir "${FINAL_OUTPUT_DIR}"
    --report-path "${FINAL_REPORT}"
  )

  run_stage train_transport "${transport_args[@]}"
}

mine_hard_slices() {
  local source_checkpoint="${FINAL_BEST_CHECKPOINT}"
  if [[ ! -f "${source_checkpoint}" ]]; then
    source_checkpoint="${FINAL_CHECKPOINT}"
  fi
  if [[ ! -f "${source_checkpoint}" ]]; then
    echo "[ERROR] No transport checkpoint found for hard-slice mining." >&2
    exit 6
  fi
  if [[ "${FORCE_FINETUNE}" != "1" && -f "${HARD_SLICE_JSON}" ]]; then
    stage_log "Hard-slice JSON exists; skipping mining."
    return
  fi
  run_stage mine_hard_slices \
    python scripts/mine_hard_brats_slices.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${source_checkpoint}" \
      --device "${DEVICE}" \
      --target-modality "${TARGET_MODALITY}" \
      --spatial-size "${SPATIAL_SIZE}" \
      --max-subjects "${MAX_SUBJECTS}" \
      --val-subjects "${VAL_SUBJECTS}" \
      --seed "${SEED}" \
      --split-seed "${SPLIT_SEED}" \
      --candidate-slices-per-subject "${MINE_CANDIDATE_SLICES_PER_SUBJECT}" \
      --top-slices-per-subject "${MINE_TOP_SLICES_PER_SUBJECT}" \
      --max-mine-subjects "${MINE_MAX_SUBJECTS}" \
      --output-json "${HARD_SLICE_JSON}"
}

train_transport_stage2_hard() {
  if [[ "${TRAIN_STAGE2_HARD}" != "1" ]]; then
    stage_log "TRAIN_STAGE2_HARD=0; skipping hard-case transport fine-tune."
    return
  fi
  local source_checkpoint="${FINAL_BEST_CHECKPOINT}"
  if [[ ! -f "${source_checkpoint}" ]]; then
    source_checkpoint="${FINAL_CHECKPOINT}"
  fi
  if [[ ! -f "${source_checkpoint}" ]]; then
    echo "[ERROR] No transport checkpoint found for stage-2 fine-tune." >&2
    exit 7
  fi
  if [[ "${FORCE_FINETUNE}" != "1" && -f "${STAGE2_CHECKPOINT}" && -f "${STAGE2_REPORT}" ]]; then
    stage_log "Stage-2 checkpoint/report exist; skipping hard fine-tune."
    return
  fi
  run_stage train_transport_stage2_hard \
    python scripts/train_slice_virtual_modality_drifting.py \
      --manifest "${MANIFEST}" \
      --device "${DEVICE}" \
      --model-kind transport \
      --target-modality "${TARGET_MODALITY}" \
      --spatial-size "${SPATIAL_SIZE}" \
      --max-subjects "${MAX_SUBJECTS}" \
      --val-subjects "${VAL_SUBJECTS}" \
      --slices-per-subject "${SLICES_PER_SUBJECT}" \
      --slice-crop-size "${STAGE2_SLICE_CROP_SIZE}" \
      --slice-crop-jitter "${SLICE_CROP_JITTER}" \
      --slice-crop-mode region_balanced \
      --hard-slice-json "${HARD_SLICE_JSON}" \
      --hard-slice-prob "${HARD_SLICE_PROB}" \
      --hard-slice-top-k "${HARD_SLICE_TOP_K}" \
      --epochs "${STAGE2_EPOCHS}" \
      --max-train-steps "${MAX_TRAIN_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --output-activation hardtanh \
      --transport-steps "${TRANSPORT_STEPS}" \
      --transport-step-scale "${TRANSPORT_STEP_SCALE}" \
      --transport-velocity-scale "${TRANSPORT_VELOCITY_SCALE}" \
      --transport-init-blur-kernel "${TRANSPORT_INIT_BLUR_KERNEL}" \
      --transport-velocity-weight 0.45 \
      --transport-path-weight 0.18 \
      --transport-monotonic-weight 0.05 \
      --lr "${STAGE2_LR}" \
      --recon-weight 1.0 \
      --nll-weight 0.02 \
      --gradient-weight 0.12 \
      --drift-weight 0.0 \
      --window-feature-weight 0.05 \
      --medical-drift-weight 0.003 \
      --medical-drift-memory-tokens 64 \
      --medical-drift-max-current-tokens 192 \
      --medical-drift-max-add-tokens 768 \
      --lesion-texture-weight 0.24 \
      --region-moment-weight 0.06 \
      --edge-weight 0.14 \
      --enhancement-under-weight 0.45 \
      --lesion-boundary-weight 0.28 \
      --enhancement-contrast-weight 0.18 \
      --top-intensity-weight 0.45 \
      --top-intensity-quantile 0.72 \
      --focus-dilation 3 \
      --focus-base 0.08 \
      --focus-et 30 \
      --focus-tc 10 \
      --focus-wt 0.25 \
      --background-weight 0.001 \
      --resume-checkpoint "${source_checkpoint}" \
      --seed "${SEED}" \
      --split-seed "${SPLIT_SEED}" \
      --log-every "${LOG_EVERY}" \
      --output-dir "${STAGE2_OUTPUT_DIR}" \
      --report-path "${STAGE2_REPORT}"
}

visualize_cases() {
  local vis_checkpoint="${STAGE2_BEST_CHECKPOINT}"
  if [[ ! -f "${vis_checkpoint}" ]]; then
    vis_checkpoint="${STAGE2_CHECKPOINT}"
  fi
  if [[ ! -f "${vis_checkpoint}" ]]; then
    vis_checkpoint="${FINAL_BEST_CHECKPOINT}"
  fi
  if [[ ! -f "${vis_checkpoint}" ]]; then
    vis_checkpoint="${FINAL_CHECKPOINT}"
  fi
  if [[ ! -f "${vis_checkpoint}" ]]; then
    echo "[ERROR] No checkpoint missing before visualization." >&2
    exit 5
  fi
  run_stage visualize_cases \
    python scripts/visualize_slice_virtual_modality_generation.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${vis_checkpoint}" \
      --device "${DEVICE}" \
      --target-modality "${TARGET_MODALITY}" \
      --spatial-size "${SPATIAL_SIZE}" \
      --num-cases "${VIS_NUM_CASES}" \
      --slice-crop-size "${SLICE_CROP_SIZE}" \
      --apply-brain-mask \
      --output-dir "${VIS_DIR}"
}

write_pipeline_report() {
  python - \
    "${PIPELINE_REPORT}" "${ROOT}" "${DATA_ROOT}" "${RAW_ROOT}" "${MODEL_READY_ROOT}" \
    "${MANIFEST}" "${BASE_REPORT}" "${FINAL_REPORT}" "${VIS_DIR}" "${LOG_DIR}" \
    "${BASE_CHECKPOINT}" "${FINAL_CHECKPOINT}" <<'PY'
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

(
    report_path,
    root,
    data_root,
    raw_root,
    model_ready_root,
    manifest,
    base_report,
    final_report,
    vis_dir,
    log_dir,
    base_checkpoint,
    final_checkpoint,
) = [Path(item) for item in sys.argv[1:]]

def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))

def tree_stats(path: Path, pattern: str = "*") -> dict[str, float | int | str]:
    files = list(path.rglob(pattern)) if path.exists() else []
    total = sum(item.stat().st_size for item in files if item.is_file())
    return {
        "path": str(path),
        "files": len([item for item in files if item.is_file()]),
        "gb": round(total / (1024 ** 3), 4),
    }

def load_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_commit = None

try:
    import torch
    torch_info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    torch_info = {"error": type(exc).__name__ + ": " + str(exc)}

base = load_json(base_report)
final = load_json(final_report)
visual_summary = load_json(vis_dir / "summary.json")

report = {
    "status": "H20_PIPELINE_DONE",
    "git_commit": git_commit,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch_info,
    "paths": {
        "root": str(root),
        "data_root": str(data_root),
        "manifest": str(manifest),
        "base_checkpoint": str(base_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "base_report": str(base_report),
        "final_report": str(final_report),
        "visual_dir": str(vis_dir),
        "log_dir": str(log_dir),
    },
    "data": {
        "manifest_rows": count_rows(manifest),
        "raw_nii_gz": tree_stats(raw_root, "*.nii.gz"),
        "model_ready_npy": tree_stats(model_ready_root, "*.npy"),
    },
    "base_final_eval": (base or {}).get("final_eval", {}),
    "transport_final_eval": (final or {}).get("final_eval", {}),
    "visual_summary": visual_summary,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
}

main() {
  stage_log "Pipeline root: ${ROOT}"
  stage_log "Run name: ${RUN_NAME}"
  stage_log "Data root: ${DATA_ROOT}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  ensure_environment
  download_and_prepare_data
  if [[ "${TRAIN_PROMPTED_BASELINE}" == "1" ]]; then
    train_base_if_needed
  fi
  train_transport
  mine_hard_slices
  train_transport_stage2_hard
  visualize_cases
  write_pipeline_report
  stage_log "ALL DONE"
  stage_log "Pipeline report: ${PIPELINE_REPORT}"
}

main "$@"
