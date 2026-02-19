#!/bin/bash
# ===========================================================================
# SLURM array job wrapper for Phase 2: DeblurNet
# Usage: sbatch --array=0-N%32 jobs/run_deblur.sh
# ===========================================================================
#SBATCH --job-name=deblur_%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/deblur_%A_%a.out
#SBATCH --error=logs/deblur_%A_%a.err

set -euo pipefail

# --- Environment ---
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-focusmamba_data}
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"

# --- Variables (set by pipeline orchestrator or export) ---
OUT_ROOT=${OUT_ROOT:?"OUT_ROOT must be set"}
RAW_ROOT=${RAW_ROOT:-./data}
SCENE_INDEX=${SCENE_INDEX:-${OUT_ROOT}/scene_index.json}
STEPS=${DEBLUR_STEPS:-8}
MODEL_ID=${MODEL_ID:-black-forest-labs/FLUX.1-dev}
LORA_PATH=${LORA_PATH:-.}
LORA_WEIGHT=${DEBLUR_LORA_WEIGHT:-deblurNet.safetensors}

echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
mkdir -p logs

python scripts/02_deblur.py \
    --out_root      "$OUT_ROOT" \
    --raw_root      "$RAW_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$SLURM_ARRAY_TASK_ID" \
    --steps         "$STEPS" \
    --model_id      "$MODEL_ID" \
    --lora_path     "$LORA_PATH" \
    --lora_weight   "$LORA_WEIGHT"

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
