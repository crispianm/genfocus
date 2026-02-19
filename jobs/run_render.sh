#!/bin/bash
# ===========================================================================
# SLURM array job wrapper for Phase 5: BokehNet Rendering + GT Focus Maps
# Usage: sbatch --array=0-N%16 jobs/run_render.sh
# ===========================================================================
#SBATCH --job-name=render_%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/render_%A_%a.out
#SBATCH --error=logs/render_%A_%a.err

set -euo pipefail

# --- Environment ---
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-focusmamba_data}
source "$CONDA_BASE/bin/activate"
conda activate "$CONDA_ENV"

# --- Variables ---
OUT_ROOT=${OUT_ROOT:?"OUT_ROOT must be set"}
MAX_COC=${MAX_COC:?"MAX_COC must be set (from Phase 4 calibration)"}
SCENE_INDEX=${SCENE_INDEX:-${OUT_ROOT}/scene_index.json}
N_SETS=${N_SETS:-4}
RENDER_STEPS=${RENDER_STEPS:-30}
MODEL_ID=${MODEL_ID:-black-forest-labs/FLUX.1-dev}
LORA_PATH=${LORA_PATH:-.}
BOKEH_LORA_WEIGHT=${BOKEH_LORA_WEIGHT:-bokehNet.safetensors}

echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "MAX_COC=$MAX_COC"
mkdir -p logs

python scripts/05_render_and_labels.py \
    --out_root      "$OUT_ROOT" \
    --scene_index   "$SCENE_INDEX" \
    --scene_idx     "$SLURM_ARRAY_TASK_ID" \
    --max_coc       "$MAX_COC" \
    --n_sets        "$N_SETS" \
    --steps         "$RENDER_STEPS" \
    --model_id      "$MODEL_ID" \
    --lora_path     "$LORA_PATH" \
    --lora_weight   "$BOKEH_LORA_WEIGHT"

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
