#!/bin/bash
# ===========================================================================
# Pipeline Orchestrator — Phases 1-4
#
# Submits all SLURM jobs in dependency order up to (and including) Phase 4
# (CoC calibration). A MANUAL STEP is required before proceeding to Phase 5.
#
# Usage:
#   bash scripts/run_pipeline.sh
#
# Override defaults with environment variables, e.g.:
#   OUT_ROOT=/scratch/mydata RAW_ROOT=./data bash scripts/run_pipeline.sh
# ===========================================================================

set -euo pipefail

export OUT_ROOT=${OUT_ROOT:-./output}
export RAW_ROOT=${RAW_ROOT:-./data}

# --- Environment activation ---
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-focusmamba_data}
if [[ -f "$CONDA_BASE/bin/activate" ]]; then
    source "$CONDA_BASE/bin/activate"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

mkdir -p "$OUT_ROOT/logs" logs

echo "============================================================"
echo " FocusMamba Data Pipeline — Phases 1–4"
echo " OUT_ROOT: $OUT_ROOT"
echo " RAW_ROOT: $RAW_ROOT"
echo "============================================================"

# --- Phase 1: Validate and index (runs locally, no GPU) ---
echo ""
echo ">>> Phase 1: Dataset Validation"
python scripts/01_validate_dataset.py \
    --raw_root   "$RAW_ROOT" \
    --out_root   "$OUT_ROOT" \
    --min_frames 16

N_SCENES=$(python -c "
import json
idx = json.load(open('$OUT_ROOT/scene_index.json'))
print(len(idx) - 1)  # 0-indexed for SLURM arrays
")
echo "  Scenes: $((N_SCENES + 1))"

export SCENE_INDEX="$OUT_ROOT/scene_index.json"

# --- Phase 2: Deblur (GPU array) ---
echo ""
echo ">>> Phase 2: Submitting DeblurNet jobs..."
JOB_DEBLUR=$(sbatch --parsable \
    --export=ALL,OUT_ROOT="$OUT_ROOT",RAW_ROOT="$RAW_ROOT",SCENE_INDEX="$SCENE_INDEX" \
    --array=0-${N_SCENES}%32 \
    jobs/run_deblur.sh)
echo "  Job ID: $JOB_DEBLUR"

# --- Phase 3: Depth (GPU array, depends on deblur) ---
echo ""
echo ">>> Phase 3: Submitting Depth (VDA) jobs..."
JOB_DEPTH=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEBLUR \
    --export=ALL,OUT_ROOT="$OUT_ROOT",SCENE_INDEX="$SCENE_INDEX" \
    --array=0-${N_SCENES}%16 \
    jobs/run_depth.sh)
echo "  Job ID: $JOB_DEPTH (depends on $JOB_DEBLUR)"

# --- Phase 4: CoC calibration (CPU, depends on depth) ---
echo ""
echo ">>> Phase 4: Submitting CoC calibration..."
JOB_COC=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEPTH \
    --partition=cpu --ntasks=1 --mem=32G --time=01:00:00 \
    --output=logs/coc_calib_%j.out --error=logs/coc_calib_%j.err \
    --export=ALL,OUT_ROOT="$OUT_ROOT" \
    --wrap="python scripts/04_calibrate_coc.py \
            --out_root '$OUT_ROOT' \
            --sample_fraction 0.2 \
            --output_stats '$OUT_ROOT/logs/coc_stats.json' \
            --output_plot  '$OUT_ROOT/logs/coc_histogram.png'")
echo "  Job ID: $JOB_COC (depends on $JOB_DEPTH)"

echo ""
echo "================================================================"
echo " Phases 1–4 submitted."
echo ""
echo " MANUAL STEP REQUIRED — do not run Phase 5 until:"
echo "   1. Job $JOB_COC completes."
echo "   2. Read: $OUT_ROOT/logs/coc_stats.json"
echo "   3. Verify recommended_max_coc is plausible (0.02–0.20)."
echo "   4. Then run:"
echo "        bash scripts/run_pipeline_phase5.sh <max_coc>"
echo "      e.g.:"
echo "        bash scripts/run_pipeline_phase5.sh 0.063"
echo "================================================================"
