#!/bin/bash
# ===========================================================================
# Pipeline Orchestrator — Phases 5-6
#
# Run AFTER Phase 4 (CoC calibration) completes and you've verified
# the recommended_max_coc value.
#
# Usage:
#   bash scripts/run_pipeline_phase5.sh <max_coc>
#   e.g.:
#   bash scripts/run_pipeline_phase5.sh 0.063
# ===========================================================================

set -euo pipefail

MAX_COC=${1:?"Usage: $0 <max_coc e.g. 0.063>"}

# Validate the value
python -c "
v = float('$MAX_COC')
assert 1e-6 < v < 1.0, f'max_coc={v} looks wrong — expected 1e-6–1.0'
print(f'max_coc={v} accepted.')
"

export OUT_ROOT=${OUT_ROOT:-./output}
export MAX_COC

# --- Environment activation ---
CONDA_BASE=${CONDA_BASE:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-focusmamba_data}
if [[ -f "$CONDA_BASE/bin/activate" ]]; then
    source "$CONDA_BASE/bin/activate"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

SCENE_INDEX="$OUT_ROOT/scene_index.json"

N_SCENES=$(python -c "
import json
idx = json.load(open('$SCENE_INDEX'))
print(len(idx) - 1)
")

mkdir -p logs

echo "============================================================"
echo " FocusMamba Data Pipeline — Phases 5–6"
echo " OUT_ROOT: $OUT_ROOT"
echo " MAX_COC:  $MAX_COC"
echo " Scenes:   $((N_SCENES + 1))"
echo "============================================================"

# --- Phase 5: Render + GT labels (GPU array) ---
echo ""
echo ">>> Phase 5: Submitting BokehNet render + focus map jobs..."
JOB_RENDER=$(sbatch --parsable \
    --export=ALL,OUT_ROOT="$OUT_ROOT",MAX_COC="$MAX_COC",SCENE_INDEX="$SCENE_INDEX" \
    --array=0-${N_SCENES}%16 \
    jobs/run_render.sh)
echo "  Job ID: $JOB_RENDER"

# --- Phase 6: Validate (CPU, depends on render) ---
echo ""
echo ">>> Phase 6: Submitting validation..."
sbatch \
    --dependency=afterok:$JOB_RENDER \
    --partition=cpu --ntasks=1 --mem=16G --time=00:30:00 \
    --output=logs/validate_%j.out --error=logs/validate_%j.err \
    --export=ALL,OUT_ROOT="$OUT_ROOT" \
    --wrap="python scripts/06_validate_outputs.py \
            --out_root    '$OUT_ROOT' \
            --scene_index '$OUT_ROOT/scene_index.json' \
            --report      '$OUT_ROOT/logs/validation_report.json'"

echo ""
echo "================================================================"
echo " Phases 5–6 submitted."
echo " Monitor with: squeue -u \$USER"
echo " Final report: $OUT_ROOT/logs/validation_report.json"
echo "================================================================"
