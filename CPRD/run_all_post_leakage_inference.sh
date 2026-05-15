#!/bin/bash
# Run inference + cohort metrics for all post-leakage-fix dual-backbone models, sequentially.
# Single backbone (V2 ablation) handled separately.

set -e
cd /Data0/swangek_data/991/CPRD
export PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD:$PYTHONPATH

PYTHON=/Data0/swangek_data/conda_envs/survivehr/bin/python
LOG_DIR=/Data0/swangek_data/991/CPRD/cohort_ctd_logs
mkdir -p "$LOG_DIR"

# Order: V4 (most recent post-V5), V3, V2 (V2 labels), dual_baseline (clean baseline pre V2 labels), crossattn
# Skip if NPZ already exists
for tag in v4 v3 v2 dual_baseline crossattn; do
    NPZ="/Data0/swangek_data/991/CPRD/data/test_cif_${tag}_full.npz"
    if [ "$tag" = "dual_baseline" ]; then
        NPZ="/Data0/swangek_data/991/CPRD/data/test_cif_dual_baseline_full.npz"
    fi
    if [ -f "$NPZ" ]; then
        echo "[$(date)] [$tag] NPZ already exists, skipping inference"
    else
        echo "[$(date)] [$tag] Starting inference..."
        CUDA_VISIBLE_DEVICES=0 $PYTHON -u inference_dual_cohort_ctd.py "$tag" > "$LOG_DIR/${tag}_infer.log" 2>&1
        echo "[$(date)] [$tag] Inference complete"
    fi

    echo "[$(date)] [$tag] Computing cohort metrics..."
    $PYTHON -u compute_cohort_ctd_generic.py "$NPZ" "$tag" > "$LOG_DIR/${tag}_metrics.log" 2>&1
    echo "[$(date)] [$tag] Metrics done. Result:"
    tail -10 "$LOG_DIR/${tag}_metrics.log"
done

# V2 ablation (single backbone) — separate
NPZ="/Data0/swangek_data/991/CPRD/data/test_cif_v2_ablation_full.npz"
if [ -f "$NPZ" ]; then
    echo "[$(date)] [v2_ablation] NPZ exists, skipping inference"
else
    echo "[$(date)] [v2_ablation] Starting inference..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON -u inference_single_v2ablation.py > "$LOG_DIR/v2_ablation_infer.log" 2>&1
fi
echo "[$(date)] [v2_ablation] Computing cohort metrics..."
$PYTHON -u compute_cohort_ctd_generic.py "$NPZ" "v2_ablation" > "$LOG_DIR/v2_ablation_metrics.log" 2>&1
tail -10 "$LOG_DIR/v2_ablation_metrics.log"

echo "[$(date)] ALL DONE"
