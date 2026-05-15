#!/bin/bash
# run_overnight_pipeline.sh
# ========================
# Orchestrates the full overnight pipeline:
#   1. Wait for V2 ablation training (PID 3789819) to complete
#   2. V2 ablation single-GPU eval
#   3. V4 test inference + calibration analysis (Approach A)
#   4. V2 ablation test inference + calibration analysis (Approach A)
#   5. V5 pipeline: V4 inference on V4 train → top 5% selection → V5 dataset
#   6. V5 training (~24h)
#   7. V5 eval + V5 test inference + V5 calibration
#
# All steps log to /Data0/swangek_data/991/CPRD/overnight_pipeline.log
# Each step also has its own log under /Data0/swangek_data/991/CPRD/*.log

set -u  # error on unset vars (don't use -e: we want to continue past individual step failures)

PIPELINE_LOG=/Data0/swangek_data/991/CPRD/overnight_pipeline.log
PYTHON=/Data0/swangek_data/conda_envs/survivehr/bin/python
PYTHONPATH_VAL="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK=/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR
CKPT_DIR=/Data0/swangek_data/991/CPRD/output/checkpoints

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }

log "============================================================"
log "Overnight pipeline started"
log "============================================================"

# --- Step 1: Wait for V2 ablation training ---
log "[Step 1] Waiting for V2 ablation training (PID 3789819) to finish..."
while ps -p 3789819 > /dev/null 2>&1; do
    sleep 300
done
log "[Step 1] V2 ablation training finished."

# --- Step 2: V2 ablation single-GPU eval ---
log "[Step 2] Running V2 ablation eval..."
rm -f "$CKPT_DIR/last.ckpt"
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" run_experiment.py \
    --config-name config_FineTune_Dementia_CR_hes_static_v2_ablation_eval \
    > /Data0/swangek_data/991/CPRD/v2_ablation_eval.log 2>&1
log "[Step 2] V2 ablation eval done. See v2_ablation_eval.log"

# --- Step 3: V4 test inference + V4 calibration ---
log "[Step 3a] V4 test inference..."
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_test_v4.py \
    > /Data0/swangek_data/991/CPRD/v4_test_inference.log 2>&1
log "[Step 3a] V4 test inference done."

log "[Step 3b] V4 calibration analysis (Approach A)..."
mkdir -p /Data0/swangek_data/991/CPRD/calibration_outputs/v4
PYTHONPATH="$PYTHONPATH_VAL" "$PYTHON" "$WORK/compute_calibration.py" \
    --input /Data0/swangek_data/991/CPRD/data/test_cif_v4.csv \
    --output-dir /Data0/swangek_data/991/CPRD/calibration_outputs/v4 \
    --label V4 \
    >> "$PIPELINE_LOG" 2>&1
log "[Step 3b] V4 calibration done."

# --- Step 4: V2 ablation test inference + calibration ---
log "[Step 4a] V2 ablation test inference..."
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_test_ablation.py \
    > /Data0/swangek_data/991/CPRD/v2_ablation_test_inference.log 2>&1
log "[Step 4a] V2 ablation test inference done."

log "[Step 4b] V2 ablation calibration analysis..."
mkdir -p /Data0/swangek_data/991/CPRD/calibration_outputs/v2_ablation
PYTHONPATH="$PYTHONPATH_VAL" "$PYTHON" "$WORK/compute_calibration.py" \
    --input /Data0/swangek_data/991/CPRD/data/test_cif_v2_ablation.csv \
    --output-dir /Data0/swangek_data/991/CPRD/calibration_outputs/v2_ablation \
    --label V2_ablation \
    >> "$PIPELINE_LOG" 2>&1
log "[Step 4b] V2 ablation calibration done."

# --- Step 5: V5 pipeline preparation ---
log "[Step 5a] V4 model inference on V4 train set (for V5 selection)..."
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_train_cif_v4.py \
    > /Data0/swangek_data/991/CPRD/v4_train_inference.log 2>&1
log "[Step 5a] V4 train inference done."

log "[Step 5b] Building V5 dataset (top 5% pseudo-labeling + overlap analysis)..."
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" "$PYTHON" build_dementia_cr_hes_aug_v5.py \
    > /Data0/swangek_data/991/CPRD/v5_build.log 2>&1
log "[Step 5b] V5 dataset built. See v5_build.log for overlap analysis."

# --- Step 6: V5 training ---
log "[Step 6] V5 training..."
rm -f "$CKPT_DIR/last.ckpt"
rm -f "$CKPT_DIR/crPreTrain_small_1337_FineTune_Dementia_CR_dual_v5.ckpt"
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" run_dual_experiment.py \
    --config-name config_FineTune_Dementia_CR_dual_v5 \
    > /Data0/swangek_data/991/CPRD/dual_v5_train.log 2>&1
log "[Step 6] V5 training done."

# --- Step 7: V5 eval + V5 calibration ---
log "[Step 7a] V5 eval (single GPU)..."
rm -f "$CKPT_DIR/last.ckpt"
cd "$WORK"
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" run_dual_experiment.py \
    --config-name config_FineTune_Dementia_CR_dual_v5_eval \
    > /Data0/swangek_data/991/CPRD/dual_v5_eval.log 2>&1
log "[Step 7a] V5 eval done."

# V5 test inference (reuse v4 inference template, just need to swap paths)
log "[Step 7b] V5 test inference for calibration..."
cd "$WORK"
# Create temporary V5 inference script by copying V4 inference and patching
sed -e 's|dual_v4.ckpt|dual_v5.ckpt|g' \
    -e 's|hes_static_v4/|hes_static_v5/|g' \
    -e 's|test_cif_v4.csv|test_cif_v5.csv|g' \
    inference_test_v4.py > inference_test_v5.py
PYTHONPATH="$PYTHONPATH_VAL" CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_test_v5.py \
    > /Data0/swangek_data/991/CPRD/v5_test_inference.log 2>&1
log "[Step 7b] V5 test inference done."

log "[Step 7c] V5 calibration analysis..."
mkdir -p /Data0/swangek_data/991/CPRD/calibration_outputs/v5
PYTHONPATH="$PYTHONPATH_VAL" "$PYTHON" "$WORK/compute_calibration.py" \
    --input /Data0/swangek_data/991/CPRD/data/test_cif_v5.csv \
    --output-dir /Data0/swangek_data/991/CPRD/calibration_outputs/v5 \
    --label V5 \
    >> "$PIPELINE_LOG" 2>&1
log "[Step 7c] V5 calibration done."

log "============================================================"
log "Overnight pipeline COMPLETE"
log "============================================================"

# --- Step 8: Summarize results ---
log "[Step 8] Summary of all metrics:"
log "  V2 ablation eval: see /Data0/swangek_data/991/CPRD/v2_ablation_eval.log"
log "  V4 calibration:   see /Data0/swangek_data/991/CPRD/calibration_outputs/v4/"
log "  V2 ablation cal:  see /Data0/swangek_data/991/CPRD/calibration_outputs/v2_ablation/"
log "  V5 training log:  see /Data0/swangek_data/991/CPRD/dual_v5_train.log"
log "  V5 eval log:      see /Data0/swangek_data/991/CPRD/dual_v5_eval.log"
log "  V5 calibration:   see /Data0/swangek_data/991/CPRD/calibration_outputs/v5/"
