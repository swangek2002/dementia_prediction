#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

LOG_FILE="${LOG_DIR}/finetune_cr_combined_log.txt"

echo "===== Experiment C: Combined weighting (event-type + reverse time) =====" | tee "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  mode: combined (event_lambda=10, alpha=2.0, tau=0.33)" | tee -a "$LOG_FILE"
echo "  w_t_max=3.0, w_total_max=20.0" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$WORK_DIR"

echo "[Step 1] Cleaning old Combined checkpoints..." | tee -a "$LOG_FILE"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_Combined.ckpt" 2>/dev/null || true

echo "[Step 2] Starting combined-weighting fine-tuning..." | tee -a "$LOG_FILE"
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_Combined 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo "===== DONE =====" | tee -a "$LOG_FILE"
