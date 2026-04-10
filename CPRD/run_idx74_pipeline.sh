#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_idx74_log.txt"

cd "$WORK_DIR"

########################################
# Step 1: Build dataset (index_age=74)
########################################
echo "===== Step 1: Build dataset (index_age=74, 31 pure codes) =====" | tee "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

$PYTHON build_dementia_cr_idx74_dataset.py 2>&1 | tee -a "$LOG_FILE"

echo "Dataset build done: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

########################################
# Step 2: Train (4-GPU DDP)
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "===== Step 2: Train (event_only, lambda=6, idx74, 15 epochs) =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"

rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_idx74.ckpt" 2>/dev/null || true

$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_idx74 2>&1 | tee -a "$LOG_FILE"

echo "End train: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

########################################
# Step 3: Single-GPU eval (best val loss)
########################################
echo "===== Step 3: Single-GPU eval (best val loss checkpoint) =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_idx74_eval 2>&1 | tee -a "$LOG_FILE"

echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
