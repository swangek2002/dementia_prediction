#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

cd "$WORK_DIR"

########################################
# Step 1: Train Experiment C (lambda=3)
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "===== Step 1: Experiment C (combined, lambda=3) =====" | tee "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
echo "Start: $(date)" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
echo "  mode: combined, event_lambda=3.0, alpha=2.0, w_t_max=3.0" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"

echo "[Cleaning old Combined checkpoints...]" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_Combined.ckpt" 2>/dev/null || true

$PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_Combined \
    2>&1 | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"

echo "End Train: $(date)" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
echo "" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"

########################################
# Step 2: Single-GPU eval of Exp C
########################################
echo "===== Step 2: Single-GPU eval of Experiment C =====" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
echo "Start Eval: $(date)" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"

CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_Combined_eval \
    2>&1 | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"

echo "End Eval: $(date)" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"
echo "===== ALL DONE =====" | tee -a "${LOG_DIR}/finetune_cr_combined_L3_log.txt"