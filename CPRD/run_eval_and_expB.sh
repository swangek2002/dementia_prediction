#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

cd "$WORK_DIR"

########################################
# Step 1: Single-GPU eval of Exp C
########################################
echo "===== Step 1: Single-GPU eval of Experiment C =====" | tee "${LOG_DIR}/eval_combined_1gpu_log.txt"
echo "Start: $(date)" | tee -a "${LOG_DIR}/eval_combined_1gpu_log.txt"

CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_Combined_eval \
    2>&1 | tee -a "${LOG_DIR}/eval_combined_1gpu_log.txt"

echo "End: $(date)" | tee -a "${LOG_DIR}/eval_combined_1gpu_log.txt"
echo ""

########################################
# Step 2: Train Experiment B (4 GPU)
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "===== Step 2: Experiment B (event_only, lambda=3) =====" | tee "${LOG_DIR}/finetune_cr_eventweight_log.txt"
echo "Start: $(date)" | tee -a "${LOG_DIR}/finetune_cr_eventweight_log.txt"
echo "  mode: event_only, event_lambda=3.0" | tee -a "${LOG_DIR}/finetune_cr_eventweight_log.txt"

rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_EventWeight.ckpt" 2>/dev/null || true

$PYTHON run_experiment.py \
    --config-name=config_FineTune_Dementia_CR_EventWeight \
    2>&1 | tee -a "${LOG_DIR}/finetune_cr_eventweight_log.txt"

echo "End: $(date)" | tee -a "${LOG_DIR}/finetune_cr_eventweight_log.txt"
echo "===== ALL DONE ====="
