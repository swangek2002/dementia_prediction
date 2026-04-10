#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_aug_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES-Augmented Fine-Tuning: idx72, no SAW, study to 2022" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

########################################
# Step 1: Build dataset + HES post-processing
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 1: Build HES-augmented dataset =====" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

$PYTHON build_dementia_cr_hes_aug.py 2>&1 | tee -a "$LOG_FILE"

echo "Dataset build done: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 2: Verify dataset statistics
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 2: Verify dataset stats =====" | tee -a "$LOG_FILE"

$PYTHON verify_hes_aug_dataset.py 2>&1 | tee -a "$LOG_FILE"

echo "Verification done: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 3: Train (4-GPU DDP, 15 epochs max)
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "" | tee -a "$LOG_FILE"
echo "===== Step 3: Train (4-GPU, 15 epochs max) =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"

rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_aug.ckpt" 2>/dev/null || true

$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug 2>&1 | tee -a "$LOG_FILE"

echo "End train: $(date)" | tee -a "$LOG_FILE"

########################################
# Step 4: Single-GPU eval (best val_loss checkpoint)
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 4: Single-GPU eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug_eval 2>&1 | tee -a "$LOG_FILE"

echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
