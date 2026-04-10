#!/bin/bash
set -e

##############################################################################
# run_idx68_cv5_pipeline.sh
# =========================
# 5-Fold Cross-Validation pipeline for dementia CR fine-tuning
# Index age = 68, lambda = 6, 31 pure dementia codes, 10 epochs max
#
# Steps per fold:
#   1. Build dataset using fold-specific practice_id splits
#   2. 4-GPU DDP fine-tuning (best val_loss checkpoint saved)
#   3. 1-GPU evaluation on test set using best checkpoint
#
# Usage:
#   bash run_idx68_cv5_pipeline.sh
#   # Or in tmux:
#   tmux new-session -d -s cv5_idx68 "bash run_idx68_cv5_pipeline.sh"
##############################################################################

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_idx68_cv5_log.txt"

cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  5-Fold CV Pipeline: idx68, lambda=6, 31 codes, 10 epochs" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

########################################
# Step 0: Generate 5-fold splits
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 0: Generate 5-fold practice_id splits =====" | tee -a "$LOG_FILE"
$PYTHON generate_5fold_splits.py 2>&1 | tee -a "$LOG_FILE"

########################################
# Step 0.5: Generate YAML configs
########################################
echo "" | tee -a "$LOG_FILE"
echo "===== Step 0.5: Generate YAML configs =====" | tee -a "$LOG_FILE"
$PYTHON generate_cv5_configs.py 2>&1 | tee -a "$LOG_FILE"

########################################
# Loop over 5 folds
########################################
for FOLD in 0 1 2 3 4; do
    FOLD_TAG="fold${FOLD}"
    FINE_TUNE_ID="FineTune_Dementia_CR_idx68_cv_${FOLD_TAG}"
    CKPT_FILE="${CKPT_DIR}/crPreTrain_small_1337_${FINE_TUNE_ID}.ckpt"
    DS_PATH="/Data0/swangek_data/991/CPRD/data/FoundationalModel/FineTune_Dementia_CR_idx68_cv/${FOLD_TAG}/"

    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo "  FOLD ${FOLD} / 4" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"

    ############################
    # Step 1: Build dataset
    ############################
    echo "" | tee -a "$LOG_FILE"
    echo "===== Fold ${FOLD} Step 1: Build dataset =====" | tee -a "$LOG_FILE"
    echo "Start: $(date)" | tee -a "$LOG_FILE"

    # Clean previous data if exists
    rm -rf "$DS_PATH" 2>/dev/null || true

    $PYTHON build_dementia_cr_idx68_cv.py --fold $FOLD 2>&1 | tee -a "$LOG_FILE"

    echo "Dataset build done: $(date)" | tee -a "$LOG_FILE"

    ############################
    # Step 2: Train (4-GPU DDP)
    ############################
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    echo "" | tee -a "$LOG_FILE"
    echo "===== Fold ${FOLD} Step 2: Train (4-GPU, 10 epochs max) =====" | tee -a "$LOG_FILE"
    echo "Start train: $(date)" | tee -a "$LOG_FILE"

    # Remove old checkpoint to avoid confusion
    rm -f "$CKPT_FILE" 2>/dev/null || true

    $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_idx68_cv_${FOLD_TAG} 2>&1 | tee -a "$LOG_FILE"

    echo "End train: $(date)" | tee -a "$LOG_FILE"

    ############################
    # Step 3: Eval (1-GPU)
    ############################
    echo "" | tee -a "$LOG_FILE"
    echo "===== Fold ${FOLD} Step 3: Eval (best val_loss checkpoint) =====" | tee -a "$LOG_FILE"
    echo "Start eval: $(date)" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_idx68_cv_${FOLD_TAG}_eval 2>&1 | tee -a "$LOG_FILE"

    echo "End eval: $(date)" | tee -a "$LOG_FILE"

done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  ALL 5 FOLDS DONE: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
