#!/bin/bash
# HES Static Covariates Pipeline (Option B)
# GP sequences + HES label augmentation + HES summary stats as static covariates
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_static_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES Static Covariates Fine-Tuning (Option B)"            | tee -a "$LOG_FILE"
echo "  GP seq + HES labels + 8 HES static features"             | tee -a "$LOG_FILE"
echo "  Start: $(date)"                                           | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Step 1: Build HES summary features
echo "===== Step 1: Build HES summary features =====" | tee -a "$LOG_FILE"
$PYTHON build_hes_summary_features.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Build dataset (GP sequences + HES labels + HES static features)
echo "===== Step 2: Build dataset =====" | tee -a "$LOG_FILE"
$PYTHON build_dementia_cr_hes_static.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Train (4-GPU DDP)
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "===== Step 3: Train =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_static.ckpt" 2>/dev/null || true
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_static 2>&1 | tee -a "$LOG_FILE"
echo "End train: $(date)" | tee -a "$LOG_FILE"

# Step 4: Single-GPU eval
echo "===== Step 4: Eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_static_eval 2>&1 | tee -a "$LOG_FILE"
echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
