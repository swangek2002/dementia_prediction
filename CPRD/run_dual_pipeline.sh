#!/bin/bash
# Dual-Backbone Fine-Tuning Pipeline
# GP backbone (pretrained) + HES backbone (pretrained) + Gated Fusion + Survival Head
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_dual_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  Dual-Backbone Fine-Tuning Pipeline"                        | tee -a "$LOG_FILE"
echo "  GP backbone:  crPreTrain_small_1337.ckpt"                  | tee -a "$LOG_FILE"
echo "  HES backbone: crPreTrain_HES_1337.ckpt"                   | tee -a "$LOG_FILE"
echo "  Fusion: gated"                                             | tee -a "$LOG_FILE"
echo "  Start: $(date)"                                            | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Step 1: Train (single GPU — dual backbone uses more VRAM)
export CUDA_VISIBLE_DEVICES=0
echo "===== Step 1: Train =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_dual.ckpt" 2>/dev/null || true
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
$PYTHON run_dual_experiment.py --config-name=config_FineTune_Dementia_CR_dual 2>&1 | tee -a "$LOG_FILE"
echo "End train: $(date)" | tee -a "$LOG_FILE"

# Step 2: Single-GPU eval
echo "===== Step 2: Eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 $PYTHON run_dual_experiment.py --config-name=config_FineTune_Dementia_CR_dual_eval 2>&1 | tee -a "$LOG_FILE"
echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
