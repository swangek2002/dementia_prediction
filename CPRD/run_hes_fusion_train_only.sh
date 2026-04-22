#!/bin/bash
# Re-run ONLY the training + eval steps of the HES fusion pipeline.
# The dataset + fused database are already built; only training failed
# previously because a stale last.ckpt caused auto-resume to skip everything.
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_fusion_train_only_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES Fusion Fine-Tuning (train + eval only, data pre-built)" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# CRITICAL: blow away anything that would cause PL to auto-resume and skip
# the real fine-tune. last.ckpt is the killer -- it's at epoch>=15 from the
# previous hes_aug run, so max_epochs=15 is "already reached" on start.
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_fusion.ckpt" 2>/dev/null || true
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
echo "Removed stale checkpoints (fusion ckpt, last.ckpt)" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "===== Train =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion 2>&1 | tee -a "$LOG_FILE"
echo "End train: $(date)" | tee -a "$LOG_FILE"

echo "===== Eval (single-GPU) =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion_eval 2>&1 | tee -a "$LOG_FILE"
echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
