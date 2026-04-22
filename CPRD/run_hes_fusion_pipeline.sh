#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

LOG_FILE="${LOG_DIR}/finetune_cr_hes_fusion_log.txt"
cd "$WORK_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "  HES Full Fusion Fine-Tuning" | tee -a "$LOG_FILE"
echo "  idx72, no SAW, OMOP mapping, all dementia patients" | tee -a "$LOG_FILE"
echo "  Start: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Step 1: Build OMOP mapping dictionary
echo "===== Step 1: Build OMOP mapping =====" | tee -a "$LOG_FILE"
$PYTHON build_omop_mapping.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Build HES events lookup (with HOTFIX 1 + 2)
echo "===== Step 2: Build HES events =====" | tee -a "$LOG_FILE"
$PYTHON build_hes_events.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Create fused database copy
echo "===== Step 3: Prepare fused database =====" | tee -a "$LOG_FILE"
$PYTHON prepare_hes_fusion_db.py 2>&1 | tee -a "$LOG_FILE"

# Step 4: Build fine-tuning dataset
echo "===== Step 4: Build dataset =====" | tee -a "$LOG_FILE"
$PYTHON build_dementia_cr_hes_fusion.py 2>&1 | tee -a "$LOG_FILE"

# Step 5: Verify dataset
echo "===== Step 5: Verify dataset =====" | tee -a "$LOG_FILE"
$PYTHON verify_hes_fusion_dataset.py 2>&1 | tee -a "$LOG_FILE"

# Step 6: Train (4-GPU DDP)
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "===== Step 6: Train =====" | tee -a "$LOG_FILE"
echo "Start train: $(date)" | tee -a "$LOG_FILE"
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR_hes_fusion.ckpt" 2>/dev/null || true
# CRITICAL: also remove last.ckpt, otherwise PL auto-resumes from the previous
# fine-tune run and max_epochs is immediately "reached", skipping all training.
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion 2>&1 | tee -a "$LOG_FILE"
echo "End train: $(date)" | tee -a "$LOG_FILE"

# Step 7: Single-GPU eval
echo "===== Step 7: Eval =====" | tee -a "$LOG_FILE"
echo "Start eval: $(date)" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_fusion_eval 2>&1 | tee -a "$LOG_FILE"
echo "End eval: $(date)" | tee -a "$LOG_FILE"
echo "===== ALL DONE =====" | tee -a "$LOG_FILE"
