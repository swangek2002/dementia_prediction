#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "===== Dementia Competing-Risk Fine-Tuning ====="
echo "Start: $(date)"
echo ""
echo "  experiment.type: fine-tune (competing-risk)"
echo "  num_risks: 2 (dementia=k1, death=k2)"
echo "  batch_size=32 x accum=4 x 4GPU = 512 effective"
echo "  block_size=512, backbone_lr=5e-5, head_lr=5e-4"
echo ""

cd "$WORK_DIR"

echo "[Step 1] Cleaning old CR checkpoints..."
rm -f "${CKPT_DIR}/crPreTrain_small_1337_FineTune_Dementia_CR.ckpt" 2>/dev/null || true
mv "${CKPT_DIR}/last.ckpt" "${CKPT_DIR}/last_backup_before_cr.ckpt" 2>/dev/null || true
mv "${CKPT_DIR}/last-v1.ckpt" "${CKPT_DIR}/last-v1_backup_before_cr.ckpt" 2>/dev/null || true

echo "[Step 2] Starting competing-risk fine-tuning..."
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR

echo ""
echo "End: $(date)"
echo "===== DONE ====="
