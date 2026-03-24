#!/bin/bash
set -e

LOG_DIR="/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "===== Ablation: SFT (from scratch, no pretrain) ====="
echo "Start: $(date)"
echo ""
echo "  run_id: SFT_small_1337 (no pretrain checkpoint to load)"
echo "  All other settings identical to FFT run"
echo ""

cd "$WORK_DIR"

echo "[Step 1] Clean stale last.ckpt..."
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
rm -f "${CKPT_DIR}/last-v1.ckpt" 2>/dev/null || true

echo "[Step 2] Starting SFT training..."
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_SFT

echo ""
echo "End: $(date)"
echo "===== DONE ====="
