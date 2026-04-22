#!/bin/bash
set -e

PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
CKPT_DIR="/Data0/swangek_data/991/CPRD/output/checkpoints"

cd "$WORK_DIR"

echo "============================================================"
echo "  HES Backbone Pretrain Pipeline"
echo "  Start: $(date)"
echo "============================================================"

# Step 1: Build HES database
echo "===== Step 1: Build HES database ====="
$PYTHON build_hes_database.py

# Step 2: Build HES pretrain dataset
echo "===== Step 2: Build HES pretrain dataset ====="
$PYTHON build_hes_pretrain_dataset.py

# Step 3: Pretrain HES backbone
echo "===== Step 3: Pretrain HES backbone ====="
rm -f "${CKPT_DIR}/last.ckpt" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
$PYTHON run_experiment.py --config-name=config_HES_Pretrain

echo "===== HES PRETRAIN DONE ====="
echo "Checkpoint: ${CKPT_DIR}/crPreTrain_HES_1337.ckpt"
echo "End: $(date)"
