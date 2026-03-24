#!/bin/bash
set -e

WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"

echo "===== Building Dementia Indexed Dataset ====="
echo "Start: $(date)"
echo "(This runs on CPU only, in parallel with fine-tuning)"
echo ""

cd "$WORK_DIR"
$PYTHON build_dementia_finetune_dataset.py

echo ""
echo "End: $(date)"
echo "===== BUILD DONE ====="
