#!/bin/bash
set -e

WORK_DIR="/Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR"
PYTHON="/Data0/swangek_data/conda_envs/survivehr/bin/python"
export PYTHONPATH="/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD"
export CUDA_VISIBLE_DEVICES=0

echo "===== Dementia CR — Test-Only Evaluation (Single GPU) ====="
echo "Start: $(date)"
echo ""
echo "  Using best checkpoint: crPreTrain_small_1337_FineTune_Dementia_CR.ckpt"
echo "  train=False, test=True, devices=1"
echo "  log_individual=True → dementia CIF + death CIF metrics separately"
echo ""

cd "$WORK_DIR"
$PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_eval

echo ""
echo "End: $(date)"
echo "===== DONE ====="
