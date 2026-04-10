#!/bin/bash
export PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD
cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR/
PYTHON=/Data0/swangek_data/conda_envs/survivehr/bin/python
LOG=/Data0/swangek_data/991/CPRD/finetune_cr_hes_aug_25ep_log.txt

echo "===== Resume training from epoch 15 to 25 =====" | tee $LOG
echo "Start: $(date)" | tee -a $LOG

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug 2>&1 | tee -a $LOG

echo "===== Eval (1-GPU) =====" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 $PYTHON run_experiment.py --config-name=config_FineTune_Dementia_CR_hes_aug_eval 2>&1 | tee -a $LOG

echo "===== DONE =====" | tee -a $LOG