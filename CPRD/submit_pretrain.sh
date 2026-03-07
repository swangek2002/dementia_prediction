#!/bin/bash
#SBATCH --job-name=survivehr_pretrain
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/swangek/991/CPRD/output/slurm_%j.log
#SBATCH --error=/home/swangek/991/CPRD/output/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=swangek@ad.unc.edu

module load anaconda/2024.02
module load cuda/11.8
conda activate survivehr
export PYTHONPATH="/home/swangek/991/CPRD:$PYTHONPATH"
cd /home/swangek/991/CPRD/examples/modelling/SurvivEHR
mkdir -p /home/swangek/991/CPRD/output/checkpoints
python run_experiment.py --config-name=config_CompetingRisk11M
