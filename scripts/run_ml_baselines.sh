#!/bin/bash
#SBATCH --job-name=train_ml_baselines
#SBATCH --partition=gpu03        # <-- Changed to gpu03
#SBATCH --gres=gpu:1             # <-- Add this if your cluster strictly requires a GPU request on GPU partitions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00          
#SBATCH --output=nlp_logs/ml_baselines_%j.out
#SBATCH --error=nlp_logs/ml_baselines_%j.err

cd ~/rizwan/NLP_Project

mkdir -p nlp_logs

source ~/.bashrc
conda activate nlp_master

export PYTHONNOUSERSITE=1

python -u src/train_ml_baselines.py