#!/bin/bash
#SBATCH --job-name=train_ml_baselines
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00  # <-- Changed to 1 hour (2 hours is overkill for ML models)
#SBATCH --output=nlp_logs/ml_baselines_%j.out
#SBATCH --error=nlp_logs/ml_baselines_%j.err

# Ensure we are in the project root directory
cd ~/rizwan/NLP_Project  # <-- ADD THIS LINE just to be safe!

mkdir -p nlp_logs

source ~/.bashrc
conda activate nlp_master

export PYTHONNOUSERSITE=1

python -u src/train_ml_baselines.py