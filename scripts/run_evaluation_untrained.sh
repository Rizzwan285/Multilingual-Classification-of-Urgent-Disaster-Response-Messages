#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=nlp_logs/eval_baselines_%j.out
#SBATCH --error=nlp_logs/eval_baselines_%j.err

# Create logs directory if it doesn't exist
mkdir -p nlp_logs

# Setup Environment
source ~/.bashrc
conda activate nlp_master

# CRITICAL OFFLINE FLAGS to prevent network hanging on the compute node
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "Starting Evaluation of Untrained Baselines and ML Models..."

python -u src/evaluate_untrained_baselines.py

echo "Evaluation Script Finished successfully!"