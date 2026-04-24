#!/bin/bash
#SBATCH --job-name=eval_all_models
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=nlp_logs/evaluation_%j.out
#SBATCH --error=nlp_logs/evaluation_%j.err

mkdir -p nlp_logs

source ~/.bashrc
conda activate nlp_master

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "Starting massive evaluation script across all ML and Transformer models..."

python -u src/evaluate_all_models.py
python -u src/evaluate_with_malayalam.py


echo "Evaluation finished successfully!"