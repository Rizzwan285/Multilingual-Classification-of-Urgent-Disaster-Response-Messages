#!/bin/bash
#SBATCH --job-name=train_base_models
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=nlp_logs/train_base_%j.out
#SBATCH --error=nlp_logs/train_base_%j.err

mkdir -p nlp_logs

source ~/.bashrc
conda activate nlp_master

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# --- ADD THESE TWO LINES ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# ---------------------------

export CUDA_VISIBLE_DEVICES=0

python -u src/train_transformers.py --model_name local_muril
python -u src/train_transformers.py --model_name local_indic_bert
python -u src/train_transformers.py --model_name local_mbert