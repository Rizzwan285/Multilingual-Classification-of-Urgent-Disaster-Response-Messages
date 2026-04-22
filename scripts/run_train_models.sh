#!/bin/bash
#SBATCH --job-name=train_all_transformers
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=nlp_logs/train_all_%j.out
#SBATCH --error=nlp_logs/train_all_%j.err

mkdir -p nlp_logs

source ~/.bashrc
conda activate nlp_master

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

python -u src/train_transformers.py --model_name local_muril
python -u src/train_transformers.py --model_name local_indic_bert
python -u src/train_transformers.py --model_name local_mbert
python -u src/train_transformers.py --model_name local_xlm_roberta