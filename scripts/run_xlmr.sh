#!/bin/bash
#SBATCH --job-name=Train_XLM_Roberta
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=nlp_logs/train_xlmr_%j.out
#SBATCH --error=nlp_logs/train_xlmr_%j.err

mkdir -p nlp_logs
source ~/.bashrc
conda activate nlp_master

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

python -u src/train_xlm_roberta.py --model_name local_xlm_roberta