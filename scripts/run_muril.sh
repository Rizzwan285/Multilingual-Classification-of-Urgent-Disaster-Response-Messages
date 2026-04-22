#!/bin/bash
#SBATCH --job-name=Train_MuRIL
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=nlp_logs/train_muril_%j.out
#SBATCH --error=nlp_logs/train_muril_%j.err

mkdir -p nlp_logs
source ~/.bashrc
conda activate nlp_master

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python -u src/train_muril.py