#!/bin/bash
#SBATCH --job-name=train_xlmr
#SBATCH --partition=gpu01         # Route to GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=nlp_logs/xlmr_%j.out
#SBATCH --error=nlp_logs/xlmr_%j.err

# 1. Force execution from the absolute project root
cd ~/rizwan/NLP_Project

# 2. Setup logs directory safely
mkdir -p nlp_logs

# 3. Initialize environment
source ~/.bashrc
conda activate nlp_master
export PYTHONNOUSERSITE=1

# 4. Run the script
python -u src/train_xlm_roberta.py