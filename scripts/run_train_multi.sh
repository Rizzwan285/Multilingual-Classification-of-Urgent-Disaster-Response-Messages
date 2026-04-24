#!/bin/bash
#SBATCH --job-name=train_multi_en_ml
#SBATCH --partition=gpu01,gpu02,gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=nlp_logs/train_multi_%j.out
#SBATCH --error=nlp_logs/train_multi_%j.err

mkdir -p nlp_logs

# Activate your isolated environment
source ~/.bashrc
conda activate nlp_master

# Stability blocks
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Generate a unique timestamp for this entire experiment
RUN_ID=$(date +"%Y%m%d_%H%M%S")

echo "========================================================="
echo "Starting Mixed Translation Training Job | RUN_ID: $RUN_ID"
echo "========================================================="

# 1. Train MuRIL using the standard script
echo -e "\n---> Training MuRIL"
python -u src/train_transformers_multi.py --model_name local_muril --run_id $RUN_ID

# 2. Train IndicBERT using the standard script (with use_fast=False)
echo -e "\n---> Training IndicBERT"
python -u src/train_transformers_multi.py --model_name local_indic_bert --run_id $RUN_ID

# 3. Train mBERT using the standard script
echo -e "\n---> Training mBERT"
python -u src/train_transformers_multi.py --model_name local_mbert --run_id $RUN_ID

# 4. Train XLM-RoBERTa using the dedicated XLM-R script
echo -e "\n---> Training XLM-RoBERTa"
python -u src/train_xlmr_multi.py --model_name local_xlm_roberta --run_id $RUN_ID

echo -e "\n========================================================="
echo "All 4 mixed-language models have finished training!"
echo "========================================================="