#!/bin/bash
#SBATCH --job-name=translate_data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=nlp_logs/translate_%j.out
#SBATCH --error=nlp_logs/translate_%j.err

source ~/.bashrc
conda activate nlp_master

python -u src/generate_mixed_dataset.py