#!/bin/bash
#SBATCH --job-name=stccc_abprep
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/abca1_prep.out
# prepare the 16 additional ABCA-1 sections (existing ones are skipped)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/mamba/bin/python -u prep_abca1.py
