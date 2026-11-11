#!/bin/bash
#SBATCH --job-name=stccc_sel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/sel_%a.out
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u select_kl.py --job $SLURM_ARRAY_TASK_ID
