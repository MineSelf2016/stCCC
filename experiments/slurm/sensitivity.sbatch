#!/bin/bash
#SBATCH --job-name=stccc_sens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/sens_%a.out
# Sensitivity of domain recovery to lambda and gamma on the development donor (NA-fixed K).
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u run_dlpfc.py --samples sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]} --conds lr --seeds 0 1 --lams 10 100 1000 --gammas 1 3 --tag sensitivity
