#!/bin/bash
#SBATCH --job-name=stccc_repr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/repr_%a.out
# Domain recovery of the frozen configuration on all 12 sections, K = number of annotated layers ('NA' excluded).
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u run_dlpfc.py --samples sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]} --conds lr --seeds 0 1 2 --lams 10 --gammas 3 --tag repr
