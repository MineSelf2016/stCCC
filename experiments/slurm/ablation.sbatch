#!/bin/bash
#SBATCH --job-name=stccc_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/abl_%a.out
# Ablations (NA-fixed K) of the frozen configuration on all 12 sections; also saves latent means for domain maps.
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u run_dlpfc.py --samples sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]} --conds lr noCCC attn randlr lr_raw depth --seeds 0 1 2 --lams 10 --gammas 3 --tag ablation --save_latent
