#!/bin/bash
#SBATCH --job-name=stccc_relabl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/relabl_%a.out
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
CONDS=(noCCC attn)
i=$SLURM_ARRAY_TASK_ID
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u ccc_reliability.py --samples sample_${SECTIONS[$((i % 12))]} --cond ${CONDS[$((i / 12))]}
