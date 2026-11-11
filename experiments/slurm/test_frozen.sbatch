#!/bin/bash
#SBATCH --job-name=stccc_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/test_%a.out
# Test donors Br5595 + Br8100. Hyper-parameters frozen from the dev sweep: lambda=10, gamma=3
# (best `lr` configuration on Br5292). Run once; do not re-tune after looking at these numbers.
SECTIONS=(151669 151670 151671 151672 151673 151674 151675 151676)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
PY=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
$PY -u run_dlpfc.py --samples $S --conds noCCC attn --seeds 0 1 2 --tag test
$PY -u run_dlpfc.py --samples $S --conds lr randlr lr_raw depth --seeds 0 1 2 --lams 10 --gammas 3 --tag test
