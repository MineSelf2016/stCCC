#!/bin/bash
#SBATCH --job-name=stccc_dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/dev_%a.out
# Development donor Br5292 only: hyper-parameters are chosen here, then frozen for the 8 test sections.
SECTIONS=(151507 151508 151509 151510)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
PY=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
$PY -u run_dlpfc.py --samples $S --conds noCCC attn --seeds 0 1 --tag dev
$PY -u run_dlpfc.py --samples $S --conds lr randlr depth --seeds 0 1 --lams 10 100 1000 --gammas 1 3 --tag dev
