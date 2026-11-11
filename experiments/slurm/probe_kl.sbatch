#!/bin/bash
#SBATCH --job-name=stccc_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/probe_%a.out
T=(42 18 24); C=(kl0.025 kl0.25 kl1)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u probe_kl.py --task ${T[$((SLURM_ARRAY_TASK_ID / 3))]} --conf ${C[$((SLURM_ARRAY_TASK_ID % 3))]}
