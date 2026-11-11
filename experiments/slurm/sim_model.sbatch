#!/bin/bash
# retrain stCCC on all 72 simulated datasets and rescan every input
#SBATCH --job-name=stccc_simm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/simm_%a.out
P=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
set -e
$P -u simulate.py --task $SLURM_ARRAY_TASK_ID --stage model
$P -u simulate.py --task $SLURM_ARRAY_TASK_ID --stage scan
