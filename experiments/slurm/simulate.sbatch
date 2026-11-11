#!/bin/bash
# semi-synthetic benchmark: 24 tasks (planted effect x depth confound x seed), see simulate.py
#SBATCH --job-name=stccc_sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=48-71
#SBATCH --output=logs/sim_%a.out
E=${CONDA_ENVS:-$HOME/miniconda3/envs}
T=$SLURM_ARRAY_TASK_ID
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
set -e
$E/spaVAE_env/bin/python -u simulate.py --task $T --stage counts_model
$E/stccc_magic/bin/python -u simulate.py --task $T --stage magic
$E/spaVAE_env/bin/python -u simulate.py --task $T --stage scan
echo done
