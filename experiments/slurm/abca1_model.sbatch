#!/bin/bash
# stCCC-dependent part of the ABCA-1 pipeline (model + noCCC reliability, interface scan)
#SBATCH --job-name=stccc_abm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/abm_%a.out
SECTIONS=(40 41 52 53 64 65 76 77)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
export STCCC_DATASET=abca1
P=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
set -e
$P -u ccc_reliability.py --samples $S
$P -u interface_scan.py --samples $S
$P -u ccc_reliability.py --cond noCCC --samples $S
