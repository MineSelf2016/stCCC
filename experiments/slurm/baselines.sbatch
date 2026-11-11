#!/bin/bash
#SBATCH --job-name=stccc_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/base_%a.out
# Baselines x 12 sections, 3 training seeds each, same read-out as stCCC (run_baselines.py).
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
METHODS=(graphst stagate nichecompass)
i=$SLURM_ARRAY_TASK_ID; M=${METHODS[$((i / 12))]}; S=sample_${SECTIONS[$((i % 12))]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/${M}_env/bin/python -u run_baselines.py --method $M --samples $S --seeds 0 1 2
