#!/bin/bash
# usage: sbatch --export=METHOD=scvi denoise.sbatch   (scvi on GPU; smooth_r / magic: add -p general --gres=none)
#SBATCH --job-name=stccc_commot
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/commot_%a.out
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/stccc_magic/bin/python -u commot_reliability.py --samples sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
