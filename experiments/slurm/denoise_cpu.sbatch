#!/bin/bash
# usage: sbatch --export=METHOD=scvi denoise.sbatch   (scvi on GPU; smooth_r / magic: add -p general --gres=none)
#SBATCH --job-name=stccc_dn
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/dn_%x_%a.out
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
PY=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
[ "$METHOD" = magic ] && PY=${CONDA_ENVS:-$HOME/miniconda3/envs}/stccc_magic/bin/python
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
$PY -u denoise_baselines.py --method $METHOD --samples sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
