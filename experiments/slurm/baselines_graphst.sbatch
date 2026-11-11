#!/bin/bash
#SBATCH --job-name=stccc_gst
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/gst_%a.out
# GraphST rerun with its own PCA-20 embedding (run_baselines.py); one task per section x seed.
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
i=$SLURM_ARRAY_TASK_ID; S=sample_${SECTIONS[$((i % 12))]}; SD=$((i / 12))
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/graphst_env/bin/python -u run_baselines.py --method graphst --samples $S --seeds $SD
