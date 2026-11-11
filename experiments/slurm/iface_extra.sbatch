#!/bin/bash
# denoise-then-score baselines in the interface scan: save full-data matrices, then scan them
#SBATCH --job-name=stccc_ifx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/ifx_%a.out
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
E=${CONDA_ENVS:-$HOME/miniconda3/envs}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
$E/spaVAE_env/bin/python -u denoise_baselines.py --save_full --method smooth_r --samples $S
$E/spaVAE_env/bin/python -u denoise_baselines.py --save_full --method scvi --samples $S
$E/stccc_magic/bin/python -u denoise_baselines.py --save_full --method magic --samples $S
$E/spaVAE_env/bin/python -u interface_scan.py --extra smooth_r scvi magic --samples $S
