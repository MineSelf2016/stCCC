#!/bin/bash
# full reliability + interface pipeline on the ABCA-1 MERFISH isocortex (STCCC_DATASET=abca1), one section per task
#SBATCH --job-name=stccc_abca1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/abca1_%a.out
SECTIONS=(40 41 52 53 64 65 76 77)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
export STCCC_DATASET=abca1
E=${CONDA_ENVS:-$HOME/miniconda3/envs}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
step() { echo "== $(date +%T) $*"; "$@" || echo "!! FAILED: $*"; }
step $E/spaVAE_env/bin/python -u ccc_reliability.py --samples $S
step $E/spaVAE_env/bin/python -u interface_scan.py --samples $S
for M in smooth_r scvi; do step $E/spaVAE_env/bin/python -u denoise_baselines.py --method $M --samples $S; done
step $E/stccc_magic/bin/python -u denoise_baselines.py --method magic --samples $S
for M in smooth_r scvi; do step $E/spaVAE_env/bin/python -u denoise_baselines.py --save_full --method $M --samples $S; done
step $E/stccc_magic/bin/python -u denoise_baselines.py --save_full --method magic --samples $S
step $E/spaVAE_env/bin/python -u interface_scan.py --extra smooth_r scvi magic --samples $S
step $E/mamba/bin/python -u liana_reliability.py --samples $S
step $E/spaVAE_env/bin/python -u ccc_reliability.py --cond noCCC --samples $S
step $E/stccc_magic/bin/python -u commot_reliability.py --samples $S
echo "== $(date +%T) done"
