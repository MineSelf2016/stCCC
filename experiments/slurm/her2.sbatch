#!/bin/bash
#SBATCH --job-name=stccc_her2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/her2_%a.out
# Full pipeline on the HER2 breast-cancer ST sections (STCCC_DATASET=her2; data from prep_her2.py), one section per task:
# stCCC reliability + interface scan, denoise-then-score baselines, LIANA+, COMMOT, and the model ablations
# (noCCC, noGate, lr_noGate: reliability + interface scan).
SECTIONS=(11 21 31 41 51 52 53 61 62 63 71 72 73 81 82 83)
S=sample_${SECTIONS[$SLURM_ARRAY_TASK_ID]}
export STCCC_DATASET=her2
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
step $E/stccc_magic/bin/python -u commot_reliability.py --samples $S
for C in noCCC noGate lr_noGate; do
  step $E/spaVAE_env/bin/python -u ccc_reliability.py --cond $C --samples $S
  step $E/spaVAE_env/bin/python -u denoise_baselines.py --save_full --method $C --samples $S
  step $E/spaVAE_env/bin/python -u iface_cond.py --method $C --samples $S
done
echo "== $(date +%T) done"
