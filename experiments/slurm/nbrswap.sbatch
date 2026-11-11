#!/bin/bash
#SBATCH --job-name=stccc_swap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-51
#SBATCH --output=logs/swap_%a.out
# Neighbourhood swap (nbr_swap.py) and gating weights (gate_map.py) on every section: tasks 0-11 DLPFC, 12-35 MERFISH, 36-51 HER2.
D=(); S=()
for s in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do D+=(dlpfc); S+=($s); done
for s in 24 25 28 29 34 35 40 41 46 47 52 53 58 59 64 65 68 69 76 77 84 85 88 89; do D+=(abca1); S+=($s); done
for s in 11 21 31 41 51 52 53 61 62 63 71 72 73 81 82 83; do D+=(her2); S+=($s); done
export STCCC_DATASET=${D[$SLURM_ARRAY_TASK_ID]}; X=sample_${S[$SLURM_ARRAY_TASK_ID]}
E=${CONDA_ENVS:-$HOME/miniconda3/envs}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
step() { echo "== $(date +%T) $*"; "$@" || echo "!! FAILED: $*"; }
step $E/spaVAE_env/bin/python -u gate_map.py --samples $X
step $E/stccc_magic/bin/python -u nbr_swap.py --variant magic_sp --samples $X
for V in model_xnbr model_xall; do step $E/spaVAE_env/bin/python -u nbr_swap.py --variant $V --samples $X; done
for V in magic_sp model_xnbr model_xall; do step $E/spaVAE_env/bin/python -u iface_cond.py --method $V --samples $X; done
echo "== $(date +%T) done"
