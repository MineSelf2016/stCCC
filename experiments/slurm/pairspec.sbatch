#!/bin/bash
#SBATCH --job-name=stccc_pairspec
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pairspec_%a.out
# Partner-swap interface scan (iface_pairspec.py), one section per task: tasks 0-11 DLPFC, 12-35 MERFISH, 36-51 HER2.
D=(); S=()
for s in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do D+=(dlpfc); S+=($s); done
for s in 24 25 28 29 34 35 40 41 46 47 52 53 58 59 64 65 68 69 76 77 84 85 88 89; do D+=(abca1); S+=($s); done
for s in 11 21 31 41 51 52 53 61 62 63 71 72 73 81 82 83; do D+=(her2); S+=($s); done
export STCCC_DATASET=${D[$SLURM_ARRAY_TASK_ID]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python -u iface_pairspec.py --samples sample_${S[$SLURM_ARRAY_TASK_ID]}
