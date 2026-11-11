#!/bin/bash
#SBATCH --job-name=stccc_bemb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/bemb_%a.out
# Re-run STAGATE and NicheCompass on 151673, seed 0, only to save embeddings for the domain-map figure.
METHODS=(stagate nichecompass); M=${METHODS[$SLURM_ARRAY_TASK_ID]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/${M}_env/bin/python -u run_baselines.py --method $M --samples sample_151673 --seeds 0 --out baselines_emb_rerun.jsonl
