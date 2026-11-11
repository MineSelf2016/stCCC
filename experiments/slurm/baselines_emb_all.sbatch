#!/bin/bash
#SBATCH --job-name=stccc_bemb2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/bemb2_%a.out
# Re-run STAGATE and NicheCompass on all 12 sections x 3 seeds only to save their embeddings, for the mclust read-out
# (mclust_readout.R). The 151673 seed-0 embeddings of the domain figure are kept in results/baseline_emb_fig_backup/.
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
METHODS=(stagate nichecompass); i=$SLURM_ARRAY_TASK_ID; M=${METHODS[$((i / 12))]}; S=sample_${SECTIONS[$((i % 12))]}
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
${CONDA_ENVS:-$HOME/miniconda3/envs}/${M}_env/bin/python -u run_baselines.py --method $M --samples $S --seeds 0 1 2 --out baselines_emb_all.jsonl
