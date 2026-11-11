#!/bin/bash
#SBATCH --job-name=stccc_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-35
#SBATCH --output=logs/gate_%a.out
# Gating ablation on both datasets: noGate = GP-VAE alone (no gating, attention or communication term),
# lr_noGate = full stCCC without the spatial gating unit. Per section: reliability (halves, 25% thinning,
# full), full-data denoised rate + interface scan; on DLPFC also domain recovery (3 seeds, frozen lam=10, gamma=3).
ALL=(dlpfc:151507 dlpfc:151508 dlpfc:151509 dlpfc:151510 dlpfc:151669 dlpfc:151670 dlpfc:151671 dlpfc:151672 dlpfc:151673 dlpfc:151674 dlpfc:151675 dlpfc:151676
     abca1:24 abca1:25 abca1:28 abca1:29 abca1:34 abca1:35 abca1:40 abca1:41 abca1:46 abca1:47 abca1:52 abca1:53
     abca1:58 abca1:59 abca1:64 abca1:65 abca1:68 abca1:69 abca1:76 abca1:77 abca1:84 abca1:85 abca1:88 abca1:89)
X=${ALL[$SLURM_ARRAY_TASK_ID]}; export STCCC_DATASET=${X%%:*}; S=sample_${X##*:}
P=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
step() { echo "== $(date +%T) $*"; "$@" || echo "!! FAILED: $*"; }
for C in noGate lr_noGate; do
  step $P -u ccc_reliability.py --cond $C --samples $S
  step $P -u denoise_baselines.py --save_full --method $C --samples $S
  step $P -u iface_cond.py --method $C --samples $S
done
if [ "$STCCC_DATASET" = dlpfc ]; then
  step $P -u run_dlpfc.py --samples $S --conds noGate lr_noGate --seeds 0 1 2 --lams 10 --gammas 3 --tag ablation_gate
fi
echo "== $(date +%T) done"
