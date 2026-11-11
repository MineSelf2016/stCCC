#!/bin/bash
# interface scan of the stCCC ablation without attention and communication term (noCCC), both datasets
#SBATCH --job-name=stccc_ifab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/ifab_%a.out
ALL=(dlpfc:151507 dlpfc:151508 dlpfc:151509 dlpfc:151510 dlpfc:151669 dlpfc:151670 dlpfc:151671 dlpfc:151672 dlpfc:151673 dlpfc:151674 dlpfc:151675 dlpfc:151676 abca1:40 abca1:41 abca1:52 abca1:53 abca1:64 abca1:65 abca1:76 abca1:77)
X=${ALL[$SLURM_ARRAY_TASK_ID]}; export STCCC_DATASET=${X%%:*}; S=sample_${X##*:}
P=${CONDA_ENVS:-$HOME/miniconda3/envs}/spaVAE_env/bin/python
cd ${STCCC_REPO:?set STCCC_REPO to the repository root}/experiments
set -e
$P -u denoise_baselines.py --save_full --method noCCC --samples $S
$P -u - <<PY
import os, sys; sys.argv=['x']
import numpy as np
from interface_scan import scan, DATA, RES
import scanpy as sc
S='$S'
ad=sc.read_h5ad(f'{DATA}/{S}_with_graph.h5ad'); raw=np.rint(ad.X.toarray() if hasattr(ad.X,'toarray') else np.asarray(ad.X)).astype('float64')
lab=np.array([str(v) for v in np.load(f'{DATA}/{S}_labels.npy', allow_pickle=True)])
df,_=scan(S, dict(noCCC=np.load(f'{RES}/denoised_noCCC/{S}.npy').astype('float64')), raw, list(ad.var_names), lab, np.asarray(ad.obsm['spatial'],'float64'))
os.makedirs(f'{RES}/interface_ablation', exist_ok=True); df.to_csv(f'{RES}/interface_ablation/{S}.csv', index=False); print(S, len(df))
PY
