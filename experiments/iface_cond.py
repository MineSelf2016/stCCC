"""Interface scan of one saved denoised input (results[/abca1]/denoised_<method>/<sample>.npy, from
denoise_baselines.py --save_full). Writes interface_ablation/ for noCCC (as iface_ablation.sbatch did) and
interface_<method>/ otherwise, so interface_extra/ is never overwritten.
Usage: [STCCC_DATASET=abca1] python iface_cond.py --method noGate --samples sample_151507 ..."""
import os, argparse
import numpy as np, scanpy as sc
from interface_scan import scan, DATA, RES

ap = argparse.ArgumentParser(); ap.add_argument('--method', required=True); ap.add_argument('--samples', nargs='+', required=True)
a = ap.parse_args()
od = f'{RES}/interface_ablation' if a.method == 'noCCC' else f'{RES}/interface_{a.method}'
os.makedirs(od, exist_ok=True)
for S in a.samples:
    ad = sc.read_h5ad(f'{DATA}/{S}_with_graph.h5ad')
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    lab = np.array([str(v) for v in np.load(f'{DATA}/{S}_labels.npy', allow_pickle=True)])
    df, _ = scan(S, {a.method: np.load(f'{RES}/denoised_{a.method}/{S}.npy').astype('float64')}, raw, list(ad.var_names), lab,
                 np.asarray(ad.obsm['spatial'], 'float64'))
    df.to_csv(f'{od}/{S}.csv', index=False); print(S, a.method, len(df), flush=True)
