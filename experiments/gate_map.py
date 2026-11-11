"""Spatial gating weights omega^s of the full stCCC model (frozen configuration, seed 0, full counts).
Trains the model exactly as ccc_reliability.train_denoise and saves, per section, omega^s of every spot with its
coordinates, label and log library size, plus the agreement of the decoder rate with the stored results/denoised rate
(a check that this run reproduces the model used throughout).
Writes RESULTS/gate/<sample>.npz.  Usage: STCCC_DATASET=<d> python gate_map.py --samples sample_X ..."""
import os, sys, argparse, warnings
import numpy as np, torch
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rel_common import load
from ccc_reliability import train_denoise
from datasets import RESULTS

ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='+', required=True); a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
out = os.path.join(RESULTS, 'gate'); os.makedirs(out, exist_ok=True)
for s in a.samples:
    ad, loc, raw, genes, labels = load(s)
    from ccc_reliability import prep
    TL0 = torch.as_tensor(loc, dtype=torch.float64, device=dev); TX0 = torch.as_tensor(prep(raw)[0], dtype=torch.float64, device=dev)
    traj = []                                           # per epoch: mean omega^s and spatial share of residual-mean variance

    def cb(model, ep):
        e, sp = model.gate_terms(TL0, TX0); model.train()
        traj.append((ep + 1, float(model.gate_weights(TL0, TX0).mean()), float(sp.var(0).sum() / (e.var(0).sum() + sp.var(0).sum()))))
        model.train()
    rate, m, TL, TX = train_denoise(raw, genes, loc, dev, return_model=True, callback=cb)
    w = m.gate_weights(TL, TX)
    te, ts = m.gate_terms(TL, TX)                      # expression and spatial terms of the residual mean
    var_e, var_s = te.var(0).sum(), ts.var(0).sum()    # variance across spots contributed by each term
    share_s = float(var_s / (var_e + var_s)); norm_ratio = np.linalg.norm(ts, axis=1) / (np.linalg.norm(te, axis=1) + 1e-12)
    ref = os.path.join(RESULTS, 'denoised', f'{s}.npy'); agree = np.nan
    if os.path.exists(ref):
        R = np.load(ref).astype('float64'); agree = float(spearmanr(np.log1p(rate).ravel()[::97], np.log1p(R).ravel()[::97])[0])
    np.savez(os.path.join(out, f'{s}.npz'), omega_s=w, xy=np.asarray(ad.obsm['spatial'], 'float64'),
             label=np.array([str(x) for x in labels]), loglib=np.log(raw.sum(1) + 1.0), agree_ref=agree,
             term_expr=te, term_spatial=ts, share_spatial_var=share_s, norm_ratio=norm_ratio, trajectory=np.array(traj))
    print(s, 'omega_s median %.3f IQR %.3f-%.3f' % tuple(np.percentile(w, [50, 25, 75])), 'agree with stored rate %.3f' % agree,
          'spatial share of residual-mean variance %.3f, median |spatial|/|expr| %.3f' % (share_s, np.median(norm_ratio)), flush=True)
