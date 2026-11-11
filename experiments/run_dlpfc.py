"""Train stCCC on one DLPFC section under one condition and score domain recovery.
Protocol (fixed, identical for every condition): default architecture, KMeans(K = #annotated layers)
on the raw latent, NMI/ARI averaged over 5 KMeans seeds. One JSON line per run in results/<tag>.jsonl.
Conditions: noCCC (no attention, no loss) | attn (neighbour attention, no communication) |
            lr | randlr | lr_raw | depth  (attention + communication score + L_CCC) |
            noGate (GP-VAE alone: no gating, attention or communication) | lr_noGate (lr without the gating unit)."""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd, torch, scanpy as sc
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from preprocess import normalize
from ccc_graph import build_scores
from stccc import stCCC
sys.path.insert(0, HERE)
from datasets import KL
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')

def run(sample, cond, seed, lam, gamma, k, maxiter, device, save_latent=False):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    loc = MinMaxScaler().fit_transform(ad.obsm['spatial'].astype('float64')) * 20.
    adata = sc.AnnData(ad.X.astype('float64'), dtype="float64"); adata.var_names = ad.var_names
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)
    keep = adata.obs_names.astype(int) if adata.n_obs != ad.n_obs else np.arange(ad.n_obs)
    loc = loc[keep]
    raw = np.asarray(adata.raw.X.todense() if hasattr(adata.raw.X, 'todense') else adata.raw.X, 'float64')
    genes = list(adata.var_names)
    score_cond = 'lr' if cond == 'lr_noGate' else (cond if cond in ('lr', 'randlr', 'lr_raw', 'depth') else 'none')
    nbr, dist, S, info = build_scores(raw, genes, loc, score_cond, DB, k=k, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    ip = np.mgrid[0:(1 + 1e-5):(1. / 8), 0:(1 + 1e-5):(1. / 8)].reshape(2, -1).T * 20.
    m = stCCC(input_dim=adata.n_vars, GP_dim=2, Normal_dim=8, encoder_layers=[128, 64], decoder_layers=[128],
                   inducing_points=ip, kernel_scale=20., N_train=adata.n_obs, nbr=nbr, nbr_dist=dist, score=S,
                   sigma_sp=float(np.median(dist)), use_attention=(cond not in ('noCCC', 'noGate')), gamma=gamma, use_gating=('noGate' not in cond),
                   lam=(lam if S is not None else 0.0), device=device, **KL)
    t0 = time.time()
    m.fit(loc, adata.X, raw, adata.obs.size_factors.values, maxiter=maxiter)
    z = m.latent(torch.as_tensor(loc, device=device), torch.as_tensor(np.asarray(adata.X), device=device))
    y = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)[keep]
    if save_latent:                                   # latent mean + kept spot indices, for domain maps
        d = os.path.join(HERE, '..', 'results', 'latent'); os.makedirs(d, exist_ok=True)
        np.savez(f'{d}/{sample}_{cond}_seed{seed}.npz', z=np.asarray(z, 'float32'), keep=keep)
    ok = ~pd.isnull(y) & (y.astype(str) != 'NA'); K = len(set(y[ok]))   # 'NA' = unannotated spots, not a layer
    nm, ar = [], []
    for cs in range(5):
        p = KMeans(K, n_init=10, random_state=cs).fit_predict(z)
        nm.append(NMI(y[ok], p[ok])); ar.append(ARI(y[ok], p[ok]))
    h = m.history[-1]
    return dict(sample=sample, cond=cond, seed=seed, lam=lam, gamma=gamma, k=k, NMI=float(np.mean(nm)), ARI=float(np.mean(ar)),
                epochs=len(m.history), secs=round(time.time() - t0, 1), recon=h['recon'], lamLccc=lam * h['L_ccc'],
                L_ccc_first=m.history[0]['L_ccc'], L_ccc_last=h['L_ccc'], **info)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--conds', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); ap.add_argument('--lams', nargs='+', type=float, default=[10.])
    ap.add_argument('--gammas', nargs='+', type=float, default=[1.]); ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--maxiter', type=int, default=100); ap.add_argument('--tag', required=True)
    ap.add_argument('--save_latent', action='store_true')
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = open(os.path.join(HERE, '..', 'results', f'{a.tag}.jsonl'), 'a')
    for s in a.samples:
        for c in a.conds:
            for lam in (a.lams if c not in ('noCCC', 'attn', 'noGate') else [0.0]):
                for g in (a.gammas if c not in ('noCCC', 'attn', 'noGate') else [0.0]):
                    for sd in a.seeds:
                        r = run(s, c, sd, lam, g, a.k, a.maxiter, dev, a.save_latent)
                        print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
