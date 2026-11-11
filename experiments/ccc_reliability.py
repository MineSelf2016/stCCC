"""Reliability of spatial LR communication scores: model-denoised vs. conventional inputs.

For one DLPFC section, builds four count versions:
  full  : observed counts
  A, B  : independent binomial halves (p = 0.5), so A + B = full
  d25   : 25% binomial downsample
stCCC (frozen dev configuration: lr, lambda = 10, gamma = 3, seed 0) is trained on each version
(--cond attn / noCCC: ablations without the communication term / without neighbour attention; results_<cond>/;
 --cond noGate: GP-VAE alone, also without the spatial gating unit; lr_noGate: full model without the gating unit)
and its decoder rate is used as denoised expression.

Edge-level LR scores on the spatial kNN graph are computed per LR pair from
  counts  : raw counts                                   (no normalisation)
  norm    : log1p counts-per-10k                         (CellChat / COMMOT-style input)
  smooth  : norm averaged over the spot and its k nbrs   (spatial-smoothing baseline)
  model   : log1p of the stCCC decoder rate, per-10k, then depth-residualised (the method)
  model_nr: same without residualisation                 (ablation)
  norm_r  : norm, depth-residualised                     (ablation)
Saved per section/version/method: per-pair layer means (P x L), per-edge total score, per-pair depth rho.
"""
import os, sys, pickle, argparse, warnings
import numpy as np, pandas as pd, torch, scanpy as sc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
from sklearn.preprocessing import MinMaxScaler
from ccc_graph import build_scores, knn_graph, lr_pairs
from stccc import stCCC
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')
from datasets import DATA, RESULTS, KL
OUT = os.path.join(RESULTS, 'reliability')


def prep(raw):
    lib = raw.sum(1)
    sf = np.maximum(lib, 1.0) / np.median(lib)
    X = np.log1p(raw / sf[:, None])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return X, sf


def train_denoise(raw, genes, loc, device, seed=0, cond='lr', return_model=False, callback=None):
    comm = cond in ('lr', 'lr_noGate')                  # noGate: GP-VAE alone (no gating, attention or communication)
    nbr, dist, S, _ = build_scores(raw, genes, loc, 'lr' if comm else 'none', DB, k=8, seed=seed)
    X, sf = prep(raw)
    torch.manual_seed(seed); np.random.seed(seed)
    ip = np.mgrid[0:(1 + 1e-5):(1. / 8), 0:(1 + 1e-5):(1. / 8)].reshape(2, -1).T * 20.
    m = stCCC(input_dim=raw.shape[1], GP_dim=2, Normal_dim=8, encoder_layers=[128, 64], decoder_layers=[128],
                   inducing_points=ip, kernel_scale=20., N_train=raw.shape[0], nbr=nbr, nbr_dist=dist, score=S,
                   sigma_sp=float(np.median(dist)), use_attention=(cond not in ('noCCC', 'noGate')), gamma=3.0, lam=(10.0 if comm else 0.0),
                   use_gating=('noGate' not in cond), device=device, **KL)
    m.fit(loc, X, raw, sf, maxiter=100, callback=callback)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    return (m.denoise(T(loc), T(X)), m, T(loc), T(X)) if return_model else m.denoise(T(loc), T(X))


def per10k(E):
    return np.log1p(E / (E.sum(1, keepdims=True) + 1e-12) * 1e4)


def resid(y, lib, nbr):
    ll = np.log(lib + 1.0)
    s = (ll[:, None] + ll[nbr]).ravel(); d = np.abs(ll[:, None] - ll[nbr]).ravel()
    Xd = np.stack([np.ones_like(s), s, d], 1)
    b, *_ = np.linalg.lstsq(Xd, y.ravel(), rcond=None)
    return (y.ravel() - Xd @ b).reshape(y.shape)


def summarise(E, genes, pairs, nbr, lib, labels, layers, residualise):
    gi = {g: i for i, g in enumerate(genes)}
    dep = np.log(lib + 1.0); dep = dep[:, None] + dep[nbr]
    ok = ~pd.isnull(labels)
    LM = np.full((len(pairs), len(layers)), np.nan); total = np.zeros(nbr.shape); rho = []
    for p, (l, r) in enumerate(pairs):
        L, R = E[:, gi[l]], E[:, gi[r]]
        y = np.log1p(L[:, None] * R[nbr] + L[nbr] * R[:, None])
        if residualise:
            y = resid(y, lib, nbr)
        total += y
        rho.append(spearmanr(y.ravel(), dep.ravel())[0] if y.std() > 0 else np.nan)
        ym = y.mean(1)
        for q, lay in enumerate(layers):
            sel = ok & (labels == lay)
            if sel.any():
                LM[p, q] = ym[sel].mean()
    return dict(LM=LM, total=total, rho=np.array(rho),
                rho_total=float(spearmanr(total.ravel(), dep.ravel())[0]))


def main(sample, device, cond='lr'):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    loc = MinMaxScaler().fit_transform(ad.obsm['spatial'].astype('float64')) * 20.
    raw = ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)
    raw = np.rint(raw).astype('float64')
    genes = list(ad.var_names)
    labels = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)
    labels = np.array([None if pd.isnull(x) or str(x) == 'NA' else str(x) for x in labels], dtype=object)   # 'NA' = unannotated
    layers = sorted({x for x in labels if x is not None})
    rng = np.random.default_rng(0)
    A = rng.binomial(raw.astype(np.int64), 0.5).astype('float64')
    versions = dict(full=raw, A=A, B=raw - A, d25=rng.binomial(raw.astype(np.int64), 0.25).astype('float64'))
    nbr, _ = knn_graph(loc, 8)
    pairs = lr_pairs(genes, DB)
    res = dict(sample=sample, pairs=pairs, layers=layers, nbr=nbr, out={})
    for v, Rv in versions.items():
        lib = Rv.sum(1)
        norm = per10k(Rv)
        smooth = (norm + norm[nbr].sum(1)) / (nbr.shape[1] + 1)
        mod = per10k(train_denoise(Rv, genes, loc, device, cond=cond))
        inputs = dict(model=(mod, True), model_nr=(mod, False))
        if cond == 'lr':
            inputs.update(counts=(Rv, False), norm=(norm, False), norm_r=(norm, True), smooth=(smooth, False))
        res['out'][v] = {k: summarise(E, genes, pairs, nbr, lib, labels, layers, r) for k, (E, r) in inputs.items()}
        print(sample, v, {k: round(o['rho_total'], 3) for k, o in res['out'][v].items()}, flush=True)
    out = OUT if cond == 'lr' else f'{OUT}_{cond}'
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{sample}.pkl'), 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--cond', default='lr', choices=['lr', 'attn', 'noCCC', 'noGate', 'lr_noGate'])
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for s in a.samples:
        main(s, dev, a.cond)
