"""Laminar-interface LR scan on one DLPFC section, using stCCC's de-confounded communication scores.
Follows the AAAI-27 scan (Spatial-ccc-gb/experiments_gb/rebuttal_aaai27/interface_fdr.py): the same LR pairs
(both genes detected in >= 2% of spots), the same six laminar boundaries, the same 6-NN Visium edges and the
same expression-matched null (partners drawn from the same mean-expression deciles, 200 draws).

Edge score y_e = log1p(L_i R_j + L_j R_i), depth-residualised on [1, log lib_i + log lib_j, |log lib_i - log lib_j|].
Statistic per (pair, boundary) = mean y over boundary edges - mean y over within-layer edges of the two layers.
Inputs:
  model : log1p per-10k stCCC decoder rate (frozen dev configuration), saved to results/denoised/
  norm  : log1p counts-per-10k (the input of the submitted scan)
  --extra smooth_r scvi magic : denoise-then-score baselines instead (log1p per-10k matrices saved by
               denoise_baselines.py --save_full in results/denoised_<name>/), written to results/interface_extra/
Writes results/interface/<sample>.csv."""
import os, sys, argparse, warnings
import numpy as np, pandas as pd, torch, scanpy as sc
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccc_reliability import train_denoise, per10k, DATA, DB
NNULL, FRAC = 200, 0.02
from datasets import BOUNDS, SCAN_K, RESULTS as RES


def main(S, device, extra=()):
    ad = sc.read_h5ad(f"{DATA}/{S}_with_graph.h5ad")
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    genes = list(ad.var_names); lib = raw.sum(1)
    lab = np.array([str(v) for v in np.load(f"{DATA}/{S}_labels.npy", allow_pickle=True)])
    fden = f"{RES}/denoised/{S}.npy"
    if extra:
        inputs = {name: np.load(f"{RES}/denoised_{name}/{S}.npy").astype('float64') for name in extra}
    elif not os.path.exists(fden):
        loc = MinMaxScaler().fit_transform(ad.obsm['spatial'].astype('float64')) * 20.
        os.makedirs(os.path.dirname(fden), exist_ok=True)
        np.save(fden, train_denoise(raw, genes, loc, device).astype('float32'))
    if not extra:
        inputs = dict(model=per10k(np.load(fden).astype('float64')), norm=per10k(raw))
    df, P = scan(S, inputs, raw, genes, lab, np.asarray(ad.obsm['spatial'], 'float64'))
    od = f"{RES}/interface_extra" if extra else f"{RES}/interface"
    os.makedirs(od, exist_ok=True); df.to_csv(f"{od}/{S}.csv", index=False)
    print(S, len(P), 'pairs', df.groupby('input').size().to_dict(), flush=True)
    print(df[(df.ligand == 'NPTX2') & (df.receptor == 'NPTXR')].to_string(), flush=True)


def scan(S, inputs, raw, genes, lab, xy, no_resid=(), bounds=BOUNDS):
    """Interface statistics and matched-null z for every (LR pair, boundary) and every input expression matrix.
    Inputs named in no_resid skip the depth residualisation (used only by the simulation)."""
    lib = raw.sum(1)
    gi = {}
    for i, g in enumerate(genes):
        gi.setdefault(g, i)
    frac = (raw > 0).mean(0); expressed = {g for g in gi if frac[gi[g]] >= FRAC}
    db = pd.read_csv(DB)
    P = sorted({(s, t) for s, t in zip(db['source'], db['target']) if isinstance(s, str) and isinstance(t, str)
                and s in expressed and t in expressed and s != t})
    lrg = sorted({g for p in P for g in p}); lgi = {g: i for i, g in enumerate(lrg)}
    _, idx = NearestNeighbors(n_neighbors=SCAN_K + 1).fit(xy).kneighbors(xy)
    ei = np.repeat(np.arange(len(xy)), SCAN_K); ej = idx[:, 1:].ravel(); k = ei < ej; ei, ej = ei[k], ej[k]
    ta, tb = lab[ei], lab[ej]
    ll = np.log(lib + 1.0)
    D = np.stack([np.ones(len(ei)), ll[ei] + ll[ej], np.abs(ll[ei] - ll[ej])], 1)
    Dp = np.linalg.pinv(D)               # (3, E): least-squares coefficients are Dp @ y

    def resid(y):
        return y - D @ (Dp @ y)

    rows = []
    for name, E in inputs.items():
        Xd = E[:, [gi[g] for g in lrg]]
        me = Xd.mean(0); dec = pd.qcut(me, 10, labels=False, duplicates='drop'); bydec = defaultdict(list)
        for g, d in zip(lrg, dec):
            bydec[d].append(g)
        cache = {}

        def edge(l, r):
            if (l, r) not in cache:
                L, R = Xd[:, lgi[l]], Xd[:, lgi[r]]
                y = np.log1p(L[ei] * R[ej] + L[ej] * R[ei])
                cache[(l, r)] = y if name in no_resid else resid(y)
            return cache[(l, r)]

        rng = np.random.default_rng(0)
        for A, B in bounds:
            bm = ((ta == A) & (tb == B)) | ((ta == B) & (tb == A)); wm = ((ta == A) & (tb == A)) | ((ta == B) & (tb == B))
            if bm.sum() < 30 or wm.sum() < 30:
                continue
            stat = lambda l, r: edge(l, r)[bm].mean() - edge(l, r)[wm].mean()
            for l, r in P:
                sv = stat(l, r); dl, dr = dec[lgi[l]], dec[lgi[r]]
                nl = np.array([stat(rng.choice(bydec[dl]), rng.choice(bydec[dr])) for _ in range(NNULL)])
                rows.append(dict(sample=S, input=name, boundary=f'{A}/{B}', ligand=l, receptor=r, stat=float(sv),
                                 z=float((sv - nl.mean()) / (nl.std() + 1e-9)), p_emp=float(((nl >= sv).sum() + 1) / (NNULL + 1)),
                                 n_bedges=int(bm.sum())))
    return pd.DataFrame(rows), P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--extra', nargs='*', default=[])
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for s in a.samples:
        main(s, dev, tuple(a.extra))
