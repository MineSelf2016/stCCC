"""Shared pieces of the reliability protocol: data loading, count versions and edge-level summaries.
Identical to the functions in ccc_reliability.py, kept free of torch so that baselines in other
environments (MAGIC, COMMOT) can use them."""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
from ccc_graph import knn_graph, lr_pairs
from datasets import DATA, RESULTS
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')


def load(sample):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    loc = MinMaxScaler().fit_transform(ad.obsm['spatial'].astype('float64')) * 20.
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    labels = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)
    # 'NA' (a handful of unannotated spots) is kept as its own column, exactly as in the stored
    # results/reliability/*.pkl; reliability_summary.centred() drops it before any comparison
    labels = np.array([None if pd.isnull(x) else str(x) for x in labels], dtype=object)
    return ad, loc, raw, list(ad.var_names), labels


def versions(raw):
    """full, two independent binomial halves and a 25% downsample (seed 0, as in ccc_reliability.py)."""
    rng = np.random.default_rng(0)
    A = rng.binomial(raw.astype(np.int64), 0.5).astype('float64')
    return dict(full=raw, A=A, B=raw - A, d25=rng.binomial(raw.astype(np.int64), 0.25).astype('float64'))


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
