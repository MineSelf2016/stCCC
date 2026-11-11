"""Spatial neighbour graph and edge-level communication scores.

Everything lives on a fixed spatial kNN graph (N x k), so storage is O(Nk) and every spot
keeps k communication partners regardless of dataset size or mini-batch composition.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def knn_graph(coords, k=8):
    """Indices (N, k) and Euclidean distances (N, k) of each spot's k nearest neighbours, self excluded."""
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    return idx[:, 1:], d[:, 1:]


def lr_pairs(genes, db_path, shuffle=False, seed=0):
    """Ligand-receptor pairs present in `genes`. With shuffle=True the receptor column is permuted,
    which keeps every gene's ligand and receptor degree but destroys the pairing."""
    db = pd.read_csv(db_path); g = set(genes)
    P = sorted({(s, t) for s, t in zip(db['source'], db['target'])
                if isinstance(s, str) and isinstance(t, str) and s in g and t in g and s != t})
    if shuffle:
        rng = np.random.default_rng(seed)
        lig = [p[0] for p in P]; rec = [p[1] for p in P]; rng.shuffle(rec)
        P = [(l, r) for l, r in zip(lig, rec) if l != r]
    return P


def lognorm(raw):
    """log1p of library-normalised counts (counts per 10k)."""
    lib = raw.sum(1, keepdims=True)
    return np.log1p(raw / (lib + 1e-8) * 1e4)


def edge_lr_score(expr, genes, pairs, nbr):
    """Saturating LR co-expression summed over pairs and both orderings, on kNN edges only.
    expr: (N, G) non-negative library-normalised expression. Returns (N, k) raw scores."""
    gi = {g: i for i, g in enumerate(genes)}
    N, k = nbr.shape
    S = np.zeros((N, k))
    prods = []
    for l, r in pairs:
        L, R = expr[:, gi[l]], expr[:, gi[r]]
        a = L[:, None] * R[nbr] + L[nbr] * R[:, None]
        prods.append(a)
    if not prods:
        return S
    allp = np.stack(prods)
    nz = allp[allp > 0]
    kh = float(np.median(nz)) if nz.size else 1.0
    return (allp / (kh + allp)).sum(0)


def depth_score(lib, nbr):
    """Pure sequencing-depth affinity: log lib_i + log lib_j on each edge (contains no LR biology)."""
    ll = np.log(lib + 1.0)
    return ll[:, None] + ll[nbr]


def residualise(S, lib, nbr):
    """Remove the sequencing-depth component of an edge score.
    Regress log1p(S) on [1, log lib_i + log lib_j, |log lib_i - log lib_j|] across all edges and keep the residual."""
    y = np.log1p(S).ravel()
    ll = np.log(lib + 1.0)
    s = (ll[:, None] + ll[nbr]).ravel(); d = np.abs(ll[:, None] - ll[nbr]).ravel()
    X = np.stack([np.ones_like(s), s, d], 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return (y - X @ beta).reshape(S.shape)


def zscore(S):
    return (S - S.mean()) / (S.std() + 1e-8)


def build_scores(raw, genes, coords, cond, db_path, k=8, seed=0):
    """Edge scores for one experimental condition.
      lr     : true LR pairs, depth-residualised   (the model)
      randlr : degree-preserving shuffled pairs, same pipeline
      lr_raw : true LR pairs, NOT residualised     (what the submitted paper used, up to normalisation)
      depth  : depth-only affinity, no LR biology
      none   : no communication input
    Returns nbr (N,k), dist (N,k), score (N,k) z-scored or None, and diagnostics."""
    nbr, dist = knn_graph(coords, k)
    lib = raw.sum(1)
    info = {}
    if cond == 'none':
        return nbr, dist, None, info
    if cond == 'depth':
        S = depth_score(lib, nbr)
    else:
        P = lr_pairs(genes, db_path, shuffle=(cond == 'randlr'), seed=seed)
        info['n_pairs'] = len(P)
        S = edge_lr_score(lognorm(raw), genes, P, nbr)
        S = residualise(S, lib, nbr) if cond in ('lr', 'randlr') else np.log1p(S)
    S = zscore(S)
    from scipy.stats import spearmanr
    info['rho_depth'] = float(spearmanr(S.ravel(), depth_score(lib, nbr).ravel())[0])
    return nbr, dist, S, info
