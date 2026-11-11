"""Core of the reliability audit: count versions, edge-level LR scores, pattern summaries and interface scans.

Every function here reproduces the protocol of the manuscript (experiments/ccc_reliability.py, rel_common.py,
reliability_summary.py, nar_summary.py, interface_scan.py and iface_pairspec.py) without depending on the dataset layout
of the experiments folder, so that any expression input can be audited on any section."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------------------------------------- basics
def knn(xy, k):
    """Indices (N, k) and distances (N, k) of the k nearest neighbours of each unit, self excluded."""
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    return idx[:, 1:], d[:, 1:]


def per10k(C):
    """log1p of counts (or a rate) per 10,000: the scale on which every input is scored."""
    C = np.asarray(C, 'float64')
    return np.log1p(C / (C.sum(1, keepdims=True) + 1e-12) * 1e4)


def lr_pairs(genes, db):
    """LR pairs whose two genes are both in `genes`. `db`: path to a csv with columns source/target, a DataFrame with
    those columns, or an iterable of (ligand, receptor) tuples."""
    if isinstance(db, str):
        db = pd.read_csv(db)
    it = zip(db['source'], db['target']) if isinstance(db, pd.DataFrame) else db
    g = set(genes)
    return sorted({(s, t) for s, t in it if isinstance(s, str) and isinstance(t, str) and s in g and t in g and s != t})


def count_versions(counts, seed=0):
    """full counts, two independent binomial halves (A + B = full) and a 25% binomial downsample."""
    rng = np.random.default_rng(seed)
    C = np.rint(np.asarray(counts)).astype(np.int64)
    A = rng.binomial(C, 0.5)
    return dict(full=C.astype('float64'), A=A.astype('float64'), B=(C - A).astype('float64'),
                d25=rng.binomial(C, 0.25).astype('float64'))


def corr(a, b):
    a, b = np.asarray(a, 'float64').ravel(), np.asarray(b, 'float64').ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[m], b[m])[0]) if m.sum() > 2 else np.nan


def _depth_design(lib, ei, ej):
    ll = np.log(lib + 1.0)
    return np.stack([np.ones(len(ei)), ll[ei] + ll[ej], np.abs(ll[ei] - ll[ej])], 1)


# ------------------------------------------------------------------------------------------------ section container
@dataclass
class Section:
    """One tissue section. counts: (N, G) integer counts; xy: (N, 2) coordinates; labels: (N,) region or layer labels
    (None/NaN/'NA' = unannotated); genes: G gene symbols. k: neighbours of the scoring graph (8 in the manuscript);
    scan_k: neighbours of the interface-scan graph (6 for Visium and single cells, 8 for a square ST array)."""
    counts: np.ndarray
    xy: np.ndarray
    labels: np.ndarray
    genes: list
    name: str = 'section'
    k: int = 8
    scan_k: int = 6
    nbr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.counts = np.rint(np.asarray(self.counts.toarray() if hasattr(self.counts, 'toarray') else self.counts)).astype('float64')
        self.xy = np.asarray(self.xy, 'float64'); self.genes = list(self.genes)
        lab = np.array([None if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ('NA', 'nan', 'None')) else str(x)
                        for x in self.labels], dtype=object)
        self.labels = lab
        self.layers = sorted({x for x in lab if x is not None})
        # scoring graph on coordinates rescaled to [0, 20] per axis, as in the manuscript (the interface scan uses raw xy)
        from sklearn.preprocessing import MinMaxScaler
        self.nbr, _ = knn(MinMaxScaler().fit_transform(self.xy) * 20., self.k)

    @classmethod
    def from_anndata(cls, adata, label_key, spatial_key='spatial', layer=None, **kw):
        X = adata.layers[layer] if layer else adata.X
        return cls(counts=X, xy=adata.obsm[spatial_key], labels=adata.obs[label_key].to_numpy(), genes=list(adata.var_names), **kw)


# -------------------------------------------------------------------------------------------------- pattern summary
def summarise(E, sec, pairs, lib, residualise=True):
    """Edge scores y_ij = log1p(L_i R_j + L_j R_i) on the scoring graph, optionally depth-residualised per pair on
    [1, log l_i + log l_j, |log l_i - log l_j|]; returns the per-pair layer means (P x L) and the edge total (N x k)."""
    gi = {}
    for i, g in enumerate(sec.genes):
        gi.setdefault(g, i)
    nbr = sec.nbr; N, k = nbr.shape
    ei = np.repeat(np.arange(N), k); ej = nbr.ravel()
    D = _depth_design(lib, ei, ej); Dp = np.linalg.pinv(D)
    ok = np.array([x is not None for x in sec.labels])
    LM = np.full((len(pairs), len(sec.layers)), np.nan); total = np.zeros((N, k))
    masks = [ok & (sec.labels == lay) for lay in sec.layers]
    for p, (l, r) in enumerate(pairs):
        L, R = E[:, gi[l]], E[:, gi[r]]
        y = np.log1p(L[:, None] * R[nbr] + L[nbr] * R[:, None]).ravel()
        if residualise:
            y = y - D @ (Dp @ y)
        y = y.reshape(N, k); total += y; ym = y.mean(1)
        for q, m in enumerate(masks):
            if m.any():
                LM[p, q] = ym[m].mean()
    return dict(LM=LM, total=total)


def centred(LM):
    return LM - np.nanmean(LM, 1, keepdims=True)


def depth_pattern_share(LM, sec, lib):
    """Share of the centred laminar/regional pattern explained by the regions' mean log library size (R^2_depth)."""
    ll = np.log(lib + 1.0)
    dp = np.array([ll[sec.labels == lay].mean() for lay in sec.layers]); dp -= dp.mean()
    C = centred(LM); C = C[~np.isnan(C).any(1)]
    if not len(C) or dp @ dp == 0:
        return np.nan
    b = (C @ dp) / (dp @ dp)
    return float(1 - ((C - np.outer(b, dp)) ** 2).sum() / (C ** 2).sum())


# --------------------------------------------------------------------------------------------------- interface scan
def default_boundaries(sec, min_edges=30):
    """Every pair of regions joined by at least `min_edges` edges of the scan graph."""
    idx, _ = knn(sec.xy, sec.scan_k)
    ei = np.repeat(np.arange(len(sec.xy)), sec.scan_k); ej = idx.ravel(); keep = ei < ej
    a, b = sec.labels[ei[keep]], sec.labels[ej[keep]]
    out = []
    for i, A in enumerate(sec.layers):
        for B in sec.layers[i + 1:]:
            if (((a == A) & (b == B)) | ((a == B) & (b == A))).sum() >= min_edges:
                out.append((A, B))
    return out


def interface_scan(E, sec, db, boundaries=None, n_null=100, frac=0.02, seed=0, min_edges=30):
    """Boundary-versus-within-region enrichment of every LR pair, standardised against two nulls:
      z_both : both genes replaced by random genes from the same expression deciles (2 x n_null draws)
      z_pair : one gene kept and its partner replaced (n_null draws each way), i.e. what the pairing adds
    LR pairs: both genes detected in >= `frac` of the units of this section. Returns a DataFrame."""
    C = sec.counts; lib = C.sum(1)
    gi = {}
    for i, g in enumerate(sec.genes):
        gi.setdefault(g, i)
    det = (C > 0).mean(0)
    P = [p for p in lr_pairs(sec.genes, db) if det[gi[p[0]]] >= frac and det[gi[p[1]]] >= frac]
    if not P:
        return pd.DataFrame()
    lrg = sorted({g for p in P for g in p}); lgi = {g: i for i, g in enumerate(lrg)}
    idx, _ = knn(sec.xy, sec.scan_k)
    ei = np.repeat(np.arange(len(sec.xy)), sec.scan_k); ej = idx.ravel(); keep = ei < ej; ei, ej = ei[keep], ej[keep]
    ta, tb = sec.labels[ei], sec.labels[ej]
    D = _depth_design(lib, ei, ej); Dp = np.linalg.pinv(D)
    B = []
    for A, Bn in (boundaries or default_boundaries(sec, min_edges)):
        bm = ((ta == A) & (tb == Bn)) | ((ta == Bn) & (tb == A)); wm = ((ta == A) & (tb == A)) | ((ta == Bn) & (tb == Bn))
        if bm.sum() >= min_edges and wm.sum() >= min_edges:
            B.append((f'{A}/{Bn}', bm, wm, int(bm.sum())))
    if not B:
        return pd.DataFrame()
    Xd = E[:, [gi[g] for g in lrg]]
    dec = pd.qcut(Xd.mean(0), 10, labels=False, duplicates='drop'); bydec = {}
    for g, d in zip(lrg, dec):
        bydec.setdefault(d, []).append(lgi[g])
    rng = np.random.default_rng(seed)

    def draw(gene, avoid):
        pool = [j for j in bydec[dec[gene]] if j not in avoid]
        return rng.choice(pool, n_null) if pool else np.full(n_null, gene)
    jobs = []
    for p, (l, r) in enumerate(P):
        a, b = lgi[l], lgi[r]
        jobs.append((p, 0, a, b))
        jobs += [(p, 1, a, x) for x in draw(b, {a, b})] + [(p, 1, x, b) for x in draw(a, {a, b})]
        jobs += [(p, 2, x, y) for x, y in zip(draw(a, {a, b}), draw(b, {a, b}))]
        jobs += [(p, 2, x, y) for x, y in zip(draw(a, {a, b}), draw(b, {a, b}))]
    J = np.array(jobs); st = np.zeros((len(B), len(J))); XI, XJ = Xd[ei], Xd[ej]
    for c in range(0, len(J), 1500):
        a, b = J[c:c + 1500, 2], J[c:c + 1500, 3]
        Y = np.log1p(XI[:, a] * XJ[:, b] + XJ[:, a] * XI[:, b]); Y -= D @ (Dp @ Y)
        for t, (_, bm, wm, _) in enumerate(B):
            st[t, c:c + 1500] = Y[bm].mean(0) - Y[wm].mean(0)
    rows = []
    for t, (bn, _, _, nb) in enumerate(B):
        for p, (l, r) in enumerate(P):
            m = J[:, 0] == p; kind = J[m, 1]; s = st[t, m]
            obs = s[kind == 0][0]; n1, n2 = s[kind == 1], s[kind == 2]
            rows.append(dict(boundary=bn, ligand=l, receptor=r, stat=obs, n_boundary_edges=nb,
                             z_both=(obs - n2.mean()) / (n2.std() + 1e-9), z_pair=(obs - n1.mean()) / (n1.std() + 1e-9)))
    return pd.DataFrame(rows)
