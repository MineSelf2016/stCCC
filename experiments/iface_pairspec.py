"""Pairing-specific interface scan: is an LR pair more boundary-enriched than either of its genes with a random partner?

The matched null of interface_scan.py replaces both genes of a pair by random genes of the same expression decile, so its
z-score also rewards a pair whose ligand (or receptor) alone is a boundary marker, whatever its partner. Here the null keeps
one gene and replaces the other (ligand kept, receptor swapped: 100 draws; receptor kept, ligand swapped: 100 draws), so the
z-score measures what the pairing adds beyond each gene's own spatial pattern:
  z_both : matched null of interface_scan.py (both partners swapped, 200 draws), recomputed here for reference
  z_pair : partner-swap null (one partner swapped, 200 draws)
Edge score, depth residualisation, edges, boundaries, LR pairs and expression deciles are those of interface_scan.scan; null
partners are drawn once per pair and shared across boundaries.
Inputs: model (results/denoised), norm (log1p per-10k counts), smooth_r, magic, scvi and the ablations (results/denoised_<m>).
Writes RESULTS/interface_pairspec/<sample>.csv.   Usage: STCCC_DATASET=<d> python iface_pairspec.py --samples sample_X ...
Summary: python iface_pairspec.py --summary <dataset> ..."""
import os, sys, glob, argparse, importlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
K, FRAC = 100, 0.02
INPUTS = ['model', 'norm', 'smooth_r', 'magic', 'scvi', 'noCCC', 'noGate', 'lr_noGate']


def per10k(E):
    return np.log1p(E / (E.sum(1, keepdims=True) + 1e-12) * 1e4)


def run(S):
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors
    from datasets import DATA, RESULTS as RES, BOUNDS, SCAN_K
    from rel_common import DB
    ad = sc.read_h5ad(f'{DATA}/{S}_with_graph.h5ad')
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    genes = list(ad.var_names); lib = raw.sum(1); xy = np.asarray(ad.obsm['spatial'], 'float64')
    lab = np.array([str(v) for v in np.load(f'{DATA}/{S}_labels.npy', allow_pickle=True)])
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
    ta, tb = lab[ei], lab[ej]; ll = np.log(lib + 1.0)
    D = np.stack([np.ones(len(ei)), ll[ei] + ll[ej], np.abs(ll[ei] - ll[ej])], 1); Dp = np.linalg.pinv(D)
    B = []
    for A, Bn in BOUNDS:
        bm = ((ta == A) & (tb == Bn)) | ((ta == Bn) & (tb == A)); wm = ((ta == A) & (tb == A)) | ((ta == Bn) & (tb == Bn))
        if bm.sum() >= 30 and wm.sum() >= 30:
            B.append((f'{A}/{Bn}', bm, wm))
    if not B:
        return
    rows = []
    for name in INPUTS:
        f = f'{RES}/denoised/{S}.npy' if name == 'model' else f'{RES}/denoised_{name}/{S}.npy'
        if name == 'norm':
            E = per10k(raw)
        elif os.path.exists(f):
            E = np.load(f).astype('float64'); E = per10k(E) if name == 'model' else E
        else:
            continue
        Xd = E[:, [gi[g] for g in lrg]]
        dec = pd.qcut(Xd.mean(0), 10, labels=False, duplicates='drop'); bydec = {}
        for g, d in zip(lrg, dec):
            bydec.setdefault(d, []).append(lgi[g])
        rng = np.random.default_rng(0)

        def draw(gene, avoid):
            pool = [j for j in bydec[dec[gene]] if j not in avoid]
            return rng.choice(pool, K) if pool else np.full(K, gene)
        jobs = []                                    # (pair index, kind, ligand idx, receptor idx)
        for p, (l, r) in enumerate(P):
            a, b = lgi[l], lgi[r]
            jobs.append((p, 0, a, b))
            jobs += [(p, 1, a, x) for x in draw(b, {a, b})]                       # receptor swapped
            jobs += [(p, 1, x, b) for x in draw(a, {a, b})]                       # ligand swapped
            jobs += [(p, 2, x, y) for x, y in zip(draw(a, {a, b}), draw(b, {a, b}))]   # both swapped (interface_scan null)
            jobs += [(p, 2, x, y) for x, y in zip(draw(a, {a, b}), draw(b, {a, b}))]
        J = np.array(jobs); st = np.zeros((len(B), len(J))); XI, XJ = Xd[ei], Xd[ej]
        for c in range(0, len(J), 1500):
            a, b = J[c:c + 1500, 2], J[c:c + 1500, 3]
            Y = np.log1p(XI[:, a] * XJ[:, b] + XJ[:, a] * XI[:, b])
            Y -= D @ (Dp @ Y)
            for t, (_, bm, wm) in enumerate(B):
                st[t, c:c + 1500] = Y[bm].mean(0) - Y[wm].mean(0)
        for t, (bn, _, _) in enumerate(B):
            for p, (l, r) in enumerate(P):
                m = J[:, 0] == p; kind = J[m, 1]; s = st[t, m]
                obs = s[kind == 0][0]; n1, n2 = s[kind == 1], s[kind == 2]
                rows.append(dict(sample=S, input=name, boundary=bn, ligand=l, receptor=r, stat=obs,
                                 z_pair=(obs - n1.mean()) / (n1.std() + 1e-9), z_both=(obs - n2.mean()) / (n2.std() + 1e-9)))
    od = f'{RES}/interface_pairspec'; os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{od}/{S}.csv', index=False); print(S, len(rows), flush=True)


def summary(name):
    from scipy.stats import spearmanr, wilcoxon
    os.environ['STCCC_DATASET'] = name
    import datasets; importlib.reload(datasets)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f'{datasets.RESULTS}/interface_pairspec/sample_*.csv')])
    df['sec'] = df['sample'].str.split('_').str[-1].astype(int); df['hyp'] = df.ligand + '-' + df.receptor + ' @ ' + df.boundary
    out = []
    for zc in ('z_both', 'z_pair'):
        for inp, d in df.groupby('input'):
            W = d.pivot_table(index='hyp', columns='sec', values=zc)
            for a, b in datasets.ADJ:
                out.append(dict(null=zc, input=inp, pair=f'{a}-{b}',
                                rho=spearmanr(*W[[a, b]].dropna().T.values)[0] if {a, b} <= set(W.columns) else np.nan))
    O = pd.DataFrame(out); O.to_csv(f'{datasets.RESULTS}/pairspec_pairs.csv', index=False)
    T = O.pivot_table(index='input', columns='null', values='rho', aggfunc='mean')
    print(f'== {name}: mean serial agreement'); print(T.round(3).to_string())
    for zc in ('z_both', 'z_pair'):
        W = O[O.null == zc].pivot(index='pair', columns='input', values='rho')
        for b in [c for c in W.columns if c != 'model']:
            x = (W['model'] - W[b]).dropna()
            if len(x):
                print(f'  {zc}: model vs {b:9s} higher {int((x > 0).sum())}/{len(x)}  diff {x.mean():+.3f}  '
                      f'P(model>) {wilcoxon(x, alternative="greater").pvalue:.3f}  P(model<) {wilcoxon(x, alternative="less").pvalue:.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='*', default=[]); ap.add_argument('--summary', nargs='*')
    a = ap.parse_args()
    for s in a.samples:
        run(s)
    for n in (a.summary or []):
        summary(n)
