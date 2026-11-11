"""Within-layer depth dependence of spot/cell-level total communication scores (reviewer: is the depth link technical
or laminar biology?). For each section and layer (>= 50 units), Spearman between the unit-level total score and log
library size among the units of that layer only; on MERFISH also within layer x subclass (>= 50 cells).
Also reports the share of log-library variance explained by layer (R^2, one-way ANOVA).
Writes results[/abca1]/within_layer.csv (per section x method) and prints means over sections.
Usage: python within_layer.py   (both datasets; STCCC_ABCA1=orig for the four original ABCA-1 pairs)"""
import os, sys, pickle, importlib, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
SUBS = ['liana', 'commot', 'denoise_smooth_r', 'denoise_magic', 'denoise_scvi']
METHODS = ['counts', 'norm', 'smooth', 'liana_cosine', 'commot',
           'liana_morans', 'commot_r', 'norm_r', 'smooth_r', 'magic', 'scvi', 'model']
MIN = 50


def run(name):
    os.environ['STCCC_DATASET'] = name
    import datasets; importlib.reload(datasets)
    import scanpy as sc
    R = os.path.join(datasets.RESULTS, 'reliability')
    rows = []
    for s in datasets.SECTIONS:
        f = f'{R}/sample_{s}.pkl'
        if not os.path.exists(f):
            print('missing', f); continue
        r = pickle.load(open(f, 'rb'))
        for sub in SUBS:
            g = f'{R}_{sub}/sample_{s}.pkl'
            if os.path.exists(g):
                r['out']['full'].update(pickle.load(open(g, 'rb'))['out']['full'])
        ad = sc.read_h5ad(f'{datasets.DATA}/sample_{s}_with_graph.h5ad')
        lib = np.log(np.asarray(ad.X.sum(1)).ravel() + 1.0)
        lab = np.array([str(v) for v in np.load(f'{datasets.DATA}/sample_{s}_labels.npy', allow_pickle=True)])
        sub = ad.obs['subclass'].astype(str).values if 'subclass' in ad.obs else None
        ok = (lab != 'NA') & (lab != 'nan') & (lab != 'None')
        layers = [l for l in np.unique(lab[ok]) if (lab == l).sum() >= MIN]
        # share of log-library variance explained by layer
        mu = {l: lib[lab == l].mean() for l in np.unique(lab[ok])}
        fit = np.array([mu[l] for l in lab[ok]])
        r2_layer = 1 - ((lib[ok] - fit) ** 2).sum() / ((lib[ok] - lib[ok].mean()) ** 2).sum()
        for m in METHODS:
            if m not in r['out']['full']:
                continue
            tot = r['out']['full'][m]['total']; tot = tot.mean(1) if tot.ndim == 2 else tot
            wl = [spearmanr(tot[lab == l], lib[lab == l])[0] for l in layers]
            row = dict(sample=s, method=m, rho_all=spearmanr(tot[ok], lib[ok])[0], rho_within_layer=np.nanmedian(wl),
                       r2_layer_lib=r2_layer)
            if sub is not None:
                ws = [spearmanr(tot[(lab == l) & (sub == c)], lib[(lab == l) & (sub == c)])[0]
                      for l in layers for c in np.unique(sub[lab == l]) if ((lab == l) & (sub == c)).sum() >= MIN]
                row['rho_within_layer_subclass'] = np.nanmedian(ws); row['n_strata'] = len(ws)
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(datasets.RESULTS, 'within_layer.csv'), index=False)
    cols = [c for c in ['rho_all', 'rho_within_layer', 'rho_within_layer_subclass'] if c in D]
    print(f'== {name}: {D["sample"].nunique()} sections; layer explains {D.groupby("sample").r2_layer_lib.first().mean():.2f} '
          f'of log-library variance (mean over sections)')
    print(D.groupby('method')[cols].mean().loc[[m for m in METHODS if m in set(D.method)]].round(3).to_string())


if __name__ == '__main__':
    for n in (sys.argv[1:] or ('dlpfc', 'abca1', 'her2')):
        run(n)
