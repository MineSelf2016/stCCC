"""Summarise results/reliability/*.pkl into four reliability metrics per scoring method.
  depth_spot: Spearman between the spot-level total score (edge methods: mean over the spot's edges) and log library size
  depth     : |Spearman| between per-pair score and depth (median over pairs; edge methods vs log lib_i + log lib_j)
  splithalf : Spearman between the layer patterns of the two independent binomial halves
  down25    : Spearman between the layer pattern at 25% depth and at full depth
  adjacent  : Spearman between the layer patterns of adjacent serial sections (same donor), full data
A layer pattern is the per-pair mean edge score in each annotated layer, centred per pair across layers,
so a method that returns a constant or a pure pair-level offset scores zero."""
import os, glob, pickle, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
from datasets import DATA, RESULTS, ADJ
R = os.path.join(RESULTS, 'reliability')
METHODS = ['counts', 'norm', 'smooth', 'liana_cosine', 'liana_morans', 'commot', 'commot_r', 'norm_r', 'smooth_r',
           'magic_nr', 'magic', 'scvi_nr', 'scvi', 'noCCC', 'noCCC_nr', 'attn', 'attn_nr', 'noGate', 'noGate_nr', 'lr_noGate', 'lr_noGate_nr', 'model_nr', 'model']


def align(ex, layers):
    """Re-index the layer columns of a merged result to the main result's layers (older pickles kept an 'NA' column)."""
    if list(ex['layers']) == list(layers):
        return ex['out']
    idx = [list(ex['layers']).index(l) if l in ex['layers'] else None for l in layers]
    out = {}
    for v, d in ex['out'].items():
        out[v] = {}
        for k, o in d.items():
            LM = np.full((o['LM'].shape[0], len(layers)), np.nan)
            for j, i in enumerate(idx):
                if i is not None:
                    LM[:, j] = o['LM'][:, i]
            out[v][k] = {**o, 'LM': LM}
    return out


def centred(LM, layers=None):
    if layers is not None and 'NA' in layers:       # 'NA' = a handful of unannotated spots, not a layer
        LM = np.delete(LM, layers.index('NA'), axis=1)
    return LM - np.nanmean(LM, 1, keepdims=True)


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[m], b[m])[0]


def frame(res, v, meth, pairs=None, layers=None):
    keep = [x for x in res['layers'] if x != 'NA']
    df = pd.DataFrame(centred(res['out'][v][meth]['LM'], res['layers']), index=[f'{l}|{r}' for l, r in res['pairs']], columns=keep)
    return df if pairs is None else df.loc[pairs, layers]


D = {int(os.path.basename(f)[7:-4]): pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'{R}/sample_*.pkl'))}
# merge the published-tool baseline, the denoise-then-score baselines and the model ablations (same thinning, same pairs and layers)
for sub, ren in [('liana', {}), ('commot', {}), ('denoise_smooth_r', {}), ('denoise_magic', {}), ('denoise_scvi', {}), ('noCCC', {'model': 'noCCC', 'model_nr': 'noCCC_nr'}), ('attn', {'model': 'attn', 'model_nr': 'attn_nr'}),
                 ('noGate', {'model': 'noGate', 'model_nr': 'noGate_nr'}), ('lr_noGate', {'model': 'lr_noGate', 'model_nr': 'lr_noGate_nr'})]:
    for s, res in D.items():
        f = f'{R}_{sub}/sample_{s}.pkl'
        if os.path.exists(f):
            ex = pickle.load(open(f, 'rb')); exo = align(ex, res['layers'])
            for v in res['out']:
                res['out'][v].update({ren.get(k, k): o for k, o in exo[v].items()})
_miss = [m for m in METHODS if not all(m in r['out']['full'] for r in D.values())]
if _miss: print('not on every section, skipped:', _miss)
METHODS = [m for m in METHODS if m not in _miss]
import scanpy as sc
LIB = {}
for s in D:
    X = sc.read_h5ad(f'{DATA}/sample_{s}_with_graph.h5ad').X
    LIB[s] = np.log(np.asarray(X.sum(1)).ravel() + 1.0)
rows = []
for s, res in D.items():
    for m in METHODS:
        o = res['out']
        tot = o['full'][m]['total']; tot = tot.mean(1) if tot.ndim == 2 else tot
        rows.append(dict(sample=s, method=m,
                         depth_spot=float(spearmanr(tot, LIB[s])[0]),
                         depth=float(np.nanmedian(np.abs(o['full'][m]['rho']))),
                         splithalf=corr(centred(o['A'][m]['LM'], res['layers']).ravel(), centred(o['B'][m]['LM'], res['layers']).ravel()),
                         down25=corr(centred(o['d25'][m]['LM'], res['layers']).ravel(), centred(o['full'][m]['LM'], res['layers']).ravel()),
                         unit_splithalf=corr(o['A'][m]['total'].ravel(), o['B'][m]['total'].ravel())))
T = pd.DataFrame(rows)
adj = []
for a, b in ADJ:
    if a in D and b in D:
        P = sorted(set(f'{l}|{r}' for l, r in D[a]['pairs']) & set(f'{l}|{r}' for l, r in D[b]['pairs']))
        L = sorted((set(D[a]['layers']) & set(D[b]['layers'])) - {'NA'})
        for m in METHODS:
            adj.append(dict(pair=f'{a}-{b}', method=m, n_pairs=len(P),
                            adjacent=corr(frame(D[a], 'full', m, P, L).values.ravel(), frame(D[b], 'full', m, P, L).values.ravel())))
A = pd.DataFrame(adj)
summ = T.groupby('method')[['depth_spot', 'depth', 'splithalf', 'down25', 'unit_splithalf']].mean()
if len(A):
    summ['adjacent'] = A.groupby('method').adjacent.mean()
summ = summ.loc[[m for m in METHODS if m in summ.index]].round(3)
print(f'{len(D)} sections'); print(summ.to_string())
print(T.pivot(index='sample', columns='method', values='down25')[METHODS].round(3).to_string())
if len(A):
    print(A.pivot(index='pair', columns='method', values='adjacent')[METHODS].round(3).to_string())
summ.to_csv(os.path.join(R, 'summary.csv')); T.to_csv(os.path.join(R, 'per_section.csv')); A.to_csv(os.path.join(R, 'adjacent.csv'))
