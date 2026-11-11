"""Re-selection of the stCCC KL pressure on the development donor only (Br5292, sections 151507-151510).

Candidates (STCCC_KL = KL target per latent dimension : min beta : initial beta):
  C0 0.025:4:10 (original)  C1 0.1:1:1  C2 0.25:0.5:1  C3 0.5:0.2:1  C4 1:0.1:1
Evidence, all from the development donor:
  sim  : semi-synthetic benchmark on section 151507 (simulate.py with STCCC_SIMSEC=151507): program mode f = 2, 3;
         gene mode f = 3; null f = 1; depth none / gradient; seeds 0, 1 (16 datasets per candidate)
  real : on each development section, laminar-pattern split-half and down-25% agreement, spot-level split-half
         agreement, the interface scan (for LR-specific agreement of the two serial pairs) and domain NMI (seed 0)
Rule, fixed before the runs: keep candidates whose null false-positive rate (z > 1.645) is at most 0.05, whose
LR-specific serial agreement is at most 0.05 below C0 and whose NMI is at most 0.02 below C0; among them choose the
highest mean AUROC over the planted simulations; candidates within 0.01 AUROC of the best are resolved towards the
smaller KL target (stronger regularisation). Results: results/selection/ (kl_select.jsonl, iface_<C>_<section>.csv)."""
import os, sys, json, argparse, warnings
CONF = {'C0': '0.025:4:10', 'C1': '0.1:1:1', 'C2': '0.25:0.5:1', 'C3': '0.5:0.2:1', 'C4': '1:0.1:1'}
SIMS = [(m, f, d, s) for m, f in (('program', 2.0), ('program', 3.0), ('gene', 3.0), ('gene', 1.0)) for d in ('none', 'gradient') for s in (0, 1)]
DEV = ['sample_151507', 'sample_151508', 'sample_151509', 'sample_151510']
ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); a = ap.parse_args()
JOBS = [('sim', c, i) for c in CONF for i in range(len(SIMS))] + [('real', c, s) for c in CONF for s in DEV]
kind, conf, item = JOBS[a.job]
os.environ['STCCC_KL'] = CONF[conf]; os.environ['STCCC_SIMSEC'] = '151507'
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import RESULTS, DATA
from ccc_reliability import train_denoise
from rel_common import load, versions, per10k, summarise, knn_graph, lr_pairs, DB
from interface_scan import scan
OUT = os.path.join(RESULTS, 'selection'); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def record(d):
    d = dict(conf=conf, kl=CONF[conf], **d); print(d, flush=True)
    with open(f'{OUT}/kl_select.jsonl', 'a') as fh:
        fh.write(json.dumps(d) + '\n')


if kind == 'sim':
    from simulate import base, simulate
    mode, f, depth, seed = SIMS[item]
    raw, genes, lab, xy = base()
    X, truth, boosted = simulate(raw, genes, lab, xy, f, depth, seed, mode)
    E = per10k(train_denoise(X, genes, MinMaxScaler().fit_transform(xy) * 20., dev))
    df, _ = scan('sim', dict(model=E), X, genes, lab, xy)
    tr = {tuple(t) for t in truth}; g = set(boosted)
    df['planted'] = [(l, r, b) in tr for l, r, b in zip(df.ligand, df.receptor, df.boundary)]
    df = df[df.planted | ~(df.ligand.isin(g) | df.receptor.isin(g))]
    pos, neg = df[df.planted], df[~df.planted]
    record(dict(kind='sim', mode=mode, f=f, depth=depth, seed=seed, fpr=float((neg.z > 1.645).mean()),
                auroc=float(roc_auc_score(df.planted, df.z)) if f > 1 else None, tpr=float((pos.z > 1.645).mean()) if f > 1 else None))
else:
    from run_dlpfc import run
    S = item
    ad, loc, raw, genes, labels = load(S)
    layers = sorted({x for x in labels if x is not None}); keep = [i for i, l in enumerate(layers) if l != 'NA']
    nbr, _ = knn_graph(loc, 8); pairs = lr_pairs(genes, DB)
    o = {}
    for v, Rv in versions(raw).items():
        E = per10k(train_denoise(Rv, genes, loc, dev))
        o[v] = summarise(E, genes, pairs, nbr, Rv.sum(1), labels, layers, True)
        if v == 'full':
            lab = np.array([str(x) for x in np.load(f'{DATA}/{S}_labels.npy', allow_pickle=True)])
            df, _ = scan(S, dict(model=E), raw, genes, lab, np.asarray(ad.obsm['spatial'], 'float64'))
            df.to_csv(f'{OUT}/iface_{conf}_{S}.csv', index=False)
    cen = lambda LM: (LM[:, keep] - np.nanmean(LM[:, keep], 1, keepdims=True)).ravel()
    corr = lambda x, y: spearmanr(x[~(np.isnan(x) | np.isnan(y))], y[~(np.isnan(x) | np.isnan(y))])[0]
    nmi = run(S, 'lr', 0, 10.0, 3.0, 8, 100, dev)['NMI']
    record(dict(kind='real', sample=S, split=corr(cen(o['A']['LM']), cen(o['B']['LM'])), down=corr(cen(o['d25']['LM']), cen(o['full']['LM'])),
                unit=corr(o['A']['total'].ravel(), o['B']['total'].ravel()), depth_rho=o['full']['rho_total'], nmi=nmi))
