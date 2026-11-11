"""One summary per dataset for the NAR manuscript (tables and the new figures read these files).
  results[/abca1]/nar_summary.csv : per method, means over sections of
      depth  : spot-level Spearman between total score and log library size
      expl   : share of the centred laminar pattern explained by the layers' mean log library size (R^2_depth)
      split, down, unit, adj : as in reliability_summary.py
      lrspec : LR-specific serial agreement = Spearman between the interface z-scores (matched-null) of adjacent
               serial sections, all hypotheses (inputs that were scanned only)
  results[/abca1]/nar_iface_pairs.csv : lrspec per serial pair and input
Usage: python nar_summary.py   (runs both datasets)"""
import os, sys, glob, pickle, importlib, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
SUBS = ['liana', 'commot', 'denoise_smooth_r', 'denoise_magic', 'denoise_scvi']
ABL = ['noCCC', 'attn', 'noGate', 'lr_noGate']

from reliability_summary_align import align

def run(name):
    os.environ['STCCC_DATASET'] = name
    import datasets; importlib.reload(datasets)
    import scanpy as sc
    R = os.path.join(datasets.RESULTS, 'reliability')
    rows = []
    for s in datasets.SECTIONS:
        r = pickle.load(open(f'{R}/sample_{s}.pkl', 'rb'))
        for sub in SUBS:
            f = f'{R}_{sub}/sample_{s}.pkl'
            if os.path.exists(f):
                ex = pickle.load(open(f, 'rb'))
                for v in r['out']:
                    r['out'][v].update(ex['out'][v])
        for abl in ABL:                                   # model ablations: rename model -> <ablation>
            f = f'{R}_{abl}/sample_{s}.pkl'
            if os.path.exists(f):
                exo = align(pickle.load(open(f, 'rb')), r['layers'])
                for v in r['out']:
                    r['out'][v][abl] = exo[v]['model']
        X = sc.read_h5ad(f'{datasets.DATA}/sample_{s}_with_graph.h5ad').X
        lib = np.log(np.asarray(X.sum(1)).ravel() + 1.0)
        y = np.array([str(v) for v in np.load(f'{datasets.DATA}/sample_{s}_labels.npy', allow_pickle=True)])
        ki = [i for i, l in enumerate(r['layers']) if l != 'NA']
        dp = np.array([lib[y == r['layers'][i]].mean() for i in ki]); dp -= dp.mean()
        for m, o in r['out']['full'].items():
            C = o['LM'][:, ki]; C = C - np.nanmean(C, 1, keepdims=True); C = C[~np.isnan(C).any(1)]
            b = (C @ dp) / (dp @ dp)
            rows.append(dict(sample=s, method=m, expl=1 - ((C - np.outer(b, dp)) ** 2).sum() / (C ** 2).sum()))
    E = pd.DataFrame(rows).groupby('method').expl.mean()
    S = pd.read_csv(f'{R}/summary.csv').set_index('method')
    S['depth'] = pd.read_csv(f'{R}/per_section.csv').groupby('method').depth_spot.mean()
    S['expl'] = E
    I = os.path.join(datasets.RESULTS, 'interface')
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{I}/sample_*.csv')) + sorted(glob.glob(f'{I}_extra/sample_*.csv'))
                    + sorted(glob.glob(f'{I}_ablation/sample_*.csv')) + sorted(glob.glob(f'{I}_noGate/sample_*.csv'))
                    + sorted(glob.glob(f'{I}_lr_noGate/sample_*.csv'))])
    df['sec'] = df['sample'].str.split('_').str[-1].astype(int); df['hyp'] = df.ligand + '-' + df.receptor + ' @ ' + df.boundary
    pr = []
    for inp, d in df.groupby('input'):
        W = d.pivot_table(index='hyp', columns='sec', values='z')
        for a, b in datasets.ADJ:
            pr.append(dict(input={'norm': 'norm_r'}.get(inp, inp), pair=f'{a}-{b}', group=datasets.GROUP[a],
                           rho=spearmanr(*W[[a, b]].dropna().T.values)[0] if {a, b} <= set(W.columns) else np.nan))
    P = pd.DataFrame(pr); S['lrspec'] = P.groupby('input').rho.mean()
    S.to_csv(os.path.join(datasets.RESULTS, 'nar_summary.csv')); P.to_csv(os.path.join(datasets.RESULTS, 'nar_iface_pairs.csv'), index=False)
    print(f'== {name}'); print(S[['depth', 'expl', 'splithalf', 'down25', 'adjacent', 'unit_splithalf', 'lrspec']].round(3).to_string())


if __name__ == '__main__':
    for n in (sys.argv[1:] or ('dlpfc', 'abca1', 'her2')):
        run(n)
