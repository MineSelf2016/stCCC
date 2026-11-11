"""Apply the pre-specified selection rule of select_kl.py to results/selection/ and print the chosen candidate."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from datasets import RESULTS
R = os.path.join(RESULTS, 'selection')
J = pd.DataFrame([json.loads(l) for l in open(f'{R}/kl_select.jsonl')]).reindex(columns=['conf', 'kl', 'kind', 'mode', 'f', 'depth', 'seed', 'sample', 'fpr', 'auroc', 'tpr', 'split', 'down', 'unit', 'depth_rho', 'nmi']).drop_duplicates(['conf', 'kind', 'mode', 'f', 'depth', 'seed', 'sample'])
sim, real = J[J.kind == 'sim'], J[J.kind == 'real']
rows = []
for c in sorted(J.conf.unique()):
    s, r = sim[sim.conf == c], real[real.conf == c]
    lr = []
    for a, b in [('sample_151507', 'sample_151508'), ('sample_151509', 'sample_151510')]:
        fa, fb = f'{R}/iface_{c}_{a}.csv', f'{R}/iface_{c}_{b}.csv'
        if os.path.exists(fa) and os.path.exists(fb):
            za = pd.read_csv(fa).assign(h=lambda d: d.ligand + d.receptor + d.boundary).set_index('h').z
            zb = pd.read_csv(fb).assign(h=lambda d: d.ligand + d.receptor + d.boundary).set_index('h').z
            k = za.index.intersection(zb.index); lr.append(spearmanr(za[k], zb[k])[0])
    rows.append(dict(conf=c, kl=s.kl.iloc[0] if len(s) else r.kl.iloc[0], n_sim=len(s), n_real=len(r),
                     auroc=s[s.f > 1].auroc.mean(), auroc_prog=s[(s.f > 1) & (s['mode'] == 'program')].auroc.mean(),
                     auroc_gene=s[(s.f > 1) & (s['mode'] == 'gene')].auroc.mean(), tpr=s[s.f > 1].tpr.mean(),
                     fpr_null=s[s.f == 1].fpr.mean(), lrspec=np.mean(lr) if lr else np.nan,
                     split=r.split.mean(), down=r.down.mean(), unit=r.unit.mean(), nmi=r.nmi.mean(), depth=r.depth_rho.abs().mean()))
T = pd.DataFrame(rows).set_index('conf'); print(T.round(3).to_string())
c0 = T.loc['C0']
ok = T[(T.fpr_null <= 0.05) & (T.lrspec >= c0.lrspec - 0.05) & (T.nmi >= c0.nmi - 0.02)]
print('\neligible:', list(ok.index))
if len(ok):
    best = ok.auroc.max(); near = ok[ok.auroc >= best - 0.01]
    kl = near.kl.str.split(':').str[0].astype(float)
    print('chosen:', kl.idxmin(), near.loc[kl.idxmin(), 'kl'])
T.to_csv(f'{R}/selection_summary.csv')
