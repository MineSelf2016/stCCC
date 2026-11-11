"""Combine results/interface/*.csv across sections and donors, separately for each input.
  stouffer : sum of section z / sqrt(n sections), hypotheses tested in >= 8 of 12 sections
  donor    : Stouffer within each donor, then Stouffer across the three donors (one-sided p), BH over all hypotheses
  replic.  : Spearman between the z vectors of adjacent serial sections; share of hypotheses with z > 0 in all donors"""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
HERE = os.path.dirname(os.path.abspath(__file__))
from datasets import RESULTS, GROUP as DONOR, ADJ, SECTIONS
R = os.path.join(RESULTS, 'interface')

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/sample_*.csv')) + sorted(glob.glob(f'{R}_extra/sample_*.csv'))])
df['sec'] = df['sample'].str.split('_').str[-1].astype(int); df['donor'] = df.sec.map(DONOR)
df['hyp'] = df.ligand + '-' + df.receptor + ' @ ' + df.boundary
out = []
for inp, d in df.groupby('input'):
    W = d.pivot_table(index='hyp', columns='sec', values='z')
    W = W[W.notna().sum(1) >= int(np.ceil(2 * len(SECTIONS) / 3))]   # 8 of 12 DLPFC sections
    st = W.sum(1) / np.sqrt(W.notna().sum(1))
    dz = pd.concat({dn: W[[c for c in W.columns if DONOR[c] == dn]].pipe(lambda x: x.sum(1) / np.sqrt(x.notna().sum(1)))
                    for dn in sorted(set(DONOR.values()))}, axis=1)
    dstat = dz.sum(1) / np.sqrt(dz.notna().sum(1)); p = norm.sf(dstat)
    q = multipletests(p, method='fdr_bh')[1]
    T = pd.DataFrame(dict(stouffer=st, donor_z=dstat, p=p, q=q, all_donors_pos=(dz > 0).all(1),
                          min_donor_z=dz.min(1))).sort_values('donor_z', ascending=False)
    T['rank'] = np.arange(1, len(T) + 1)
    T.to_csv(f'{R}/combined_{inp}.csv')
    adj = [spearmanr(*W[[a, b]].dropna().T.values)[0] for a, b in ADJ if a in W and b in W]
    out.append(dict(input=inp, hypotheses=len(T), adjacent_rho=np.mean(adj), q05=int((q < 0.05).sum()), q10=int((q < 0.10).sum()),
                    all_donors_pos_top20=int(T.all_donors_pos.head(20).sum())))
    print(f'\n== {inp}: top 10 by donor-level z')
    print(T.head(10).round(3).to_string())
    nx = [h for h in T.index if h.startswith('NPTX2-NPTXR')]
    print(T.loc[nx].round(3).to_string())
print('\n', pd.DataFrame(out).round(3).to_string(index=False))
