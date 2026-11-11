"""Check that stccc_audit reproduces the stored manuscript values on DLPFC sections 151673/151674 for the normalised
and smoothed inputs (with and without depth residualisation).  Run from the repository root:
    python -m stccc_audit.tests.test_reproduce"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
from stccc_audit import Section, run
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(HERE, '..', '..', 'data')), 'dlpfc')
RES = os.path.join(ROOT, 'results')
BOUNDS = [('Layer1', 'Layer2'), ('Layer2', 'Layer3'), ('Layer3', 'Layer4'), ('Layer4', 'Layer5'), ('Layer5', 'Layer6'), ('Layer6', 'WM')]
S = [151673, 151674]
secs = []
for s in S:
    a = sc.read_h5ad(f'{DATA}/sample_{s}_with_graph.h5ad')
    lab = np.load(f'{DATA}/sample_{s}_labels.npy', allow_pickle=True)
    secs.append(Section(counts=a.X, xy=a.obsm['spatial'], labels=lab, genes=list(a.var_names), name=str(s)))
rep = run(secs, ['norm', 'smooth'], serial=[(0, 1)], residualise='both', boundaries=BOUNDS, verbose=False)
ref = pd.read_csv(f'{RES}/reliability/per_section.csv'); adj = pd.read_csv(f'{RES}/reliability/adjacent.csv')
ps = pd.read_csv(f'{RES}/pairspec_pairs.csv')
name = {'norm': 'norm_r', 'norm_nr': 'norm', 'smooth': 'smooth_r', 'smooth_nr': 'smooth'}
rows = []
for inp, m in name.items():
    for s in S:
        got = rep.per_section[(rep.per_section.section == str(s))].loc[inp]
        exp = ref[(ref['sample'] == s) & (ref.method == m)].iloc[0]
        for k in ('depth_spot', 'splithalf', 'down25', 'unit_splithalf'):
            rows.append(dict(input=inp, section=s, metric=k, audit=got[k], stored=exp[k]))
    e = adj[(adj.pair == '151673-151674') & (adj.method == m)]
    g = rep.per_pair[rep.per_pair.input == inp].iloc[0]
    rows.append(dict(input=inp, section='pair', metric='serial', audit=g.serial, stored=e.adjacent.iloc[0] if len(e) else np.nan))
    if inp in ('norm', 'smooth'):
        e = ps[(ps.pair == '151673-151674') & (ps.input == {'norm': 'norm', 'smooth': 'smooth_r'}[inp])]
        for z in ('z_both', 'z_pair'):
            rows.append(dict(input=inp, section='pair', metric=f'lrspec_{z[2:]}', audit=g[f'lrspec_{z[2:]}'],
                             stored=e[e.null == z].rho.iloc[0] if len(e) else np.nan))
T = pd.DataFrame(rows); T['diff'] = (T.audit - T.stored).abs()
print(T.round(4).to_string()); print('max |diff|:', T['diff'].max())
assert T['diff'].max() < 1e-6, 'audit does not reproduce stored values'
print('OK')
