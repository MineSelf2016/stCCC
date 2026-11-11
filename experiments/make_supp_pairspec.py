"""Supplementary Table S12: LR-specific serial agreement under the matched null (interface scan of the main text,
results[/d]/nar_iface_pairs.csv) and under the partner-swap null (iface_pairspec.py, results[/d]/pairspec_pairs.csv).
Writes paper/sections/supp_pairspec.tex.  Run iface_pairspec.py --summary dlpfc abca1 her2 first."""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results')
DS = [('', 'DLPFC'), ('abca1/', 'MERFISH'), ('her2/', 'Breast cancer')]
INP = [('norm', 'norm_r', 'Normalised + residualised'), ('smooth_r', 'smooth_r', 'Smoothing + residualised'),
       ('magic', 'magic', 'MAGIC + residualised'), ('scvi', 'scvi', 'scVI + residualised'), ('model', 'model', 'stCCC')]
M, P = {}, {}
for d, _ in DS:
    a = pd.read_csv(f'{R}/{d}nar_iface_pairs.csv'); M[d] = a.pivot(index='pair', columns='input', values='rho')
    b = pd.read_csv(f'{R}/{d}pairspec_pairs.csv'); P[d] = b[b.null == 'z_pair'].pivot(index='pair', columns='input', values='rho')
L = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
     ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{n}}}' for _, n in DS) + r' \\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
     'Input & ' + ' & '.join(['Matched', 'Partner-swap'] * 3) + r' \\', r'\midrule']
for ps, mm, lab in INP:
    v = []
    for d, _ in DS:
        v += [f'{M[d][mm].mean():.2f}', f'{P[d][ps].mean():.2f}']
    L.append(f'{lab} & ' + ' & '.join(v) + r' \\')
L.append(r'\midrule')
for b, lab in (('magic', 'MAGIC'), ('scvi', 'scVI')):
    v = []
    for d, _ in DS:
        v += [f"{int((M[d]['model'] > M[d][b]).sum())}/{len(M[d])}", f"{int((P[d]['model'] > P[d][b]).sum())}/{len(P[d])}"]
    L.append(f'stCCC higher than {lab} (pairs) & ' + ' & '.join(v) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(HERE, '..', 'paper', 'sections', 'supp_pairspec.tex'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
