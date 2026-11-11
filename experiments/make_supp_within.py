"""Supplementary Table S10 (paper/sections/supp_within.tex) from results[/abca1]/within_layer.csv (within_layer.py)."""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
NM = [('counts', 'Raw counts'), ('norm', 'Normalised'), ('smooth', 'Normalised + smoothing'), ('liana_cosine', 'LIANA+ cosine'),
      ('commot', 'COMMOT'), ('liana_morans', 'LIANA+ local Moran'), ('commot_r', 'COMMOT + residualised'),
      ('norm_r', 'Normalised + residualised'), ('smooth_r', 'Smoothing + residualised'), ('magic', 'MAGIC + residualised'),
      ('scvi', 'scVI + residualised'), ('model', 'stCCC')]
D = pd.read_csv(f'{RES}/within_layer.csv').groupby('method').mean(numeric_only=True)
A = pd.read_csv(f'{RES}/abca1/within_layer.csv')
na = A['sample'].nunique(); A = A.groupby('method').mean(numeric_only=True)
H = pd.read_csv(f'{RES}/her2/within_layer.csv')
nh = H['sample'].nunique(); H = H.groupby('method').mean(numeric_only=True)
f = lambda v: f'{v:.2f}'.replace('-', '$-$')
L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
     r' & \multicolumn{2}{c}{DLPFC (12 sections)} & \multicolumn{3}{c}{MERFISH (%d sections)} & \multicolumn{2}{c}{Breast cancer (%d sections)} \\' % (na, nh),
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}',
     r'Score & All & \shortstack{Within\\layer} & All & \shortstack{Within\\layer} & \shortstack{Within layer\\$\times$ subclass} & All & \shortstack{Within\\region} \\', r'\midrule']
for k, n in NM:
    if k == 'liana_morans':
        L.append(r'\midrule')
    L.append(f'{n} & {f(D.loc[k, "rho_all"])} & {f(D.loc[k, "rho_within_layer"])} & {f(A.loc[k, "rho_all"])} & '
             f'{f(A.loc[k, "rho_within_layer"])} & {f(A.loc[k, "rho_within_layer_subclass"])} & {f(H.loc[k, "rho_all"])} & {f(H.loc[k, "rho_within_layer"])} \\\\')
L += [r'\midrule', f'Layer $R^2$ of log library size & \\multicolumn{{2}}{{c}}{{{D.r2_layer_lib.mean():.2f}}} & \\multicolumn{{3}}{{c}}{{{A.r2_layer_lib.mean():.2f}}} & \\multicolumn{{2}}{{c}}{{{H.r2_layer_lib.mean():.2f}}} \\\\',
      r'\bottomrule', r'\end{tabular}']
open(os.path.join(HERE, '..', 'paper', 'sections', 'supp_within.tex'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
