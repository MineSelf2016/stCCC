"""Supplementary Table S11 (paper/sections/supp_mclust.tex): domain recovery under K-means (Table 1) and mclust
(mclust_readout.py), means over 12 sections (each averaged over training seeds) and over the eight test sections."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
K = pd.read_csv(f'{RES}/domain_table.csv', header=[0, 1, 2], index_col=0)
K = K[[i.isdigit() for i in K.index.astype(str)]]; K.index = K.index.astype(int)
M = pd.read_csv(f'{RES}/mclust_readout.csv')
fails = M.groupby('method').failed.sum(); runs = M.groupby('method').size()
M = M.groupby(['method', 'sample'])[['NMI', 'ARI']].mean()
TEST = [s for s in K.index if s not in (151507, 151508, 151509, 151510)]
NAMES = [('stccc', 'stCCC'), ('stagate', 'STAGATE'), ('graphst', 'GraphST'), ('nichecompass', 'NicheCompass')]
L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
     r' & \multicolumn{3}{c}{$K$-means} & \multicolumn{3}{c}{mclust (EEE)} & \\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'Method & NMI (12) & ARI (12) & NMI (test 8) & NMI (12) & ARI (12) & NMI (test 8) & Failed fits \\', r'\midrule']
for k, n in NAMES:
    km = K[n]; mc = M.loc[k]
    L.append(f'{n} & {km["NMI"]["mean"].mean():.3f} & {km["ARI"]["mean"].mean():.3f} & {km["NMI"]["mean"].loc[TEST].mean():.3f} & '
             f'{mc.NMI.mean():.3f} & {mc.ARI.mean():.3f} & {mc.NMI.loc[TEST].mean():.3f} & {int(fails[k])}/{int(runs[k])} \\\\')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(HERE, '..', 'paper', 'sections', 'supp_mclust.tex'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
