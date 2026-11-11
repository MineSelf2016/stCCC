"""LR-specific serial agreement of the neighbourhood-swap variants (nbr_swap.py) next to the original inputs.
Writes results[/d]/nbr_swap_pairs.csv and prints means and paired comparisons per dataset."""
import os, glob, importlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
SRC = {'model': 'interface', 'magic': 'interface_extra', 'magic_sp': 'interface_magic_sp',
       'model_xnbr': 'interface_model_xnbr', 'model_xall': 'interface_model_xall', 'noGate': 'interface_noGate'}


def run(name):
    os.environ['STCCC_DATASET'] = name
    import datasets; importlib.reload(datasets)
    rows = []
    for inp, d in SRC.items():
        fs = glob.glob(f'{datasets.RESULTS}/{d}/sample_*.csv')
        df = pd.concat([pd.read_csv(f) for f in fs]); df = df[df.input == inp]
        df['sec'] = df['sample'].str.split('_').str[-1].astype(int); df['hyp'] = df.ligand + '-' + df.receptor + ' @ ' + df.boundary
        W = df.pivot_table(index='hyp', columns='sec', values='z')
        for a, b in datasets.ADJ:
            rows.append(dict(input=inp, pair=f'{a}-{b}', rho=spearmanr(*W[[a, b]].dropna().T.values)[0] if {a, b} <= set(W.columns) else np.nan))
    P = pd.DataFrame(rows); P.to_csv(f'{datasets.RESULTS}/nbr_swap_pairs.csv', index=False)
    W = P.pivot(index='pair', columns='input', values='rho')
    print(f'== {name}  mean LR-specific serial agreement'); print(W.mean().round(3).to_string())
    for a, b in (('model_xnbr', 'model'), ('model_xall', 'model'), ('magic_sp', 'magic'), ('model_xall', 'magic'), ('magic_sp', 'model')):
        d = (W[a] - W[b]).dropna()
        print(f'  {a:10s} vs {b:6s}: higher {int((d > 0).sum())}/{len(d)}  diff {d.mean():+.3f}  two-sided P {wilcoxon(d).pvalue:.3f}')


if __name__ == '__main__':
    for n in ('dlpfc', 'abca1', 'her2'):
        run(n)
