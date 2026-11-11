"""Summarise the semi-synthetic benchmark (results/simulation/task_*/scan.csv).
Non-planted hypotheses that involve any gene changed by planting are dropped (the planted genes themselves in
'gene' mode, also their co-expression programs in 'program' mode), because their scores change legitimately. On the rest, per task and input:
  auroc : planted vs non-planted, ranked by matched-null z  (tasks with f > 1)
  tpr   : share of planted hypotheses with z > 1.645 (one-sided 5%)
  fpr   : share of non-planted hypotheses with z > 1.645 (nominal 0.05)
Writes results/simulation/summary.csv (mean and sd over the three seeds)."""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from datasets import RESULTS
R = os.path.join(RESULTS, 'simulation')
rows = []
for d in sorted(glob.glob(f'{R}/task_*/scan.csv')):
    t = json.load(open(os.path.join(os.path.dirname(d), 'truth.json')))
    g = set(t.get('boosted') or [x for l, r, _ in t['truth'] for x in (l, r)])   # every gene planting changed
    df = pd.read_csv(d)
    df = df[df.planted | ~(df.ligand.isin(g) | df.receptor.isin(g))]
    for inp, x in df.groupby('input'):
        pos, neg = x[x.planted], x[~x.planted]
        rows.append(dict(mode=t.get('mode', 'gene'), width=t.get('width', 1), f=t['f'], depth=t['depth'], seed=t['seed'], input=inp,
                         auroc=roc_auc_score(x.planted, x.z) if t['f'] > 1 and len(pos) else np.nan,
                         tpr=(pos.z > 1.645).mean() if t['f'] > 1 and len(pos) else np.nan,
                         fpr=(neg.z > 1.645).mean(), n_pos=len(pos), n_neg=len(neg)))
T = pd.DataFrame(rows)
S = T.groupby(['mode', 'width', 'depth', 'f', 'input'])[['auroc', 'tpr', 'fpr']].agg(['mean', 'std']).round(3)
S.to_csv(f'{R}/summary.csv'); T.to_csv(f'{R}/per_task.csv', index=False)
order = ['norm_nr', 'norm', 'smooth_r', 'magic', 'scvi', 'model']
for m in ('auroc', 'tpr', 'fpr'):
    print(f'\n== {m} (mean over seeds)')
    print(T.pivot_table(index=['mode', 'width', 'depth', 'f'], columns='input', values=m)[[c for c in order if c in T.input.unique()]].round(3).to_string())
