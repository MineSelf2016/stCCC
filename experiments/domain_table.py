"""Table 2: domain recovery of stCCC (results/repr.jsonl, cond lr) and the baselines (results/baselines.jsonl).
For each baseline the read-out (raw or standardised embedding) with the higher 12-section mean NMI is used for all sections.
Writes results/domain_table.csv and paper/sections/tab_domain.tex."""
import os, json, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
r = pd.DataFrame([json.loads(l) for l in open(f'{R}/repr.jsonl')]); r = r[r.cond == 'lr']
cols = {'stCCC': r.groupby('sample')[['NMI', 'ARI']].agg(['mean', 'std'])}
seedmean = {'stCCC': r.groupby('seed')[['NMI', 'ARI']].mean()}   # 12-section mean of each training seed
b = pd.DataFrame([json.loads(l) for l in open(f'{R}/baselines.jsonl')])
names = {'graphst': 'GraphST', 'stagate': 'STAGATE', 'nichecompass': 'NicheCompass'}
for m, nm in names.items():
    d = b[b.method == m]
    if d.empty: continue
    tag = max(['raw', 'std'], key=lambda t: d[f'NMI_{t}'].mean())
    d = d.rename(columns={f'NMI_{tag}': 'NMI', f'ARI_{tag}': 'ARI'}); print(nm, 'read-out', tag, 'n =', len(d))
    cols[nm] = d.groupby('sample')[['NMI', 'ARI']].agg(['mean', 'std'])
    assert (d.groupby('seed')['sample'].nunique() == d['sample'].nunique()).all()
    seedmean[nm] = d.groupby('seed')[['NMI', 'ARI']].mean()
T = pd.concat(cols, axis=1); T.index = [s.replace('sample_', '') for s in T.index]
T.to_csv(f'{R}/domain_table.csv')
meth = list(cols); lines = []
for s, row in T.iterrows():
    best = {k: max(meth, key=lambda m: row[(m, k, 'mean')]) for k in ['NMI', 'ARI']}
    cells = []
    for m in meth:
        for k in ['NMI', 'ARI']:
            v = f"{row[(m, k, 'mean')]:.3f}"; v = f'\\textbf{{{v}}}' if best[k] == m else v
            cells.append(f"{v}$\\pm${row[(m, k, 'std')]:.3f}")
    lines.append(f"{s} & " + ' & '.join(cells) + r' \\')
mean = T.mean()
lines.append(r'\midrule')
lines.append('Mean & ' + ' & '.join(f"{mean[(m, k, 'mean')]:.3f}$\\pm${seedmean[m][k].std():.3f}" for m in meth for k in ['NMI', 'ARI']) + r' \\')
head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{m}}}' for m in meth)
cm = ' '.join(f'\\cmidrule(lr){{{2+2*i}-{3+2*i}}}' for i in range(len(meth)))
tex = ('\\begin{tabular}{l' + 'cc' * len(meth) + '}\n\\toprule\n & ' + head + ' \\\\\n' + cm + '\nSection & ' +
       ' & '.join(['NMI & ARI'] * len(meth)) + ' \\\\\n\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
open(os.path.join(R, '..', 'paper', 'sections', 'tab_domain.tex'), 'w').write(tex)
print(T.xs('mean', axis=1, level=2).round(3).to_string()); print(mean.xs('mean', level=2).round(3))
sd = {m: T[(m, 'NMI', 'std')].median() for m in meth}; print('median seed sd NMI', sd)
