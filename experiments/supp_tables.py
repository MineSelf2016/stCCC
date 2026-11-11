"""Supplementary tables for the BIB manuscript, generated from results/. Writes paper/sections/supp_*.tex.
  supp_data          per-section data summary
  supp_sensitivity   lambda x gamma on the development donor (results/sensitivity.jsonl)
  supp_ablation      model variants: domain recovery (results/ablation.jsonl) and reliability (results/reliability/per_section.csv)
  supp_reliability   per-section depth dependence and split-half agreement (results/reliability/per_section.csv)
  supp_adjacent      serial-section agreement per section pair (results/reliability/adjacent.csv)
  supp_interface     all donor-level discoveries of the interface scan (results/interface/combined_*.csv)
  supp_readout       baseline read-outs (raw vs standardised embedding) and run time (results/baselines.jsonl)"""
import os, json, sys
import numpy as np, pandas as pd, scanpy as sc
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results'); OUT = os.path.join(HERE, '..', 'paper', 'sections')
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
SECTIONS = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
DONOR = {s: 'Br5292' if s < 151600 else ('Br5595' if s < 151673 else 'Br8100') for s in SECTIONS}
NAME = {'counts': 'Raw counts', 'norm': 'Normalised', 'smooth': 'Normalised + smoothing', 'liana_cosine': 'LIANA+ cosine',
        'liana_morans': 'LIANA+ local Moran', 'norm_r': 'Normalised + residualised', 'model': 'stCCC'}
jl = lambda f: pd.DataFrame([json.loads(l) for l in open(f'{R}/{f}')])


def write(name, body):
    open(f'{OUT}/{name}.tex', 'w').write(body); print('wrote', name)


def tabular(df, fmt, align=None):
    cols = list(df.columns); align = align or 'l' + 'r' * (len(cols) - 1)
    rows = [' & '.join(cols) + r' \\', r'\midrule']
    for _, r in df.iterrows():
        rows.append(' & '.join(fmt.get(c, str)(r[c]) for c in cols) + r' \\')
    return '\\begin{tabular}{' + align + '}\n\\toprule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'


f3 = lambda v: '--' if pd.isnull(v) else f'{v:.3f}'


def supp_data():
    rep = jl('repr.jsonl'); npairs = rep.groupby('sample').n_pairs.first()
    rows = []
    for s in SECTIONS:
        ad = sc.read_h5ad(f'{DATA}/sample_{s}_with_graph.h5ad'); X = ad.X
        lib = np.asarray(X.sum(1)).ravel(); y = np.array([str(v) for v in np.load(f'{DATA}/sample_{s}_labels.npy', allow_pickle=True)])
        lay = sorted({v for v in y if v not in ('NA', 'nan', 'None')})
        rows.append({'Section': str(s), 'Donor': DONOR[s], 'Spots': f'{ad.n_obs:,}', 'Annotated': f'{int(np.isin(y, lay).sum()):,}',
                     'Genes': f'{ad.n_vars:,}', 'LR pairs': str(int(npairs[f'sample_{s}'])), 'Layers': str(len(lay)),
                     'Median library': f'{int(np.median(lib)):,}'})
    write('supp_data', tabular(pd.DataFrame(rows), {}, 'llrrrrrr'))


def supp_sensitivity():
    d = jl('sensitivity.jsonl'); d = d[d.cond == 'lr']
    g = d.groupby(['lam', 'gamma'])[['NMI', 'ARI']].mean().reset_index()
    sd = d.groupby(['lam', 'gamma', 'sample']).NMI.mean().groupby(['lam', 'gamma']).std().values
    g['Section SD (NMI)'] = sd
    g = g.rename(columns={'lam': r'$\lambda$', 'gamma': r'$\gamma$'})
    write('supp_sensitivity', tabular(g, {r'$\lambda$': lambda v: f'{v:g}', r'$\gamma$': lambda v: f'{v:g}', 'NMI': f3, 'ARI': f3, 'Section SD (NMI)': f3}))


def supp_ablation():
    d = jl('ablation.jsonl'); d['s'] = d['sample'].str[-6:].astype(int)
    lab = {'noCCC': 'GP-VAE backbone (no attention, no communication)', 'attn': '+ neighbour attention (no LR bias, no consistency)',
           'lr': '+ LR bias and communication consistency (stCCC)',
           'randlr': 'stCCC, LR pairing shuffled (degree-preserving)', 'lr_raw': 'stCCC, LR score not depth-residualised',
           'depth': 'stCCC, depth-only affinity in place of LR'}
    ps = pd.read_csv(f'{R}/reliability/per_section.csv'); rel = ps.groupby('method')[['splithalf', 'down25']].mean()
    rmap = {'noCCC': 'noCCC', 'attn': 'attn', 'lr': 'model'}
    rows = []
    for c in ['noCCC', 'attn', 'lr', 'randlr', 'lr_raw', 'depth']:
        x = d[d.cond == c]; t = x[x.s > 151600]; rr = rel.loc[rmap[c]] if c in rmap else {'splithalf': np.nan, 'down25': np.nan}
        rows.append({'Variant': lab[c], 'NMI (12)': x.NMI.mean(), 'ARI (12)': x.ARI.mean(), 'NMI (test 8)': t.NMI.mean(),
                     'Split-half': rr['splithalf'], 'Down-25\\%': rr['down25']})
    write('supp_ablation', tabular(pd.DataFrame(rows), {k: f3 for k in ['NMI (12)', 'ARI (12)', 'NMI (test 8)', 'Split-half', 'Down-25\\%']},
                                   'lrrrrr'))


SHORT = {'counts': 'Raw\\\\counts', 'norm': 'Normalised', 'smooth': 'Normalised\\\\+ smoothing', 'liana_cosine': 'LIANA+\\\\cosine',
         'liana_morans': 'LIANA+\\\\local Moran', 'norm_r': 'Normalised +\\\\residualised', 'model': 'stCCC'}


def grouped(W, lead, fmt, bold=True):
    """Wide per-section table with the seven scores grouped into depth-confounded and depth-independent; with bold,
    the best (highest) depth-independent value per row is set in bold."""
    nl = len(lead); ms = list(NAME); BS = '\\'
    hdr = ' & '.join(lead + [BS + 'shortstack{' + SHORT[m] + '}' for m in ms]) + ' ' + BS * 2
    top = ' & ' * nl + BS + 'multicolumn{4}{c}{Depth-confounded} & ' + BS + 'multicolumn{3}{c}{Depth-independent} ' + BS * 2
    rule = (BS + 'cmidrule(lr){%d-%d}') % (nl + 1, nl + 4) + (BS + 'cmidrule(lr){%d-%d}') % (nl + 5, nl + 7)
    rows = []
    for _, r in W.iterrows():
        best = max(r[m] for m in ms[4:])
        cells = [str(r[c]) for c in lead] + [(BS + 'textbf{%s}' if (bold and m in ms[4:] and abs(r[m] - best) < 5e-4) else '%s') % fmt(r[m]) for m in ms]
        rows.append(' & '.join(cells) + ' ' + BS * 2)
    mean = ['Mean'] + [''] * (nl - 1) + [fmt(W[m].mean()) for m in ms]
    return (BS + 'begin{tabular}{' + 'l' * nl + 'r' * 7 + '}\n' + BS + 'toprule\n' + top + '\n' + rule + '\n' + hdr + '\n' + BS + 'midrule\n'
            + '\n'.join(rows) + '\n' + BS + 'midrule\n' + ' & '.join(mean) + ' ' + BS * 2 + '\n' + BS + 'bottomrule\n' + BS + 'end{tabular}\n')


def supp_reliability():
    ps = pd.read_csv(f'{R}/reliability/per_section.csv'); ps = ps[ps.method.isin(NAME)]
    for col, nm in [('depth_spot', 'supp_rel_depth'), ('splithalf', 'supp_rel_split')]:
        W = ps.pivot(index='sample', columns='method', values=col)[list(NAME)].reset_index().rename(columns={'sample': 'Section'})
        W['Section'] = W['Section'].astype(str)
        write(nm, grouped(W, ['Section'], f3, bold=(col == 'splithalf')))


def supp_adjacent():
    a = pd.read_csv(f'{R}/reliability/adjacent.csv'); a = a[a.method.isin(NAME)]
    W = a.pivot(index='pair', columns='method', values='adjacent')[list(NAME)]
    W.insert(0, 'LR pairs', a.groupby('pair').n_pairs.first().astype(int).astype(str)); W = W.reset_index().rename(columns={'pair': 'Sections'})
    W['Sections'] = W['Sections'].str.replace('-', '--')
    write('supp_adjacent', grouped(W, ['Sections', 'LR pairs'], f3))


def supp_interface():
    out = []
    for inp, nm in [('model', 'stCCC'), ('norm', 'Normalised + residualised')]:
        d = pd.read_csv(f'{R}/interface/combined_{inp}.csv'); d = d[d.q < 0.05].copy()
        d['LR pair'] = d.hyp.str.split(' @ ').str[0].str.replace('-', '--'); d['Boundary'] = d.hyp.str.split(' @ ').str[1].str.replace('Layer', 'L')
        d['Input'] = nm
        d['all_donors_pos'] = d.min_donor_z > 0   # positive in every donor in which the boundary is annotated
        out.append(d[['Input', 'LR pair', 'Boundary', 'donor_z', 'min_donor_z', 'q', 'all_donors_pos']])
    D = pd.concat(out).rename(columns={'donor_z': 'Donor $z$', 'min_donor_z': 'Min', 'q': '$q$', 'all_donors_pos': 'All donors'})
    fq = lambda v: f'{v:.1e}' if v < 1e-3 else f'{v:.3f}'
    write('supp_interface', tabular(D, {'Donor $z$': lambda v: f'{v:.2f}', 'Min': lambda v: f'{v:.2f}', '$q$': fq,
                                        'All donors': lambda v: 'yes' if v else 'no'}, 'lllrrrc'))


def supp_readout():
    b = jl('baselines.jsonl'); g = jl('baselines_graphst_rawemb.jsonl').assign(method='graphst_raw')
    a = jl('ablation.jsonl'); a = a[a.cond == 'lr']
    nm = {'graphst': 'GraphST (PCA-20 of emb)', 'graphst_raw': 'GraphST (full emb output)', 'stagate': 'STAGATE', 'nichecompass': 'NicheCompass'}
    rows = [{'Method': 'stCCC', 'Dim.': '10', 'NMI raw': a.NMI.mean(), 'NMI std.': np.nan, 'ARI raw': a.ARI.mean(), 'ARI std.': np.nan,
             'Minutes': a.secs.mean() / 60}]
    for k in ['graphst', 'graphst_raw', 'stagate', 'nichecompass']:
        x = pd.concat([b, g])[lambda d: d.method == k]
        rows.append({'Method': nm[k], 'Dim.': str(int(x.dim.median())), 'NMI raw': x.NMI_raw.mean(), 'NMI std.': x.NMI_std.mean(),
                     'ARI raw': x.ARI_raw.mean(), 'ARI std.': x.ARI_std.mean(), 'Minutes': x.secs.mean() / 60})
    write('supp_readout', tabular(pd.DataFrame(rows), {**{k: f3 for k in ['NMI raw', 'NMI std.', 'ARI raw', 'ARI std.']},
                                                       'Minutes': lambda v: f'{v:.1f}'}))


NAME2 = {**NAME, 'commot': 'COMMOT', 'commot_r': 'COMMOT + residualised', 'smooth_r': 'Smoothing + residualised',
         'magic': 'MAGIC + residualised', 'scvi': 'scVI + residualised'}


def supp_mainrel():
    """Main-text Table: depth dependence, reliability and LR-specific agreement on both datasets
    (results/nar_summary.csv and results/abca1/nar_summary.csv, written by nar_summary.py)."""
    cols = ['depth', 'expl', 'splithalf', 'down25', 'adjacent', 'unit_splithalf', 'lrspec']
    groups = [('Depth-confounded', ['counts', 'norm', 'smooth', 'liana_cosine', 'commot']),
              ('Depth-independent', ['liana_morans', 'commot_r', 'norm_r']),
              ('Denoised, then depth-residualised', ['smooth_r', 'magic', 'scvi']),
              (None, ['model'])]
    lines = []
    for ds, title in [('', 'Human DLPFC, Visium (12 sections, 6 serial pairs)'), ('abca1/', 'Mouse isocortex, MERFISH (24 sections, 12 serial pairs)'),
                      ('her2/', 'HER2+ breast cancer, Spatial Transcriptomics (16 sections, 8 serial pairs)')]:
        f = pd.read_csv(f'{R}/{ds}nar_summary.csv').set_index('method')
        indep = [m for g, ms in groups[1:] for m in ms]
        best = {c: f.loc[indep, c].max() for c in cols[2:]}
        lines.append(r'\multicolumn{8}{l}{\textbf{' + title + r'}} \\')
        for g, ms in groups:
            if g:
                lines.append(r'\multicolumn{8}{l}{\quad\textit{' + g + r'}} \\')
            for m in ms:
                v = f.loc[m]
                cells = [f'{v[c]:.3f}' if abs(v[c]) < 0.01 else f'{v[c]:.2f}' for c in cols[:2]]
                cells += ['--' if pd.isnull(v[c]) else ((r'\textbf{%.2f}' if m in indep and abs(v[c] - best[c]) < 5e-4 else '%.2f') % v[c]) for c in cols[2:]]
                lines.append((r'\quad\quad ' if g else r'\quad ') + NAME2[m] + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\addlinespace')
    body = ('\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Depth dependence} & \\multicolumn{5}{c}{Agreement} \\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\n'
            'Score & $\\rho$ spot & Pattern & Split-half & Down-25\\% & Serial & Spot split-half & LR-specific serial \\\\\n\\midrule\n'
            + '\n'.join(lines[:-1]) + '\n\\bottomrule\n\\end{tabular}\n')
    write('tab_reliability', body)


def supp_kl():
    """KL pressure of stCCC on the development donor (results/selection/selection_summary.csv, select_kl_summary.py)."""
    T = pd.read_csv(f'{R}/selection/selection_summary.csv').set_index('conf')
    rows = []
    for c, r in T.iterrows():
        kl, mb, ib = r.kl.split(':')
        rows.append({'KL target': kl + (' (used)' if c == 'C0' else ''), r'Min.\ $\beta$': mb, 'AUROC program': r.auroc_prog, 'AUROC gene': r.auroc_gene,
                     'FPR null': r.fpr_null, 'Split-half': r.split, 'Down-25\\%': r.down, 'Spot split-half': r.unit,
                     'LR-specific serial': r.lrspec, 'NMI': r.nmi})
    write('supp_kl', tabular(pd.DataFrame(rows), {k: f3 for k in ['AUROC program', 'AUROC gene', 'FPR null', 'Split-half', 'Down-25\\%',
                                                                   'Spot split-half', 'LR-specific serial', 'NMI']}, 'llrrrrrrrr'))


if __name__ == '__main__':
    for w in (sys.argv[1:] or ['data', 'sensitivity', 'ablation', 'reliability', 'adjacent', 'interface', 'readout']):
        globals()[f'supp_{w}']()
