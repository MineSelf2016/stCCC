"""Figures for the BIB manuscript that read spatial data, read from results/ (no retraining):
  fig_depthmap         spatial maps of library size and spot-level communication totals on one section (results/reliability)
  fig_laminar          de-confounded laminar communication patterns (results/reliability) -> results/laminar_patterns.csv
  fig_domain           domain maps and per-section NMI (results/latent, results/baseline_emb, results/ablation.jsonl, results/baselines.jsonl)
  supp_fig_domain_all  stCCC domain maps on all 12 sections (written by fig_domain)
  supp_fig_depthmap_*  fig_depthmap on one section of each other donor (fig_suppdepth)
Helpers (load, edge_scores, matched, cluster, laminar_table) are shared with make_figures.py.
Usage: python make_figures_extra.py [depthmap laminar domain suppdepth]"""
import os, sys, json, pickle, glob, warnings
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI
from figstyle import *
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); OUT = os.path.join(HERE, '..', 'paper', 'figures')
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
SEC = 151673
SECTIONS = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
DONOR = lambda s: 'Br5292' if s < 151600 else ('Br5595' if s < 151673 else 'Br8100')
TOP = [('NPTX2', 'NPTXR', 'Layer5/Layer6'), ('NPTX1', 'NPTXR', 'Layer5/Layer6'), ('CALM3', 'PDE1A', 'Layer5/Layer6'),
       ('CALM1', 'PDE1A', 'Layer5/Layer6'), ('NPTX2', 'NPTXR', 'Layer2/Layer3'), ('NPTX1', 'NPTXR', 'Layer6/WM'),
       ('LGI3', 'STX1A', 'Layer3/Layer4'), ('LGI3', 'STX1A', 'Layer4/Layer5'), ('RIMS1', 'SLC17A7', 'Layer3/Layer4'),
       ('CD99', 'CD99L2', 'Layer4/Layer5')]


def load(s):
    ad = sc.read_h5ad(f'{DATA}/sample_{s}_with_graph.h5ad')
    X = ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)
    y = np.array([str(v) for v in np.load(f'{DATA}/sample_{s}_labels.npy', allow_pickle=True)])
    return np.rint(X), list(ad.var_names), np.asarray(ad.obsm['spatial'], 'float64'), y


def edge_scores(s, lig, rec):
    """Depth-residualised edge score of one LR pair on the 6-NN Visium graph, averaged per spot, for stCCC and normalised input."""
    X, genes, xy, y = load(s); gi = {}
    for i, g in enumerate(genes): gi.setdefault(g, i)
    lib = X.sum(1); per10k = lambda E: np.log1p(E / E.sum(1, keepdims=True) * 1e4)
    _, idx = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy)
    ei = np.repeat(np.arange(len(xy)), 6); ej = idx[:, 1:].ravel(); k = ei < ej; ei, ej = ei[k], ej[k]
    ll = np.log(lib + 1.0); D = np.stack([np.ones(len(ei)), ll[ei] + ll[ej], np.abs(ll[ei] - ll[ej])], 1); Dp = np.linalg.pinv(D)
    out = {}
    for name, E in [('model', per10k(np.load(f'{RES}/denoised/sample_{s}.npy').astype('float64'))), ('norm', per10k(X))]:
        L, R = E[:, gi[lig]], E[:, gi[rec]]
        yv = np.log1p(L[ei] * R[ej] + L[ej] * R[ei]); yv = yv - D @ (Dp @ yv)
        sv = np.zeros(len(xy)); n = np.zeros(len(xy)); np.add.at(sv, ei, yv); np.add.at(sv, ej, yv); np.add.at(n, ei, 1); np.add.at(n, ej, 1)
        out[name] = sv / np.maximum(n, 1)
        out[name + '_edge'] = yv
    return xy, y, ei, ej, out


def matched(lab, ok, p):
    """Relabel clusters to layers by maximum overlap (Hungarian) for display."""
    L = [l for l in LAYERS if (lab[ok] == l).any()]; C = np.unique(p)
    M = np.array([[np.sum((p[ok] == c) & (lab[ok] == l)) for l in L] for c in C])
    ri, ci = linear_sum_assignment(-M); mp = {C[a]: L[b] for a, b in zip(ri, ci)}
    return np.array([mp.get(c, 'other') for c in p])


def cluster(Z, K, std):
    E = StandardScaler().fit_transform(Z) if std else Z
    return KMeans(K, n_init=10, random_state=0).fit_predict(E)


# ---------------------------------------------------------------- Figure 2: depth maps
def fig_depthmap(sec=SEC, name='fig_depthmap'):
    X, genes, xy, y = load(sec)
    lib = np.log(X.sum(1) + 1)
    o = pickle.load(open(f'{RES}/reliability/sample_{sec}.pkl', 'rb'))['out']['full']
    tot = {m: o[m]['total'].sum(1) for m in ['norm', 'norm_r', 'model']}
    fig = plt.figure(figsize=(TEXTW, 3.9))
    mw = 0.17; xs0 = [0.015, 0.215, 0.41, 0.605, 0.80]; y0, mh = 0.6, 0.31
    lays = [l for l in LAYERS if (y == l).any()]; short = lambda l: 'WM' if l == 'WM' else l.replace('Layer', 'L')
    ax = fig.add_axes([xs0[0], y0, mw, mh]); layermap(ax, xy, y); ax.set_title('Annotated layers', pad=11)
    letter(fig, 0.005, 0.99, 'A'); letter(fig, 0.2, 0.99, 'B')
    layer_legend(fig, xs0[0] + mw / 2, y0 - 0.02, lays, ncol=4, loc='upper center')
    specs = [('lib', lib, 'Log library size'), ('norm', tot['norm'], 'Normalised'),
             ('norm_r', tot['norm_r'], 'Normalised + residualised'), ('model', tot['model'], 'stCCC')]
    for j, (k, v, t) in enumerate(specs):
        ax = fig.add_axes([xs0[j + 1], y0, mw, mh]); spotmap(ax, xy, pct(v), SEQ); ax.set_title(t, pad=11)
        if k != 'lib':
            ax.text(0.5, 1.01, f'$\\rho$ with depth = {spearmanr(v, lib)[0]:.2f}', transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=6.3, color=INK2)
    hbar(fig, fig.add_axes([0.60, y0 - 0.04, 0.12, 0.013]), SEQ, [0, 1], ['low', 'high'])
    fig.text(0.59, y0 - 0.033, 'Spot percentile within section', fontsize=6.3, color=INK2, ha='right', va='center')
    # (C) depth differs between layers
    ax = fig.add_axes([0.075, 0.1, 0.25, 0.32]); letter(fig, 0.005, 0.47, 'C')
    data = [lib[y == l] for l in lays]
    vp = ax.violinplot(data, positions=range(len(lays)), widths=0.8, showextrema=False)
    for b_, l in zip(vp['bodies'], lays): b_.set_facecolor(LCOL[l]); b_.set_edgecolor('none'); b_.set_alpha(0.85)
    for i, d in enumerate(data):
        q1, md, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK, lw=1.8, solid_capstyle='butt'); ax.scatter([i], [md], s=9, color='white', edgecolor=INK, lw=0.5, zorder=3)
    ax.set_xticks(range(len(lays))); ax.set_xticklabels([short(l) for l in lays]); ax.set_ylabel('Log library size')
    ax.set_title('Sequencing depth by layer', loc='left'); ax.grid(axis='y', color=RULE, lw=0.5); ax.set_axisbelow(True)
    # (D) spot score against depth: density + layer means
    letter(fig, 0.375, 0.47, 'D'); zs = lambda a: (a - a.mean()) / a.std()
    for j, k in enumerate(['norm', 'norm_r', 'model']):
        ax = fig.add_axes([0.42 + j * 0.195, 0.1, 0.165, 0.32]); v = zs(tot[k])
        cm = LinearSegmentedColormap.from_list('c', ['#ffffff', MCOL[k]])
        ax.hexbin(lib, v, gridsize=28, cmap=cm, mincnt=1, lw=0.1, edgecolors='white', extent=(lib.min(), lib.max(), -3.5, 3.5))
        mx = [lib[y == l].mean() for l in lays]; my = [v[y == l].mean() for l in lays]
        ax.plot(mx, my, color=INK2, lw=0.7, zorder=3)
        ax.scatter(mx, my, s=22, c=[LCOL[l] for l in lays], edgecolor=INK, lw=0.6, zorder=4)
        b1, b0 = np.polyfit(lib, v, 1); xx = np.array([lib.min(), lib.max()])
        ax.plot(xx, b0 + b1 * xx, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_ylim(-3.5, 3.5); ax.set_title(NAME[k], fontsize=7, pad=3)
        ax.text(0.04, 0.96, f'$\\rho$ = {spearmanr(tot[k], lib)[0]:.2f}', transform=ax.transAxes, va='top', fontsize=6.5,
                fontweight='bold' if k == 'model' else 'normal')
        ax.set_xlabel('Log library size')
        if j == 0: ax.set_ylabel('Spot total (z-score)')
        else: ax.set_yticklabels([])
    fig.text(0.42, 0.455, 'Spot-level total against depth; points: layer means (colours as in A); dashed: least-squares fit',
             fontsize=6.3, color=INK2)
    save(fig, OUT, name)
    print(name, {k: round(spearmanr(v, lib)[0], 3) for k, v in tot.items()})


def fig_suppdepth():
    for s in [151507, 151669]:
        fig_depthmap(s, f'supp_fig_depthmap_{s}')


# ---------------------------------------------------------------- Figure 4: laminar map
def laminar_table():
    """Per pair x section: centred laminar pattern (scaled to max |.| = 1) and its correlation with the depth profile."""
    recs = []
    for f in sorted(glob.glob(f'{RES}/reliability/sample_*.pkl')):
        r = pickle.load(open(f, 'rb')); s = int(f[-10:-4])
        X, genes, xy, y = load(s); lib = np.log(X.sum(1) + 1)
        lays = [l for l in LAYERS if l in r['layers']]
        dp = np.array([lib[y == l].mean() for l in lays]); dp -= dp.mean()
        for m in ['model', 'norm', 'norm_r']:
            LM = pd.DataFrame(r['out']['full'][m]['LM'], index=[f'{a}–{b}' for a, b in r['pairs']], columns=r['layers'])[lays]
            C = LM.sub(LM.mean(1), axis=0)
            for p, row in C.iterrows():
                v = row.values
                if np.isnan(v).any() or v.std() == 0: continue
                recs.append(dict(s=s, d=DONOR(s), m=m, p=p, cd=pearsonr(v, dp)[0], **dict(zip(lays, v / np.abs(v).max()))))
    return pd.DataFrame(recs)


def laminar_agreement(R):
    agree = {}
    for m in ['norm', 'norm_r', 'model']:
        cs = []
        for p, g in R[R.m == m].groupby('p'):
            D = g.groupby('d')[LAYERS].mean()
            for i in range(len(D)):
                for j in range(i + 1, len(D)):
                    a, b = D.iloc[i], D.iloc[j]; k = a.notna() & b.notna()
                    if k.sum() >= 4: cs.append(spearmanr(a[k], b[k])[0])
        agree[m] = np.array(cs)
    return agree


def depth_profiles():
    """Centred, max-scaled layer means of log library size per section."""
    out = {}
    for s in SECTIONS:
        X, genes, xy, y = load(s); lib = np.log(X.sum(1) + 1)
        v = np.array([lib[y == l].mean() if (y == l).any() else np.nan for l in LAYERS]); v = v - np.nanmean(v)
        out[s] = v / np.nanmax(np.abs(v))
    return out


def fig_laminar():
    from scipy.stats import wilcoxon, mannwhitneyu
    from sklearn.decomposition import PCA
    R = laminar_table(); R.to_csv(f'{RES}/laminar_patterns.csv', index=False)
    n = R[R.m == 'model'].groupby('p').size(); keep = n[n >= 8].index
    M = {m: R[(R.m == m) & R.p.isin(keep)].groupby('p')[LAYERS].mean() for m in ['model', 'norm']}
    order = sorted(keep, key=lambda p: (int(np.nanargmax(M['model'].loc[p].values)), -np.nanmax(M['model'].loc[p].values)))
    peak = [LAYERS[int(np.nanargmax(M['model'].loc[p].values))] for p in order]
    agree = laminar_agreement(R); DP = depth_profiles()
    short = lambda l: 'WM' if l == 'WM' else l.replace('Layer', 'L')
    fig = plt.figure(figsize=(TEXTW, 5.8))
    # (A) heatmaps
    top, bot = 0.95, 0.43
    gs = fig.add_gridspec(1, 3, wspace=0.05, width_ratios=[0.07, 1, 1], left=0.135, right=0.585, top=top, bottom=bot)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(np.array([[matplotlib.colors.to_rgb(LCOL[l])] for l in peak]), aspect='auto')
    ax0.set_yticks(range(len(order))); ax0.set_yticklabels(order, fontsize=6); ax0.set_xticks([]); ax0.tick_params(length=0)
    for sp in ax0.spines.values(): sp.set_visible(False)
    ax0.set_title('Peak', fontsize=6.3, color=INK2, pad=3)
    for j, m in enumerate(['model', 'norm']):
        ax = fig.add_subplot(gs[j + 1]); A = M[m].loc[order, LAYERS].values
        im = ax.imshow(np.ma.masked_invalid(A), cmap=DIV, vmin=-1, vmax=1, aspect='auto'); ax.set_facecolor(RULE)
        ax.set_xticks(range(7)); ax.set_xticklabels([short(l) for l in LAYERS])
        ax.set_yticks([]); ax.tick_params(length=0); ax.set_title(NAME[m], pad=3, fontweight='bold' if m == 'model' else 'normal')
        for sp in ax.spines.values(): sp.set_visible(False)
        for yy in np.where(np.array(peak[1:]) != np.array(peak[:-1]))[0]: ax.axhline(yy + 0.5, color='white', lw=1.2)
    letter(fig, 0.005, 0.995, 'A')
    hbar(fig, fig.add_axes([0.37, bot - 0.045, 0.14, 0.011]), DIV, [-1, 0, 1], ['−1', '0', '1'], vmin=-1, vmax=1)
    fig.text(0.36, bot - 0.04, 'Centred layer mean (scaled per pair and section)', fontsize=6.1, color=INK2, ha='right', va='center')
    # (B, C) rainclouds
    ms = ['model', 'norm_r', 'norm']; ys = [2, 1, 0]
    P = R.pivot_table(index=['s', 'p'], columns='m', values='cd').abs().dropna()
    for i, (vals, lab, ttl, lt, xl) in enumerate([({m: P[m].values for m in ms}, '|Pearson r| with depth profile', 'Depth dependence (pair × section)', 'B', (0, 1)),
                                                  (agree, 'Spearman agreement', 'Agreement between donors (pair × donor pair)', 'C', (-1, 1))]):
        ax = fig.add_axes([0.715, [0.735, 0.445][i], 0.255, 0.2]); letter(fig, 0.6, [0.995, 0.705][i], lt)
        raincloud(ax, [vals[m] for m in ms], ys, [MCOL[m] for m in ms], width=0.42, s=1.6, alpha=0.4)
        ax.set_yticks(np.array(ys) - 0.06); ax.set_yticklabels(['stCCC', 'Normalised +\nresidualised', 'Normalised'], fontsize=6.2)
        ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
        ax.set_xlabel(lab); ax.grid(axis='x', color=RULE, lw=0.5); ax.set_axisbelow(True); ax.set_xlim(*xl); ax.set_ylim(-1.25, 2.65)
        ax.set_title(ttl, loc='left', fontsize=7, pad=3, x=-0.42)
        for y_, m in zip(ys, ms):
            ax.text(1.01, y_ + 0.12, f'{np.median(vals[m]):.2f}', transform=ax.get_yaxis_transform(), fontsize=6.3, va='center',
                    color=MCOL[m] if m != 'norm' else INK2, fontweight='bold')
        pv = [wilcoxon(P['model'], P[o]).pvalue if lt == 'B' else mannwhitneyu(vals['model'], vals[o]).pvalue for o in ['norm_r', 'norm']]
        ax.text(0.99, 0.02, f'vs Norm. + resid.: {fmt_p(pv[0])}\nvs Normalised: {fmt_p(pv[1])}', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=5.6, color=INK2)
    fig.text(0.985, 0.955, 'median', fontsize=6, color=INK2, ha='right')
    # (D) example pairs: per-section profiles
    letter(fig, 0.005, 0.36, 'D')
    ex = [('COL3A1–MAG', 'white matter'), ('RIMS1–SLC17A7', 'L5–L6'), ('NPTX2–NPTXR', 'upper layers')]
    for j, (p, where) in enumerate(ex):
        ax = fig.add_axes([0.06 + j * 0.155, 0.075, 0.13, 0.24]); xs = np.arange(7)
        for m in ['norm', 'model']:
            sub = R[(R.m == m) & (R.p == p)]
            for _, row in sub.iterrows(): ax.plot(xs, row[LAYERS].values.astype(float), color=MCOL[m], lw=0.4, alpha=0.35)
            ax.plot(xs, sub[LAYERS].mean().values, color=MCOL[m], lw=1.8, marker='o', ms=2.8, mec='white', mew=0.4, label=NAME[m], zorder=4)
        dpm = np.nanmean(np.array([DP[s] for s in R[(R.m == 'model') & (R.p == p)].s]), 0)
        ax.plot(xs, dpm, color=INK, lw=1.0, ls=(0, (3, 1.5)), label='Depth profile', zorder=5)
        ax.set_xticks(xs); ax.set_xticklabels(['1', '2', '3', '4', '5', '6', 'W'], fontsize=6); ax.set_ylim(-1.15, 1.15)
        ax.axhline(0, color=GREY, lw=0.5); ax.set_title(p, fontsize=6.8, pad=2, style='italic')
        ax.set_xlabel('Layer', fontsize=6.3, labelpad=1)
        if j == 0: ax.set_ylabel('Centred, scaled pattern')
        else: ax.set_yticklabels([])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.27, 0.375), ncol=3, fontsize=6.2, handlelength=1.8)
    # (E) pattern space: PCA of pair x section patterns on fully annotated sections, coloured by correlation with depth
    letter(fig, 0.53, 0.36, 'E')
    for j, m in enumerate(['norm', 'model']):
        ax = fig.add_axes([0.6 + j * 0.2, 0.075, 0.16, 0.24])
        sub = R[(R.m == m)].dropna(subset=LAYERS); Z = PCA(2, random_state=0).fit(sub[LAYERS].values)
        U = Z.transform(sub[LAYERS].values); o = np.argsort(np.abs(sub.cd.values))
        sc_ = ax.scatter(U[o, 0], U[o, 1], c=sub.cd.values[o], cmap=DIV, vmin=-1, vmax=1, s=3, lw=0, rasterized=True)
        clean(ax); [ax.spines[k].set_visible(True) for k in ['left', 'bottom']]; [ax.spines[k].set_color(GREY) for k in ['left', 'bottom']]
        ev = Z.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({ev[0] * 100:.0f}%)', fontsize=6.2, color=INK2, labelpad=1); ax.set_ylabel(f'PC2 ({ev[1] * 100:.0f}%)', fontsize=6.2, color=INK2, labelpad=1)
        ax.set_title(NAME[m], fontsize=6.8, pad=2, fontweight='bold' if m == 'model' else 'normal')
    hbar(fig, fig.add_axes([0.8, 0.37, 0.12, 0.011]), DIV, [-1, 0, 1], ['−1', '0', '1'], vmin=-1, vmax=1)
    fig.text(0.79, 0.375, 'r with depth profile', fontsize=6.1, color=INK2, ha='right', va='center')
    save(fig, OUT, 'fig_laminar')
    print('pairs', len(order), {m: round(float(np.median(v)), 3) for m, v in agree.items()},
          {m: round(float(R[R.m == m].cd.abs().median()), 3) for m in ms})


# ---------------------------------------------------------------- Figure 6: domains
BASE = {'graphst': 'GraphST', 'stagate': 'STAGATE', 'nichecompass': 'NicheCompass'}
BSTY = {'stCCC': (BLUE, 'o', 5.0), 'GraphST': ('#6f6e69', 's', 4.0), 'STAGATE': (INK3, '^', 4.4), 'NicheCompass': ('#b3b2ac', 'D', 3.8)}


def per_section_nmi():
    abl = pd.DataFrame([json.loads(l) for l in open(f'{RES}/ablation.jsonl')])
    bl = pd.DataFrame([json.loads(l) for l in open(f'{RES}/baselines.jsonl')])
    per = [abl[abl.cond == 'lr'].assign(m='stCCC', s=lambda d: d['sample'].str[-6:].astype(int))[['m', 's', 'NMI']]]
    for k, v in BASE.items():
        per.append(bl[bl.method == k].assign(m=v, s=lambda d: d['sample'].str[-6:].astype(int), NMI=lambda d: d.NMI_std)[['m', 's', 'NMI']])
    return pd.concat(per).groupby(['m', 's']).NMI.mean().unstack(0)


def fig_domain():
    import umap
    from scipy.stats import wilcoxon
    per = per_section_nmi()
    fig = plt.figure(figsize=(TEXTW, 5.3))
    X, genes, xy, y = load(SEC); ok = np.isin(y, LAYERS); K = len(set(y[ok]))
    lays = [l for l in LAYERS if (y == l).any()]
    # (A) domain maps
    xs0 = [0.01, 0.21, 0.41, 0.61, 0.81]; y0, mh, mw = 0.69, 0.25, 0.18
    ax = fig.add_axes([xs0[0], y0, mw, mh]); layermap(ax, xy, y, 'Annotated layers\n '); letter(fig, 0.005, 0.995, 'A')
    z = np.load(f'{RES}/latent/sample_{SEC}_lr_seed0.npz')
    p = cluster(z['z'], K, False); xs_ = xy[z['keep']]; ys_ = y[z['keep']]; ok_ = np.isin(ys_, LAYERS)
    embs = [('stCCC', xs_, ys_, ok_, p, z['z'], False)]
    for k, v in BASE.items():
        Zb = np.load(f'{RES}/baseline_emb/{k}_sample_{SEC}_seed0.npy'); embs.append((v, xy, y, ok, cluster(Zb, K, True), Zb, True))
    for j, (t, xy_, y_, ok_, p, Z, std) in enumerate(embs):
        ax = fig.add_axes([xs0[j + 1], y0, mw, mh]); layermap(ax, xy_, matched(y_, ok_, p), f'{t}\nNMI = {NMI(y_[ok_], p[ok_]):.2f}')
        if t == 'stCCC': ax.title.set_fontweight('bold')
    layer_legend(fig, 0.5, y0 - 0.005, lays, loc='upper center')
    # (B) UMAP of each embedding, coloured by annotated layer
    letter(fig, 0.005, 0.635, 'B')
    for j, (t, xy_, y_, ok_, p, Z, std) in enumerate(embs):
        ax = fig.add_axes([0.035 + j * 0.245, 0.375, 0.2, 0.235])
        E = StandardScaler().fit_transform(Z) if std else Z
        U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(E[ok_])
        lab = y_[ok_]; o = np.random.default_rng(0).permutation(len(lab))
        ax.scatter(U[o, 0], U[o, 1], c=[LCOL[l] for l in lab[o]], s=0.6, lw=0, rasterized=True)
        clean(ax); ax.set_title(t, fontweight='bold' if t == 'stCCC' else 'normal', pad=2)
        for sp in ['left', 'bottom']: ax.spines[sp].set_visible(True); ax.spines[sp].set_color(GREY)
        ax.set_xlabel('UMAP 1', fontsize=6, color=INK3, labelpad=1); ax.set_ylabel('UMAP 2', fontsize=6, color=INK3, labelpad=1)
    # (C) per-section NMI
    ax = fig.add_axes([0.07, 0.075, 0.56, 0.22]); letter(fig, 0.005, 0.33, 'C')
    xs = np.arange(12)
    for i, m in enumerate(BSTY):
        c, mk, ms = BSTY[m]
        ax.plot(xs + (i - 1.5) * 0.15, per.loc[SECTIONS, m], ls='none', marker=mk, ms=ms, color=c, mec='white', mew=0.5, label=m,
                zorder=4 if m == 'stCCC' else 3)
    ax.set_xticks(xs); ax.set_xticklabels([str(s)[-3:] for s in SECTIONS], fontsize=6.2); ax.set_xlabel('Section (151xxx)')
    ax.set_ylabel('NMI'); ax.set_ylim(0.3, 0.82); ax.set_xlim(-0.6, 11.6)
    for x in [3.5, 7.5]: ax.axvline(x, color=RULE, lw=0.8)
    for x, d in zip([1.5, 5.5, 9.5], ['Br5292 (selection)', 'Br5595', 'Br8100']):
        ax.text(x, 0.815, d, ha='center', va='top', fontsize=6.2, color=INK2)
    ax.grid(axis='y', color=RULE, lw=0.5); ax.set_axisbelow(True)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    # (D) paired comparison over sections
    ax = fig.add_axes([0.72, 0.075, 0.27, 0.22]); letter(fig, 0.66, 0.33, 'D')
    ms = list(BSTY); pos = np.arange(len(ms))
    raincloud(ax, [per[m].values for m in ms], pos, [BSTY[m][0] for m in ms], horizontal=False, width=0.36, s=6, alpha=0.8)
    for i in range(12):
        ax.plot(pos - 0.24, [per[m].values[i] for m in ms], color=GREY, lw=0.3, alpha=0.6, zorder=1)
    ax.set_xticks(pos); ax.set_xticklabels(['stCCC', 'GraphST', 'STAGATE', 'Niche-\nCompass'], fontsize=6.2)
    ax.set_ylabel('NMI (12 sections)'); ax.set_ylim(0.3, 0.98); ax.grid(axis='y', color=RULE, lw=0.5); ax.set_axisbelow(True)
    for k, m in enumerate(ms[1:]):
        pval = wilcoxon(per['stCCC'].values, per[m].values).pvalue
        bracket(ax, 0, k + 1, 0.76 + k * 0.07, fmt_p(pval), h=0.015, fs=5.6)
    save(fig, OUT, 'fig_domain')
    # supplementary: stCCC maps on all 12 sections
    fig, axs = plt.subplots(4, 6, figsize=(TEXTW, 5.7)); plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.05, wspace=0.04, hspace=0.25)
    for i, s in enumerate(SECTIONS):
        X, genes, xy, y = load(s); z = np.load(f'{RES}/latent/sample_{s}_lr_seed0.npz'); xs_, ys_ = xy[z['keep']], y[z['keep']]
        ok_ = np.isin(ys_, LAYERS); p = cluster(z['z'], len(set(ys_[ok_])), False)
        a0, a1 = axs[(i // 3), (i % 3) * 2], axs[(i // 3), (i % 3) * 2 + 1]
        layermap(a0, xy, y, f'{s} ({DONOR(s)})', s=0.8); layermap(a1, xs_, matched(ys_, ok_, p), f'stCCC, NMI {NMI(ys_[ok_], p[ok_]):.2f}', s=0.8)
        a0.title.set_fontsize(6.5); a1.title.set_fontsize(6.5)
    layer_legend(fig, 0.5, 0.015, loc='lower center')
    save(fig, OUT, 'supp_fig_domain_all')
    print(per.mean().round(3).to_dict())


if __name__ == '__main__':
    for w in (sys.argv[1:] or ['depthmap', 'laminar', 'domain', 'suppdepth']):
        globals()[f'fig_{w}']()
