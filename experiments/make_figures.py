"""Main-text figures for the BIB manuscript, read from results/ (no retraining):
  fig_overview     Figure 1: schematic of stCCC with thumbnails from the results
  fig_reliability  Figure 3: depth dependence and reliability (results/reliability/figure_summary.csv; `summary` rebuilds it)
  fig_interface    Figure 5: laminar-interface scan, replication, discoveries and the leading hypotheses (results/interface)
Usage: python make_figures.py [summary overview reliability interface]"""
import os, sys, glob, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, ConnectionPatch
from figstyle import *
from matplotlib.colors import LinearSegmentedColormap
from make_figures_extra import load, edge_scores, matched, cluster, TOP, SECTIONS, DONOR, SEC, RES, OUT, DATA
warnings.filterwarnings("ignore")
ADJ = [(151507, 151508), (151509, 151510), (151669, 151670), (151671, 151672), (151673, 151674), (151675, 151676)]


# ---------------------------------------------------------------- reliability summary (Table 1 / Figure 3)
def reliability_summary():
    import scanpy as sc
    D = {}
    for f in sorted(glob.glob(f'{RES}/reliability/sample_*.pkl')):
        s = int(os.path.basename(f)[7:13]); r = pickle.load(open(f, 'rb'))
        ex = pickle.load(open(f'{RES}/reliability_liana/sample_{s}.pkl', 'rb'))
        for v in r['out']: r['out'][v].update(ex['out'][v])
        D[s] = r
    cen = lambda LM: LM - np.nanmean(LM, 1, keepdims=True)
    corr = lambda a, b: spearmanr(a[~(np.isnan(a) | np.isnan(b))], b[~(np.isnan(a) | np.isnan(b))])[0]
    rows = []
    for s, r in D.items():
        ad = sc.read_h5ad(f'{DATA}/sample_{s}_with_graph.h5ad'); lib = np.log(np.asarray(ad.X.sum(1)).ravel() + 1)
        y = np.load(f'{DATA}/sample_{s}_labels.npy', allow_pickle=True).astype(str)
        ki = [i for i, l in enumerate(r['layers']) if l != 'NA']; lays = [r['layers'][i] for i in ki]
        dp = np.array([lib[y == l].mean() for l in lays]); dp -= dp.mean()
        for m in ['counts', 'norm', 'smooth', 'liana_cosine', 'liana_morans', 'norm_r', 'model']:
            o = r['out']; P = lambda v: cen(o[v][m]['LM'][:, ki])
            C = P('full'); C = C[~np.isnan(C).any(1)]; b = (C @ dp) / (dp @ dp)
            tot = o['full'][m]['total']; tot = tot.mean(1) if tot.ndim == 2 else tot
            rows.append(dict(sample=s, m=m, depth=spearmanr(tot, lib)[0], expl=1 - ((C - np.outer(b, dp)) ** 2).sum() / (C ** 2).sum(),
                             split=corr(P('A').ravel(), P('B').ravel()), down=corr(P('d25').ravel(), P('full').ravel()),
                             unit=corr(o['A'][m]['total'].ravel(), o['B'][m]['total'].ravel())))
    T = pd.DataFrame(rows); adj = []
    for a, b in ADJ:
        P_ = sorted(set(D[a]['pairs']) & set(D[b]['pairs'])); L_ = sorted((set(D[a]['layers']) & set(D[b]['layers'])) - {'NA'})
        for m in T.m.unique():
            f = lambda s: pd.DataFrame(cen(np.delete(D[s]['out']['full'][m]['LM'], [D[s]['layers'].index('NA')] if 'NA' in D[s]['layers'] else [], 1)),
                                       index=D[s]['pairs'], columns=[l for l in D[s]['layers'] if l != 'NA']).loc[P_, L_].values.ravel()
            adj.append(dict(m=m, adj=corr(f(a), f(b))))
    S = T.groupby('m').mean(numeric_only=True); S['adj'] = pd.DataFrame(adj).groupby('m').adj.mean()
    S.round(3).to_csv(f'{RES}/reliability/figure_summary.csv'); print(S.round(3))


def fig_summary():
    reliability_summary()


# ---------------------------------------------------------------- Figure 3
CONF = ['counts', 'norm', 'liana_cosine', 'smooth']; INDEP = ['liana_morans', 'norm_r', 'model']


def fig_reliability():
    from scipy.stats import wilcoxon
    S = pd.read_csv(f'{RES}/reliability/figure_summary.csv').set_index('m')
    PS = pd.read_csv(f'{RES}/reliability/per_section.csv')
    S['depth'] = PS.groupby('method').depth_spot.mean()
    col = lambda m: MCOL[m] if m in INDEP else '#9b9a95'
    fig = plt.figure(figsize=(TEXTW, 4.75))
    # (A) per-section depth dependence, raincloud
    ax = fig.add_axes([0.165, 0.585, 0.2, 0.34]); letter(fig, 0.005, 0.995, 'A')
    order = CONF + INDEP; ys = np.array([7.6, 6.6, 5.6, 4.6, 2.4, 1.4, 0.4])
    raincloud(ax, [PS[PS.method == m].depth_spot.values for m in order], ys, [col(m) for m in order], width=0.38, s=5, alpha=0.8)
    ax.set_yticks(ys - 0.06); ax.set_yticklabels([NAME[m] for m in order]); ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.axvline(0, color=INK2, lw=0.6); ax.set_xlim(-0.35, 1.0); ax.set_ylim(-0.1, 8.5)
    ax.text(-0.78, 8.3, 'Depth-confounded', fontsize=6.3, color=INK2, style='italic', va='center', transform=ax.get_yaxis_transform())
    ax.text(-0.78, 3.15, 'Depth-independent', fontsize=6.3, color=INK2, style='italic', va='center', transform=ax.get_yaxis_transform())
    ax.set_xlabel('Spearman $\\rho$ with log library size'); ax.set_title('Depth dependence per section', loc='left', x=-0.75)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0); ax.grid(axis='x', color=RULE, lw=0.5); ax.set_axisbelow(True)
    # (B) apparent reliability vs share explained by depth (bubble = spot-level split-half)
    ax = fig.add_axes([0.45, 0.585, 0.2, 0.34]); letter(fig, 0.395, 0.995, 'B')
    ax.axvspan(0.5, 0.82, color=GREY_T, lw=0, zorder=0)
    ax.text(0.66, 0.703, '$R^2$ > 0.5', ha='center', va='bottom', fontsize=5.8, color=INK2, style='italic')
    lab = {'counts': ('Raw counts', (-7, 0), 'right'), 'norm': ('Normalised', (7, -2), 'left'), 'smooth': ('Norm. + smoothing', (-8, 0), 'right'),
           'liana_cosine': ('LIANA+ cosine', (0, -10), 'center'), 'liana_morans': ('LIANA+ local Moran', (0, -9), 'center'),
           'norm_r': ('Normalised +\nresidualised', (8, 0), 'left'), 'model': ('stCCC', (9, 3), 'left')}
    for m in order:
        x, y = S.loc[m, 'expl'], S.loc[m, 'split']
        ax.scatter(x, y, s=12 + 110 * S.loc[m, 'unit'] ** 2, color=col(m), edgecolor='white', linewidth=0.8, zorder=3, alpha=0.95)
        t, off, ha = lab[m]
        ax.annotate(t, (x, y), xytext=off, textcoords='offset points', ha=ha, va='center', fontsize=6.1,
                    color=INK if m in INDEP else INK2, fontweight='bold' if m == 'model' else 'normal')
    for v in [0.25, 0.85]:
        ax.scatter([], [], s=12 + 110 * v ** 2, color=GREY, label=f'{v:.2f}')
    ax.legend(title='Spot split-half', title_fontsize=5.8, fontsize=5.8, loc='upper left', bbox_to_anchor=(0.0, 0.56), ncol=2,
              borderpad=0.2, handletextpad=0.1, columnspacing=0.6)
    ax.set_xlim(-0.03, 0.82); ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_ylim(0.70, 0.95)
    ax.grid(color=RULE, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel('Laminar pattern explained by depth ($R^2$)'); ax.set_ylabel('Split-half agreement')
    ax.set_title('Apparent reliability', loc='left')
    # (C) radar: reliability and depth independence
    ax = fig.add_axes([0.755, 0.6, 0.19, 0.3], projection='polar'); letter(fig, 0.69, 0.995, 'C')
    axes = [('split', 'Split-half'), ('down', '25% vs\nfull'), ('adj', 'Serial'), ('unit', 'Spot\nsplit-half'),
            ('indep_r', '1 − |$\\rho$|'), ('indep_p', '1 − $R^2$')]
    S['indep_r'] = 1 - S['depth'].abs(); S['indep_p'] = 1 - S['expl']
    rm = ['norm', 'liana_morans', 'norm_r', 'model']
    radar(ax, [l for _, l in axes], [[S.loc[m, k] for k, _ in axes] for m in rm], ['#9b9a95'] + [MCOL[m] for m in rm[1:]],
          [NAME[m] for m in rm], rmin=0.0, rmax=1.0)
    ax.tick_params(axis='x', pad=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, handlelength=1.2, columnspacing=0.8)
    ax.set_title('Reliability and depth independence', fontsize=7.5, pad=14)
    # (D) spot/edge-level split-half on one section
    r = pickle.load(open(f'{RES}/reliability/sample_{SEC}.pkl', 'rb'))['out']; letter(fig, 0.005, 0.44, 'D')
    for j, m in enumerate(['norm_r', 'model']):
        ax = fig.add_axes([0.065 + j * 0.175, 0.085, 0.14, 0.27])
        a_, b_ = pct(r['A'][m]['total'].ravel()), pct(r['B'][m]['total'].ravel())
        cm = LinearSegmentedColormap.from_list('c', ['#ffffff', MCOL[m]])
        ax.hexbin(a_, b_, gridsize=22, cmap=cm, mincnt=1, lw=0.1, edgecolors='white', extent=(0, 1, 0, 1))
        ax.plot([0, 1], [0, 1], color=INK, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_title(NAME[m].replace(' + ', ' +\n') if m == 'norm_r' else NAME[m], fontsize=6.8, pad=2)
        ax.text(0.04, 0.96, f'$\\rho$ = {spearmanr(a_, b_)[0]:.2f}', transform=ax.transAxes, va='top', fontsize=6.5,
                fontweight='bold' if m == 'model' else 'normal')
        ax.set_xlabel('Half A (percentile)')
        if j == 0: ax.set_ylabel('Half B (percentile)')
        else: ax.set_yticklabels([])
    fig.text(0.065, 0.415, f'Edge-level totals, section {SEC}', fontsize=6.5, color=INK2)
    # (E) per-section paired comparison
    letter(fig, 0.395, 0.44, 'E'); mm = ['norm_r', 'liana_morans', 'model']
    for j, (k, t) in enumerate([('splithalf', 'Split-half'), ('down25', '25% vs full'), ('unit_splithalf', 'Spot split-half')]):
        ax = fig.add_axes([0.45 + j * 0.19, 0.085, 0.145, 0.3])
        W = PS[PS.method.isin(mm)].pivot(index='sample', columns='method', values=k)[mm]
        for _, row in W.iterrows(): ax.plot(range(3), row.values, color=GREY, lw=0.45, alpha=0.8, zorder=1)
        for i, m in enumerate(mm):
            ax.scatter(np.full(len(W), i), W[m], s=11, color=MCOL[m], edgecolor='white', lw=0.4, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [W[m].median()] * 2, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(3)); ax.set_xticklabels(['Norm. +\nresid.', 'LIANA+\nMoran', 'stCCC'], fontsize=5.9)
        ax.set_xlim(-0.45, 2.45); lo = W.values.min()
        ax.set_ylim(lo - 0.05, 1.08 if k != 'unit_splithalf' else 1.2); ax.set_title(t, fontsize=7, pad=2)
        ax.grid(axis='y', color=RULE, lw=0.5); ax.set_axisbelow(True)
        ax.set_yticks([t for t in ax.get_yticks() if lo - 0.05 <= t <= 1.0001])
        if j == 0: ax.set_ylabel('Spearman agreement (per section)')
        top = W.values.max()
        for kk, other in enumerate(['norm_r', 'liana_morans']):
            pval = wilcoxon(W['model'], W[other]).pvalue
            bracket(ax, mm.index(other), 2, top + 0.025 + kk * 0.06 * (1 if k != 'unit_splithalf' else 1.6), fmt_p(pval), h=0.012, fs=5.3)
    save(fig, OUT, 'fig_reliability')


# ---------------------------------------------------------------- Figure 5
def interface_stats():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RES}/interface/sample_*.csv'))])
    df['sec'] = df['sample'].str[-6:].astype(int); df['hyp'] = df.ligand + '-' + df.receptor + ' @ ' + df.boundary
    adjrows, disc = [], []
    for inp, d in df.groupby('input'):
        W = d.pivot_table(index='hyp', columns='sec', values='z')
        for a, b in ADJ:
            adjrows.append(dict(inp=inp, pair=f'{str(a)[-2:]}–{str(b)[-2:]}', donor=DONOR(a), rho=spearmanr(*W[[a, b]].dropna().T.values)[0]))
        for mn in [1, 4, 8]:
            V = W[W.notna().sum(1) >= mn]; cols = {k: [c for c in V if DONOR(c) == k] for k in ['Br5292', 'Br5595', 'Br8100']}
            dz = pd.concat({k: (V[c].sum(1) / np.sqrt(V[c].notna().sum(1))).where(V[c].notna().sum(1) > 0) for k, c in cols.items()}, axis=1)
            z = dz.sum(1) / np.sqrt(dz.notna().sum(1)); q = multipletests(norm.sf(z), method='fdr_bh')[1]
            disc.append(dict(inp=inp, mn=mn, n=len(V), d=int((q < 0.05).sum())))
    return df, pd.DataFrame(adjrows), pd.DataFrame(disc)


def top_matrix(df):
    hs = [f'{l}–{r}  {b.replace("Layer", "L")}' for l, r, b in TOP]; out = {}
    for inp in ['model', 'norm']:
        M = np.full((len(TOP), 12), np.nan)
        for i, (l, r, b) in enumerate(TOP):
            d = df[(df.ligand == l) & (df.receptor == r) & (df.boundary == b) & (df.input == inp)].set_index('sec').z
            M[i] = [d.get(s, np.nan) for s in SECTIONS]
        out[inp] = M
    return hs, out


def fig_interface():
    from scipy.stats import wilcoxon
    df, A, Dd = interface_stats(); hs, Z = top_matrix(df)
    fig = plt.figure(figsize=(TEXTW, 6.5))
    R1, R2 = 0.745, 0.415          # bottoms of rows 1 and 2
    # (A) section z between two serial sections, all hypotheses
    letter(fig, 0.005, 0.995, 'A'); a_, b_ = 151675, 151676
    for j, (inp, key) in enumerate([('norm', 'norm_r'), ('model', 'model')]):
        ax = fig.add_axes([0.06 + j * 0.135, R1, 0.115, 0.19])
        W = df[df.input == inp].pivot_table(index='hyp', columns='sec', values='z')[[a_, b_]].dropna()
        ax.scatter(W[a_], W[b_], s=2, color=MCOL[key], alpha=0.35, lw=0, rasterized=True)
        lim = 4.5; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.5, ls=(0, (3, 2)))
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        ax.text(0.04, 0.96, f'$\\rho$ = {spearmanr(W[a_], W[b_])[0]:.2f}', transform=ax.transAxes, va='top', fontsize=6.3,
                fontweight='bold' if inp == 'model' else 'normal')
        ax.set_title('Normalised +\nresidualised' if inp == 'norm' else 'stCCC', fontsize=6.6, pad=2)
        ax.set_xlabel(f'$z$, {a_}', fontsize=6.2)
        if j == 0: ax.set_ylabel(f'$z$, {b_}', fontsize=6.2)
        else: ax.set_yticklabels([])
    # (B) replication over all serial pairs
    ax = fig.add_axes([0.38, R1, 0.25, 0.19]); letter(fig, 0.335, 0.995, 'B')
    piv = A.pivot(index='pair', columns='inp', values='rho').loc[A.pair.unique()]; xs = np.arange(len(piv))
    for x_, (_, r) in zip(xs, piv.iterrows()):
        ax.plot([x_, x_], [r['norm'], r['model']], color=GREY, lw=1.2, zorder=1)
    for inp, k in [('norm', 'norm_r'), ('model', 'model')]:
        ax.scatter(xs, piv[inp], s=30, color=MCOL[k], edgecolor='white', lw=0.8, zorder=3, label=NAME[k])
        ax.axhline(piv[inp].mean(), color=MCOL[k], lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(5.5, piv[inp].mean() + 0.02, f'mean {piv[inp].mean():.2f}', color=MCOL[k], fontsize=6, va='bottom', ha='right')
    ax.set_xticks(xs); ax.set_xticklabels(piv.index, fontsize=5.9); ax.set_xlim(-0.5, 5.5)
    for x in [1.5, 3.5]: ax.axvline(x, color=RULE, lw=0.8)
    for x, d in zip([0.5, 2.5, 4.5], ['Br5292', 'Br5595', 'Br8100']):
        ax.text(x, 0.86, d, ha='center', va='top', fontsize=6, color=INK2)
    ax.axhline(0, color=INK2, lw=0.5); ax.set_ylim(-0.12, 0.88)
    ax.set_xlabel('Serial section pair'); ax.set_ylabel('Spearman $\\rho$ of $z$')
    ax.set_title('Replication between serial sections', loc='left', fontsize=7)
    pv = wilcoxon(piv['model'], piv['norm'], alternative='greater').pvalue
    ax.text(0.98, 0.03, f'one-sided Wilcoxon, {fmt_p(pv)}', transform=ax.transAxes, fontsize=5.6, color=INK2, va='bottom', ha='right')
    # (C) discoveries
    ax = fig.add_axes([0.72, R1, 0.12, 0.19]); letter(fig, 0.665, 0.995, 'C'); w = 0.38
    for k, (inp, key) in enumerate([('norm', 'norm_r'), ('model', 'model')]):
        sub = Dd[Dd.inp == inp].sort_values('mn'); xs = np.arange(3) + (k - 0.5) * w
        ax.bar(xs, sub.d, width=w - 0.05, color=MCOL[key], lw=0)
        for x_, v in zip(xs, sub.d): ax.text(x_, v + 0.4, str(v), ha='center', fontsize=6)
    n = Dd[Dd.inp == 'model'].sort_values('mn').n.values
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([f'≥{m}\n({k:,})' for m, k in zip([1, 4, 8], n)], fontsize=5.9)
    ax.set_xlabel('Sections tested', fontsize=6.5); ax.set_ylabel('Discoveries ($q$ < 0.05)'); ax.set_ylim(0, 24)
    ax.set_title('Discoveries', loc='left', fontsize=7); ax.grid(axis='y', color=RULE, lw=0.5); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], ls='none', marker='o', ms=4.5, color=MCOL[k], label='Normalised +\nresidualised' if k == 'norm_r' else NAME[k]) for k in ['norm_r', 'model']]
    fig.legend(handles=h, loc='center left', bbox_to_anchor=(0.85, R1 + 0.11), fontsize=6, labelspacing=0.8, handletextpad=0.2)
    # (D) volcano over the 116 hypotheses tested in >= 8 sections
    ax = fig.add_axes([0.065, R2, 0.23, 0.24]); letter(fig, 0.005, 0.69, 'D')
    for inp, key in [('norm', 'norm_r'), ('model', 'model')]:
        c = pd.read_csv(f'{RES}/interface/combined_{inp}.csv'); sig = c.q < 0.05
        ax.scatter(c.donor_z[~sig], -np.log10(c.q[~sig]), s=7, color=MCOL[key], alpha=0.3, lw=0)
        ax.scatter(c.donor_z[sig], -np.log10(c.q[sig]), s=16, color=MCOL[key], edgecolor='white', lw=0.5, zorder=3)
        if inp == 'model':
            top = c.sort_values('q')
            for t, i in [('NPTX1/2–NPTXR L5/L6', 0), ('CALM1/3–PDE1A L5/L6', 2), ('NPTX2–NPTXR L2/L3', 4)]:
                r = top.iloc[i]
                ax.annotate(t, (r.donor_z, -np.log10(r.q)), xytext=(-5, 0), textcoords='offset points', ha='right', va='center', fontsize=5.4)
        else:
            r = c.sort_values('q').iloc[0]; lr, bd = r.hyp.split(' @ ')
            ax.annotate(lr.replace('-', '–') + ' ' + bd.replace('Layer', 'L') + ' (normalised)', (r.donor_z, -np.log10(r.q)), xytext=(-5, -1),
                        textcoords='offset points', ha='right', va='center', fontsize=5.4, color=MCOL[key])
    ax.axhline(-np.log10(0.05), color=INK2, lw=0.6, ls=(0, (3, 2))); ax.text(-3.4, -np.log10(0.05) + 0.1, '$q$ = 0.05', fontsize=5.6, color=INK2)
    ax.axvline(0, color=GREY, lw=0.5); ax.set_xlim(-3.6, 6.3)
    ax.set_xlabel('Donor-level interface $z$'); ax.set_ylabel('$-\\log_{10}\\,q$'); ax.set_title('116 hypotheses (≥ 8 sections)', loc='left', fontsize=7)
    ax.grid(color=RULE, lw=0.5); ax.set_axisbelow(True)
    # (E) NPTX2-NPTXR maps on 151673
    xy, y, ei, ej, sc_ = edge_scores(SEC, 'NPTX2', 'NPTXR')
    bnd = ((y[ei] == 'Layer5') & (y[ej] == 'Layer6')) | ((y[ei] == 'Layer6') & (y[ej] == 'Layer5')); mid = (xy[ei[bnd]] + xy[ej[bnd]]) / 2
    letter(fig, 0.335, 0.69, 'E')
    for j, k in enumerate(['lay', 'model', 'norm']):
        ax = fig.add_axes([0.345 + j * 0.115, R2 + 0.01, 0.11, 0.21])
        if k == 'lay': layermap(ax, xy, y, 'Annotated layers', s=0.6)
        else: spotmap(ax, xy, pct(sc_[k]), SEQ, 'stCCC' if k == 'model' else 'Normalised +\nresidualised', s=0.6)
        ax.scatter(mid[:, 0], -mid[:, 1], s=0.5, c=INK, lw=0, rasterized=True)
        ax.title.set_fontsize(6.6)
    # zoom window around the L5/L6 boundary
    c0 = mid[np.argmin(np.linalg.norm(mid - np.median(mid, 0), axis=1))]
    sp = np.median(np.linalg.norm(xy[ei] - xy[ej], axis=1)); rad = 3.6 * sp
    for ax in fig.axes[-3:]:
        ax.add_patch(Rectangle((c0[0] - rad, -c0[1] - rad), 2 * rad, 2 * rad, fill=False, ec=INK, lw=0.7))
    fig.text(0.345, R2 - 0.005, 'NPTX2–NPTXR, section 151673; black: L5/L6 boundary edges; box: window in F', fontsize=6, color=INK2, va='top')
    # (F) edge-level network in the zoom window
    letter(fig, 0.69, 0.69, 'F')
    win = (np.abs(xy[ei, 0] - c0[0]) < rad) & (np.abs(xy[ei, 1] - c0[1]) < rad) & (np.abs(xy[ej, 0] - c0[0]) < rad) & (np.abs(xy[ej, 1] - c0[1]) < rad)
    spots = np.where((np.abs(xy[:, 0] - c0[0]) < rad) & (np.abs(xy[:, 1] - c0[1]) < rad))[0]
    for j, k in enumerate(['model', 'norm']):
        ax = fig.add_axes([0.71 + j * 0.145, R2 + 0.035, 0.135, 0.185])
        ev = pct(sc_[k + '_edge'])
        for e in np.where(win & bnd)[0]:
            ax.plot(xy[[ei[e], ej[e]], 0], -xy[[ei[e], ej[e]], 1], color=INK, lw=4.6, solid_capstyle='round', zorder=1)
        for e in np.where(win)[0]:
            ax.plot(xy[[ei[e], ej[e]], 0], -xy[[ei[e], ej[e]], 1], color=SEQ(ev[e]), lw=2.4, solid_capstyle='round', zorder=2)
        ax.scatter(xy[spots, 0], -xy[spots, 1], s=14, c=[LCOL.get(y[i], GREY) for i in spots], edgecolor='white', lw=0.4, zorder=3)
        ax.set_aspect('equal'); ax.axis('off')
        bm = ev[win & bnd].mean(); wm = ev[win & ~bnd & (((y[ei] == 'Layer5') & (y[ej] == 'Layer5')) | ((y[ei] == 'Layer6') & (y[ej] == 'Layer6')))].mean()
        ax.set_title(('stCCC' if k == 'model' else 'Normalised +\nresidualised'), fontsize=6.6, pad=2)
    hbar(fig, fig.add_axes([0.80, R2 - 0.035, 0.12, 0.011]), SEQ, [0, 1], ['low', 'high'])
    fig.text(0.79, R2 - 0.03, 'edge score (percentile)', fontsize=6, color=INK2, ha='right', va='center')
    # (G) section-level interface z of the ten leading hypotheses
    letter(fig, 0.005, 0.33, 'G')
    for j, k in enumerate(['model', 'norm']):
        ax = fig.add_axes([0.145 + j * 0.42, 0.055, 0.385, 0.235])
        im = ax.imshow(np.ma.masked_invalid(Z[k]), cmap=DIV, vmin=-4, vmax=4, aspect='auto'); ax.set_facecolor(RULE)
        ax.set_xticks(range(12)); ax.set_xticklabels([str(s)[-2:] for s in SECTIONS], fontsize=6)
        for x in [3.5, 7.5]: ax.axvline(x, color='white', lw=2)
        ax.set_yticks(range(len(hs))); ax.set_yticklabels(hs if j == 0 else [], fontsize=6); ax.tick_params(length=0)
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        for i in range(Z[k].shape[0]):
            for jj in range(12):
                if np.isfinite(Z[k][i, jj]) and Z[k][i, jj] <= 0: ax.text(jj, i, '–', ha='center', va='center', fontsize=6, color=INK2)
        pos = int((Z[k] > 0).sum()); tot = int(np.isfinite(Z[k]).sum())
        ax.set_title(f'{NAME["model" if k == "model" else "norm_r"]}: $z$ > 0 in {pos} of {tot}', pad=3,
                     fontweight='bold' if k == 'model' else 'normal')
        ax.set_xlabel('Section (last two digits)', fontsize=6.3)
    hbar(fig, fig.add_axes([0.83, 0.325, 0.12, 0.011]), DIV, [-4, 0, 4], ['−4', '0', '4'], vmin=-4, vmax=4)
    fig.text(0.82, 0.33, 'Section-level interface $z$ (–: $z$ ≤ 0; grey: not annotated)', fontsize=6.1, color=INK2, ha='right', va='center')
    save(fig, OUT, 'fig_interface')
    print('positives', {k: (int((Z[k] > 0).sum()), int(np.isfinite(Z[k]).sum())) for k in Z}, 'wilcoxon', pv)


# ---------------------------------------------------------------- Figure 1
def fig_overview():
    W, H = TEXTW, 3.05
    fig = plt.figure(figsize=(W, H)); ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off')
    fa = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

    def bx(x, y, w, h, title, sub=None, comm=False):
        ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=BLUE_T if comm else GREY_T,
                                    ec=BLUE if comm else '#b9b8b2', lw=0.8, zorder=2))
        if sub is None:
            ov.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=6.6, fontweight='bold', zorder=3)
        else:
            ov.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=6.6, fontweight='bold', zorder=3)
            ov.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=6.2, color=INK2, zorder=3)

    def ar(p0, p1, lab=None, lo=(0, 0.07), col=INK2, rad=0.0, ha='center'):
        ov.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>,head_length=3.2,head_width=1.8', color=col, lw=0.8,
                                     connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=1))
        if lab: ov.text((p0[0] + p1[0]) / 2 + lo[0], (p0[1] + p1[1]) / 2 + lo[1], lab, fontsize=6.2, ha=ha, va='center', color=col)

    def head(x, y, l, t):
        ov.text(x, y, l, fontsize=9, fontweight='bold', va='top'); ov.text(x + 0.17, y - 0.015, t, fontsize=7.5, va='top')

    # ---------- A: input
    head(0.02, H - 0.03, 'A', 'Input: one Visium section')
    X, genes, xy, y = load(SEC); lib = np.log(X.sum(1) + 1)
    ax = fa(0.08, 1.2, 1.05, 1.5); layermap(ax, xy, y, s=0.55)
    ov.text(1.2, 2.45, 'counts $c_i$\nlibrary size $\\ell_i$\nposition $x_i$\nLR pairs $\\mathcal{P}$', fontsize=6.3, va='top', linespacing=1.5)
    # local graph around a Layer5/Layer6 boundary spot
    _, idx = NearestNeighbors(n_neighbors=9).fit(xy).kneighbors(xy)
    cand = [i for i in range(len(y)) if y[i] == 'Layer5' and (y[idx[i, 1:]] == 'Layer6').any()]
    c0 = cand[int(np.argmin([np.linalg.norm(xy[i] - np.median(xy, 0)) for i in cand]))]
    sp = np.median(np.linalg.norm(xy[idx[:, 1]] - xy, axis=1)); P = np.where(np.linalg.norm(xy - xy[c0], axis=1) < 2.6 * sp)[0]
    ax.add_patch(Rectangle((xy[c0, 0] - 2.7 * sp, -xy[c0, 1] - 2.7 * sp), 5.4 * sp, 5.4 * sp, fill=False, ec=INK, lw=0.7))
    ia = fa(0.2, 0.12, 1.3, 0.92); Ps = set(P)
    for i in P:
        for j in idx[i, 1:]:
            if j in Ps: ia.plot(xy[[i, j], 0], -xy[[i, j], 1], color=GREY, lw=0.5, zorder=1)
    for j in idx[c0, 1:]:
        ia.plot(xy[[c0, j], 0], -xy[[c0, j], 1], color=BLUE, lw=1.3, zorder=2)
    ia.scatter(xy[P, 0], -xy[P, 1], c=[LCOL.get(y[i], GREY) for i in P], s=34, ec='white', lw=0.6, zorder=3)
    ia.scatter(xy[c0, 0], -xy[c0, 1], c=LCOL['Layer5'], s=60, ec=INK, lw=1.0, zorder=4)
    ia.text(xy[c0, 0] - 0.55 * sp, -xy[c0, 1] + 0.35 * sp, '$i$', fontsize=7.5, fontweight='bold', zorder=5)
    ia.set_aspect('equal'); ia.axis('off')
    for (x0, y0), (x1, y1) in [((0, 1), (0, 1)), ((1, 1), (1, 1))]:
        cp = ConnectionPatch(xyA=(xy[c0, 0] + (x0 - 0.5) * 5.4 * sp, -xy[c0, 1] - 2.7 * sp), coordsA=ax.transData,
                             xyB=(x1, y1), coordsB=ia.transAxes, color=INK3, lw=0.5, ls=(0, (2, 2)))
        fig.add_artist(cp)
    ov.text(0.85, 0.05, '8-NN graph $\\mathcal{N}(i)$ (blue)', fontsize=6.1, color=INK2, ha='center')
    # ---------- B: model
    ov.plot([1.78, 1.78], [0.12, H - 0.12], color=RULE, lw=0.8); ov.plot([5.42, 5.42], [0.12, H - 0.12], color=RULE, lw=0.8)
    head(1.86, H - 0.03, 'B', 'stCCC: spatially-gated VAE with LR-guided attention')
    bx(1.95, 1.3, 0.62, 1.1, 'Encoder', '$y_i\\rightarrow q(z_i)$')
    bx(2.78, 2.0, 1.95, 0.5, 'Gaussian-process spatial latent', '$z^{\\mathrm{GP}}_i$ with SVGP prior over $x_i$')
    bx(2.78, 1.25, 0.88, 0.52, 'Spatial gating', '$\\omega^{\\mathrm{e}}_i\\tilde\\mu_i+\\omega^{\\mathrm{s}}_i\\Psi(x_i)$')
    bx(3.82, 1.25, 0.91, 0.52, 'LR-guided attention', 'bias $\\gamma S_{ij}$ over $\\mathcal{N}(i)$', comm=True)
    bx(4.92, 1.3, 0.42, 1.2, 'NB\ndecoder')
    bx(2.78, 0.42, 0.88, 0.52, 'LR edge scores', '$S_{ij}$, depth-residualised', comm=True)
    bx(3.82, 0.42, 0.91, 0.52, 'Consistency', '$\\mathrm{KL}(P_i\\,\\|\\,Q_i)$', comm=True)
    ar((1.66, 1.85), (1.95, 1.85))
    ar((2.57, 2.25), (2.78, 2.25)); ar((2.57, 1.51), (2.78, 1.51)); ar((3.66, 1.51), (3.82, 1.51))
    ar((4.73, 1.51), (4.92, 1.51), '$z^{\\mathrm{N}}_i$', lo=(0, 0.09)); ar((4.73, 2.25), (4.92, 2.25))
    ar((3.22, 0.94), (3.95, 1.25), '$\\gamma S_{ij}$', lo=(-0.16, 0.06), col=BLUE)
    ar((3.66, 0.68), (3.82, 0.68), col=BLUE); ov.text(3.74, 0.77, '$P_i$', fontsize=6.2, ha='center', color=BLUE)
    ar((4.45, 1.25), (4.45, 0.94), col=BLUE); ov.text(4.5, 1.1, '$Q_i$', fontsize=6.2, ha='left', color=BLUE)
    ov.text(5.13, 1.2, 'rate $\\rho_i$', fontsize=6.2, ha='center', va='top', color=INK2)
    ov.text(3.62, 0.13, '$\\mathcal{L}=-\\log p_{\\mathrm{NB}}(c_i\\mid z_i,s_i)+\\beta\\,\\mathrm{KL}+\\lambda\\,\\mathrm{KL}(P\\,\\|\\,Q)$',
            fontsize=6.4, ha='center', va='bottom')
    ov.add_patch(FancyBboxPatch((4.88, 2.62), 0.12, 0.12, boxstyle='round,pad=0,rounding_size=0.02', fc=BLUE_T, ec=BLUE, lw=0.8))
    ov.text(5.04, 2.68, 'communication', fontsize=6, va='center', color=INK2)
    # ---------- C: read-out
    head(5.5, H - 0.03, 'C', 'De-confounded read-out')
    ar((5.34, 1.9), (5.58, 2.1), rad=-0.2)
    bx(5.58, 2.0, 1.38, 0.58, 'Communication score', 'LR edge scores on $\\rho_i$,\ndepth-residualised', comm=True)
    ar((6.27, 2.0), (6.27, 1.86))
    th = [fa(5.6, 1.0, 0.62, 0.7), fa(6.33, 1.0, 0.62, 0.7), fa(5.6, 0.08, 0.62, 0.7), fa(6.33, 0.08, 0.62, 0.7)]
    R = pd.read_csv(f'{RES}/laminar_patterns.csv'); n = R[R.m == 'model'].groupby('p').size(); keep = n[n >= 8].index
    M = R[(R.m == 'model') & R.p.isin(keep)].groupby('p')[LAYERS].mean()
    M = M.loc[sorted(M.index, key=lambda p: (int(np.nanargmax(M.loc[p].values)), -np.nanmax(M.loc[p].values)))]
    th[0].imshow(M.values, cmap=DIV, vmin=-1, vmax=1, aspect='auto'); clean(th[0])
    th[0].set_xticks(range(7)); th[0].set_xticklabels(['1', '', '3', '', '5', '', 'W'], fontsize=5.5); th[0].tick_params(length=0, pad=1)
    xy2, y2, ei, ej, sc_ = edge_scores(SEC, 'NPTX2', 'NPTXR')
    bnd = ((y2[ei] == 'Layer5') & (y2[ej] == 'Layer6')) | ((y2[ei] == 'Layer6') & (y2[ej] == 'Layer5')); mid = (xy2[ei[bnd]] + xy2[ej[bnd]]) / 2
    spotmap(th[1], xy2, pct(sc_['model']), SEQ, s=0.3); th[1].scatter(mid[:, 0], -mid[:, 1], s=0.3, c=INK, lw=0, rasterized=True)
    S = pd.read_csv(f'{RES}/reliability/figure_summary.csv').set_index('m'); a = th[2]; w = 0.36
    for k, m in enumerate(['norm_r', 'model']):
        a.bar(np.arange(3) + (k - 0.5) * w, S.loc[m, ['split', 'down', 'adj']].values, width=w - 0.04, color=MCOL[m], lw=0)
    a.set_ylim(0.5, 0.9); a.set_xticks(range(3)); a.set_xticklabels(['split', '25%', 'serial'], fontsize=5.5)
    a.tick_params(length=0, pad=1); a.set_yticks([]); a.spines['left'].set_visible(False)
    z = np.load(f'{RES}/latent/sample_{SEC}_lr_seed0.npz'); xs_, ys_ = xy[z['keep']], y[z['keep']]; ok_ = np.isin(ys_, LAYERS)
    layermap(th[3], xs_, matched(ys_, ok_, cluster(z['z'], len(set(ys_[ok_])), False)), s=0.3)
    for a, t in zip(th, ['Laminar map', 'Interface scan', 'Reliability', 'Domains']):
        a.set_title(t, fontsize=6.3, pad=2)
    save(fig, OUT, 'fig_overview')


if __name__ == '__main__':
    for w in (sys.argv[1:] or ['overview', 'reliability', 'interface']):
        globals()[f'fig_{w}']()
