"""New main-text figures for the NAR manuscript, read from results/ (no retraining):
  fig_specificity  denoise-then-score baselines and LR-specific agreement on both datasets (nar_summary.py output)
  fig_abca1        the depth confound and reliability at single-cell resolution (ABCA-1 MERFISH isocortex)
  fig_simulation   semi-synthetic benchmark with planted communication and a depth gradient (sim_summary.py output)
Usage: NUMBA_CACHE_DIR=... python make_figures_nar.py [specificity abca1 simulation]"""
import os, sys, glob, json, pickle, warnings
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import spearmanr, wilcoxon
from figstyle import *
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results'); OUT = os.path.join(HERE, '..', 'paper', 'figures')
MC = {**MCOL, 'commot_r': GREEN, 'smooth_r': PINK, 'magic': YELLOW, 'scvi': PURPLE, 'commot': '#8f8e88'}
NM = {**NAME, 'commot': 'COMMOT', 'commot_r': 'COMMOT + resid.', 'smooth_r': 'Smoothing + resid.', 'magic': 'MAGIC + resid.',
      'scvi': 'scVI + resid.', 'norm_r': 'Normalised + resid.', 'liana_morans': 'LIANA+ local Moran'}
INDEP = ['liana_morans', 'commot_r', 'norm_r', 'smooth_r', 'magic', 'scvi', 'model']
SCANNED = ['norm_r', 'smooth_r', 'scvi', 'magic', 'model']
METRICS = [('splithalf', 'Split-\nhalf'), ('down25', 'Down-\n25%'), ('adjacent', 'Serial'), ('unit_splithalf', 'Spot\nsplit-half'), ('lrspec', 'LR-specific\nserial')]
DS = {'dlpfc': ('', 'Human DLPFC (Visium)'), 'abca1': ('abca1/', 'Mouse isocortex (MERFISH)'),
      'her2': ('her2/', 'HER2+ breast cancer (ST)')}


def summ(ds):
    return pd.read_csv(f'{RES}/{DS[ds][0]}nar_summary.csv').set_index('method')


def dotmatrix(ax, S, title):
    """Value tiles: one row per depth-independent score, one column per agreement measure; bold = best in column."""
    from matplotlib.patches import FancyBboxPatch
    for i, m in enumerate(INDEP):
        for j, (c, _) in enumerate(METRICS):
            v = S.loc[m, c]
            if pd.isnull(v):
                ax.text(j, i, '–', ha='center', va='center', color=INK3, fontsize=6.5); continue
            ax.add_patch(FancyBboxPatch((j - 0.42, i - 0.36), 0.84, 0.72, boxstyle='round,pad=0,rounding_size=0.12',
                                        fc=SEQ(np.clip(v, 0, 1)), ec='white', lw=0.8, zorder=2))
            best = S.loc[INDEP, c].max()
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.0, zorder=4,
                    color='white' if v > 0.55 else INK, fontweight='bold' if abs(v - best) < 5e-4 else 'normal')
    ax.axhline(len(INDEP) - 1.5, color=INK3, lw=0.6)
    ax.set_xlim(-0.5, len(METRICS) - 0.5); ax.set_ylim(len(INDEP) - 0.5, -0.5)
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([l for _, l in METRICS], fontsize=6.2)
    ax.set_yticks(range(len(INDEP))); ax.set_yticklabels([NM[m] for m in INDEP], fontsize=6.4)
    for t, m in zip(ax.get_yticklabels(), INDEP):
        t.set_color(MC[m] if m == 'model' else INK); t.set_fontweight('bold' if m == 'model' else 'normal')
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, pad=22, fontsize=7.2)


def fig_specificity():
    """Three datasets: value tiles per dataset (A-C), patterns vs LR-specific agreement (D), per-pair agreement (E)."""
    fig = plt.figure(figsize=(TEXTW, 7.7))
    axA = fig.add_axes([0.155, 0.715, 0.30, 0.235]); dotmatrix(axA, summ('dlpfc'), DS['dlpfc'][1])
    axB = fig.add_axes([0.66, 0.715, 0.30, 0.235]); dotmatrix(axB, summ('abca1'), DS['abca1'][1])
    axC = fig.add_axes([0.155, 0.395, 0.30, 0.235]); dotmatrix(axC, summ('her2'), DS['her2'][1])
    MK = [('dlpfc', 'o', -0.22, 'DLPFC'), ('abca1', 's', 0.0, 'MERFISH'), ('her2', '^', 0.22, 'Breast cancer')]
    # D: pattern agreement against LR-specific agreement, per input and dataset
    axD = fig.add_axes([0.60, 0.405, 0.24, 0.215])
    xs = []
    for ds, mk, _, _ in MK:
        S = summ(ds)
        for m in SCANNED:
            axD.scatter(S.loc[m, 'adjacent'], S.loc[m, 'lrspec'], s=40 if m == 'model' else 26, marker=mk, color=MC[m], edgecolor='white', lw=0.6, zorder=3)
            xs.append(S.loc[m, 'adjacent'])
    axD.set_xlabel('Serial agreement of laminar or regional patterns', fontsize=6.4); axD.set_ylabel('LR-specific serial agreement', fontsize=6.4)
    axD.set_xlim(np.floor(min(xs) * 20) / 20 - 0.02, np.ceil(max(xs) * 20) / 20 + 0.02); axD.set_ylim(0.0, 0.75)
    h = [plt.Line2D([], [], ls='none', marker='o', ms=4.5, mfc=MC[m], mec='white', label=NM[m]) for m in SCANNED]
    h += [plt.Line2D([], [], ls='none', marker=k, ms=4, mfc=INK3, mec='white', label=l) for _, k, _, l in MK]
    axD.legend(handles=h, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=5.8, handletextpad=0.2, ncol=1)
    # E: LR-specific serial agreement per serial pair, all datasets
    axE = fig.add_axes([0.075, 0.055, 0.86, 0.255])
    for k, (ds, mk, off, _) in enumerate(MK):
        P = pd.read_csv(f'{RES}/{DS[ds][0]}nar_iface_pairs.csv').pivot(index='pair', columns='input', values='rho')[SCANNED]
        for _, r in P.iterrows():
            axE.plot(np.arange(len(SCANNED)) + off, r.values, color=GREY, lw=0.45, zorder=1)
        for j, m in enumerate(SCANNED):
            axE.scatter(np.full(len(P), j + off), P[m], s=10, marker=mk, color=MC[m], edgecolor='white', lw=0.4, zorder=3)
            axE.plot([j + off - 0.08, j + off + 0.08], [P[m].mean()] * 2, color=INK, lw=1.3, zorder=4)
        yc = -0.13 - 0.075 * k                                  # one row of counts per dataset
        for j, m in enumerate(SCANNED[:-1]):
            n = int((P['model'] > P[m]).sum())
            axE.scatter([j - 0.12], [yc], s=6, marker=mk, color=INK3, lw=0, zorder=3)
            axE.text(j - 0.06, yc, f'{n}/{len(P)}', ha='left', va='center', fontsize=5.3, color=INK2)
    axE.text(len(SCANNED) - 1, -0.205, 'serial pairs with\nstCCC higher', ha='center', va='center', fontsize=5.3, color=INK2)
    axE.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    axE.set_xticks(range(len(SCANNED))); axE.set_xticklabels([NM[m] for m in SCANNED], fontsize=6.2)
    axE.set_xlim(-0.5, len(SCANNED) - 0.5); axE.set_ylim(-0.31, 0.85); axE.axhline(0, color=RULE, lw=0.6, zorder=0)
    axE.set_ylabel('LR-specific serial agreement\n(Spearman of interface $z$)')
    h = [plt.Line2D([], [], ls='none', marker=k, ms=4, mfc=INK3, mec='white', label=f'{l} pair') for _, k, _, l in MK]
    axE.legend(handles=h, loc='upper left', ncol=3, bbox_to_anchor=(0.0, 1.08))
    letter(fig, 0.005, 0.995, 'A'); letter(fig, 0.51, 0.995, 'B'); letter(fig, 0.005, 0.675, 'C'); letter(fig, 0.51, 0.675, 'D')
    letter(fig, 0.005, 0.335, 'E')
    save(fig, OUT, 'fig_specificity')


ALAY = ['L1', 'L2/3', 'L4', 'L5', 'L6a', 'L6b']; ACOL = dict(zip(ALAY, [BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN]))


def fig_abca1(sec=64):
    D = f'{HERE}/../data/abca1'
    ad = sc.read_h5ad(f'{D}/sample_{sec}_with_graph.h5ad'); xy = ad.obsm['spatial'].copy(); lab = ad.obs['label'].astype(str).values
    lib = np.log(np.asarray(ad.X.sum(1)).ravel() + 1)
    r = pickle.load(open(f'{RES}/abca1/reliability/sample_{sec}.pkl', 'rb'))
    tot = {m: r['out']['full'][m]['total'].mean(1) for m in ('norm', 'norm_r', 'model')}
    fig = plt.figure(figsize=(TEXTW, 4.9))
    W, H, y0 = 0.205, 0.37, 0.59
    axs = [fig.add_axes([0.01 + i * 0.245, y0, W, H]) for i in range(4)]
    for l in ALAY:
        m = lab == l; axs[0].scatter(xy[m, 0], -xy[m, 1], s=0.35, color=ACOL[l], lw=0, rasterized=True)
    axs[0].set_aspect('equal'); axs[0].axis('off'); axs[0].set_title(f'Isocortex layers (section {sec})', pad=2)
    h = [plt.Line2D([], [], ls='none', marker='o', ms=3.8, mec='none', color=ACOL[l], label=l) for l in ALAY]
    axs[0].legend(handles=h, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.02), handletextpad=0.05, columnspacing=0.5, fontsize=6)
    for ax, (v, t) in zip(axs[1:], [(lib, 'Log transcripts per cell'), (tot['norm'], 'Normalised total'), (tot['model'], 'stCCC total')]):
        spotmap(ax, xy, pct(v), SEQ, t, s=0.35)
        if t != 'Log transcripts per cell':
            ax.text(0.5, -0.02, f'$\\rho$ = {spearmanr(v, lib)[0]:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    hbar(fig, fig.add_axes([0.80, y0 - 0.035, 0.12, 0.014]), SEQ, [0, 0.5, 1], ['0', '50', '100'], None)
    fig.text(0.86, y0 - 0.08, 'Percentile within section', ha='center', fontsize=6.2, color=INK2)
    # C: depth correlation per section, all scores
    ps = pd.read_csv(f'{RES}/abca1/reliability/per_section.csv')
    order = ['counts', 'norm', 'liana_cosine', 'commot', 'liana_morans', 'commot_r', 'norm_r', 'smooth_r', 'magic', 'scvi', 'model']
    axC = fig.add_axes([0.085, 0.165, 0.39, 0.31])
    for i, m in enumerate(order):
        v = ps[ps.method == m].depth_spot.values; c = MC.get(m, INK3)
        axC.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=9, color=c, lw=0, alpha=0.85, zorder=3)
        axC.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.1, zorder=4)
    axC.axhline(0, color=RULE, lw=0.6, zorder=0); axC.axvline(3.5, color=GREY, lw=0.6, ls=(0, (2, 2)))
    axC.set_xticks(range(len(order))); axC.set_xticklabels([NM[m] for m in order], rotation=40, ha='right', fontsize=5.9)
    axC.set_ylabel('Spearman $\\rho$, cell-level total\nscore vs log transcripts'); axC.set_ylim(-0.3, 1.0)
    axC.text(1.5, 0.95, 'Depth-confounded', ha='center', fontsize=6, color=INK2); axC.text(7.5, 0.95, 'Depth-independent', ha='center', fontsize=6, color=INK2)
    # D: cell-level split-half agreement per section, depth-independent scores
    axD = fig.add_axes([0.62, 0.165, 0.35, 0.31])
    raincloud(axD, [ps[ps.method == m].unit_splithalf.values for m in INDEP], list(range(len(INDEP))), [MC[m] for m in INDEP], horizontal=True, s=5)
    axD.set_yticks(range(len(INDEP))); axD.set_yticklabels([NM[m] for m in INDEP], fontsize=6.2); axD.set_ylim(len(INDEP) - 0.5, -0.7)
    axD.set_xlabel('Cell-level split-half agreement'); axD.set_xlim(0, 1.02)
    letter(fig, 0.005, 0.995, 'A'); letter(fig, 0.25, 0.995, 'B'); letter(fig, 0.005, 0.5, 'C'); letter(fig, 0.5, 0.5, 'D')
    save(fig, OUT, 'fig_abca1')

SIMIN = ['norm_nr', 'norm', 'smooth_r', 'magic', 'scvi', 'model']
SNM = {'norm_nr': 'Normalised', 'norm': 'Normalised + resid.', 'smooth_r': 'Smoothing + resid.', 'magic': 'MAGIC + resid.',
       'scvi': 'scVI + resid.', 'model': 'stCCC'}
SMC = {**MC, 'norm_nr': INK3, 'norm': MC['norm_r']}


def fig_simulation():
    """Supplementary figure: semi-synthetic benchmark (results/simulation/per_task.csv from sim_summary.py)."""
    sys.path.insert(0, HERE)
    from simulate import base, band, TASKS
    from sklearn.neighbors import NearestNeighbors
    T = pd.read_csv(f'{RES}/simulation/per_task.csv')
    raw, genes, lab, xy = base(); _, idx = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy); nb = idx[:, 1:]
    fig = plt.figure(figsize=(TEXTW, 6.2))
    # A: design on section 151673
    ax = fig.add_axes([0.0, 0.70, 0.2, 0.27]); layermap(ax, xy, lab, 'Layers (151673)', s=1.0)
    ax = fig.add_axes([0.2, 0.70, 0.2, 0.27])
    m = np.zeros(len(lab)); m[band(lab, nb, 'Layer5', 'Layer6', 1)] = 1; m[band(lab, nb, 'Layer6', 'Layer5', 1)] = 2
    for v, c in ((0, RULE), (1, ORANGE), (2, BLUE)):
        k = m == v; ax.scatter(xy[k, 0], -xy[k, 1], s=1.0, color=c, lw=0, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Planted band (L5/L6, width 1)', pad=2)
    h = [plt.Line2D([], [], ls='none', marker='o', ms=3.5, mec='none', color=c, label=l) for c, l in ((ORANGE, 'ligand raised'), (BLUE, 'receptor raised'))]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=5.8, handletextpad=0.1, columnspacing=0.6)
    X = np.load(f'{RES}/simulation/task_15/counts.npz')['X']
    spotmap(fig.add_axes([0.4, 0.70, 0.2, 0.27]), xy, pct(np.log(X.sum(1) + 1)), SEQ, 'Log library, depth gradient', s=1.0)
    # B: AUROC against effect size, width 1
    W1 = T[T.width == 1]
    for k, (mode, depth) in enumerate([('program', 'none'), ('program', 'gradient'), ('gene', 'none'), ('gene', 'gradient')]):
        ax = fig.add_axes([0.075 + k * 0.24, 0.385, 0.185, 0.23])
        d = W1[(W1['mode'] == mode) & (W1.depth == depth) & (W1.f > 1)]
        for m_ in SIMIN:
            g = d[d.input == m_].groupby('f').auroc.agg(['mean', 'std'])
            ax.errorbar(g.index, g['mean'], yerr=g['std'], color=SMC[m_], lw=1.6 if m_ == 'model' else 1.0, marker='o', ms=3, capsize=1.5,
                        elinewidth=0.6, label=SNM[m_], zorder=3 if m_ == 'model' else 2)
        ax.set_ylim(0.45, 1.02); ax.set_xticks([1.5, 2, 3]); ax.axhline(0.5, color=RULE, lw=0.6)
        ax.set_title(f"{'Program' if mode == 'program' else 'Gene'} mode, {'no added' if depth == 'none' else '4-fold'} depth gradient", fontsize=6.5)
        ax.set_xlabel('Effect size $f$')
        if k == 0: ax.set_ylabel('AUROC, planted vs other')
    hl, ll = ax.get_legend_handles_labels()
    fig.legend(hl, ll, loc='center left', bbox_to_anchor=(0.66, 0.835), fontsize=6.3, title='Input of the interface scan', title_fontsize=6.5)
    # C: AUROC against band width, program mode, f = 3
    for k, depth in enumerate(['none', 'gradient']):
        ax = fig.add_axes([0.075 + k * 0.24, 0.07, 0.185, 0.21])
        d = T[(T['mode'] == 'program') & (T.f == 3.0) & (T.depth == depth)]
        for m_ in SIMIN:
            g = d[d.input == m_].groupby('width').auroc.agg(['mean', 'std'])
            ax.errorbar(g.index, g['mean'], yerr=g['std'], color=SMC[m_], lw=1.6 if m_ == 'model' else 1.0, marker='o', ms=3, capsize=1.5, elinewidth=0.6)
        ax.set_xticks([1, 2, 3]); ax.set_ylim(0.6, 1.02); ax.set_xlabel('Band width (spot rings)')
        ax.set_title(f"Program mode, $f$ = 3, {'no added' if depth == 'none' else '4-fold'} gradient", fontsize=6.5)
        if k == 0: ax.set_ylabel('AUROC')
    # D: false-positive rate under the null (f = 1)
    ax = fig.add_axes([0.6, 0.07, 0.37, 0.21])
    d = T[(T.f == 1.0)]
    for j, m_ in enumerate(SIMIN):
        for k, (mode, depth, mk) in enumerate([('gene', 'none', 'o'), ('gene', 'gradient', 's'), ('program', 'none', '^'), ('program', 'gradient', 'D')]):
            v = d[(d.input == m_) & (d['mode'] == mode) & (d.depth == depth)].fpr
            ax.scatter(np.full(len(v), j + (k - 1.5) * 0.15), v, s=9, marker=mk, color=SMC[m_], lw=0, zorder=3)
    ax.axhline(0.05, color=INK2, lw=0.7, ls=(0, (3, 2))); ax.text(len(SIMIN) - 0.5, 0.051, 'nominal 0.05', fontsize=5.6, color=INK2, va='bottom', ha='right')
    ax.set_xticks(range(len(SIMIN))); ax.set_xticklabels([SNM[m_].replace(' + resid.', '\n+ resid.') for m_ in SIMIN], fontsize=5.6)
    ax.set_ylim(0, 0.1); ax.set_ylabel('False-positive rate\n(no planted signal)')
    h = [plt.Line2D([], [], ls='none', marker=mk, ms=3.5, mfc=INK3, mec='none', label=l) for mk, l in
         [('o', 'gene, none'), ('s', 'gene, gradient'), ('^', 'program, none'), ('D', 'program, gradient')]]
    ax.legend(handles=h, loc='upper left', ncol=2, fontsize=5.4, handletextpad=0.1, columnspacing=0.6)
    letter(fig, 0.005, 0.995, 'A'); letter(fig, 0.005, 0.66, 'B'); letter(fig, 0.005, 0.325, 'C'); letter(fig, 0.53, 0.325, 'D')
    save(fig, OUT, 'supp_fig_simulation')



if __name__ == '__main__':
    for w in (sys.argv[1:] or ['specificity', 'abca1']):
        globals()[f'fig_{w}']()
