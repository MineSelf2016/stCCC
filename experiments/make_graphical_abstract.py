"""Graphical abstract for the NAR submission (5:2 landscape, 10 x 4 in, sans-serif 12-16 pt; PDF + 600-dpi TIF/PNG).
Left: spot-level depth dependence of the normalised score vs stCCC; middle: the model; right: LR-specific serial
agreement of stCCC, MAGIC and the residualised normalised score on the three datasets. Values are read from
results[/d]/nar_summary.csv.  Usage: python make_graphical_abstract.py"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results'); OUT = os.path.join(HERE, '..', 'paper', 'figures')
INK, INK2, RULE, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#ffffff'
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#a9a8a2'      # stCCC, MAGIC, conventional (validated pair + neutral)
plt.rcParams.update({'font.family': 'Nimbus Sans', 'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
DS = [('', 'Human\ncortex'), ('abca1/', 'Mouse\ncortex'), ('her2/', 'Breast\ncancer')]
S = {d: pd.read_csv(f'{R}/{d}nar_summary.csv', index_col=0) for d, _ in DS}

fig = plt.figure(figsize=(10, 4), facecolor=SURF)


def bars(ax, series, ylab, title, ylim):
    n = len(series); w = 0.8 / n
    for k, (col, lab, vals) in enumerate(series):
        x = [i + (k - (n - 1) / 2) * w for i in range(len(DS))]
        ax.bar(x, vals, width=w, color=col, edgecolor=SURF, linewidth=1.5, label=lab, zorder=3)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([n for _, n in DS], fontsize=12)
    ax.set_ylim(*ylim); ax.set_ylabel(ylab, fontsize=12); ax.tick_params(axis='y', labelsize=11, length=0)
    ax.tick_params(axis='x', length=0); ax.grid(axis='y', color=RULE, lw=0.8, zorder=0); ax.axhline(0, color=INK2, lw=0.8, zorder=4)


# left: depth dependence
fig.text(0.015, 0.93, '1  Scores track depth', fontsize=14, fontweight='bold', va='center')
fig.text(0.37, 0.93, '2  stCCC', fontsize=14, fontweight='bold', va='center')
fig.text(0.665, 0.93, '3  Reliability protocol', fontsize=14, fontweight='bold', va='center')
a1 = fig.add_axes([0.075, 0.17, 0.245, 0.58])
bars(a1, [(GREY, 'Normalised score', [S[d].loc['norm', 'depth'] for d, _ in DS]),
          (BLUE, 'stCCC', [S[d].loc['model', 'depth'] for d, _ in DS])],
     'Correlation with\nsequencing depth', '1  Scores track depth', (-0.05, 1.0))
a1.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=11, handlelength=1)
for i, (d, _) in enumerate(DS):
    a1.text(i + 0.2, 0.06, '≈0', ha='center', va='bottom', fontsize=11, color=BLUE)

# middle: model schematic
am = fig.add_axes([0.355, 0.05, 0.27, 0.9]); am.axis('off'); am.set_xlim(0, 1); am.set_ylim(0, 1)
steps = [('Counts + spot positions', '#f2f1ee', INK2), ('Spatially-gated VAE\nGP prior · LR-biased attention', '#e3eefb', BLUE),
         ('Library-free expression rate', '#f2f1ee', INK2), ('Depth-residualised\nLR edge scores', '#e3eefb', BLUE)]
ys = [0.74, 0.535, 0.345, 0.15]; h = [0.1, 0.14, 0.1, 0.14]
for (t, fc, ec), y, hh in zip(steps, ys, h):
    am.add_patch(FancyBboxPatch((0.04, y - hh / 2), 0.92, hh, boxstyle='round,pad=0.008,rounding_size=0.03', fc=fc, ec=ec, lw=1.2))
    am.text(0.5, y, t, ha='center', va='center', fontsize=12, linespacing=1.15)
for y0, y1, h0, h1 in zip(ys[:-1], ys[1:], h[:-1], h[1:]):
    am.add_patch(FancyArrowPatch((0.5, y0 - h0 / 2 - 0.008), (0.5, y1 + h1 / 2 + 0.012), arrowstyle='-|>', mutation_scale=14, color=INK2, lw=1.2))

# right: LR-specific replication
a3 = fig.add_axes([0.715, 0.17, 0.27, 0.58])
bars(a3, [(GREY, 'Normalised + resid.', [S[d].loc['norm_r', 'lrspec'] for d, _ in DS]),
          (ORANGE, 'MAGIC', [S[d].loc['magic', 'lrspec'] for d, _ in DS]),
          (BLUE, 'stCCC', [S[d].loc['model', 'lrspec'] for d, _ in DS])],
     'LR-specific agreement\nbetween serial sections', '3  Reliability protocol', (0, 0.7))
a3.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=11, handlelength=1, columnspacing=0.5, handletextpad=0.4)
fig.text(0.99, 0.025, 'Best input depends on tissue: stCCC in cortex, MAGIC in tumour', ha='right', fontsize=11, color=INK2)

for ext, kw in (('pdf', {}), ('png', {'dpi': 300}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}})):
    fig.savefig(f'{OUT}/graphical_abstract.{ext}', facecolor=SURF, **kw)
print('wrote', OUT)
