"""Supplementary figure: spatial gating weights (gate_map.py) and the neighbourhood swap (nbr_swap.py, nbr_swap_summary.py).
  A  omega^s of every spot on one section per dataset
  B  spatial share of the variance of the gated residual mean over training (median and IQR over sections)
  C  spatial share at convergence, per section
  D  LR-specific serial agreement with physical or transcriptional neighbours, for stCCC and MAGIC
Writes paper/figures/supp_fig_gate_swap.{pdf,png}.  Usage: python make_fig_gate_swap.py"""
import os, glob
import numpy as np, pandas as pd
from figstyle import *
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'results'); OUT = os.path.join(HERE, '..', 'paper', 'figures')
DS = [('', 'DLPFC (Visium)', 151673, 'o', '-'), ('abca1/', 'Mouse cortex (MERFISH)', 64, 's', '--'), ('her2/', 'Breast cancer (ST)', 72, '^', ':')]
G = {d: {int(os.path.basename(f)[7:-4]): np.load(f) for f in glob.glob(f'{R}/{d}gate/sample_*.npz')} for d, *_ in DS}

fig = plt.figure(figsize=(TEXTW, 5.0))
# A: omega^s maps
for k, (d, name, s, *_) in enumerate(DS):
    z = G[d][s]; w = z['omega_s']; xy = z['xy']; hi = float(np.percentile(w, 99))
    ax = fig.add_axes([0.02 + k * 0.33, 0.56, 0.27, 0.36])
    spotmap(ax, xy, w, 'Blues', s={'': 1.4, 'abca1/': 0.25, 'her2/': 9}[d], vmin=0, vmax=hi)
    ax.set_title(f'{name}, section {s}', fontsize=7, pad=2)
    cax = fig.add_axes([0.07 + k * 0.33, 0.525, 0.17, 0.012])
    hbar(fig, cax, 'Blues', [0, hi], ['0', f'{hi:.3f}'], vmin=0, vmax=hi, title=f'$\\omega^s$ (median {np.median(w):.3f})')
letter(fig, 0.005, 0.985, 'A')
fig.text(0.04, 0.975, 'Spatial gating weight $\\omega^s_i$ (share of the residual latent mean taken from position)', fontsize=7, va='top')

# B: trajectory of spatial share
axB = fig.add_axes([0.07, 0.08, 0.25, 0.33])
for d, name, s, mk, ls in DS:
    T = [z['trajectory'] for z in G[d].values()]; n = min(len(t) for t in T)
    M = np.stack([t[:n, 2] for t in T]); ep = T[0][:n, 0]
    q1, md, q3 = np.percentile(M, [25, 50, 75], 0)
    axB.fill_between(ep, q1, q3, color=INK3, alpha=0.18, lw=0); axB.plot(ep, md, color=INK, ls=ls, lw=1.1, label={'': 'DLPFC', 'abca1/': 'MERFISH', 'her2/': 'Breast cancer'}[d])
axB.set_xscale('log'); axB.set_xlabel('Training epoch'); axB.set_ylabel('Spatial share of variance\nof the gated residual mean'); axB.set_ylim(0, 1)
axB.legend(fontsize=6, loc='upper right', handlelength=2.2)
letter(fig, 0.005, 0.45, 'B')

# C: final share per section
axC = fig.add_axes([0.41, 0.08, 0.17, 0.33])
for i, (d, name, s, mk, ls) in enumerate(DS):
    v = np.array([float(z['share_spatial_var']) for z in G[d].values()])
    axC.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=10, marker=mk, color=INK2, lw=0, alpha=0.8)
    axC.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.4)
axC.set_xticks(range(3)); axC.set_xticklabels(['DLPFC', 'MERFISH', 'Breast\ncancer'], fontsize=6.5); axC.set_xlim(-0.6, 2.6)
axC.set_ylabel('Spatial share at convergence'); axC.set_ylim(0, 0.4)
letter(fig, 0.345, 0.45, 'C')

# D: neighbourhood swap
for k, (d, name, *_ ) in enumerate(DS):
    P = pd.read_csv(f'{R}/{d}nbr_swap_pairs.csv').pivot(index='pair', columns='input', values='rho')
    ax = fig.add_axes([0.665 + k * 0.115, 0.08, 0.085, 0.33])
    for (phys, trans), col, lab in ((('model', 'model_xall'), BLUE, 'stCCC'), (('magic_sp', 'magic'), YELLOW, 'MAGIC')):
        for _, r in P.iterrows():
            ax.plot([0, 1], [r[phys], r[trans]], color=col, lw=0.4, alpha=0.35)
        ax.plot([0, 1], [P[phys].mean(), P[trans].mean()], color=col, lw=1.8, marker='o', ms=3.5, label=lab)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['phys.', 'transc.'], fontsize=6); ax.set_xlim(-0.3, 1.3); ax.set_ylim(0, 0.8)
    ax.set_title(name.split(' (')[0].replace('Mouse cortex', 'MERFISH'), fontsize=6.5, pad=2)
    if k == 0:
        ax.set_ylabel('LR-specific serial agreement')
    else:
        ax.set_yticklabels([])
h = [plt.Line2D([], [], color=BLUE, lw=1.8, marker='o', ms=3.5, label='stCCC'), plt.Line2D([], [], color=YELLOW, lw=1.8, marker='o', ms=3.5, label='MAGIC')]
fig.legend(handles=h, loc='center', bbox_to_anchor=(0.8, 0.475), ncol=2, fontsize=6.5, handlelength=1.6)
fig.text(0.79, 0.02, 'neighbours used for pooling', ha='center', fontsize=6.5, color=INK2)
letter(fig, 0.625, 0.45, 'D')
save(fig, OUT, 'supp_fig_gate_swap'); print('saved')
