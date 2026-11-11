"""Shared figure style for the BIB manuscript. Figures are drawn at final print size
(text width 7.0 in, column width 3.38 in) so that font sizes are the printed sizes."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

TEXTW, COLW = 7.0, 3.38
INK, INK2, INK3, GREY, RULE = '#0b0b0b', '#52514e', '#8a8984', '#c9c8c2', '#e6e5e0'
BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6'
BLUE_T, GREY_T = '#e3eefb', '#f2f1ee'          # box tints for schematics

# one colour per scoring method, used in every figure (colour follows the entity)
MCOL = {'model': BLUE, 'norm_r': ORANGE, 'liana_morans': AQUA,
        'counts': '#6f6e69', 'norm': INK3, 'smooth': '#a9a8a2', 'liana_cosine': GREY, 'lib': INK}
NAME = {'counts': 'Raw counts', 'norm': 'Normalised', 'smooth': 'Normalised + smoothing', 'liana_cosine': 'LIANA+ cosine',
        'liana_morans': 'LIANA+ local Moran', 'norm_r': 'Normalised + residualised', 'model': 'stCCC', 'lib': 'Log library size'}
LAYERS = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
LCOL = dict(zip(LAYERS, [BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, PURPLE]))   # validated categorical order
SEQ = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f2a19f', '#c23a39'])

plt.rcParams.update({
    'font.family': 'Nimbus Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Nimbus Sans',
    'mathtext.it': 'Nimbus Sans:italic', 'mathtext.bf': 'Nimbus Sans:bold', 'mathtext.cal': 'cmsy10',
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.titlepad': 3, 'axes.labelpad': 2.5, 'legend.frameon': False,
    'legend.handletextpad': 0.4, 'legend.columnspacing': 1.0, 'pdf.fonttype': 42, 'savefig.dpi': 300})


def panel(fig, ax, s, dx=-0.0, dy=0.0):
    """Panel letter at the top-left corner of the axes' tight bounding box, in figure coordinates."""
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=9, fontweight='bold', va='bottom', ha='left')


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=9, fontweight='bold', va='top', ha='left')


def pct(v):
    return (rankdata(v) - 0.5) / len(v)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def spotmap(ax, xy, v, cmap, title=None, s=1.6, vmin=0, vmax=1):
    o = np.argsort(v)
    ax.scatter(xy[o, 0], -xy[o, 1], c=v[o], cmap=cmap, s=s, lw=0, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, pad=2)


def layermap(ax, xy, lab, title=None, s=1.6):
    for l in LAYERS:
        m = lab == l
        ax.scatter(xy[m, 0], -xy[m, 1], c=LCOL[l], s=s, lw=0, rasterized=True)
    m = ~np.isin(lab, LAYERS)
    ax.scatter(xy[m, 0], -xy[m, 1], c=GREY, s=s, lw=0, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, pad=2)


def layer_legend(fig, x, y, layers=LAYERS, ncol=None, loc='center', **kw):
    h = [plt.Line2D([], [], ls='none', marker='o', ms=4.2, mec='none', color=LCOL[l],
                    label='WM' if l == 'WM' else l.replace('Layer', 'L')) for l in layers]
    return fig.legend(handles=h, ncol=ncol or len(h), loc=loc, bbox_to_anchor=(x, y), handletextpad=0.1,
                      columnspacing=0.7, **kw)


def hbar(fig, cax, cmap, ticks, labels, title=None, vmin=0, vmax=1):
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_clim(vmin, vmax)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
    cb.set_ticks(ticks); cb.set_ticklabels(labels); cb.ax.tick_params(length=0, labelsize=6.2, pad=1.5)
    if title: cax.set_title(title, fontsize=6.5, color=INK2, pad=2)
    return cb


def save(fig, out, name):
    """Save PDF + PNG. The PDF is re-written by ghostscript so that the OpenType (CFF) fonts are embedded with a
    matching font type (matplotlib writes them as TrueType), without downsampling the rasterised layers."""
    import os, subprocess
    raw = f'{out}/{name}.raw.pdf'
    fig.savefig(raw); fig.savefig(f'{out}/{name}.png', dpi=220); plt.close(fig)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite', '-dEmbedAllFonts=true', '-dSubsetFonts=true',
                    '-dCompatibilityLevel=1.5', '-dAutoRotatePages=/None', '-dDownsampleColorImages=false',
                    '-dDownsampleGrayImages=false', '-dDownsampleMonoImages=false', '-dAutoFilterColorImages=false',
                    '-dColorImageFilter=/FlateEncode', f'-sOutputFile={out}/{name}.pdf', raw], check=True)
    os.remove(raw)


def fmt_p(p):
    if p >= 0.001: return f'$P$ = {p:.3f}'
    m, e = f'{p:.1e}'.split('e'); return f'$P$ = {m}$\\times$10$^{{{int(e)}}}$'


def bracket(ax, a, b, y, text, h=None, vertical=True, fs=6):
    """Significance bracket between positions a and b at height y (data units)."""
    h = h if h is not None else 0.02 * np.ptp(ax.get_ylim() if vertical else ax.get_xlim())
    if vertical:
        ax.plot([a, a, b, b], [y, y + h, y + h, y], color=INK2, lw=0.6, clip_on=False)
        ax.text((a + b) / 2, y + h, text, ha='center', va='bottom', fontsize=fs, color=INK2)
    else:
        ax.plot([y, y + h, y + h, y], [a, a, b, b], color=INK2, lw=0.6, clip_on=False)
        ax.text(y + h * 1.4, (a + b) / 2, text, ha='left', va='center', fontsize=fs, color=INK2, rotation=0)


def raincloud(ax, data, pos, cols, horizontal=True, width=0.42, seed=0, s=3, alpha=0.55, box=True):
    """Half violin (cloud) + narrow box + jittered points (rain) per group."""
    rng = np.random.default_rng(seed)
    for d, p, c in zip(data, pos, cols):
        d = np.asarray(d); d = d[np.isfinite(d)]
        v = ax.violinplot([d], positions=[p], vert=not horizontal, widths=width * 2, showextrema=False)
        for b in v['bodies']:
            path = b.get_paths()[0].vertices
            if horizontal: path[:, 1] = np.clip(path[:, 1], p, np.inf)
            else: path[:, 0] = np.clip(path[:, 0], p, np.inf)
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.75)
        if box:
            q1, md, q3 = np.percentile(d, [25, 50, 75])
            if horizontal:
                ax.plot([q1, q3], [p - 0.06, p - 0.06], color=INK, lw=2.2, solid_capstyle='butt', zorder=4)
                ax.scatter([md], [p - 0.06], s=10, color='white', edgecolor=INK, lw=0.6, zorder=5)
            else:
                ax.plot([p - 0.06, p - 0.06], [q1, q3], color=INK, lw=2.2, solid_capstyle='butt', zorder=4)
                ax.scatter([p - 0.06], [md], s=10, color='white', edgecolor=INK, lw=0.6, zorder=5)
        j = p - 0.16 - rng.uniform(0, 0.16, len(d))
        if horizontal: ax.scatter(d, j, s=s, color=c, alpha=alpha, lw=0, zorder=3, rasterized=True)
        else: ax.scatter(j, d, s=s, color=c, alpha=alpha, lw=0, zorder=3, rasterized=True)


def radar(ax, labels, series, cols, names, rmin=0.0, rmax=1.0, ticks=(0.25, 0.5, 0.75, 1.0)):
    """Radar chart on a polar axes."""
    n = len(labels); ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist(); ang += ang[:1]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(rmin, rmax); ax.set_yticks(list(ticks)); ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=5.5, color=INK3)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=6.3)
    ax.grid(color=RULE, lw=0.6); ax.spines['polar'].set_color(GREY); ax.spines['polar'].set_linewidth(0.6)
    for v, c, nm in zip(series, cols, names):
        v = list(v) + [v[0]]
        ax.plot(ang, v, color=c, lw=1.4, label=nm, zorder=3); ax.fill(ang, v, color=c, alpha=0.12, zorder=2)
        ax.scatter(ang[:-1], v[:-1], s=12, color=c, edgecolor='white', lw=0.5, zorder=4)
