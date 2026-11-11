"""Third dataset: HER2-positive breast cancer, Spatial Transcriptomics (Andersson et al. 2021, her2st).

Sections: the pathologist-annotated section of each of the eight patients (A1, B1, C1, D1, E1, F1, G2, H1) and, for
patients E-H, whose three sections are adjacent (patients A-D have six evenly spaced sections, not adjacent), the two
other sections (E2, E3, F2, F3, G1, G3, H2, H3). Section ids: 10 * patient index + section number (A1 = 11, ..., H3 = 83).

Genes (one gene set for all sections): genes present in all 16 count tables; the 3,000 most variable (scanpy seurat_v3
on raw counts, batch_key = section) plus every gene of the LR resource detected in >= 2% of the pooled spots.
Spots: tissue spots of the spot-selection files; coordinates are the adjusted array coordinates (new_x, new_y).
Labels: pathologist regions; 'undetermined' -> 'NA'. For the unannotated sections the labels are transferred from the
annotated section of the same patient by PASTE pairwise alignment (Zeira et al. 2022; alpha = 0.1): each target spot
takes the region with the largest transported mass, and spots receiving < 50% of the uniform mass are set to 'NA'.
Round-trip check: labels transferred annotated -> target -> annotated are compared with the original annotation.

Writes data/her2/sample_<id>_with_graph.h5ad (integer counts in X) and sample_<id>_labels.npy, and
data/her2/label_transfer.csv. Run in envs/paste_env:  python prep_her2.py"""
import os
import numpy as np, pandas as pd, scanpy as sc, anndata as ad

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'her2st')
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'her2')
ANNOT = {'A': 'A1', 'B': 'B1', 'C': 'C1', 'D': 'D1', 'E': 'E1', 'F': 'F1', 'G': 'G2', 'H': 'H1'}
EXTRA = ['E2', 'E3', 'F2', 'F3', 'G1', 'G3', 'H2', 'H3']
SECS = list(ANNOT.values()) + EXTRA
sid = lambda s: 10 * ('ABCDEFGH'.index(s[0]) + 1) + int(s[1])


def load(s):
    c = pd.read_csv(f'{SRC}/ST-cnts/{s}.tsv.gz', sep='\t', index_col=0)
    sp = pd.read_csv(f'{SRC}/ST-spotfiles/{s}_selection.tsv', sep='\t')
    sp.index = [f'{a}x{b}' for a, b in zip(sp.x, sp.y)]
    sp = sp[sp.selected == 1]
    keep = c.index.intersection(sp.index)
    c, sp = c.loc[keep], sp.loc[keep]
    c = c.T.groupby(level=0).sum().T                     # duplicated gene symbols -> summed
    a = ad.AnnData(c.values.astype('float32'), obs=pd.DataFrame(index=[f'{s}_{i}' for i in c.index]))
    a.var_names = list(c.columns); a.obsm['spatial'] = sp[['new_x', 'new_y']].to_numpy('float64'); a.obs['section'] = s
    lab = np.array(['NA'] * a.n_obs, dtype=object)
    f = f'{SRC}/ST-pat/lbl/{s}_labeled_coordinates.tsv'
    if os.path.exists(f):
        lb = pd.read_csv(f, sep='\t').dropna(subset=['x'])
        lb.index = [f'{int(round(x))}x{int(round(y))}' for x, y in zip(lb.x, lb.y)]
        m = lb.label.reindex(c.index)
        lab = np.array(['NA' if (pd.isnull(v) or v == 'undetermined') else v for v in m], dtype=object)
    a.obs['label'] = lab
    return a


def transfer(src, tgt):
    """PASTE pairwise alignment; returns transferred labels for tgt and the share of tgt spots that were assigned."""
    # PASTE pairwise alignment (Zeira et al. 2022), written directly against POT because the paste-bio package no
    # longer runs with current POT: fused Gromov-Wasserstein with expression cost = KL divergence between the spots'
    # expression profiles (pseudocount 0.01), structure = Euclidean distances within each section, uniform marginals.
    import ot
    from scipy.spatial.distance import cdist
    X = np.asarray(src.X, 'float64') + 0.01; Y = np.asarray(tgt.X, 'float64') + 0.01
    X /= X.sum(1, keepdims=True); Y /= Y.sum(1, keepdims=True)
    M = (X * np.log(X)).sum(1)[:, None] - X @ np.log(Y).T
    DA, DB = cdist(src.obsm['spatial'], src.obsm['spatial']), cdist(tgt.obsm['spatial'], tgt.obsm['spatial'])
    a, b = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))
    pi = ot.gromov.fused_gromov_wasserstein(M, DA, DB, a, b, loss_fun='square_loss', alpha=0.1, max_iter=200)
    labs = sorted(set(src.obs.label) - {'NA'})
    M = np.stack([pi[(src.obs.label == l).to_numpy()].sum(0) for l in labs], 1)       # (n_tgt, n_labels)
    tot = pi.sum(0); out = np.array([labs[i] for i in M.argmax(1)], dtype=object)
    out[tot < 0.5 / tgt.n_obs] = 'NA'
    out[M.max(1) <= 0] = 'NA'
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    A = {s: load(s) for s in SECS}
    common = sorted(set.intersection(*[set(a.var_names) for a in A.values()]))
    pool = ad.concat([a[:, common] for a in A.values()], join='inner')
    pool.obs_names_make_unique()
    sc.pp.highly_variable_genes(pool, flavor='seurat_v3', n_top_genes=3000, batch_key='section')
    db = pd.read_csv(DB); lrg = set(db['source'].dropna()) | set(db['target'].dropna())
    det = np.asarray((pool.X > 0).mean(0)).ravel()
    genes = [g for g, h, d in zip(common, pool.var.highly_variable, det) if h or (g in lrg and d >= 0.02)]
    print(len(common), 'common genes ->', len(genes), 'kept (', sum(g in lrg for g in genes), 'LR-resource genes )', flush=True)
    rows = []
    for p, s0 in ANNOT.items():
        for s in [x for x in EXTRA if x[0] == p]:
            src, tgt = A[s0][:, genes].copy(), A[s][:, genes].copy()
            tgt.obs['label'] = transfer(src, tgt)
            back = transfer(tgt, src)                                                   # round trip on the annotated section
            ok = (src.obs.label.values != 'NA') & (back != 'NA')
            acc = float((back[ok] == src.obs.label.values[ok]).mean())
            A[s].obs['label'] = tgt.obs['label'].values
            rows.append(dict(source=s0, target=s, n_target=tgt.n_obs, assigned=float((tgt.obs.label != 'NA').mean()),
                             round_trip_accuracy=acc, **{f'n_{k}': int(v) for k, v in tgt.obs.label.value_counts().items()}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/label_transfer.csv', index=False)
    for s, a in A.items():
        b = a[:, genes].copy(); b.X = np.rint(b.X).astype('float32')
        b = b[np.asarray(b.X.sum(1)).ravel() > 0].copy()
        b.write_h5ad(f'{OUT}/sample_{sid(s)}_with_graph.h5ad')
        np.save(f'{OUT}/sample_{sid(s)}_labels.npy', b.obs['label'].to_numpy(dtype=object), allow_pickle=True)
        print(s, sid(s), b.n_obs, 'spots, median library', float(np.median(b.X.sum(1))), b.obs.label.value_counts().to_dict(), flush=True)
