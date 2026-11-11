"""Second dataset: isocortex of the Zhuang ABCA-1 MERFISH mouse brain (1,122-gene panel, single-cell resolution).

Twelve pairs of adjacent coronal sections (71-147 um apart) at twelve anterior-posterior positions play the role
of the DLPFC serial pairs; the positions play the role of donors (the first submission used four of these pairs). Only isocortex cells are kept, and each
cell's cortical layer (L1, L2/3, L4, L5, L6a, L6b) is read from its CCF substructure, so the laminar analyses
carry over unchanged. Coordinates are the registered CCF (ml, dv) in mm, so adjacent sections are aligned.
Gene symbols are upper-cased to match the human ligand-receptor resource.
Writes data/abca1/sample_<section>_with_graph.h5ad (integer counts in X) and sample_<section>_labels.npy."""
import os, re
import numpy as np, pandas as pd, anndata as ad
PREP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'abca1', 'prep')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'abca1')
PAIRS = [(24, 25), (28, 29), (34, 35), (40, 41), (46, 47), (52, 53), (58, 59), (64, 65), (68, 69), (76, 77), (84, 85), (88, 89)]   # = datasets.ADJ
LAYER = re.compile(r'(1|2/3|4|5|6a|6b)$')

obs = pd.read_parquet(f'{PREP}/obs.parquet', columns=['section', 'division', 'substructure', 'ccf_ml', 'ccf_dv', 'subclass'])
genes = [g.strip().upper() for g in open(f'{PREP}/genes.txt')]
X = np.load(f'{PREP}/counts_u16.npy', mmap_mode='r')
assert X.shape == (len(obs), len(genes))
os.makedirs(OUT, exist_ok=True)
for sec in [s for p in PAIRS for s in p]:
    if os.path.exists(f'{OUT}/sample_{sec}_with_graph.h5ad'):
        continue                                                                 # already prepared
    sel = np.flatnonzero(((obs.section == sec) & (obs.division == 'Isocortex')).to_numpy())
    lay = np.array([('L' + m.group(1)) if (m := LAYER.search(str(v))) else None for v in obs.substructure.to_numpy()[sel]], dtype=object)
    sel, lay = sel[lay != None], lay[lay != None]                                # noqa: E711
    o = obs.iloc[sel]
    a = ad.AnnData(np.asarray(X[sel], dtype='float32'), obs=pd.DataFrame(index=o.index.astype(str)))
    o = o.assign(layer=lay)
    a.var_names = genes; a.var_names_make_unique()
    a.obsm['spatial'] = o[['ccf_ml', 'ccf_dv']].to_numpy('float64')
    a.obs['label'] = o.layer.values; a.obs['subclass'] = o.subclass.astype(str).values
    a = a[a.X.sum(1) > 0].copy()
    a.write_h5ad(f'{OUT}/sample_{sec}_with_graph.h5ad')
    np.save(f'{OUT}/sample_{sec}_labels.npy', a.obs['label'].to_numpy(dtype=object), allow_pickle=True)
    print(sec, a.n_obs, 'cells', a.obs.label.value_counts().to_dict(), 'median counts', float(np.median(a.X.sum(1))), flush=True)
