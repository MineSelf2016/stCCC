"""Whole-brain MERFISH (Zhuang-ABCA-1, Yao/Zhuang 2023 ABC atlas) as one registered 3D stack.

Cells: the 2.85M QC-passed cells of cell_metadata_with_cluster_annotation, restricted to the 92%
that carry CCF coordinates (one hemisphere, CCF x = anterior-posterior, y = dorsal-ventral,
z = medial-lateral, mm). The metadata's own reconstructed z repeats values across sections, so
CCF is used for all three axes; parcellation_index gives the Allen CCF division / structure /
substructure of every cell, which the scan never sees.

Isotropy, as for the other real stacks: anterior-posterior is compressed so that the spacing of
adjacent sections equals the median in-plane nearest-neighbour distance (measured within each
section). Coordinates are kept in mm in `ccf` and the scaled version in `iso`.

Counts: X of the raw h5ad (dense float32, gzip chunks of 16281 rows x 5 genes) read column block
by column block, checked integer, stored as a uint16 memmap ordered by section.

-> data/raw/abca1/prep/counts_u16.npy (n x 1122), data/raw/abca1/prep/obs.parquet, data/raw/abca1/prep/genes.txt,
   data/raw/abca1/prep/meta.json
"""
import json, os, time
import numpy as np, pandas as pd, h5py
from sklearn.neighbors import NearestNeighbors

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'abca1')
O = os.path.join(R, 'prep')
os.makedirs(O, exist_ok=True)
t0 = time.time()

m = pd.read_csv(f"{R}/cell_metadata_with_cluster_annotation.csv", dtype={"cell_label": str},
                usecols=["cell_label", "brain_section_label", "neurotransmitter", "class",
                         "subclass", "supertype", "cluster"])
c = pd.read_csv(f"{R}/ccf_coordinates.csv", dtype={"cell_label": str})
c = c.rename(columns={"x": "ccf_ap", "y": "ccf_dv", "z": "ccf_ml"})
par = pd.read_csv(f"{R}/parcellation_to_parcellation_term_membership_acronym.csv")
par = par.set_index("parcellation_index")[["division", "structure", "substructure"]]
obs = m.merge(c, on="cell_label", how="inner")
obs = obs.join(par, on="parcellation_index")
print(f"{len(m)} QC cells, {len(obs)} with CCF ({len(obs) / len(m):.3f})", flush=True)

# section order along anterior-posterior, by median CCF AP of each section
sec_ap = obs.groupby("brain_section_label")["ccf_ap"].median().sort_values()
obs["section"] = obs["brain_section_label"].map({s: i for i, s in enumerate(sec_ap.index)})
obs = obs.sort_values(["section", "ccf_dv", "ccf_ml"]).reset_index(drop=True)

# in-plane nearest-neighbour distance, within each section
nn = []
for s, g in obs.groupby("section"):
    xy = g[["ccf_dv", "ccf_ml"]].to_numpy()
    d = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]
    nn.append(np.median(d))
dxy = float(np.median(nn))
dz = float(np.median(np.diff(sec_ap.to_numpy())))
f = dxy / dz
obs["iso_ap"] = obs["ccf_ap"] * f
obs["iso_dv"] = obs["ccf_dv"]
obs["iso_ml"] = obs["ccf_ml"]
print(f"in-plane NN {dxy * 1000:.1f} um, section spacing {dz * 1000:.1f} um, AP scaled x{f:.3f}",
      flush=True)

# counts: rows of the h5ad that are kept, in obs order
h = h5py.File(f"{R}/Zhuang-ABCA-1-raw.h5ad", "r")
lab = pd.Index(h["obs"]["cell_label"][:].astype(str))
rows = lab.get_indexer(obs["cell_label"])
assert (rows >= 0).all()
genes = h["var"]["gene_symbol"][:].astype(str)
X = h["X"]
n, G = len(obs), X.shape[1]
out = np.lib.format.open_memmap(f"{O}/counts_u16.npy", mode="w+", dtype=np.uint16, shape=(n, G))
step = X.chunks[1]
maxv, nonint = 0.0, 0
for j in range(0, G, step):
    blk = X[:, j:j + step][rows]
    nonint += int((blk != np.round(blk)).sum())
    maxv = max(maxv, float(blk.max()))
    out[:, j:j + step] = np.clip(np.round(blk), 0, 65535).astype(np.uint16)
    if j % 100 == 0:
        print(f"  genes {j}/{G}  {time.time() - t0:.0f}s", flush=True)
out.flush()
print(f"max count {maxv}, non-integer entries {nonint}", flush=True)

obs.drop(columns=["cell_label"]).to_parquet(f"{O}/obs.parquet")
obs[["cell_label"]].to_parquet(f"{O}/cell_label.parquet")
open(f"{O}/genes.txt", "w").write("\n".join(genes))
json.dump(dict(n_cells=int(n), n_genes=int(G), n_sections=int(obs.section.nunique()),
               nn_inplane_mm=dxy, section_spacing_mm=dz, ap_scale=f, max_count=maxv,
               non_integer=nonint, seconds=round(time.time() - t0)),
          open(f"{O}/meta.json", "w"), indent=1)
print("done", round(time.time() - t0), "s")
