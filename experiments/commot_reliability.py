"""Second published spatial CCC tool as a baseline for the reliability analysis: COMMOT.
Collective optimal transport between ligand senders and receptor receivers within a distance cut-off.
Same count versions as ccc_reliability.py (identical binomial thinning, seed 0), same LR pairs, same layers.
Input is log1p counts-per-10k, the COMMOT tutorial default; the cut-off is twice the median
nearest-neighbour distance (about one Visium ring), the same range as the LIANA+ bandwidth.
Per-spot score per pair = sent + received signal, the per-spot summary COMMOT reports.
  commot   : as COMMOT returns it
  commot_r : the same score with log library size regressed out per pair (depth-residualised)
Saved per section/version/metric: per-pair layer means (P x L), per-spot total score, per-pair depth rho."""
import os, pickle, argparse, warnings
import numpy as np, pandas as pd, scanpy as sc, commot as ct
from sklearn.neighbors import NearestNeighbors
from rel_common import HERE, DB, DATA, RESULTS, versions, lr_pairs
from liana_summary import summarise
warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, 'reliability_commot')


def run_commot(Rv, genes, xy, thr, pairs):
    ad = sc.AnnData(Rv.copy()); ad.var_names = genes; ad.var_names_make_unique(); ad.obsm['spatial'] = xy
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    lr = pd.DataFrame([(l, r, l) for l, r in pairs], columns=['ligand', 'receptor', 'pathway'])
    ct.tl.spatial_communication(ad, database_name='db', df_ligrec=lr, dis_thr=thr, heteromeric=False, pathway_sum=False)
    S = np.zeros((ad.n_obs, len(pairs)))
    for p, (l, r) in enumerate(pairs):
        k = f'{l}-{r}'
        s_col, r_col = f's-{k}', f'r-{k}'
        snd = ad.obsm['commot-db-sum-sender']; rcv = ad.obsm['commot-db-sum-receiver']
        S[:, p] = (snd[s_col].values if s_col in snd else 0) + (rcv[r_col].values if r_col in rcv else 0)
    return S


def depth_resid(S, lib):
    X = np.stack([np.ones_like(lib), np.log(lib + 1.0)], 1)
    b, *_ = np.linalg.lstsq(X, S, rcond=None)
    return S - X @ b


def main(sample):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    xy = ad.obsm['spatial'].astype('float64')
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    genes = list(ad.var_names)
    labels = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)
    layers = sorted({str(x) for x in labels if not pd.isnull(x)})
    labels = np.array([None if pd.isnull(x) else str(x) for x in labels], dtype=object)
    thr = 2 * float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
    pairs = lr_pairs(genes, DB)
    out = dict(sample=sample, pairs=pairs, layers=layers, out={})
    for v, Rv in versions(raw).items():
        S = run_commot(Rv, genes, xy, thr, pairs); lib = Rv.sum(1)
        out['out'][v] = dict(commot=summarise(S, lib, labels, layers), commot_r=summarise(depth_resid(S, lib), lib, labels, layers))
        print(sample, v, {k: round(o['rho_total'], 3) for k, o in out['out'][v].items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{sample}.pkl'), 'wb') as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='+', required=True)
    for s in ap.parse_args().samples:
        main(s)
