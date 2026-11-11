"""Published spatial CCC tool as a baseline for the reliability analysis: LIANA+ bivariate local scores.
  liana_cosine : LIANA+ default local metric (spatially weighted cosine similarity)
  liana_morans : local bivariate Moran's I (the SpatialDM statistic)
Same count versions as ccc_reliability.py (identical binomial thinning, seed 0), same LR pairs, same layers.
Input is log1p counts-per-10k, the LIANA+ default; spatial weights from li.ut.spatial_neighbors with the
bandwidth set to twice the median nearest-neighbour distance (about one Visium ring).
Saved per section/version/metric: per-pair layer means (P x L), per-spot total score, per-pair depth rho."""
import os, sys, pickle, argparse, warnings
import numpy as np, pandas as pd, scanpy as sc, liana as li
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model'))
from ccc_graph import lr_pairs
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')
from datasets import DATA, RESULTS
OUT = os.path.join(RESULTS, 'reliability_liana')


def run_liana(Rv, genes, xy, bw, pairs, local):
    ad = sc.AnnData(Rv.copy()); ad.var_names = genes; ad.var_names_make_unique(); ad.obsm['spatial'] = xy
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    li.ut.spatial_neighbors(ad, bandwidth=bw, cutoff=0.1, set_diag=True)
    res = pd.DataFrame(pairs, columns=['ligand', 'receptor'])
    lr = li.mt.bivariate(ad, resource=res, local_name=local, nz_prop=0.0, use_raw=False, mask_negatives=False,
                         add_categories=False, remove_self_interactions=True, verbose=False)
    S = pd.DataFrame(np.asarray(lr.X.todense() if hasattr(lr.X, 'todense') else lr.X), columns=lr.var_names)
    return S.reindex(columns=[f'{l}^{r}' for l, r in pairs]).fillna(0.0).values   # (N, P)


def summarise(S, lib, labels, layers):
    ok = ~pd.isnull(labels); dep = np.log(lib + 1.0)
    LM = np.full((S.shape[1], len(layers)), np.nan)
    for q, lay in enumerate(layers):
        sel = ok & (labels == lay)
        LM[:, q] = S[sel].mean(0)
    rho = np.array([spearmanr(S[:, p], dep)[0] if S[:, p].std() > 0 else np.nan for p in range(S.shape[1])])
    tot = S.sum(1)
    return dict(LM=LM, total=tot, rho=rho, rho_total=float(spearmanr(tot, dep)[0]))


def main(sample):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    xy = ad.obsm['spatial'].astype('float64')
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype('float64')
    genes = list(ad.var_names)
    labels = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)
    layers = sorted({str(x) for x in labels if not pd.isnull(x)})
    labels = np.array([None if pd.isnull(x) else str(x) for x in labels], dtype=object)
    rng = np.random.default_rng(0)                       # identical thinning to ccc_reliability.py
    A = rng.binomial(raw.astype(np.int64), 0.5).astype('float64')
    versions = dict(full=raw, A=A, B=raw - A, d25=rng.binomial(raw.astype(np.int64), 0.25).astype('float64'))
    bw = 2 * float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
    pairs = lr_pairs(genes, DB)
    out = dict(sample=sample, pairs=pairs, layers=layers, out={})
    for v, Rv in versions.items():
        out['out'][v] = {f'liana_{m}': summarise(run_liana(Rv, genes, xy, bw, pairs, m), Rv.sum(1), labels, layers)
                         for m in ('cosine', 'morans')}
        print(sample, v, {k: round(o['rho_total'], 3) for k, o in out['out'][v].items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{sample}.pkl'), 'wb') as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--samples', nargs='+', required=True)
    for s in ap.parse_args().samples:
        main(s)
