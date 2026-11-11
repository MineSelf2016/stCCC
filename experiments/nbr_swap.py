"""Neighbourhood swap: does it matter whether an input borrows information from physical or from transcriptional neighbours?

Variants (each saved as the read-out input, log1p per-10k, to RESULTS/denoised_<variant>/<sample>.npy and then scanned by
iface_cond.py exactly like the other inputs; the interface-scan edges themselves stay physical):
  magic_sp   : MAGIC with its diffusion operator built on spatial coordinates instead of expression
               (same settings as the MAGIC baseline: knn = 5, decay = 1, t = 3)
  model_xnbr : stCCC whose neighbour graph (attention, consistency term, LR edge scores) is the 8-nearest-neighbour
               graph in expression space (first 30 principal components of log-normalised expression); the GP prior
               and the gating unit still see spot coordinates
  model_xall : as model_xnbr, and the coordinates seen by the GP prior and the gating unit are replaced by a 2-D
               expression embedding (first two principal components, rescaled to [0, 20] like coordinates), so that
               every component pools over transcriptional neighbours
Usage: STCCC_DATASET=<d> python nbr_swap.py --variant model_xnbr --samples sample_X ...   (magic_sp needs env stccc_magic)"""
import os, sys, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'model'))
from rel_common import load, per10k, DB
from datasets import RESULTS


def expr_pcs(raw, n):
    from sklearn.decomposition import PCA
    E = per10k(raw); E = (E - E.mean(0)) / np.where(E.std(0) > 0, E.std(0), 1.0)
    return PCA(n, random_state=0).fit_transform(E)


def model_swap(raw, genes, loc, variant, seed=0):
    import torch
    from ccc_graph import build_scores
    from stccc import stCCC
    from ccc_reliability import prep
    from datasets import KL
    from sklearn.preprocessing import MinMaxScaler
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    P30 = expr_pcs(raw, 30)
    nbr, dist, S, _ = build_scores(raw, genes, P30, 'lr', DB, k=8, seed=seed)   # neighbours in expression space
    x = loc if variant == 'model_xnbr' else MinMaxScaler().fit_transform(P30[:, :2]) * 20.
    X, sf = prep(raw)
    torch.manual_seed(seed); np.random.seed(seed)
    ip = np.mgrid[0:(1 + 1e-5):(1. / 8), 0:(1 + 1e-5):(1. / 8)].reshape(2, -1).T * 20.
    m = stCCC(input_dim=raw.shape[1], GP_dim=2, Normal_dim=8, encoder_layers=[128, 64], decoder_layers=[128],
                   inducing_points=ip, kernel_scale=20., N_train=raw.shape[0], nbr=nbr, nbr_dist=dist, score=S,
                   sigma_sp=float(np.median(dist)), use_attention=True, gamma=3.0, lam=10.0, use_gating=True, device=dev, **KL)
    m.fit(x, X, raw, sf, maxiter=100)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    return per10k(m.denoise(T(x), T(X)))


def magic_spatial(raw, loc, seed=0):
    import magic, graphtools
    norm = per10k(raw)
    G = graphtools.Graph(loc, knn=5, decay=1, n_pca=None, random_state=seed, verbose=0)
    Y = magic.MAGIC(knn=5, decay=1, t=3, random_state=seed, verbose=0).fit(norm, graph=G).transform(norm)
    return per10k(np.expm1(np.clip(np.asarray(Y), 0, None)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, choices=['magic_sp', 'model_xnbr', 'model_xall'])
    ap.add_argument('--samples', nargs='+', required=True); a = ap.parse_args()
    out = os.path.join(RESULTS, f'denoised_{a.variant}'); os.makedirs(out, exist_ok=True)
    for s in a.samples:
        ad, loc, raw, genes, labels = load(s)
        E = magic_spatial(raw, loc) if a.variant == 'magic_sp' else model_swap(raw, genes, loc, a.variant)
        np.save(os.path.join(out, f'{s}.npy'), E.astype('float32')); print(s, a.variant, 'saved', E.shape, flush=True)
