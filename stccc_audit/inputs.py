"""Expression inputs that can be audited. An input is any callable f(counts, section) -> (N, G) non-negative expression
on the log1p per-10k scale; the audit refits it on every count version. Built-in inputs:
  norm        log1p counts per 10k (CellChat/COMMOT-style input)
  smooth      norm averaged over each unit and its k scoring-graph neighbours
  magic       MAGIC diffusion of norm (knn 5, decay 1, t 3), back-transformed      [needs the `magic-impute` package]
  scvi        scVI decoder mean (NB likelihood, 10 latent dimensions)               [needs `scvi-tools`]
  stccc  stCCC decoder rate, frozen configuration of the manuscript     [needs torch; uses ../model]"""
import os, sys, math
import numpy as np
from .core import per10k

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, 'data', 'consensus.csv')


def norm(counts, sec):
    return per10k(counts)


def smooth(counts, sec):
    E = per10k(counts)
    return (E + E[sec.nbr].sum(1)) / (sec.nbr.shape[1] + 1)


def magic(counts, sec, seed=0):
    import magic as _magic
    Y = _magic.MAGIC(random_state=seed, n_jobs=4, verbose=0).fit_transform(per10k(counts))
    return per10k(np.expm1(np.clip(np.asarray(Y), 0, None)))


def scvi(counts, sec, seed=0):
    import scvi as _scvi, anndata
    _scvi.settings.seed = seed
    ad = anndata.AnnData(np.asarray(counts, 'float32'))
    _scvi.model.SCVI.setup_anndata(ad)
    m = _scvi.model.SCVI(ad, n_latent=10, gene_likelihood='nb')
    bs = 128
    while any(n % bs == 1 for n in (math.ceil(0.9 * len(counts)), math.floor(0.9 * len(counts)))):
        bs += 1
    m.train(max_epochs=400, early_stopping=True, enable_progress_bar=False, batch_size=bs)
    return per10k(np.asarray(m.get_normalized_expression(library_size=1e4, return_numpy=True)))


def stccc(counts, sec, seed=0, db=DEFAULT_DB, device=None, return_model=False):
    """stCCC with the frozen configuration of the manuscript (lambda 10, gamma 3, KL target 0.025, 8-NN graph,
    at least 1,200 gradient steps); coordinates rescaled to [0, 20]."""
    import torch
    sys.path.insert(0, os.path.join(HERE, '..', 'model'))
    from sklearn.preprocessing import MinMaxScaler
    from ccc_graph import build_scores
    from stccc import stCCC
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    raw = np.asarray(counts, 'float64')
    loc = MinMaxScaler().fit_transform(sec.xy) * 20.
    nbr, dist, S, _ = build_scores(raw, sec.genes, loc, 'lr', db, k=8, seed=seed)
    lib = raw.sum(1); sf = np.maximum(lib, 1.0) / np.median(lib)
    X = np.log1p(raw / sf[:, None]); X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    torch.manual_seed(seed); np.random.seed(seed)
    ip = np.mgrid[0:(1 + 1e-5):(1. / 8), 0:(1 + 1e-5):(1. / 8)].reshape(2, -1).T * 20.
    m = stCCC(input_dim=raw.shape[1], GP_dim=2, Normal_dim=8, encoder_layers=[128, 64], decoder_layers=[128],
                   inducing_points=ip, kernel_scale=20., N_train=raw.shape[0], nbr=nbr, nbr_dist=dist, score=S,
                   sigma_sp=float(np.median(dist)), use_attention=True, gamma=3.0, lam=10.0, use_gating=True, device=device,
                   KL_loss=0.025, min_beta=4, init_beta=10)
    m.fit(loc, X, raw, sf, maxiter=100)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    E = per10k(m.denoise(T(loc), T(X)))
    return (E, m) if return_model else E


BUILTIN = dict(norm=norm, smooth=smooth, magic=magic, scvi=scvi, stccc=stccc)
