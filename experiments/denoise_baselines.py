"""Denoise-then-score baselines for the reliability analysis.

Question: is a de-confounded, reproducible communication score obtained by any denoiser followed by
the same read-out, or does it need stCCC? Each baseline replaces the stCCC decoder rate by
another estimate of expression and is then scored exactly as the model is (per-10k, log1p edge product,
depth residualisation), on the same count versions, LR pairs, graph and layers.
  smooth_r : one-ring spatial smoothing of log-normalised expression, then residualised   (no training)
  scvi     : scVI decoder mean (NB likelihood, library-free normalised expression)          (non-spatial VAE)
  magic    : MAGIC diffusion of log-normalised expression, back-transformed                 (graph imputation)
Each method is saved residualised (<name>) and, for scvi and magic, unresidualised (<name>_nr)."""
import os, pickle, argparse, warnings
import numpy as np
from rel_common import HERE, DB, RESULTS, load, versions, per10k, summarise, knn_graph, lr_pairs
warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, 'reliability_denoise')


def denoise(method, Rv, nbr, seed=0):
    """Expression on the scale the read-out uses: log1p counts-per-10k (smooth_r is exactly the
    'smooth' input of ccc_reliability.py; scvi and magic are re-normalised like the model rate)."""
    norm = per10k(Rv)
    if method == 'smooth_r':
        return (norm + norm[nbr].sum(1)) / (nbr.shape[1] + 1)
    if method == 'scvi':
        import scvi, anndata
        scvi.settings.seed = seed
        ad = anndata.AnnData(Rv.astype('float32'))
        scvi.model.SCVI.setup_anndata(ad)
        m = scvi.model.SCVI(ad, n_latent=10, gene_likelihood='nb')
        import math
        bs = 128                                   # avoid a 1-cell last batch (BatchNorm fails); only changes bs when it would occur
        while any(n % bs == 1 for n in (math.ceil(0.9 * len(Rv)), math.floor(0.9 * len(Rv)))):
            bs += 1
        m.train(max_epochs=400, early_stopping=True, enable_progress_bar=False, batch_size=bs)
        return per10k(np.asarray(m.get_normalized_expression(library_size=1e4, return_numpy=True)))
    if method in ('noCCC', 'attn', 'noGate', 'lr_noGate'):                      # stCCC ablations: no attention / no communication term
        from ccc_reliability import train_denoise
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        return per10k(train_denoise(Rv, GENES, LOC, dev, seed=seed, cond=method))
    if method == 'magic':
        import magic
        Y = magic.MAGIC(random_state=seed, n_jobs=4, verbose=0).fit_transform(norm)
        return per10k(np.expm1(np.clip(np.asarray(Y), 0, None)))
    raise ValueError(method)


def save_full(sample, method):
    """Denoise the full counts only and save the read-out input (log1p per-10k) for interface_scan.py."""
    global GENES, LOC
    ad, loc, raw, genes, labels = load(sample)
    GENES, LOC = genes, loc
    nbr, _ = knn_graph(loc, 8)
    out = os.path.join(RESULTS, f'denoised_{method}')
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f'{sample}.npy'), denoise(method, raw, nbr).astype('float32'))
    print(sample, method, 'saved', flush=True)


def main(sample, method):
    ad, loc, raw, genes, labels = load(sample)
    layers = sorted({x for x in labels if x is not None})
    nbr, _ = knn_graph(loc, 8)
    pairs = lr_pairs(genes, DB)
    res = dict(sample=sample, pairs=pairs, layers=layers, nbr=nbr, out={})
    for v, Rv in versions(raw).items():
        E = denoise(method, Rv, nbr)
        lib = Rv.sum(1)
        res['out'][v] = {method: summarise(E, genes, pairs, nbr, lib, labels, layers, True)}
        if method != 'smooth_r':
            res['out'][v][f'{method}_nr'] = summarise(E, genes, pairs, nbr, lib, labels, layers, False)
        print(sample, method, v, {k: round(o['rho_total'], 3) for k, o in res['out'][v].items()}, flush=True)
    os.makedirs(f'{OUT}_{method}', exist_ok=True)
    with open(os.path.join(f'{OUT}_{method}', f'{sample}.pkl'), 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', required=True)
    ap.add_argument('--method', required=True, choices=['smooth_r', 'scvi', 'magic', 'noCCC', 'attn', 'noGate', 'lr_noGate'])
    ap.add_argument('--save_full', action='store_true', help='only denoise the full counts and save them')
    a = ap.parse_args()
    for s in a.samples:
        (save_full if a.save_full else main)(s, a.method)
