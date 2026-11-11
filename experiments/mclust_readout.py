"""Alternative domain read-out (reviewer: baselines are usually clustered with mclust, not K-means).
mclust with the GraphST protocol: Gaussian mixture, model 'EEE', G = number of annotated layers, fitted on the
embedding (PCA to 20 components if it has more), via R (envs/r_env). Same embeddings as Table 1:
stCCC latent means (results/latent/<s>_lr_seed<k>.npz) and baseline embeddings (results/baseline_emb/).
Writes results/mclust_readout.csv (method, sample, seed, NMI, ARI) and prints means over sections.
Usage: python mclust_readout.py [--methods stccc graphst stagate nichecompass]"""
import os, sys, glob, argparse, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
RSCRIPT = os.environ.get('STCCC_RSCRIPT', 'Rscript')
SECTIONS = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
R_CODE = r'''
suppressMessages(library(mclust))
a <- commandArgs(TRUE); X <- as.matrix(read.csv(a[1], header = FALSE)); set.seed(2020)
fit <- Mclust(X, G = as.integer(a[2]), modelNames = "EEE", verbose = FALSE)
cl <- if (is.null(fit)) rep(NA, nrow(X)) else fit$classification
write.table(cl, a[3], row.names = FALSE, col.names = FALSE)
'''


def mclust(Z, K, tmp):
    Z = np.asarray(Z, 'float64')
    if Z.shape[1] > 20:
        Z = PCA(20, random_state=0).fit_transform(Z)
    fx, fo, fr = (os.path.join(tmp, n) for n in ('x.csv', 'cl.txt', 'm.R'))
    np.savetxt(fx, Z, delimiter=','); open(fr, 'w').write(R_CODE)
    subprocess.run([RSCRIPT, fr, fx, str(K), fo], check=True, capture_output=True)
    return pd.read_csv(fo, header=None, na_values='NA')[0].to_numpy('float64')


def embeddings(method, s):
    for sd in (0, 1, 2):
        if method == 'stccc':
            f = f'{RES}/latent/sample_{s}_lr_seed{sd}.npz'
            if os.path.exists(f):
                d = np.load(f); yield sd, d['z'], d['keep']
        else:
            f = f'{RES}/baseline_emb/{method}_sample_{s}_seed{sd}.npy'
            if os.path.exists(f):
                Z = np.load(f); yield sd, Z, np.arange(len(Z))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--methods', nargs='+', default=['stccc', 'graphst', 'stagate', 'nichecompass'])
    a = ap.parse_args()
    out = f'{RES}/mclust_readout.csv'
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=['method'])
    old = old[~old.method.isin(a.methods)]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for m in a.methods:
            for s in SECTIONS:
                y = np.load(f'{DATA}/sample_{s}_labels.npy', allow_pickle=True)
                for sd, Z, keep in embeddings(m, s):
                    yy = y[keep]; ok = ~pd.isnull(yy) & (yy.astype(str) != 'NA'); K = len(set(yy[ok]))
                    cl = mclust(Z, K, tmp)
                    good = ok & ~np.isnan(cl)
                    rows.append(dict(method=m, sample=s, seed=sd, NMI=NMI(yy[good], cl[good]) if good.any() else np.nan, ARI=ARI(yy[good], cl[good]) if good.any() else np.nan,
                                     failed=bool(np.isnan(cl).all())))
                    print(rows[-1], flush=True)
    D = pd.concat([old, pd.DataFrame(rows)], ignore_index=True); D.to_csv(out, index=False)
    print(D.groupby(['method', 'sample'])[['NMI', 'ARI']].mean().groupby('method').agg(['mean', 'count']).round(3))
