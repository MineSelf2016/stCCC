"""Probe: does relaxing the KL pressure restore stCCC's sensitivity in the semi-synthetic benchmark?
Retrains stCCC with a different KL target / beta floor on one simulated dataset, scans it and reports
AUROC / TPR / FPR of the planted hypotheses. Writes results/simulation/probe_kl.jsonl (one line per run)."""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'model')); sys.path.insert(0, HERE)
from simulate import base, RESULTS
from ccc_reliability import prep, DB
from ccc_graph import build_scores
from stccc import stCCC
from rel_common import per10k
from interface_scan import scan
CONF = {'kl0.025': (0.025, 4), 'kl0.25': (0.25, 0.5), 'kl1': (1.0, 0.1)}


def main(task, conf):
    kl, minb = CONF[conf]
    raw, genes, lab, xy = base(); d = f'{RESULTS}/simulation/task_{task}'
    X = np.load(f'{d}/counts.npz')['X'].astype('float64'); T = json.load(open(f'{d}/truth.json'))
    loc = MinMaxScaler().fit_transform(xy) * 20.; dev = 'cuda'
    nbr, dist, S, _ = build_scores(X, genes, loc, 'lr', DB, k=8, seed=0)
    Xs, sf = prep(X); torch.manual_seed(0); np.random.seed(0)
    ip = np.mgrid[0:(1 + 1e-5):(1. / 8), 0:(1 + 1e-5):(1. / 8)].reshape(2, -1).T * 20.
    m = stCCC(input_dim=X.shape[1], GP_dim=2, Normal_dim=8, encoder_layers=[128, 64], decoder_layers=[128], inducing_points=ip,
                   kernel_scale=20., N_train=X.shape[0], nbr=nbr, nbr_dist=dist, score=S, sigma_sp=float(np.median(dist)),
                   use_attention=True, gamma=3.0, lam=10.0, KL_loss=kl, init_beta=max(minb, 1.0), min_beta=minb, device=dev)
    m.fit(loc, Xs, X, sf, maxiter=100)
    Tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    E = per10k(m.denoise(Tt(loc), Tt(Xs)))
    df, _ = scan(f'task_{task}', dict(model=E), X, genes, lab, xy)
    tr = {tuple(t) for t in T['truth']}; g = set(T.get('boosted') or [x for t in T['truth'] for x in t[:2]])
    df['planted'] = [(l, r, b) in tr for l, r, b in zip(df.ligand, df.receptor, df.boundary)]
    df = df[df.planted | ~(df.ligand.isin(g) | df.receptor.isin(g))]
    pos, neg = df[df.planted], df[~df.planted]
    out = dict(task=task, conf=conf, mode=T.get('mode'), f=T['f'], beta=float(m.beta),
               auroc=float(roc_auc_score(df.planted, df.z)) if T['f'] > 1 else None,
               tpr=float((pos.z > 1.645).mean()) if T['f'] > 1 else None, fpr=float((neg.z > 1.645).mean()))
    print(out, flush=True)
    with open(f'{RESULTS}/simulation/probe_kl.jsonl', 'a') as fh:
        fh.write(json.dumps(out) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int); ap.add_argument('--conf'); a = ap.parse_args()
    main(a.task, a.conf)
