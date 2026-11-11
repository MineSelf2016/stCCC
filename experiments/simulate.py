"""Semi-synthetic benchmark with known communication and a known depth confound.

Base: DLPFC section 151673 (real spot layout, real expression, real layer labels).
Planted communication: K = 12 (LR pair, laminar boundary) hypotheses with disjoint genes. For a hypothesis
(l, r, A/B) the ligand is raised in the A-side boundary spots (spots of layer A with a 6-NN neighbour in B) and
the receptor in the B-side boundary spots, by adding Poisson((f - 1) * rate) counts, where rate is the gene's
layer-mean expression per unit library times the spot's library. f = 1 plants nothing (pure null).
Depth confound: 'none' keeps the observed depth; 'gradient' binomially thins every spot with a probability that
falls four-fold from layer 1 to white matter, so depth changes at every boundary but expression rates do not.
Grid: f in {1, 1.5, 2, 3} x depth in {none, gradient} x seed in {0, 1, 2} = 24 tasks per planting mode:
  gene    (tasks 0-23)  : only the ligand and the receptor are raised (a change no other gene shares)
  program (tasks 24-47) : the ligand and the receptor are raised together with their co-expression programs, the
                          19 genes most correlated with each across spots of the base section, as when a cell
                          population that expresses the ligand or receptor is enriched at the boundary.

Spatial scale (tasks 48-71, program mode, f in {2, 3}): the planted band is widened from the spots touching the
boundary (width 1) to all spots of the layer within 2 or 3 steps of the other layer on the 6-NN graph.

Stages (separate environments): --stage counts_model (spaVAE_env: simulate, stCCC, scVI, smoothing),
--stage magic (stccc_magic), --stage scan (spaVAE_env: interface scan of every input). Writes results/simulation/task_<i>/."""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import DATA, RESULTS, BOUNDS
from rel_common import DB, per10k, knn_graph
SEC, K, FRAC = 'sample_' + os.environ.get('STCCC_SIMSEC', '151673'), 12, 0.02
TASKS = [(f, d, s, mode, 1) for mode in ('gene', 'program') for f in (1.0, 1.5, 2.0, 3.0) for d in ('none', 'gradient') for s in (0, 1, 2)]
TASKS += [(f, d, s, 'program', w) for w in (2, 3) for f in (2.0, 3.0) for d in ('none', 'gradient') for s in (0, 1, 2)]   # 48-71: wider bands
NPROG = 20
ORDER = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']


def base():
    import scanpy as sc
    ad = sc.read_h5ad(f"{DATA}/{SEC}_with_graph.h5ad")
    raw = np.rint(ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)).astype(np.int64)
    lab = np.array([str(v) for v in np.load(f"{DATA}/{SEC}_labels.npy", allow_pickle=True)])
    return raw, list(ad.var_names), lab, np.asarray(ad.obsm['spatial'], 'float64')


def programs(raw):
    """Pearson correlation of log-normalised expression between all genes of the base section."""
    Z = per10k(raw.astype('float64')); Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    return (Z.T @ Z) / len(Z)


def band(lab, nb, side, other, width):
    """Spots of layer `side` within `width` graph steps of a spot of layer `other` (6-NN graph)."""
    near = lab == other; reach = near.copy()
    for _ in range(width):
        reach = reach | reach[nb].any(1)
    return np.flatnonzero((lab == side) & reach)


def simulate(raw, genes, lab, xy, f, depth, seed, mode='gene', width=1):
    rng = np.random.default_rng(seed)
    gi = {g: i for i, g in enumerate(genes)}
    frac = (raw > 0).mean(0); db = pd.read_csv(DB)
    P = sorted({(s, t) for s, t in zip(db['source'], db['target']) if isinstance(s, str) and isinstance(t, str)
                and s in gi and t in gi and s != t and frac[gi[s]] >= FRAC and frac[gi[t]] >= FRAC})
    _, idx = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy); nb = idx[:, 1:]
    truth, used = [], set()
    for p in rng.permutation(len(P)):
        l, r = P[p]
        if l in used or r in used:
            continue
        A, B = BOUNDS[rng.integers(len(BOUNDS))]
        truth.append((l, r, f'{A}/{B}')); used |= {l, r}
        if len(truth) == K:
            break
    X = raw.copy(); lib = raw.sum(1).astype('float64')
    Cg = programs(raw) if mode == 'program' else None
    boosted = set()
    for l, r, b in truth:
        A, B = b.split('/')
        for g, side, other in ((l, A, B), (r, B, A)):
            prog = [g] if mode == 'gene' else [genes[j] for j in np.argsort(-Cg[gi[g]])[:NPROG]]
            if gi[g] not in [gi[x] for x in prog]:
                prog = [g] + prog[:-1]
            boosted |= set(prog)
            if f == 1:
                continue
            spots = band(lab, nb, side, other, width)
            inl = lab == side
            for h in prog:
                rate = raw[inl, gi[h]].sum() / lib[inl].sum() * lib[spots]
                X[spots, gi[h]] += rng.poisson((f - 1) * rate)
    if depth == 'gradient':
        g = np.array([ORDER.index(v) / (len(ORDER) - 1) if v in ORDER else 0.5 for v in lab])
        X = rng.binomial(X, (0.25 ** g)[:, None])          # layer 1 keeps all counts, white matter a quarter
    return X.astype('float64'), truth, sorted(boosted)


def main(task, stage):
    f, depth, seed, mode, width = TASKS[task]
    out = os.path.join(RESULTS, 'simulation', f'task_{task}'); os.makedirs(out, exist_ok=True)
    raw, genes, lab, xy = base()
    if stage == 'counts_model':
        import torch
        from ccc_reliability import train_denoise
        from denoise_baselines import denoise
        X, truth, boosted = simulate(raw, genes, lab, xy, f, depth, seed, mode, width)   # deterministic given the task
        np.savez_compressed(f'{out}/counts.npz', X=X.astype(np.int32))
        json.dump(dict(f=f, depth=depth, seed=seed, mode=mode, width=width, truth=truth, boosted=boosted), open(f'{out}/truth.json', 'w'))
        loc = MinMaxScaler().fit_transform(xy) * 20.
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(f'{out}/den_model.npy'):
            np.save(f'{out}/den_model.npy', per10k(train_denoise(X, genes, loc, dev)).astype('float32'))
        torch.set_default_dtype(torch.float32)                            # stCCC switches it to float64
        nbr, _ = knn_graph(loc, 8)
        for m in ('scvi', 'smooth_r'):
            np.save(f'{out}/den_{m}.npy', denoise(m, X, nbr).astype('float32'))
    elif stage == 'model':                                                 # retrain stCCC only (current STCCC_KL)
        import torch
        from ccc_reliability import train_denoise
        X = np.load(f'{out}/counts.npz')['X'].astype('float64')
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        np.save(f'{out}/den_model.npy', per10k(train_denoise(X, genes, MinMaxScaler().fit_transform(xy) * 20., dev)).astype('float32'))
    elif stage == 'magic':
        from denoise_baselines import denoise
        X = np.load(f'{out}/counts.npz')['X'].astype('float64')
        np.save(f'{out}/den_magic.npy', denoise('magic', X, None).astype('float32'))
    elif stage == 'scan':
        from interface_scan import scan
        X = np.load(f'{out}/counts.npz')['X'].astype('float64')
        inputs = dict(norm_nr=per10k(X), norm=per10k(X),
                      **{m: np.load(f'{out}/den_{m}.npy').astype('float64') for m in ('smooth_r', 'scvi', 'magic', 'model')})
        df, _ = scan(f'task_{task}', inputs, X, genes, lab, xy, no_resid=('norm_nr',))
        T = {tuple(t) for t in json.load(open(f'{out}/truth.json'))['truth']}
        df['planted'] = [(l, r, b) in T for l, r, b in zip(df.ligand, df.receptor, df.boundary)]
        df.to_csv(f'{out}/scan.csv', index=False)
        print(task, TASKS[task], df.groupby('input').z.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--stage', required=True, choices=['counts_model', 'model', 'magic', 'scan'])
    a = ap.parse_args(); main(a.task, a.stage)
