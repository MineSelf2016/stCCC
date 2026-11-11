"""Domain-recovery baselines on the 12 DLPFC sections under the stCCC protocol.
Every method gets the same input as stCCC (the processed count matrix and spot coordinates of each section;
all of its genes are used as the feature set) and the same read-out: K-means on the method's embedding,
K = number of annotated layers ('NA' spots excluded), NMI/ARI averaged over five clustering seeds.
The read-out is computed on the raw and on the standardised embedding; the paper reports the better of the two
for each baseline. Runs in the method's own conda env:
  graphst      -> graphst_env       (GraphST, default 600 epochs; 'emb' is reconstructed expression, so the
                                     20-component PCA of GraphST's own clustering step is the embedding)
  stagate      -> stagate_env       (STAGATE_pyG, 6-NN spatial net, default 1000 epochs)
  nichecompass -> nichecompass_env  (same consensus LR resource as stCCC, 100 epochs)
Writes one JSON line per (method, section, training seed) to results/baselines.jsonl."""
import os, sys, json, time, argparse, random, warnings
import numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stccc_audit', 'data', 'consensus.csv')


def load(sample):
    ad = sc.read_h5ad(f"{DATA}/{sample}_with_graph.h5ad")
    X = ad.X.toarray() if hasattr(ad.X, 'toarray') else np.asarray(ad.X)
    a = sc.AnnData(np.rint(X).astype('float32')); a.var_names = list(ad.var_names); a.var_names_make_unique()
    a.obsm['spatial'] = np.asarray(ad.obsm['spatial'], 'float64')
    y = np.load(f"{DATA}/{sample}_labels.npy", allow_pickle=True)
    ok = ~pd.isnull(y) & (y.astype(str) != 'NA')
    return a, y, ok


def evaluate(Z, y, ok):
    K = len(set(y[ok])); out = {}
    for tag, E in [('raw', np.asarray(Z, 'float64')), ('std', StandardScaler().fit_transform(np.asarray(Z, 'float64')))]:
        n, r = [], []
        for cs in range(5):
            p = KMeans(K, n_init=10, random_state=cs).fit_predict(E)
            n.append(NMI(y[ok], p[ok])); r.append(ARI(y[ok], p[ok]))
        out[f'NMI_{tag}'] = float(np.mean(n)); out[f'ARI_{tag}'] = float(np.mean(r))
    return out


def seed_all(seed):
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graphst(a, seed, device):
    from GraphST import GraphST
    a.var['highly_variable'] = True                 # all genes of the processed panel; GraphST's own preprocessing follows
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); sc.pp.scale(a, zero_center=False, max_value=10)
    m = GraphST.GraphST(a, device=device, random_seed=seed)
    from sklearn.decomposition import PCA
    return PCA(n_components=20, random_state=42).fit_transform(m.train().obsm['emb'])   # as GraphST.utils.clustering


def stagate(a, seed, device):
    import STAGATE_pyG as ST
    a.var['highly_variable'] = True
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    ST.Cal_Spatial_Net(a, k_cutoff=6, model='KNN', verbose=False)
    a = ST.train_STAGATE(a, device=device, random_seed=seed, verbose=False)
    return a.obsm['STAGATE']


def nichecompass(a, seed, device):
    import scipy.sparse as sp, squidpy as sq
    from nichecompass.models import NicheCompass
    from nichecompass.utils import add_gps_from_gp_dict_to_adata
    a.X = sp.csr_matrix(a.X); a.var_names = [g.upper() for g in a.var_names]; a.layers['counts'] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sq.gr.spatial_neighbors(a, coord_type='generic', n_neighs=6)
    db = pd.read_csv(DB); g = set(a.var_names); gp = {}
    for s, t in zip(db['source'], db['target']):
        if isinstance(s, str) and isinstance(t, str) and s in g and t in g and s != t:
            d = gp.setdefault(f'{s}_ligand_receptor_GP', {'sources': [s], 'targets': [], 'sources_categories': ['ligand'], 'targets_categories': []})
            if t not in d['targets']:
                d['targets'].append(t); d['targets_categories'].append('receptor')
    add_gps_from_gp_dict_to_adata(gp_dict=gp, adata=a, genes_uppercase=True, gp_targets_mask_key='nichecompass_gp_targets',
                                  gp_sources_mask_key='nichecompass_gp_sources', gp_names_key='nichecompass_gp_names', min_genes_per_gp=1)
    m = NicheCompass(a, counts_key='counts', adj_key='spatial_connectivities', cond_embed_injection=[], n_fc_layers_encoder=1,
                     conv_layer_encoder='gcnconv', active_gp_thresh_ratio=0.03, seed=seed)
    m.train(n_epochs=100, n_epochs_all_gps=25, lr=0.001, verbose=False)
    return m.get_latent_representation()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['graphst', 'stagate', 'nichecompass'])
    ap.add_argument('--samples', nargs='+', required=True); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--out', default='baselines.jsonl')
    args = ap.parse_args()
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = open(os.path.join(HERE, '..', 'results', args.out), 'a')
    for s in args.samples:
        for sd in args.seeds:
            a, y, ok = load(s); seed_all(sd); t0 = time.time()
            Z = globals()[args.method](a, sd, dev)
            ed = os.path.join(HERE, '..', 'results', 'baseline_emb'); os.makedirs(ed, exist_ok=True)
            np.save(f'{ed}/{args.method}_{s}_seed{sd}.npy', np.asarray(Z, 'float32'))
            r = dict(method=args.method, sample=s, seed=sd, dim=int(np.asarray(Z).shape[1]), secs=round(time.time() - t0, 1), **evaluate(Z, y, ok))
            print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
