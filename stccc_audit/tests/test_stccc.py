"""Check that the built-in stccc input reproduces the stored stCCC rate of the manuscript (DLPFC 151673, GPU).
    python -m stccc_audit.tests.test_stccc"""
import os
import numpy as np, scanpy as sc
from scipy.stats import spearmanr
from stccc_audit import Section, per10k
from stccc_audit.inputs import stccc
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(HERE, '..', '..', 'data')), 'dlpfc')
a = sc.read_h5ad(f'{DATA}/sample_151673_with_graph.h5ad')
sec = Section(counts=a.X, xy=a.obsm['spatial'], labels=np.load(f'{DATA}/sample_151673_labels.npy', allow_pickle=True), genes=list(a.var_names))
E = stccc(sec.counts, sec)
R = per10k(np.load(os.path.join(HERE, '..', '..', 'results', 'denoised', 'sample_151673.npy')).astype('float64'))
r = spearmanr(E.ravel()[::31], R.ravel()[::31])[0]; d = np.abs(E - R).max()
print('Spearman with stored rate %.6f, max |diff| %.2e' % (r, d)); assert r > 0.999; print('OK')
