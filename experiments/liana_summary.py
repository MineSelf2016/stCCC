"""Per-spot score summary used by liana_reliability.py and commot_reliability.py."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def summarise(S, lib, labels, layers):
    ok = ~pd.isnull(labels); dep = np.log(lib + 1.0)
    LM = np.full((S.shape[1], len(layers)), np.nan)
    for q, lay in enumerate(layers):
        sel = ok & (labels == lay)
        LM[:, q] = S[sel].mean(0)
    rho = np.array([spearmanr(S[:, p], dep)[0] if S[:, p].std() > 0 else np.nan for p in range(S.shape[1])])
    tot = S.sum(1)
    return dict(LM=LM, total=tot, rho=rho, rho_total=float(spearmanr(tot, dep)[0]))
