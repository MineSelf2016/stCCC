import numpy as np
def align(ex, layers):
    """Re-index the layer columns of a merged result to the main result's layers (older pickles kept an 'NA' column)."""
    if list(ex['layers']) == list(layers):
        return ex['out']
    idx = [list(ex['layers']).index(l) if l in ex['layers'] else None for l in layers]
    out = {}
    for v, d in ex['out'].items():
        out[v] = {}
        for k, o in d.items():
            LM = np.full((o['LM'].shape[0], len(layers)), np.nan)
            for j, i in enumerate(idx):
                if i is not None:
                    LM[:, j] = o['LM'][:, i]
            out[v][k] = {**o, 'LM': LM}
    return out
