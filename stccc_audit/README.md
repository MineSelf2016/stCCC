# stccc_audit

A ground-truth-free reliability audit for spatial ligand–receptor (LR) communication scores. It implements the
protocol of the stCCC manuscript for any expression input, so that a user can check, on their own sections,
whether a communication score measures sequencing depth, whether it survives depth correction, and which input gives
the most reproducible LR-specific signal.

## What it reports

For every input and section (the input is refitted on every count version):

| metric | meaning |
|---|---|
| `depth_spot` | Spearman between the unit-level total score and log library size (0 = depth-independent) |
| `depth_pattern` | share of the centred regional pattern explained by the regions' mean log library size |
| `splithalf` | agreement of centred regional patterns between two binomial halves of the counts |
| `down25` | agreement of centred regional patterns between a 25% downsample and the full counts |
| `unit_splithalf` | agreement of edge-level totals between the two halves |

For every pair of adjacent sections:

| metric | meaning |
|---|---|
| `serial` | agreement of centred regional patterns over shared LR pairs and regions |
| `lrspec_both` | agreement of interface z-scores against a null that replaces both genes by expression-matched random genes |
| `lrspec_pair` | the same against a partner-swap null (one gene kept, its partner replaced): what the pairing adds |

Edge scores are `log1p(L_i R_j + L_j R_i)` on an 8-nearest-neighbour graph, depth-residualised by default on
`[1, log l_i + log l_j, |log l_i − log l_j|]`. The interface scan compares boundary with within-region edges for
every pair of regions joined by at least 30 edges.

## Use

```python
import anndata as ad
from stccc_audit import Section, run

secs = [Section.from_anndata(ad.read_h5ad(f), label_key='layer') for f in ['s1.h5ad', 's2.h5ad']]
rep = run(secs, inputs=['norm', 'smooth', 'magic', 'stccc'], serial=[(0, 1)])
print(rep.summary())                       # one row per input
print(rep.compare('stccc', 'magic'))  # paired comparison over serial pairs
print(rep.recommend()[1])                  # the input to prefer on these data

# any callable f(counts, section) -> expression on the log1p per-10k scale can be audited:
rep = run(secs, inputs={'mine': lambda C, sec: my_denoiser(C, sec.xy)}, serial=[(0, 1)])
```

Command line:

```
python -m stccc_audit --h5ad s1.h5ad s2.h5ad --label-key layer --inputs norm smooth magic --serial 0,1 --out audit.csv
```

`Section(scan_k=8)` for a square Spatial Transcriptomics array; `run(..., residualise='both')` also reports the
uncorrected scores (`<input>_nr`); `boundaries=[('L5', 'L6'), ...]` restricts the scan.

Built-in inputs: `norm`, `smooth`, `magic` (needs `magic-impute`), `scvi` (needs `scvi-tools`) and `stccc`
(needs `torch`; frozen configuration of the manuscript). LR pairs default to the LIANA consensus resource in `data/`.

## Checks

`python -m stccc_audit.tests.test_reproduce` reproduces the stored manuscript values for the normalised and smoothed
inputs on DLPFC sections 151673/151674 (maximum difference 7e-13); `python -m stccc_audit.tests.test_stccc`
checks the `stccc` input against the stored stCCC rate (GPU).
