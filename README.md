# stCCC

Code for *Depth-confounded spatial communication scores and how to recover reproducible ligand–receptor signal with
stCCC* (Qi and Wei, 2026).

Spot-level spatial communication scores built from sparse ligand and receptor counts largely measure sequencing
depth, and regressing depth out removes most of their reproducibility. This repository contains

| folder | what it is |
|---|---|
| `stccc_audit/` | **the reliability audit** — a ground-truth-free protocol that checks any expression input for depth dependence and reproducibility (count splitting, downsampling, serial sections, LR-specific comparison against expression-matched random gene pairs) and recommends the input to use on your data |
| `model/` | **stCCC** — a spatially-gated variational autoencoder (Gaussian-process prior, spatial gating, LR-biased neighbour attention, communication consistency term) whose library-free decoder rate is the input of the depth-residualised communication score |
| `experiments/` | every script that produces the results, tables and figures of the manuscript |
| `environment/` | package versions of the environments used |
| `data/` | where the inputs go (see `data/README.md`) |

Data: processed inputs and all result tables are on figshare, https://doi.org/10.6084/m9.figshare.34988539 ;
public raw data are linked in `data/README.md`.

## Quick start: audit a communication input

```python
import anndata as ad
from stccc_audit import Section, run

secs = [Section.from_anndata(ad.read_h5ad(f), label_key='layer') for f in ['s1.h5ad', 's2.h5ad']]   # integer counts in X
rep = run(secs, inputs=['norm', 'smooth', 'magic', 'stccc'], serial=[(0, 1)])
print(rep.summary())                       # depth dependence and agreement measures, one row per input
print(rep.compare('stccc', 'magic'))  # paired comparison over serial pairs
print(rep.recommend()[1])                  # depth-independent input with the most reproducible LR-specific signal
```

Any function `f(counts, section) -> expression` (log1p per-10k scale) can be audited next to the built-in inputs; see
`stccc_audit/README.md`. From the command line:

```
python -m stccc_audit --h5ad s1.h5ad s2.h5ad --label-key layer --inputs norm smooth magic --serial 0,1 --out audit.csv
```

## Quick start: stCCC

```python
from stccc_audit import Section
from stccc_audit.inputs import stccc
sec = Section.from_anndata(adata, label_key='layer')
rate = stccc(sec.counts, sec)          # (spots x genes) denoised, library-free expression, log1p per 10k
```

The frozen configuration of the manuscript (λ = 10, γ = 3, KL target 0.025 per latent dimension, 8-nearest-neighbour
graph, at least 1,200 gradient steps) is used. A section of a few thousand spots trains in about a minute on one GPU.

## Install

```
conda create -n spaVAE_env python=3.9
conda activate spaVAE_env
pip install -r environment/stccc.txt
```

The baselines need their own environments (`environment/README.md`). Run everything from the repository root or
from `experiments/`; the slurm scripts in `experiments/slurm/` expect `STCCC_REPO` (repository root) and `CONDA_ENVS`
(folder of the conda environments) to be set.

## Reproducing the manuscript

Datasets are selected with `STCCC_DATASET=dlpfc` (default), `abca1` (MERFISH) or `her2`; results go to `results/`,
`results/abca1/` and `results/her2/`. Unzipping `stCCC_source_data.zip` (figshare, https://doi.org/10.6084/m9.figshare.34988539) in the repository root restores
every result table, so the summary, table and figure steps can be run without retraining.

| step | scripts (`experiments/`) | slurm (`experiments/slurm/`) | manuscript |
|---|---|---|---|
| data preparation | `prep_abca1_atlas.py`, `prep_abca1.py`, `prep_her2.py` | `abca1_prep` | Methods; Supp. Note 1 |
| hyper-parameter selection (development donor) and frozen test | `run_dlpfc.py` | `dev_sweep`, `sensitivity`, `test_frozen`, `repr` | Supp. Table S2 |
| reliability of stCCC and conventional scores | `ccc_reliability.py`, `reliability_summary.py`, `nar_summary.py` | `reliability`, `abca1`, `abca1_model`, `abca1_ext`, `her2` | Table 1, Figs 2–3 |
| LIANA+ and COMMOT | `liana_reliability.py`, `commot_reliability.py` | `commot` | Table 1 |
| denoise-then-score baselines (smoothing, MAGIC, scVI) | `denoise_baselines.py` | `denoise`, `denoise_cpu` | Table 1, Fig 6 |
| model variants (gating ablation) | `ccc_reliability.py --cond`, `run_dlpfc.py` | `reliability_ablation`, `gating`, `ablation` | Supp. Table S3 |
| laminar-interface scan and replication | `interface_scan.py`, `iface_cond.py`, `interface_summary.py` | `interface`, `iface_extra`, `iface_ablation` | Table 2, Figs 5–6, Supp. Table S7 |
| partner-swap null | `iface_pairspec.py`, `make_supp_pairspec.py` | `pairspec` | Supp. Table S12 |
| within-layer depth dependence | `within_layer.py`, `make_supp_within.py` | | Supp. Table S10 |
| neighbourhood exchange and gating weights | `nbr_swap.py`, `nbr_swap_summary.py`, `gate_map.py`, `make_fig_gate_swap.py` | `nbrswap`, `gate` | Supp. Note 12, Supp. Fig. S7 |
| semi-synthetic benchmark and KL pressure | `simulate.py`, `sim_summary.py`, `select_kl.py`, `select_kl_summary.py`, `probe_kl.py` | `simulate`, `sim_model`, `select_kl`, `probe_kl` | Supp. Note 8, Supp. Fig. S5, Supp. Table S9 |
| domain recovery and baselines | `run_dlpfc.py`, `run_baselines.py`, `domain_table.py`, `mclust_readout.py`, `make_supp_mclust.py` | `repr`, `baselines*` | Table 3, Supp. Tables S8, S11, Supp. Figs S3–S4 |
| tables | `supp_tables.py` (`mainrel`, `data`, `sensitivity`, `reliability`, `adjacent`, `interface`, `readout`, `kl`) | | Table 1, Supp. Tables |
| figures | `make_figures.py`, `make_figures_extra.py`, `make_figures_nar.py`, `make_fig_gate_swap.py`, `make_graphical_abstract.py` | | all figures |

Figures are written to `paper/figures/` and LaTeX tables to `paper/sections/`. Supplementary Table S3 was assembled from
the outputs of `reliability_summary.py` and `nar_summary.py` for the model variants.

## Checks

```
python -m stccc_audit.tests.test_reproduce     # audit reproduces the stored DLPFC values (needs data/dlpfc and results/)
python -m stccc_audit.tests.test_stccc    # stccc input reproduces the stCCC rate in results/denoised/ (GPU; written by interface_scan.py)
```

## Licence

MIT (code). The data are subject to the licences of their original sources.
