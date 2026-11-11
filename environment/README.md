# Environments

The analyses used separate environments because some tools pin incompatible dependencies. Versions are those
of the runs reported in the manuscript; the slurm scripts in `experiments/slurm/` call each environment as
`$CONDA_ENVS/<name>/bin/python`.

| file | environment name | used for |
|---|---|---|
| `stccc.txt` | `spaVAE_env` (Python 3.9) | stCCC, scVI, interface scans, summaries, figures, `stccc_audit` |
| `magic_commot.txt` | `stccc_magic` (Python 3.10) | MAGIC, COMMOT (`denoise_baselines.py --method magic`, `commot_reliability.py`, `nbr_swap.py --variant magic_sp`) |
| `liana.txt` | `mamba` (Python 3.9) | LIANA+ (`liana_reliability.py`), `within_layer.py` |
| `paste.txt` | `paste_env` (Python 3.10) | HER2 preparation and label transfer (`prep_her2.py`) |
| `graphst.txt`, `stagate.txt`, `nichecompass.txt` | `graphst_env`, `stagate_env`, `nichecompass_env` (Python 3.9) | domain-recovery baselines (`run_baselines.py`) |
| R 4.5.2 with mclust 6.1.3 | `r_env` | mclust read-out (`mclust_readout.py`; set `STCCC_RSCRIPT` to its `Rscript`) |

Create one with, for example,
```
conda create -n spaVAE_env python=3.9 && conda activate spaVAE_env && pip install -r environment/stccc.txt
```
