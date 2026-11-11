# Data

Every script reads its inputs from this folder (or from `$STCCC_DATA` if set). Unzip the figshare archives
(https://doi.org/10.6084/m9.figshare.34988539) in the repository root and obtain the MERFISH data from the Allen Brain Cell Atlas as below.

```
data/
  dlpfc/   sample_<id>_with_graph.h5ad, sample_<id>_labels.npy      12 Visium sections   (figshare)
  her2/    sample_<id>_with_graph.h5ad, sample_<id>_labels.npy      16 ST sections       (figshare, or prep_her2.py)
  abca1/   sample_<id>_with_graph.h5ad, sample_<id>_labels.npy      24 MERFISH sections  (prep_abca1_atlas.py + prep_abca1.py)
  raw/     public raw files used by the preparation scripts
```

## Human DLPFC (Visium)

Processed sections are on figshare (`stCCC_DLPFC_processed.zip`). They are the 12 sections of Maynard et al.
(2021), filtered per section to 1,011–3,000 genes, with the expert layer annotation. Raw data: spatialLIBD,
http://spatial.libd.org/spatialLIBD/ and https://github.com/LieberInstitute/HumanPilot.

## HER2-positive breast cancer (Spatial Transcriptomics)

Processed sections are on figshare (`stCCC_HER2_processed.zip`). To rebuild them, download the data of
Andersson et al. (2021) from https://github.com/almaan/her2st (Zenodo record 3957257; the 7z passwords are given in
that README) and place

```
data/raw/her2st/ST-cnts/<section>.tsv.gz
data/raw/her2st/ST-spotfiles/<section>_selection.tsv
data/raw/her2st/ST-pat/lbl/<section>_labeled_coordinates.tsv
```

then run `python experiments/prep_her2.py` (environment `paste_env`).

## Mouse isocortex (MERFISH, Zhuang-ABCA-1)

Download from the Allen Brain Cell Atlas (https://alleninstitute.github.io/abc_atlas_access/, public S3 bucket
`allen-brain-cell-atlas`) into `data/raw/abca1/`:

| file | ABC atlas directory |
|---|---|
| `Zhuang-ABCA-1-raw.h5ad` | `expression_matrices/Zhuang-ABCA-1/` |
| `cell_metadata_with_cluster_annotation.csv` | `metadata/Zhuang-ABCA-1/…/views/` |
| `ccf_coordinates.csv` | `metadata/Zhuang-ABCA-1-CCF/` |
| `parcellation_to_parcellation_term_membership_acronym.csv` | `metadata/Allen-CCF-2020/` |

then run

```
python experiments/prep_abca1_atlas.py    # registered whole-brain stack -> data/raw/abca1/prep/
python experiments/prep_abca1.py          # 24 isocortex sections with CCF layers -> data/abca1/
```

## Ligand–receptor pairs

The LIANA consensus resource is in `stccc_audit/data/consensus.csv`.
