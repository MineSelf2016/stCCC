"""Dataset switch for every reliability / interface script: STCCC_DATASET=dlpfc (default) or abca1.
  DATA     : folder with sample_<id>_with_graph.h5ad (integer counts) and sample_<id>_labels.npy
  RESULTS  : results root (DLPFC keeps the original results/ layout)
  SECTIONS : section ids;  ADJ : adjacent serial pairs;  GROUP : section -> donor (DLPFC) or AP position (ABCA-1)
  BOUNDS   : laminar boundaries scanned by interface_scan.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get('STCCC_DATASET', 'dlpfc')
if NAME == 'dlpfc':
    DATA = os.path.join(os.environ.get('STCCC_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'dlpfc')
    RESULTS = os.path.join(HERE, '..', 'results')
    SECTIONS = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
    ADJ = [(151507, 151508), (151509, 151510), (151669, 151670), (151671, 151672), (151673, 151674), (151675, 151676)]
    GROUP = {**{s: 'Br5292' for s in SECTIONS[:4]}, **{s: 'Br5595' for s in SECTIONS[4:8]}, **{s: 'Br8100' for s in SECTIONS[8:]}}
    BOUNDS = [('Layer1', 'Layer2'), ('Layer2', 'Layer3'), ('Layer3', 'Layer4'), ('Layer4', 'Layer5'), ('Layer5', 'Layer6'), ('Layer6', 'WM')]
elif NAME == 'abca1':
    DATA = os.path.join(HERE, '..', 'data', 'abca1')
    RESULTS = os.path.join(HERE, '..', 'results', 'abca1')
    # 12 pairs of consecutive sections (71-147 um apart) at 12 anterior-posterior positions; STCCC_ABCA1=orig gives the
    # four pairs of the first NAR submission (40/41, 52/53, 64/65, 76/77)
    ADJ = [(24, 25), (28, 29), (34, 35), (40, 41), (46, 47), (52, 53), (58, 59), (64, 65), (68, 69), (76, 77), (84, 85), (88, 89)]
    if os.environ.get('STCCC_ABCA1') == 'orig':
        ADJ = [(40, 41), (52, 53), (64, 65), (76, 77)]
    SECTIONS = [s for p in ADJ for s in p]
    GROUP = {s: f'AP{i + 1}' for i, p in enumerate(ADJ) for s in p}
    BOUNDS = [('L1', 'L2/3'), ('L2/3', 'L4'), ('L4', 'L5'), ('L5', 'L6a'), ('L6a', 'L6b')]
elif NAME == 'her2':
    # HER2+ breast cancer, Spatial Transcriptomics (Andersson et al. 2021); prep_her2.py. Patients E-H have three adjacent
    # sections; pairs of neighbouring sections serve as serial pairs (labels of the unannotated sections by PASTE transfer).
    DATA = os.path.join(HERE, '..', 'data', 'her2')
    RESULTS = os.path.join(HERE, '..', 'results', 'her2')
    SECTIONS = [11, 21, 31, 41, 51, 52, 53, 61, 62, 63, 71, 72, 73, 81, 82, 83]
    ADJ = [(51, 52), (52, 53), (61, 62), (62, 63), (71, 72), (72, 73), (81, 82), (82, 83)]
    GROUP = {s: 'ABCDEFGH'[s // 10 - 1] for s in SECTIONS}
    _R = ['adipose tissue', 'breast glands', 'cancer in situ', 'connective tissue', 'immune infiltrate', 'invasive cancer']
    BOUNDS = [(a, b) for i, a in enumerate(_R) for b in _R[i + 1:]]
    SCAN_K = 8                               # square ST array: the 8 surrounding spots
else:
    raise ValueError(NAME)
SCAN_K = globals().get('SCAN_K', 6)          # neighbours per spot in the interface-scan graph (6: Visium lattice, MERFISH)

# stCCC KL pressure, STCCC_KL="<KL target per latent dim>:<min beta>:<initial beta>"; default = original frozen configuration
_kl = [float(v) for v in os.environ.get('STCCC_KL', '0.025:4:10').split(':')]
KL = dict(KL_loss=_kl[0], min_beta=_kl[1], init_beta=_kl[2])
