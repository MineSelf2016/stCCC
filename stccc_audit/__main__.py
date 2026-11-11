"""Command line: python -m stccc_audit --h5ad s1.h5ad s2.h5ad --label-key layer --inputs norm smooth magic \
                    --serial 0,1 --out report.csv"""
import argparse
import anndata as ad
from . import Section, run


def main():
    ap = argparse.ArgumentParser(description='Reliability audit of spatial LR communication inputs')
    ap.add_argument('--h5ad', nargs='+', required=True, help='one AnnData per section (integer counts in X or --layer)')
    ap.add_argument('--label-key', required=True, help='obs column with regions or layers')
    ap.add_argument('--spatial-key', default='spatial'); ap.add_argument('--layer', default=None)
    ap.add_argument('--inputs', nargs='+', default=['norm', 'smooth'])
    ap.add_argument('--serial', nargs='*', default=[], help='adjacent sections as i,j (0-based, order of --h5ad)')
    ap.add_argument('--scan-k', type=int, default=6, help='neighbours of the interface-scan graph (8 for a square ST array)')
    ap.add_argument('--no-interface', action='store_true'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    secs = [Section.from_anndata(ad.read_h5ad(f), a.label_key, a.spatial_key, a.layer, name=f.rsplit('/', 1)[-1].rsplit('.', 1)[0],
                                 scan_k=a.scan_k) for f in a.h5ad]
    rep = run(secs, a.inputs, serial=[tuple(int(x) for x in p.split(',')) for p in a.serial], interface=not a.no_interface)
    S = rep.summary(); print(S.to_string()); print(rep.recommend()[1])
    if a.out:
        S.to_csv(a.out); rep.per_section.to_csv(a.out.replace('.csv', '_sections.csv'))
        if len(rep.per_pair):
            rep.per_pair.to_csv(a.out.replace('.csv', '_pairs.csv'))


if __name__ == '__main__':
    main()
