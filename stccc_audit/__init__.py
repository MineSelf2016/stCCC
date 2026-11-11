"""stccc_audit: a ground-truth-free reliability audit for spatial ligand-receptor communication scores.

    from stccc_audit import Section, run
    secs = [Section.from_anndata(a, label_key='layer') for a in adatas]
    rep = run(secs, inputs=['norm', 'magic', 'stccc'], serial=[(0, 1)])
    print(rep.summary()); print(rep.recommend())

Any callable f(counts, section) -> expression (log1p per-10k scale) can be audited next to the built-in inputs."""
from .core import Section, interface_scan, count_versions, lr_pairs, per10k
from .audit import audit_section, audit_serial, run, Report, SectionAudit
from .inputs import BUILTIN, DEFAULT_DB

__all__ = ['Section', 'run', 'audit_section', 'audit_serial', 'Report', 'SectionAudit', 'interface_scan',
           'count_versions', 'lr_pairs', 'per10k', 'BUILTIN', 'DEFAULT_DB']
