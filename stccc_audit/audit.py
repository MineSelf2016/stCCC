"""Run the audit on sections and serial pairs, collect a report and choose an input."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .core import (count_versions, summarise, centred, corr, depth_pattern_share, interface_scan, lr_pairs)
from .inputs import BUILTIN, DEFAULT_DB


@dataclass
class SectionAudit:
    name: str
    pairs: list
    layers: list
    results: dict          # input -> dict(metrics, LM, iface)

    def metrics(self):
        return pd.DataFrame({k: v['metrics'] for k, v in self.results.items()}).T.assign(section=self.name)


def _resolve(inputs):
    if isinstance(inputs, (list, tuple)):
        return {k: BUILTIN[k] for k in inputs}
    return {k: (BUILTIN[v] if isinstance(v, str) else v) for k, v in inputs.items()}


def audit_section(sec, inputs=('norm', 'smooth'), db=DEFAULT_DB, residualise=True, interface=True, boundaries=None,
                  n_null=100, seed=0, verbose=True):
    """Audit one section. inputs: names of built-in inputs or {name: callable(counts, section)}.
    residualise: depth-residualise edge scores (True), not (False) or report both ('both': '<name>_nr' added).
    Metrics per input:
      depth_spot     Spearman between the unit-level total score and log library size (0 = depth-independent)
      depth_pattern  share of the centred regional pattern explained by the regions' mean log library size
      splithalf      agreement of centred regional patterns between two binomial halves of the counts
      down25         agreement of centred regional patterns between a 25% downsample and the full counts
      unit_splithalf agreement of edge-level totals between the two halves
    The input is refitted on every count version. With interface=True the full-count input is also scanned."""
    inputs = _resolve(inputs)
    pairs = lr_pairs(sec.genes, db)
    vers = count_versions(sec.counts, seed)
    modes = [True, False] if residualise == 'both' else [bool(residualise)]
    res = {}
    for name, fn in inputs.items():
        S = {r: {} for r in modes}; iface = None
        for v, C in vers.items():
            E = fn(C, sec)
            for r in modes:
                S[r][v] = summarise(E, sec, pairs, C.sum(1), residualise=r)
            if v == 'full' and interface:
                iface = interface_scan(E, sec, db, boundaries=boundaries, n_null=n_null, seed=seed)
        lib = vers['full'].sum(1)
        for r in modes:
            o = S[r]; key = name if (len(modes) == 1 or r) else f'{name}_nr'
            res[key] = dict(metrics=dict(
                depth_spot=corr(o['full']['total'].mean(1), np.log(lib + 1.0)),
                depth_pattern=depth_pattern_share(o['full']['LM'], sec, lib),
                splithalf=corr(centred(o['A']['LM']), centred(o['B']['LM'])),
                down25=corr(centred(o['d25']['LM']), centred(o['full']['LM'])),
                unit_splithalf=corr(o['A']['total'], o['B']['total'])),
                LM=o['full']['LM'], iface=iface if (len(modes) == 1 or r) else None, residualised=r)
        if verbose:
            print(sec.name, name, {k: round(v, 3) for k, v in res[name]['metrics'].items()}, flush=True)
    return SectionAudit(sec.name, pairs, sec.layers, res)


def audit_serial(a, b):
    """Agreement between two adjacent sections, per input: centred regional patterns over shared LR pairs and regions
    (serial), and interface z-scores over shared hypotheses under both nulls (lrspec_both, lrspec_pair)."""
    rows = []
    for name in sorted(set(a.results) & set(b.results)):
        P = sorted(set(a.pairs) & set(b.pairs)); L = sorted(set(a.layers) & set(b.layers))
        fa = pd.DataFrame(centred(a.results[name]['LM']), index=a.pairs, columns=a.layers).loc[P, L]
        fb = pd.DataFrame(centred(b.results[name]['LM']), index=b.pairs, columns=b.layers).loc[P, L]
        row = dict(pair=f'{a.name}-{b.name}', input=name, serial=corr(fa.values, fb.values))
        ia, ib = a.results[name]['iface'], b.results[name]['iface']
        if ia is not None and ib is not None and len(ia) and len(ib):
            m = ia.merge(ib, on=['boundary', 'ligand', 'receptor'])
            row.update(n_hypotheses=len(m), lrspec_both=corr(m.z_both_x, m.z_both_y), lrspec_pair=corr(m.z_pair_x, m.z_pair_y))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Report:
    per_section: pd.DataFrame
    per_pair: pd.DataFrame

    def summary(self):
        S = self.per_section.groupby(level=0).mean(numeric_only=True)
        if len(self.per_pair):
            S = S.join(self.per_pair.groupby('input').mean(numeric_only=True))
        return S.round(3)

    def compare(self, a, b, metric='lrspec_pair'):
        """Paired comparison of inputs a and b over serial pairs (or sections for section-level metrics)."""
        T = self.per_pair if metric in self.per_pair.columns else self.per_section.reset_index().rename(columns={'index': 'input', 'section': 'pair'})
        W = T.pivot_table(index='pair', columns='input', values=metric)
        d = (W[a] - W[b]).dropna()
        p = wilcoxon(d, alternative='greater').pvalue if len(d) > 1 and (d != 0).any() else np.nan
        return dict(metric=metric, n=len(d), higher=int((d > 0).sum()), mean_diff=float(d.mean()), p_one_sided=float(p))

    def recommend(self, max_depth=0.1):
        """Among depth-independent inputs (|depth_spot| <= max_depth), the one with the most reproducible LR-specific
        signal between serial sections (partner-swap null); without serial pairs, the highest unit-level split-half."""
        S = self.summary()
        ok = S[S.depth_spot.abs() <= max_depth]
        if not len(ok):
            return None, 'no input is depth-independent; residualise the edge scores'
        col = 'lrspec_pair' if 'lrspec_pair' in ok.columns and ok.lrspec_pair.notna().any() else 'unit_splithalf'
        best = ok[col].idxmax()
        return best, f'{best}: highest {col} ({ok.loc[best, col]:.3f}) among depth-independent inputs {list(ok.index)}'


def run(sections, inputs=('norm', 'smooth'), serial=(), **kw):
    """Audit a list of Section objects with the given inputs; serial: list of (i, j) index pairs of adjacent sections."""
    A = [audit_section(s, inputs, **kw) for s in sections]
    per_section = pd.concat([a.metrics() for a in A])
    per_pair = pd.concat([audit_serial(A[i], A[j]) for i, j in serial]) if serial else pd.DataFrame()
    return Report(per_section, per_pair)
