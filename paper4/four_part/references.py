"""Reference definitions, within-donor coverage comparisons and random-panel calibration."""

import collections
import math
import re

import numpy as np

from paper4.plaque_validation.panel_definition import ACCESSIONS, DEN, NUM, draw_panels
from paper4.four_part.inference import paired_stats

STRICT = "ec3_lysosomal_lumen"
PUMP = "v_atpase"
GO = "lysosomal_lumen_hydrolases"
NBOOT = 10000


def classes(go, ec):
    if go["release"] != ec["release"] or go["entry_fields"] != [
        "accession",
        "primary_gene",
    ]:
        raise ValueError("Incompatible annotation snapshots")
    lumen = dict(go["classes"][GO]["entries"])
    lysosome = dict(go["classes"]["lysosome_hydrolases"]["entries"])
    sets = {PUMP: {ACCESSIONS[g]: g for g in DEN}, GO: lumen}
    for label, entries in (
        (STRICT, lumen),
        ("ec3_lysosome", lysosome),
        ("ec34_lysosome", lysosome),
    ):
        sets[label] = {
            a: g
            for a, g in entries.items()
            if any(
                re.fullmatch(r"3\.\d+\.\d+\.\d+", code)
                and not code.startswith("3.6.")
                and (label != "ec34_lysosome" or code.startswith("3.4."))
                for code in ec["ec_by_accession"][a]
            )
        }
    return sets


def reference_tables(paired, sets, target, key):
    members, comparisons = [], []
    strict = {}
    for label, entries in sets.items():
        eligible = {}
        freq = collections.Counter(entries.values())
        for a, g in sorted(entries.items()):
            mapped = a in paired and paired[a][0] == g and freq[g] == 1
            v = paired[a][1] if mapped else np.full(len(target), np.nan)
            keep = g not in NUM and (label == PUMP or g not in DEN)
            n = int(np.isfinite(v).sum())
            members.append(
                dict(
                    **key,
                    reference=label,
                    gene=g,
                    accession=a,
                    target=g in NUM,
                    mapped=mapped,
                    n_pair=n,
                    n_eligible=len(target),
                    complete_control=keep and n == len(target),
                )
            )
            if keep and n == len(target):
                eligible[g] = v
            if label == STRICT and keep and n > 0:
                strict[g] = v
        if len(eligible) < 2:
            raise ValueError("Insufficient complete controls: " + label)
        score = target - np.mean(list(eligible.values()), axis=0)
        comparisons.append(
            dict(
                **key,
                reference=label,
                n_controls=len(eligible),
                genes="|".join(sorted(eligible)),
                **paired_stats(score),
            )
        )
    return members, comparisons, strict


def coverage(strict, target, key, rng):
    n = len(target)
    thresholds = (100, 80, 50)
    panels = {
        t: sorted(g for g, x in strict.items() if np.isfinite(x).sum() * 100 >= n * t)
        for t in thresholds
    }
    # Conditional on the membership observed in the original sample. Reuse resampled donors across cutoffs.
    weights = rng.multinomial(n, np.full(n, 1 / n), size=NBOOT)
    weights = np.vstack((np.ones(n), weights))
    baseline = target - np.mean([strict[g] for g in panels[100]], axis=0)
    base = weights @ baseline / n
    effects, matched = {}, {}
    genes = []
    for g in panels[50]:
        x = strict[g]
        ok = np.isfinite(x)
        donor_weights = weights[:, ok]
        counts = donor_weights.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            effects[g] = donor_weights @ (target[ok] - x[ok]) / counts
            matched[g] = donor_weights @ baseline[ok] / counts
        genes.append(
            dict(
                **key,
                gene=g,
                n_paired=int(ok.sum()),
                estimate=effects[g][0],
                baseline_same_donors=matched[g][0],
                composition=effects[g][0] - matched[g][0],
                subset=matched[g][0] - base[0],
            )
        )
    rows = []
    for t, panel in panels.items():
        eff = np.mean([effects[g] for g in panel], axis=0)
        old = np.mean([matched[g] for g in panel], axis=0)
        current = {
            "estimate": eff,
            "change_vs_100": eff - base,
            "composition": eff - old,
            "donor_subset": old - base,
        }
        if panel == panels[100]:
            current = {
                "estimate": base,
                **{
                    k: np.zeros_like(base)
                    for k in ("change_vs_100", "composition", "donor_subset")
                },
            }
        for metric, values in current.items():
            draws = values[1:]
            valid = draws[np.isfinite(draws)]
            if len(valid) < 0.99 * NBOOT:
                raise ValueError("Insufficient valid bootstrap draws")
            low, high = np.quantile(valid, [0.025, 0.975])
            rows.append(
                dict(
                    **key,
                    coverage_pct=t,
                    min_pairs=math.ceil(t * n / 100),
                    n_donors=n,
                    n_controls=len(panel),
                    genes="|".join(panel),
                    endpoint=metric,
                    estimate=values[0],
                    low=low,
                    high=high,
                    valid_draws=len(valid),
                )
            )
    return rows, genes


def calibration(single, key, rng, record_draws=None):
    slots = NUM + DEN
    for g in slots:
        if g not in single or single[g][0] != ACCESSIONS[g]:
            raise ValueError("Fixed panel identity mismatch: " + g)
    all_delta = np.array([single[g][1] for g in slots])
    ok = np.isfinite(all_delta).all(axis=0)
    if not ok.all():
        raise ValueError("Frozen cohort has incomplete fixed-panel target donors")
    delta = all_delta[:, ok]
    target_abundance = np.array([np.median(single[g][2][ok]) for g in slots])
    genes = sorted(
        g
        for g, (_, d, a) in single.items()
        if g not in slots and np.isfinite(d[ok]).all() and np.isfinite(a[ok]).all()
    )
    med = np.array([np.median(single[g][2][ok]) for g in genes])
    change = np.array([np.mean(single[g][1][ok]) for g in genes])
    numerator = delta[:2].mean()
    denominator = delta[2:].mean()
    observed = (delta[:2].mean(axis=0) - delta[2:].mean(axis=0)).mean()
    out = []
    for window in (0.5, 1.0):
        pools = [np.flatnonzero(np.abs(med - m) <= window) for m in target_abundance]
        for name, choice, count in (
            ("both_arms_matched", pools, 2),
            ("hydrolases_replaced", pools[:2], 2),
            ("v_atpase_replaced", pools[2:], 0),
        ):
            draws = draw_panels(choice, 10000, rng)
            x = change[draws]
            a = x[:, :count].mean(axis=1) if count else numerator
            b = x[:, count:].mean(axis=1) if count < len(choice) else denominator
            mock = a - b
            if record_draws is not None:
                record_draws(key, window, name, mock)
            hits = int(np.sum(mock >= observed - 1e-12))
            lo, median, hi = np.quantile(mock, [0.025, 0.5, 0.975])
            out.append(
                dict(
                    **key,
                    caliper_log2=window,
                    design=name,
                    n_background=len(genes),
                    min_candidates=min(map(len, choice)),
                    draws=len(mock),
                    hits=hits,
                    upper_tail=hits / len(mock),
                    observed=observed,
                    null_q025=lo,
                    null_median=median,
                    null_q975=hi,
                )
            )
    return out
