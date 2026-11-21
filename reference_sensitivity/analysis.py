"""Donor-paired contrasts, fixed-size panels and proteome-wide direction summaries."""

import hashlib
import itertools
import math

import numpy as np
from scipy.stats import t

from paper4.plaque_validation.panel_definition import DEN
from paper4.four_part.references import PUMP, classes


EPS = 1e-12


def random_generator(seed, *labels):
    text = repr((int(seed), *labels)).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(text).digest()[:8], "big"))


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, ranked)
    return adjusted


def score_intervals(values, draws, seed, labels, family_size=None):
    """Columns share donors and bootstrap weights. Bonferroni uses t assumptions."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n, comparisons = values.shape
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("At least two complete donors are required for intervals")
    family_size = family_size or comparisons
    if family_size < comparisons:
        raise ValueError("The comparison family cannot be smaller than the reported tests")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / math.sqrt(n)
    nominal = t.ppf(0.975, n - 1) * se
    simultaneous = t.ppf(1 - 0.05 / (2 * family_size), n - 1) * se
    pvalues = np.ones(comparisons)
    nonconstant = se > 0
    pvalues[nonconstant] = 2 * t.sf(np.abs(mean[nonconstant] / se[nonconstant]), n - 1)
    pvalues[(~nonconstant) & (np.abs(mean) > EPS)] = 0
    adjusted = holm_adjust(np.concatenate([pvalues, np.ones(family_size - comparisons)]))[:comparisons]
    rng = random_generator(seed, *labels)
    boot = np.empty((draws, comparisons))
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count)
        boot[start:start + count] = weights @ values / n
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return [dict(n=n, estimate=float(mean[i]), se=float(se[i]),
                 t_low=float(mean[i] - nominal[i]), t_high=float(mean[i] + nominal[i]),
                 bootstrap_low=float(low[i]), bootstrap_high=float(high[i]),
                 bonferroni_low=float(mean[i] - simultaneous[i]),
                 bonferroni_high=float(mean[i] + simultaneous[i]),
                 p_two_sided=float(pvalues[i]), p_holm_within_family=float(adjusted[i]),
                 family_size=family_size, bootstrap_draws=draws,
                 precision_note="small_sample" if n < 10 else "") for i in range(comparisons)]


def coverage_rows(cohort, targets, references):
    rows = []
    for role, groups in (("target", {"targets": targets}), ("reference", references)):
        for name, genes in groups.items():
            for gene in genes:
                record = cohort.proteins.get(gene)
                count = int(np.isfinite(record[1]).sum()) if record else 0
                rows.append(dict(**cohort.key, role=role, reference=name, gene=gene,
                                 accession=record[0] if record else "", n_donors=len(cohort.donors),
                                 n_paired=count, paired_fraction=count / len(cohort.donors),
                                 mapping_status="unique" if record else cohort.missing_reason(gene)))
    return rows


def common_mask(cohort, genes):
    missing = sorted(set(genes) - cohort.proteins.keys())
    if missing:
        return np.zeros(len(cohort.donors), dtype=bool), "unmapped=" + "|".join(missing)
    mask = np.isfinite(cohort.values(list(dict.fromkeys(genes)))).all(axis=1)
    return mask, "" if mask.sum() >= 2 else "fewer_than_two_complete_donors"


def panel_combinations(cohort, config, draws, seed):
    targets, pool = config["primary_targets"], config["combination_pool"]
    panels = list(itertools.combinations(pool, config["combination_size"]))
    mask, reason = common_mask(cohort, targets + pool)
    status = dict(**cohort.key, n_total=len(cohort.donors), n_common=int(mask.sum()),
                  n_panels=len(panels), status="unavailable" if reason else "estimated", reason=reason)
    if reason:
        return [], [], status, []
    target = cohort.values(targets)[mask].mean(axis=1)
    values = np.column_stack([target - cohort.values(list(panel))[mask].mean(axis=1) for panel in panels])
    estimates = score_intervals(values, draws, seed, ("combinations", cohort.dataset, cohort.group))
    full = (target - cohort.values(pool)[mask].mean(axis=1)).mean()
    np.testing.assert_allclose(values.mean(), full, atol=1e-12, rtol=0)
    rows, donors = [], []
    names = [donor for donor, keep in zip(cohort.donors, mask) if keep]
    for i, (panel, stats) in enumerate(zip(panels, estimates), 1):
        label = f"P{i:02d}"
        rows.append(dict(**cohort.key, panel=label, targets="|".join(targets),
                         reference_genes="|".join(panel), **stats))
        donors.extend(dict(**cohort.key, donor=donor, panel=label, score=float(score))
                      for donor, score in zip(names, values[:, i - 1]))
    status.update(point_min=float(values.mean(axis=0).min()), point_max=float(values.mean(axis=0).max()),
                  n_positive_point=int((values.mean(axis=0) > EPS).sum()), full_panel_estimate=float(full),
                  n_positive_simultaneous=sum(r["bonferroni_low"] > 0 for r in rows),
                  n_negative_simultaneous=sum(r["bonferroni_high"] < 0 for r in rows))
    return rows, donors, status, names


def resolve_references(cohort, config, go, ec):
    if config["reference_mode"] == "fixed_lists":
        return config["references"], []
    definitions = classes(go, ec)
    lists, members = {}, []
    for name, entries in definitions.items():
        controls = []
        for accession, gene in sorted(entries.items()):
            record = cohort.proteins.get(gene)
            mapped = record is not None and record[0] == accession
            count = int(np.isfinite(record[1]).sum()) if mapped else 0
            eligible = gene not in config["primary_targets"] and (name == PUMP or gene not in DEN)
            retained = eligible and mapped and count == len(cohort.donors)
            members.append(dict(**cohort.key, reference=name, accession=accession, gene=gene,
                                mapped=bool(mapped), eligible_annotation=eligible,
                                n_paired=count, n_total=len(cohort.donors), retained=bool(retained)))
            if retained:
                controls.append(gene)
        lists[name] = sorted(controls)
    return lists, members


def compare_references(cohort, targets, references, draws, seed, application):
    """Each reference has an exact list. Cross-reference contrasts use common donors."""
    estimates, donor_rows, statuses, contrasts = [], [], [], []
    measurable = {}
    for name, genes in references.items():
        mask, reason = common_mask(cohort, targets + genes)
        if not genes:
            reason = "empty_measured_reference"
        statuses.append(dict(**cohort.key, application=application, reference=name,
                             n_total=len(mask), n_common=int(mask.sum()),
                             status="unavailable" if reason else "estimated", reason=reason))
        if not reason:
            measurable[name] = genes
    # Per-reference results preserve usable comparisons when another reference is absent.
    scopes = [("per_reference", {name: genes}) for name, genes in measurable.items()]
    if len(measurable) >= 2:
        scopes.append(("common_donors", measurable))
    for scope, panels in scopes:
        needed = targets + list(dict.fromkeys(g for genes in panels.values() for g in genes))
        mask, reason = common_mask(cohort, needed)
        if reason:
            statuses.append(dict(**cohort.key, application=application, reference="|".join(panels),
                                 n_total=len(mask), n_common=int(mask.sum()), status="unavailable_common_subset", reason=reason))
            continue
        target = cohort.values(targets)[mask].mean(axis=1)
        scores = np.column_stack([target - cohort.values(genes)[mask].mean(axis=1) for genes in panels.values()])
        names = [d for d, keep in zip(cohort.donors, mask) if keep]
        stats = score_intervals(scores, draws, seed, (application, scope, cohort.dataset, cohort.group, *panels), len(references))
        for i, ((name, genes), result) in enumerate(zip(panels.items(), stats)):
            estimates.append(dict(**cohort.key, application=application, scope=scope, reference=name,
                                  targets="|".join(targets), reference_genes="|".join(genes), **result))
            donor_rows.extend(dict(**cohort.key, application=application, scope=scope, reference=name,
                                   donor=d, score=float(s)) for d, s in zip(names, scores[:, i]))
        if scope == "common_donors":
            pairs = list(itertools.combinations(range(len(panels)), 2))
            differences = np.column_stack([scores[:, i] - scores[:, j] for i, j in pairs])
            pair_stats = score_intervals(differences, draws, seed, (application, "reference_differences", cohort.dataset, cohort.group))
            panel_names = list(panels)
            for (i, j), result in zip(pairs, pair_stats):
                contrasts.append(dict(**cohort.key, application=application,
                                      reference_A=panel_names[i], reference_B=panel_names[j],
                                      contrast="score_A_minus_score_B", **result))
    for scope in ("per_reference", "common_donors"):
        block = [row for row in estimates if row["scope"] == scope]
        pvalues = [row["p_two_sided"] for row in block]
        pvalues += [1.0] * (len(references) - len(block))
        for row, adjusted in zip(block, holm_adjust(pvalues)):
            row["p_holm_within_family"] = float(adjusted)
    return estimates, donor_rows, statuses, contrasts


def direction_categories(scores):
    lower, upper = scores.min(axis=-1), scores.max(axis=-1)
    positive = lower > EPS
    negative = upper < -EPS
    changed = (lower < -EPS) & (upper > EPS)
    return np.stack([positive, negative, changed, ~(positive | negative | changed)], axis=-1)


def proteome_directions(cohort, targets, references, draws, seed):
    """Bootstrap category fractions over donors, conditional on observed membership."""
    usable = {name: genes for name, genes in references.items() if genes and all(g in cohort.proteins for g in genes)}
    ref_genes = list(dict.fromkeys(g for genes in usable.values() for g in genes))
    mask, reason = common_mask(cohort, targets + ref_genes)
    if len(usable) < 2:
        reason = "fewer_than_two_measurable_references"
    if reason:
        return [], [], [], dict(**cohort.key, status="unavailable", reason=reason)
    excluded = set(targets + ref_genes)
    genes = sorted(g for g, record in cohort.proteins.items()
                   if g not in excluded and np.isfinite(record[1][mask]).all())
    if not genes:
        return [], [], [], dict(**cohort.key, status="unavailable", reason="empty_complete_background")
    n = int(mask.sum())
    delta = cohort.values(genes)[mask]
    reference = np.column_stack([cohort.values(v)[mask].mean(axis=1) for v in usable.values()])
    observed = delta.mean(axis=0)[:, None] - reference.mean(axis=0)[None, :]
    span = np.ptp(observed, axis=1)
    np.testing.assert_allclose(span, np.ptp(reference.mean(axis=0)), atol=1e-12, rtol=0)
    categories = direction_categories(observed)
    category_names = ["positive_under_all", "negative_under_all", "point_sign_change", "touches_zero"]
    boot_fractions = np.empty((draws, 4))
    counts = np.zeros((len(genes), 4), dtype=np.int64)
    rng = random_generator(seed, "proteome", cohort.dataset, cohort.group)
    for start in range(0, draws, 128):
        batch = min(128, draws - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=batch)
        gene_means = weights @ delta / n
        ref_means = weights @ reference / n
        scores = gene_means[:, :, None] - ref_means[:, None, :]
        states = direction_categories(scores)
        boot_fractions[start:start + batch] = states.mean(axis=1)
        counts += states.sum(axis=0)
    quantiles = np.quantile(boot_fractions, [.025, .975], axis=0)
    summary = [dict(**cohort.key, category=category, n_donors=n, n_proteins=len(genes),
                    n_references=len(usable), count=int(categories[:, i].sum()),
                    fraction=float(categories[:, i].mean()), bootstrap_low=float(quantiles[0, i]),
                    bootstrap_high=float(quantiles[1, i]), bootstrap_draws=draws)
               for i, category in enumerate(category_names)]
    profiles = [dict(**cohort.key, gene=g, accession=cohort.proteins[g][0], n_donors=n,
                     regional_change=float(delta[:, i].mean()),
                     min_relative_score=float(observed[i].min()), max_relative_score=float(observed[i].max()),
                     category=category_names[int(np.flatnonzero(categories[i])[0])],
                     bootstrap_sign_change_fraction=float(counts[i, 2] / draws)) for i, g in enumerate(genes)]
    effects = []
    for j, name in enumerate(usable):
        scores = delta - reference[:, j, None]
        means = scores.mean(axis=0)
        width = t.ppf(.975, n - 1) * scores.std(axis=0, ddof=1) / math.sqrt(n)
        effects.extend(dict(**cohort.key, gene=g, reference=name, n=n,
                            estimate=float(means[i]), t_low=float(means[i] - width[i]),
                            t_high=float(means[i] + width[i])) for i, g in enumerate(genes))
    status = dict(**cohort.key, status="estimated", reason="", n_donors=n, n_proteins=len(genes),
                  reference_names="|".join(usable), common_span=float(span[0]),
                  span_max_minus_min=float(np.ptp(span)))
    return summary, profiles, effects, status

