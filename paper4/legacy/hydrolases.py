"""Run the ten follow-up checks on the published AD plaque and CAA workbooks."""

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from paper4.plaque_validation.analysis import ACCESSIONS, DEN, NUM, complete_columns, exact_inference
from paper4.plaque_validation.cli import PLAQUE_GROUPS, ROOT, DRUMMOND as DRUMMOND_DATASET
from paper4.plaque_validation.inputs import (
    CAA_VESSELS, DRUMMOND, cohort_inputs, read_drummond_matrices, read_vessel_matrices,
    single_gene_matrix, verify_workbooks,
)


ANNOTATIONS = ROOT / 'plaque_validation' / 'uniprot_hydrolase_sets.json'
PRIMARY = 'lysosomal_lumen_hydrolases'
CLASSES = (PRIMARY, 'lysosome_hydrolases', 'lysosome_peptidases')
CALIPERS = (0.5, 1.0)
BOOTSTRAPS = 10000
SEED = 20261118


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def class_matrices(genes, delta, classes):
    """Map a reviewed accession and require a matching single reported gene symbol."""
    for label, entries in classes.items():
        membership = entries.loc[~entries.primary_gene.duplicated(keep=False)]
        valid = genes.Genes.reindex(membership.index).eq(membership.primary_gene)
        yield label, delta.reindex(membership.index[valid.fillna(False)])


def annotation_tables(payload):
    classes = {}
    for label in CLASSES:
        entries = pd.DataFrame(payload['classes'][label]['entries'], columns=payload['entry_fields'])
        entries = entries.set_index('accession').sort_index()
        if not entries.index.is_unique:
            raise ValueError(f'Duplicate annotated accessions in {label}')
        if any(entries.primary_gene.get(ACCESSIONS[g]) != g for g in NUM):
            raise ValueError(f'Expected two annotated hydrolases in {label}')
        classes[label] = entries
    return classes


def plaque_data(data_dir):
    cohorts = {name: (meta, genes, delta, adjacent)
               for name, meta, genes, delta, adjacent in cohort_inputs(data_dir)}
    labels, genes, delta, adjacent = read_drummond_matrices(data_dir / DRUMMOND)
    cohorts[DRUMMOND_DATASET] = (labels.rename('Group').to_frame(), genes, delta, adjacent)
    return cohorts


def control_coverage(entries, measured, adjacent, references, key):
    coverage = entries.copy()
    coverage['paired_donors'] = measured.count(axis=1).reindex(coverage.index, fill_value=0)
    coverage['eligible_donors'] = measured.shape[1]
    coverage['matches_workbook_accession_and_gene'] = coverage.index.isin(measured.index)
    coverage['median_adjacent_log2'] = adjacent.median(axis=1)
    coverage['target'] = coverage.index.isin([ACCESSIONS[g] for g in NUM])
    coverage['eligible_control'] = coverage.index.isin(references.index)
    return coverage.reset_index().assign(**key)


def compare_pairs(references, baseline, targets, target_baseline, key):
    """Enumerate unique unordered replacement pairs with at least one valid assignment."""
    observed = targets.mean(axis=1).mean()
    control_means = references.mean(axis=1)
    for caliper in CALIPERS:
        pools = [baseline.index[(baseline - target_baseline[gene]).abs().le(caliper)] for gene in NUM]
        pairs = sorted({tuple(sorted((a, b))) for a, b in product(*pools) if a != b})
        values = np.array([(control_means[a] + control_means[b]) / 2 for a, b in pairs])
        record = {**key, 'caliper_log2': caliper, 'n_donors': len(targets), 'n_ctsd_matches': len(pools[0]),
                  'n_tpp1_matches': len(pools[1]), 'n_distinct_pairs': len(values), 'observed_log2': observed}
        if len(values):
            count = int(np.count_nonzero(values >= observed - 1e-12))
            record.update(status='estimated', reference_median_log2=np.median(values),
                          n_pairs_ge_observed=count, fraction_ge_observed=count / len(values))
        else:
            record['status'] = 'no_distinct_matched_pair'
        yield record


def peer_ranks(references, baseline, targets, target_baseline, key):
    changes = references.mean(axis=1)
    for gene in NUM:
        for caliper in (None, *CALIPERS):
            peers = changes if caliper is None else changes.loc[(baseline - target_baseline[gene]).abs().le(caliper)]
            observed = targets[gene].mean()
            yield {**key, 'gene': gene, 'caliper_log2': caliper, 'n_controls': len(peers),
                   'observed_log2': observed, 'n_controls_ge_observed': int(peers.ge(observed - 1e-12).sum()),
                   'control_median_log2': peers.median() if len(peers) else np.nan}


def reference_means(scores, controls, key, targets=NUM):
    if len(controls) < 2:
        return [{**key, 'comparison': 'CTSD+TPP1 vs controls', 'n_controls': len(controls),
                 'n_eligible': len(scores), 'status': 'fewer_than_two_complete_controls'}]
    baseline = controls.mean(axis=0)
    numerator = {'CTSD+TPP1': scores[NUM].mean(axis=1)}
    numerator.update({gene: scores[gene] for gene in targets})
    return [{**key, 'comparison': f'{label} vs controls', 'n_controls': len(controls),
             'control_genes': '|'.join(controls.index), **exact_inference(y - baseline)}
            for label, y in numerator.items()]


def sensitivity(values, key, rng):
    y = values.to_numpy()
    boot = y[rng.integers(len(y), size=(BOOTSTRAPS, len(y)))].mean(axis=1)
    omit_one = (y.sum() - y) / (len(y) - 1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {**key, 'n_donors': len(y), 'estimate_log2': y.mean(),
            'bootstrap_ci95_low': lo, 'bootstrap_ci95_high': hi, 'donor_median_log2': np.median(y),
            'leave_one_out_min_log2': omit_one.min(), 'leave_one_out_max_log2': omit_one.max()}


def usable_controls(matrix, adjacent):
    quantified = (matrix.notna() & adjacent.reindex(matrix.index).notna()).all(axis=1)
    return matrix.loc[quantified & ~matrix.index.isin(ACCESSIONS.values())]


def plaque_checks(cohorts, classes):
    target_rows, coverage, comparisons, matched, ranks, robust = ([] for _ in range(6))
    panels = {}
    rng = np.random.default_rng(SEED)
    for dataset, (meta, genes, delta, adjacent) in cohorts.items():
        for group in PLAQUE_GROUPS[dataset]:
            print(f'{dataset} / {group}', flush=True)
            key = {'dataset': dataset, 'group': group}
            ids = delta.columns[delta.columns.isin(meta.index[meta.Group.eq(group)])]
            fixed = single_gene_matrix(genes, delta.loc[:, ids], ACCESSIONS)
            donors = complete_columns(fixed)
            if len(donors) < 4:
                raise ValueError(f'Fewer than four donors with a complete fixed panel in {key}')
            targets = fixed.loc[:, donors].T
            near = adjacent.loc[:, donors]
            target_baseline = single_gene_matrix(genes, near, ACCESSIONS).median(axis=1, skipna=False)
            if not np.isfinite(targets.to_numpy()).all() or not np.isfinite(target_baseline).all():
                raise ValueError(f'Invalid fixed-panel quantities in {key}')
            hydro = targets[NUM].mean(axis=1)
            vatpase = targets[DEN].mean(axis=1)
            fixed_contrast = hydro - vatpase
            series = dict(CTSD=targets.CTSD, TPP1=targets.TPP1, hydrolase_mean=hydro,
                          v_atpase_mean=vatpase, target_vs_v_atpase=fixed_contrast)
            target_rows.extend({**key, 'endpoint': name, **exact_inference(y)} for name, y in series.items())
            robust.append(sensitivity(fixed_contrast, {**key, 'endpoint': 'target_vs_v_atpase'}, rng))
            matrices = {}
            for label, matrix in class_matrices(genes, delta.loc[:, donors], classes):
                tagged = {**key, 'reference_class': label}
                near_matrix = near.reindex(matrix.index)
                if any(np.isinf(x.to_numpy(dtype=float)).any() for x in (matrix, near_matrix)):
                    raise ValueError(f'Infinite reference quantities in {tagged}')
                accessions = usable_controls(matrix, near_matrix)
                names = classes[label].primary_gene.loc[accessions.index]
                controls = accessions.set_axis(names, axis=0)
                medians = near_matrix.loc[accessions.index].median(axis=1).set_axis(names, axis=0)
                coverage.append(control_coverage(classes[label], matrix, near_matrix, accessions, tagged))
                comparisons.extend(reference_means(targets, controls, tagged))
                matched.extend(compare_pairs(controls, medians, targets, target_baseline, tagged))
                ranks.extend(peer_ranks(controls, medians, targets, target_baseline, tagged))
                if len(controls) >= 2:
                    robust.append(sensitivity(hydro - controls.mean(axis=0),
                                              {**key, 'endpoint': label}, rng))
                matrices[label] = accessions
            panels[dataset, group] = (targets, matrices)
    return panels, {
        '02_target_effects.csv': target_rows,
        '03_annotation_coverage.csv': pd.concat(coverage, ignore_index=True),
        '04_class_comparisons.csv': comparisons,
        '05_abundance_matched_pairs.csv': matched,
        '06_protein_ranks.csv': ranks,
        '08_donor_robustness.csv': robust,
    }


def shared_background(panels, classes):
    rows = []
    for label, entries in classes.items():
        shared = sorted(set.intersection(*(set(m[label].index) for _, m in panels.values())))
        for (dataset, group), (targets, matrices) in panels.items():
            controls = matrices[label].loc[shared].set_axis(entries.primary_gene.loc[shared], axis=0)
            key = {'dataset': dataset, 'group': group, 'reference_class': label,
                   'control_accessions': '|'.join(shared)}
            rows.extend(reference_means(targets, controls, key, targets=()))
    return rows


def caa_checks(data_dir, leitner, classes):
    meta, plaque_genes, plaque_delta, _ = leitner
    vessel_genes, vessel_delta, labels, shared = read_vessel_matrices(data_dir / CAA_VESSELS, meta)
    ad = shared[shared.map(labels) == 'AD']
    if len(ad) != 6 or not meta.loc[ad, 'Group'].eq('AD').all():
        raise ValueError('Expected six verified AD donor links')
    plaque = single_gene_matrix(plaque_genes, plaque_delta.loc[:, ad], ACCESSIONS)
    vessel = single_gene_matrix(vessel_genes, vessel_delta.loc[:, ad], ACCESSIONS)
    paired = complete_columns(plaque).intersection(complete_columns(vessel))
    if len(paired) < 4:
        raise ValueError('Fewer than four linked AD donors with complete fixed panels')
    plaque, vessel = plaque.loc[:, paired], vessel.loc[:, paired]
    references = []
    p_classes = dict(class_matrices(plaque_genes, plaque_delta.loc[:, paired], classes))
    v_classes = dict(class_matrices(vessel_genes, vessel_delta.loc[:, paired], classes))
    for label, entries in classes.items():
        p, v = p_classes[label].dropna(), v_classes[label].dropna()
        common = p.index.intersection(v.index).difference(ACCESSIONS.values())
        if any(np.isinf(x.loc[common].to_numpy(dtype=float)).any() for x in (p, v)):
            raise ValueError(f'Infinite linked reference quantities in {label}')
        key = {'reference_class': label, 'n_controls': len(common),
               'control_genes': '|'.join(entries.primary_gene.loc[common])}
        if len(common) < 2:
            references.append({**key, 'endpoint': 'plaque minus CAA', 'n_eligible': len(paired),
                               'status': 'fewer_than_two_complete_controls'})
            continue
        ph = plaque.loc[NUM].mean(axis=0) - p.loc[common].mean(axis=0)
        vh = vessel.loc[NUM].mean(axis=0) - v.loc[common].mean(axis=0)
        for endpoint, y in (('plaque', ph), ('CAA', vh), ('plaque minus CAA', ph - vh)):
            references.append({**key, 'endpoint': endpoint, **exact_inference(y)})
    return references
