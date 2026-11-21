"""Compare the locked plaque panel with abundance- and coverage-matched proteins."""


import numpy as np
import pandas as pd

from paper4.plaque_validation.analysis import ACCESSIONS, DEN, NUM, complete_columns, contrast
from paper4.plaque_validation.panel_definition import draw_panels
from paper4.plaque_validation.inputs import (
    DRUMMOND, LEITNER, MARTA, cohort_inputs, read_drummond_matrices,
    single_gene_matrix, verify_workbooks,
)


TARGETS = NUM + DEN
CALIPERS = (0.5, 1.0)
SEED = 20261117


def plaque_groups(data_dir):
    for name, meta, genes, delta, adjacent in cohort_inputs(data_dir):
        groups = ('AD',) if name == 'Leitner' else ('EOAD', 'LOAD')
        for group in groups:
            donors = delta.columns[delta.columns.isin(meta.index[meta.Group.eq(group)])]
            yield name, group, genes, delta.loc[:, donors], adjacent.loc[:, donors]
    labels, genes, delta, adjacent = read_drummond_matrices(data_dir / DRUMMOND)
    donors = delta.columns[labels.reindex(delta.columns).eq('eoAD')]
    yield 'Drummond 2022', 'eoAD', genes, delta.loc[:, donors], adjacent.loc[:, donors]




def compare_group(genes, delta, adjacent, key, count, rng):
    if not delta.columns.is_unique or delta.columns.empty:
        raise ValueError(f'Invalid donor IDs for {key}')
    if any(np.isinf(matrix.to_numpy(dtype=float)).any() for matrix in (delta, adjacent)):
        raise ValueError(f'Infinite protein quantities for {key}')
    target_delta = single_gene_matrix(genes, delta, ACCESSIONS).loc[TARGETS]
    donors = complete_columns(target_delta)
    if len(donors) < 4:
        raise ValueError(f'Fewer than four donors with complete target coverage for {key}')
    target_delta = target_delta.loc[:, donors]
    target_abundance = single_gene_matrix(genes, adjacent.loc[:, donors], ACCESSIONS).median(axis=1)
    target_numerator = target_delta.loc[NUM].to_numpy().mean()
    target_denominator = target_delta.loc[DEN].to_numpy().mean()
    observed = float(contrast(target_delta).mean())

    controls = single_gene_matrix(genes, delta.loc[:, donors]).drop(index=TARGETS)
    baseline = single_gene_matrix(genes, adjacent.loc[:, donors]).reindex(controls.index)
    complete = (controls.notna() & baseline.notna()).all(axis=1)
    medians = baseline.loc[complete].median(axis=1).sort_index()
    changes = controls.loc[medians.index].mean(axis=1).to_numpy()
    estimates, matches = [], []
    for caliper in CALIPERS:
        pools = []
        for target in TARGETS:
            offsets = medians - target_abundance[target]
            indices = np.flatnonzero(offsets.abs().le(caliper))
            pools.append(indices)
            for index in indices:
                matches.append({
                    **key, 'caliper_log2': caliper, 'target_gene': target,
                    'target_median_adjacent_log2': target_abundance[target],
                    'control_gene': medians.index[index],
                    'control_median_adjacent_log2': medians.iloc[index],
                    'abundance_difference_log2': offsets.iloc[index],
                })
        middle = len(NUM)
        for design, slots, n_mock_hydrolases in (
            ('both_arms_matched', pools, middle),
            ('hydrolases_replaced', pools[:middle], middle),
            ('v_atpase_replaced', pools[middle:], 0),
        ):
            try:
                draws = draw_panels(slots, count, rng)
            except ValueError as exc:
                raise ValueError(f'{key}, caliper {caliper}, {design}: {exc}') from exc
            matched = changes[draws]
            numerator = matched[:, :n_mock_hydrolases].mean(axis=1) if n_mock_hydrolases else target_numerator
            denominator = (matched[:, n_mock_hydrolases:].mean(axis=1)
                           if n_mock_hydrolases < len(slots) else target_denominator)
            reference = numerator - denominator
            lo, median, hi = np.quantile(reference, [0.025, 0.5, 0.975])
            estimates.append({
                **key, 'caliper_log2': caliper, 'design': design,
                'n_eligible_donors': delta.shape[1], 'n_complete_donors': len(donors),
                'n_complete_background_genes': len(medians), 'min_candidates_per_slot': min(map(len, slots)),
                'n_panel_draws': len(draws), 'n_distinct_ordered_panels': len(np.unique(draws, axis=0)),
                'observed_log2': observed, 'observed_hydrolase_log2': target_numerator,
                'observed_v_atpase_log2': target_denominator,
                'reference_q025_log2': lo, 'reference_median_log2': median, 'reference_q975_log2': hi,
                'observed_minus_reference_median_log2': observed - median,
                'reference_n_ge_observed': int(np.count_nonzero(reference >= observed - 1e-12)),
                'reference_fraction_ge_observed': float(np.mean(reference >= observed - 1e-12)),
            })
    return estimates, matches
