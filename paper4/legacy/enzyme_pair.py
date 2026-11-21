"""Measure the donor-paired difference in CTSD and TPP1 plaque enrichment."""

from itertools import combinations, islice
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from paper4.plaque_validation.analysis import ACCESSIONS, exact_inference
from paper4.plaque_validation.inputs import DRUMMOND, cohort_inputs, read_drummond_matrices, single_gene_matrix


TARGETS = {gene: ACCESSIONS[gene] for gene in ('CTSD', 'TPP1')}
SCORE = 'ctsd_minus_tpp1_enrichment_log2'
PLAQUE = 'plaque_ctsd_minus_tpp1_log2'
ADJACENT = 'adjacent_ctsd_minus_tpp1_log2'


def donor_table(dataset, meta, genes, delta, adjacent, groups, source_column):
    cases = meta.index[meta.Group.isin(groups)]
    if not cases.is_unique or not cases.isin(delta.columns).all() or not cases.isin(adjacent.columns).all():
        raise ValueError(f'Unpaired or duplicated case identifiers in {dataset}')
    d = single_gene_matrix(genes, delta.loc[:, cases], TARGETS).T
    a = single_gene_matrix(genes, adjacent.loc[:, cases], TARGETS).T
    if np.isinf(d.to_numpy()).any() or np.isinf(a.to_numpy()).any():
        raise ValueError(f'Infinite log2 abundance in {dataset}')
    plaque = a + d
    source = (meta.loc[cases, source_column].astype("string").str.strip() if source_column
              else pd.Series(pd.NA, index=cases, dtype="string"))
    table = pd.DataFrame({'dataset': dataset, 'group': meta.loc[cases, 'Group'], 'source': source,
                          'ctsd_enrichment_log2': d.CTSD, 'tpp1_enrichment_log2': d.TPP1,
                          PLAQUE: plaque.CTSD - plaque.TPP1, ADJACENT: a.CTSD - a.TPP1})
    table['complete_pair'] = (a.notna() & d.notna()).all(axis=1)
    table[SCORE] = (d.CTSD - d.TPP1).where(table.complete_pair)
    np.testing.assert_allclose(table[SCORE], table[PLAQUE] - table[ADJACENT], atol=1e-10, rtol=0,
                               equal_nan=True)
    return table.rename_axis('case_id').reset_index()


def read_donors(data):
    tables = []
    for dataset, meta, genes, delta, adjacent in cohort_inputs(data):
        groups, source = (('AD',), 'Study') if dataset == 'Leitner' else (('EOAD', 'LOAD'), 'Source')
        tables.append(donor_table(dataset, meta, genes, delta, adjacent, groups, source))
    labels, genes, delta, adjacent = read_drummond_matrices(data / DRUMMOND)
    tables.append(donor_table('Drummond 2022', labels.rename('Group').to_frame(), genes, delta,
                              adjacent, ('eoAD',), None))
    return pd.concat(tables, ignore_index=True)


def cohort_estimates(donors):
    results = []
    for (dataset, group), t in donors.groupby(['dataset', 'group'], sort=False):
        y = t[SCORE].dropna()
        row = {'dataset': dataset, 'group': group, **exact_inference(t[SCORE])}
        if len(y) >= 4:
            leave_one = (y.sum() - y) / (len(y) - 1)
            row.update(median_log2=y.median(), leave_one_donor_out_min=leave_one.min(),
                       leave_one_donor_out_max=leave_one.max())
        results.append(row)
    return pd.DataFrame(results)


def batched(values, size):
    values = iter(values)
    while batch := tuple(islice(values, size)):
        yield batch


def exact_label_test(early, late):
    """Two-sided difference of means under label exchangeability within a brain bank."""
    y = np.concatenate((early, late))
    k = len(early)
    n_allocations = comb(len(y), k)
    observed = abs(early.mean() - late.mean())
    total = y.sum()
    exceed = 0
    for chunk in batched(combinations(range(len(y)), k), 4096):
        sums = y[np.array(chunk)].sum(axis=1)
        null = sums / k - (total - sums) / len(late)
        exceed += np.count_nonzero(abs(null) >= observed - 1e-12)
    return n_allocations, exceed / n_allocations


def compare_onset(t, scope, shared_bank=False):
    rows = []
    for endpoint in (SCORE, PLAQUE, ADJACENT):
        early = t.loc[t.group.eq('EOAD'), endpoint].to_numpy()
        late = t.loc[t.group.eq('LOAD'), endpoint].to_numpy()
        result = {'scope': scope, 'endpoint': endpoint, 'n_early': len(early), 'n_late': len(late)}
        if min(len(early), len(late)) < 2:
            rows.append({**result, 'status': 'insufficient_donors'})
            continue
        test = stats.ttest_ind(early, late, equal_var=False)
        ci = test.confidence_interval()
        difference = early.mean() - late.mean()
        result.update(status='estimated', early_mean=early.mean(), late_mean=late.mean(),
                      early_minus_late_log2=difference, ci95_low=ci.low, ci95_high=ci.high,
                      welch_p=test.pvalue)
        if endpoint == SCORE:
            omit_early = (early.sum() - early) / (len(early) - 1) - late.mean()
            omit_late = early.mean() - (late.sum() - late) / (len(late) - 1)
            omit = np.concatenate((omit_early, omit_late))
            result.update(leave_one_donor_out_min=omit.min(), leave_one_donor_out_max=omit.max())
            if shared_bank:
                result['n_label_allocations'], result['exact_label_p'] = exact_label_test(early, late)
        rows.append(result)
    if all(r['status'] == 'estimated' for r in rows):
        np.testing.assert_allclose(rows[0]['early_minus_late_log2'],
                                   rows[1]['early_minus_late_log2'] - rows[2]['early_minus_late_log2'],
                                   atol=1e-10, rtol=0)
    return rows


def source_analysis(donors):
    marta = donors.loc[donors.dataset.eq('Marta-Ariza')].copy()
    if marta.source.isna().any() or marta.source.eq('').any():
        raise ValueError('Missing Marta-Ariza brain-bank label')
    source_counts = (marta.groupby(['source', 'group']).agg(eligible=('case_id', 'size'),
                     complete=('complete_pair', 'sum'), mean_enrichment_difference=(SCORE, 'mean')).reset_index())
    complete = marta.loc[marta.complete_pair]
    results = compare_onset(complete, 'Marta-Ariza, all sources')
    for name, part in complete.groupby('source'):
        if {'EOAD', 'LOAD'} == set(part.group):
            results.extend(compare_onset(part, name, shared_bank=True))
    return source_counts, pd.DataFrame(results)
