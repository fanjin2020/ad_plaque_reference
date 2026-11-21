"""One command to reproduce the plaque lysosome results on a server."""


from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import ACCESSIONS, DEN, NUM, complete_columns, contrast, grade_results, summarize
from .inputs import CAA_VESSELS, DRUMMOND as DRUMMOND_FILE, cohort_inputs, read_drummond, read_vessel
from .inputs import single_gene_matrix, verify_workbooks


ROOT = Path(__file__).resolve().parents[1]
DRUMMOND = 'Drummond 2022'
LINKED = 'Leitner linked plaque and CAA'
PLAQUE_GROUPS = {'Leitner': ['AD'], 'Marta-Ariza': ['EOAD', 'LOAD'], DRUMMOND: ['eoAD']}


def write_outputs(folder, data_dir):
    cohorts = {name: (meta, single_gene_matrix(genes, delta, ACCESSIONS))
               for name, meta, genes, delta, _ in cohort_inputs(data_dir)}
    drummond, dgroup, _ = read_drummond(data_dir / DRUMMOND_FILE, ACCESSIONS)
    cohorts[DRUMMOND] = (dgroup.rename('Group').to_frame(), drummond)
    lmeta, lpanel = cohorts['Leitner']
    vessels, vgroup, linked = read_vessel(data_dir / CAA_VESSELS, ACCESSIONS, lmeta)
    shared_ad = linked[linked.map(vgroup) == 'AD']
    if len(shared_ad) != 6 or not lmeta.loc[shared_ad, 'Group'].eq('AD').all():
        raise ValueError('The linked plaque and CAA check did not recover the expected six AD donors')
    p, v = lpanel.loc[:, shared_ad], vessels.loc[:, shared_ad]
    jointly_complete = complete_columns(p).intersection(complete_columns(v))
    caa_donors = pd.DataFrame({'plaque_log2': contrast(p), 'caa_log2': contrast(v)})
    caa_donors.loc[~caa_donors.index.isin(jointly_complete), ['plaque_log2', 'caa_log2']] = np.nan
    caa_donors['plaque_minus_caa_log2'] = caa_donors.plaque_log2 - caa_donors.caa_log2

    comparisons = []
    for name, (meta, panel) in cohorts.items():
        for group in PLAQUE_GROUPS[name]:
            ids = meta.index[meta.Group.eq(group)].intersection(panel.columns)
            comparisons.append(summarize(panel.loc[:, ids], name, group, 'Aβ plaque minus adjacent tissue'))
    n_main = len(comparisons)
    for label, matrix in [('plaque', p), ('CAA', v), ('plaque minus CAA', p - v)]:
        comparisons.append(summarize(matrix, LINKED, 'AD', label, jointly_complete))

    estimates, *_ = zip(*comparisons)
    main = pd.DataFrame(estimates[:n_main])
    linked_results = pd.DataFrame(estimates[n_main:])
    main.to_csv(folder / 'main_results.csv', index=False)
    linked_results.to_csv(folder / 'caa_paired_results.csv', index=False)
    caa_donors.to_csv(folder / 'caa_paired_donors.csv', index_label='case_code')
    grade_results(cohorts["Leitner"], cohorts["Marta-Ariza"]).to_csv(
        folder / "caa_grades_exploratory.csv", index=False)
    return main, linked_results
