"""Fixed panel and statistics for the human plaque analyses."""

from itertools import product
import re

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .inputs import AGE, CAA, PMI


from .panel_definition import ACCESSIONS, DEN, NUM
GRADES = {'mild': 1., 'mild to moderate': 1.5, 'moderate': 2., 'severe': 3., 'abundant': 3.}
AFTER_CAA = re.compile(r'\bCAA\s*\(\s*(mild to moderate|mild|moderate|severe|abundant)\s*\)', re.I)
ARTERIOLAR = re.compile(r'\b(mild|moderate|severe)\s+arteriolar\s+A[ßβb]-CAA\s*\(present\)', re.I)


def paired_estimate(y):
    n = len(y)
    if n < 4:
        return {'n_paired':n,'status':'insufficient_coverage'}
    test = stats.ttest_1samp(y,0)
    ci = test.confidence_interval()
    return {'n_paired':n,'n_positive':int((y>0).sum()),'estimate_log2':float(y.mean()),
            'ci95_low':float(ci.low),'ci95_high':float(ci.high),'p_value':float(test.pvalue),'status':'estimated'}


def contrast(data):
    return data.loc[NUM].mean(axis=0, skipna=False) - data.loc[DEN].mean(axis=0, skipna=False)


def exact_inference(values):
    x = values.dropna()
    estimate = paired_estimate(x)
    if 4 <= len(x) <= 17:
        signs = np.array(list(product((-1, 1), repeat=len(x))))
        null = signs @ x.to_numpy() / len(x)
        estimate['sign_flip_p'] = np.mean(np.abs(null) >= abs(x.mean()) - 1e-12)
    return {'n_eligible': len(values), **estimate}


def complete_columns(panel):
    return panel.columns[panel.notna().all(axis=0)]


def summarize(panel, dataset, group, endpoint, eligible=None):
    if not panel.columns.is_unique or panel.columns.empty or np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError(f'Invalid donor keys or quantities for {dataset}/{group}/{endpoint}')
    complete = complete_columns(panel)
    if eligible is not None:
        complete = complete.intersection(eligible)
    complete_panel = panel.loc[:, complete]
    score = contrast(complete_panel).reindex(panel.columns)
    key = {'dataset': dataset, 'group': group, 'endpoint': endpoint}
    result = {**key, **exact_inference(score), 'positive_fraction': score.gt(0).sum() / len(complete) if len(complete) else np.nan}
    donors = score.rename('score_log2').rename_axis('case_code').reset_index().assign(**key)
    coverage = [{**key, 'gene': gene, 'accession': ACCESSIONS[gene],
                 'donors_with_paired_values': panel.loc[gene].count(), 'donors_eligible': panel.shape[1],
                 'complete_panel_donors_used': len(complete)} for gene in ACCESSIONS]
    checks = []
    if len(complete) >= 4:
        for case in complete:
            v = score.drop(index=case).dropna()
            checks.append({**key, 'omission_type': 'donor', 'omitted': case, **exact_inference(v)})
        for gene in ACCESSIONS:
            h = complete_panel.loc[[g for g in NUM if g != gene]].mean(axis=0, skipna=False)
            v = complete_panel.loc[[g for g in DEN if g != gene]].mean(axis=0, skipna=False)
            checks.append({**key, 'omission_type': 'protein', 'omitted': gene, **exact_inference(h - v)})
    return result, donors, coverage, checks


def severity(text):
    grades = {m.lower() for rule in [AFTER_CAA, ARTERIOLAR] for m in rule.findall(str(text))}
    if len(grades) > 1:
        raise ValueError('Conflicting CAA grades in one source case summary')
    return next(iter(grades)) if grades else None


def evaluate(values, covars):
    data = values[['score', 'grade', *covars]].dropna()
    if len(data) < 6 or data.grade.nunique() < 2:
        return {'n': len(data), 'status': 'insufficient graded coverage'}
    numeric = pd.get_dummies(data.drop(columns='score'), columns=[c for c in ('Group', 'Source') if c in covars],
                             drop_first=True, dtype=float)
    design = sm.add_constant(numeric)
    if np.linalg.matrix_rank(design) != design.shape[1] or len(design) <= design.shape[1] + 3:
        return {'n': len(data), 'status': 'rank deficient'}
    result = sm.OLS(data.score, design).fit(cov_type='HC3', use_t=True)
    lo, hi = result.conf_int().loc['grade']
    rho = stats.spearmanr(data.grade, data.score)
    return {'n': len(data), 'status': 'estimated', 'slope_grade': result.params.grade, 'ci95_low': lo,
            'ci95_high': hi, 'p_HC3': result.pvalues.grade, 'unadjusted_spearman': rho.statistic}


def grade_results(leitner, marta):
    lmeta, lpanel = leitner
    grade = lmeta.loc[lmeta.Group.eq('AD')].copy()
    grade['score'] = contrast(lpanel)
    grade['grade'] = pd.to_numeric(grade[CAA], errors='coerce')
    grade['age_10y'] = pd.to_numeric(grade[AGE], errors='coerce') / 10
    grade['female'] = grade.Sex.map({'F': 1., 'M': 0.})
    grade['PMI_hours'] = pd.to_numeric(grade[PMI], errors='coerce')
    results = [{'dataset': 'Leitner', 'group': 'AD', 'scale': 'author 0-5 regional score',
                **evaluate(grade, ['age_10y', 'female', 'PMI_hours'])}]

    meta, panel = marta
    grade = meta.loc[meta.Group.isin(['EOAD', 'LOAD'])].assign(score=contrast(panel))
    grade['grade'] = meta['Pathology details'].map(severity).map(GRADES)
    for label, data in grade.groupby('Group'):
        results.append({'dataset': 'Marta-Ariza', 'group': label, 'scale': 'explicit author mild/moderate/severe wording',
                        **evaluate(data, [])})
    return pd.DataFrame(results)
