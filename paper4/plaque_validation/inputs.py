"""Read and validate the four published human proteomics workbooks."""

import hashlib
import json
import re

import numpy as np
import pandas as pd


LEITNER = 'leitner_2026_S5.xlsx'
MARTA = 'marta_2025_S10.xlsx'
DRUMMOND = 'drummond2022_plaque_supp.xlsx'
CAA_VESSELS = 'leitner_2024_caa_S2.xlsx'
LEITNER_SAMPLE = re.compile(r'(AD|MCI|Control)__(p|np)__(B\d+|ROS\d+)')
MARTA_SAMPLE = re.compile(r'((DS|EOAD|LOAD|Control)_\d+) (Plaque|Non-plaque)')
CAA_SAMPLE = re.compile(r'CAA(pos|neg)_(AD|MCI|Control)_(R\d+)')
AGE, PMI, CASE = 'Age', 'PMI', 'Case ID'
CAA = 'Regional CAA \n(score 0-5)'
GROUPS = {'AD': 53, 'MCI': 36, 'Control': 62}
LINK_FIELDS = ['Group', 'Age', 'Sex', 'PMI', 'Study', 'APOE Genotype']


def read_workbook_manifest(manifest):
    sources = json.loads(manifest.read_text(encoding='utf-8'))
    if len(sources) != 4 or {row['name'] for row in sources} != {LEITNER, MARTA, DRUMMOND, CAA_VESSELS}:
        raise ValueError('Data manifest must list the four validated workbooks once')
    return sources


def verify_workbooks(data_dir, manifest, *, required=None):
    hashes = {}
    for source in read_workbook_manifest(manifest):
        if required is not None and source['name'] not in required:
            continue
        path = data_dir / source['name']
        if not path.is_file():
            raise FileNotFoundError(f'Input workbook missing: {path}. See README.md.')
        with path.open('rb') as handle:
            value = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != source['bytes'] or value != source['sha256']:
            raise ValueError(f'Workbook does not match the validated edition: {path}. See data_manifest.json.')
        hashes[path.name] = value
    return hashes


def within_donor(matrix, samples, plaque, adjacent):
    plaque_cols = samples.loc[samples.tissue.eq(plaque)].set_index('case').sort_index()
    adjacent_cols = samples.loc[samples.tissue.eq(adjacent)].set_index('case')
    found = samples.reset_index(names='column').pivot(index='case', columns='tissue', values='column')
    same = found.loc[plaque_cols.index, [plaque, adjacent]]
    if same.isna().any().any() or not same.index.isin(adjacent_cols.index).all():
        raise ValueError('Plaque and adjacent samples are not aligned')
    inside = matrix.loc[:, same[plaque]].set_axis(same.index, axis=1)
    outside = matrix.loc[:, same[adjacent]].set_axis(same.index, axis=1)
    return inside - outside, outside


def single_gene_matrix(genes, delta, expected_accessions=None):
    g = genes.Genes.dropna().astype(str)
    g = g.loc[g.str.fullmatch(r'[A-Z][A-Z0-9-]+') & ~g.duplicated(keep=False)]
    if expected_accessions is not None:
        expected = pd.Series({accession: gene for gene, accession in expected_accessions.items()})
        matching = g.reindex(expected.index)
        mismatches = expected.loc[matching.fillna('').ne(expected)]
        if not mismatches.empty:
            raise ValueError('Missing or ambiguous protein identity: ' + ', '.join(mismatches))
        g = matching
    return delta.loc[g.index].set_axis(g, axis=0)


def read_leitner(path):
    f = pd.ExcelFile(path)
    meta = f.parse('Supplementary Table 1', header=1, dtype={CASE:str})
    meta = meta.loc[meta['Analysis Performed'].str.contains(r'\bProteomics\b', na=False)].copy()
    if meta['Group'].value_counts().to_dict() != GROUPS or meta[CASE].duplicated().any():
        raise ValueError('The author case table does not contain the expected 151 unique proteomics donors')
    meta = meta.set_index(CASE)
    t = f.parse('Supplementary Table 2', header=1)
    f.close()
    cols = [k for k in t.columns if LEITNER_SAMPLE.fullmatch(str(k))]
    sample_info = {k:LEITNER_SAMPLE.fullmatch(k).groups() for k in cols}
    si = pd.DataFrame.from_dict(sample_info, orient='index', columns=['group','tissue','case'])
    si.index.name = 'author_sample'
    n = si.groupby(['group','tissue']).size().to_dict()
    if n != {('AD','p'):53,('AD','np'):53,('MCI','p'):36,('MCI','np'):36,('Control','np'):62}:
        raise ValueError(f'Unexpected sample counts by group and tissue: {n}')
    if si.duplicated(['case','tissue']).any() or set(si['case'])!=set(meta.index):
        raise ValueError('The author LC-MS sample headers do not uniquely match their case table')
    if not si['case'].map(meta['Group']).eq(si['group']).all():
        raise ValueError('A sample header contradicted its clinical group')
    if not si.groupby('case').size().eq(meta.loc[si.groupby('case').size().index,'Group'].map({'AD':2,'MCI':2,'Control':1})).all():
        raise ValueError('Wrong plaque/adjacent pairing within donors')
    row_ids = t['ProteinGroupRepresentative'].astype('string')
    t = t.loc[~row_ids.duplicated(keep=False)].copy().set_index('ProteinGroupRepresentative')
    mat = t[cols].apply(pd.to_numeric, errors='raise')
    if np.isinf(mat).any().any():
        raise ValueError('Protein table has invalid log quantities')
    return meta, t, si, mat


def read_marta(path):
    with pd.ExcelFile(path) as book:
        meta = book.parse('Supplementary Table 1',header=1,usecols=['Group','Case ID','Source','Pathology details'])
        p = book.parse('Supplementary Table 3',header=1)
    meta = meta.loc[meta['Group'].isin(('DS','EOAD','LOAD','Control'))].set_index('Case ID')
    if not meta.index.is_unique:
        raise ValueError('Duplicate case ID in Martá-Ariza case table')
    if meta.Group.value_counts().to_dict()!={'DS':20,'EOAD':20,'LOAD':20,'Control':20}:
        raise ValueError('Unexpected number of cases in the author workbook')
    entries = {c:MARTA_SAMPLE.fullmatch(str(c)).groups() for c in p.columns if MARTA_SAMPLE.fullmatch(str(c))}
    si = pd.DataFrame.from_dict(entries,orient='index',columns=['case','group','tissue'])
    if len(si)!=140 or set(si['case'])!=set(meta.index) or si.duplicated(['case','tissue']).any():
        raise ValueError('The author sample identifiers could not be matched')
    if not si['case'].map(meta.Group).eq(si.group).all():
        raise ValueError('The sample names disagreed with case-group designations')
    expected = meta.Group.ne('Control').astype(int)+1
    if not si.groupby('case').size().eq(expected).all():
        raise ValueError('Some samples had an unexpected pairing structure')
    return meta, p, si


def cohort_inputs(data_dir):
    meta, proteins, samples, matrix = read_leitner(data_dir / LEITNER)
    delta, adjacent = within_donor(matrix, samples, 'p', 'np')
    yield 'Leitner', meta, proteins[['Genes']], delta, adjacent

    meta, proteins, samples = read_marta(data_dir / MARTA)
    proteins = proteins.dropna(subset=['Uniprot ID']).set_index('Uniprot ID')
    proteins = proteins.loc[~proteins.index.duplicated(keep=False)]
    matrix = proteins.loc[:, samples.index].apply(pd.to_numeric, errors='raise')
    delta, adjacent = within_donor(matrix, samples, 'Plaque', 'Non-plaque')
    yield 'Marta-Ariza', meta, proteins[['Gene ID']].rename(columns={'Gene ID': 'Genes'}), delta, adjacent


def read_drummond_matrices(book):
    p = pd.read_excel(book, sheet_name='Table 2 - Unimputed Data', header=4)
    info = pd.read_excel(book, sheet_name='Table 1 - Imputed Data', header=4,
                         usecols=['Uniprot ID', 'Majority Gene ID', 'All Gene IDs', 'Peptides'])
    info = info.dropna(subset=['Uniprot ID']).set_index('Uniprot ID')
    p = p.dropna(subset=['Uniprot ID']).set_index('Uniprot ID')
    p = p.loc[~p.index.duplicated(keep=False)]
    info = info.loc[~info.index.duplicated(keep=False)].reindex(p.index)
    p = p.loc[p.index.astype(str).str.fullmatch(r'[A-Z0-9]{6,10}')]
    header = re.compile(r'(DS|eoAD)_(p|np)_(0[1-5])')
    parsed = {c: (m[1] + m[3], m[1], m[2]) for c in p if (m := header.fullmatch(c))}
    s = pd.DataFrame.from_dict(parsed, orient='index', columns=['case', 'group', 'tissue'])
    if len(s) != 20:
        raise ValueError('Expected 10 plaque-adjacent pairs')
    x = p.loc[:, s.index].apply(pd.to_numeric, errors='raise')
    delta, adjacent = within_donor(x, s, 'p', 'np')
    metadata = info.rename(columns={'Majority Gene ID': 'Genes'}).loc[p.index]
    labels = s.drop_duplicates('case').set_index('case').group
    return labels, metadata, delta, adjacent


def read_drummond(book, expected_accessions):
    labels, metadata, delta, _ = read_drummond_matrices(book)
    return single_gene_matrix(metadata, delta, expected_accessions), labels, metadata


def read_vessel_matrices(book, plaque_cases):
    p = pd.read_excel(book, sheet_name='Supplemental Table 3', header=1)
    meta = pd.read_excel(book, sheet_name='Supplemental Table 1', header=1)
    key = meta['Proteomics ID'].str.extract(r'((?:AD|MCI|Control)_R\d+)')[0]
    meta = meta.assign(author_key=key).dropna(subset=['author_key']).set_index('author_key')
    if not meta.index.is_unique:
        raise ValueError('Ambiguous CAA sample metadata')
    parsed = {c: (meta.at[f'{m[2]}_{m[3]}', 'Case ID'], m[2], m[1]) for c in p if (m := CAA_SAMPLE.fullmatch(c))}
    s = pd.DataFrame.from_dict(parsed, orient='index', columns=['case', 'group', 'tissue'])
    p = p.dropna(subset=['UniprotAcc']).set_index('UniprotAcc')
    p = p.loc[~p.index.duplicated(keep=False)]
    d, _ = within_donor(p.loc[:, s.index].apply(pd.to_numeric, errors='raise'), s, 'pos', 'neg')
    labels = s.drop_duplicates('case').set_index('case').group
    if labels.reindex(d.columns).value_counts().to_dict() != {'AD': 6, 'MCI': 4}:
        raise ValueError('Unexpected paired CAA dataset')
    relevant = plaque_cases.loc[plaque_cases.Group.isin(['AD', 'MCI'])]
    shared = relevant.index.intersection(d.columns)
    clinical = meta.loc[meta['Analysis performed'].str.contains(r'\bProteomics\b', na=False)].set_index('Case ID')
    if not clinical.index.is_unique:
        raise ValueError('CAA metadata contain duplicate case IDs')
    mismatch = relevant.loc[shared, LINK_FIELDS].ne(clinical.loc[shared, LINK_FIELDS]).sum()
    if mismatch.any():
        raise ValueError(f'Linked case identifiers disagree on published metadata: {mismatch.to_dict()}')
    return p[['Genes']], d, labels, shared


def read_vessel(book, expected_accessions, plaque_cases):
    genes, delta, labels, shared = read_vessel_matrices(book, plaque_cases)
    return single_gene_matrix(genes, delta, expected_accessions), labels, shared
