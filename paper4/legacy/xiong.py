"""Paper 4: one fixed tissue contrast, different inference for group vs donor data."""
import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics as st
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import load_workbook
from scipy import stats

XIONG = '911ae6c1fd61db37253d2937d4d21975534466daccefc353dc85daaa7e43d8f6'
MARTA = 'da46728a666766b66fc05ba1e7bc3aded5bcbb6e17683bf7856df4f61762385d'
SOURCE = 'https://ars.els-cdn.com/content/image/1-s2.0-S1552526018335805-mmc2.xlsx'
TARGETS = {'CTSD': 'P07339', 'TPP1': 'O14773'}
SIGNALS = ['ctsd_plaque_log2','ctsd_adjacent_log2','tpp1_plaque_log2','tpp1_adjacent_log2']
FIELDS = ['study_id','bank','donor_id','onset_group',*SIGNALS]
FORMULA = '(CTSD_plaque - CTSD_adjacent) - (TPP1_plaque - TPP1_adjacent), log2 signals'

def digest(p):
    with p.open('rb') as f:
        h = hashlib.sha256()
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
        return h.hexdigest()

def check(flag, message):
    if not flag: raise ValueError(message)

def savecsv(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

def jdump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')

def pick(sheet, header_row, id_col, gene_col):
    it = sheet.iter_rows(min_row=header_row, values_only=True); names = next(it); idx = names.index(id_col)
    records = {}
    for row in it:
        if row[idx] in TARGETS.values():
            g = next(g for g, a in TARGETS.items() if a == row[idx]); r = dict(zip(names, row))
            check(g not in records and (r[gene_col] == g or ('GN='+g) in str(r[gene_col]).split()), 'Duplicate or ambiguous target '+g)
            records[g] = r
    check(set(records)==set(TARGETS),'Missing one or both fixed targets'); return records

def xiong(book, out):
    check(digest(book)==XIONG, 'Xiong Supplementary Table 1 SHA-256 mismatch')
    w=load_workbook(book,read_only=True,data_only=True)
    try:
        records=pick(w['Human_Pro_All'],1,'Accession','Description')
        ids=[r[2] for r in w['Patient_information'].values if isinstance(r[2],str) and re.fullmatch(r'(?:AD|nonAD)\d+',r[2])]
        check(Counter(ids)==Counter(['AD1','AD2','AD3','nonAD1','nonAD2','nonAD3']), 'Unexpected donor metadata')
    finally: w.close()
    details=[]; effects=[]
    for group in ['AD','nonAD']:
        e={}
        for g,r in records.items():
            p=float(r['Abundances (Normalized):'+group+'_SPR']); a=float(r['Abundances (Normalized):'+group+'_NPR'])
            check(all(math.isfinite(x) and x>0 for x in (p,a)), 'Missing/invalid intensity')
            ratio=p/a; published=float(r[f'Abundance Ratio: {group}_SPR / {group}_NPR']); check(abs(ratio-published)<=0.0006, 'Reported ratio differs from group intensities')
            e[g]=math.log2(ratio); details.append(dict(group=group,gene=g,accession=TARGETS[g],unique_peptides_in_table=r['# Unique Peptides'],plaque_intensity=p,nonplaque_intensity=a,ratio_from_intensities=ratio,published_rounded_ratio=published,enrichment_log2=e[g]))
        D=e['CTSD']-e['TPP1']; effects.append(dict(group=group,metadata_donor_count=3,available_group_contrasts=1,donor_level_pairs_available=0,D_log2=D,relative_enrichment_ratio=2**D,inference='descriptive_group_only'))
    savecsv(out/'group_protein_values.csv',details); savecsv(out/'group_effects.csv',effects)
    return ('Xiong: descriptive group-level check', [f"{r['group']}: D={r['D_log2']:+.5f} log2; relative fold={r['relative_enrichment_ratio']:.5f}." for r in effects]+['No donor-resolved human quantification in this table. No CI, p-value, onset comparison or pooling with discovery.'])
