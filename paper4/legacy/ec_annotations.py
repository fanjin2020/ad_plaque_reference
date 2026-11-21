"""Check whether plaque enrichment persists against functionally defined hydrolases."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paper4.plaque_validation.analysis import ACCESSIONS, NUM, exact_inference
from paper4.plaque_validation.inputs import verify_workbooks
from paper4.legacy.hydrolases import (
    ANNOTATIONS, BOOTSTRAPS, CALIPERS, PRIMARY, ROOT, SEED, annotation_tables, caa_checks,
    plaque_checks, plaque_data, sha256, shared_background,
)


EC_SNAPSHOT = ROOT / 'plaque_validation' / 'uniprot_ec_snapshot.json'
EC_LUMEN = 'ec3_lysosomal_lumen'
EC_LYSOSOME = 'ec3_lysosome'
EC_PEPTIDASE = 'ec34_lysosome'
COMPLETE_EC3 = re.compile(r'3\.\d+\.\d+\.\d+')


def enzyme_classes(go, snapshot):
    if snapshot['release'] != go['release']:
        raise ValueError('GO membership and EC numbers must come from the same UniProt release')
    original = annotation_tables(go)
    ec = snapshot['ec_by_accession']
    if set(ec) != set().union(*(set(t.index) for t in original.values())):
        raise ValueError('EC accession inventory differs from the frozen GO snapshot')
    classes = {
        PRIMARY: original[PRIMARY],
        EC_LUMEN: original[PRIMARY],
        EC_LYSOSOME: original['lysosome_hydrolases'],
        EC_PEPTIDASE: original['lysosome_hydrolases'],
    }
    decisions = []
    for label, members in classes.items():
        applicable = {
            accession: [code for code in ec[accession]
                        if COMPLETE_EC3.fullmatch(code)
                        and not code.startswith('3.6.')
                        and (label != EC_PEPTIDASE or code.startswith('3.4.'))]
            for accession in members.index
        }
        keep = [acc for acc, codes in applicable.items() if label == PRIMARY or codes]
        classes[label] = members.loc[keep]
        if not set(ACCESSIONS[g] for g in NUM).issubset(keep):
            raise ValueError(f'Target enzyme missing from {label}')
        decisions.extend({'reference_class': label, 'accession': accession, 'gene': gene,
                          'all_ec': '|'.join(ec[accession]), 'qualifying_ec': '|'.join(applicable[accession]),
                          'included_by_annotation_rule': accession in keep,
                          'target': gene in NUM}
                         for accession, gene in members.primary_gene.items())
    return classes, decisions


def omit_controls(panels, classes):
    for (dataset, group), (targets, matrices) in panels.items():
        for label, matrix in matrices.items():
            if len(matrix) < 3:
                continue
            hydro = targets[NUM].mean(axis=1)
            for accession in matrix.index:
                score = hydro - matrix.drop(index=accession).mean(axis=0)
                yield {'dataset': dataset, 'group': group, 'reference_class': label,
                       'excluded_gene': classes[label].at[accession, 'primary_gene'],
                       'n_remaining_controls': len(matrix) - 1, **exact_inference(score)}
