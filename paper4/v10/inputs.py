"""Verify published input versions and align the tangle and neuropil columns."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
from openpyxl import load_workbook

from paper4.four_part.study_tables import DRUMMOND, LEITNER, MARTA
from paper4.reference_extensions.io import Cohort, application_matrix, validate_protocol
from paper4.review_followup.analysis import read_groups
from paper4.source_inputs import STUDY_DOWNLOAD_LIMIT, check_file, download


PAPER4 = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("protocol.json")
DATASET = "Thierry 2026"
SAMPLE_COLUMN = re.compile(r"QuantityNorm (Tangles|Non-tangles)_(PART|INTER|ADV)_Case#(\d+)")


def load_inputs(work, input_dir):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    reference = validate_protocol(json.loads(
        (PAPER4 / "reference_extensions/data/protocol.json").read_text(encoding="utf-8")))
    original_manifest = json.loads((PAPER4 / "data_manifest.json").read_text(encoding="utf-8"))
    records = [row for row in original_manifest if row["name"] in {DRUMMOND, LEITNER, MARTA}]
    if len(records) != 3:
        raise ValueError("Three original plaque workbooks are required")
    records.append(protocol["external_source"])
    if sum(row["bytes"] for row in records) > STUDY_DOWNLOAD_LIMIT:
        raise ValueError("The four public inputs exceed the study download limit")

    if input_dir is None:
        input_dir = work / "inputs"
        input_dir.mkdir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda row: download(input_dir, row), records))
    for row in records:
        check_file(input_dir / row["name"], row)

    def read(name):
        if name.startswith("data/"):
            return (input_dir / Path(name).name).read_bytes()
        return (PAPER4 / name).read_bytes()

    _, original, _, _ = read_groups(read)
    original_accessions = original[("Leitner", "AD")].accessions
    external = protocol["external_application"]
    accessions = {
        gene: original_accessions[gene]
        for app in reference["applications"] for gene in application_matrix(app)[0]
    }
    accessions.update(protocol["external_accessions"])
    if set(application_matrix(external)[0]) - accessions.keys():
        raise ValueError("The external target and all comparators require fixed accessions")

    tangle, mapping, measurements = read_tangles(
        input_dir / protocol["external_source"]["name"], protocol["external_groups"], accessions)
    return protocol, reference, list(original.values()), tangle, mapping, measurements, records


def intensity(value):
    if value is None or str(value).strip().lower() in {"", "nan", "na", "#n/a"}:
        return np.nan
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Unexpected nonfinite intensity {value!r}")
    return number


def read_tangles(path, group_plan, accessions):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book["Online Resource 1"]
        rows = sheet.iter_rows(min_row=6, values_only=True)
        headings = next(rows)
        if list(headings[:2]) != ["Gene ID", "Protein ID"]:
            raise ValueError("The Thierry workbook metadata columns changed")
        pairs = defaultdict(dict)
        for j, heading in enumerate(headings):
            match = SAMPLE_COLUMN.fullmatch(" ".join(str(heading or "").split()))
            if match is None:
                if "QuantityNorm" in str(heading or ""):
                    raise ValueError(f"Unrecognized Thierry intensity column {heading!r}")
                continue
            arm, group, donor = match.groups()
            if arm in pairs[group, donor]:
                raise ValueError("A Thierry donor arm appears twice")
            pairs[group, donor][arm] = j
        if any(set(arms) != {"Tangles", "Non-tangles"} for arms in pairs.values()):
            raise ValueError("Unmatched tangle or neuropil sample")
        if len(pairs) != sum(entry["donors"] for entry in group_plan.values()):
            raise ValueError("Unexpected number of Thierry donor pairs")
        if len({donor for _, donor in pairs}) != len(pairs):
            raise ValueError("A Thierry donor appears in more than one group")

        seen = defaultdict(list)
        selected = {}
        for row in rows:
            gene, accession = (str(value or "").strip() for value in row[:2])
            if gene not in accessions:
                continue
            seen[gene].append(accession)
            if accession == accessions[gene]:
                if gene in selected:
                    raise ValueError(f"More than one row matches the fixed accession for {gene}")
                selected[gene] = row
    finally:
        book.close()

    genes = sorted(selected)
    mapping = [
        dict(gene=gene, required_accession=accessions[gene],
             included_accession=accessions[gene] if gene in selected else "",
             workbook_accessions="|".join(seen[gene]), available=gene in selected)
        for gene in sorted(accessions)
    ]
    cohorts, measurements = [], []
    for code, description in group_plan.items():
        donors = sorted(donor for group, donor in pairs if group == code)
        if len(donors) != description["donors"]:
            raise ValueError(f"Unexpected external coverage in {code}")
        raw = {
            arm: np.array([[intensity(selected[g][pairs[code, donor][arm]]) for g in genes]
                           for donor in donors])
            for arm in ("Tangles", "Non-tangles")
        }
        tangle, adjacent = raw["Tangles"], raw["Non-tangles"]
        ids = [f"Case{donor}" for donor in donors]
        group = description["group"]
        cohorts.append(Cohort(DATASET, group, ids, genes,
                              {g: accessions[g] for g in genes}, tangle-adjacent, adjacent))
        measurements.extend(
            dict(dataset=DATASET, group=group, donor=donor, gene=gene, accession=accessions[gene],
                 tangle_log2=tangle[i, j], neuropil_log2=adjacent[i, j])
            for i, donor in enumerate(ids) for j, gene in enumerate(genes))
    return cohorts, mapping, measurements
