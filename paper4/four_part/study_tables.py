"""Source tables for the three frozen human LCM studies."""

import collections
import hashlib
import io
import json
import posixpath
import re
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from paper4.priorities.workbook import NS, REL, SAMPLE, worksheet_rows

LEITNER = "leitner_2026_S5.xlsx"
MARTA = "marta_2025_S10.xlsx"
DRUMMOND = "drummond2022_plaque_supp.xlsx"
ORDER = (
    ("Leitner", "AD", 53),
    ("Marta-Ariza", "EOAD", 20),
    ("Marta-Ariza", "LOAD", 20),
    ("Drummond 2022", "eoAD", 5),
)


class Sheets:
    def __init__(self, archive):
        self.book = archive
        self.strings = [
            "".join(t.text or "" for t in si.iter(NS + "t"))
            for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        rels = {
            r.get("Id"): posixpath.normpath(
                posixpath.join("xl", r.get("Target"))
            ).lstrip("/")
            for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        self.paths = {
            s.get("name"): rels[s.get(REL + "id")]
            for s in ET.fromstring(archive.read("xl/workbook.xml")).iter(NS + "sheet")
        }

    def records(self, sheet, header=2):
        cols = None
        for n, row in worksheet_rows(self.book, self.paths[sheet], self.strings):
            if n == header:
                cols = row
                yield list(cols.values())
            elif n > header and cols is not None:
                yield {cols[k]: v for k, v in row.items() if k in cols}


def source_labels(study, header):
    labels = {}
    for c in header:
        if study == "Leitner":
            m = SAMPLE.fullmatch(c)
            if m:
                g, arm, case = m.groups()
        elif study == "Marta-Ariza":
            m = re.fullmatch(r"((DS|EOAD|LOAD|Control)_\d+) (Plaque|Non-plaque)", c)
            if m:
                case, g, tissue = m.groups()
                arm = "p" if tissue == "Plaque" else "np"
        else:
            m = re.fullmatch(r"(DS|eoAD)_(p|np)_(0[1-5])", c)
            if m:
                g, arm, number = m.groups()
                case = g + number
        if m:
            if (g, case, arm) in labels:
                raise ValueError(f"Duplicate donor/tissue: {study} {case} {arm}")
            labels[g, case, arm] = c
    return labels


def read_study(study, raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        ws = Sheets(z)
        meta = {}
        names = {}
        if study == "Drummond 2022":
            records = ws.records("Table 1 - Imputed Data", 5)
            next(records)
            for row in records:
                a = row.get("Uniprot ID")
                names[a] = None if a in names else row.get("Majority Gene ID")
            tab, header, idcol, genecol = (
                "Table 2 - Unimputed Data",
                5,
                "Uniprot ID",
                None,
            )
        else:
            records = ws.records("Supplementary Table 1")
            next(records)
            for row in records:
                if study == "Leitner" and not re.search(
                    r"\bProteomics\b", row.get("Analysis Performed", "")
                ):
                    continue
                if row.get("Group") not in {
                    "AD",
                    "MCI",
                    "Control",
                    "DS",
                    "EOAD",
                    "LOAD",
                }:
                    continue
                case = row["Case ID"]
                if case in meta:
                    raise ValueError("Duplicate clinical donor: " + case)
                meta[case] = row
            tab, header, idcol, genecol = (
                ("Supplementary Table 2", 2, "ProteinGroupRepresentative", "Genes")
                if study == "Leitner"
                else ("Supplementary Table 3", 2, "Uniprot ID", "Gene ID")
            )
        records = ws.records(tab, header)
        labels = source_labels(study, next(records))
        if meta and (
            set(c for _, c, _ in labels) != set(meta)
            or any(meta[c]["Group"] != g for g, c, t in labels)
        ):
            raise ValueError("Clinical donor mismatch: " + study)
        expected_headers = {"Leitner": 240, "Marta-Ariza": 140, "Drummond 2022": 20}
        if len(labels) != expected_headers[study]:
            raise ValueError("Unexpected number of sample columns: " + study)
        cols = list(labels.values())
        proteins = {}
        for row in records:
            a = row.get(idcol)
            if not a or (
                study == "Drummond 2022" and not re.fullmatch(r"[A-Z0-9]{6,10}", a)
            ):
                continue
            if a in proteins:
                proteins[a] = None
                continue
            gene = row.get(genecol) if genecol else names.get(a)
            # The unimputed Drummond table uses literal NA as well as empty cells.
            raw_values = (row.get(c) for c in cols)
            vals = np.array(
                [
                    np.nan if v in (None, "", "NA", "NaN") else float(v)
                    for v in raw_values
                ]
            )
            if np.isinf(vals).any():
                raise ValueError("Infinite quantity at " + a)
            proteins[a] = (gene, vals)
        proteins = {a: v for a, v in proteins.items() if v is not None}
    return meta, {k: i for i, k in enumerate(labels)}, proteins


def paired_group(labels, proteins, group, n):
    cases = sorted(c for g, c, t in labels if g == group and t == "p")
    if len(cases) != n or set(cases) != {
        c for g, c, t in labels if g == group and t == "np"
    }:
        raise ValueError(f"Invalid {group} tissue pairs")
    p = [labels[group, c, "p"] for c in cases]
    a = [labels[group, c, "np"] for c in cases]
    freq = collections.Counter(g for g, x in proteins.values() if g)
    paired = {}
    single = {}
    for acc, (gene, values) in proteins.items():
        d = values[p] - values[a]
        paired[acc] = (gene, d)
        if gene and freq[gene] == 1 and re.fullmatch(r"[A-Z][A-Z0-9-]+", gene):
            single[gene] = (acc, d, values[a])
    return cases, paired, single


def checked_inputs(read):
    manifest = json.loads(read("data_manifest.json"))
    sources = {}
    for rec in manifest:
        name = rec["name"]
        if name not in (LEITNER, MARTA, DRUMMOND):
            continue
        raw = read("data/" + name)
        if len(raw) != rec["bytes"] or hashlib.sha256(raw).hexdigest() != rec["sha256"]:
            raise ValueError("Workbook failed source checksum: " + name)
        sources[name] = rec
    if len(sources) != 3:
        raise ValueError("Three workbooks required")
    return sources
