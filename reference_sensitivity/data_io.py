"""Frozen workbook loading and the reusable paired CSV interface."""

import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper4.four_part.study_tables import (
    DRUMMOND, LEITNER, MARTA, ORDER, checked_inputs, paired_group, read_study,
)


@dataclass
class Cohort:
    dataset: str
    group: str
    donors: list
    proteins: dict
    excluded: dict

    @property
    def key(self):
        return {"dataset": self.dataset, "group": self.group}

    def values(self, genes):
        return np.column_stack([self.proteins[g][1] for g in genes])

    def missing_reason(self, gene):
        return self.excluded.get(gene, "absent")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    if fields is None and rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError("CSV field names are required for an empty table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def frozen_cohorts(read):
    sources = checked_inputs(read)
    cohorts = []
    for study, filename in (("Leitner", LEITNER), ("Marta-Ariza", MARTA), ("Drummond 2022", DRUMMOND)):
        print("Reading " + filename, flush=True)
        _, labels, proteins = read_study(study, read("data/" + filename))
        for dataset, group, count in ORDER:
            if dataset != study:
                continue
            donors, _, single = paired_group(labels, proteins, group, count)
            excluded = {}
            for gene, _ in proteins.values():
                if not gene:
                    continue
                for token in re.findall(r"[A-Z][A-Z0-9-]+", gene):
                    if token not in single:
                        excluded[token] = "ambiguous_gene_or_protein_group"
            cohorts.append(Cohort(study, group, donors, single, excluded))
    return cohorts, sources


def read_paired_csv(path):
    """One row per study, group, donor and unambiguous gene, already on log2 scale."""
    opener = gzip.open if path.suffix == ".gz" else open
    required = {"dataset", "group", "donor", "gene", "accession", "plaque_log2", "adjacent_log2"}
    observations = defaultdict(dict)
    accessions = defaultdict(dict)
    donor_sets = defaultdict(set)
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Input CSV requires " + ", ".join(sorted(required)))
        for line, row in enumerate(reader, 2):
            identifiers = [row[name].strip() for name in ("dataset", "group", "donor", "gene", "accession")]
            if not all(identifiers):
                raise ValueError(f"Empty identifier on input line {line}")
            dataset, group, donor, gene, accession = identifiers
            if not re.fullmatch(r"[A-Z][A-Z0-9-]+", gene):
                raise ValueError(f"Ambiguous gene symbol on line {line}: {gene}")
            key = (dataset, group)
            if (donor, gene) in observations[key]:
                raise ValueError(f"Duplicate donor/gene measurement on line {line}")
            if gene in accessions[key] and accessions[key][gene] != accession:
                raise ValueError(f"Multiple accessions for {gene} in {key}")
            accessions[key][gene] = accession
            vals = []
            for field in ("plaque_log2", "adjacent_log2"):
                text = row[field].strip()
                value = np.nan if text.lower() in ("", "na", "nan") else float(text)
                if np.isinf(value):
                    raise ValueError(f"Infinite intensity on line {line}")
                vals.append(value)
            observations[key][donor, gene] = vals
            donor_sets[key].add(donor)
    if not observations:
        raise ValueError("The input table has no measurements")
    cohorts = []
    for (dataset, group), values in observations.items():
        key = (dataset, group)
        repeated = [acc for acc, count in Counter(accessions[key].values()).items() if count > 1]
        if repeated:
            raise ValueError(f"Accessions map to multiple genes in {key}: {repeated}")
        donors = sorted(donor_sets[key])
        proteins = {}
        for gene, accession in sorted(accessions[key].items()):
            mat = np.array([values.get((d, gene), [np.nan, np.nan]) for d in donors])
            proteins[gene] = (accession, mat[:, 0] - mat[:, 1], mat[:, 1])
        cohorts.append(Cohort(dataset, group, donors, proteins, {}))
    return cohorts


def export_paired_csv(path, cohorts):
    fields = ["dataset", "group", "donor", "gene", "accession", "plaque_log2", "adjacent_log2"]
    # A fixed gzip timestamp keeps identical exports byte-reproducible.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(fields)
            for cohort in cohorts:
                for gene, (accession, delta, adjacent) in sorted(cohort.proteins.items()):
                    for i, donor in enumerate(cohort.donors):
                        # The reusable interface records paired data. An unpaired plaque
                        # intensity cannot be recovered from a missing paired difference.
                        plaque = delta[i] + adjacent[i]
                        writer.writerow([cohort.dataset, cohort.group, donor, gene, accession,
                                         float(plaque) if np.isfinite(plaque) else "",
                                         float(adjacent[i]) if np.isfinite(adjacent[i]) else ""])


def validate_protocol(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    def genes(values, label):
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not re.fullmatch(r"[A-Z][A-Z0-9-]+", v) for v in values):
            raise ValueError("Invalid gene list: " + label)
        if len(values) != len(set(values)):
            raise ValueError("Repeated gene in " + label)
    def references(mapping, targets):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("At least one named reference is required")
        for name, values in mapping.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("Invalid reference name: " + name)
            genes(values, name)
            if set(values) & set(targets):
                raise ValueError("Target/reference overlap in " + name)
    genes(config["primary_targets"], "primary_targets")
    genes(config["combination_pool"], "combination_pool")
    if set(config["primary_targets"]) & set(config["combination_pool"]):
        raise ValueError("Combination pool overlaps targets")
    size = config["combination_size"]
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= len(config["combination_pool"]):
        raise ValueError("Invalid combination_size")
    if config["reference_mode"] == "fixed_lists":
        references(config["references"], config["primary_targets"])
    elif config["reference_mode"] != "frozen_annotation_complete":
        raise ValueError("Invalid reference_mode")
    secondary = config["secondary"]
    genes(secondary["targets"], "secondary.targets")
    references(secondary["references"], secondary["targets"])
    for label in ("name", "rationale", "selection", "interpretation"):
        if not isinstance(secondary.get(label), str) or not secondary[label].strip():
            raise ValueError("Provide secondary." + label + " in the protocol")

