"""Validated paired inputs; no normalization, imputation, or silent aggregation."""
import csv
import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(row) for row in rows]
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError("Empty output needs an explicit schema: " + str(path))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rng_for(seed, *keys):
    token = json.dumps([int(seed), *keys], sort_keys=True).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(token).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


@dataclass
class Cohort:
    dataset: str
    group: str
    donors: list
    genes: list
    accessions: dict
    delta: np.ndarray
    adjacent: np.ndarray

    @property
    def key(self):
        return {"dataset": self.dataset, "group": self.group}

    def matrix(self, genes, arm="delta"):
        index = {g: i for i, g in enumerate(self.genes)}
        source = self.delta if arm == "delta" else self.adjacent
        result = np.full((len(self.donors), len(genes)), np.nan)
        for j, gene in enumerate(genes):
            if gene in index:
                result[:, j] = source[:, index[gene]]
        return result


def read_paired(raw):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    required = {"dataset", "group", "donor", "gene", "accession",
                "plaque_log2", "adjacent_log2"}
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not required <= set(reader.fieldnames or []):
        raise ValueError("Input requires columns: " + ", ".join(sorted(required)))
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Duplicate input column names")
    groups = {}
    count = 0
    for line, row in enumerate(reader, 2):
        ids = [str(row.get(k) or "").strip() for k in
               ["dataset", "group", "donor", "gene", "accession"]]
        if not all(ids):
            raise ValueError(f"Empty identifier on line {line}")
        dataset, group, donor, gene, accession = ids
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", gene):
            raise ValueError(f"Ambiguous gene symbol on line {line}: {gene}")
        state = groups.setdefault((dataset, group), {"obs": {}, "acc": {}, "donors": set()})
        if (donor, gene) in state["obs"]:
            raise ValueError(f"Duplicate donor/gene row on line {line}")
        if gene in state["acc"] and state["acc"][gene] != accession:
            raise ValueError(f"Conflicting accessions for {gene} in {dataset}/{group}")
        values = []
        for key in ["plaque_log2", "adjacent_log2"]:
            value = str(row.get(key) or "").strip()
            try:
                number = np.nan if value.lower() in ("", "na", "nan", "null") else float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid number on line {line}, {key}") from exc
            if np.isinf(number):
                raise ValueError(f"Infinite intensity on line {line}")
            values.append(number)
        state["obs"][donor, gene] = values
        state["acc"][gene] = accession
        state["donors"].add(donor)
        count += 1
    if not count:
        raise ValueError("Input contains no observations")
    cohorts = []
    for (dataset, group), state in groups.items():
        if len(set(state["acc"].values())) != len(state["acc"]):
            raise ValueError(f"One accession assigned to multiple genes in {dataset}/{group}")
        genes, donors = sorted(state["acc"]), sorted(state["donors"])
        gi, di = {g: i for i, g in enumerate(genes)}, {d: i for i, d in enumerate(donors)}
        plaque = np.full((len(donors), len(genes)), np.nan)
        adjacent = np.full_like(plaque, np.nan)
        for (donor, gene), (p, a) in state["obs"].items():
            plaque[di[donor], gi[gene]], adjacent[di[donor], gi[gene]] = p, a
        cohorts.append(Cohort(dataset, group, donors, genes, state["acc"], plaque-adjacent, adjacent))
    return cohorts, {"rows": count, "groups": len(cohorts), "paired_study_records":
                     sum(len(c.donors) for c in cohorts),
                     "unit_note": "Study-coded donor records; cross-study identity is not resolved. No pooling."}


def validate_protocol(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema_version")
    if not isinstance(config.get("applications"), list) or not config["applications"]:
        raise ValueError("At least one application is required")
    names = set()
    def gene_list(values, label):
        if not isinstance(values, list) or not values or any(
                not isinstance(g, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", g) for g in values):
            raise ValueError("Invalid gene list: " + label)
        if len(set(values)) != len(values):
            raise ValueError("Repeated genes: " + label)
    for app in config["applications"]:
        name = app.get("name", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("Invalid or repeated application name")
        names.add(name)
        gene_list(app.get("targets"), name)
        refs = app.get("references")
        if not isinstance(refs, dict) or len(refs) < 2:
            raise ValueError("A reference-sensitivity application requires at least two references")
        lists = set()
        for label, value in refs.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
                raise ValueError("Invalid reference identifier: " + label)
            gene_list(value, label)
            if set(value) & set(app["targets"]):
                raise ValueError("Target/reference overlap: " + label)
            canonical = tuple(sorted(value))
            if canonical in lists:
                raise ValueError("Identical reference lists must not be counted as different tests")
            lists.add(canonical)
        if app.get("primary_reference") not in refs:
            raise ValueError("primary_reference must name a planned reference")
        for key in ["question", "selection_disclosure", "interpretation"]:
            if not isinstance(app.get(key), str) or not app[key].strip():
                raise ValueError("Missing application rationale: " + key)
    sim = config["simulation"]
    for label, value, minimum in [("minimum_donors", config["minimum_donors"], 3),
                                  ("calibration_minimum_donors", sim["calibration_minimum_donors"], 10),
                                  ("minimum_complete_donors", sim["minimum_complete_donors"], 3)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("Invalid " + label)
    if not 0 < config["alpha"] < 1:
        raise ValueError("alpha must be between zero and one")
    if not config["margins_log2"] or any(not np.isfinite(x) or x < 0 for x in config["margins_log2"]):
        raise ValueError("Margins must be finite and nonnegative")
    if not sim["sample_sizes"] or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in sim["sample_sizes"]):
        raise ValueError("Simulation sample sizes must be integers >= 3")
    if len(set(sim["sample_sizes"])) != len(sim["sample_sizes"]):
        raise ValueError("Repeated simulation sample sizes")
    allowed = {"gaussian_empirical", "gaussian_shrinkage", "donor_resampling"}
    if not sim["generators"] or not set(sim["generators"]) <= allowed:
        raise ValueError("Unknown generator")
    if len(set(sim["generators"])) != len(sim["generators"]):
        raise ValueError("Repeated simulation generators")
    if not 0 <= sim["diagonal_shrinkage"] <= 1:
        raise ValueError("Invalid shrinkage coefficient")
    if sim["calibration_draws"] < 1000 or sim["mnar_slope"] <= 0:
        raise ValueError("Invalid MNAR calibration settings")
    for mechanism, rate in sim["missingness"]:
        if mechanism not in ("complete", "MCAR", "MNAR") or not 0 <= rate < 1:
            raise ValueError("Invalid missingness setting")
        if (mechanism == "complete") != (rate == 0):
            raise ValueError("Only complete data may specify zero missingness")
    if not sim["missingness"] or len({tuple(x) for x in sim["missingness"]}) != len(sim["missingness"]):
        raise ValueError("Missing or repeated missingness settings")
    return config


def application_matrix(app):
    refs = [app["primary_reference"]] + [r for r in app["references"] if r != app["primary_reference"]]
    genes = list(dict.fromkeys(app["targets"] + [g for r in refs for g in app["references"][r]]))
    gi = {g: i for i, g in enumerate(genes)}
    coefficients = np.zeros((len(genes), len(refs)))
    for j, label in enumerate(refs):
        for g in app["targets"]:
            coefficients[gi[g], j] = 1/len(app["targets"])
        for g in app["references"][label]:
            coefficients[gi[g], j] = -1/len(app["references"][label])
    return genes, refs, coefficients
