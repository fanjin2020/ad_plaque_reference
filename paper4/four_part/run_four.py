"""Generate the tissue, coverage and conditional benchmark tables."""
import csv
import json

import numpy as np
from paper4.plaque_validation.panel_definition import ACCESSIONS, NUM
from paper4.four_part.inference import tissue_models
from paper4.four_part.references import PUMP, calibration, classes, coverage, reference_tables
from paper4.four_part.study_tables import (
    LEITNER,
    MARTA,
    DRUMMOND,
    ORDER,
    checked_inputs,
    paired_group,
    read_study,
)


def write_table(folder, name, rows):
    with (folder / name).open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def run(read, folder):
    sources = checked_inputs(read)
    files = ("uniprot_hydrolase_sets.json", "uniprot_ec_snapshot.json")
    annotation = [read("plaque_validation/" + f) for f in files]
    go, ec = map(json.loads, annotation)
    sets = classes(go, ec)
    mc = np.random.default_rng(20261117)
    boot = np.random.default_rng(20261119)
    tissues = []
    members = []
    estimates = []
    benchmarks = []
    thresholds = []
    controls = []
    for study, file in (
        ("Leitner", LEITNER),
        ("Marta-Ariza", MARTA),
        ("Drummond 2022", DRUMMOND),
    ):
        print("Reading", file, flush=True)
        meta, labels, proteins = read_study(study, read("data/" + file))
        if study == "Leitner":
            for g in NUM:
                if proteins[ACCESSIONS[g]][0] != g:
                    raise ValueError("Unverified tissue model target " + g)
            observations, tissues = tissue_models(meta, labels, proteins, ACCESSIONS)
            write_table(folder, "01_donor_tissues.csv", observations)
            write_table(folder, "01_tissue_contrasts.csv", tissues)
        for dataset, group, n in ORDER:
            if dataset != study:
                continue
            print("Estimating", dataset, group, flush=True)
            cases, paired, single = paired_group(labels, proteins, group, n)
            targets = np.mean([single[g][1] for g in NUM], axis=0)
            key = dict(dataset=dataset, group=group)
            m, e, s = reference_tables(paired, sets, targets, key)
            members.extend(m)
            estimates.extend(e)
            bench = calibration(single, key, mc)
            benchmarks.extend(bench)
            pump = next(r for r in e if r["reference"] == PUMP)
            np.testing.assert_allclose(
                [r["observed"] for r in bench], pump["estimate"], atol=1e-11
            )
            if study != "Drummond 2022":
                rows, genes = coverage(s, targets, key, boot)
                thresholds.extend(rows)
                controls.extend(genes)
    for name, rows in (
        ("02_coverage.csv", thresholds),
        ("02_control_effects.csv", controls),
        ("03_reference_members.csv", members),
        ("03_reference_estimates.csv", estimates),
        ("03_panel_benchmark.csv", benchmarks),
    ):
        write_table(folder, name, rows)
    pump = [row for row in estimates if row["reference"] == PUMP]
    if len(pump) != 4:
        raise ValueError("Expected four pump estimates")
    write_table(folder, "04_concordance.csv", pump)
