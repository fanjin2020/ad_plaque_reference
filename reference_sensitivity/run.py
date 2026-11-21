"""Regenerate the original four-step reference sensitivity analyses."""
import json

import numpy as np

from paper4.reference_sensitivity.analysis import (
    compare_references, coverage_rows, panel_combinations, proteome_directions, resolve_references,
)
from paper4.reference_sensitivity.data_io import (
    export_paired_csv, frozen_cohorts, write_csv,
)


def regression_checks(read, cohorts, primary, combo_status, proteome_status, config):
    if config != json.loads(read("reference_sensitivity/protocol.json")):
        return {"status": "custom_protocol", "checks": []}
    expected = json.loads(read("reference_sensitivity/expected_regression.json"))
    checks = []
    for cohort in cohorts:
        key = cohort.dataset + "|" + cohort.group
        baseline = expected[key]
        if len(cohort.donors) != baseline["n"]:
            raise AssertionError("Frozen donor count differs: " + key)
        estimates = {r["reference"]: r["estimate"] for r in primary
                     if (r["dataset"], r["group"], r["scope"]) == (cohort.dataset, cohort.group, "common_donors")}
        if set(estimates) != set(baseline["reference_estimates"]):
            raise AssertionError("Frozen reference definitions differ: " + key)
        for name, value in baseline["reference_estimates"].items():
            np.testing.assert_allclose(estimates[name], value, atol=1e-9, rtol=0)
        background = next(r for r in proteome_status if (r["dataset"], r["group"]) == (cohort.dataset, cohort.group))
        if background["n_proteins"] != baseline["background"]:
            raise AssertionError("Unexpected proteome denominator: " + key)
        if "combination_min" in baseline:
            result = next(r for r in combo_status if (r["dataset"], r["group"]) == (cohort.dataset, cohort.group))
            np.testing.assert_allclose([result["point_min"], result["point_max"], result["n_common"]],
                                       [baseline["combination_min"], baseline["combination_max"], baseline["combination_n"]], atol=1e-9, rtol=0)
        checks.append({"cohort": key, "status": "passed"})
    return {"status": "passed", "checks": checks}

def execute(args, read, config, folder):
    cohorts, _ = frozen_cohorts(read)
    go_bytes = read("plaque_validation/uniprot_hydrolase_sets.json")
    ec_bytes = read("plaque_validation/uniprot_ec_snapshot.json")
    go, ec = json.loads(go_bytes), json.loads(ec_bytes)
    tables = folder / "tables"
    figures = folder / "figures"
    tables.mkdir()
    figures.mkdir()
    combo, combo_donors, combo_status, combo_coverage = [], [], [], []
    members, primary, primary_donors, primary_status, primary_contrasts, primary_coverage = [], [], [], [], [], []
    direction_summary, profiles, effects, direction_status = [], [], [], []
    second, second_donors, second_status, second_contrasts, second_coverage = [], [], [], [], []
    for cohort in cohorts:
        print(f"Analyzing {cohort.dataset} {cohort.group}, {len(cohort.donors)} donors", flush=True)
        combo_coverage.extend(coverage_rows(cohort, config["primary_targets"], {"combination_pool": config["combination_pool"]}))
        a, b, c, _ = panel_combinations(cohort, config, args.draws, args.seed)
        combo.extend(a)
        combo_donors.extend(b)
        combo_status.append(c)
        references, annotation = resolve_references(cohort, config, go, ec)
        members.extend(annotation)
        primary_coverage.extend(coverage_rows(cohort, config["primary_targets"], references))
        a, b, c, d = compare_references(cohort, config["primary_targets"], references, args.draws, args.seed, "primary")
        primary.extend(a)
        primary_donors.extend(b)
        primary_status.extend(c)
        primary_contrasts.extend(d)
        a, b, c, d = proteome_directions(cohort, config["primary_targets"], references, args.draws, args.seed)
        direction_summary.extend(a)
        profiles.extend(b)
        effects.extend(c)
        direction_status.append(d)
        secondary = config["secondary"]
        second_coverage.extend(coverage_rows(cohort, secondary["targets"], secondary["references"]))
        a, b, c, d = compare_references(cohort, secondary["targets"], secondary["references"], args.draws, args.seed, "secondary")
        second.extend(a)
        second_donors.extend(b)
        second_status.extend(c)
        second_contrasts.extend(d)
    output_tables = {
        "01_combination_estimates": combo, "01_combination_donors": combo_donors,
        "01_combination_status": combo_status, "01_combination_coverage": combo_coverage,
        "02_reference_estimates": primary, "02_reference_donors": primary_donors,
        "02_reference_status": primary_status, "02_reference_differences": primary_contrasts,
        "02_reference_coverage": primary_coverage, "02_annotation_members": members,
        "03_direction_summary": direction_summary, "03_protein_profiles": profiles,
        "03_protein_reference_estimates": effects, "03_direction_status": direction_status,
        "04_secondary_estimates": second, "04_secondary_donors": second_donors,
        "04_secondary_status": second_status, "04_secondary_reference_differences": second_contrasts,
        "04_secondary_coverage": second_coverage,
    }
    for name, rows in output_tables.items():
        write_csv(tables / (name + ".csv"), rows, None if rows else ["dataset", "group", "status"])
    regression_checks(read, cohorts, primary, combo_status, direction_status, config)
    export_paired_csv(folder / "paired_measurements.csv.gz", cohorts)
    from paper4.reference_sensitivity.figures import style, combination_figure, proteome_figure, secondary_figure
    style()
    combination_figure(combo, config["combination_pool"], figures)
    proteome_figure(direction_summary, figures)
    secondary_figure(second_coverage, second, config["secondary"], figures)
