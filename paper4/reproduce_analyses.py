"""Run the published calculations on the checksum-verified workbooks."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd

from paper4.legacy.xiong import xiong
from paper4.four_part.run_four import run as four_part
from paper4.legacy.locked_panels import run as locked_swaps
from paper4.legacy.reference_audit import derive as reference_audit
from paper4.plaque_validation.cli import write_outputs
from paper4.reference_extensions.application import analyze as fixed_panel_analysis
from paper4.reference_extensions.io import read_paired, validate_protocol, write_csv
from paper4.reference_extensions.simulation import build_models, simulate
from paper4.reference_extensions.validation import audit
from paper4.reference_strengthening.run import execute as strengthening
from paper4.review_followup.analysis import run_analysis as followup
from paper4.review_followup.validation import verify_tables as verify_followup
from paper4.legacy.enzyme_pair import read_donors, cohort_estimates, source_analysis
from paper4.legacy.ec_annotations import enzyme_classes, omit_controls
from paper4.legacy.hydrolases import caa_checks, plaque_checks, plaque_data, shared_background
from paper4.legacy.abundance_matching import compare_group, plaque_groups

import numpy as np


ROOT = Path(__file__).resolve().parent
DRAWS = 10_000
REPEATS = 2_000
SEED = 20261119
LEGACY_NAMES = {
    "main_results.csv": "01_fixed_panel.csv",
    "caa_paired_results.csv": "08_linked_CAA.csv",
    "caa_grades_exploratory.csv": "16_grade_exploratory.csv",
    "caa_paired_donors.csv": "paired_vascular_donors.csv",
}
EC_NAMES = {
    "02_target_effects.csv": "02_components.csv",
    "04_class_comparisons.csv": "03_reference_sets.csv",
    "03_annotation_coverage.csv": "04_reference_membership.csv",
    "07_shared_background.csv": "05_shared_background.csv",
    "05_abundance_matched_pairs.csv": "06_abundance_matching.csv",
    "09_caa_same_donor.csv": "09_CAA_references.csv",
    "10_ec_membership.csv": "13_EC_membership.csv",
    "11_leave_one_control_out.csv": "14_leave_one_control_out.csv",
    "08_donor_robustness.csv": "15_donor_robustness.csv",
}
FOUR_PART_NAMES = {
    "01_donor_tissues.csv": "24_tissue_donors.csv",
    "01_tissue_contrasts.csv": "25_tissue_models.csv",
    "02_coverage.csv": "26_coverage_thresholds.csv",
    "02_control_effects.csv": "27_threshold_members.csv",
    "03_reference_members.csv": "28_class_members.csv",
    "03_reference_estimates.csv": "29_class_effects.csv",
    "03_panel_benchmark.csv": "30_random_benchmarks.csv",
    "04_concordance.csv": "31_concordance.csv",
}


def read_source(name, input_dir):
    if name.startswith("data/"):
        return (input_dir / Path(name).name).read_bytes()
    return (ROOT / name).read_bytes()


def enzymes(data, folder, work):
    donors = read_donors(data)
    results = cohort_estimates(donors).rename(columns={
        "n_paired": "n", "n_positive": "positive", "estimate_log2": "mean_D",
        "ci95_low": "ci_low", "ci95_high": "ci_high",
    })
    results[["dataset", "group", "n", "positive", "mean_D", "ci_low", "ci_high"]].to_csv(
        folder / "10_enzyme_difference.csv", index=False)
    _, onset = source_analysis(donors)
    onset = onset.rename(columns={
        "early_minus_late_log2": "difference", "ci95_low": "ci_low", "ci95_high": "ci_high",
    })
    cols = ["scope", "endpoint", "n_early", "n_late", "difference",
            "ci_low", "ci_high", "welch_p", "exact_label_p"]
    onset[cols].to_csv(folder / "11_onset.csv", index=False)
    xiong(data / "xiong_2019_table1.xlsx", work)
    proteins = pd.read_csv(work / "group_protein_values.csv", float_precision="round_trip")
    fold = proteins.pivot(index="group", columns="gene", values="ratio_from_intensities")
    raw = pd.read_csv(work / "group_effects.csv").rename(
        columns={"relative_enrichment_ratio": "ratio_of_folds"})
    raw["ctsd_fold"] = raw["group"].map(fold["CTSD"])
    raw["tpp1_fold"] = raw["group"].map(fold["TPP1"])
    raw["inference"] = "group-level descriptive only; no CI or p"
    raw[["group", "ctsd_fold", "tpp1_fold", "ratio_of_folds", "D_log2", "inference"]].to_csv(
        folder / "12_Xiong.csv", index=False)


def legacy_tables(read, data, legacy, work, go, ec):
    scratch = work / "historical_calculations"
    scratch.mkdir()
    write_outputs(scratch, data)
    for original, name in LEGACY_NAMES.items():
        (scratch / original).replace(legacy / name)

    classes, annotation = enzyme_classes(go, ec)
    cohorts = plaque_data(data)
    panels, outputs = plaque_checks(cohorts, classes)
    outputs["07_shared_background.csv"] = shared_background(panels, classes)
    outputs["09_caa_same_donor.csv"] = caa_checks(data, cohorts["Leitner"], classes)
    outputs["10_ec_membership.csv"] = annotation
    outputs["11_leave_one_control_out.csv"] = list(omit_controls(panels, classes))
    for original, name in EC_NAMES.items():
        pd.DataFrame(outputs[original]).to_csv(legacy / name, index=False)

    rng = np.random.default_rng(20261117)
    matching = []
    for dataset, group, genes, regional_changes, adjacent in plaque_groups(data):
        rows, _ = compare_group(
            genes, regional_changes, adjacent, dict(dataset=dataset, group=group), DRAWS, rng)
        matching.extend(rows)
    pd.DataFrame(matching).to_csv(legacy / "07_general_proteome_matching.csv", index=False)

    enzymes(data, legacy, scratch)
    reference_audit(legacy)
    locked_swaps(ROOT, legacy, data)
    four_part(read, scratch)
    for original, name in FOUR_PART_NAMES.items():
        (scratch / original).replace(legacy / name)


def run(work, data, workers):
    def read(name):
        return read_source(name, data)

    observed_protocol = json.loads(read("reference_sensitivity/protocol.json"))
    fixed_protocol = validate_protocol(json.loads(read("reference_extensions/data/protocol.json")))
    go = json.loads(read("plaque_validation/uniprot_hydrolase_sets.json"))
    ec = json.loads(read("plaque_validation/uniprot_ec_snapshot.json"))

    legacy = work / "legacy" / "tables"
    legacy.mkdir(parents=True)
    legacy_tables(read, data, legacy, work, go, ec)

    enhancement = work / "enhancement"
    enhancement.mkdir()
    args = Namespace(draws=DRAWS, sim_reps=REPEATS, seed=SEED)
    strengthening(args, read, enhancement)
    paired_path = enhancement / "baseline" / "paired_measurements.csv.gz"

    cohorts, _ = read_paired(paired_path.read_bytes())
    tables = fixed_panel_analysis(cohorts, fixed_protocol, DRAWS, SEED)
    models, diagnostics = build_models(cohorts, fixed_protocol)
    tables.update(diagnostics)
    tables.update(simulate(models, fixed_protocol, REPEATS, SEED, workers))
    empirical_check = audit(cohorts, fixed_protocol, tables)
    extensions = work / "three_extensions" / "tables"
    for name, records in tables.items():
        write_csv(extensions / (name + ".csv"), records, fields=None if records else ["status", "reason"])

    _, s1_tables = followup(read, fixed_protocol, (observed_protocol, go, ec))
    checks_inputs = {
        52: enhancement / "tables" / "01_margin_curves.csv",
        57: enhancement / "tables" / "02_reference_members.csv",
        59: enhancement / "tables" / "02_target_effects.csv",
        66: extensions / "01_coverage.csv",
        67: extensions / "02_family_availability.csv",
        68: extensions / "03_shared_donor_estimates.csv",
        69: extensions / "04_donor_scores.csv",
    }
    followup_check = verify_followup(lambda name: checks_inputs[int(Path(name).stem[1:])].read_bytes(), s1_tables)
    return SimpleNamespace(
        work=work, legacy=legacy, enhancement=enhancement, extensions=extensions,
        paired=paired_path, s1=s1_tables, ec=ec, empirical_check=empirical_check,
        followup_check=followup_check,
    )
