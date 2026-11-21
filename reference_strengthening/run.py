"""Regenerate the reference harmonization and equicorrelation tables."""
import json

from paper4.reference_sensitivity.analysis import resolve_references
from paper4.reference_sensitivity.data_io import frozen_cohorts, write_csv
from paper4.reference_sensitivity.run import execute as baseline_execute
from paper4.reference_strengthening.analysis import analyze_all
from paper4.reference_strengthening.simulation import simulate
from paper4.reference_strengthening.reporting import inference_map


def execute(args, read, folder):
    protocol = json.loads(read("reference_strengthening/protocol.json"))
    original = json.loads(read("reference_sensitivity/protocol.json"))
    base = folder / "baseline"
    base.mkdir()
    baseline_execute(args, read, original, base)
    cohorts, _ = frozen_cohorts(read)
    go, ec = [json.loads(read("plaque_validation/" + name)) for name in (
        "uniprot_hydrolase_sets.json", "uniprot_ec_snapshot.json")]
    references = {(c.dataset, c.group): resolve_references(c, original, go, ec)[0] for c in cohorts}
    tables = analyze_all(cohorts, references, original["primary_targets"], protocol, args.draws, args.seed)
    tables.update(simulate(protocol["simulation"], args.sim_reps, args.seed))
    tables["04_inference_map"] = inference_map()
    table_folder = folder / "tables"
    table_folder.mkdir()
    for name, rows in tables.items():
        write_csv(table_folder / (name + ".csv"), rows, None if rows else ["status"])
