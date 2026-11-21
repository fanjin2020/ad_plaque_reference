"""Assemble the v9 numerical supplement and regenerate its display tables and figures."""
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib as mpl
from matplotlib.transforms import Bbox
import numpy as np

from paper4.legacy.figures import render as render_legacy
from paper4.plaque_validation.panel_definition import ACCESSIONS, DEN
from paper4.reference_extensions.io import write_csv
from paper4.review_followup.reporting import figures as render_followup
from . import cases, evidence_figures as evidence, main_figures, strengthening_art
from .atlas import common, pump_structure, reference_plane, virtual_plaque
from .decision_art import workflow
from .derived_s1 import csv_bytes, derive as derived_s1, read_csv
from .derived_s2 import matched_draws, confirmed_donors, enzyme_donors, coverage_steps, numbered
from .tipping import tipping


ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ("Leitner", "AD", "Leitner AD"), ("Marta-Ariza", "EOAD", "Martá-Ariza EOAD"),
    ("Marta-Ariza", "LOAD", "Martá-Ariza LOAD"), ("Drummond 2022", "eoAD", "Drummond EOAD"),
    ("Marta-Ariza", "DS", "Martá-Ariza DS"), ("Drummond 2022", "DS", "Drummond DS"),
]
GEOMETRY = {
    "lysosomal_anchor": ("v_atpase", "ec3_lysosomal_lumen"),
    "apoe_lipid": ("apod_clu", "lrp1_sort1"),
}


class Tables:
    """Small file-backed interface shared by table builders and plotters."""

    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True)

    def namelist(self):
        return sorted(p.relative_to(self.folder).as_posix() for p in self.folder.rglob("*.csv"))

    def read(self, name):
        return (self.folder / name).read_bytes()

    def write(self, name, rows):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rows if isinstance(rows, bytes) else csv_bytes(rows))

    def table(self, name):
        return read_csv(self.read(name))


def protocol(name):
    return json.loads((ROOT / "protocols" / name).read_text(encoding="utf-8"))


def write_numbered(tables, analyses):
    registry = protocol("table_registry.json")
    for record in registry:
        name = record["file"]
        if record["number"] == 83:
            raw = csv_bytes(tipping(numbered(tables, 28), numbered(tables, 68)))
        else:
            raw = (analyses.work / record["source"]).read_bytes()
        if len(read_csv(raw)) != record["rows"]:
            raise ValueError(f"Unexpected row count for {name}")
        tables.write(name, raw)
    tables.write("Table_index.csv", registry)


def index_data(tables, filename, registry):
    index = []
    for record in registry:
        raw = tables.read(record["file"])
        with (tables.folder / record["file"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = sum(1 for _ in csv.reader(stream)) - 1
        if "rows" in record and rows != record["rows"]:
            raise ValueError(f"Unexpected row count for {record['file']}")
        if filename == "Data_S1_index.csv":
            index.append(dict(file=record["file"], contents=record["contents"], rows=rows,
                              source=record["source"], sha256=sha256(raw).hexdigest()))
        else:
            index.append(dict(file=record["file"], rows=rows, contents=record["contents"],
                              sha256=sha256(raw).hexdigest()))
    tables.write(filename, index)


def write_s1(tables, analyses):
    outputs = analyses.s1.copy()
    outputs.update(derived_s1(tables, outputs["fixed_estimates"], outputs["reference_profiles"], analyses.ec))
    for name, rows in outputs.items():
        write_csv(tables.folder / "Data_S1" / (name + ".csv"), rows)
    registry = protocol("data_s1_registry.json")
    index_data(tables, "Data_S1_index.csv", registry)


def write_s2(tables, data):
    tables.write("Data_S2/matched_draws.csv", matched_draws(tables, data))
    tables.write("Data_S2/caa_donors.csv", confirmed_donors(tables, data))
    tables.write("Data_S2/regional_enzyme_donors.csv", enzyme_donors(tables))
    tables.write("Data_S2/coverage_steps.csv", coverage_steps(tables))
    donor_rows = numbered(tables, 69)
    summary_rows = numbered(tables, 68)
    geometry, summary = [], []
    for family, references in GEOMETRY.items():
        for reference in references:
            matches = [r for r in donor_rows if (r["application"], r["dataset"], r["group"], r["comparison"])
                       == (family, "Leitner", "AD", reference)]
            assert len(matches) == 53 and len({r["donor"] for r in matches}) == 53
            for i, r in enumerate(matches, 1):
                geometry.append(dict(family=family, reference=reference, donor=i, target=r["target_regional"],
                                     ref=r["reference_regional"], difference=r["score"]))
            r, = [r for r in summary_rows if (r["application"], r["dataset"], r["group"], r["comparison"])
                  == (family, "Leitner", "AD", reference)]
            summary.append(dict(family=family, reference=reference, target=r["target_regional_mean"],
                                ref=r["reference_regional_mean"], difference=r["mean"], lower=r["simultaneous_low"],
                                upper=r["simultaneous_high"], positive=r["positive_donors"], n=r["n"]))
    tables.write("Data_S2/paired_geometry.csv", geometry)
    tables.write("Data_S2/paired_summary.csv", summary)
    tables.write("Data_S2/pump_panel.csv", [dict(gene=gene, accession=ACCESSIONS[gene]) for gene in DEN])
    fields = ("application", "n", "missingness", "requested_missing_fraction", "repetitions", "n_estimable",
              "analytic_count_eligibility_fraction", "variance_degenerate_repeats",
              "minimum_complete_donors", "calibration_source_n")
    simulation = [{k: r[k] for k in fields} for r in numbered(tables, 80)
                  if (r["generator"], r["truth"]) == ("gaussian_empirical", "all_null")]
    tables.write("Data_S2/simulation_checks.csv", simulation)


def human_tables(tables, output):
    coverage = tables.table("Data_S1/family_coverage.csv")
    panels = tables.table("Data_S1/panel_specific.csv")
    cohorts = []
    calibration = []
    bench = [r for r in numbered(tables, 30) if float(r["caliper_log2"]) == .5]
    for dataset, group, label in GROUPS:
        apoe, lysosomal = (next(r for r in coverage if (r["dataset"], r["group"], r["application"])
                                == (dataset, group, application)) for application in ("apoe_lipid", "lysosomal_anchor"))
        primary = next(r for r in panels if (r["dataset"], r["group"], r["application"], r["comparison"])
                       == (dataset, group, "apoe_lipid", "apod_clu"))
        cohorts.append(dict(zip(
            ("Study group", "Paired records", "APOE with APOD / CLU", "All APOE panels", "All lysosomal panels"),
            (label, apoe["total_donors"], primary["n"], apoe["shared_donors"], lysosomal["shared_donors"]),
        )))
        if group != "DS":
            modes = ("both_arms_matched", "hydrolases_replaced", "v_atpase_replaced")
            rows = [next(r for r in bench if (r["dataset"], r["group"], r["design"]) == (dataset, group, mode))
                    for mode in modes]
            calibration.append({
                "Study group": label, "Observed (log2)": f"{float(rows[0]['observed']):+.3f}",
                **{name: f"{int(row['hits']):,} ({100 * float(row['upper_tail']):.2f}%)" for name, row in
                   zip(("Replace targets and pump", "Fix pump, replace targets", "Fix targets, replace pump"), rows)},
            })
    reporting = [
        ("Tissue and target", "State both sampled regions, the target list and the biological question."),
        ("Biological references", "Name a primary panel and the alternatives. State their rationale, full membership and selection history."),
        ("Coverage", "Give candidate and measured counts, missing members and shared donor counts. Preserve unavailable comparisons."),
        ("Inference", "Report the regional estimate, each relative estimate, the simultaneous intervals, the multiplicity family and the margin."),
        ("Sensitivity", "Separate fixed-donor membership changes, panel-specific donors and conditional matched-protein benchmarks."),
        ("Reuse", "Supply paired inputs, analysis rules, source identifiers, versions, seeds and numerical output checks."),
    ]
    reporting = [{"Item": item, "Information needed to interpret the result": text} for item, text in reporting]
    display = [
        ("Table_1_cohorts.csv", "Table 1. Paired records and donors with complete named references", cohorts),
        ("Table_2_calibration.csv", "Table 2. Abundance-matched panels reaching the observed enzyme-to-pump contrast", calibration),
        ("Table_3_reporting.csv", "Table 3. Reporting a biological reference analysis", reporting),
    ]
    labels = {
        "v_atpase": "V-ATPase", "lysosomal_lumen_hydrolases": "GO lumen", "ec3_lysosomal_lumen": "EC 3 lumen",
        "ec3_lysosome": "EC 3 lysosome", "ec34_lysosome": "EC 3.4 lysosome",
    }
    membership = [
        {"Groups": "Three" if row["scope"] == "three_larger_groups" else "Four",
         "Reference": labels[row["reference"]], "Members": row["n_members"],
         "Gene symbols": row["genes"].replace("|", ", ")}
        for row in numbered(tables, 57)
    ]
    text = protocol("si_display_text.json")
    display += [
        ("SI_shared_reference_membership.csv", "Shared reference membership", membership),
        ("SI_reading_guide.csv", "Reading guide", text["reading_guide"]),
        ("SI_numerical_table_archive.csv", "Numerical table archive", text["archive_index"]),
    ]
    output.mkdir()
    body = []
    for filename, title, rows in display:
        (output / filename).write_bytes(csv_bytes(rows))
        header = "".join(f"<th>{escape(k)}</th>" for k in rows[0])
        content = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
        body.append(f"<table><caption>{escape(title)}</caption><thead><tr>{header}</tr></thead>"
                    f"<tbody>{content}</tbody></table>")
    style = "body{font:15px Arial,sans-serif;color:#000;background:#fff;margin:2rem}table{border-collapse:collapse;" \
            "width:100%;max-width:960px;border-block:2px solid #000;margin-bottom:2rem}th,td{text-align:left;" \
            "padding:7px 10px;vertical-align:top}thead{border-bottom:1px solid #000}caption{font-weight:bold;" \
            "text-align:left;margin-bottom:6px}"
    (output / "Manuscript_Tables.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Manuscript tables</title>'
        f"<style>{style}</style><body>{''.join(body)}</body></html>\n", encoding="utf-8")


def atlas_artifacts(tables, data, main, supplement):
    common.DATA = tables.folder / "Data_S2"
    shutil.copyfile(data / "6WM2.cif.gz", common.DATA / "6WM2.cif.gz")
    placement = {
        "01_virtual_plaque": (supplement, "FigS16_virtual_section"),
        "03_reference_plane": (main, "Fig3_paired_geometry"),
        "04_vatpase_structure": (supplement, "FigS17_pump_structure"),
    }

    def export_plate(fig, stem):
        title, details = fig.texts[:2]
        title.set_position((.052,.831))
        title.set_fontsize(7.3)
        details.set_position((.952,.831))
        details.set_fontsize(7.0)
        fig.texts[2].remove()
        for artist in fig.artists:
            if hasattr(artist, "get_ydata") and np.array_equal(artist.get_ydata(), [.831,.831]):
                artist.set_visible(False)
        common.rule(fig, .807)
        crop = Bbox.from_extents(0, 0, 7, fig.get_figheight() * .865)
        folder, name = placement[stem]
        evidence.export(fig, folder, name, crop)

    original = common.save
    common.save = export_plate
    try:
        for module in (reference_plane, virtual_plaque, pump_structure):
            module.run()
        # Fig. S15 uses the v9 intervals. Its numerical companion uses the same repetitions.
        from scipy.stats import binom
        checks = []
        simulation = tables.table("Data_S2/simulation_checks.csv")
        for app, size, family in (("lysosomal_anchor", 22, "Lysosomal panel"), ("apoe_lipid", 6, "APOE panel")):
            for n in (5, 20, 53):
                for missing in (.1, .2):
                    r, = [r for r in simulation
                          if (r["application"], int(r["n"]), r["missingness"], float(r["requested_missing_fraction"]))
                          == (app, n, "MCAR", missing)]
                    prob = float(binom.sf(2, n, (1 - missing)**size))
                    assert abs(prob - float(r["analytic_count_eligibility_fraction"])) < 1e-12
                    checks.append(dict(family=family, proteins=size, n=n, missing_probability=missing,
                                       model_probability=prob, simulated_estimable=r["n_estimable"], repetitions=r["repetitions"]))
        tables.write("Data_S2/design_map_crosschecks.csv", checks)
        main_figures.plot_tissue(tables, main)
    finally:
        common.save = original
        (common.DATA / "6WM2.cif.gz").unlink()


def draw_all(tables, data, analyses, dest):
    main = dest / "figures"
    supplement = main / "supplement"
    supplement.mkdir(parents=True)
    main_art = {
        "Fig5_reference_drivers": evidence.reference_drivers,
        "Fig6_matched_panels": evidence.matched_panels,
    }
    si_art = {
        "FigS5_group_references": evidence.coverage_figure,
        "FigS12_family_span": evidence.family_span,
        "FigS13_linked_CAA": evidence.caa_pairs,
        "FigS14_donor_distributions": evidence.donor_distributions,
    }
    with mpl.rc_context():
        evidence.appearance()
        atlas_artifacts(tables, data, main, supplement)
        for folder, family in ((main, main_art), (supplement, si_art)):
            for stem, func in family.items():
                evidence.export(func(tables), folder, stem)
        evidence.export(evidence.toc_graphic(), main, "TOC_graphic")
        main_figures.plot_simulations(tables, main)
        main_figures.plot_probability_checks(tables, supplement)
    with mpl.rc_context():
        evidence.appearance()
        cases.cases(tables.table("Data_S1/fixed_estimates.csv"), main)
        cases.specification(tables.table("Data_S1/ordered_specifications.csv"), supplement)
        workflow(main)
        render_legacy(analyses.legacy, supplement)
        strengthening_art.magnitude(analyses.enhancement / "tables", main)
        strengthening_art.harmonized(analyses.enhancement / "tables", supplement)
        strengthening_art.simulation(analyses.enhancement / "tables", supplement)

    followup = analyses.work / "phenotype"
    followup.mkdir()
    render_followup(followup, analyses.s1)
    baseline = analyses.enhancement / "baseline" / "figures"
    for source, old, stem in (
        (baseline, "Fig1_fixed_size_reference_panels", "FigS2_fixed_panels"),
        (baseline, "Fig3_secondary_application_coverage", "FigS3_C1q_coverage"),
        (baseline, "Fig2_proteome_relative_direction", "FigS4_direction_profiles"),
        (followup / "figures", "02_DS_APOE", "FigS7_DS_APOE"),
        (followup / "figures", "01_reference_membership_profiles", "FigS9_membership"),
    ):
        for extension in ("pdf", "png", "tiff"):
            shutil.copyfile(source / f"{old}.{extension}", supplement / f"{stem}.{extension}")


def generate(analyses, data, output):
    tables = Tables(analyses.work / "publication")
    write_numbered(tables, analyses)
    write_s1(tables, analyses)
    write_s2(tables, data)
    draw_all(tables, data, analyses, output)
    index_data(tables, "Data_S2_index.csv", [
        dict(file=name, contents=text) for name, text in protocol("data_s2_registry.json").items()])
    human_tables(tables, output / "tables")
    with ZipFile(output / "Supporting_Data.zip", "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name in tables.namelist():
            archive.write(tables.folder / name, name)
    shutil.copyfile(analyses.paired, output / "paired_measurements.csv.gz")
    return tables
