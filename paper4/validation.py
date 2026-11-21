"""Check the manuscript's key numerical claims and its regenerated deliverables."""
import math

from PIL import Image
import pymupdf

from paper4.publication.derived_s2 import numbered
from paper4.publication.results import protocol


def select(rows, **criteria):
    row, = (row for row in rows if all(row[k] == str(v) for k, v in criteria.items()))
    return row


def validate_numbers(tables):
    primary = numbered(tables, 1)
    assert [int(row["n_paired"]) for row in primary] == [53, 20, 20, 5]
    assert [round(float(row["estimate_log2"]), 3) for row in primary] == [.833, .850, .821, .777], "Four fixed pump estimates"
    caa = numbered(tables, 8)
    assert all(int(row["n_paired"]) == 6 for row in caa)
    assert round(float(select(caa, endpoint="plaque minus CAA")["estimate_log2"]), 3) == 1.416
    enzyme = select(numbered(tables, 10), dataset="Leitner", group="AD")
    assert (int(enzyme["n"]), int(enzyme["positive"]), round(float(enzyme["mean_D"]), 3)) == (53, 43, .193)

    fixed = numbered(tables, 68)
    strict = select(fixed, application="lysosomal_anchor", dataset="Leitner", group="AD",
                    comparison="ec3_lysosomal_lumen")
    assert round(float(strict["mean"]), 3) == .037
    assert [round(float(strict[k]), 3) for k in ("simultaneous_low", "simultaneous_high")] == [-.050, .125]
    family = tables.table("Data_S1/family_summary.csv")
    apoe = select(family, application="apoe_lipid", dataset="Leitner", group="AD")
    assert round(float(apoe["smallest_simultaneous_lower"]), 3) == 1.817, "Leitner APOE simultaneous lower bound"
    assert [int(row["n_proteins"]) for row in tables.table("Data_S2/coverage_steps.csv")] == [41, 12, 11, 8, 7, 5]

    matched = numbered(tables, 30)
    assert [int(select(matched, dataset="Leitner", group="AD", caliper_log2=.5, design=kind)["hits"])
            for kind in ("both_arms_matched", "hydrolases_replaced", "v_atpase_replaced")] == [110, 106, 5914]
    assert sum(int(row["draws"]) for row in matched) == 240_000
    assert len(tables.table("Data_S2/virtual_cells.csv")) == 310
    structure = tables.table("Data_S2/structure_mapping.csv")
    assert len(structure) == 8 and sum(row["exact_match"] == "True" for row in structure) == 7

    simulation = numbered(tables, 80)
    methods = numbered(tables, 81)
    equicorrelation = numbered(tables, 62)
    assert len(equicorrelation) == 90 and sum(int(row["repetitions"]) for row in equicorrelation) == 180_000
    assert len(simulation) == 360 and sum(int(row["repetitions"]) for row in simulation) == 720_000
    focus = dict(generator="gaussian_empirical", truth="all_null", n=53)
    mc = {}
    for app, nominal in (("lysosomal_anchor", (279, 62)), ("apoe_lipid", (176, 70))):
        row = select(simulation, **focus, application=app, missingness="MCAR", requested_missing_fraction=.2)
        assert int(row["repetitions"]) == 2_000
        mc[app] = {"estimable_at_20_percent_MCAR": int(row["n_estimable"])}
        for method, expected in zip(("selected_min_p_nominal", "all_holm"), nominal):
            row = select(methods, **focus, application=app, missingness="complete", method=method)
            count = int(row["false_family_count"])
            assert int(row["false_family_denominator"]) == 2_000
            assert math.isclose(float(row["false_family_rate"]), count / 2_000, abs_tol=1e-12)
            # Multivariate Gaussian draws can differ across numerical library builds.
            assert abs(count / 2_000 - expected / 2_000) < .025
            mc[app][method] = {"reproduced_count": count, "v9_count": expected}
        assert mc[app]["selected_min_p_nominal"]["reproduced_count"] > mc[app]["all_holm"]["reproduced_count"]
    return {
        "paired_record_counts": [53, 20, 20, 5],
        "benchmark_draws": 240_000,
        "equicorrelation_scenarios": 90,
        "equicorrelation_repeats": 180_000,
        "simulation_scenarios": 360,
        "simulation_repeats": 720_000,
        "structure_mapped_accessions": 7,
        "monte_carlo_check": mc,
    }


def validate_figures(output):
    expected = protocol("figures.json")
    files = {}
    with pymupdf.open() as gallery:
        bookmarks = []
        for index, (label, record) in enumerate(expected.items(), 1):
            folder = output / "figures"
            if label.startswith("Figure S"):
                folder /= "supplement"
            stem = record["file"]
            paths = {suffix: folder / f"{stem}.{suffix}" for suffix in ("pdf", "png", "tiff")}
            with pymupdf.open(paths["pdf"]) as figure:
                assert len(figure) == 1
                page = figure[0]
                spans = [s for block in page.get_text("dict")["blocks"] if block["type"] == 0
                         for line in block["lines"] for s in line["spans"] if s["text"].strip()]
                assert spans and all(s["color"] == 0 for s in spans), f"Figure text color: {stem}"
                assert min(s["size"] for s in spans) >= 4.5, f"Figure font size: {stem}"
                assert page.rect.width <= 7.02 * 72, f"Figure width: {stem}"
                gallery.insert_pdf(figure)
                bookmarks.append([1, label, index])
            with Image.open(paths["png"]) as image:
                image.verify()
            with Image.open(paths["tiff"]) as image:
                dpi = float(min(image.info["dpi"]))
                assert abs(dpi - record["tiff_dpi"]) < 1, f"TIFF resolution: {stem}"
                image.verify()
            files[label] = {"file": paths["pdf"].relative_to(output).as_posix(), "tiff_dpi": dpi}
        gallery.set_toc(bookmarks)
        gallery.save(output / "Figures.pdf", garbage=4, deflate=True)
    return files
