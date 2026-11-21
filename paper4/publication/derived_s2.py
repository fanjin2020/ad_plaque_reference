"""Archive the numeric inputs for the revised figures and verify archived comparisons."""
import csv
import io
import math

import numpy as np
from paper4.four_part.references import calibration
from paper4.four_part.study_tables import LEITNER, MARTA, DRUMMOND, ORDER, checked_inputs, paired_group, read_study
from paper4.plaque_validation.analysis import ACCESSIONS, DEN, NUM, exact_inference
from paper4.plaque_validation.inputs import (
    CAA_VESSELS, read_leitner, read_vessel_matrices, single_gene_matrix, within_donor,
)

AD = "Leitner", "AD"
STRICT = "ec3_lysosomal_lumen"


def read_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def numbered(archive, n):
    (name,) = [f for f in archive.namelist() if f.startswith(f"Table_S{n:02d}_")]
    return read_csv(archive.read(name))


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def matched_draws(archive, data_dir):
    expected = {
        (r["dataset"], r["group"], float(r["caliper_log2"]), r["design"]): r
        for r in numbered(archive, 30)
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("dataset", "group", "caliper_log2", "design", "draw", "contrast_log2"))
    rng = np.random.default_rng(20261117)

    def collect(key, caliper, design, scores):
        writer.writerows(
            (key["dataset"], key["group"], caliper, design, i, format(float(s), ".17g"))
            for i, s in enumerate(scores, start=1)
        )

    checked = 0
    for dataset, filename in (("Leitner", LEITNER), ("Marta-Ariza", MARTA), ("Drummond 2022", DRUMMOND)):
        _, labels, proteins = read_study(dataset, (data_dir / filename).read_bytes())
        for name, group, n in ORDER:
            if name != dataset:
                continue
            _, _, single = paired_group(labels, proteins, group, n)
            for computed in calibration(single, {"dataset": name, "group": group}, rng, record_draws=collect):
                key = name, group, computed["caliper_log2"], computed["design"]
                original = expected[key]
                for field in ("n_background", "min_candidates", "draws", "hits"):
                    assert int(original[field]) == int(computed[field]), (key, field)
                for field in ("upper_tail", "observed", "null_q025", "null_median", "null_q975"):
                    assert abs(float(original[field]) - computed[field]) < 1e-12, (key, field)
                checked += 1
    assert checked == len(expected) == 24
    return stream.getvalue().encode("utf-8")


def confirmed_donors(archive, data_dir):
    vessel_path = data_dir / CAA_VESSELS
    archived_donors = {r["case_code"]: r for r in numbered(archive, 32)}
    meta, proteins, headers, values = read_leitner(data_dir / LEITNER)
    plaque, _ = within_donor(values, headers, "p", "np")
    vessel_gene, vessel, clinical, links = read_vessel_matrices(vessel_path, meta)
    ids = sorted(links[links.map(clinical) == "AD"])
    assert len(ids) == 6 and set(ids) == set(archived_donors)
    vascular_results = [r for r in numbered(archive, 9) if r["reference_class"] == STRICT]
    luminal_genes = vascular_results[0]["control_genes"].split("|")
    controls = {
        r["gene"]: r["accession"] for r in numbered(archive, 28)
        if (r["dataset"], r["group"]) == AD and r["reference"] == STRICT and r["gene"] in luminal_genes
    }
    assert set(controls) == set(luminal_genes)
    identities = ACCESSIONS | controls
    p = single_gene_matrix(proteins[["Genes"]], plaque.loc[:, ids], identities)
    v = single_gene_matrix(vessel_gene, vessel.loc[:, ids], identities)
    rows = []
    for family, reference, baseline in [
        ("v_atpase", DEN, numbered(archive, 8)), (STRICT, luminal_genes, vascular_results),
    ]:
        p_scores = p.loc[NUM].mean() - p.loc[reference].mean()
        v_scores = v.loc[NUM].mean() - v.loc[reference].mean()
        for endpoint, scores in [("plaque", p_scores), ("CAA", v_scores), ("plaque minus CAA", p_scores - v_scores)]:
            result = exact_inference(scores)
            archived, = [r for r in baseline if r["endpoint"] == endpoint]
            assert int(archived["n_paired"]) == result["n_paired"] == 6
            for field in ("estimate_log2", "ci95_low", "ci95_high"):
                assert abs(result[field] - float(archived[field])) < 5e-11, (family, endpoint, field)
        for i, case in enumerate(ids, 1):
            pp, vv = p_scores[case], v_scores[case]
            if family == "v_atpase":
                for column, val in (("plaque_log2", pp), ("caa_log2", vv), ("plaque_minus_caa_log2", pp - vv)):
                    assert abs(val - float(archived_donors[case][column])) < 5e-11
            rows.append(dict(reference=family, panel_members="|".join(reference), donor=f"D{i}",
                             plaque_minus_adjacent_log2=pp, caapos_minus_caaneg_log2=vv,
                             plaque_score_minus_caa_score_log2=pp-vv))
    return rows


def enzyme_donors(archive):
    arm = {(r["case"], r["arm"]): r for r in numbered(archive, 24) if r["group"] == "AD"}
    cases = sorted({key[0] for key in arm})
    assert len(cases) == 53 and len(arm) == 106
    rows = []
    for i, case in enumerate(cases, 1):
        ctsd, tpp1 = (float(arm[case, "p"][g]) - float(arm[case, "np"][g]) for g in ("CTSD", "TPP1"))
        rows.append(dict(donor=f"D{i:02d}", CTSD_log2=ctsd, TPP1_log2=tpp1, CTSD_minus_TPP1_log2=ctsd-tpp1))
    assert sum(r["CTSD_minus_TPP1_log2"] > 0 for r in rows) == 43
    actual = round(float(np.mean([r["CTSD_minus_TPP1_log2"] for r in rows])), 3)
    assert actual == .193
    return rows


def coverage_steps(archive):
    members = [
        r for r in numbered(archive, 28)
        if (r["dataset"], r["group"]) == AD and r["reference"] == STRICT and r["target"].lower() == "false"
    ]
    pairs = [int(r["n_pair"]) for r in members]
    assert set(int(r["n_eligible"]) for r in members) == {53}
    rows = [
        ("Annotated candidates", len(pairs), ""),
        ("Mapped to workbook", sum(r["mapped"].lower() == "true" for r in members), ""),
        ("At least one pair", sum(n >= 1 for n in pairs), 1),
        ("At least 50%", sum(n >= math.ceil(.5 * 53) for n in pairs), math.ceil(.5 * 53)),
        ("At least 80%", sum(n >= math.ceil(.8 * 53) for n in pairs), math.ceil(.8 * 53)),
        ("All 53 donors", sum(n == 53 for n in pairs), 53),
    ]
    assert [n for _, n, _ in rows] == [41, 12, 11, 8, 7, 5]
    return [dict(step=s, n_proteins=n, minimum_donor_pairs=p) for s, n, p in rows]


