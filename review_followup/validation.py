"""Check every retained v6 estimate against its deposited table and recount the new results."""
import csv
import io
import math
from collections import Counter


def baseline(read, number):
    return list(csv.DictReader(io.StringIO(read(f"v6/S{number}.csv").decode("utf-8-sig"))))


def key(row, columns):
    return tuple(str(row[c]) for c in columns.split())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def equivalent(value, ref):
    try:
        return math.isclose(float(value), float(ref), abs_tol=2e-10, rel_tol=2e-9)
    except (ValueError, TypeError):
        return str(value) == str(ref)


def compare(actual, expected, keys, fields):
    lookup = {key(r, keys): r for r in actual}
    expected_keys = {key(r, keys) for r in expected}
    require(len(actual) == len(lookup) == len(expected) == len(expected_keys)
            and set(lookup) == expected_keys, "v6 row identifiers disagree: " + keys)
    for ref in expected:
        k = key(ref, keys)
        for name in fields.split():
            if not equivalent(lookup[k][name], ref[name]):
                raise ValueError(f"v6 mismatch {k} {name}: {lookup[k][name]!r} vs {ref[name]!r}")
    return len(expected)


def verify_tables(read, tables):
    def original(name):
        return [r for r in tables[name] if r["group"] != "DS"]

    def lookup(name, **terms):
        rows = [r for r in tables[name] if all(r[k] == v for k,v in terms.items())]
        require(len(rows) == 1, f"Expected one row for {name} {terms}, found {len(rows)}")
        return rows[0]

    results = {}
    app = "application dataset group"
    results["S52_four_margin_fractions_and_donor_intervals"] = compare(tables["background_summary"],
        [r for r in baseline(read,52) if float(r["margin_log2"]) == .2], "dataset group",
        "n_donors n_references n_proteins count fraction bootstrap_low bootstrap_high loo_min loo_max bootstrap_draws")
    results["S57_intersection_members"] = compare(tables["shared_reference_lists"],
        [r for r in baseline(read,57) if r["scope"] == "all_four_groups"], "scope reference",
        "n_members genes panel_kind")
    profiles = [r for r in baseline(read,59) if (r["scope"],r["definition"]) in
        {("original","observed"),("all_four_groups","harmonized_with_singletons")}]
    results["S59_target_profiles_and_intervals"] = compare(tables["reference_profiles"], profiles,
        "dataset group scope definition reference",
        "reference_genes panel_kind n family_size estimate se t_low t_high bonferroni_low bonferroni_high "
        "bootstrap_low bootstrap_high p_two_sided p_holm_within_family")
    results["S66_unique_mappings_and_coverage"] = compare(original("member_coverage"), baseline(read,66),
        app + " role gene", "accession n_donors_total n_paired paired_fraction")
    results["S67_family_availability"] = compare(original("family_coverage"), baseline(read,67),
        app, "total_donors shared_donors planned_references union_protein_count missing_genes status shared_donor_ids")
    results["S68_fixed_estimates_and_intervals"] = compare(original("fixed_estimates"), baseline(read,68),
        app + " comparison", "n family_size members positive_donors mean se nominal_low nominal_high "
        "simultaneous_low simultaneous_high p p_holm bootstrap_low bootstrap_high")
    results["S69_individual_scores"] = compare(original("fixed_donor_scores"), baseline(read,69),
        app + " comparison donor", "score target_regional reference_regional")

    apo = dict(application="apoe_lipid", group="DS")
    ds = dict(**apo, dataset="Marta-Ariza")
    chk = lookup("family_coverage", **ds)
    require(chk["shared_donors"] == 20 and chk["union_protein_count"] == 6, "DS six-protein coverage changed")
    # These anchors are independent workbook read-through results, rounded only for QA.
    anchors = {"regional":(2.992717,2.497089,3.488345), "apod_clu":(2.319484,1.788565,2.850404),
               "apoa1_apod_clu":(2.533096,1.953262,3.112931), "lrp1_sort1":(2.700453,2.171462,3.229444)}
    for name, expected in anchors.items():
        row = lookup("fixed_estimates", **ds, comparison=name)
        require(row["n"] == 20 and row["family_size"] == 4, "DS simultaneous family changed")
        for field, value in zip(("mean","simultaneous_low","simultaneous_high"), expected):
            require(abs(row[field]-value) < 2e-6, f"Independent DS workbook estimate mismatch: {name} {field}")
        if name != "regional":
            require(row["positive_donors"] == 20, "Unexpected DS relative scores")
    other = dict(**apo, dataset="Drummond 2022")
    partial = lookup("panel_specific", **other, comparison="apod_clu")
    require(partial["n"] == 5 and partial["family_size"] == 3 and abs(partial["mean"]-2.935085)<2e-6,
            "Independent Drummond DS panel disagrees")
    require(lookup("family_coverage", **other)["shared_donors"] == 0 and
            lookup("member_coverage", **other, role="lrp1_sort1", gene="SORT1")["n_paired"] == 0,
            "Drummond fixed-family availability differs")
    lys = dict(application="lysosomal_anchor", dataset="Marta-Ariza", group="DS")
    require(lookup("family_coverage", **lys)["shared_donors"] == 0 and
            lookup("member_coverage", **lys, role="ec3_lysosome", gene="OCRL")["n_paired"] == 0
            and lookup("panel_specific", **lys, comparison="v_atpase")["n"] == 20
            and lookup("panel_specific", **lys, comparison="ec3_lysosomal_lumen")["n"] == 18,
            "DS lysosomal coverage differs")
    sums = Counter((r["dataset"],r["group"]) for r in tables["background_proteins"] if r["crosses_both_margins"])
    counts = Counter((r["dataset"],r["group"]) for r in tables["background_proteins"])
    for r in tables["background_summary"]:
        k = r["dataset"], r["group"]
        require((sums[k], counts[k]) == (r["count"], r["n_proteins"]), "Background recount failed")
    for ds_name, group, a, b, unknown in (("Leitner","AD",23,30,0),("Marta-Ariza","EOAD",8,12,0),
        ("Marta-Ariza","LOAD",12,8,0),("Marta-Ariza","DS",6,13,1)):
        cov = lookup("genotype_coverage", dataset=ds_name, group=group)
        require((cov["n_carriers"],cov["n_noncarriers"],cov["n_unknown_or_unlinked"]) == (a,b,unknown),
                "Genotype denominator differs " + group)
    gen = [r for r in tables["genotype_estimates"] if r["adjustment_family"] == "original_AD_six_reference_changes"]
    require(len(gen)==6 and abs(min(r["p_holm"] for r in gen)-0.1096981)<2e-6, "Genotype Holm check failed")
    tost = tables["equivalence"][0]
    require(tost["n"] == 53 and abs(tost["p_equivalence"]-2.40975551e-6)<1e-12 and
            abs(tost["ci90_low"]+.0159147118)<1e-8 and abs(tost["ci90_high"]-.0908049781)<1e-8,
            "Independent equivalence result disagrees")
    results.update(new_DS_APOE_estimates_checked=4, reviewed_Drummond_DS_primary=True,
        verified_DS_coverage_boundaries=True, background_gene_rows_recounted=sum(counts.values()),
        unknown_genotype_handling_verified=True, verified_exploratory_Holm_family_of_six=True,
        verified_TOST_and_CI_decision=True)
    return results
