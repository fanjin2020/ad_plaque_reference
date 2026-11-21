"""Independent numerical checks and output-level consistency checks."""
import math
import numpy as np
from scipy.stats import t


def audit(cohorts, protocol, tables):
    applications = {a["name"]: a for a in protocol["applications"]}
    cohort_map = {(c.dataset, c.group): c for c in cohorts}
    checked = 0
    worst_error = 0.
    for row in tables.get("03_shared_donor_estimates", []):
        app = applications[row["application"]]
        cohort = cohort_map[row["dataset"], row["group"]]
        union = set(app["targets"])
        for members in app["references"].values():
            union.update(members)
        donor_mask = np.isfinite(cohort.matrix(sorted(union))).all(axis=1)
        gene_values = {g: cohort.delta[:, cohort.genes.index(g)] for g in union}
        values = []
        for donor in np.flatnonzero(donor_mask):
            target = math.fsum(float(gene_values[g][donor]) for g in app["targets"])/len(app["targets"])
            reference = 0.
            if row["comparison"] != "regional":
                members = app["references"][row["comparison"]]
                reference = math.fsum(float(gene_values[g][donor]) for g in members)/len(members)
            values.append(target-reference)
        n = len(values)
        estimate = math.fsum(values)/n
        se = math.sqrt(math.fsum((x-estimate)**2 for x in values)/(n-1)/n)
        error = max(abs(estimate-row["mean"]), abs(se-row["se"]))
        worst_error = max(worst_error, error)
        if n != row["n"] or error > 1e-10:
            raise AssertionError("Independent paired-donor reconstruction failed")
        if se > 1e-14:
            pvalue = float(2*t.sf(abs(estimate/se), n-1))
            half = float(t.ppf(1-protocol["alpha"]/(2*row["family_size"]), n-1))*se
            if abs(pvalue-row["p"]) > 1e-10 or abs(estimate-half-row["simultaneous_low"]) > 1e-10:
                raise AssertionError("Independent inference reconstruction failed")
        checked += 1
    keyed = {}
    fields = ["application", "generator", "truth", "n", "missingness", "requested_missing_fraction"]
    for row in tables.get("16_simulation_methods", []):
        key = tuple(row[k] for k in fields)
        keyed.setdefault(key, {})[row["method"]] = row
        for prefix in ["false_family", "any_true_rejection", "coverage"]:
            rate = row.get(prefix+"_rate")
            if rate is not None:
                if not 0 <= rate <= 1 or row[prefix+"_denominator"] != row["n_estimable"]:
                    raise AssertionError("Invalid simulation denominator")
                if not row[prefix+"_low"]-1e-12 <= rate <= row[prefix+"_high"]+1e-12:
                    raise AssertionError("Monte Carlo interval excludes its estimate")
    for key, methods in keyed.items():
        if not all(x in methods for x in ["all_nominal", "all_holm", "all_bonferroni"]):
            raise AssertionError("Missing planned simulation method")
        a, h, b = [methods[x].get("false_family_count") for x in ["all_nominal", "all_holm", "all_bonferroni"]]
        if a is not None and not b <= h <= a:
            raise AssertionError("Nested rejection procedures have inconsistent error counts")
        if key[2] == "all_null":
            if methods["selected_min_p_nominal"]["false_family_count"] != a:
                raise AssertionError("Selected minimum-p and any nominal rejection must coincide under the global null")
    return {"status": "passed", "independently_reconstructed_contrasts": checked,
            "maximum_absolute_reconstruction_error": worst_error,
            "simulation_scenarios_consistency_checked": len(keyed),
            "scope": "Numerical and logical validation; scientific significance is not a completion criterion."}
