"""Fixed-panel biological applications with shared-donor primary comparisons."""
import itertools
import numpy as np

from .io import application_matrix, rng_for
from .stats import bootstrap_means, summarize_columns


def numeric_row(summary, column):
    return {k: float(v[column]) for k, v in summary.items() if k != "valid"}


def decision_label(low, high, margin):
    if not np.isfinite(low) or not np.isfinite(high):
        return "unresolved"
    if low > margin:
        return "above_margin"
    if high < -margin:
        return "below_margin"
    if low >= -margin and high <= margin:
        return "contained_within_margin"
    return "unresolved"


def analyze(cohorts, protocol, draws, seed):
    tables = {k: [] for k in ["01_coverage", "02_family_availability", "03_shared_donor_estimates",
              "04_donor_scores", "05_reference_differences", "06_member_influence",
              "07_panel_specific_donors", "08_decisions", "09_candidate_coverage"]}
    alpha, minimum = protocol["alpha"], protocol["minimum_donors"]
    for app in protocol["applications"]:
        genes, refs, coefficients = application_matrix(app)
        gi = {g: i for i, g in enumerate(genes)}
        target_indices = [gi[g] for g in app["targets"]]
        for cohort in cohorts:
            base = dict(application=app["name"], **cohort.key)
            d = cohort.matrix(genes)
            observed = np.isfinite(d)
            common = observed.all(axis=1)
            n = int(common.sum())
            roles = {"target": app["targets"], **app["references"]}
            for label, members in roles.items():
                for gene in members:
                    j = gi[gene]
                    tables["01_coverage"].append(dict(**base, role=label, gene=gene,
                        accession=cohort.accessions.get(gene, ""), n_donors_total=len(cohort.donors),
                        n_paired=int(observed[:, j].sum()),
                        paired_fraction=float(observed[:, j].mean()),
                        present_in_input=gene in cohort.genes,
                        member_definition="fixed_named_comparators_not_exhaustive_class"))
            missing = [g for j, g in enumerate(genes) if not observed[:, j].any()]
            tables["02_family_availability"].append(dict(**base, total_donors=len(cohort.donors),
                shared_donors=n, minimum_donors=minimum, planned_references=len(refs),
                union_protein_count=len(genes), missing_genes="|".join(missing),
                status="estimable" if n >= minimum else "not_estimable",
                shared_donor_ids="|".join(np.array(cohort.donors)[common])))

            # Available-panel analyses retain their own donor set and are never used
            # to attribute between-panel changes solely to reference composition.
            for label, members in {"regional": [], **app["references"]}.items():
                required = app["targets"] + members
                present = np.isfinite(cohort.matrix(required)).all(axis=1)
                native_n = int(present.sum())
                row = dict(**base, comparison=label, n=native_n,
                           donor_scope="panel_specific_sensitivity", status="not_estimable",
                           donors="|".join(np.array(cohort.donors)[present]),
                           family_size=1 if label == "regional" else len(refs))
                if native_n >= minimum:
                    values = cohort.matrix(app["targets"])[present].mean(axis=1)
                    if members:
                        values -= cohort.matrix(members)[present].mean(axis=1)
                    fit = summarize_columns(values, alpha, row["family_size"])
                    row.update(numeric_row(fit, 0), status="estimable" if fit["valid"][0] else "zero_variance")
                    # A single-row call cannot give the family-wide Holm adjustment.
                    # Only the family-size Bonferroni bounds are retained here.
                    row.pop("p_holm", None)
                tables["07_panel_specific_donors"].append(row)

            if n < minimum:
                for margin in protocol["margins_log2"]:
                    tables["08_decisions"].append(dict(**base, margin_log2=margin, n=n,
                        regional_status="not_estimable", relative_status="not_estimable",
                        robust_above_all_planned_references=False,
                        reason="insufficient_shared_donors_or_absent_fixed_comparator",
                        inference_scope="exploratory_fixed_family"))
                continue

            data = d[common]
            donor_ids = np.array(cohort.donors)[common]
            regional = data[:, target_indices].mean(axis=1)
            scores = data @ coefficients
            values = np.column_stack([regional, scores])
            labels = ["regional", *refs]
            fit = summarize_columns(values, alpha)
            bounds = bootstrap_means(values, draws, rng_for(seed, "application_bootstrap", app["name"], cohort.dataset, cohort.group))
            loo = (values.sum(axis=0)[None, :]-values)/(n-1)
            for j, label in enumerate(labels):
                refmean = 0. if j == 0 else float((regional-scores[:, j-1]).mean())
                row = dict(**base, comparison=label, n=n, family_size=len(labels),
                    targets="|".join(app["targets"]),
                    members="" if j == 0 else "|".join(app["references"][label]),
                    designated_primary=label == app["primary_reference"],
                    target_regional_mean=float(regional.mean()), reference_regional_mean=refmean,
                    bootstrap_low=float(bounds[0, j]), bootstrap_high=float(bounds[1, j]),
                    bootstrap_draws=draws, bootstrap_scope="pointwise_percentile_descriptive",
                    leave_one_donor_min=float(loo[:, j].min()), leave_one_donor_max=float(loo[:, j].max()),
                    positive_donors=int((values[:, j] > 0).sum()),
                    status="estimable" if fit["valid"][j] else "zero_variance",
                    interval_procedure="Bonferroni_including_regional_and_all_planned_references")
                row.update(numeric_row(fit, j))
                tables["03_shared_donor_estimates"].append(row)
                for i, donor in enumerate(donor_ids):
                    tables["04_donor_scores"].append(dict(**base, donor=donor, comparison=label,
                        score=float(values[i, j]), target_regional=float(regional[i]),
                        reference_regional=0. if j == 0 else float(regional[i]-scores[i, j-1])))

            pairs = list(itertools.combinations(range(len(refs)), 2))
            differences = np.column_stack([scores[:, a]-scores[:, b] for a, b in pairs])
            pairfit = summarize_columns(differences, alpha)
            for j, (a, b) in enumerate(pairs):
                tables["05_reference_differences"].append(dict(**base, first=refs[a], second=refs[b],
                    definition="S_first_minus_S_second_equals_reference_second_minus_first",
                    n=n, family_size=len(pairs), **numeric_row(pairfit, j)))

            for j, label in enumerate(refs):
                members = app["references"][label]
                if len(members) < 2:
                    continue
                for omitted in members:
                    retained = [g for g in members if g != omitted]
                    alternative = regional-data[:, [gi[g] for g in retained]].mean(axis=1)
                    tables["06_member_influence"].append(dict(**base, reference=label, omitted=omitted,
                        remaining="|".join(retained), remaining_count=len(retained), n=n,
                        original_estimate=float(scores[:, j].mean()), alternative_estimate=float(alternative.mean()),
                        change=float((alternative-scores[:, j]).mean()),
                        interpretation="descriptive_same_donors_single_member_reference_if_count_one"))

            for margin in protocol["margins_log2"]:
                regional_status = decision_label(fit["simultaneous_low"][0], fit["simultaneous_high"][0], 0)
                states = [decision_label(fit["simultaneous_low"][j+1], fit["simultaneous_high"][j+1], margin)
                          for j in range(len(refs))]
                if all(s == "above_margin" for s in states):
                    relative = "above_all_planned_references"
                elif all(s == "below_margin" for s in states):
                    relative = "below_all_planned_references"
                elif "above_margin" in states and "below_margin" in states:
                    relative = "opposite_supported_directions"
                elif all(s == "contained_within_margin" for s in states):
                    relative = "all_intervals_within_margin"
                else:
                    relative = "mixed_or_unresolved"
                tables["08_decisions"].append(dict(**base, margin_log2=margin, n=n,
                    regional_status=regional_status, relative_status=relative,
                    robust_above_all_planned_references=regional_status == "above_margin" and relative == "above_all_planned_references",
                    reference_states="|".join(r+":"+s for r, s in zip(refs, states)),
                    reason="conditional_on_named_panels_measured_donors_and_family",
                    inference_scope="exploratory_fixed_family"))

    for candidate in protocol.get("coverage_screen", []):
        for cohort in cohorts:
            target = np.isfinite(cohort.matrix(candidate["targets"])).all(axis=1)
            for label, members in candidate["references"].items():
                observed = np.isfinite(cohort.matrix(members))
                for j, gene in enumerate(members):
                    tables["09_candidate_coverage"].append(dict(candidate=candidate["name"], **cohort.key,
                        reference=label, gene=gene, n_paired=int(observed[:, j].sum()),
                        n_target_and_full_panel=int((target & observed.all(axis=1)).sum()),
                        planned_members=len(members), selection_basis="measurement_availability_not_effect_or_p_value"))
    return tables
