"""Rebuild phenotype, background and membership results from the three source workbooks."""
import re

import numpy as np
from scipy.stats import t

from paper4.reference_extensions.application import analyze
from paper4.reference_extensions.io import Cohort as MatrixCohort
from paper4.reference_extensions.stats import holm
from paper4.reference_sensitivity.analysis import common_mask, resolve_references, score_intervals
from paper4.reference_sensitivity.data_io import Cohort as GeneCohort
from paper4.reference_strengthening.analysis import complete_background, evaluate, intersect_references, panel_kind
from paper4.four_part.study_tables import DRUMMOND, LEITNER, MARTA, ORDER, checked_inputs, paired_group, read_study

GROUPS = (*ORDER, ("Marta-Ariza", "DS", 20), ("Drummond 2022", "DS", 5))
DRAWS = 10000
SEED = 20261119
MARGIN = 0.2


def read_groups(read):
    sources = checked_inputs(read)
    matrices, original, genotypes = {}, {}, {}
    for study, file in (("Leitner", LEITNER), ("Marta-Ariza", MARTA), ("Drummond 2022", DRUMMOND)):
        print("Reading paired source measurements:", file, flush=True)
        metadata, labels, proteins = read_study(study, read("data/" + file))
        for dataset, group, n in GROUPS:
            if dataset != study:
                continue
            donors, _, records = paired_group(labels, proteins, group, n)
            key = (study, group)
            genes = sorted(records)
            matrices[key] = MatrixCohort(study, group, donors, genes,
                {g: records[g][0] for g in genes},
                np.column_stack([records[g][1] for g in genes]),
                np.column_stack([records[g][2] for g in genes]))
            if group != "DS":
                original[key] = GeneCohort(study, group, donors, records, {})
            if metadata:
                column = "APOE Genotype" if study == "Leitner" else "APOE"
                genotypes[key] = {d: carrier(metadata[d][column]) for d in donors}
    return sources, matrices, original, genotypes


def carrier(value):
    code = str(value or "").upper().strip().replace("Ε", "E")
    if code in {"", "NA", "N/A", "ND", "UNKNOWN", "NAN"}:
        return None
    parsed = re.fullmatch(r"E?([234])\s*[/|,\-]?\s*E?([234])", code)
    if not parsed:
        raise ValueError(f"Unexpected APOE genotype encoding {value!r}")
    return "4" in parsed.groups()


def background_and_membership(original, cfg, go, ec):
    targets = cfg["primary_targets"]
    studies = [original[(s, g)] for s, g, _ in ORDER]
    measured = [resolve_references(c, cfg, go, ec)[0] for c in studies]
    shared = intersect_references(measured)
    output = {"background_summary": [], "background_proteins": [], "reference_profiles": [],
              "reference_membership": [], "shared_reference_lists": []}
    for ref, panel in shared.items():
        output["shared_reference_lists"].append(dict(scope="all_four_groups", reference=ref,
            n_members=len(panel), genes="|".join(panel), panel_kind=panel_kind(panel)))
    for cohort, panels in zip(studies, measured):
        mask, problem = common_mask(cohort, targets + sorted({g for p in panels.values() for g in p}))
        if problem or not mask.all():
            raise ValueError(f"Expected complete observed reference donors in {cohort.key}: {problem}")
        excluded = set(targets).union(*panels.values())
        genes = complete_background(cohort, mask, excluded)
        print(f"Recomputing background and five references: {cohort.dataset} {cohort.group}, "
              f"{len(genes)} eligible proteins", flush=True)
        rows, _, _, native_scores, _ = evaluate(cohort, targets, {"observed": panels}, mask, genes,
            {"study_measured": set(genes)}, [MARGIN], DRAWS, SEED, "original")
        summary = rows[0]
        output["background_summary"].append(summary)
        refs = list(panels)
        reference_means = np.array([cohort.values(panels[r])[mask].mean(axis=1).mean() for r in refs])
        rel = cohort.values(genes)[mask].mean(axis=0)[:, None] - reference_means
        count = 0
        for gene, values in zip(genes, rel):
            imin, imax = int(values.argmin()), int(values.argmax())
            flip = values[imin] < -MARGIN-1e-12 and values[imax] > MARGIN+1e-12
            count += int(flip)
            output["background_proteins"].append(dict(**cohort.key, gene=gene,
                accession=cohort.proteins[gene][0], minimum_score=values[imin], minimum_reference=refs[imin],
                maximum_score=values[imax], maximum_reference=refs[imax], crosses_both_margins=bool(flip),
                **{name: float(value) for name, value in zip(refs, values)}))
        if count != summary["count"] or len(genes) != summary["n_proteins"]:
            raise ValueError("Independent protein-by-protein recount disagrees with bootstrap analysis")
        target = cohort.values(targets)[mask].mean(axis=1)
        same_members = np.column_stack([target - cohort.values(panel)[mask].mean(axis=1)
                                        for panel in shared.values()])
        fits = score_intervals(same_members, DRAWS, SEED,
            ("strengthening_targets", "all_four_groups", "harmonized_with_singletons", cohort.dataset, cohort.group))
        uniform = [dict(**cohort.key, scope="all_four_groups", definition="harmonized_with_singletons",
            reference=ref, reference_genes="|".join(panel), panel_kind=panel_kind(panel),
            targets="|".join(targets), **fit) for (ref, panel), fit in zip(shared.items(), fits)]
        for row in native_scores + uniform:
            output["reference_profiles"].append(dict(**row, n_members=len(row["reference_genes"].split("|"))))
        for ref, panel in panels.items():
            for gene in panel:
                record = cohort.proteins[gene]
                output["reference_membership"].append(dict(**cohort.key, reference=ref, gene=gene,
                    accession=record[0], n_total=len(cohort.donors),
                    n_paired=int(np.isfinite(record[1]).sum()), retained_in_four_group_intersection=gene in shared[ref]))
    return output


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y)
    se = np.sqrt(vx + vy)
    if min(len(x), len(y)) < 2 or se <= 0 or not np.isfinite(se):
        raise ValueError("Welch contrast requires measurements and positive sampling variance")
    df = (vx + vy)**2 / (vx**2/(len(x)-1) + vy**2/(len(y)-1))
    effect = x.mean()-y.mean()
    half_width = t.ppf(.975, df)*se
    return dict(mean_carrier=x.mean(), mean_noncarrier=y.mean(), carrier_minus_noncarrier=effect,
                se=se, df=df, nominal_low=effect-half_width, nominal_high=effect+half_width,
                p_two_sided=2*t.sf(abs(effect/se), df))


def genotype_tables(fixed, genotypes, config):
    app = next(a for a in config["applications"] if a["name"] == "apoe_lipid")
    primary = app["primary_reference"]
    scores = {}
    for row in fixed["04_donor_scores"]:
        if row["application"] == app["name"]:
            key = row["dataset"], row["group"]
            scores.setdefault(key, {}).setdefault(row["donor"], {})[row["comparison"]] = row["score"]
    coverage, tests, donors = [], [], []
    for dataset, group, n in GROUPS:
        key = dataset, group
        calls = genotypes.get(key, {})
        both = scores.get(key, {})
        usable = sorted(d for d in both if calls.get(d) is not None)
        counts = dict(n_carriers=sum(calls[d] for d in usable),
                      n_noncarriers=sum(not calls[d] for d in usable))
        coverage.append(dict(dataset=dataset, group=group, n_total=n, n_shared_family=len(both),
            n_known_genotypes=sum(v is not None for v in calls.values()),
            n_unknown_or_unlinked=n-sum(v is not None for v in calls.values()),
            genotype_source="paired_workbook_metadata" if calls else "no_linked_genotype_in_workbook",
            **counts))
        if min(counts.values()) < 2:
            continue
        for ref in app["references"]:
            is_primary = ref == primary
            contrast = "primary_score" if is_primary else "alternative_minus_primary_score"
            vals = {d: both[d][ref] if is_primary else both[d][ref]-both[d][primary] for d in usable}
            row = dict(dataset=dataset, group=group, reference=ref, primary_reference=primary,
                       contrast=contrast, **counts, **welch([v for d,v in vals.items() if calls[d]],
                                                            [v for d,v in vals.items() if not calls[d]]))
            row["adjustment_family"] = ("descriptive_primary" if is_primary else
                ("DS_two_reference_changes" if group == "DS" else "original_AD_six_reference_changes"))
            tests.append(row)
            for donor, value in vals.items():
                donors.append(dict(dataset=dataset, group=group, donor=donor, carrier=int(calls[donor]),
                                   reference=ref, contrast=contrast, value=value))
    for family, expected in (("DS_two_reference_changes", 2), ("original_AD_six_reference_changes", 6)):
        members = [r for r in tests if r["adjustment_family"] == family]
        if len(members) != expected:
            raise ValueError("Unexpected estimability of genotype contrasts: " + family)
        for row, adjusted in zip(members, holm([r["p_two_sided"] for r in members])):
            row["p_holm"] = float(adjusted)
            row["family_size"] = expected
    return coverage, tests, donors


def equivalence(fixed):
    def selected(r):
        return (r["dataset"], r["group"], r["application"], r["comparison"]) == (
            "Leitner", "AD", "lysosomal_anchor", "ec3_lysosomal_lumen")
    rows = [r for r in fixed["04_donor_scores"] if selected(r)]
    fit = next(r for r in fixed["03_shared_donor_estimates"] if selected(r))
    values = np.array([r["score"] for r in rows])
    n = len(values)
    mean, se = values.mean(), values.std(ddof=1)/np.sqrt(n)
    width = t.ppf(.95, n-1)*se
    p_lower, p_upper = t.sf((mean+MARGIN)/se, n-1), t.cdf((mean-MARGIN)/se, n-1)
    passes = bool(max(p_lower, p_upper) < .05)
    if passes != bool(mean-width > -MARGIN and mean+width < MARGIN):
        raise ValueError("TOST and 90% interval decisions disagree")
    return dict(dataset=fit["dataset"], group=fit["group"], reference=fit["comparison"], members=fit["members"],
        target=fit["targets"], n=n, mean=mean, se=se, lower_margin=-MARGIN, upper_margin=MARGIN,
        ci90_low=mean-width, ci90_high=mean+width, p_lower=p_lower, p_upper=p_upper,
        p_equivalence=max(p_lower,p_upper), meets_exploratory_margin=passes,
        simultaneous_low=fit["simultaneous_low"], simultaneous_high=fit["simultaneous_high"],
        multiplicity="one_exploratory_TOST; simultaneous_95pct_bounds_from_v6_family_of_six")


def run_analysis(read, config, annotations):
    sources, matrices, original, genotypes = read_groups(read)
    cohorts = [matrices[s, g] for s, g, _ in GROUPS]
    print("Recomputing locked APOE and CTSD/TPP1 comparisons for six groups", flush=True)
    fixed = analyze(cohorts, config, DRAWS, SEED)
    outputs = background_and_membership(original, *annotations)
    coverage, fits, donor_fits = genotype_tables(fixed, genotypes, config)
    outputs.update(genotype_coverage=coverage, genotype_estimates=fits, genotype_donor_scores=donor_fits,
                   equivalence=[equivalence(fixed)])
    export = {"member_coverage":"01_coverage", "family_coverage":"02_family_availability",
              "fixed_estimates":"03_shared_donor_estimates", "fixed_donor_scores":"04_donor_scores",
              "panel_specific":"07_panel_specific_donors", "fixed_decisions":"08_decisions"}
    outputs.update({name:fixed[code] for name,code in export.items()})
    return sources, outputs
