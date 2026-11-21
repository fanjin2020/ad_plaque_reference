"""Rebuild the phenotype and decision analyses added to the v10 manuscript."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t
from statsmodels.api import OLS

from paper4.four_part.study_tables import DRUMMOND, LEITNER, MARTA, ORDER, paired_group, read_study
from paper4.reference_extensions.io import application_matrix, validate_protocol, write_csv, write_json
from paper4.reference_extensions.stats import summarize_columns
from paper4.reference_sensitivity.analysis import common_mask, resolve_references
from paper4.reference_sensitivity.data_io import Cohort
from paper4.reference_strengthening.analysis import complete_background, intersect_references
from paper4.source_inputs import check_file, download, output_hashes, verify_output


ROOT = Path(__file__).resolve().parents[1]
ALPHA = .05
MARGINS = (0., .2)
STATE = ("negative_supported", "unresolved", "positive_supported")


def input_manifest():
    return [r for r in json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
            if r["name"] in (LEITNER, MARTA, DRUMMOND)]


def load_data(inputs, manifest):
    data, phenotype = {}, None
    for name, filename in (("Leitner", LEITNER), ("Marta-Ariza", MARTA), ("Drummond 2022", DRUMMOND)):
        record = next(r for r in manifest if r["name"] == filename)
        check_file(inputs / filename, record)
        meta, labels, proteins = read_study(name, (inputs / filename).read_bytes())
        groups = [(study, group, n) for study, group, n in ORDER if study == name]
        if name == "Leitner":
            groups.append(("Leitner", "MCI", 36))
            phenotype = meta
        for study, group, n in groups:
            donors, _, values = paired_group(labels, proteins, group, n)
            data[study, group] = Cohort(study, group, donors, values, {})
    return data, phenotype


def infer_stage(stage, meta):
    def stats(x, y):
        vx = x.var(ddof=1) / len(x)
        vy = y.var(ddof=1) / len(y)
        se = np.sqrt(vx + vy)
        df = (vx + vy)**2 / (vx**2/(len(x)-1) + vy**2/(len(y)-1))
        return x.mean() - y.mean(), se, df

    mci, ad = (np.asarray(list(stage[g].values())) for g in ("MCI", "AD"))
    effect, se, df = stats(mci, ad)
    ci = t.ppf(.975, df) * se
    rows = [dict(model="unadjusted_Welch", n_mci=len(mci), n_ad=len(ad),
                 mean_gap_mci=mci.mean(), mean_gap_ad=ad.mean(),
                 mci_minus_ad=effect, se=se, df=df, ci_low=effect-ci, ci_high=effect+ci,
                 p=2 * t.sf(abs(effect/se), df))]
    cases = [(int(g == "MCI"), score, meta[donor]) for g, entries in stage.items()
             for donor, score in entries.items()]
    records = [(is_mci, score, float(m["Age"]), float(m["PMI"]), m["Sex"], m["Study"])
               for is_mci, score, m in cases if all(m.get(k) for k in ("Age", "PMI", "Sex", "Study"))]
    g, y, age, pmi, sex, study = map(np.asarray, zip(*records))
    dummies = [col == category for col in (sex, study) for category in sorted(set(col))[1:]]
    design = np.column_stack([np.ones(len(y)), g, age-age.mean(), pmi-pmi.mean(), *dummies])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The stage adjustment design is rank deficient")
    fit = OLS(y, design).fit(cov_type="HC3", use_t=True)
    rows.append(dict(model="adjusted_age_sex_PMI_source_HC3", n_mci=int(g.sum()), n_ad=int((1-g).sum()),
                     mean_gap_mci=None, mean_gap_ad=None, mci_minus_ad=fit.params[1],
                     se=fit.bse[1], df=fit.df_resid, ci_low=fit.conf_int()[1, 0],
                     ci_high=fit.conf_int()[1, 1], p=fit.pvalues[1]))
    return rows


def phenotype_results(data, meta, protocol):
    estimates, donors, stage, robust = [], [], {}, []
    for group in ("AD", "MCI"):
        cohort = data["Leitner", group]
        for app in protocol["applications"]:
            genes, names, coef = application_matrix(app)
            mask = np.isfinite(cohort.values(genes)).all(axis=1)
            x = cohort.values(genes)[mask]
            cases = np.asarray(cohort.donors)[mask]
            target = cohort.values(app["targets"])[mask].mean(axis=1)
            scores = np.column_stack([target, x @ coef])
            states = summarize_columns(scores, ALPHA)
            for j, name in enumerate(("regional", *names)):
                row = dict(application=app["name"], group=group, comparison=name, n=len(x),
                           n_total=len(mask), family_size=len(names)+1)
                estimates.append(dict(**row, **{k: float(v[j]) for k, v in states.items() if k != "valid"},
                                      all_clear_0=bool(states["simultaneous_low"].min()>0),
                                      all_clear_0_2=bool(states["simultaneous_low"][0]>0
                                                        and states["simultaneous_low"][1:].min()>.2)))
                donors.extend(dict(**row, donor=d, score=float(y)) for d, y in zip(cases,scores[:, j]))
            if app["name"] == "lysosomal_anchor":
                v = scores[:, 1+names.index("v_atpase")]-scores[:, 1+names.index("ec3_lysosomal_lumen")]
                stage[group] = dict(zip(cases, v))
                lumen = cohort.values(app["references"]["ec3_lysosomal_lumen"])[mask]
                alternatives = {"equal_mean": lumen.mean(axis=1), "median": np.median(lumen, axis=1),
                                "trim_one_each_tail": np.sort(lumen, axis=1)[:, 1:-1].mean(axis=1)}
                fit = summarize_columns(np.column_stack([target-r for r in alternatives.values()]), ALPHA)
                for j, method in enumerate(alternatives):
                    robust.append(dict(group=group, method=method, n=len(x),
                                       target_mean=target.mean(), reference_mean=alternatives[method].mean(),
                                       **{k: float(a[j]) for k, a in fit.items() if k != "valid"}))
    return estimates, donors, infer_stage(stage, meta), robust


def decision_tables(scope, cohort, genes, panels):
    quantified = cohort.values(genes)
    reference = np.column_stack([cohort.values(v).mean(axis=1) for v in panels.values()])
    scores = quantified[:, :, None] - reference[:, None, :]
    n, n_genes, n_refs = scores.shape
    mean = scores.mean(axis=0)
    se = scores.std(axis=0, ddof=1) / np.sqrt(n)
    usable = (np.isfinite(scores).all(axis=0) & (se > 1e-14)).all(axis=1)
    hw = t.ppf(1-ALPHA/(2*n_refs), n-1) * se
    low, high = mean-hw, mean+hw
    keys = list(panels)
    base = dict(scope=scope, **cohort.key, n_donors=n, family_size=n_refs, n_background=n_genes,
                n_estimable=int(usable.sum()), n_unavailable=int((~usable).sum()))
    counts, transitions, detail = [], [], []
    for j, gene in enumerate(genes):
        for k, reference_name in enumerate(keys):
            detail.append(dict(scope=scope, **cohort.key, gene=gene, accession=cohort.proteins[gene][0],
                               comparison=reference_name, members="|".join(panels[reference_name]),
                               n_donors=n, family_size=n_refs, estimable=bool(usable[j]), mean=mean[j, k],
                               low=low[j, k] if usable[j] else None, high=high[j, k] if usable[j] else None))
    for margin in MARGINS:
        state = (low > margin).astype(int) - (high < -margin).astype(int)
        point_flip = (mean.min(axis=1) < -margin-1e-12) & (mean.max(axis=1) > margin+1e-12)
        strict = (state.min(axis=1) == -1) & (state.max(axis=1) == 1) & usable
        differs = (state.min(axis=1) != state.max(axis=1)) & usable
        state1 = dict(**base, margin_log2=margin)
        counts.append(dict(**state1, point_flip_count=int(point_flip.sum()), strict_flip_count=int(strict.sum()),
                           decision_change_count=int(differs.sum()), strict_fraction=float(strict.sum()/usable.sum())))
        donor_states = state[usable]
        for a, b in combinations(range(n_refs), 2):
            for source in (-1, 0, 1):
                for destination in (-1, 0, 1):
                    count = ((donor_states[:, a] == source) & (donor_states[:, b] == destination)).sum()
                    transitions.append(dict(**state1, from_reference=keys[a], to_reference=keys[b],
                                            from_state=STATE[source+1], to_state=STATE[destination+1], count=int(count)))
    return counts, transitions, detail


def background_results(groups):
    params = json.loads((ROOT/"reference_sensitivity/protocol.json").read_text(encoding="utf-8"))
    go = json.loads((ROOT/"plaque_validation/uniprot_hydrolase_sets.json").read_text(encoding="utf-8"))
    ec = json.loads((ROOT/"plaque_validation/uniprot_ec_snapshot.json").read_text(encoding="utf-8"))
    targets = params["primary_targets"]
    cohorts = [groups[s, g] for s, g, _ in ORDER]
    definitions = [resolve_references(c, params, go, ec)[0] for c in cohorts]
    excluded_all = set(targets) | {g for panels in definitions for genes in panels.values() for g in genes}
    common = set.intersection(*(set(complete_background(c, np.ones(len(c.donors), bool), excluded_all)) for c in cohorts))
    harmonized = {k: p for k, p in intersect_references(definitions).items() if len(p) >= 2}
    result = ([], [], [])
    for c, panels in zip(cohorts, definitions):
        excluded = set(targets).union(*panels.values())
        mask, reason = common_mask(c, sorted(excluded))
        if reason or not mask.all():
            raise ValueError(f"Incomplete original sampling units for {c.key}")
        original = complete_background(c, mask, excluded)
        for name, genes, refs in (("observed", original, panels), ("shared_all_four", sorted(common), harmonized)):
            for out, rows in zip(result, decision_tables(name, c, genes, refs)):
                out.extend(rows)
    return result


def run(inputs, output, manifest):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    groups, meta = load_data(inputs, manifest)
    protocol_file = ROOT/"reference_extensions/data/protocol.json"
    protocol = validate_protocol(json.loads(protocol_file.read_text(encoding="utf-8")))
    e, donors, interaction, robust = phenotype_results(groups, meta, protocol)
    print("Completed fixed-panel phenotype comparisons and the stage interaction", flush=True)
    counts, matrix, details = background_results(groups)
    baseline = [(r["n_background"], r["point_flip_count"], r["strict_flip_count"]) for r in counts
                if r["scope"]=="observed" and r["margin_log2"]==.2]
    assert baseline == [(2157, 246, 72), (1473, 266, 0), (1615, 339, 3), (1480, 151, 0)]
    assert {r["n"] for r in e if r["group"]=="MCI"} == {36}
    tables = dict(phenotype_estimates=e, phenotype_donor_scores=donors, phenotype_stage_interaction=interaction,
                  background_decisions=counts, panel_transitions=matrix, background_intervals=details, robust_weights=robust)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper4-work-followup-", dir=output.parent) as temp:
        stage = Path(temp) / "final"
        stage.mkdir()
        for name, values in tables.items():
            write_csv(stage / f"{name}.csv", values)
        write_json(stage / "verification.json", dict(status="passed", sources=manifest,
            design=dict(margins_log2=MARGINS, alpha=ALPHA,
                        interval_scope="each_protein_and_named_reference_family",
                        phase_contrast="MCI_minus_AD_of_EC3_luminal_minus_V_ATPase_gap"),
            tables={name: len(values) for name, values in tables.items()}, outputs_sha256=output_hashes(stage)))
        count = verify_output(stage)
        stage.rename(output)
    print(f"Verified {count} publication-follow-up files in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = input_manifest()
    if args.input_dir is not None:
        run(args.input_dir, args.out, manifest)
    else:
        with TemporaryDirectory(prefix="paper4-followup-") as tmp, ThreadPoolExecutor(3) as pool:
            list(pool.map(lambda r: download(Path(tmp), r), manifest))
            run(Path(tmp), args.out, manifest)


if __name__ == "__main__":
    main()
