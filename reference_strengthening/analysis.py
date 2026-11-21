"""Paired donor bootstrap with shared weights across definitions and backgrounds."""
import itertools
import numpy as np

from paper4.reference_sensitivity.analysis import common_mask, random_generator, score_intervals


def panel_kind(genes, minimum=2):
    return "empty" if not genes else "single_protein" if len(genes) < minimum else "multi_protein_panel"


def intersect_references(mappings):
    names = list(mappings[0])
    if any(set(m) != set(names) for m in mappings):
        raise ValueError("Harmonization requires the same reference-class names")
    return {name: sorted(set.intersection(*(set(m[name]) for m in mappings))) for name in names}


def magnitude_states(gene_means, reference_means, margins):
    """Both positive and negative relative scores must exceed the margin strictly."""
    genes = np.atleast_2d(gene_means)
    refs = np.atleast_2d(reference_means)
    margin = np.asarray(margins)
    low = genes[:, :, None] - refs.max(axis=1)[:, None, None]
    high = genes[:, :, None] - refs.min(axis=1)[:, None, None]
    return (low < -margin[None, None, :] - 1e-12) & (high > margin[None, None, :] + 1e-12)


def complete_background(cohort, mask, excluded):
    return sorted(g for g, record in cohort.proteins.items()
                  if g not in excluded and np.isfinite(record[1][mask]).all())


def evaluate(cohort, targets, definitions, mask, genes, backgrounds, margins, draws, seed, scope):
    """One shared resampling distribution for every comparison in this block."""
    n = int(mask.sum())
    if n < 3 or not genes:
        raise ValueError("At least three paired donors and one background gene are required")
    for name, panels in definitions.items():
        if len(panels) < 2 or any(not genes_ for genes_ in panels.values()):
            raise ValueError("At least two nonempty references are required: " + name)
    x = cohort.values(genes)[mask]
    r = {name: np.column_stack([cohort.values(g)[mask].mean(axis=1) for g in panels.values()])
         for name, panels in definitions.items()}
    bg = {name: np.array([g in members for g in genes]) for name, members in backgrounds.items()}
    if any(not keep.any() for keep in bg.values()):
        raise ValueError("Empty requested protein background")
    combos = list(itertools.product(definitions, backgrounds))
    boot = np.empty((draws, len(combos), len(margins)))
    observed, leave = {}, {}
    gene_loo = (x.sum(axis=0)[None, :] - x) / (n - 1)
    for name in definitions:
        point = magnitude_states(x.mean(axis=0), r[name].mean(axis=0), margins)[0]
        loo = magnitude_states(gene_loo, (r[name].sum(axis=0)[None, :] - r[name]) / (n-1), margins)
        for background, keep in bg.items():
            observed[name, background] = point[keep].mean(axis=0)
            leave[name, background] = loo[:, keep, :].mean(axis=1)
    rng = random_generator(seed, "strengthening", scope, cohort.dataset, cohort.group)
    for start in range(0, draws, 128):
        count = min(128, draws-start)
        weights = rng.multinomial(n, np.full(n, 1/n), size=count)
        gene_means = weights @ x / n
        for name in definitions:
            states = magnitude_states(gene_means, weights @ r[name] / n, margins)
            for background, keep in bg.items():
                index = combos.index((name, background))
                boot[start:start+count, index] = states[:, keep, :].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    rows, differences, members, effects, donor_records = [], [], [], [], []
    for j, (name, background) in enumerate(combos):
        size = int(bg[background].sum())
        for k, margin in enumerate(margins):
            fraction = float(observed[name, background][k])
            rows.append(dict(**cohort.key, scope=scope, definition=name, background=background,
                n_donors=n, n_proteins=size, n_references=len(definitions[name]), margin_log2=margin,
                count=int(round(fraction*size)), fraction=fraction, bootstrap_low=float(low[j,k]),
                bootstrap_high=float(high[j,k]), loo_min=float(leave[name,background][:,k].min()),
                loo_max=float(leave[name,background][:,k].max()), bootstrap_draws=draws,
                precision_note="small_sample" if n < 10 else "",
                interval_scope="pointwise_donor_bootstrap_conditional_on_fixed_membership"))
        if name != "observed":
            index = combos.index(("observed", background))
            changes = boot[:, j] - boot[:, index]
            ci = np.quantile(changes, [.025, .975], axis=0)
            for k, margin in enumerate(margins):
                differences.append(dict(**cohort.key, scope=scope, definition=name, background=background,
                    margin_log2=margin, n_donors=n, n_proteins=size,
                    fraction_difference=float(observed[name,background][k]-observed["observed",background][k]),
                    bootstrap_low=float(ci[0,k]), bootstrap_high=float(ci[1,k]), bootstrap_draws=draws,
                    contrast="harmonized_minus_observed_same_donors_and_background"))
    for background, keep in bg.items():
        members.extend(dict(**cohort.key, scope=scope, background=background, gene=g,
                            accession=cohort.proteins[g][0]) for g, retained in zip(genes,keep) if retained)
    target = cohort.values(targets)[mask].mean(axis=1)
    names = [d for d, retained in zip(cohort.donors, mask) if retained]
    for name, panels in definitions.items():
        score = target[:, None] - r[name]
        statistics = score_intervals(score, draws, seed, ("strengthening_targets",scope,name,cohort.dataset,cohort.group))
        for j, ((ref, panel), stats) in enumerate(zip(panels.items(),statistics)):
            effects.append(dict(**cohort.key, scope=scope, definition=name, reference=ref,
                reference_genes="|".join(panel), panel_kind=panel_kind(panel), targets="|".join(targets), **stats))
            donor_records.extend(dict(**cohort.key, scope=scope, definition=name, reference=ref,
                donor=d, score=float(value)) for d,value in zip(names,score[:,j]))
    return rows, differences, members, effects, donor_records


def analyze_all(cohorts, references, targets, protocol, draws, seed):
    outputs = {key: [] for key in ("01_margin_curves", "02_harmonized_curves", "02_fraction_changes",
        "02_background_members", "02_reference_members", "02_target_effects", "02_target_donors", "02_design_audit")}
    lookup = {(c.dataset,c.group):c for c in cohorts}
    margins = protocol["margins_log2"]
    for cohort in cohorts:
        key = (cohort.dataset,cohort.group)
        panels = references[key]
        excluded = set(targets) | {g for panel in panels.values() for g in panel}
        mask,reason = common_mask(cohort, sorted(excluded))
        if reason:
            raise ValueError(str(key)+": "+reason)
        genes = complete_background(cohort,mask,excluded)
        print(f"Magnitude curves: {cohort.dataset} {cohort.group}, {len(genes)} proteins",flush=True)
        rows,_,members,effects,donors = evaluate(cohort,targets,{"observed":panels},mask,genes,
            {"study_measured":set(genes)},margins,draws,seed,"original")
        outputs["01_margin_curves"].extend(rows)
        outputs["02_background_members"].extend(members)
        outputs["02_target_effects"].extend(effects)
        outputs["02_target_donors"].extend(donors)
    for scope, keys in protocol["harmonization_scopes"].items():
        keys = [tuple(k) for k in keys]
        if not set(keys) <= set(lookup):
            raise ValueError("Missing required cohorts for scope "+scope)
        shared = intersect_references([references[k] for k in keys])
        minimum = protocol["minimum_members_for_class_label"]
        class_panels = {name:genes for name,genes in shared.items() if len(genes)>=minimum}
        all_panels = {name:genes for name,genes in shared.items() if genes}
        if len(class_panels)<2:
            raise ValueError("Fewer than two multi-protein panels in "+scope)
        # Common exclusion union prevents former reference proteins re-entering the
        # background when a definition loses members. Observed and harmonized scores
        # are compared on the same donors, exclusion rules and background genes.
        excluded = set(targets) | {g for k in keys for panel in references[k].values() for g in panel}
        masks, backgrounds = {}, {}
        for key in keys:
            needed = targets + list({g for panel in references[key].values() for g in panel})
            masks[key],reason = common_mask(lookup[key],needed)
            if reason:
                raise ValueError(str(key)+": "+reason)
            backgrounds[key] = complete_background(lookup[key],masks[key],excluded)
        common = set.intersection(*(set(backgrounds[k]) for k in keys))
        if not common:
            raise ValueError("No common measured background in "+scope)
        for ref, panel in shared.items():
            outputs["02_reference_members"].append(dict(scope=scope,reference=ref,
                n_members=len(panel),genes="|".join(panel),panel_kind=panel_kind(panel,minimum),
                included_in_class_analysis=len(panel)>=minimum,included_in_singleton_sensitivity=bool(panel)))
        for key in keys:
            cohort=lookup[key]
            print(f"Harmonization {scope}: {cohort.dataset} {cohort.group}; common background {len(common)}",flush=True)
            definitions={"observed":references[key],"harmonized_classes":class_panels,
                         "harmonized_with_singletons":all_panels}
            rows,diffs,members,effects,donors=evaluate(cohort,targets,definitions,masks[key],backgrounds[key],
                {"study_measured":set(backgrounds[key]),"common_intersection":common},margins,draws,seed,scope)
            for name,items in (("02_harmonized_curves",rows),("02_fraction_changes",diffs),
                ("02_background_members",members),("02_target_effects",effects),("02_target_donors",donors)):
                outputs[name].extend(items)
            outputs["02_design_audit"].append(dict(**cohort.key,scope=scope,n_total=len(cohort.donors),
                n_common_donors=int(masks[key].sum()),n_study_background=len(backgrounds[key]),
                n_shared_background=len(common),excluded_union="|".join(sorted(excluded)),
                donors="|".join(d for d,keep in zip(cohort.donors,masks[key]) if keep),
                cross_study_donor_independence="unresolved_no_pooling"))
    expected = {("Leitner","AD"):[2157,936,509,246],("Marta-Ariza","EOAD"):[1473,656,444,266],
                ("Marta-Ariza","LOAD"):[1615,789,547,339],("Drummond 2022","eoAD"):[1480,522,294,151]}
    for key, counts in expected.items():
        rows=[r for r in outputs["01_margin_curves"] if (r["dataset"],r["group"])==key]
        assert rows[0]["n_proteins"]==counts[0],key
        for margin,count in zip((0,.1,.2),counts[1:]):
            assert next(r["count"] for r in rows if r["margin_log2"]==margin)==count,(key,margin)
    for name in ("01_margin_curves","02_harmonized_curves"):
        groups={}
        for row in outputs[name]:
            key=tuple(row[k] for k in ("dataset","group","scope","definition","background"))
            groups.setdefault(key,[]).append(row)
        for block in groups.values():
            block.sort(key=lambda r:r["margin_log2"])
            assert all(a["count"]>=b["count"] for a,b in zip(block,block[1:]))
    return outputs
