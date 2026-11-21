"""Held-out simulations of the signal-to-span ratio and the v9 interval rule."""
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import gzip
from io import TextIOWrapper
from itertools import product

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import t

from paper4.reference_extensions.io import application_matrix, rng_for
from paper4.reference_extensions.simulation import (
    calibrations, fit_batch, generate, scenario_batches, truth_means,
)
from paper4.reference_extensions.stats import rate_summary, summarize_columns


@dataclass
class Scenario:
    number: int
    model: dict
    partition: str
    generator: str
    truth: str
    means: np.ndarray
    donors: int
    missingness: str
    missing_rate: float
    intercept: float | None
    repetitions: int


@dataclass
class Repeats:
    scenario: Scenario
    counts: np.ndarray
    v9_available: np.ndarray
    eligible: np.ndarray
    region: np.ndarray
    region_low: np.ndarray
    contrasts: np.ndarray
    min_low: np.ndarray
    score: np.ndarray
    missing_fraction: float

    def truth(self, margin):
        means = self.scenario.means
        model = self.scenario.model
        return bool(means[model["target_indices"]].mean() > 0
                    and (means @ model["coefficients"]).min() > margin + 1e-12)

    def ssr_gate(self):
        return self.eligible & (self.region_low > 0) & np.isfinite(self.score)


def boundary_means(model, profile, minimum_effect, span, regional_mean):
    """Set the true weakest contrast and span while retaining panel overlap."""
    coefficient = model["coefficients"]
    targets = model["target_indices"]
    references = np.ones(len(model["genes"]), dtype=bool)
    references[targets] = False
    means = np.zeros(len(model["genes"]))
    direction = np.linspace(-1, 1, references.sum())
    means[references] = direction if profile == "forward" else direction[::-1]
    current_span = np.ptp(means @ coefficient)
    if current_span <= 1e-12:
        raise ValueError("The boundary profile must distinguish the reference means")
    means *= span / current_span
    means[targets] = regional_mean
    means[references] += (means @ coefficient).min() - minimum_effect
    truth = means @ coefficient
    np.testing.assert_allclose([truth.min(), np.ptp(truth)], [minimum_effect, span], atol=1e-12)
    return means


def mnar_intercepts(model, config, residual, means, rates):
    proteins = len(means)
    plaque = residual[:, :proteins] + means + residual[:, proteins:] + model["adjacent_means"]
    standardized = config["mnar_slope"] * (plaque - model["plaque_center"]) / model["plaque_scale"]
    return {
        rate: brentq(lambda intercept: expit(intercept-standardized).mean()-rate, -40, 40)
        for rate in rates
    }


def make_scenarios(models, original, protocol):
    config = original["simulation"]
    settings = protocol["ssr"]
    scenarios = []

    def add(model, partition, generator, truth, means, donors, mechanism, rate, intercept, repetitions):
        scenarios.append(Scenario(len(scenarios)+1, model, partition, generator, truth,
                                  means, donors, mechanism, rate, intercept, repetitions))

    for model in models:
        intercepts, _, _ = calibrations(model, original, protocol["v9_seed"])
        for generator, (truth, means), donors, (mechanism, rate) in product(
                config["generators"], truth_means(model).items(), config["sample_sizes"], config["missingness"]):
            add(model, "v9_replay", generator, truth, means, donors, mechanism, rate,
                intercepts.get((generator, truth, rate)), settings["v9_repetitions"])

        positive = list(product(settings["minimum_true_effect_log2"], settings["true_span_log2"],
                                [settings["regional_mean_log2"]]))
        challenges = [(minimum, settings["true_span_log2"][1], -0.05)
                      for minimum in settings["negative_regional_challenge"]]
        rates = [rate for mechanism, rate in settings["missingness"] if mechanism == "MNAR"]
        for partition, plan in settings["partitions"].items():
            generator, profile = plan["generator"], plan["profile"]
            rng = rng_for(protocol["seed"], "boundary_mnar", model["application"], partition)
            residual = generate(model, generator, rng, (config["calibration_draws"],))
            for minimum, span, region in positive + challenges:
                truth = f"min_{minimum:.2f}_span_{span:.2f}_region_{region:.2f}"
                means = boundary_means(model, profile, minimum, span, region)
                intercepts = mnar_intercepts(model, config, residual, means, rates)
                for donors, (mechanism, rate) in product(settings["sample_sizes"], settings["missingness"]):
                    add(model, partition, generator, truth, means, donors, mechanism, rate,
                        intercepts[rate] if mechanism == "MNAR" else None, settings["boundary_repetitions"])
    return scenarios


def repeat_estimates(scenario, original, protocol):
    model = scenario.model
    config = original["simulation"]
    if scenario.partition == "v9_replay":
        rng = rng_for(protocol["v9_seed"], "empirical_simulation", model["application"],
                      scenario.generator, scenario.truth, scenario.donors, scenario.missingness, scenario.missing_rate)
    else:
        rng = rng_for(protocol["seed"], "ssr_boundary", model["application"], scenario.partition,
                      scenario.generator, scenario.truth, scenario.donors, scenario.missingness, scenario.missing_rate)
    repetitions = scenario.repetitions
    nrefs = len(model["references"])
    counts = np.empty(repetitions, dtype=int)
    relative_available = np.zeros(repetitions, dtype=bool)
    eligible = np.zeros(repetitions, dtype=bool)
    region, region_low, min_low, score = [np.full(repetitions, np.nan) for _ in range(4)]
    contrasts = np.full((repetitions, nrefs), np.nan)
    start = missing = entries = 0
    for delta, observed in scenario_batches(
            model, config, rng, repetitions, scenario.generator, scenario.means, scenario.donors,
            scenario.missingness, scenario.missing_rate, scenario.intercept):
        count, available, means, se, *_ = fit_batch(
            delta, observed, model["coefficients"], config["minimum_complete_donors"], original["alpha"])
        stop = start + len(count)
        counts[start:stop] = count
        relative_available[start:stop] = available
        index = start + np.flatnonzero(available)
        contrasts[index] = means
        if len(index):
            complete = observed[available].all(axis=2)
            target = delta[available][:, :, model["target_indices"]].mean(axis=2)
            size = count[available]
            avg = (target * complete).sum(axis=1) / size
            var = (((target-avg[:, None])**2) * complete).sum(axis=1) / (size-1)
            target_se = np.sqrt(var/size)
            multiplier = t.ppf(1-original["alpha"]/(2*(nrefs+1)), size-1)
            region[index] = avg
            region_low[index] = avg-multiplier*target_se
            min_low[index] = (means-multiplier[:, None]*se).min(axis=1)
            eligible[index] = (size >= original["minimum_donors"]) & (target_se > 1e-14)
            span = np.ptp(means, axis=1)
            score[index] = np.divide(means.min(axis=1), span, out=np.full(len(index), np.nan),
                                     where=span > protocol["ssr"]["span_tolerance_log2"])
        missing += int((~observed).sum())
        entries += observed.size
        start = stop
    return Repeats(scenario, counts, relative_available, eligible, region, region_low, contrasts,
                   min_low, score, missing/entries)


def scenario_row(repeats):
    s, model = repeats.scenario, repeats.scenario.model
    truths = s.means @ model["coefficients"]
    return dict(
        scenario=s.number, partition=s.partition, application=model["application"], generator=s.generator,
        truth=s.truth, n=s.donors, missingness=s.missingness, requested_missing_fraction=s.missing_rate,
        actual_missing_fraction=repeats.missing_fraction, repetitions=s.repetitions,
        calibration_study=model["source_dataset"], calibration_group=model["source_group"],
        calibration_donors=model["source_n"], mnar_intercept=s.intercept,
        true_target_regional_mean=s.means[model["target_indices"]].mean(),
        true_min_relative=truths.min(), true_reference_span=np.ptp(truths),
        reference_order="|".join(model["references"]), true_relative_means="|".join(map(str, truths)),
        gene_order="|".join(model["genes"]), true_gene_means="|".join(map(str, s.means)),
        n_estimable_v9=int(repeats.v9_available.sum()), n_estimable_full_decision=int(repeats.eligible.sum()),
        n_defined_ssr=int((repeats.eligible & np.isfinite(repeats.score)).sum()),
    )


def write_repeats(writer, result, max_references):
    pad = [None] * (max_references-result.contrasts.shape[1])

    def numeric(row):
        return [format(float(num), ".17g") if np.isfinite(num) else "" for num in row]

    for i, (n, available, eligible, region, lcb, lowest_lcb, score, contrasts) in enumerate(zip(
            result.counts, result.v9_available, result.eligible, result.region, result.region_low,
            result.min_low, result.score, result.contrasts), start=1):
        writer.writerow([result.scenario.number, i, int(n), int(available), int(eligible),
                         *numeric((region, lcb, lowest_lcb, score)), *numeric(contrasts), *pad])


def choose_cutoff(results, margin, desired_fpr):
    negatives = [r for r in results if r.scenario.partition == "calibration" and not r.truth(margin)]
    denominator = sum(int(r.eligible.sum()) for r in negatives)
    if not denominator:
        raise ValueError("The calibration design must include estimable null repeats")
    scores = np.sort(np.concatenate([r.score[r.ssr_gate()] for r in negatives]))
    allowed = int(np.floor(desired_fpr*denominator))
    cutoff = float(scores[-allowed-1]) if allowed < len(scores) else -np.inf
    return cutoff, denominator, int((scores > cutoff).sum())


def decisions(result, margin, cutoff):
    regional = result.eligible & (result.region_low > 0)
    return {
        "v9_simultaneous": regional & (result.min_low > margin),
        "SSR_cutoff_1": result.ssr_gate() & (result.score > 1.0),
        "SSR_calibrated": result.ssr_gate() & (result.score > cutoff),
    }


def roc_points(results, margin):
    score = np.concatenate([np.where(r.ssr_gate(), r.score, -np.inf)[r.eligible] for r in results])
    truth = np.concatenate([np.full(int(r.eligible.sum()), r.truth(margin)) for r in results]).astype(bool)
    if not truth.any() or truth.all():
        return [], None
    order = np.argsort(-score, kind="stable")
    score, truth = score[order], truth[order]
    ends = np.flatnonzero(np.r_[score[1:] != score[:-1], True])
    cumulative = np.cumsum(truth)[ends]
    tpr = np.r_[0, cumulative/truth.sum()]
    fpr = np.r_[0, (ends+1-cumulative)/(~truth).sum()]
    auc = float(np.trapezoid(tpr, fpr))
    keep = np.unique(np.linspace(0, len(fpr)-1, min(601, len(fpr))).astype(int))
    rows = [dict(margin_log2=margin, false_positive_rate=fpr[i], sensitivity=tpr[i]) for i in keep]
    return rows, auc


def evaluate(results, original, protocol):
    settings = protocol["ssr"]
    partitions = ("v9_replay", "calibration", "held_out")
    applications = sorted({r.scenario.model["application"] for r in results})
    calibration, by_scenario, summary, roc = [], [], [], []
    for margin in original["margins_log2"]:
        cutoff, eligible_negatives, false_positives = choose_cutoff(
            results, margin, settings["desired_false_positive_rate"])
        calibration.append(dict(
            margin_log2=margin, cutoff=cutoff,
            rule="SSR_strictly_greater_than_cutoff" if np.isfinite(cutoff) else "all_finite_scores_pass",
            requested_calibration_fpr=settings["desired_false_positive_rate"],
            calibration_estimable_negatives=eligible_negatives, calibration_false_positives=false_positives,
            calibration_fpr=false_positives/eligible_negatives))
        counts = []
        for r in results:
            for method, predicted in decisions(r, margin, cutoff).items():
                conditional = rate_summary(predicted[r.eligible])
                counts.append(dict(scenario=r.scenario.number, partition=r.scenario.partition,
                    application=r.scenario.model["application"], margin_log2=margin, method=method,
                    true_robust_above_margin=r.truth(margin), requested=r.scenario.repetitions,
                    experiment_event="true_positive" if r.truth(margin) else "false_positive",
                    called_count=conditional["count"], eligible_count=conditional["denominator"],
                    call_rate_conditional=conditional["rate"], wilson_low=conditional["low"],
                    wilson_high=conditional["high"], call_rate_all_repeats=float(predicted.mean())))
        by_scenario.extend(counts)

        for partition in partitions:
            part = [r for r in results if r.scenario.partition == partition]
            for application in ("All", *applications):
                group = [r for r in part if application == "All" or r.scenario.model["application"] == application]
                roc_rows, auc = roc_points(group, margin)
                for row in roc_rows:
                    roc.append(dict(partition=partition, application=application, **row))
                matching = [r for r in counts if r["partition"] == partition
                            and (application == "All" or r["application"] == application)]
                for method in ("v9_simultaneous", "SSR_cutoff_1", "SSR_calibrated"):
                    methods = [r for r in matching if r["method"] == method]
                    row = dict(partition=partition, application=application, margin_log2=margin,
                               method=method, ssr_roc_auc=auc if method == "SSR_calibrated" else None)
                    for label, truth in (("sensitivity", True), ("false_positive_rate", False)):
                        cells = [r for r in methods if r["true_robust_above_margin"] == truth]
                        called, eligible, requested = (
                            sum(r[k] for r in cells) for k in ("called_count", "eligible_count", "requested"))
                        row.update({
                            label+"_numerator": called, label+"_eligible": eligible,
                            label+"_requested": requested, label+"_conditional": called/eligible if eligible else None,
                            label+"_all_repeats": called/requested if requested else None,
                            label+"_worst_estimable_scenario": max(
                                (r["call_rate_conditional"] for r in cells if r["eligible_count"] >= 100),
                                default=None),
                        })
                    summary.append(row)
    return {
        "ssr_cutoffs": calibration, "ssr_performance": summary, "ssr_by_scenario": by_scenario,
        "ssr_roc": roc, "ssr_scenarios": [scenario_row(r) for r in results],
    }


def observed_ssr(study_apps, cutoffs, original, tolerance):
    """Record all measured diagnostic ratios and retain unavailable families."""
    results = []
    for cohorts, applications, calibration_scope in study_apps:
        for app in applications:
            genes, refs, coefficients = application_matrix(app)
            for cohort in cohorts:
                source = cohort.matrix(genes)
                values = source[np.isfinite(source).all(axis=1)]
                n = len(values)
                base = dict(**cohort.key, application=app["name"], n=n, references="|".join(refs),
                            calibration_scope=calibration_scope)
                scores = (None, None, None)
                region_low, relative_low = None, None
                if n >= original["minimum_donors"]:
                    regional = values[:, :len(app["targets"])].mean(axis=1)
                    fit = summarize_columns(np.column_stack([regional, values @ coefficients]), original["alpha"])
                    effects = fit["mean"][1:]
                    minimum, span = effects.min(), np.ptp(effects)
                    scores = (minimum, span, minimum/span if span > tolerance else None)
                    if fit["valid"].all():
                        region_low = fit["simultaneous_low"][0]
                        relative_low = fit["simultaneous_low"][1:].min()
                minimum, span, ratio = scores
                for row in cutoffs:
                    margin = row["margin_log2"]
                    ready = region_low is not None
                    ratio_ready = ready and ratio is not None
                    results.append(dict(**base, margin_log2=margin, minimum_relative_mean=minimum,
                        reference_span_log2=span, ssr=ratio, regional_simultaneous_low=region_low,
                        relative_minimum_simultaneous_low=relative_low, simulated_ssr_cutoff=row["cutoff"],
                        v9_simultaneous_pass=bool(region_low > 0 and relative_low > margin) if ready else None,
                        ssr_cutoff_1_pass=bool(region_low > 0 and ratio > 1) if ratio_ready else None,
                        ssr_calibrated_pass=bool(region_low > 0 and ratio > row["cutoff"]) if ratio_ready else None))
    return results


def run_ssr(models, original, protocol, workers, output):
    scenarios = make_scenarios(models, original, protocol)
    max_references = max(len(model["references"]) for model in models)
    results = []
    path = output / "ssr_repeats.csv.gz"
    with path.open("wb") as raw, gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=4, mtime=0) as archive, \
            TextIOWrapper(archive, encoding="utf-8", newline="") as stream, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["scenario", "repeat", "complete_donors", "estimable_v9_relative_only",
                         "estimable_full_decision", "regional", "regional_simultaneous_low",
                         "minimum_relative_simultaneous_low", "ssr",
                         *[f"reference_{j}_estimate" for j in range(1, max_references+1)]])
        for i, result in enumerate(pool.map(lambda s: repeat_estimates(s, original, protocol), scenarios), 1):
            write_repeats(writer, result, max_references)
            results.append(result)
            if i % 120 == 0 or i == len(scenarios):
                print(f"Simulation scenarios complete: {i}/{len(scenarios)}", flush=True)
    return evaluate(results, original, protocol)
