"""Simulations conditional on observed joint donor covariance and known truths."""
import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import expit
from scipy.stats import binom, t

from .io import application_matrix, rng_for
from .stats import holm, rate_summary


def psd_root(covariance):
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 1e-10*max(1., float(np.max(np.abs(values))))
    if values.min() < -tolerance:
        raise ValueError("Empirical covariance has a material negative eigenvalue")
    # Clip floating-point roundoff only; retain rank deficiency.
    return vectors*np.sqrt(np.maximum(values, 0.))[None, :]


def build_models(cohorts, protocol):
    models = []
    summaries, covrows, coeffrows = [], [], []
    minimum = protocol["simulation"]["calibration_minimum_donors"]
    shrinkage = protocol["simulation"]["diagonal_shrinkage"]
    for app in protocol["applications"]:
        genes, refs, coefficients = application_matrix(app)
        candidates = []
        for c in cohorts:
            delta, adjacent = c.matrix(genes), c.matrix(genes, "adjacent")
            mask = np.isfinite(delta).all(axis=1) & np.isfinite(adjacent).all(axis=1)
            candidates.append((int(mask.sum()), c, mask, delta, adjacent))
        n, cohort, mask, delta, adjacent = max(candidates, key=lambda item: item[0])
        base = dict(application=app["name"], **cohort.key, source_complete_donors=n,
                    proteins=len(genes), joint_dimensions=2*len(genes),
                    source_selection="largest_complete_donor_count_input_order_breaks_ties")
        if n < minimum:
            summaries.append(dict(**base, status="not_estimable", reason="insufficient_calibration_donors"))
            continue
        delta, adjacent = delta[mask], adjacent[mask]
        joint = np.column_stack([delta, adjacent])
        means = joint.mean(axis=0)
        residual = joint-means
        covariance = residual.T @ residual/(n-1)
        if np.any(np.diag(covariance) <= 1e-14):
            summaries.append(dict(**base, status="not_estimable", reason="constant_calibration_feature"))
            continue
        shrunk = (1-shrinkage)*covariance+shrinkage*np.diag(np.diag(covariance))
        p = len(genes)
        marginal = np.sqrt(np.diag(covariance))
        correlation = covariance/np.outer(marginal, marginal)
        eigen = np.linalg.eigvalsh(covariance)
        source_plaque = adjacent+(delta-delta.mean(axis=0))
        model = dict(application=app["name"], genes=genes, references=refs, coefficients=coefficients,
            target_indices=[genes.index(g) for g in app["targets"]],
            adjacent_means=adjacent.mean(axis=0), residual=residual*math.sqrt(n/(n-1)),
            covariance=covariance, correlation=correlation,
            roots={"gaussian_empirical": psd_root(covariance), "gaussian_shrinkage": psd_root(shrunk)},
            source_n=n, source_dataset=cohort.dataset, source_group=cohort.group,
            plaque_center=float(source_plaque.mean()), plaque_scale=float(source_plaque.std(ddof=1)))
        models.append(model)
        offdiag = correlation[:p, :p][np.triu_indices(p, 1)]
        summaries.append(dict(**base, status="estimable", reason="", genes="|".join(genes),
            source_donors="|".join(np.array(cohort.donors)[mask]),
            diagonal_shrinkage=shrinkage, covariance_min_eigenvalue=float(eigen.min()),
            covariance_max_eigenvalue=float(eigen.max()), covariance_rank=int(np.linalg.matrix_rank(covariance)),
            regional_correlation_min=float(offdiag.min()), regional_correlation_median=float(np.median(offdiag)),
            regional_correlation_max=float(offdiag.max()), plaque_scale=model["plaque_scale"],
            scope="conditional_on_measured_complete_donor_empirical_distribution"))
        variable_names = ["D:"+g for g in genes]+["A:"+g for g in genes]
        for i, a in enumerate(variable_names):
            for j, b in enumerate(variable_names):
                covrows.append(dict(application=app["name"], variable_a=a, variable_b=b,
                    sample_covariance=float(covariance[i, j]), sample_correlation=float(correlation[i, j]),
                    shrunk_covariance=float(shrunk[i, j])))
        for i, gene in enumerate(genes):
            for j, label in enumerate(refs):
                coeffrows.append(dict(application=app["name"], gene=gene, reference=label,
                                      coefficient=float(coefficients[i, j])))
    return models, {"10_covariance_models": summaries, "11_joint_covariance": covrows,
                    "12_contrast_coefficients": coeffrows}


def truth_means(model):
    p = len(model["genes"])
    c = model["coefficients"]
    zero = np.zeros(p)
    shifted = zero.copy()
    shifted[model["target_indices"]] = .3
    # Orthogonal to the designated primary contrast, but nonzero for another.
    # This construction works for overlapping and linearly dependent panels.
    primary = c[:, 0]
    projected = c-primary[:, None]*(primary @ c/(primary @ primary))[None, :]
    best = int(np.argmax(np.linalg.norm(projected, axis=0)))
    partial = projected[:, best]
    maximum = np.max(np.abs(partial @ c))
    if maximum <= 1e-12:
        raise ValueError("Distinct reference effects cannot be generated by these contrast coefficients")
    partial = partial*.3/maximum
    heterogeneous = np.linspace(-.3, .3, p)
    heterogeneous[model["target_indices"]] = .2
    return {"all_null": zero, "target_shift": shifted,
            "partial_null": partial, "reference_heterogeneity": heterogeneous}


def generate(model, generator, rng, shape):
    if generator == "donor_resampling":
        index = rng.integers(0, model["source_n"], size=shape)
        return model["residual"][index]
    root = model["roots"][generator]
    return rng.normal(size=(*shape, root.shape[0])) @ root.T


def calibrations(model, protocol, seed):
    config = protocol["simulation"]
    intercepts, rows, truths = {}, [], []
    for truth, means in truth_means(model).items():
        effects = means @ model["coefficients"]
        for j, label in enumerate(model["references"]):
            truths.append(dict(application=model["application"], truth=truth, reference=label,
                true_effect=float(effects[j]), is_true_null=bool(abs(effects[j]) < 1e-10),
                genes="|".join(model["genes"]), population_means="|".join(f"{x:.17g}" for x in means)))
        for generator in config["generators"]:
            rng = rng_for(seed, "mnar_calibration", model["application"], generator, truth)
            residual = generate(model, generator, rng, (config["calibration_draws"],))
            p = len(means)
            plaque = residual[:, :p]+means+residual[:, p:]+model["adjacent_means"]
            z = (plaque-model["plaque_center"])/model["plaque_scale"]
            for mechanism, rate in config["missingness"]:
                if mechanism != "MNAR":
                    continue
                lo, hi = -40., 40.
                for _ in range(45):
                    middle = (lo+hi)/2
                    expected = expit(middle-config["mnar_slope"]*z).mean()
                    if expected < rate:
                        lo = middle
                    else:
                        hi = middle
                intercept = (lo+hi)/2
                intercepts[generator, truth, rate] = intercept
                rows.append(dict(application=model["application"], generator=generator, truth=truth,
                    requested_missing_fraction=rate, intercept=intercept, slope=config["mnar_slope"],
                    calibrated_fraction=float(expit(intercept-config["mnar_slope"]*z).mean()),
                    calibration_draws=config["calibration_draws"],
                    definition="global_logistic_dropout_on_standardized_latent_plaque_intensity"))
    return intercepts, rows, truths


def fit_batch(delta, observed, coefficients, minimum, alpha):
    complete = observed.all(axis=2)
    counts = complete.sum(axis=1)
    scores = delta @ coefficients
    denominator = np.maximum(counts, 1)
    mean = (scores*complete[:, :, None]).sum(axis=1)/denominator[:, None]
    squares = (((scores-mean[:, None, :])**2)*complete[:, :, None]).sum(axis=1)
    se = np.sqrt(squares/np.maximum(counts-1, 1)[:, None]/denominator[:, None])
    usable = (counts >= minimum) & np.isfinite(se).all(axis=1) & (se > 1e-14).all(axis=1)
    mean, se, df = mean[usable], se[usable], counts[usable]-1
    pvalues = 2*t.sf(np.abs(mean/se), df[:, None])
    nominal = t.ppf(1-alpha/2, df)[:, None]*se
    joint = t.ppf(1-alpha/(2*coefficients.shape[1]), df)[:, None]*se
    return counts, usable, mean, se, pvalues, nominal, joint


def performance(base, effects, estimates, pvalues, nominal, joint, alpha):
    n, m = estimates.shape
    truth_null = np.abs(effects) < 1e-10
    error = estimates-effects[None, :]
    indices = np.arange(n)
    selected = pvalues.argmin(axis=1) if n else np.zeros(0, int)
    selected_rejection = np.zeros((n, m), bool)
    selected_rejection[indices, selected] = pvalues[indices, selected] < alpha
    primary_rejection = np.zeros((n, m), bool)
    primary_rejection[:, 0] = pvalues[:, 0] < alpha
    adjusted = holm(pvalues)
    plans = [
        ("fixed_primary_nominal", primary_rejection, error[:, :1], nominal[:, :1], "primary_interval"),
        ("selected_min_p_nominal", selected_rejection, error[indices, selected, None], nominal[indices, selected, None], "selected_reference_own_truth"),
        ("all_nominal", pvalues < alpha, error, nominal, "joint_family"),
        ("all_holm", adjusted < alpha, error, None, "no_interval_defined_for_this_method"),
        ("all_bonferroni", pvalues < alpha/m, error, joint, "joint_family")]
    rows = []
    for method, rejected, errors, halfwidth, coverage_scope in plans:
        row = dict(**base, method=method, n_estimable=n, reference_count=m,
                   true_null_count=int(truth_null.sum()), true_nonnull_count=int((~truth_null).sum()),
                   coverage_scope=coverage_scope,
                   bias=float(errors.mean()) if n else None,
                   rmse=float(np.sqrt(np.mean(errors**2))) if n else None,
                   mean_interval_width=float(2*halfwidth.mean()) if halfwidth is not None and n else None)
        for label, mask in [("false_family", truth_null), ("any_true_rejection", ~truth_null)]:
            if mask.any():
                events = rejected[:, mask].any(axis=1)
                row.update({label+"_"+k: v for k, v in rate_summary(events).items()})
                row[label+"_per_requested_repeat"] = float(events.sum()/base["repetitions"])
            else:
                row[label+"_rate"] = None
        if halfwidth is not None:
            covered = (np.abs(errors) <= halfwidth).all(axis=1)
            row.update({"coverage_"+k: v for k, v in rate_summary(covered).items()})
            row["success_and_coverage_per_requested_repeat"] = float(covered.sum()/base["repetitions"])
        rows.append(row)
    details = []
    for j in range(m):
        row = dict(**base, reference_index=j, true_effect=float(effects[j]),
                   is_true_null=bool(truth_null[j]), n_estimable=n,
                   bias=float(error[:, j].mean()) if n else None,
                   rmse=float(np.sqrt(np.mean(error[:, j]**2))) if n else None)
        for label, events in [
            ("nominal_rejection", pvalues[:, j] < alpha),
            ("holm_rejection", adjusted[:, j] < alpha),
            ("bonferroni_rejection", pvalues[:, j] < alpha/m),
            ("nominal_coverage", np.abs(error[:, j]) <= nominal[:, j]),
            ("bonferroni_coverage", np.abs(error[:, j]) <= joint[:, j]),
            ("selected", selected == j)]:
            row.update({label+"_"+k: v for k, v in rate_summary(events).items()})
        details.append(row)
    return rows, details


def scenario_batches(model, config, rng, repetitions, generator, means, n, mechanism, rate, intercept):
    """Draw residuals and missingness in the original 256-repeat batches."""
    p = len(means)
    for start in range(0, repetitions, 256):
        size = min(256, repetitions-start)
        residual = generate(model, generator, rng, (size, n))
        delta = residual[:, :, :p]+means
        if mechanism == "complete":
            observed = np.ones(delta.shape, bool)
        elif mechanism == "MCAR":
            observed = rng.random(delta.shape) >= rate
        else:
            plaque = delta+residual[:, :, p:]+model["adjacent_means"]
            z = (plaque-model["plaque_center"])/model["plaque_scale"]
            probability = expit(intercept-config["mnar_slope"]*z)
            observed = rng.random(delta.shape) >= probability
        yield delta, observed


def one_scenario(task):
    model, config, seed, repetitions, alpha, intercepts, generator, truth, n, mechanism, rate = task
    means = truth_means(model)[truth]
    p = len(means)
    rng = rng_for(seed, "empirical_simulation", model["application"], generator, truth, n, mechanism, rate)
    collected = [[] for _ in range(5)]
    counts_all, usable_all = [], []
    missing_total, cells_total = 0, 0
    intercept = intercepts.get((generator, truth, rate))
    for delta, observed in scenario_batches(model, config, rng, repetitions, generator, means, n,
                                            mechanism, rate, intercept):
        counts, usable, *results = fit_batch(delta, observed, model["coefficients"],
                                            config["minimum_complete_donors"], alpha)
        counts_all.append(counts)
        usable_all.append(usable)
        for storage, result in zip(collected, results):
            storage.append(result)
        missing_total += int((~observed).sum())
        cells_total += observed.size
    estimates, se, pvalues, nominal, joint = [np.concatenate(x, axis=0) for x in collected]
    counts, usable = np.concatenate(counts_all), np.concatenate(usable_all)
    base = dict(application=model["application"], generator=generator, truth=truth,
        n=n, missingness=mechanism, requested_missing_fraction=rate, repetitions=repetitions,
        precision_note="fewer_than_100_estimable_repeats" if usable.sum() < 100 else "",
        calibration_source_n=model["source_n"], calibration_source_dataset=model["source_dataset"],
        calibration_source_group=model["source_group"])
    rows, details = performance(base, means @ model["coefficients"], estimates, pvalues, nominal, joint, alpha)
    for row in details:
        row["reference"] = model["references"][row.pop("reference_index")]
    expected = 1. if mechanism == "complete" else float(binom.sf(
        config["minimum_complete_donors"]-1, n, (1-rate)**p)) if mechanism == "MCAR" else None
    availability = dict(**base, n_estimable=int(usable.sum()), n_unavailable=int((~usable).sum()),
        estimable_fraction=float(usable.mean()), actual_missing_fraction=missing_total/cells_total,
        mean_complete_donors=float(counts.mean()), minimum_complete_donors=config["minimum_complete_donors"],
        analytic_count_eligibility_fraction=expected,
        variance_degenerate_repeats=int(((counts >= config["minimum_complete_donors"]) & ~usable).sum()))
    availability.update({"estimability_"+k: v for k, v in rate_summary(usable).items()})
    return availability, rows, details


def simulate(models, protocol, repetitions, seed, workers, progress=print):
    tables = {k: [] for k in ["13_simulation_truths", "14_missingness_calibration",
                              "15_simulation_availability", "16_simulation_methods", "17_simulation_by_reference"]}
    tasks = []
    config = protocol["simulation"]
    for model in models:
        progress("Calibrating empirical simulations: " + model["application"])
        intercepts, rows, truths = calibrations(model, protocol, seed)
        tables["14_missingness_calibration"].extend(rows)
        tables["13_simulation_truths"].extend(truths)
        for generator, truth, n, (mechanism, rate) in itertools.product(
                config["generators"], truth_means(model), config["sample_sizes"], config["missingness"]):
            tasks.append((model, config, seed, repetitions, protocol["alpha"], intercepts,
                          generator, truth, n, mechanism, rate))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (availability, performance_rows, details) in enumerate(pool.map(one_scenario, tasks), 1):
            tables["15_simulation_availability"].append(availability)
            tables["16_simulation_methods"].extend(performance_rows)
            tables["17_simulation_by_reference"].extend(details)
            if index % 10 == 0 or index == len(tasks):
                progress(f"Empirical simulations: {index}/{len(tasks)} scenarios complete")
    return tables
