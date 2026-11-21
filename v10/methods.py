"""Posterior reference choice, GLS sensitivity and compositional identity checks."""
import math

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.stats import t

from paper4.reference_extensions.io import application_matrix, rng_for
from paper4.reference_extensions.stats import rate_summary, summarize_columns


def mean_posterior(values, sd, prior_precision):
    """Normal–inverse-Wishart model and marginal multivariate t for the means."""
    n, d = values.shape
    avg = values.mean(axis=0)
    scatter = (values-avg).T @ (values-avg)
    precision = n+prior_precision
    posterior_scale = sd**2*np.eye(d)+scatter+n*prior_precision/precision*np.outer(avg, avg)
    df = n+3  # inverse-Wishart prior df=d+2, with E[Sigma]=sd**2 I
    return n*avg/precision, posterior_scale/(precision*df), df


def mixture_quantile(probability, weights, mean, scale, df):
    limits = mean + scale*t.ppf(probability, df)
    if np.ptp(limits) < 1e-14:
        return float(limits.mean())
    pad = 1e-12*max(1.0, float(np.max(np.abs(limits))))
    return brentq(lambda x: weights @ t.cdf((x-mean)/scale, df)-probability,
                  limits.min()-pad, limits.max()+pad)


def joint_tail(mean, shape, df, thresholds, marginal, draws, rng):
    """Condition on the least probable marginal event before sampling the remaining t scores."""
    j = int(np.argmin(marginal))
    weight = float(marginal[j])
    if weight == 0:
        return 0., 0, 0., 0., 0.
    keep = np.arange(len(mean)) != j
    sigma = math.sqrt(shape[j, j])
    z = t.isf(weight*(1-rng.random(draws)), df)
    condition = shape[keep, j]/sigma
    residual = shape[np.ix_(keep, keep)] - np.outer(condition, condition)
    standard = rng.standard_normal((draws, int(keep.sum()))) @ np.linalg.cholesky(residual).T
    conditional_mean = mean[keep] + z[:, None]*condition
    values = conditional_mean + standard*np.sqrt((df+z*z)/rng.chisquare(df+1, draws))[:, None]
    rates = rate_summary((values > thresholds[keep]).all(axis=1))
    return weight*rates["rate"], rates["count"], weight*rates["mcse"], weight*rates["low"], weight*rates["high"]


def bayesian_results(base, values, refs, margins, settings, seed):
    fits, mixtures, joints = [], [], []
    draws = settings["draws"]
    weights = {"equal_probabilities": np.full(len(refs), 1/len(refs))}
    for p in settings["primary_reference_probabilities"]:
        if not math.isclose(p, 1/len(refs)):
            weights[f"primary_probability_{p:g}"] = np.r_[p, np.full(len(refs)-1, (1-p)/(len(refs)-1))]

    for sd in settings["prior_sd_log2"]:
        mean, shape, df = mean_posterior(values, sd, settings["prior_mean_precision"])
        scale = np.sqrt(np.diag(shape))
        quantile = t.ppf(.975, df)
        key = dict(**base, prior_sd_log2=sd, posterior_df=df)
        # Use common random numbers across the prior-scale sensitivity analysis.
        rng = rng_for(seed, "joint_mean_posterior", base["dataset"], base["group"], base["application"])
        standard = rng.standard_normal((draws, len(mean)))
        simulated = mean + (standard @ np.linalg.cholesky(shape).T) / np.sqrt(
            rng.chisquare(df, draws)/df)[:, None]
        regional_pass = simulated[:, 0] > 0
        for margin in margins:
            thresholds = np.r_[0, np.full(len(refs), margin)]
            marginal = t.sf((thresholds-mean)/scale, df)
            ref_pass = simulated[:, 1:] > margin
            probability, hits, mcse, mc_low, mc_high = joint_tail(
                mean, shape, df, thresholds, marginal, draws,
                rng_for(seed, "joint_conditioned", base["dataset"], base["group"], base["application"], margin))
            relative_prob, _, _, _, _ = joint_tail(
                mean[1:], shape[1:, 1:], df, thresholds[1:], marginal[1:], draws,
                rng_for(seed, "relative_conditioned", base["dataset"], base["group"], base["application"], margin))
            lowest_marginal = float(marginal.min())
            if not 0 <= probability <= lowest_marginal:
                raise ValueError("Joint posterior probability exceeds its marginal bound")
            joints.append(dict(**key, margin_log2=margin, conditioned_draws=draws, conditioned_hits=hits,
                conditioning_comparison=["regional", *refs][int(np.argmin(marginal))],
                p_region_and_all_references=probability, p_all_relative=relative_prob,
                p_regional_increase=marginal[0], minimum_marginal_probability=lowest_marginal,
                mcse=mcse, monte_carlo_low=mc_low, monte_carlo_high=mc_high))
            for j, label in enumerate(["regional", *refs]):
                fits.append(dict(**key, comparison=label, margin_log2=margin, threshold=thresholds[j],
                    posterior_mean=mean[j], credible_low=mean[j]-quantile*scale[j],
                    credible_high=mean[j]+quantile*scale[j], p_above_threshold=marginal[j]))
            for choice, w in weights.items():
                mixture = float(w @ marginal[1:])
                if not marginal[1:].min()-1e-12 <= mixture <= marginal[1:].max()+1e-12:
                    raise ValueError("Posterior mixture escaped its component probability bounds")
                mixtures.append(dict(**key, margin_log2=margin, choice_prior=choice, primary_reference=refs[0],
                    reference_order="|".join(refs), posterior_choice_weights="|".join(f"{v:.17g}" for v in w),
                    weighting_assumption="common_joint_likelihood_choice_probabilities_equal_specified_prior",
                    mixture_mean=float(w @ mean[1:]),
                    mixture_credible_low=mixture_quantile(.025, w, mean[1:], scale[1:], df),
                    mixture_credible_high=mixture_quantile(.975, w, mean[1:], scale[1:], df),
                    p_selected_reference_above_margin=mixture,
                    p_region_and_selected_reference=float(w @ (regional_pass[:, None] & ref_pass).mean(axis=0))))
    return fits, mixtures, joints


def nonnegative_weights(reference, shrinkage):
    count = reference.shape[1]
    equal = np.full(count, 1/count)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    scale = covariance.diagonal().max()
    if scale == 0:
        return equal
    cov = ((1-shrinkage)*covariance + shrinkage*np.diag(covariance.diagonal())) / scale
    fit = minimize(lambda w: w @ cov @ w, equal, jac=lambda w: 2*cov @ w, method="SLSQP",
                   bounds=[(0, 1)]*count, constraints=[dict(type="eq", fun=lambda w: w.sum()-1,
                   jac=lambda w: np.ones_like(w))], options=dict(ftol=1e-12, maxiter=250))
    if not fit.success or abs(fit.x.sum()-1) > 1e-8 or fit.x.min() < -1e-10:
        raise ValueError("The nonnegative GLS fit failed: " + fit.message)
    weights = np.maximum(fit.x, 0)
    return weights/weights.sum()


def gls_results(base, target, reference, genes, donors, settings):
    n, count = reference.shape
    equal = np.full(count, 1/count)
    cov = np.atleast_2d(np.cov(reference, rowvar=False))
    condition = float(np.linalg.cond(cov))
    equal_reference = reference @ equal
    equal_score = target-equal_reference
    rows, weights, donor_rows = [], [], []
    sd = np.sqrt(cov.diagonal())
    correlation = cov/np.outer(sd, sd) if np.all(sd > 0) else None

    def record(method, per_donor_weights, expected_ref=None, expected_score=None):
        np.testing.assert_allclose(per_donor_weights.sum(axis=1), 1, atol=1e-9)
        ref = (reference*per_donor_weights).sum(axis=1)
        score = target-ref
        avg_w = per_donor_weights.mean(axis=0)
        variance_equivalent = float(avg_w @ correlation @ avg_w) if correlation is not None else 0.0
        rows.append(dict(**base, method=method, status="estimable", members=count,
            covariance_condition=condition, target_mean=target.mean(), reference_mean=ref.mean(),
            estimate_log2=score.mean(), change_from_equal_log2=(score-equal_score).mean(),
            reference_sd=ref.std(ddof=1), contrast_sd=score.std(ddof=1),
            independent_equivalent_standardized_references=1/variance_equivalent if variance_equivalent > 0 else None,
            negative_weights=int((avg_w < -1e-10).sum()),
            held_out_reference_rmse=np.sqrt(np.mean((ref-expected_ref)**2)) if expected_ref is not None else None,
            held_out_contrast_rmse=np.sqrt(np.mean((score-expected_score)**2)) if expected_score is not None else None))
        weights.extend(dict(**base, method=method, gene=gene, weight=avg_w[j],
                            minimum=per_donor_weights[:, j].min(), maximum=per_donor_weights[:, j].max())
                       for j, gene in enumerate(genes))
        donor_rows.extend(dict(**base, method=method, donor=donor, reference=ref[i], score=score[i],
                               weight_order="|".join(genes),
                               weights="|".join(f"{w:.17g}" for w in per_donor_weights[i]))
                          for i, donor in enumerate(donors))

    record("equal_weight", np.tile(equal, (n, 1)), (equal_reference.sum()-equal_reference)/(n-1),
           (equal_score.sum()-equal_score)/(n-1))
    if n > count and condition < settings["unconstrained_condition_limit"]:
        inv = np.linalg.solve(cov, np.ones(count))
        record("unconstrained_inverse_covariance", np.tile(inv/inv.sum(), (n, 1)))
    else:
        rows.append(dict(**base, method="unconstrained_inverse_covariance", status="covariance_unavailable",
                         members=count, covariance_condition=condition))

    loo = []
    expected_reference, expected_contrast = [], []
    for i in range(n):
        training = np.arange(n) != i
        weights_i = nonnegative_weights(reference[training], settings["diagonal_shrinkage"])
        ref_prediction = reference[training].mean(axis=0) @ weights_i
        loo.append(weights_i)
        expected_reference.append(ref_prediction)
        expected_contrast.append(target[training].mean()-ref_prediction)
    record("nonnegative_shrinkage_leave_one_donor_out", np.array(loo),
           np.array(expected_reference), np.array(expected_contrast))
    return rows, weights, donor_rows


def coda_results(base, adjacent, delta, target_count, donors, margins, alpha, family_size):
    ref_count = adjacent.shape[1]-target_count
    factor = math.sqrt(target_count*ref_count/(target_count+ref_count))
    after = adjacent+delta

    def ratio(logs):
        return logs[:, :target_count].mean(axis=1)-logs[:, target_count:].mean(axis=1)

    def closure(logs):
        return logs-logsumexp(logs*math.log(2), axis=1, keepdims=True)/math.log(2)

    score = ratio(delta)
    alr = ratio(after)-ratio(adjacent)
    closed_alr = ratio(closure(after))-ratio(closure(adjacent))
    clr = ratio(after-after.mean(axis=1, keepdims=True))-ratio(adjacent-adjacent.mean(axis=1, keepdims=True))
    coordinate = np.r_[np.full(target_count, factor/target_count), np.full(ref_count, -factor/ref_count)]
    ilr = after @ coordinate-adjacent @ coordinate
    reconstructed = np.column_stack([alr, closed_alr, clr, ilr/factor])
    for x in reconstructed.T:
        np.testing.assert_allclose(x, score, rtol=0, atol=2e-12)
    fit = summarize_columns(score, alpha, family_size)
    ilr_fit = summarize_columns(ilr, alpha, family_size)
    for level in ("mean", "simultaneous_low", "simultaneous_high"):
        np.testing.assert_allclose(ilr_fit[level]/factor, fit[level], rtol=0, atol=2e-12)
    np.testing.assert_allclose(ilr_fit["p"], fit["p"], rtol=1e-9, atol=2e-12)
    row = dict(**base, target_members=target_count, reference_members=ref_count,
               reference_relative_log2=score.mean(), ilr_scale=factor, ilr_balance_change_log2=ilr.mean(),
               ilr_balance_change_natural_log=ilr.mean()*math.log(2), ilr_rescaled_log2=ilr.mean()/factor,
               original_simultaneous_low=fit["simultaneous_low"][0],
               original_simultaneous_high=fit["simultaneous_high"][0],
               ilr_simultaneous_low=ilr_fit["simultaneous_low"][0],
               ilr_simultaneous_high=ilr_fit["simultaneous_high"][0],
               original_p=fit["p"][0], ilr_p=ilr_fit["p"][0],
               max_abs_reconstruction_error=np.max(np.abs(reconstructed-score[:, None])))
    for margin in margins:
        key = str(margin).replace(".", "_")
        row[f"ilr_margin_{key}"] = factor*margin
        if bool(fit["simultaneous_low"][0] > margin) != bool(ilr_fit["simultaneous_low"][0] > factor*margin):
            raise ValueError("Rescaling CoDA units changed a margin decision")
    details = [dict(**base, donor=donor, original_reference_relative_log2=score[i], alr_log2=alr[i],
                    closed_alr_log2=closed_alr[i], clr_contrast_log2=clr[i], ilr_log2=ilr[i],
                    ilr_rescaled=ilr[i]/factor) for i, donor in enumerate(donors)]
    return row, details


def observed_methods(cohorts, external, original, protocol):
    keys = ("bayes_marginal", "bayes_model_average", "bayes_joint", "gls_estimates", "gls_weights",
            "gls_donor_scores", "coda_equivalence", "coda_donors", "observed_eligibility")
    tables = {key: [] for key in keys}
    minimum, alpha, margins = original["minimum_donors"], original["alpha"], original["margins_log2"]
    by_study = [(cohorts, original["applications"]),
                (external, [protocol["external_application"], *original["applications"]])]
    for studies, apps in by_study:
        for cohort in studies:
            for app in apps:
                genes, refs, coefficients = application_matrix(app)
                delta = cohort.matrix(genes)
                common = np.isfinite(delta).all(axis=1)
                n = int(common.sum())
                base = dict(application=app["name"], **cohort.key)
                tables["observed_eligibility"].append(dict(**base, n_total=len(cohort.donors),
                    n_shared=n, minimum_donors=minimum, bayes_joint_estimable=n >= minimum))
                if n >= minimum:
                    target = delta[common, :len(app["targets"])].mean(axis=1)
                    fit_data = np.column_stack([target, delta[common] @ coefficients])
                    result = bayesian_results(dict(**base, n=n), fit_data, refs, margins,
                                              protocol["bayes"], protocol["seed"])
                    for key, rows in zip(keys[:3], result):
                        tables[key].extend(rows)
                for ref in refs:
                    members = app["references"][ref]
                    panel = app["targets"]+members
                    change = cohort.matrix(panel)
                    native = np.isfinite(change).all(axis=1)
                    scopes = []
                    if n >= minimum:
                        scopes.append(("shared_family", common, len(refs)+1))
                    if native.sum() >= minimum and (n < minimum or not np.array_equal(native, common)):
                        scopes.append(("panel_specific", native, len(refs)))
                    for scope, mask, family_size in scopes:
                        size = int(mask.sum())
                        donors = np.asarray(cohort.donors)[mask]
                        ident = dict(**base, comparison=ref, donor_scope=scope, n=size)
                        target_count = len(app["targets"])
                        vals = change[mask]
                        weights = gls_results(ident, vals[:, :target_count].mean(axis=1), vals[:, target_count:],
                                              members, donors, protocol["gls"])
                        for key, rows in zip(keys[3:6], weights):
                            tables[key].extend(rows)
                        row, details = coda_results(ident, cohort.matrix(panel, "adjacent")[mask], vals,
                                                    target_count, donors, margins, alpha, family_size)
                        tables["coda_equivalence"].append(row)
                        tables["coda_donors"].extend(details)
    return tables
