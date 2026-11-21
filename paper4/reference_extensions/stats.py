"""Paired donor inference and explicit multiplicity families."""
import math
import numpy as np
from scipy.stats import t


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    missing = ~np.isfinite(p)
    work = np.where(missing, 1., p)
    order = np.argsort(work, axis=1, kind="stable")
    sorted_p = np.take_along_axis(work, order, axis=1)
    adjusted = np.minimum(1., np.maximum.accumulate(
        sorted_p * np.arange(p.shape[1], 0, -1)[None, :], axis=1))
    result = np.empty_like(adjusted)
    np.put_along_axis(result, order, adjusted, axis=1)
    result[missing] = np.nan
    return result[0] if single else result


def summarize_columns(values, alpha=.05, family_size=None):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError("Inference requires an explicitly complete common donor set")
    n, k = values.shape
    if n < 3:
        raise ValueError("At least three paired donor observations are required")
    family_size = k if family_size is None else family_size
    if family_size < k:
        raise ValueError("Multiplicity family smaller than the tests being reported")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/math.sqrt(n)
    valid = np.isfinite(se) & (se > 1e-14)
    statistic = np.divide(mean, se, out=np.full(k, np.nan), where=valid)
    p = 2*t.sf(np.abs(statistic), n-1)
    nominal = t.ppf(1-alpha/2, n-1)*se
    joint = t.ppf(1-alpha/(2*family_size), n-1)*se
    nominal[~valid] = joint[~valid] = np.nan
    # Preserve planned-but-unestimable tests in the family as p=1.
    padded = np.r_[p, np.ones(family_size-k)]
    adjusted = holm(padded)[:k]
    return {"mean": mean, "se": se, "p": p, "p_holm": adjusted,
            "nominal_low": mean-nominal, "nominal_high": mean+nominal,
            "simultaneous_low": mean-joint, "simultaneous_high": mean+joint,
            "valid": valid}


def bootstrap_means(values, draws, rng):
    output = np.empty((draws, values.shape[1]))
    for start in range(0, draws, 500):
        stop = min(start+500, draws)
        indices = rng.integers(0, len(values), (stop-start, len(values)))
        output[start:stop] = values[indices].mean(axis=1)
    return np.quantile(output, [.025, .975], axis=0)


def rate_summary(events):
    events = np.asarray(events, bool)
    n = len(events)
    if not n:
        return {"count": 0, "denominator": 0, "rate": None,
                "mcse": None, "low": None, "high": None}
    count = int(events.sum())
    p, z = count/n, 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"count": count, "denominator": n, "rate": p,
            "mcse": math.sqrt(p*(1-p)/n), "low": max(0., center-half),
            "high": min(1., center+half)}
