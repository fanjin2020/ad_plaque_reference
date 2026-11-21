"""Intervals and models for donor-level regional contrasts."""

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def critical_t(df):
    # Integrate the t density by Gaussian quadrature, then invert its central 95% interval.
    x, w = np.polynomial.legendre.leggauss(48)
    norm = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(
        df * math.pi
    )
    lo, hi = 0.0, 16.0
    for _ in range(53):
        mid = (lo + hi) / 2
        y = mid * (x + 1) / 2
        area = mid / 2 * norm * np.dot(w, (1 + y * y / df) ** (-(df + 1) / 2))
        if area < 0.475:
            lo = mid
        else:
            hi = mid
    return (hi + lo) / 2


def paired_stats(score):
    n = len(score)
    if n < 2 or not np.isfinite(score).all():
        raise ValueError("Incomplete donor comparison")
    avg = score.mean()
    err = critical_t(n - 1) * score.std(ddof=1) / math.sqrt(n)
    return dict(
        n=n,
        positive=int((score > 0).sum()),
        estimate=avg,
        low=avg - err,
        high=avg + err,
    )


def tissue_models(meta, labels, proteins, acc):
    records = []
    for (group, c, arm), col in labels.items():
        if group not in ("Control", "AD"):
            continue
        row = meta[c]
        vals = [proteins[acc[g]][1][col] for g in ("CTSD", "TPP1")]
        grade = next(v for k, v in row.items() if k.startswith("Regional Aβ plaques"))
        records.append(
            dict(
                case=c,
                group=group,
                arm=arm,
                source=row["Study"].strip(),
                age=float(row.get("Age", "nan")),
                sex=row.get("Sex", ""),
                PMI=float(row.get("PMI", "nan")),
                regional_score=float(grade),
                CTSD=vals[0],
                TPP1=vals[1],
                CTSD_minus_TPP1=vals[0] - vals[1],
            )
        )
    result = []
    scopes = {
        "all": records,
        "ROS_only": [r for r in records if r["source"] == "ROS"],
        "score0_controls": [
            r for r in records if r["group"] == "AD" or r["regional_score"] == 0
        ],
    }
    for scope, subset in scopes.items():
        for adjusted in (False, True):
            # Use the same complete donors for the two enzymes and their difference.
            exclude = {
                r["case"]
                for r in subset
                if not np.isfinite(r["CTSD_minus_TPP1"])
                or adjusted
                and (
                    not np.isfinite(r["age"] + r["PMI"])
                    or r["sex"] not in ("F", "M")
                    or r["source"] not in ("ROS", "MAP")
                )
            }
            used = [r for r in subset if r["case"] not in exclude]
            clusters = np.array([r["case"] for r in used])
            cases, which = np.unique(clusters, return_inverse=True)
            G = len(cases)
            names = ["intercept", "AD_adjacent", "AD_plaque"]
            X = [
                [
                    1,
                    r["group"] == "AD" and r["arm"] == "np",
                    r["group"] == "AD" and r["arm"] == "p",
                ]
                for r in used
            ]
            if adjusted:
                names += ["age", "male", "PMI"]
                source_varies = len({r["source"] for r in used}) > 1
                if source_varies:
                    names.append("MAP")
                for row, r in zip(X, used):
                    row.extend([r["age"] - 80, r["sex"] == "M", r["PMI"]])
                    if source_varies:
                        row.append(r["source"] == "MAP")
            X = np.asarray(X, dtype=float)
            n, k = X.shape
            if np.linalg.matrix_rank(X) != k or G < 2 or n <= k:
                raise ValueError("Unidentified tissue model: " + scope)
            bread = np.linalg.inv(X.T @ X)
            for outcome in ("CTSD_minus_TPP1", "CTSD", "TPP1"):
                y = np.array([r[outcome] for r in used])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                scores = np.zeros((G, k))
                np.add.at(scores, which, X * (y - X @ beta)[:, None])
                cov = (
                    (G / (G - 1))
                    * ((n - 1) / (n - k))
                    * bread
                    @ scores.T
                    @ scores
                    @ bread
                )
                for contrast, wa, wp in (
                    ("AD_plaque_minus_AD_adjacent", -1, 1),
                    ("AD_adjacent_minus_control", 1, 0),
                    ("AD_plaque_minus_control", 0, 1),
                ):
                    l = np.zeros(k)
                    l[1:3] = [wa, wp]
                    est = l @ beta
                    err = critical_t(G - 1) * math.sqrt(max(0, l @ cov @ l))
                    result.append(
                        dict(
                            scope=scope,
                            model="adjusted" if adjusted else "crude",
                            outcome=outcome,
                            contrast=contrast,
                            n_AD=len({r["case"] for r in used if r["group"] == "AD"}),
                            n_control=len(
                                {r["case"] for r in used if r["group"] == "Control"}
                            ),
                            n_tissue=n,
                            estimate=est,
                            low=est - err,
                            high=est + err,
                            covariates="|".join(names[3:]),
                            cluster_df=G - 1,
                        )
                    )
    return records, result
