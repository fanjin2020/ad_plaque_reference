"""Reproduce the full-class point thresholds reported in Table S83."""
import csv
import io
import math


def csv_bytes(rows):
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode("utf-8")


def read_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def tipping(annotated, estimates):
    rows = [r for r in annotated if r["dataset"]=="Leitner" and r["group"]=="AD" and
            r["reference"]=="ec3_lysosomal_lumen" and r["target"]=="False"]
    e, = [r for r in estimates if r["application"]=="lysosomal_anchor" and
          r["dataset"]=="Leitner" and r["comparison"]=="ec3_lysosomal_lumen"]
    n = int(e["n"])
    complete = sum(int(r["n_pair"]) == n for r in rows)
    partial = sum(0 < int(r["n_pair"]) < n for r in rows)
    zero = sum(int(r["n_pair"]) == 0 for r in rows)
    assert (n,len(rows),complete,partial,zero) == (53,41,5,6,30)
    fraction = (partial + zero) / len(rows)
    mean, lo, hi = [float(e[k]) for k in ("mean","simultaneous_low","simultaneous_high")]
    output = []
    for margin in (0.0,-0.2,0.2):
        delta = (mean-margin)/fraction
        assert math.isclose(mean-fraction*delta,margin,abs_tol=1e-12)
        output.append(dict(dataset="Leitner",group="AD",donors=n,annotated_non_targets=len(rows),
                           complete=complete,partial=partial,no_observed_pairs=zero,
                           incomplete_weight=fraction,observed_score=mean,
                           observed_simultaneous_low=lo,observed_simultaneous_high=hi,
                           full_class_score_threshold=margin,delta_for_point=delta,
                           delta_for_lower_limit=(lo-margin)/fraction,
                           delta_for_upper_limit=(hi-margin)/fraction,
                           delta_definition="mean regional change among the 36 incomplete candidates minus mean among the five complete candidates, in the same 53 donors",
                           uncertainty="Transformed measured-panel limits; Delta is a sensitivity parameter, not an observed or imputed effect"))
    return output
