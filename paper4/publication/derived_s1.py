"""Derive the manuscript summaries from the numerical supplement and reviewed follow-up."""
import csv
from collections import defaultdict
from io import StringIO
from statistics import median


def read_csv(raw):
    return list(csv.DictReader(StringIO(raw.decode("utf-8-sig"))))


def csv_bytes(rows):
    rows = list(rows)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def derive(archive, estimates, profiles, annotation):
    def table(number):
        path, = [name for name in archive.namelist() if name.startswith(f"Table_S{number:02d}_")]
        return read_csv(archive.read(path))

    protein_intensities = {r["accession"]: r for r in table(4)
                           if (r["dataset"], r["group"], r["reference_class"]) ==
                           ("Leitner", "AD", "ec3_lysosomal_lumen")}
    candidates = [r for r in table(28) if (r["dataset"], r["group"], r["reference"], r["target"]) ==
                  ("Leitner", "AD", "ec3_lysosomal_lumen", "False")]
    membership = []
    for row in candidates:
        n = int(row["n_pair"])
        accession = row["accession"]
        protein = protein_intensities[accession]
        assert protein["primary_gene"] == row["gene"] and int(protein["paired_donors"]) == n
        classes = annotation["ec_by_accession"][accession]
        category = "complete" if n == 53 else ("partial" if n else "no_paired_value")
        membership.append(dict(gene=row["gene"], accession=accession, pairs=n, eligible_donors=53,
                               coverage=category, ec_numbers="|".join(classes),
                               ec_subclasses="|".join(sorted({".".join(c.split(".")[:2]) for c in classes})),
                               median_adjacent_log2=protein["median_adjacent_log2"]))
    groups = defaultdict(list)
    for row in estimates:
        if row["comparison"] != "regional":
            groups[(row["application"], row["dataset"], row["group"])].append(row)
    summary = []
    for (app, ds, group), rows in groups.items():
        assert len(set(row["n"] for row in rows)) == 1
        minimum = min(rows, key=lambda r: float(r["mean"]))
        maximum = max(rows, key=lambda r: float(r["mean"]))
        weakest = min(rows, key=lambda r: float(r["simultaneous_low"]))
        upper = max(rows, key=lambda r: float(r["simultaneous_high"]))
        lo, hi = float(minimum["mean"]), float(maximum["mean"])
        summary.append(dict(application=app, dataset=ds, group=group, n=rows[0]["n"],
                            relative_panels=len(rows), minimum_panel=minimum["comparison"], minimum_mean=lo,
                            maximum_panel=maximum["comparison"], maximum_mean=hi, span=hi-lo,
                            lower_limit_panel=weakest["comparison"], smallest_simultaneous_lower=float(weakest["simultaneous_low"]),
                            upper_limit_panel=upper["comparison"], largest_simultaneous_upper=float(upper["simultaneous_high"]),
                            minimum_mean_over_span=lo/(hi-lo) if hi > lo else ""))
    ordered = sorted((r for r in profiles if r["definition"] == "observed" and r["scope"] == "original"),
                     key=lambda r: (float(r["estimate"]), r["dataset"], r["group"], r["reference"]))
    specifications = [dict(rank=i, **{k: r[k] for k in ("dataset","group","reference","reference_genes","n_members",
                      "n","estimate","bonferroni_low","bonferroni_high","family_size")}) for i,r in enumerate(ordered,1)]
    assert len(specifications) == 20 and len(summary) == 5 and len(membership) == len({r["accession"] for r in membership}) == 41
    assert [sum(r["coverage"] == s for r in membership) for s in ("complete","partial","no_paired_value")] == [5,6,30]
    assert [round(median(float(r["median_adjacent_log2"]) for r in membership if r["coverage"]==s),2)
            for s in ("complete","partial")] == [15.02,12.89]
    ec32 = [r for r in membership if "3.2" in r["ec_subclasses"].split("|")]
    assert [len(ec32),sum(r["pairs"]==53 for r in ec32),sum(r["pairs"]==0 for r in ec32)] == [19,1,16]
    assert [r["gene"] for r in membership if r["pairs"]==0 and r["median_adjacent_log2"]] == ["CTSA"]
    return {"family_summary":summary, "candidate_coverage":membership, "ordered_specifications":specifications}
