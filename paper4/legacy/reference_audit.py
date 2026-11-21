"""Recompute annotation coverage, five-reference span and member influence."""
import csv


REFERENCES = [
    "pump", "lysosomal_lumen_hydrolases", "ec3_lysosomal_lumen",
    "ec3_lysosome", "ec34_lysosome",
]


def derive(folder):
    def load(name):
        with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def write(name, records):
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

    comparisons = load("03_reference_sets.csv")
    membership = load("04_reference_membership.csv")
    annotation = load("13_EC_membership.csv")
    omitted = load("14_leave_one_control_out.csv")
    coverage, estimates, span, influence = [], [], [], []
    for pump in load("01_fixed_panel.csv"):
        dataset, group = pump["dataset"], pump["group"]
        identity = dict(dataset=dataset, group=group)
        values = []

        def matches(row, reference):
            return (row["dataset"], row["group"], row["reference_class"]) == (dataset, group, reference)

        for reference in REFERENCES:
            if reference == "pump":
                record, controls = pump, 8
            else:
                record, = (
                    row for row in comparisons if matches(row, reference)
                    and row["comparison"] == "CTSD+TPP1 vs controls"
                )
                controls = int(record["n_controls"])
            estimate = float(record["estimate_log2"])
            values.append(estimate)
            estimates.append(dict(**identity, reference=reference, controls=controls, **{
                field: record[field] for field in ("estimate_log2", "ci95_low", "ci95_high")
            }))
            if reference == "pump":
                continue
            rows = [row for row in membership if matches(row, reference)]
            expected = {row["accession"] for row in annotation if row["reference_class"] == reference
                        and row["included_by_annotation_rule"] == "True"}
            assert len(rows) == len(expected) and {row["accession"] for row in rows} == expected
            non_targets = [row for row in rows if row["target"] == "False"]
            assert len(rows) - len(non_targets) == 2
            assert sum(row["eligible_control"] == "True" for row in non_targets) == controls
            pairs = [(int(row["paired_donors"]), int(row["eligible_donors"])) for row in rows]
            coverage.append(dict(
                **identity, reference_class=reference, annotated=len(rows),
                mapped=sum(row["matches_workbook_accession_and_gene"] == "True" for row in rows),
                any_pair=sum(n > 0 for n, _ in pairs), at_least_half=sum(2 * n >= total for n, total in pairs),
                all_pairs=sum(n == total for n, total in pairs), annotated_non_targets=len(non_targets),
                usable_controls=controls, usable_control_pct=100 * controls / len(non_targets),
            ))
            influence.extend(dict(
                **identity, reference=reference, omitted=row["excluded_gene"],
                delta_log2=float(row["estimate_log2"]) - estimate,
            ) for row in omitted if matches(row, reference))
        span.append(dict(
            **identity, five_set_span=max(values) - min(values),
            min_reference=REFERENCES[values.index(min(values))], max_reference=REFERENCES[values.index(max(values))],
        ))
    for name, rows in zip(("17_coverage", "18_reference_estimates", "19_reference_span", "20_influence"),
                          (coverage, estimates, span, influence)):
        write(name + ".csv", rows)

