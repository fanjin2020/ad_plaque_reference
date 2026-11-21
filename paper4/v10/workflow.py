"""Run and verify the five experiments through the published input route."""
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
import pymupdf
from PIL import Image

from paper4.reference_extensions.application import analyze
from paper4.reference_extensions.io import write_csv, write_json
from paper4.reference_extensions.simulation import build_models
from paper4.source_inputs import output_hashes, sha256_file, verify_output

from .figures import prepare_figures
from .inputs import PAPER4, PROTOCOL, load_inputs
from .methods import observed_methods
from .ssr import observed_ssr, run_ssr


EXTERNAL_FILES = {
    "01_coverage": "external_protein_coverage",
    "02_family_availability": "external_family_coverage",
    "03_shared_donor_estimates": "external_effects",
    "04_donor_scores": "external_donor_scores",
    "05_reference_differences": "external_reference_differences",
    "06_member_influence": "external_member_influence",
    "07_panel_specific_donors": "external_available_panels",
    "08_decisions": "external_decisions",
}


def check_numerics(external_tables, methods, simulation, original, protocol):
    groups = ["Advanced AD", "Intermediate AD", "PART"]
    expected = {"mapt_tangle": [6, 6, 5], "apoe_lipid": [3, 5, 3], "lysosomal_anchor": [0, 0, 1]}
    availability = {(r["application"], r["group"]): r for r in external_tables["02_family_availability"]}
    for app, expected_counts in expected.items():
        observed = [availability[app, group]["shared_donors"] for group in groups]
        if observed != expected_counts:
            raise ValueError(f"External coverage differs for {app}: {observed}")
    estimates = external_tables["03_shared_donor_estimates"]
    lookup = {(r["group"], r["application"], r["comparison"]): r["mean"] for r in estimates}
    donor_rows = external_tables["04_donor_scores"]
    for key, estimate in lookup.items():
        values = [r["score"] for r in donor_rows if (r["group"], r["application"], r["comparison"]) == key]
        np.testing.assert_allclose(np.mean(values), estimate, rtol=0, atol=2e-12)

    anchors = [r for r in methods["coda_equivalence"] if
               (r["dataset"], r["group"], r["donor_scope"], r["application"]) ==
               ("Leitner", "AD", "shared_family", "lysosomal_anchor")]
    anchor_means = {r["comparison"]: r["reference_relative_log2"] for r in anchors}
    np.testing.assert_allclose(
        [anchor_means["v_atpase"], anchor_means["ec3_lysosomal_lumen"]],
        [.8332312916809744, .03744513313726627], atol=2e-12, rtol=0)
    maximum_error = max(r["max_abs_reconstruction_error"] for r in methods["coda_equivalence"])
    if maximum_error > 2e-12:
        raise ValueError("The CoDA back-transformation error exceeds tolerance")
    for r in methods["bayes_model_average"]:
        weights = [float(w) for w in r["posterior_choice_weights"].split("|")]
        if not math.isclose(sum(weights), 1) or not 0 <= r["p_selected_reference_above_margin"] <= 1:
            raise ValueError("Invalid Bayesian reference-choice weights or probability")
    for r in methods["bayes_joint"]:
        if not 0 <= r["p_region_and_all_references"] <= r["minimum_marginal_probability"] <= 1:
            raise ValueError("Bayesian joint probability violates its analytic upper bound")
    for r in methods["gls_weights"]:
        if r["method"] == "nonnegative_shrinkage_leave_one_donor_out" and r["minimum"] < 0:
            raise ValueError("Cross-fitted constrained GLS produced a negative weight")

    design = protocol["ssr"]
    baseline = original["simulation"]
    scenario_rows = simulation["ssr_scenarios"]
    counts = dict(Counter(r["partition"] for r in scenario_rows))
    applications = len(original["applications"])
    v9_count = applications * 4 * len(baseline["generators"]) * len(baseline["sample_sizes"]) * len(baseline["missingness"])
    boundary = applications * (len(design["minimum_true_effect_log2"]) * len(design["true_span_log2"])
                               + len(design["negative_regional_challenge"]))
    boundary *= len(design["sample_sizes"]) * len(design["missingness"])
    if counts != {"v9_replay": v9_count, "calibration": boundary, "held_out": boundary}:
        raise ValueError("The simulation partitions are incomplete")
    for r in scenario_rows:
        if not 0 <= r["n_defined_ssr"] <= r["n_estimable_full_decision"] <= r["n_estimable_v9"] <= r["repetitions"]:
            raise ValueError("Inconsistent simulation denominators")
    thresholds = simulation["ssr_cutoffs"]
    if any(r["calibration_fpr"] > design["desired_false_positive_rate"]+1e-12 for r in thresholds):
        raise ValueError("A calibrated threshold misses its training-set error target")
    external_decisions = {(r["dataset"], r["group"], r["application"], r["margin_log2"]):
                          r["robust_above_all_planned_references"] for r in external_tables["08_decisions"]
                          if r["n"] >= original["minimum_donors"]}
    measured_decisions = {(r["dataset"], r["group"], r["application"], r["margin_log2"]):
                          r["v9_simultaneous_pass"] for r in simulation["ssr_observed"]}
    if any(measured_decisions[key] != result for key, result in external_decisions.items()):
        raise ValueError("The external measured decision disagrees with the shared-donor inference")

    return dict(
        external_common_donors=expected, external_group_order=groups, external_donor_mean_checks=len(lookup),
        coda_contrasts=len(methods["coda_equivalence"]), maximum_coda_reconstruction_error=maximum_error,
        observed_ssr_diagnostics=len(simulation["ssr_observed"]),
        gls_estimate_rows=len(methods["gls_estimates"]), bayesian_average_rows=len(methods["bayes_model_average"]),
        bayesian_joint_rows=len(methods["bayes_joint"]), simulation_scenarios=counts,
        simulation_repeats=sum(r["repetitions"] for r in scenario_rows),
        n_estimable_v9_relative_only=sum(r["n_estimable_v9"] for r in scenario_rows if r["partition"] == "v9_replay"),
    )


def check_figures(output):
    folder = output / "figures"
    names = sorted(p.stem for p in folder.glob("*.pdf"))
    if len(names) != 3:
        raise ValueError("Expected three experiment figures")
    for name in names:
        with pymupdf.open(folder / f"{name}.pdf") as doc:
            if len(doc) != 1 or not doc[0].get_text().strip():
                raise ValueError(f"Missing editable content in {name}.pdf")
        with Image.open(folder / f"{name}.png") as image:
            image.verify()
        with Image.open(folder / f"{name}.tiff") as image:
            if min(image.info["dpi"]) < 599:
                raise ValueError(f"Unexpected TIFF resolution for {name}")
            image.verify()
    return names


def summary_text(external_tables, simulation, methods, checks, prior_sd):
    lines = [
        "# 论文4 v10 实验结果", "",
        "## 1. 外部海马 tau 缠结配对分析", "",
        "以公开工作簿的 P10636 定量行考察 MAPT。晚期 AD 为主要应用，中期 AD 与 PART 分组报告。",
        "三个对比使用同一批供体。表内区间是同时 95% 置信区间，效应单位是 log₂。", "",
        "| 组别 | 共同供体 | 区域变化 | 超过细胞骨架 | 超过突触蛋白 |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for group in checks["external_group_order"]:
        rows = [r for r in external_tables["03_shared_donor_estimates"]
                if (r["application"], r["group"]) == ("mapt_tangle", group)]
        values = " | ".join(f"{r['mean']:+.3f} [{r['simultaneous_low']:+.3f}, {r['simultaneous_high']:+.3f}]"
                            for r in rows)
        lines.append(f"| {group} | {rows[0]['n']} | {values} |")
    lines.extend(["", "原有 APOE 面板的完整配对数为 3、5、3，CTSD/TPP1 为 0、0、1（按上表组别顺序）。"
        "各研究的编号来自各自工作簿，跨研究供体身份未能对照。参照的稳健性只针对这里指定的成员与区域。"
        " [Thierry 等人原始研究](https://doi.org/10.1007/s00401-026-03064-9)", "",
        "## 2. SSR 校准与留出评估", "",
        f"共 {sum(checks['simulation_scenarios'].values()):,} 个场景，{checks['simulation_repeats']:,} 次重复。"
        "校准与留出设计共享斑块协方差来源，采用不同生成机制、参考效应排序与独立随机流。"
        "分母为可估计的重复，另有以全部重复为分母的结果。", "",
        "| log₂ 门槛 | 方法 | 留出误报率 | 留出检出率 | 单个场景最高误报率¹ |",
        "| ---: | --- | ---: | ---: | ---: |"])
    rule_labels = {"v9_simultaneous": "同时区间", "SSR_cutoff_1": "SSR > 1", "SSR_calibrated": "校准后 SSR"}
    for r in simulation["ssr_performance"]:
        if (r["partition"], r["application"]) == ("held_out", "All"):
            lines.append(f"| {r['margin_log2']:g} | {rule_labels[r['method']]} | "
                         f"{r['false_positive_rate_conditional']:.2%} | {r['sensitivity_conditional']:.2%} | "
                         f"{r['false_positive_rate_worst_estimable_scenario']:.2%} |")
    lines.extend(["", "¹ 每个计入极值的场景有至少 100 次可估计重复。"
                  "逐队列诊断比值与判定存于 ssr_observed.csv。", "",
        "## 3. 贝叶斯参照平均和联合概率", "",
        "共同似然下，参照选择的平均权重取自预设先验。联合概率对应区域上升且全部参照同时超过门槛。"
        "两种概率在同一批供体和同一个后验下计算。", "",
        "| 数据组及靶标 | 等概率选择一个参照并满足区域条件 | 区域条件和全部参照同时通过 |",
        "| --- | ---: | ---: |"])
    cases = [("Leitner", "AD", "lysosomal_anchor", "斑块 CTSD/TPP1"),
             ("Leitner", "AD", "apoe_lipid", "斑块 APOE"),
             ("Thierry 2026", "Advanced AD", "mapt_tangle", "晚期 AD 缠结 MAPT")]
    for study, group, app, label in cases:
        key = (study, group, app)
        joint, = [r for r in methods["bayes_joint"] if (r["dataset"], r["group"], r["application"]) == key
                  and (r["prior_sd_log2"], r["margin_log2"]) == (prior_sd, .2)]
        avg, = [r for r in methods["bayes_model_average"] if (r["dataset"], r["group"], r["application"]) == key
                and (r["prior_sd_log2"], r["margin_log2"], r["choice_prior"]) ==
                (prior_sd, .2, "equal_probabilities")]
        lines.append(f"| {label} | {avg['p_region_and_selected_reference']:.2%} | "
                     f"{100*joint['p_region_and_all_references']:.4g}% |")
    lines.extend(["", f"表内采用 +0.2 log₂ 门槛和 {prior_sd:g} log₂ 协方差先验尺度。"
        "完整结果覆盖三种尺度、两种门槛和预设的参照选择权重。", "",
        "## 4. GLS 权重敏感性", "",
        "以下为 Leitner AD 中 CTSD/TPP1 相对腔内 EC3 水解酶的同供体比较。"
        "留一供体权重只使用其余供体拟合，误差指标为描述性。", "",
        "| 权重 | 相对点估计 log₂ | 负权重数 |",
        "| --- | ---: | ---: |"])
    for r in methods["gls_estimates"]:
        if (r["dataset"], r["group"], r["comparison"], r["donor_scope"]) == (
                "Leitner", "AD", "ec3_lysosomal_lumen", "shared_family"):
            label = {"equal_weight": "等权", "unconstrained_inverse_covariance": "逆协方差",
                     "nonnegative_shrinkage_leave_one_donor_out": "非负收缩权重（留一供体）"}[r["method"]]
            lines.append(f"| {label} | {r['estimate_log2']:+.3f} | {r['negative_weights']} |")
    lines.extend(["", "## 5. CoDA 同尺度核对", "",
        f"共 {checks['coda_contrasts']} 个对比。ALR、闭合后 ALR、CLR 对比及反变换后的 ILR "
        f"与原始逐供体估计的最大绝对差为 {checks['maximum_coda_reconstruction_error']:.3g} log₂。"
        "同时区间和生物学门槛按同一正常数换算。", "",
        "所有计划场景的分母、效应、权重及判定均随结果保存。逐文件校验和见 verification.json。", ""])
    return "\n".join(lines)


def run(args, pins):
    started = time.monotonic()
    dest = args.out.resolve()
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = dest.with_name(dest.name + "_" + stamp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper4-work-v10-", dir=dest.parent) as path:
        work = Path(path)
        print("Verifying four public workbooks and rebuilding paired inputs", flush=True)
        protocol, original, cohorts, external, mapping, measurements, records = load_inputs(work, args.input_dir)
        output = work / "final"
        output.mkdir()
        table_folder = output / "tables"

        external_protocol = dict(original, applications=[protocol["external_application"], *original["applications"]],
                                 coverage_screen=[])
        external_tables = analyze(external, external_protocol, protocol["observed_bootstrap_draws"], protocol["seed"])
        print("External MAPT and observed Bayesian, GLS and CoDA experiments", flush=True)
        methods = observed_methods(cohorts, external, original, protocol)

        models, diagnostics = build_models(cohorts, original)
        if {m["application"] for m in models} != {app["name"] for app in original["applications"]}:
            raise ValueError("Both original covariance models must be estimable")
        print("Running SSR calibration, held-out scenarios and the v9 replay", flush=True)
        simulation = run_ssr(models, original, protocol, args.workers, output)
        simulation["ssr_observed"] = observed_ssr(
            [(cohorts, original["applications"], "observed_diagnostic"),
             (external, external_protocol["applications"], "outside_simulated_studies")],
            simulation["ssr_cutoffs"], original, protocol["ssr"]["span_tolerance_log2"])
        tables = {
            "external_accession_mapping": mapping, "external_quantities": measurements,
            **{name: external_tables[key] for key, name in EXTERNAL_FILES.items()},
            "ssr_covariance_models": diagnostics["10_covariance_models"],
            "ssr_joint_covariance": diagnostics["11_joint_covariance"],
            "ssr_contrast_coefficients": diagnostics["12_contrast_coefficients"],
            **methods, **simulation,
        }
        for name, rows in tables.items():
            write_csv(table_folder / f"{name}.csv", rows)
        checks = check_numerics(external_tables, methods, simulation, original, protocol)
        prepare_figures(output, external, external_tables, simulation, methods, protocol)
        figures = check_figures(output)
        (output / "Summary.md").write_text(
            summary_text(external_tables, simulation, methods, checks, protocol["bayes"]["default_prior_sd_log2"]),
            encoding="utf-8")
        write_json(output / "analysis_protocol.json", {
            "v10": protocol, "original_fixed_protocol": original, "workbooks": records})
        write_json(output / "verification.json", {
            "status": "passed", "elapsed_seconds": round(time.monotonic()-started, 1),
            "data_source": "verified_publisher_downloads" if args.input_dir is None else "verified_local_workbooks",
            "public_source_bytes": sum(row["bytes"] for row in records), "workbooks": records,
            "environment": {"python": platform.python_version(), "platform": platform.platform(), **pins},
            "protocol_sha256": sha256_file(PROTOCOL),
            "base_protocol_sha256": sha256_file(PAPER4 / "reference_extensions/data/protocol.json"),
            "figures": figures, "tables": {name: len(rows) for name, rows in tables.items()},
            "numerical_checks": checks, "outputs_sha256": output_hashes(output),
        })
        verify_output(output)
        output.rename(dest)
    return dest
