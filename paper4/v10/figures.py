"""Figures for the external application and the supplementary method comparisons."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


DARK = "#000000"
TEAL = "#007f86"
BLUE = "#275399"
AMBER = "#aa6819"
GRAY = "#798695"


def export(fig, folder, name):
    for ext in ("pdf", "png", "tiff"):
        options = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        if ext == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(folder / f"{name}.{ext}", facecolor="white", **options)
    plt.close(fig)


def external_figure(folder, external, estimates, donors):
    groups = ("Advanced AD", "Intermediate AD")
    comparisons = ("regional", "neuronal_cytoskeleton", "synaptic_proteins")
    labels = ("Tangle − neuropil", "vs cytoskeleton", "vs synaptic\nproteins")
    selected = {(r["group"], r["comparison"]): r for r in estimates if r["application"] == "mapt_tangle"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.28, top=.8, wspace=.22)
    fig.suptitle("MAPT in tau tangles and adjacent neuropil", x=.13, ha="left", weight="bold", fontsize=10)
    for ax, group, letter in zip(axes, groups, "ab"):
        cohort = next(c for c in external if c.group == group)
        values = {(r["donor"], r["comparison"]): r["score"] for r in donors
                  if r["group"] == group and r["application"] == "mapt_tangle"}
        stats = [selected[group, ref] for ref in comparisons]
        x = np.arange(len(comparisons))
        for donor in cohort.donors:
            ax.plot(x, [values[donor, r] for r in comparisons], "o-", color=GRAY, alpha=.53,
                    linewidth=.65, markersize=2.7, zorder=1)
        for i, row in enumerate(stats):
            y = row["mean"]
            ax.errorbar(i, y, yerr=[[y-row["simultaneous_low"]], [row["simultaneous_high"]-y]],
                        fmt="D", markersize=4.8, color=TEAL, capsize=2.3, linewidth=1.05, zorder=3)
        ax.axhline(0, color="#445460", linewidth=.75)
        ax.hlines(.2, .65, 2.3, color=AMBER, linestyle="--", linewidth=.9)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_xlim(-.29, 2.29)
        ax.set_title(f"{letter}   {group} · {stats[0]['n']} paired donors", loc="left", weight="bold", pad=8)
    axes[0].set_ylabel("Paired log$_2$ difference")
    fig.text(.13, .075, "Lines connect each donor. Diamonds and whiskers show means and 95% simultaneous intervals.",
             fontsize=7, color=DARK)
    fig.text(.13, .029, "Dashed line marks the +0.2 log$_2$ margin for reference-relative contrasts.",
             fontsize=7, color=DARK)
    export(fig, folder, "01_MAPT_external")


def simulation_figure(folder, simulation):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.26, top=.79, wspace=.22)
    fig.suptitle("Sensitivity and false positives on held-out simulations", x=.13, ha="left",
                 weight="bold", fontsize=10)
    for ax, margin, letter in zip(axes, (0.0, .2), "ab"):
        rows = [r for r in simulation["ssr_roc"] if
                (r["partition"], r["application"], r["margin_log2"]) == ("held_out", "All", margin)]
        summary = {r["method"]: r for r in simulation["ssr_performance"] if
                   (r["partition"], r["application"], r["margin_log2"]) == ("held_out", "All", margin)}
        ax.plot([r["false_positive_rate"] for r in rows], [r["sensitivity"] for r in rows],
                color=BLUE, linewidth=1.35, label="SSR curve")
        ax.axvline(.05, color=GRAY, linewidth=.75, linestyle=":")
        for method, name, marker, color in (
                ("SSR_calibrated", "Calibrated SSR", "o", BLUE),
                ("SSR_cutoff_1", "SSR > 1", "s", AMBER),
                ("v9_simultaneous", "v9 simultaneous bounds", "D", TEAL)):
            row = summary[method]
            ax.scatter(row["false_positive_rate_conditional"], row["sensitivity_conditional"],
                       marker=marker, color=color, s=28, label=name, zorder=4)
        auc = summary["SSR_calibrated"]["ssr_roc_auc"]
        ax.set_title(f"{letter}   Margin {margin:g} log$_2$ · AUC {auc:.3f}", weight="bold", loc="left", pad=8)
        ax.set(xlim=(-.006, .206), ylim=(-.01, .43), xlabel="False positive rate")
        ax.set_xticks([0, .05, .10, .15, .20])
        ax.set_yticks([0, .10, .20, .30, .40])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Sensitivity among estimable repeats")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .067), ncol=4, fontsize=7)
    fig.text(.13, .025, "Axes zoomed near the decision thresholds. AUC uses the full ranking.", fontsize=7)
    export(fig, folder, "02_SSR_held_out")


def methods_figure(folder, methods, prior_sd):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))
    fig.subplots_adjust(left=.205, right=.975, bottom=.24, top=.79, wspace=.9)
    fig.suptitle("Reference choice and reference weighting", x=.205, ha="left", weight="bold", fontsize=10)
    groups = [("Leitner", "AD", "lysosomal_anchor", "CTSD/TPP1 · plaque"),
              ("Leitner", "AD", "apoe_lipid", "APOE · plaque"),
              ("Thierry 2026", "Advanced AD", "mapt_tangle", "MAPT · tangle")]
    y = np.arange(len(groups))
    for i, (study, group, app, _) in enumerate(groups):
        joint, = [r for r in methods["bayes_joint"] if
                  (r["dataset"], r["group"], r["application"], r["margin_log2"], r["prior_sd_log2"])
                  == (study, group, app, .2, prior_sd)]
        mixture, = [r for r in methods["bayes_model_average"] if
                    (r["dataset"], r["group"], r["application"], r["margin_log2"], r["prior_sd_log2"],
                     r["choice_prior"]) == (study, group, app, .2, prior_sd, "equal_probabilities")]
        x = [joint["p_region_and_all_references"], mixture["p_region_and_selected_reference"]]
        axes[0].hlines(i, *sorted(x), color="#c6cfd6", linewidth=1.0)
        axes[0].scatter(x[0], i, marker="D", color=TEAL, s=28, label="All references" if i == 0 else None)
        axes[0].scatter(x[1], i, marker="o", facecolor="white", edgecolor=BLUE, linewidth=.9, s=28,
                        label="Averaged choice" if i == 0 else None)
    axes[0].set(xlim=(-.035, 1.055), xlabel="Posterior probability · margin +0.2 log$_2$")
    axes[0].set_yticks(y, [row[3] for row in groups])
    axes[0].set_title("a   Joint and averaged posterior", loc="left", weight="bold", pad=8)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-.2, -.35), ncol=2, fontsize=7, handletextpad=.25)

    rows = [r for r in methods["gls_estimates"]
            if (r["dataset"], r["group"], r["comparison"], r["donor_scope"]) ==
            ("Leitner", "AD", "ec3_lysosomal_lumen", "shared_family")]
    by_method = {r["method"]: r["estimate_log2"] for r in rows}
    labels = ("Equal", "Inverse covariance", "Nonnegative LOO")
    ordered = ("equal_weight", "unconstrained_inverse_covariance", "nonnegative_shrinkage_leave_one_donor_out")
    axes[1].scatter([by_method[name] for name in ordered], y, c=[TEAL, AMBER, BLUE], marker="D", s=32)
    axes[1].axvline(.2, color=GRAY, linestyle="--", linewidth=.75)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("CTSD/TPP1 vs luminal enzymes (log$_2$)")
    axes[1].set_title("b   Effect by weight", loc="left", weight="bold", pad=8)
    for ax in axes:
        ax.set_ylim(2.48, -.48)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    export(fig, folder, "03_Bayesian_GLS")


def prepare_figures(output, external, external_tables, simulation, methods, protocol):
    folder = output / "figures"
    folder.mkdir()
    with mpl.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": .75, "legend.frameon": False, "text.color": DARK,
    }):
        external_figure(folder, external, external_tables["03_shared_donor_estimates"],
                        external_tables["04_donor_scores"])
        simulation_figure(folder, simulation)
        methods_figure(folder, methods, protocol["bayes"]["default_prior_sd_log2"])
