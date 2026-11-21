"""Publication-sized statistical figures with black text and accessible colors."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": "black", "axes.linewidth": .6, "lines.linewidth": .9,
        "axes.spines.right": False, "axes.spines.top": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig, folder, name):
    for ext in ("pdf", "png", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(folder / (name + "." + ext), dpi=300, **options)
    plt.close(fig)


def combination_figure(rows, pool, folder):
    if not rows:
        return
    groups = list(dict.fromkeys((r["dataset"], r["group"]) for r in rows))
    panels = list(dict.fromkeys(r["panel"] for r in rows))
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(7, max(4.5, len(panels) * .22 + 2.3)),
                             gridspec_kw={"width_ratios": [1.4] + [1] * len(groups)})
    fig.subplots_adjust(left=.065, right=.99, bottom=.30, top=.79, wspace=.3)
    membership = axes[0]
    example = {r["panel"]: set(r["reference_genes"].split("|")) for r in rows}
    for i, panel in enumerate(panels):
        for j, gene in enumerate(pool):
            membership.scatter(j, i, s=17, color="black" if gene in example[panel] else "#DDDDDD")
    membership.set_xticks(range(len(pool)), pool, rotation=65, ha="left")
    membership.xaxis.tick_top()
    membership.tick_params(axis="x", length=0, pad=5)
    membership.set_yticks(range(len(panels)), panels)
    membership.set_xlim(-.6, len(pool) - .4)
    membership.set_ylim(len(panels) - .5, -.5)
    membership.set_xlabel("Reference members", labelpad=12)
    for ax in axes[1:]:
        ax.set_ylim(membership.get_ylim())
        ax.set_yticks([])
        ax.axvline(0, color="black", linestyle="--", linewidth=.6)
    for j, ((dataset, group), ax) in enumerate(zip(groups, axes[1:])):
        block = [r for r in rows if (r["dataset"], r["group"]) == (dataset, group)]
        color = COLORS[j % len(COLORS)]
        for i, row in enumerate(block):
            ax.plot([row["bonferroni_low"], row["bonferroni_high"]], [i, i], color="#999999", linewidth=.6)
            ax.plot([row["bootstrap_low"], row["bootstrap_high"]], [i, i], color=color, linewidth=1.7)
            ax.scatter(row["estimate"], i, s=13, color=color, marker=["o", "s", "^", "D"][j % 4], zorder=3)
        ax.set_title(dataset.replace("Marta-Ariza", "Martá-Ariza").replace(" 2022", "") + "\n" + group + f"  n = {block[0]['n']}", pad=24)
        ax.set_xlabel("Relative score\n(log2)")
        ax.locator_params(axis="x", nbins=3)
    fig.legend([Line2D([], [], color=COLORS[0], lw=1.7), Line2D([], [], color="#999999", lw=.6)],
               ["95% donor bootstrap interval", "Bonferroni t interval within the panel family"],
               loc="lower center", bbox_to_anchor=(.5, .055), ncol=1, frameon=False, fontsize=7)
    fig.text(.5, .015, "All panels within a group use the same complete donors. Horizontal scales vary by group.", ha="center", fontsize=6.5)
    save(fig, folder, "Fig1_fixed_size_reference_panels")


def proteome_figure(rows, folder):
    if not rows:
        return
    groups = list(dict.fromkeys((r["dataset"], r["group"]) for r in rows))
    categories = ["positive_under_all", "negative_under_all", "point_sign_change", "touches_zero"]
    labels = ["Positive under all", "Negative under all", "Point sign change", "Touches zero"]
    colors = ["#0072B2", "#E69F00", "#CC79A7", "#BBBBBB"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.18, right=.98, bottom=.28, top=.82, wspace=.28)
    for i, key in enumerate(groups):
        block = {r["category"]: r for r in rows if (r["dataset"], r["group"]) == key}
        offset = 0
        for name, color in zip(categories, colors):
            width = block[name]["fraction"] * 100
            left.barh(i, width, left=offset, height=.55, color=color, edgecolor="white", linewidth=.5)
            offset += width
        change = block["point_sign_change"]
        right.plot([100 * change["bootstrap_low"], 100 * change["bootstrap_high"]], [i, i], color="black")
        right.scatter(100 * change["fraction"], i, color=colors[2], s=24, edgecolor="black", linewidth=.5)
        right.text(101, i, f"{change['count']}/{change['n_proteins']}", va="center", ha="right", fontsize=7)
    left.set_yticks(range(len(groups)), [d.replace(" 2022", "") + "\n" + g for d, g in groups])
    right.set_yticks([])
    for ax in (left, right):
        ax.set_ylim(len(groups) - .5, -.5)
        ax.set_xlim(0, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Proteins (%)")
    left.set_title("A   Direction across measured references", loc="left")
    right.set_title("B   Sign change and donor uncertainty", loc="left")
    fig.legend([Patch(facecolor=c) for c in colors], labels, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(.5, .04), fontsize=7)
    fig.text(.5, .012, "Point signs are descriptive. Intervals resample donors and hold the observed protein lists fixed.", ha="center", fontsize=6.5)
    save(fig, folder, "Fig2_proteome_relative_direction")


def secondary_figure(coverage, estimates, config, folder):
    groups = list(dict.fromkeys((r["dataset"], r["group"]) for r in coverage))
    genes = list(dict.fromkeys(config["targets"] + [g for v in config["references"].values() for g in v]))
    matrix = np.zeros((len(genes), len(groups)))
    text = {}
    for row in coverage:
        i, j = genes.index(row["gene"]), groups.index((row["dataset"], row["group"]))
        matrix[i, j] = row["paired_fraction"]
        text[i, j] = f"{row['n_paired']}/{row['n_donors']}"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.28, wspace=.8)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto", alpha=.4)
    for (i, j), label in text.items():
        ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
    ax.set_yticks(range(len(genes)), genes)
    ax.set_xticks(range(len(groups)), [d.replace("Marta-Ariza", "Martá").replace(" 2022", "") + "\n" + g for d, g in groups], rotation=40, ha="right", fontsize=6.7)
    ax.set_title("A   Exact-list paired coverage", loc="left")
    block = [r for r in estimates if r["scope"] == "per_reference"]
    if block:
        labels = []
        for i, row in enumerate(block):
            bx.plot([row["t_low"], row["t_high"]], [i, i], color="black")
            bx.scatter(row["estimate"], i, s=25, color=COLORS[0])
            labels.append(row["dataset"].replace(" 2022", "") + " " + row["group"] + f"\nn = {row['n']}  " + row["reference"].replace("_", " "))
        bx.set_yticks(range(len(block)), labels, fontsize=6.8)
        bx.set_ylim(len(block) - .5, -.5)
        bx.axvline(0, linestyle="--", linewidth=.6, color="black")
        bx.set_xlabel("Target minus reference (log2)")
    else:
        bx.text(.5, .5, "Paired coverage is insufficient\nfor the specified comparisons", ha="center", va="center", transform=bx.transAxes)
        bx.set_axis_off()
    bx.set_title("B   Available paired comparisons", loc="left")
    fig.text(.5, .95, config["name"], ha="center", fontsize=8)
    fig.text(.5, .015, "Lists remain fixed. Missing components are reported. Small donor sets have limited precision.", ha="center", fontsize=6.5)
    save(fig, folder, "Fig3_secondary_application_coverage")

