"""Figures and exact results generated from the analysis tables."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper4.four_part.study_tables import ORDER

REFS = {"v_atpase":"V-ATPase", "lysosomal_lumen_hydrolases":"GO lumen hydrolases",
        "ec3_lysosomal_lumen":"EC 3 lumen", "ec3_lysosome":"EC 3 lysosome", "ec34_lysosome":"EC 3.4 lysosome"}
LIPID = {"regional":"Plaque vs adjacent", "apod_clu":"APOD + CLU", "apoa1_apod_clu":"APOA1 + APOD + CLU",
         "lrp1_sort1":"LRP1 + SORT1"}
STUDY = {"Marta-Ariza":"Martá-Ariza", "Drummond 2022":"Drummond 2022", "Leitner":"Leitner"}
BLUE, ORANGE = "#0072B2", "#B95713"


def save(fig, folder, name):
    for suffix in ("pdf","png","tiff"):
        kwargs = {"pil_kwargs":{"compression":"tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(folder / f"{name}.{suffix}", dpi=350, facecolor="white", **kwargs)
    plt.close(fig)


def interval(ax, y, mean, low, high, color, mark="o", hollow=False):
    ax.errorbar(mean, y, xerr=[[max(0.,mean-low)],[max(0.,high-mean)]], fmt=mark, color=color,
                markerfacecolor="white" if hollow else color, markersize=4, elinewidth=.9,
                capsize=2.3, capthick=.8, zorder=3)


def figures(output, tables):
    folder = Path(output)/"figures"
    folder.mkdir()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "text.color":"black",
        "axes.labelcolor":"black", "axes.titlecolor":"black", "xtick.color":"black", "ytick.color":"black",
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.7, "pdf.fonttype":42})
    fig, axes = plt.subplots(2,2,figsize=(7,5.0), sharex=True)
    fig.subplots_adjust(left=.205,right=.94,bottom=.18,top=.84,wspace=.34,hspace=.40)
    for j, ((dataset, group, n), ax) in enumerate(zip(ORDER,axes.flat)):
        rows = [r for r in tables["reference_profiles"] if (r["dataset"],r["group"]) == (dataset,group)]
        lookup = {(r["reference"],r["definition"]):r for r in rows}
        ax.axvline(0, color=".60", lw=.7, zorder=0)
        for i, name in enumerate(REFS):
            for tag, color, mark, shift in (("observed",BLUE,"o",.13),("harmonized_with_singletons",ORANGE,"s",-.13)):
                row = lookup[name,tag]
                singleton = row["n_members"] == 1
                interval(ax,4-i+shift,row["estimate"],row["bonferroni_low"],row["bonferroni_high"],
                         color,mark,singleton)
            a, b = lookup[name,"observed"]["n_members"], lookup[name,"harmonized_with_singletons"]["n_members"]
            ax.text(1.012,4-i,f"{a}/{b}",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=7)
        ax.set_title(f"{STUDY[dataset]} · {group}  (n = {n})",fontsize=8.6,loc="left",pad=8,fontweight="bold")
        ax.set_yticks(range(5), list(REFS.values())[::-1] if j%2 == 0 else [""]*5)
        ax.tick_params(axis="y",length=0)
        ax.set_ylim(-.5,4.5)
        ax.grid(axis="x", color=".91", lw=.5)
    fig.text(.07,.944,"CTSD/TPP1 across biological references",fontsize=13,fontweight="bold")
    fig.text(.07,.902,"95% simultaneous intervals for five contrasts. Counts at right give study-specific / shared members.",fontsize=7.7)
    fig.text(.57,.100,"Target minus reference, log₂",ha="center")
    handles=[Line2D([],[],marker="o",lw=0,color=BLUE,label="Study-specific members"),
             Line2D([],[],marker="s",lw=0,color=ORANGE,label="Members shared by all four groups"),
             Line2D([],[],marker="s",lw=0,color=".35",mfc="white",label="One protein")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.008),ncol=3,frameon=False,fontsize=7.2)
    save(fig,folder,"01_reference_membership_profiles")

    fig, axes = plt.subplots(2,1,figsize=(7,4.95),sharex=True)
    fig.subplots_adjust(left=.26,right=.87,top=.84,bottom=.17,hspace=.58)
    for ax, dataset in zip(axes,("Marta-Ariza","Drummond 2022")):
        full = dataset == "Marta-Ariza"
        src = tables["fixed_estimates"] if full else tables["panel_specific"]
        by_ref = {r["comparison"]:r for r in src if (r["application"],r["dataset"],r["group"]) ==
                  ("apoe_lipid",dataset,"DS")}
        title = "Martá-Ariza · same 20 donors across all comparisons" if full else "Drummond 2022 · panel-specific donor sets"
        ax.set_title(title,loc="left",fontsize=9.1,fontweight="bold")
        ax.axvline(0,color=".6",lw=.7)
        ax.set_yticks(range(4),list(LIPID.values())[::-1])
        ax.tick_params(axis="y",length=0)
        for i, name in enumerate(LIPID):
            row = by_ref[name]
            yy = 3-i
            if row["status"] == "estimable":
                interval(ax,yy,row["mean"],row["simultaneous_low"],row["simultaneous_high"],
                         BLUE if full else ORANGE, "o" if full else "s")
            else:
                ax.text(.02,yy,"Below the 5-donor minimum",transform=ax.get_yaxis_transform(),va="center",fontsize=7.4)
            ax.text(1.02,yy,f"n = {row['n']}",transform=ax.get_yaxis_transform(),va="center",fontsize=7.6)
        ax.set_ylim(-.6,3.6)
        ax.grid(axis="x",color=".92",lw=.5)
    fig.text(.06,.946,"APOE in Down syndrome plaque tissue",fontsize=13,fontweight="bold")
    axes[1].set_xlabel("Mean donor-paired contrast, log₂")
    fig.text(.06,.037,"Martá-Ariza intervals cover four contrasts. Drummond intervals cover three planned relative contrasts, with a separate regional interval.",fontsize=6.5)
    save(fig,folder,"02_DS_APOE")

    fig, ax = plt.subplots(figsize=(6.6,3.3))
    fig.subplots_adjust(left=.29,right=.78,top=.77,bottom=.23)
    for j,r in enumerate(tables["background_summary"]):
        yy = 3-j
        interval(ax,yy,100*r["fraction"],100*r["bootstrap_low"],100*r["bootstrap_high"],BLUE)
        ax.text(1.04,yy,f"{r['count']:,} / {r['n_proteins']:,}",transform=ax.get_yaxis_transform(),va="center",fontsize=8)
    ax.set_yticks(range(4),[f"{STUDY[s]} · {g}  (n = {n})" for s,g,n in ORDER][::-1])
    ax.set_ylim(-.45,3.55)
    ax.set_xlim(left=0)
    ax.set_xlabel("Background proteins crossing both margins (%)",labelpad=6)
    ax.grid(axis="x",color=".9",lw=.5)
    ax.tick_params(axis="y",length=0)
    fig.text(.065,.91,"Direction changes across five measured references",fontsize=12,fontweight="bold")
    fig.text(.065,.83,"Strict margins of +0.2 and −0.2 log₂  ·  95% donor-bootstrap intervals",fontsize=8)
    fig.text(.065,.048,"Each denominator contains the completely measured, unambiguously mapped background proteins for that group.",fontsize=7.4)
    save(fig,folder,"03_background_direction_changes")


def write_summary(folder, tables):
    bg = tables["background_summary"]
    percentages = [100*r["fraction"] for r in bg]
    sentence = (f"Across four study groups, {min(percentages):.1f}–{max(percentages):.1f}% of completely "
        "measured background proteins had mean contrasts above +0.2 log₂ units against one reference "
        "and below −0.2 against another.")
    (folder/"abstract_candidate.txt").write_text(sentence+"\n", encoding="utf-8")
    lines = ["# 论文4 v6 后续复核", "", "## 唐氏综合征伴 AD 病理组", "",
        "下表为 Martá-Ariza 同一组完整供体的斑块相对邻近组织变化。区域比较直接使用 APOE 的变化，后三项减去具名参照的平均变化。区间对四项比较同时校正。", "",
        "| 比较 | 配对供体 | 平均值（log₂） | 同时 95% 区间 |",
        "|---|---:|---:|---:|"]
    ds = [r for r in tables["fixed_estimates"] if (r["application"],r["dataset"],r["group"]) ==
          ("apoe_lipid","Marta-Ariza","DS")]
    for r in ds:
        lines.append(f"| {LIPID[r['comparison']]} | {r['n']} | {r['mean']:+.3f} | {r['simultaneous_low']:+.3f} 至 {r['simultaneous_high']:+.3f} |")
    partial = next(r for r in tables["panel_specific"] if (r["application"],r["dataset"],r["group"],r["comparison"]) ==
                   ("apoe_lipid","Drummond 2022","DS","apod_clu"))
    lines += ["",f"Drummond 2022 的 DS 组中，APOE 对 APOD、CLU 有 {partial['n']} 名供体，"
        f"均值 {partial['mean']:+.3f}，三项预定相对比较的同时区间为 {partial['simultaneous_low']:+.3f} 至 {partial['simultaneous_high']:+.3f}。"
        "SORT1 无配对观测。两项研究间供体独立性待核实，所有结果按研究报告。", "",
        "Martá-Ariza DS 组的固定溶酶体五参照家族缺少 OCRL。八个泵亚基可比较 20 人，五个 EC 3 腔酶可比较 18 人。", "",
        "## 可测背景蛋白的方向变化", "", sentence, "",
        "| 组别 | 蛋白数 | 比例 | 95% 供体自助区间 |","|---|---:|---:|---:|"]
    for r in bg:
        lines.append(f"| {STUDY[r['dataset']]} {r['group']} | {r['count']}/{r['n_proteins']} | {r['fraction']:.1%} | {r['bootstrap_low']:.1%} 至 {r['bootstrap_high']:.1%} |")
    r = tables["equivalence"][0]
    adjusted = [s["p_holm"] for s in tables["genotype_estimates"] if s["adjustment_family"] == "original_AD_six_reference_changes"]
    lines += ["","## 参照成员及两项补充分析","",
        "四组依次为 Leitner AD、Martá-Ariza EOAD、Martá-Ariza LOAD 和 Drummond eoAD。每组均有八个完全配对的泵亚基。"
        "EC 3 腔酶在四组分别有 5、3、5、4 个，四组交集有 2 个。EC 3.4 溶酶体参照的交集只有 USP5，图中以空心符号标示。", "",
        f"在三组可分层的原 AD 数据中，ε4 对两个换参照分数差的六项探索性检验，最小 Holm 校正 p 为 {min(adjusted):.3f}。"
        "原始主参照分数的基因型组间差、DS 的两项检验和每组的未知数另见 genotype_estimates.csv 和 genotype_coverage.csv。", "",
        f"Leitner 的 {r['n']} 名 AD 供体中，CTSD/TPP1 相对五个 EC 3 腔酶的均值为 {r['mean']:+.4f}，"
        f"90% 区间 {r['ci90_low']:+.4f} 至 {r['ci90_high']:+.4f}，单对比 TOST p = {r['p_equivalence']:.3g}。"
        "±0.2 log₂ 为探索性幅度。", "", "## 文件", "",
        "tables/ 的 background_proteins.csv 给出每个背景蛋白的五个分数和两个极值。reference_profiles.csv 和 reference_membership.csv 记录构成图上每一点的具名蛋白、覆盖和供体数。fixed_estimates.csv 提供完整共同供体比较，panel_specific.csv 给出单套参照能估计的结果。", "",
        "figures/ 提供 PDF、PNG 和 350 dpi TIFF。protocols/ 保存本轮使用的两个协议，provenance.json 记录源数据和运行环境，validation.json 对照 v6 七张数值表。"]
    (folder/"研究结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
