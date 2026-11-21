"""Generate the figures revised for the v10 manuscript from indexed result files."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

from paper4.publication.atlas import common, reference_plane, virtual_plaque
from paper4.publication.cases import GROUPS, REFS, point
from paper4.publication.decision_art import workflow
from paper4.publication.evidence_figures import (
    BLUE, GOLD, TEAL, VIOLET, appearance, matched_panels, panel, records, reference_drivers, signed, tab,
)
from paper4.publication.main_figures import plot_simulations, plot_tissue
from paper4.source_inputs import check_file, download, verify_output
from .figures import external_figure, methods_figure, simulation_figure


def read(path, name):
    return pd.read_csv(path / f"{name}.csv", float_precision="round_trip").to_dict("records")


def save(fig, folder, stem):
    for t in fig.findobj(mpl.text.Text):
        t.set_color("black")
    for ext in ("pdf", "png", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(folder / f"{stem}.{ext}", dpi=450 if ext != "png" else 360,
                    facecolor="white", **options)
    plt.close(fig)


def compare_panels(original, stage):
    old = {(r["application"], r["dataset"], r["group"], r["comparison"]): r for r in original}
    new = {(r["application"], "Leitner", r["group"], r["comparison"]): r for r in stage}
    fit = old | new
    fig = plt.figure(figsize=(7, 5.60))
    a, b = (fig.add_axes([.347, y, .607, h]) for y, h in ((.609,.257),(.146,.281)))
    panel(fig, .055, .961, "A", "CTSD and TPP1 on the same five references")
    panel(fig, .055, .519, "B", "APOE across five complete groups")
    fig.text(.95,.928,"Simultaneous 95% CIs within each group",ha="right",fontsize=7.8)
    def draw(ax, key, y, color, marker):
        r = fit[key]
        point(ax, float(r["mean"]), y, float(r["simultaneous_low"]), float(r["simultaneous_high"]), color, marker)
    for i, (ref, _) in enumerate(REFS):
        for group, y, color, marker in (("AD",4-i+.15,BLUE,"o"),("MCI",4-i-.15,TEAL,"D")):
            draw(a, ("lysosomal_anchor","Leitner",group,ref), y, color, marker)
    counts = [8,7,5,10,2]
    a.set(yticks=range(4,-1,-1),yticklabels=[f"{r[1]} ({n})" for r,n in zip(REFS,counts)],
          ylim=(-.55,4.55),xlim=(-.26,1.08),xticks=[-.2,0,.2,.4,.6,.8,1.0])
    handles = [Line2D([],[],color=c,marker=m,ms=5,label=name)
               for c,m,name in [(BLUE,"o","AD (53 donors)"),(TEAL,"D","MCI (36 donors)")]]
    fig.legend(handles=handles,ncol=2,loc="center",bbox_to_anchor=(.325,.906),fontsize=7.75,
               handlelength=1.4,columnspacing=1.05)
    studies = [GROUPS[0],("Leitner","MCI","Leitner MCI"),*GROUPS[1:3],("Marta-Ariza","DS","Martá-Ariza DS")]
    apoe = [("apod_clu","APOD / CLU (2)",BLUE,"o"),("apoa1_apod_clu","APOA1 / APOD / CLU (3)",TEAL,"s"),
            ("lrp1_sort1","LRP1 / SORT1 (2)","#A65316","^")]
    for i, (dataset, group, _) in enumerate(studies):
        for shift, (ref, _, color, marker) in zip((.22,0,-.22), apoe):
            draw(b,("apoe_lipid",dataset,group,ref),4-i+shift,color,marker)
    b.set(yticks=[4,3,2,1,0],yticklabels=[f"{label} ({n})" for (_,_,label),n in zip(studies,(53,36,20,20,20))],
          xlim=(-.12,3.92),ylim=(-.52,4.52),xticks=[0,1,2,3])
    for ax in (a,b):
        ax.set_xlabel("Target minus reference regional change (log₂)",labelpad=2)
        ax.axvline(0,color="black",lw=.75,zorder=0)
        ax.axvline(.2,color="#59646A",lw=.9,ls=":",zorder=0)
        ax.tick_params(axis="y",length=0,pad=5,labelsize=7.9)
        ax.spines["left"].set_visible(False)
    h=[Line2D([],[],marker=m,color=c,lw=1,ms=4.9,label=lab) for _,lab,c,m in apoe]
    fig.legend(handles=h,loc="lower center",bbox_to_anchor=(.50,.012),ncol=3,fontsize=7.7,
               handlelength=1.25,columnspacing=1.24)
    return fig


def direction_and_decision(archive, counts, moves):
    fig = plt.figure(figsize=(7,5.78))
    panel(fig,.054,.954,"A","Opposite group means across effect margins")
    panel(fig,.054,.470,"B","Reversals at ±0.2 log₂")
    panel(fig,.560,.470,"C","Leitner AD decision transitions")
    a=fig.add_axes([.125,.572,.832,.315])
    b=fig.add_axes([.247,.151,.258,.230])
    c=fig.add_axes([.708,.151,.230,.232])
    old=tab(archive,52)
    colors=[BLUE,TEAL,"#A65316",VIOLET]
    for idx,((dataset,group,label),color,style) in enumerate(zip(GROUPS,colors,("-","--","-.",":"))):
        rows=sorted((r for r in old if (r["dataset"],r["group"])==(dataset,group)),
                    key=lambda r:float(r["margin_log2"]))
        x,y,lo,hi=np.array([[float(r[k]) for r in rows]
                            for k in ("margin_log2","fraction","bootstrap_low","bootstrap_high")])
        a.fill_between(x,100*lo,100*hi,color=color,alpha=.12,lw=0)
        a.plot(x,100*y,color=color,ls=style,lw=1.55,label=label)
        focus,=[r for r in counts if (r["dataset"],r["group"],r["scope"],r["margin_log2"])==
                (dataset,group,"observed",.2)]
        n=focus["n_estimable"]
        t0=100*focus["point_flip_count"]/n
        t1=100*focus["strict_flip_count"]/n
        b.hlines(3-idx,t1,t0,color="#AEBDC4",lw=1.2)
        b.scatter([t0],[3-idx],facecolor="white",edgecolor=color,s=30,marker="o",linewidth=1.05,zorder=3)
        b.scatter([t1],[3-idx],color="#203B45",s=21,marker="D",zorder=3)
        b.text(t0+.9,3-idx,f"{t0:.1f}%",va="center",fontsize=7.45)
        strict_label = f"{t1:.2f}%" if t1 else "0%"
        b.text(t1+.72,3-idx,strict_label,va="center",fontsize=7.45)
    a.axvline(.2,color="#5E6970",ls=":",lw=.9)
    a.set(xlim=(0,.4),ylim=(0,67),xticks=[0,.1,.2,.3,.4],yticks=[0,20,40,60],
          xlabel="Effect margin δ (log₂)",ylabel="Background proteins (%)")
    a.legend(loc="upper right",bbox_to_anchor=(1,.995),ncol=2,fontsize=7.45,columnspacing=1.0,handlelength=1.35)
    fig.text(.127,.505,"Bands are pointwise 95% donor-bootstrap intervals.",fontsize=7.55)
    b.set(xlim=(-1.0,29),ylim=(-.48,3.48),xticks=[0,10,20],yticks=range(3,-1,-1),
          yticklabels=["Leitner AD (2,157)","Martá EOAD (1,473)","Martá LOAD (1,615)","Drummond (1,480)"],
          xlabel="Measured background (%)")
    b.tick_params(axis="y",length=0,labelsize=7.5)
    b.spines["left"].set_visible(False)
    legend=[Line2D([],[],ls="",marker="o",mfc="white",mec=BLUE,label="Opposite means"),
            Line2D([],[],ls="",marker="D",color="#203B45",label="Opposite supported directions")]
    b.legend(handles=legend,loc="upper left",bbox_to_anchor=(-.75,-.29),fontsize=7.35,handletextpad=.35)
    state=["positive_supported","unresolved","negative_supported"]
    names=["Positive","Unresolved","Negative"]
    chosen={(r["from_state"],r["to_state"]):r["count"] for r in moves if
            (r["dataset"],r["group"],r["scope"],r["margin_log2"],r["from_reference"],r["to_reference"])==
            ("Leitner","AD","observed",.2,"v_atpase","ec3_lysosomal_lumen")}
    values=np.array([[chosen[x,y] for y in state] for x in state])
    c.imshow(np.sqrt(values),cmap=mpl.colors.LinearSegmentedColormap.from_list("count",["#F6F9FA","#9CBBC6"]),
             aspect="auto",interpolation="nearest")
    for i,j in np.ndindex(3,3):
        c.text(j,i,f"{values[i,j]:,}",ha="center",va="center",fontsize=8.2,
               fontweight="bold" if (i,j) in [(0,1),(0,2)] else "normal")
    c.set(xticks=range(3),xticklabels=names,yticks=range(3),yticklabels=names,
          xlabel="To EC 3 luminal enzymes",ylabel="From V-ATPase")
    c.tick_params(length=0,labelsize=7.1)
    for sp in c.spines.values(): sp.set_visible(False)
    return fig


def membership_and_weights(archive, gls):
    fig=reference_drivers(archive)
    height=fig.get_figheight()
    foot=1.83
    fig.set_figheight(height+foot)
    scale=height/(height+foot)
    shift=1-scale
    for ax in fig.axes:
        x,y,w,h=ax.get_position().bounds
        ax.set_position([x,shift+y*scale,w,h*scale])
    for text in fig.texts:
        x,y=text.get_position()
        text.set_position((x,shift+y*scale))
    for artist in fig.artists:
        artist.set_ydata(shift+np.asarray(artist.get_ydata())*scale)
    total=[r for r in tab(archive,26) if (r["dataset"],r["group"],r["endpoint"])==("Leitner","AD","estimate")]
    delta=next(float(r["estimate"]) for r in total if r["coverage_pct"]=="80")-next(
        float(r["estimate"]) for r in total if r["coverage_pct"]=="100")
    fig.text(.952,shift+.885*scale,f"Total change {signed(delta)} log₂",ha="right",fontsize=7.7,weight="bold")
    panel(fig,.046,.315,"C","Reference weighting on the same 53 donors")
    a=fig.add_axes([.325,.106,.605,.158])
    rows={r["method"]:r for r in gls if (r["dataset"],r["group"],r["comparison"],r["donor_scope"])==
          ("Leitner","AD","ec3_lysosomal_lumen","shared_family")}
    methods=["equal_weight","nonnegative_shrinkage_leave_one_donor_out","unconstrained_inverse_covariance"]
    for i,(method,color,symbol) in enumerate(zip(methods,(BLUE,TEAL,"#A65316"),("o","s","D"))):
        estimate=rows[method]["estimate_log2"]
        a.scatter(estimate,2-i,c=color,marker=symbol,s=28,zorder=3)
        a.text(estimate+.013,2-i,signed(estimate),va="center",fontsize=7.7)
    a.axvline(.2,color="#68787C",lw=.8,ls=":",zorder=0)
    a.set(xlim=(0,.375),ylim=(-.41,2.41),yticks=[2,1,0],
          yticklabels=["Equal weights","Non-negative leave-one-out","Inverse covariance"],
          xticks=[0,.1,.2,.3],xlabel="CTSD and TPP1 minus five luminal enzymes (log₂)")
    a.tick_params(axis="y",length=0)
    a.spines["left"].set_visible(False)
    return fig


def graphical_abstract(archive):
    lookup={r["comparison"]:float(r["mean"]) for r in records(archive,"Data_S1/fixed_estimates.csv")
            if (r["dataset"],r["group"],r["application"])==("Leitner","AD","lysosomal_anchor")}
    fig=plt.figure(figsize=(3.25,1.72))
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,3.25),ylim=(0,1.72))
    ax.axis("off")
    for x, y, text, face, edge in ((.42,1.14,"Plaque","#FAE4BC","#A76C19"),
                                   (.42,.46,"Adjacent","#D6ECEA","#347C79")):
        ax.add_patch(Circle((x,y),.195,fc=face,ec=edge,lw=.75))
        offsets=np.asarray([[-.08,.032],[.061,.015],[-.045,-.072],[.055,-.079],[0,.10]])
        for i,(dx,dy) in enumerate(offsets):
            ax.plot([x+dx-.028,x+dx+.028],[y+dy,y+dy+(.030 if i%2 else -.023)],c=edge,lw=.68)
        ax.text(x,y-.245,text,ha="center",va="top",fontsize=7.5)
    ax.annotate("",xy=(1.10,.83),xytext=(.70,.83),arrowprops=dict(arrowstyle="->",lw=.85,color="#466065"))
    ax.text(1.17,1.55,"CTSD and TPP1",fontsize=8.45,weight="bold")
    for y,name,comparison,fc,edge in [(1.05,"V-ATPase","v_atpase","#E1EFF6",BLUE),
                                      (.45,"EC 3 luminal enzymes","ec3_lysosomal_lumen","#E2F1EC",TEAL)]:
        ax.add_patch(FancyBboxPatch((1.16,y),1.93,.33,boxstyle="round,pad=.015,rounding_size=.05",
                                    facecolor=fc,edgecolor=edge,lw=.6))
        ax.text(1.23,y+.165,name,va="center",fontsize=7.1)
        ax.text(3.0,y+.165,signed(lookup[comparison]),va="center",ha="right",fontsize=8.0,weight="bold")
    ax.text(1.17,.16,"Target change − mean reference change (log₂)",fontsize=6.9)
    return fig


def generate(base, followup, v10, structure, destination):
    verify_output(v10)
    verify_output(followup)
    main=destination/"figures"
    supp=destination/"supporting/figures"
    main.mkdir(parents=True,exist_ok=True)
    supp.mkdir(parents=True,exist_ok=True)
    tables=v10/"tables"
    external=read(tables,"external_donor_scores")
    samples=[SimpleNamespace(group=g,donors=sorted({r["donor"] for r in external if r["group"]==g and
                          r["application"]=="mapt_tangle"})) for g in ("Advanced AD","Intermediate AD")]
    methods={name:read(tables,name) for name in ("bayes_joint","bayes_model_average","gls_estimates")}
    simulations={name:read(tables,name) for name in ("ssr_roc","ssr_performance")}
    appearance()
    with ZipFile(base) as archive, TemporaryDirectory(prefix="paper4-v10-art-") as temp:
        work=Path(temp)
        common.DATA=common.FIG=work
        for name in ("paired_geometry.csv","paired_summary.csv","pump_panel.csv"):
            (work/name).write_bytes(archive.read("Data_S2/"+name))
        manifest = Path(__file__).resolve().parents[1] / "data_manifest.json"
        structure_record = next(item for item in json.loads(manifest.read_text(encoding="utf-8"))
                                if item["name"] == "6WM2.cif.gz")
        if structure is None:
            download(work, structure_record)
        else:
            check_file(structure, structure_record)
            copyfile(structure, work / structure_record["name"])
        plot_tissue(archive,main,emphasize_data=True)
        save(compare_panels(records(archive,"Data_S1/fixed_estimates.csv"),read(followup,"phenotype_estimates")),
             main,"Fig2_cases")
        reference_plane.run(highlight=True)
        for ext in ("pdf","png","tiff"):
            copyfile(work/f"03_reference_plane.{ext}",main/f"Fig3_paired_geometry.{ext}")
        save(direction_and_decision(archive,read(followup,"background_decisions"),read(followup,"panel_transitions")),
             main,"Fig4_background")
        save(membership_and_weights(archive,methods["gls_estimates"]),main,"Fig5_reference_drivers")
        save(matched_panels(archive,explicit_tail=True),main,"Fig6_matched_panels")
        plot_simulations(archive,main,separate_missingness=True)
        workflow(main,shared_wording=True)
        save(graphical_abstract(archive),main,"TOC_graphic")
        virtual_plaque.run(legible_scale=True)
        for ext in ("pdf","png","tiff"):
            copyfile(work/f"01_virtual_plaque.{ext}",supp/f"FigS16_virtual_section.{ext}")
        external_figure(work,samples,read(tables,"external_effects"),external)
        simulation_figure(work,simulations)
        methods_figure(work,methods,.5)
        for old,new in [("01_MAPT_external","FigS18_tangle_application"),("02_SSR_held_out","FigS19_ssr_holdout"),
                        ("03_Bayesian_GLS","FigS20_weighting_posterior")]:
            for ext in ("pdf","png","tiff"):
                copyfile(work/f"{old}.{ext}",supp/f"{new}.{ext}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Indexed Supporting_Data.zip from run.py")
    parser.add_argument("--followup", type=Path, required=True, help="Completed publication_followup output")
    parser.add_argument("--v10", type=Path, required=True, help="Completed run_v10.py output")
    parser.add_argument("--structure", type=Path, help="Local 6WM2.cif.gz, otherwise downloaded and verified")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the revised figures")
    args = parser.parse_args()
    generate(args.base, args.followup, args.v10, args.structure, args.out)
    print(f"Generated 13 figure sets in {args.out.resolve()}")


if __name__=="__main__":
    main()
