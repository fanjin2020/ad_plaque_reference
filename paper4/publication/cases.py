"""Draw the updated main comparison and the ordered specification plot from Data S1."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


GROUPS = [("Leitner","AD","Leitner AD"), ("Marta-Ariza","EOAD","Martá-Ariza EOAD"),
          ("Marta-Ariza","LOAD","Martá-Ariza LOAD"), ("Drummond 2022","eoAD","Drummond EOAD")]
REFS = [("v_atpase","V-ATPase subunits"),("lysosomal_lumen_hydrolases","GO luminal hydrolases"),
        ("ec3_lysosomal_lumen","EC 3 luminal enzymes"),("ec3_lysosome","EC 3 lysosomal enzymes"),
        ("ec34_lysosome","EC 3.4 lysosomal enzymes")]
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#00699B","#B94B00","#00735E","#775284","#425767"
COLORS = [BLUE, GREEN, ORANGE, PURPLE, GRAY]
SHAPES = ["s","o","^","D","v"]


def save(fig, directory, name):
    for ext in ("png","tiff","pdf"):
        kw = {"pil_kwargs":{"compression":"tiff_lzw"}} if ext=="tiff" else {}
        fig.savefig(directory/(name+"."+ext), dpi=400, facecolor="white", **kw)
    plt.close(fig)


def point(ax,x,y,lo,hi,color,marker):
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,mec="black",mew=.5,
                ms=4.8,capsize=2.2,lw=1.15,zorder=3)


def cases(rows, output):
    lookup = {(r["application"],r["dataset"],r["group"],r["comparison"]):r for r in rows}
    fig=plt.figure(figsize=(6.55,5.25))
    a=fig.add_axes([.32,.595,.64,.265]); b=fig.add_axes([.32,.155,.64,.286])
    for y,letter,title in [(0.972,"A","Lysosomal hydrolases in 53 paired Leitner AD donors"),
                           (.530,"B","APOE in four groups with complete named panels")]:
        fig.text(.025,y,letter,va="top",weight="bold",size=11)
        fig.text(.08,y-.002,title,va="top",weight="bold",size=8.8)
    def show(ax,key,y,color,marker):
        r=lookup[key]
        point(ax,float(r["mean"]),y,float(r["simultaneous_low"]),float(r["simultaneous_high"]),color,marker)
    for i,(ref,label) in enumerate(REFS):
        show(a,("lysosomal_anchor","Leitner","AD",ref),4-i,COLORS[i],SHAPES[i])
    a.set(yticks=range(4,-1,-1),yticklabels=[r[1] for r in REFS],ylim=(-.55,4.55),xlim=(-.15,1.05),
          xticks=[0,.2,.4,.6,.8,1.])
    studies=GROUPS[:3]+[("Marta-Ariza","DS","Martá-Ariza DS")]
    apoe=[("apod_clu","APOD / CLU",BLUE,"o"),("apoa1_apod_clu","APOA1 / APOD / CLU",GREEN,"s"),
          ("lrp1_sort1","LRP1 / SORT1",ORANGE,"^")]
    for i,(ds,g,label) in enumerate(studies):
        for delta,(ref,_,color,mark) in zip((.22,0,-.22),apoe):
            show(b,("apoe_lipid",ds,g,ref),3-i+delta,color,mark)
    b.set(yticks=[3,2,1,0],yticklabels=[label+("  (53)" if ds=="Leitner" else "  (20)") for ds,_,label in studies],
          xlim=(-.12,3.92),ylim=(-.52,3.52),xticks=[0,1,2,3])
    for ax in [a,b]:
        ax.set_xlabel(r"Target minus reference regional change (log$_2$)",labelpad=2)
        ax.axvline(0,color="black",lw=.75,zorder=0)
        ax.axvline(.2,color="#555555",lw=.85,ls=":",zorder=0)
        ax.tick_params(axis="y",length=0,pad=5)
        ax.spines["left"].set_visible(False)
    handles=[Line2D([],[],marker=m,color=c,lw=1.0,mec="black",mew=.5,ms=4.7,label=lab) for _,lab,c,m in apoe]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.01),ncol=3,fontsize=7.4,
               handlelength=1.25,columnspacing=1.4,frameon=False)
    save(fig,output,"Fig2_cases")


def specification(rows, output):
    fig=plt.figure(figsize=(6.55,5.05))
    a=fig.add_axes([.275,.53,.69,.345]); b=fig.add_axes([.275,.105,.69,.33],sharex=a)
    fig.text(.025,.978,"A",va="top",weight="bold",size=11)
    fig.text(.075,.975,"CTSD/TPP1 effects ordered by their estimated magnitude",va="top",weight="bold",size=8.6)
    fig.text(.025,.479,"B",va="top",weight="bold",size=11)
    fig.text(.075,.477,"Study group and reference underlying each estimate",va="top",weight="bold",size=8.6)
    group_map={(ds,g):i for i,(ds,g,_) in enumerate(GROUPS)}
    for r in rows:
        i=int(r["rank"]); v,lo,hi=(float(r[k]) for k in ("estimate","bonferroni_low","bonferroni_high"))
        j=group_map[(r["dataset"],r["group"])]; k=[v[0] for v in REFS].index(r["reference"])
        color=COLORS[j]; marker=SHAPES[j]
        a.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt=marker,color=color,mec="black",mew=.4,ms=4.0,capsize=1.5,lw=.8,zorder=3)
        b.scatter(i,8-j,marker=marker,s=18,c=color,edgecolors="black",linewidths=.35,zorder=3)
        b.scatter(i,4-k,marker="o",s=12,c="black",zorder=3)
    a.axhline(0,color="black",lw=.7);a.axhline(.2,color="#555555",ls=":",lw=.75)
    a.set_ylabel(r"Relative change (log$_2$)")
    a.tick_params(axis="x",labelbottom=False,length=0)
    a.grid(axis="y",color="#eeeeee",lw=.55)
    names=[g[2] for g in GROUPS]+[r[1] for r in REFS]
    b.set_yticks(range(8,-1,-1),names)
    b.tick_params(axis="y",labelsize=7,length=0,pad=5)
    b.set(xticks=[1,5,10,15,20],xlim=(.3,20.7),ylim=(-.55,8.55),xlabel="Rank among the 20 observed specifications")
    b.axhline(4.5,color="#9da7ac",lw=.65);b.spines["left"].set_visible(False)
    b.grid(axis="x",color="#e9e9e9",lw=.55)
    save(fig,output,"FigS8_specifications")


