"""Publication figures drawn directly from the indexed manuscript archive."""
import csv
import io

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

BLUE = "#3667A0"
TEAL = "#217E79"
GOLD = "#D48B39"
VIOLET = "#805D93"
DARK = "#3F535B"
PALE = "#E5EBEC"
GROUPS = (("Leitner", "AD", "Leitner AD"), ("Marta-Ariza", "EOAD", "Martá-Ariza EOAD"),
          ("Marta-Ariza", "LOAD", "Martá-Ariza LOAD"), ("Drummond 2022", "eoAD", "Drummond EOAD"))
REFERENCES = (
    ("v_atpase", "V-ATPase"), ("lysosomal_lumen_hydrolases", "GO lumen"),
    ("ec3_lysosomal_lumen", "EC 3 lumen"), ("ec3_lysosome", "EC 3 lysosome"),
    ("ec34_lysosome", "EC 3.4 lysosome"),
)
COLORS = (BLUE, TEAL, GOLD, VIOLET)
MARKS = ("o", "s", "^", "D")


def appearance():
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "text.color": "black",
        "axes.labelcolor": "black", "axes.labelsize": 8, "axes.linewidth": .65, "axes.edgecolor": "#69787D",
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "xtick.color": "black", "ytick.color": "black",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": .65, "ytick.major.width": .65,
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def records(archive, path):
    return list(csv.DictReader(io.StringIO(archive.read(path).decode("utf-8-sig"))))


def tab(archive, n):
    (path,) = [p for p in archive.namelist() if p.startswith(f"Table_S{n:02d}_")]
    return records(archive, path)


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=10.7, fontweight="bold", va="center")
    fig.text(x+.036, y, title, fontsize=8.8, fontweight="bold", va="center")


def signed(x):
    return f"{float(x):+.3f}".replace("-", "−")


def export(fig, folder, stem, crop=None):
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(folder / (stem + ".pdf"), dpi=450, bbox_inches=crop)
    fig.savefig(folder / (stem + ".png"), dpi=290, bbox_inches=crop)
    fig.savefig(folder / (stem + ".tiff"), dpi=450, bbox_inches=crop, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def reference_drivers(archive):
    fig = plt.figure(figsize=(7, 3.45))
    panel(fig, .046, .948, "A", "Omit one member of the GO lumen panel")
    panel(fig, .547, .948, "B", "Relax coverage from 100% to 80%")
    rows = sorted(
        (r for r in tab(archive, 20) if (r["dataset"], r["group"], r["reference"]) ==
         ("Leitner", "AD", "lysosomal_lumen_hydrolases")), key=lambda r: float(r["delta_log2"]),
    )
    assert len(rows) == 7
    ax = fig.add_axes([.145, .20, .339, .655])
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        effect = float(r["delta_log2"])
        c = VIOLET if effect < 0 else TEAL
        ax.barh(i, effect, .53, color=c, zorder=3)
        ax.text(effect+(.005 if effect >= 0 else -.005), i, signed(effect), ha="left" if effect >= 0 else "right",
                va="center", fontsize=6.7)
    ax.axvline(0, color=DARK, lw=.85)
    ax.set_ylim(-.55, 6.55)
    ax.set_xlim(-.164, .153)
    ax.set_xticks([-.12, -.06, 0, .06,.12])
    ax.set_yticks(y, [r["omitted"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change after omission (log₂)", labelpad=3)
    ax.grid(axis="x", color="#E9EEEF", lw=.5, zorder=0)

    source = {(int(r["coverage_pct"]), r["endpoint"]): r for r in tab(archive, 26)
              if (r["dataset"], r["group"]) == ("Leitner", "AD")}
    start, end = [float(source[i, "estimate"]["estimate"]) for i in (100, 80)]
    subset, member = [source[80, term] for term in ("donor_subset", "composition")]
    a, b = float(subset["estimate"]), float(member["estimate"])
    assert abs(start + a + b - end) < 1e-12
    bx = fig.add_axes([.608, .327, .355, .525])
    bases = [0, start, start+a, 0]
    heights = [start, a, b, end]
    bx.bar(range(4), heights, .61, bottom=bases, color=[BLUE, GOLD, TEAL, BLUE], linewidth=.55,
           edgecolor=[BLUE, "#A06119", TEAL, BLUE], zorder=3)
    for i, next_base in enumerate([start, start+a, end]):
        bx.plot([i+.31,i+.69], [next_base,next_base], color="#607075", linestyle="--", lw=.74)
    for i, (v, top) in enumerate(zip(heights, [start, start+a, end, end])):
        bx.text(i,top+.007,signed(v),ha="center",va="bottom",fontsize=7.2,fontweight="bold")
    bx.set_xlim(-.49,3.49); bx.set_ylim(0,.172)
    bx.set_xticks(range(4),["Five\nproteins","Donor\nsubset","Panel\nmembers","Seven\nproteins"])
    bx.set_yticks([0,.05,.10,.15]); bx.set_ylabel("Relative score (log₂)", labelpad=3)
    bx.grid(axis="y", color="#E7EEEE", lw=.55, zorder=0)
    for yy, title, r, color in [(.135,"Panel membership",member,TEAL),(.075,"Donor subset",subset,GOLD)]:
        fig.add_artist(Line2D([.558,.580],[yy,yy],c=color,lw=3.4,transform=fig.transFigure,solid_capstyle="butt"))
        fig.text(.592,yy,title,fontsize=6.9,va="center")
        fig.text(.966,yy,f'{signed(r["estimate"])}  [{signed(r["low"])}, {signed(r["high"])}]',
                 fontsize=6.8,va="center",ha="right")
    return fig


def matched_panels(archive, explicit_tail=False):
    names = ("both_arms_matched","hydrolases_replaced","v_atpase_replaced")
    labels = ["Replace both panels", "Keep pump subunits", "Keep CTSD and TPP1"]
    action = ["Replace targets and pump", "Replace two target proteins", "Replace eight pump subunits"]
    values = {key: [] for key in names}
    path = "Data_S2/matched_draws.csv"
    for r in csv.DictReader(io.StringIO(archive.read(path).decode("utf-8"))):
        if (r["dataset"], r["group"], float(r["caliper_log2"])) == ("Leitner","AD",.5):
            values[r["design"]].append(float(r["contrast_log2"]))
    benchmark = {r["design"]: r for r in tab(archive,30) if (r["dataset"],r["group"],float(r["caliper_log2"])) ==
                 ("Leitner","AD",.5)}
    threshold = float(benchmark[names[0]]["observed"])
    minimum = min(min(v) for v in values.values())
    maximum = max(max(v) for v in values.values())
    edges = np.linspace(np.floor(minimum*4)/4, np.ceil(maximum*4)/4, 51)
    y_max = 1.1*max(max(np.histogram(v,edges)[0]) for v in values.values())/100
    fig = plt.figure(figsize=(7, 3.38))
    for i, (key, title, second) in enumerate(zip(names,labels,action)):
        left = .104+i*.298
        panel(fig,left-.054,.944,"ABC"[i],title)
        if i>0:fig.text(left-.018,.879,second,fontsize=7.1)
        else:fig.text(left-.018,.879,"Same abundance-matching rule",fontsize=7.1)
        ax = fig.add_axes([left,.245,.247,.456])
        x = np.array(values[key])
        assert len(x)==10000
        reaches = x>=threshold-1e-12
        hits=int(reaches.sum())
        assert hits==int(benchmark[key]["hits"])
        below=np.histogram(x[~reaches],edges)[0]/100
        above=np.histogram(x[reaches],edges)[0]/100
        ax.bar(edges[:-1],below,width=np.diff(edges),align="edge",color="#ABCCCA",linewidth=0)
        ax.bar(edges[:-1],above,width=np.diff(edges),align="edge",bottom=below,color=GOLD,linewidth=0)
        ax.axvline(threshold,ls=(0,(3,2)),color="#232829",lw=.85)
        ax.set_xlim(edges[0],edges[-1]);ax.set_ylim(0,y_max)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="x", pad=2)
        ax.grid(axis="y",color="#E8EDED",lw=.55,zorder=0);ax.set_axisbelow(True)
        if i==0:ax.set_ylabel("Draws per bin (%)",labelpad=4)
        else:ax.tick_params(labelleft=False)
        fig.text(left+.247,.777,f"{hits:,} / 10,000 ({hits/100:.2f}%)",ha="right",fontsize=8.2,fontweight="bold")
    fig.text(.52,.137,"Drawn target-minus-reference score (log₂)",ha="center",fontsize=8)
    handles=[Line2D([],[],color="#232829",lw=.85,ls="--",label="Observed "+signed(threshold)),
             Line2D([],[],lw=0,marker="s",mfc=GOLD,mec=GOLD,ms=5,
                    label="Empirical upper-tail frequency" if explicit_tail else "Scores reaching the observed value")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),fontsize=7.1,ncol=2,columnspacing=1.8)
    return fig


def coverage_figure(archive):
    fig=plt.figure(figsize=(7,3.80))
    panel(fig,.042,.947,"A","Estimates with each group’s measured members")
    panel(fig,.556,.947,"B","From annotation to complete pairs")
    ax=fig.add_axes([.209,.243,.290,.61])
    y = np.arange(len(REFERENCES))[::-1]
    t={ (r["dataset"],r["group"],r["reference"]): r for r in tab(archive,29)}
    for j,(dataset,group,label) in enumerate(GROUPS):
        for yy,(key,_) in zip(y,REFERENCES):
            r=t[dataset,group,key]
            m,lo,hi=[float(r[k]) for k in ("estimate","low","high")]
            assert lo<=m<=hi
            ax.errorbar(m,yy+(.255-.17*j),xerr=[[m-lo],[hi-m]],fmt=MARKS[j],ms=3.8,c=COLORS[j],
                        mfc=COLORS[j] if j<3 else "white",elinewidth=.85,capsize=1.4)
    ax.axvline(0,c="#ADBABA",lw=.66,zorder=0);ax.set_ylim(-.49,4.49)
    ax.set_yticks(y,[label for _,label in REFERENCES]);ax.tick_params(axis="y",length=0)
    ax.spines["left"].set_visible(False);ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("CTSD/TPP1 relative score (log₂)",labelpad=3)
    handles=[Line2D([],[],ls="",marker=m,color=c,ms=4.5,label=g[2]) for m,c,g in zip(MARKS,COLORS,GROUPS)]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.052,.025),ncol=2,fontsize=6.5,
               labelspacing=.7,columnspacing=.87,handletextpad=.25)
    bx=fig.add_axes([.724,.233,.228,.62])
    r=records(archive,"Data_S2/coverage_steps.csv")
    vals=[int(x["n_proteins"]) for x in r]
    c=["#BDCBCD","#9FBEBD","#72A8A3","#53918B","#347E78","#1E6665"]
    yy=np.arange(len(r))[::-1]
    bx.barh(yy,vals,color=c,height=.61)
    for y0,(v,row) in zip(yy,zip(vals,r)):
        bx.text(v+1.4,y0,str(v),fontsize=7.2,fontweight="bold",va="center")
        bx.text(-1.3,y0,row["step"],fontsize=7.15,ha="right",va="center",clip_on=False)
    bx.set_xlim(0,47);bx.set_ylim(-.6,len(r)-.37);bx.set_yticks([]);bx.set_xticks([])
    for spine in bx.spines.values():spine.set_visible(False)
    total, any_pair, complete=vals[0],vals[2],vals[-1]
    fig.text(.556,.135,f"{complete} complete   ·   {any_pair-complete} partial   ·   {total-any_pair} no pairs",
             fontsize=7.45,fontweight="bold")
    fig.text(.556,.071,"Leitner AD · 53 paired donors",fontsize=7.1)
    return fig


def family_span(archive):
    observed=records(archive,"Data_S1/family_summary.csv")
    chosen=[next(r for r in observed if r["dataset"]=="Leitner" and r["application"]==key)
            for key in ("lysosomal_anchor","apoe_lipid")]
    fig=plt.figure(figsize=(7,2.72))
    panel(fig,.042,.935,"A","Extremes across named reference estimates")
    panel(fig,.692,.935,"B","Minimum divided by span")
    ax=fig.add_axes([.195,.335,.433,.444])
    bx=fig.add_axes([.697,.335,.254,.444]);bx.set_axis_off()
    for y, row, name, color in zip([1,0],chosen,["CTSD/TPP1", "APOE"],[TEAL,BLUE]):
        low,high,bound,num=[float(row[k]) for k in ("minimum_mean","maximum_mean","smallest_simultaneous_lower","minimum_mean_over_span")]
        disp_ratio=f"{num:.3f}" if num<1 else f"{num:.2f}"
        assert abs((high-low)-float(row["span"]))<1e-10
        ax.plot([low,high],[y+.07,y+.07],lw=3.0,c=color,solid_capstyle="round")
        ax.scatter([low,high],[y+.07]*2,s=18,c=color,zorder=3)
        ax.scatter([bound],[y-.135],s=24,marker="D",facecolor="white",edgecolor="#253F46",zorder=3)
        ax.text(-.24,y,name,ha="right",va="center",fontsize=8.0,fontweight="bold",clip_on=False)
        ax.text((low+high)/2,y+.25,f"{high-low:.3f}",ha="center",fontsize=7.15)
        bx.add_patch(FancyBboxPatch((.03,.585 if y else .06),.94,.34,boxstyle="round,pad=.021,rounding_size=.036",
                                    ec="#D9E5E5",fc="#F1F6F6",lw=.6,transform=bx.transAxes))
        bx.text(.13,.753 if y else .227,disp_ratio,va="center",fontweight="bold",fontsize=12,transform=bx.transAxes)
        bx.text(.93,.753 if y else .227,f"{low:.4f} / {high-low:.4f}",ha="right",va="center",fontsize=6.8,transform=bx.transAxes)
    ax.axvline(0,color="#B6C4C5",lw=.7,ls="--");ax.set_xlim(-.21,2.78);ax.set_ylim(-.36,1.47)
    ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xticks([0,.5,1,1.5,2,2.5])
    ax.set_xlabel("Relative estimate or lower bound (log₂)")
    handles=[Line2D([],[],c=TEAL,lw=2.8,marker="o",ms=3.4,label="Span of point estimates"),
             Line2D([],[],c="#253F46",ls="",marker="D",mfc="white",ms=4.1,label="Smallest simultaneous lower bound")]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.041,.036),fontsize=7.05,ncol=2)
    return fig


def caa_pairs(archive):
    all_rows=records(archive,"Data_S2/caa_donors.csv")
    p=[r for r in tab(archive,8) if r["endpoint"]=="plaque minus CAA"][0]
    h=[r for r in tab(archive,9) if r["reference_class"]=="ec3_lysosomal_lumen" and r["endpoint"]=="plaque minus CAA"][0]
    fig=plt.figure(figsize=(7,3.36))
    overall=[float(r[k]) for r in all_rows for k in ("plaque_minus_adjacent_log2","caapos_minus_caaneg_log2")]
    lim=(min(overall)-.13,max(overall)+.18)
    for i,(ref,lab,stats) in enumerate([("v_atpase","Eight V-ATPase subunits",p),("ec3_lysosomal_lumen","Common EC 3 luminal enzymes",h)]):
        left=.111+.46*i
        panel(fig,left-.057,.942,"AB"[i],lab)
        ax=fig.add_axes([left,.294,.365,.525])
        r=[r for r in all_rows if r["reference"]==ref];assert len(r)==6
        pl=np.array([float(x["plaque_minus_adjacent_log2"]) for x in r])
        v=np.array([float(x["caapos_minus_caaneg_log2"]) for x in r])
        y=np.arange(6)[::-1]
        ax.hlines(y,np.minimum(pl,v),np.maximum(pl,v),linewidth=1.15,colors="#A3B4B8",zorder=1)
        ax.scatter(pl,y,c=TEAL,s=23,zorder=2)
        ax.scatter(v,y,marker="D",s=22,facecolors="white",edgecolor=VIOLET,lw=1.0,zorder=2)
        ax.set_yticks(y,[x["donor"] for x in r]);ax.tick_params(axis="y",length=0)
        ax.spines["left"].set_visible(False);ax.axvline(0,c="#B9C7C8",lw=.7,ls="--",zorder=0)
        ax.set_xlim(lim);ax.set_ylim(-.48,5.48);ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_xlabel("Target-minus-reference score (log₂)",labelpad=2)
        mean,low,high=[signed(stats[k]) for k in ("estimate_log2","ci95_low","ci95_high")]
        fig.text(left-.02,.176,f"Plaque minus CAA  {mean} [{low}, {high}]",fontsize=7.25,fontweight="bold")
    handles=[
        Line2D([],[],ls="",marker="o",c=TEAL,ms=5,label="Plaque versus adjacent tissue"),
        Line2D([],[],ls="",marker="D",c=VIOLET,mfc="white",ms=4.7,label="CAA-positive versus CAA-negative vessels"),
    ]
    fig.legend(handles=handles,fontsize=7.0,ncol=2,loc="lower center",bbox_to_anchor=(.5,.025),columnspacing=1.5)
    return fig


def donor_distributions(archive):
    r=records(archive,"Data_S2/regional_enzyme_donors.csv")
    values=[np.array([float(e[k]) for e in r]) for k in ("CTSD_log2","TPP1_log2","CTSD_minus_TPP1_log2")]
    assert len(r)==53 and (values[-1]>0).sum()==43
    rng=np.random.default_rng(20261120)
    fig=plt.figure(figsize=(7,3.10))
    panel(fig,.044,.941,"A","Regional changes by enzyme")
    panel(fig,.548,.941,"B","Within-donor CTSD minus TPP1")
    a=fig.add_axes([.135,.225,.349,.573])
    b=fig.add_axes([.590,.225,.357,.573])
    def half(ax,x,y,color,scale):
        axis=np.linspace(x.min(),x.max(),180)
        density=gaussian_kde(x)(axis)
        h=density/density.max()*scale
        ax.fill_between(axis,y,y+h,fc=color,alpha=.23,lw=0)
        ax.plot(axis,y+h,c=color,lw=1.25)
        bands=rng.uniform(-.22,-.064,len(x))
        ax.scatter(x,y+bands,c=color,s=9.3,alpha=.78,lw=0,zorder=2)
        q1,median,q3=np.quantile(x,[.25,.5,.75])
        ax.plot([q1,q3],[y-.285]*2,c="#252A2C",lw=1.5,solid_capstyle="round")
        ax.scatter([median],[y-.285],s=18,marker="|",c="black")
    half(a,values[0],1.02,TEAL,.34)
    half(a,values[1],0,BLUE,.34)
    half(b,values[2],.25,VIOLET,.68)
    for ax in (a,b):
        ax.set_ylim(-.39,1.45);ax.axvline(0,c="#B7C1C2",lw=.65,ls="--",zorder=0)
        ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
    a.set_yticks([.91,-.11],["CTSD","TPP1"])
    b.set_yticks([])
    a.set_xlabel("Plaque minus adjacent tissue (log₂)")
    b.set_xlabel("Difference between enzyme responses (log₂)")
    fig.text(.590,.756,"43 of 53 differences above zero",fontsize=7.15,fontweight="bold")
    fig.text(.044,.047,"Points show all donors. Lines under the points mark medians and interquartile ranges.",
             fontsize=6.85)
    return fig


def toc_graphic():
    fig=plt.figure(figsize=(3.25,1.72))
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,3.25);ax.set_ylim(0,1.72);ax.set_axis_off()
    rng=np.random.default_rng(718)
    for x, title, ec, bg in [(.39,"Plaque","#AF711F","#F5E2BD"),(.97,"Adjacent","#4C7B7C","#DFEEEB")]:
        cy=.99
        ax.add_patch(Circle((x,cy),.232,fc=bg,ec=ec,lw=.78))
        for _ in range(16):
            theta=rng.uniform(0,np.pi*2);radius=rng.uniform(.035,.193)
            xx=x+np.cos(theta)*radius;yy=cy+np.sin(theta)*radius
            phi=rng.uniform(0,np.pi*2)
            length=.027 if title=="Adjacent" else .04
            ax.plot([xx-length*np.cos(phi),xx+length*np.cos(phi)],[yy-length*np.sin(phi),yy+length*np.sin(phi)],
                    color=ec,lw=.57,alpha=.72,solid_capstyle="round")
        ax.text(x,.656,title,ha="center",fontsize=6.7,va="center")
    ax.plot([.29,.29,1.08,1.08],[.54,.49,.49,.54],c="#36565B",lw=.75)
    ax.text(.683,.367,"Paired tissue",ha="center",va="center",fontsize=6.85)
    ax.add_patch(FancyArrowPatch((1.26,.89),(1.50,.89),arrowstyle="-|>",mutation_scale=7.5,lw=.82,color="#36565B"))
    for y,title,lines,c in [
        (.910,"CTSD / TPP1","Above pump subunits\nNear fellow luminal enzymes",GOLD),
        (.225,"APOE","Above apolipoprotein\nand receptor panels",TEAL)
    ]:
        ax.add_patch(FancyBboxPatch((1.59,y),1.56,.565,boxstyle="round,pad=.025,rounding_size=.07",
                                    fc="#F4F7F7",ec="#D1DFDF",lw=.6))
        ax.add_patch(Rectangle((1.629,y+.085),.025,.392,fc=c,ec="none"))
        ax.text(1.72,y+.408,title,fontsize=7.65,fontweight="bold",va="center")
        ax.text(1.72,y+.194,lines,fontsize=6.55,va="center",linespacing=1.13)
    return fig

