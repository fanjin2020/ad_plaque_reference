"""Compose the sampling and estimability figures from the reviewed tables and artwork generators."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.stats import binom


def plot_tissue(archive, folder, emphasize_data=False):
    from .atlas.common import label, rule
    from .evidence_figures import appearance, export, records, tab, TEAL, BLUE, GOLD, VIOLET
    from .atlas.virtual_plaque import SEED, PX, WIDTH, HEIGHT, fields_and_cells, pseudo_channels
    from .atlas.pump_structure import render_views, SECTOR_ONE, SECTOR_ZERO, ISOFORM, NEUTRAL

    appearance()
    side, top, atoms, entities, matches = render_views()
    assert (atoms, entities, matches) == (8380, 16, 7)
    rng = np.random.default_rng(SEED)
    xx, yy, rad, ang, mask, _, points, _, signal, _, _ = fields_and_cells(rng)
    assert len(points) == 310
    archived_cells = records(archive, "Data_S2/virtual_cells.csv")
    assert np.allclose(points, [[float(r["x_um"]), float(r["y_um"])] for r in archived_cells], atol=5.1e-7, rtol=0)
    image = pseudo_channels(rng, xx, yy, rad, ang, mask, points, signal)

    fig = plt.figure(figsize=(7, 5.53))
    divider = .620 if emphasize_data else .492
    scale = (.951-divider)/(.951-.492)
    y = lambda position: .951 + (position-.951)*scale
    data_top = .535 if emphasize_data else .406
    data_heading = divider-.038
    for x, letter, title in [(.052, "A", "Virtual plaque sampling"), (.528, "B", "Human V-ATPase structure"),
                             (.052, "C", "Two enzymes in the same donors"), (.528, "D", "Adjusted tissue contrasts")]:
        label(fig, x, .951 if letter in "AB" else data_heading, letter, 10.4, "bold")
        label(fig, x+.038, .951 if letter in "AB" else data_heading, title, 8.8, "bold")
    rule(fig, divider, .052, .952, color="#C4CED0", lw=.58)

    a = fig.add_axes([.055, y(.560), .421, .353*scale])
    a.set_axis_off()
    half = .5/PX
    a.imshow(image, origin="lower", extent=(-half,WIDTH-half,-half,HEIGHT-half), interpolation="antialiased")
    a.set_xlim(0,WIDTH)
    a.set_ylim(0,HEIGHT)
    a.contour(xx[::4,::4],yy[::4,::4],mask[::4,::4],levels=[.5],colors=["#FFFAF1"],linewidths=.82)
    a.add_patch(Circle((150,69),20,fill=False,ec="#FFFAF1",lw=.86,ls=(0,(4,2))))
    a.add_patch(Rectangle((4,118),91,13,facecolor="white",edgecolor="none",alpha=.98))
    a.text(9,124.5,"SIMULATED SECTION",fontsize=6.25,fontweight="bold",va="center")
    a.add_patch(Rectangle((5,4.5),44,10.5,facecolor="white",edgecolor="none",alpha=.97))
    a.plot([8.5,28.5],[9.6,9.6],c="black",lw=1.3,solid_capstyle="butt")
    a.text(30.0,9.7,"20 µm",fontsize=5.8,va="center")
    for x, ls, txt in [(.061, "-", "Plaque region"), (.244, "--", "Adjacent region")]:
        fig.add_artist(Line2D([x,x+.026],[y(.541)]*2,transform=fig.transFigure,color="#303939",lw=.80,ls=ls))
        label(fig,x+.036,y(.541),txt,7.2 if emphasize_data else 7.05)
    label(fig,.055,y(.511),"Synthetic field with 310 virtual cells",7.15 if emphasize_data else 6.95)

    b = fig.add_axes([.536,y(.533),.205,.380*scale])
    b.imshow(side,interpolation="antialiased")
    b.set_axis_off()
    top_axis = fig.add_axes([.763,y(.746),.171,.167*scale])
    top_axis.imshow(top,interpolation="antialiased")
    top_axis.set_axis_off()
    label(fig,.850,y(.732),"Cytosolic view",6.9,ha="center")
    for row_y, color, name in [(.687,SECTOR_ONE,"Matched V1 subunits"),(.648,SECTOR_ZERO,"Matched V0A1"),
                           (.609,ISOFORM,"G1 in deposited structure"),(.570,NEUTRAL,"Other human subunits")]:
        fig.add_artist(Line2D([.763],[y(row_y)],transform=fig.transFigure,ls="",marker="s",ms=5.5,
                              mfc=color,mec=color,mew=.4))
        label(fig,.782,y(row_y),name,6.8)
    label(fig,.540,y(.511),"Cryo-EM model 6WM2 · Seven of eight panel accessions match",6.8)

    c = fig.add_axes([.157,.137,.286,data_top-.137])
    donors = records(archive,"Data_S2/regional_enzyme_donors.csv")
    ctsd = np.array([float(r["CTSD_log2"]) for r in donors])
    tpp1 = np.array([float(r["TPP1_log2"]) for r in donors])
    assert len(donors)==53 and np.isclose((ctsd-tpp1).mean(),.19334408022323238)
    c.plot([0,1],np.stack((ctsd,tpp1)),c="#8BA2A3",lw=.45,alpha=.47,zorder=1)
    for x, values, color in [(0,ctsd,TEAL),(1,tpp1,BLUE)]:
        c.scatter(np.full(len(values),x),values,s=8,fc=color,ec="none",alpha=.76,zorder=2)
        c.scatter([x],[values.mean()],s=38,marker="D",fc="white",ec="#173C42",lw=1.0,zorder=3)
    c.set_xlim(-.27,1.27)
    c.set_ylim(0,1.88)
    c.set_yticks([0,.5,1,1.5])
    c.set_xticks([0,1],["CTSD","TPP1"])
    c.tick_params(axis="x",length=0)
    c.set_ylabel("Plaque minus adjacent (log₂)",labelpad=3)
    c.grid(axis="y",color="#E0E9E9",lw=.5,zorder=0)
    label(fig,.055,.055,"Leitner AD · 53 donors · Diamonds mark means",6.85)

    d = fig.add_axes([.726,.137,.228,data_top-.137])
    models = {r["contrast"]: r for r in tab(archive,25)
              if r["scope"]=="all" and r["model"]=="adjusted" and r["outcome"]=="CTSD_minus_TPP1"}
    contrasts = [("AD_plaque_minus_AD_adjacent", "Plaque − adjacent"),
                 ("AD_adjacent_minus_control", "Adjacent − control"),
                 ("AD_plaque_minus_control", "Plaque − control")]
    for i, ((key,_), color, marker) in enumerate(zip(contrasts,(TEAL,GOLD,VIOLET),("o","s","D"))):
        row = models[key]
        value,low,high = [float(row[k]) for k in ("estimate","low","high")]
        d.errorbar(value,2-i,xerr=[[value-low],[high-value]],fmt=marker,c=color,ms=4.6,lw=1.03,capsize=2.2)
    d.axvline(0,c="#688082",ls=(0,(3,2)),lw=.72,zorder=0)
    d.set_xlim(-.485,.295)
    d.set_ylim(-.49,2.49)
    d.set_yticks([2,1,0],[names for _,names in contrasts])
    d.tick_params(axis="y",length=0,pad=4.5)
    d.spines["left"].set_visible(False)
    d.set_xticks([-.4,-.2,0,.2])
    d.set_xlabel("CTSD minus TPP1 contrast (log₂)",labelpad=2)
    label(fig,.530,.055,"Nominal 95% CIs · 62 separate controls",6.85)
    export(fig,folder,"Fig1_tissue")


def plot_simulations(archive, folder, separate_missingness=False):
    from .atlas.common import label, rule
    from .evidence_figures import appearance, export, tab, TEAL, VIOLET

    appearance()
    fig = plt.figure(figsize=(7,5.22))
    label(fig,.052,.950,"A",10.4,"bold")
    label(fig,.089,.950,"False rejection under the global null",8.85,"bold")
    label(fig,.951,.950,"Complete data · 53 donors · 2,000 repeats",7.05,ha="right")
    rate = fig.add_axes([.277,.652,.669,.217])
    methods=[("fixed_primary_nominal","Fixed primary test"),("selected_min_p_nominal","Smallest nominal p value"),
             ("all_holm","Holm adjustment")]
    families = [("lysosomal_anchor","CTSD and TPP1",TEAL,"o",22),("apoe_lipid","APOE",VIOLET,"D",6)]
    observed = {(r["application"],r["method"]):r for r in tab(archive,81)
                if (r["generator"],r["truth"],r["n"],r["missingness"])==("gaussian_empirical","all_null","53","complete")}
    for j,(key,name,c,m,k) in enumerate(families):
        for i,(method,_) in enumerate(methods):
            row=observed[key,method]
            total=int(row["false_family_denominator"])
            count=int(row["false_family_count"])
            value,low,high=[100*float(row[x]) for x in ("false_family_rate","false_family_low","false_family_high")]
            assert total==2000 and np.isclose(value,100*count/total)
            rate.errorbar(value,2-i+(.14 if j==0 else -.14),xerr=[[value-low],[high-value]],
                          fmt=m,color=c,ms=4.5,lw=.91,capsize=1.5,zorder=2)
    rate.axvline(5,c="#647378",ls=(0,(3,2)),lw=.77,zorder=1)
    rate.set_ylim(-.4,2.43)
    rate.set_xlim(0,16)
    rate.set_yticks([2,1,0],[name for _,name in methods])
    rate.tick_params(axis="y",length=0)
    rate.spines["left"].set_visible(False)
    rate.set_xticks([0,5,10,15])
    rate.set_xlabel("Repeats with ≥1 false rejection (%)",labelpad=2)
    markers=[Line2D([],[],ls="",marker=m,c=c,label=n,ms=4.3) for _,n,c,m,_ in families]
    fig.legend(handles=markers,loc="center right",bbox_to_anchor=(.952,.899),ncol=2,fontsize=7,
               columnspacing=1.2,handletextpad=.34,borderaxespad=0)
    rule(fig,.581,.052,.952,color="#C4CED0",lw=.58)
    text = "Independent missingness · Chance of ≥3 complete donors" if separate_missingness else "Chance of retaining ≥3 complete donors"
    label(fig,.053,.545,text,7.5,"bold")
    ramp=colors.LinearSegmentedColormap.from_list("design",["#FAF6F0","#F2CCAC","#CBDCCE","#7ABAAE","#327F82","#295575"])
    colorbar=fig.add_axes([.609,.537,.337,.018])
    colorbar.imshow(np.linspace(0,1,256)[None,:],cmap=ramp,aspect="auto",extent=[0,100,0,1],origin="lower")
    colorbar.set_yticks([])
    colorbar.set_xticks([0,50,100],["0%","50%","100%"])
    colorbar.tick_params(length=0,pad=1,labelsize=6.7)
    for sp in colorbar.spines.values():sp.set_visible(False)
    n=np.arange(3,81)
    p=np.linspace(0,.30,301)
    xx,yy=np.meshgrid(100*p,n)
    source={(r["application"],r["missingness"]):r for r in tab(archive,80)
            if (r["generator"],r["truth"],r["n"],float(r["requested_missing_fraction"]))==
               ("gaussian_empirical","all_null","53",.20)}
    linked_counts = []
    for i,(key,short,c,m,k) in enumerate(families):
        left=.109+.474*i
        label(fig,left-.057,.477,"BC"[i],10.4,"bold")
        name="Lysosomal family" if i==0 else "APOE family"
        label(fig,left-.020,.477,f"{name} · {k} proteins",8.5,"bold")
        ax=fig.add_axes([left,.224,.365,.215])
        surface=binom.sf(2,n[:,None],(1-p[None,:])**k)
        ax.pcolormesh(xx,yy,surface,cmap=ramp,shading="nearest",vmin=0,vmax=1,rasterized=True)
        ax.contour(xx,yy,surface,levels=[.50,.90],colors=["#35484C","#35484C"],linewidths=[.67,.83],
                   linestyles=["dashed","solid"])
        ax.scatter([10,20]*3,np.repeat([5,20,53],2),s=15,facecolors="white",edgecolors="#1C2B31",lw=.62,zorder=2)
        ax.set_ylim(3,80)
        ax.set_xlim(0,30)
        ax.set_xticks([0,10,20,30],[f"{j}%" for j in [0,10,20,30]])
        ax.set_yticks([5,20,53,80])
        ax.tick_params(pad=2)
        if i==0:ax.set_ylabel("Number of donors",labelpad=3)
        ax.set_xlabel("Missing probability per protein",labelpad=2)
        pair=[source[key,mechanism] for mechanism in ("MCAR","MNAR")]
        assert all(int(row["repetitions"])==2000 for row in pair)
        unlinked,linked=[f'{int(row["n_estimable"]):,}/2,000' for row in pair]
        linked_counts.append(linked)
        if separate_missingness:
            label(fig,left-.029,.081,f"Independent simulation {unlinked}",7.2)
        else:
            label(fig,left-.029,.074,f"Independent {unlinked}   ·   Intensity-linked {linked}",6.9)
    contour_handles=[Line2D([],[],c="#35484C",ls="--",lw=.78,label="50% contour"),
                     Line2D([],[],c="#35484C",lw=.83,label="90% contour"),
                     Line2D([],[],marker="o",ls="",mfc="white",mec="#1C2B31",ms=4,label="Simulation settings")]
    fig.legend(handles=contour_handles,ncol=3,loc="center",bbox_to_anchor=(.507,.133),
               fontsize=6.9,handletextpad=.45,columnspacing=1.35,borderaxespad=0)
    if separate_missingness:
        fig.add_artist(Rectangle((.045,.007),.907,.050,transform=fig.transFigure,fc="#F5ECDC",ec="none"))
        label(fig,.054,.032,"Separate intensity-linked simulation  ·  Lysosomal "+linked_counts[0]+
              "  ·  APOE "+linked_counts[1],7.15)
    else:
        label(fig,.051,.032,"Counts use 53 donors with 20% missingness. Map probabilities assume independent missingness.",6.7)
    export(fig,folder,"Fig7_empirical")


def plot_probability_checks(archive, folder):
    from .atlas.common import label
    from .evidence_figures import appearance, export, tab, TEAL, VIOLET

    appearance()
    cases=[(p,n) for p in (.10,.20) for n in (5,20,53)]
    rows={(r["application"],float(r["requested_missing_fraction"]),int(r["n"])):r for r in tab(archive,80)
          if (r["generator"],r["truth"],r["missingness"])==("gaussian_empirical","all_null","MCAR")}
    fig=plt.figure(figsize=(7,3.37))
    for i,(key,k,element,c) in enumerate([("lysosomal_anchor",22,"Lysosomal family",TEAL),
                                         ("apoe_lipid",6,"APOE family",VIOLET)]):
        left=.201+.423*i
        label(fig,.053+.484*i,.939,"AB"[i],10.5,"bold")
        label(fig,.091+.484*i,.939,f"{element} · {k} proteins",8.85,"bold")
        ax=fig.add_axes([left,.229,.335,.598])
        y=np.arange(len(cases))[::-1]
        for level,(p,n) in zip(y,cases):
            row=rows[key,p,n]
            model=100*binom.sf(2,n,(1-p)**k)
            assert abs(model-100*float(row["analytic_count_eligibility_fraction"]))<1e-10
            value,low,high=[100*float(row[q]) for q in ("estimability_rate","estimability_low","estimability_high")]
            ax.errorbar(value,level-.13,xerr=[[max(0.,value-low)],[max(0.,high-value)]],fmt="o",ms=3.9,
                        c=c,lw=.94,capsize=1.6)
            ax.scatter([model],[level+.13],s=15,marker="D",facecolors="white",edgecolor="#303C42",lw=.84)
        ax.axhline(2.5,ls=":",c="#95A7AB",lw=.76)
        ax.set_yticks(y,[f"{p:.0%} · {n} donors" if i==0 else "" for p,n in cases])
        ax.tick_params(axis="y",length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-.49,5.49)
        ax.set_xlim(-4,104)
        ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel("Estimable repeats or probability (%)",labelpad=3)
        ax.grid(axis="x",color="#E8EEEE",lw=.55,zorder=0)
    handles=[
        Line2D([],[],marker="D",ls="",mfc="white",mec="#303C42",ms=4.2,label="Binomial probability"),
        Line2D([],[],marker="o",ls="",color=TEAL,ms=4.4,label="Monte Carlo estimate and 95% interval"),
    ]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.50,.042),ncol=2,fontsize=7.1)
    export(fig,folder,"FigS15_estimability_map")


