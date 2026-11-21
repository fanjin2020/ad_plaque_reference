"""Render publication figures directly from the reviewed statistical tables."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

COLORS=["#0072B2","#D55E00","#009E73","#CC79A7"]
COHORT_COLORS=["#226C93","#11786D","#BD6428","#805789"]
GROUPS=[("Leitner","AD"),("Marta-Ariza","EOAD"),("Marta-Ariza","LOAD"),("Drummond 2022","eoAD")]


def read(tables, name):
    with (tables/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def choose(rows,**query):
    rows=[r for r in rows if all(r[k]==str(v) for k,v in query.items())]
    assert len(rows)==1,query
    return rows[0]


def save(fig, output, name):
    for text in fig.findobj(matplotlib.text.Text):
        text.set_color("black")
    fig.savefig(output/(name+".pdf"))
    fig.savefig(output/(name+".png"),dpi=300)
    fig.savefig(output/(name+".tiff"),dpi=300,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)


def magnitude(tables, output):
    rows=read(tables, "01_margin_curves.csv")
    fig,axes=plt.subplots(2,2,figsize=(7,4.65),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.15,top=.91,hspace=.40,wspace=.22)
    for j,(key,ax) in enumerate(zip(GROUPS,axes.flat)):
        data=sorted([r for r in rows if (r["dataset"],r["group"])==key],key=lambda r:float(r["margin_log2"]))
        x=np.array([float(r["margin_log2"]) for r in data])
        ax.fill_between(x,[100*float(r["bootstrap_low"]) for r in data],[100*float(r["bootstrap_high"]) for r in data],color=COHORT_COLORS[j],alpha=.18)
        ax.plot(x,[100*float(r["fraction"]) for r in data],color=COHORT_COLORS[j],marker=["o","s","^","D"][j],markersize=3.2,lw=1.2)
        title=key[0].replace("Marta-Ariza","Martá-Ariza").replace(" 2022","")+" "+key[1].upper()
        ax.set_title(title+f"  (n = {data[0]['n_donors']})",fontsize=8.5)
        ax.text(0,1.06,"ABCD"[j],transform=ax.transAxes,fontweight="bold",fontsize=10)
        ax.set_ylim(0, 70)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_xlim(0, .4)
        ax.set_xticks([0, .1, .2, .3, .4])
        if j%2==0:ax.set_ylabel("Reference-dependent direction (%)",fontsize=7.5)
        if j>=2:ax.set_xlabel("Magnitude required on each side (log$_2$)")
    fig.text(.5,.035,"Shading: pointwise 95% donor-bootstrap intervals. Background membership is fixed.",ha="center",fontsize=7)
    save(fig, output, "Fig4_background")


def harmonized(tables, output):
    rows=[r for r in read(tables, "02_harmonized_curves.csv") if r["scope"]=="three_larger_groups" and r["background"]=="common_intersection"]
    names=["observed","harmonized_classes","harmonized_with_singletons"]
    labels=["Study-specific members","Shared multi-protein panels","Shared panels plus USP5"]
    fig,axes=plt.subplots(1,3,figsize=(7,3.45),sharey=True)
    fig.subplots_adjust(left=.17,right=.985,bottom=.31,top=.85,wspace=.20)
    for p,(ax,margin) in enumerate(zip(axes,["0","0.1","0.2"])):
        for j,definition in enumerate(names):
            for i,key in enumerate(GROUPS[:3]):
                r=choose(rows,dataset=key[0],group=key[1],definition=definition,margin_log2=margin)
                y=i+(j-1)*.20
                ax.plot([100*float(r["bootstrap_low"]),100*float(r["bootstrap_high"])],[y,y],color=COLORS[j],linewidth=1.1)
                ax.plot(100*float(r["fraction"]),y,marker=["o","s","^"][j],color=COLORS[j],markersize=4)
        ax.set_title(f"Margin = {margin} log$_2$",fontsize=8.5)
        ax.text(-.03,1.18,"ABC"[p],transform=ax.transAxes,fontweight="bold",fontsize=10)
        ax.set_xlim(0,70);ax.set_xticks([0,20,40,60]);ax.set_xlabel("Proteins (%)");ax.set_ylim(2.5,-.5)
    axes[0].set_yticks(range(3),[a.replace("Marta-Ariza","Martá-Ariza")+"\n"+b for a,b in GROUPS[:3]])
    handles=[Line2D([0],[0],color=COLORS[i],marker=["o","s","^"][i],label=label,markersize=4) for i,label in enumerate(labels)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.065),frameon=False,ncol=1,fontsize=7)
    fig.text(.5,.025,"Common background: 1,344 proteins. Bars: pointwise 95% donor-bootstrap intervals.",ha="center",fontsize=7)
    save(fig, output, "FigS10_harmonization")


def simulation(tables, output):
    performance=read(tables, "03_simulation_performance.csv");availability=read(tables, "03_simulation_availability.csv")
    settings=[("complete","0"),("MCAR","0.2"),("MNAR","0.2"),("MCAR","0.4"),("MNAR","0.4")]
    methods=["fixed_reference_nominal","selected_reference_nominal","all_reference_nominal","all_reference_bonferroni"]
    labels=["Fixed reference","Selected smallest p","All references, nominal","All references, Bonferroni"]
    fig,axes=plt.subplots(1,2,figsize=(7,3.75),gridspec_kw={"width_ratios":[1.25,1]})
    fig.subplots_adjust(left=.09,right=.99,bottom=.33,top=.82,wspace=.29)
    for j,method in enumerate(methods):
        for i,(mechanism,rate) in enumerate(settings):
            r=choose(performance,truth="all_null",n=53,rho="0.2",missingness=mechanism,requested_missing_fraction=rate,method=method)
            x=i+(j-1.5)*.15
            axes[0].plot([x,x],[100*float(r["any_rejection_mc_low"]),100*float(r["any_rejection_mc_high"])],color=COLORS[j],linewidth=.9)
            axes[0].plot(x,100*float(r["any_rejection_rate"]),marker=["o","s","^","D"][j],color=COLORS[j],markersize=3.5)
    axes[0].axhline(5,color="black",ls="--",lw=.7)
    axes[0].set_title("A  False rejection among estimable repeats\nAll-null truth; n = 53; correlation = 0.2",fontsize=7.5)
    xticks=["None","MCAR\n20%","MNAR\n20%","MCAR\n40%","MNAR\n40%"]
    axes[0].set_xticks(range(5),xticks,fontsize=7)
    axes[0].set_ylabel("Repeats with any false rejection (%)",fontsize=7.5)
    axes[0].set_ylim(bottom=0)
    matrix=np.zeros((3,5))
    for j,n in enumerate([5,20,53]):
        for i,(mechanism,rate) in enumerate(settings):
            r=choose(availability,truth="all_null",n=n,rho="0.2",missingness=mechanism,requested_missing_fraction=rate)
            matrix[j,i]=100*float(r["estimable_fraction"])
    axes[1].imshow(matrix,cmap=ListedColormap(plt.get_cmap("Blues")(np.linspace(.025,.56,256))),vmin=0,vmax=100,aspect="auto")
    for j in range(3):
        for i in range(5):axes[1].text(i,j,f"{matrix[j,i]:.1f}",ha="center",va="center",fontsize=7.5)
    axes[1].set_yticks(range(3),["n = 5","n = 20","n = 53"])
    axes[1].set_xticks(range(5),xticks,fontsize=7)
    axes[1].set_title("B  Estimable repeats (%)\nAt least 3 common complete donors",fontsize=7.5)
    handles=[Line2D([0],[0],color=COLORS[j],marker=["o","s","^","D"][j],label=labels[j],markersize=3.5) for j in range(4)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.11),frameon=False,ncol=2,fontsize=7)
    fig.text(.5,.045,"Known-truth simulations, 2,000 repeats per scenario. Bars: Wilson 95% Monte Carlo intervals.",ha="center",fontsize=7)
    save(fig, output, "FigS6_equicorrelation")


