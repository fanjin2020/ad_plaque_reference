"""Reference-plane visualization on the same 53 human donors."""
import numpy as np
from matplotlib.lines import Line2D
from . import common
from .common import page, panel, label, footer, read, BLUE, GOLD

def fmt(x):
    return f"{float(x):+.3f}".replace("-","−")

def run(highlight=False):
    donor=read("paired_geometry.csv")
    summaries=read("paired_summary.csv")
    lookup={(r["family"],r["reference"]):r for r in summaries}
    cases=[
        ("lysosomal_anchor","CTSD and TPP1",[("v_atpase","V-ATPase"),("ec3_lysosomal_lumen","Five luminal enzymes")],(-.52,1.86),[-.5,0,.5,1,1.5]),
        ("apoe_lipid","APOE",[("apod_clu","APOD and CLU"),("lrp1_sort1","LRP1 and SORT1")],(-.55,4.82),[0,1,2,3,4]),
    ]
    fig=page("PAIRED HUMAN TISSUE","Changing the reference shifts the contrast","Leitner AD   /   53 plaque–adjacent pairs",5.30)
    for idx,(family,name,refs,lim,ticks) in enumerate(cases):
        x0=.054+.478*idx
        panel(fig,x0,.776,"AB"[idx],name)
        ax=fig.add_axes([x0+.023,.344,.365,.388])
        ax.set_aspect("equal",adjustable="box")
        lo,hi=lim
        ax.fill_between([lo,hi],[lo,hi],hi,color="#E9F3F1",zorder=0)
        ax.fill_between([lo,hi],lo,[lo,hi],color="#F7F2ED",zorder=0)
        ax.plot([lo,hi],[lo,hi],color="#555D60",lw=.87,ls=(0,(4,2)),zorder=1)
        ax.axvline(0,color="#CAD1CF",lw=.55,zorder=1)
        ax.axhline(0,color="#CAD1CF",lw=.55,zorder=1)
        allvals=[]
        for ref,_ in refs:
            dr=sorted([r for r in donor if r["family"]==family and r["reference"]==ref],key=lambda r:int(r["donor"]))
            assert len(dr)==53
            ids=[r["donor"] for r in dr]
            x=np.array([float(r["ref"]) for r in dr])
            y=np.array([float(r["target"]) for r in dr])
            z=np.array([float(r["difference"]) for r in dr])
            assert np.allclose(y-x,z)
            check=lookup[family,ref]
            assert abs(z.mean()-float(check["difference"]))<1e-10 and int((z>0).sum())==int(check["positive"])
            allvals.append((ids,x,y))
        ids,x0val,y0=allvals[0]; ids2,x1,y1=allvals[1]
        assert ids==ids2 and np.allclose(y0,y1)
        if highlight:
            crosses=(y0>x0val)!=(y0>x1)
            for take,c,alpha,lw in ((~crosses,"#89999A",.24,.55),(crosses,"#3F5051",.76,.8)):
                ax.hlines(y0[take],np.minimum(x0val,x1)[take],np.maximum(x0val,x1)[take],
                          lw=lw,color=c,alpha=alpha,zorder=2)
        else:
            ax.hlines(y0,np.minimum(x0val,x1),np.maximum(x0val,x1),lw=.61,color="#89999A",alpha=.64,zorder=2)
        colors=[BLUE,GOLD];markers=["o","D"]
        for j,(_,x,y) in enumerate(allvals):
            ax.scatter(x,y,s=11 if not j else 12,marker=markers[j],facecolors=colors[j] if j==0 else "white",
                       edgecolors=colors[j],linewidths=.68,zorder=3,alpha=.89)
            ax.scatter([x.mean()],[y.mean()],s=42,marker=markers[j],facecolor=colors[j],edgecolor="black",linewidth=.95,zorder=4)
        ax.set_xlim(lim);ax.set_ylim(lim);ax.set_xticks(ticks);ax.set_yticks(ticks)
        ax.tick_params(pad=2)
        ax.set_xlabel("Reference change (log₂)",labelpad=2)
        if not idx: ax.set_ylabel("Target change (log₂)",labelpad=2)
        label(fig,x0,.222,"Mean target excess  [simultaneous 95% interval]",6.70,"bold")
        for j,(key,title) in enumerate(refs):
            r=lookup[family,key]; yk=[.169,.111][j]
            fig.add_artist(Line2D([x0+.009],[yk],transform=fig.transFigure,ls="none",marker=markers[j],
                                  mfc=colors[j] if not j else "white",mec=colors[j],mew=1,ms=4.8))
            label(fig,x0+.025,yk,title,7.0)
            label(fig,x0+.420,yk,f'{fmt(r["difference"])}  [{fmt(r["lower"])}, {fmt(r["upper"])}]',7.05,"bold",ha="right")
    footer(fig,"Dark lines cross the equality diagonal. Each line joins one donor under two references. Large symbols show means."
           if highlight else "Each line joins one donor under two references. Large symbols show means. The diagonal marks equal regional changes.")
    common.save(fig,"03_reference_plane")
