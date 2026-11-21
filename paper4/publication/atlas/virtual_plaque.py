"""Generate one virtual section and plot two views of the same synthetic cells."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from . import common
from .common import page, panel, label, footer, write, TEAL, VIOLET

SEED=20261120
WIDTH,HEIGHT=200.,136.
PX=7
CENTER=np.array([80.,70.])
COLS={"nuclei":"#927DE5","target":"#159589","plaque":"#EAA137"}
def radius(a):
    return 24*(1+.125*np.cos(3*a+.4)+.081*np.sin(5*a-1.1)+.058*np.cos(9*a+.1))

def annotate_key(fig,x,y,color,text,shape="s"):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([x],[y],marker=shape,ms=4.5,mfc=color,mec=color,ls="none",transform=fig.transFigure))
    label(fig,x+.011,y,text,6.8)

def fields_and_cells(rng):
    h,w=int(HEIGHT*PX),int(WIDTH*PX)
    yy,xx=np.mgrid[0:h,0:w].astype("float32")/PX
    dx,dy=xx-CENTER[0],(yy-CENTER[1])/.84
    ang=np.arctan2(dy,dx)
    radial=np.hypot(dx,dy)
    outline=radius(ang)
    mask=radial<=outline
    distance=distance_transform_edt(~mask,sampling=1/PX).astype("float32")
    pts=[]
    while len(pts)<310:
        q=rng.uniform((2.8,2.8),(WIDTH-2.8,HEIGHT-2.8))
        iy,ix=np.round(q[::-1]*PX).astype(int)
        if distance[iy,ix]<2.6: continue
        if pts and np.min(np.linalg.norm(np.asarray(pts)-q,axis=1))<4.7: continue
        pts.append(q)
    pts=np.asarray(pts)
    ids=np.rint(pts[:,::-1]*PX).astype(int)
    d=distance[ids[:,0],ids[:,1]].astype(float)
    enrichment=np.exp(-d/23)
    shared=rng.normal(0,.24,len(d))
    target=1.42*enrichment+shared+rng.normal(0,.15,len(d))
    flat=shared+rng.normal(0,.13,len(d))
    tracking=1.29*enrichment+shared+rng.normal(0,.13,len(d))
    return xx,yy,radial,ang,mask,distance,pts,d,target,flat,tracking

def pseudo_channels(rng,xx,yy,rad,ang,mask,pts,target):
    h,w=mask.shape
    nuclei=np.zeros((h,w),"f")
    soma=np.zeros_like(nuclei)
    processes=Image.new("L",(w,h));pen=ImageDraw.Draw(processes)
    for (x,y),level in zip(pts,target):
        ix,iy=np.rint([x*PX,y*PX]).astype(int)
        d=30
        y0,y1=max(iy-d,0),min(iy+d+1,h)
        x0,x1=max(ix-d,0),min(ix+d+1,w)
        xxq=xx[y0:y1,x0:x1]-x;yyq=yy[y0:y1,x0:x1]-y
        a=rng.uniform(0,np.pi)
        r1=rng.uniform(.8,1.36);r2=rng.uniform(.56,.95)
        rx=(xxq*np.cos(a)+yyq*np.sin(a))/r1
        ry=(-xxq*np.sin(a)+yyq*np.cos(a))/r2
        r=np.hypot(rx,ry)
        nu=np.exp(-.40*r**2)*(1+.17*np.cos(8*rx+2*ry))
        nuclei[y0:y1,x0:x1]+=rng.uniform(.55,1.2)*nu
        ring=np.exp(-.5*((r-2.25)/.54)**2)
        amp=2**level
        soma[y0:y1,x0:x1]+=.78*amp*ring*(.78+.22*np.sin(2*r+3*a+rx))
        # Procedural processes give the simulated marker a local morphology.
        for _ in range(rng.integers(2,5)):
            theta=rng.uniform(0,2*np.pi)
            line=[[x,y]]
            for k in range(rng.integers(3,7)):
                theta+=rng.normal(0,.35)
                line.append(np.asarray(line[-1])+rng.uniform(.8,1.4)*np.array([np.cos(theta),np.sin(theta)]))
            line=np.asarray(line)
            pen.line([tuple(v) for v in np.rint(line*PX).astype(int)],fill=int(np.clip(42*amp,18,210)),width=rng.integers(1,3))
            if len(line)>4:
                a2=theta+rng.uniform(.65,1.1)
                branch=np.stack((line[-3],line[-3]+rng.uniform(1.5,3.6)*np.array([np.cos(a2),np.sin(a2)])))
                pen.line([tuple(v) for v in np.rint(branch*PX).astype(int)],fill=int(np.clip(36*amp,12,160)),width=1)
    textured=rng.normal(size=(h,w)).astype("f")
    fine=gaussian_filter(textured,1.4)
    lobes=gaussian_filter(textured,9)
    warp=gaussian_filter(textured,8)*23
    ribbons=(.5+.5*np.sin(38*ang+1.2*rad+3*np.sin(9*ang+.18*rad)+warp))**4
    fibrils=(.5+.5*np.cos(54*ang-.75*rad+.9*np.sin(1.4*rad)+.6*warp))**7
    boundary=gaussian_filter(mask.astype("f"),2)
    plaque=boundary*(.22+.70*ribbons+.36*fibrils+.75*np.maximum(fine,0)+np.clip(3*lobes,-.15,.15))
    plaque+=.16*gaussian_filter(boundary,17)
    processes=gaussian_filter(np.asarray(processes,dtype="f")/90,.75)
    soma=gaussian_filter(soma,1.05)+processes
    nuclei=gaussian_filter(nuclei,.8)
    channels=[(nuclei,[.48,.36,.91],1.3),(soma,[.06,.9,.70],.84),(plaque,[1.0,.56,.10],1.40)]
    rgb=np.zeros((*mask.shape,3),"f")+np.array([.016,.026,.064],"f")
    rgb+=(np.clip(.008+.009*textured,0,.05))[...,None]
    for arr,color,gain in channels:
        signal=(1-np.exp(-gain*np.maximum(arr,0)))**.88
        rgb+=signal[...,None]*np.asarray(color,"f")
    return np.uint8(np.clip(rgb,0,1)*255)

def run(legible_scale=False):
    rng=np.random.default_rng(SEED)
    xx,yy,rad,ang,mask,dist,pts,d,target,flat,tracking=fields_and_cells(rng)
    far=(d>=75)&(d<90)
    assert far.sum()>8
    rgb=pseudo_channels(rng,xx,yy,rad,ang,mask,pts,target)
    ratios=[target-flat,target-tracking]
    centered=[v-v[far].mean() for v in ratios]
    write("virtual_cells.csv",[dict(cell=i+1,x_um=f"{xy[0]:.6f}",y_um=f"{xy[1]:.6f}",
        edge_distance_um=f"{d[i]:.6f}",T_log2=f"{target[i]:.8f}",flat_R_log2=f"{flat[i]:.8f}",
        tracking_R_log2=f"{tracking[i]:.8f}") for i,xy in enumerate(pts)])
    profile=[]
    for lo in range(0,90,15):
        use=(d>=lo)&(d<lo+15)
        assert use.sum()>4
        profile.append(dict(start_um=lo,end_um=lo+15,cells=int(use.sum()),T_vs_flat=centered[0][use].mean(),T_vs_tracking=centered[1][use].mean()))
    write("virtual_profiles.csv",profile)
    fig=page("SPATIAL SIMULATION","Two views of one virtual plaque section","One synthetic field   /   310 simulated cells",5.72)
    panel(fig,.052,.779,"A","Simulated marker channels")
    panel(fig,.528,.779,"B","Distances from the plaque edge")
    ax=fig.add_axes([.052,.407,.425,.335]);ax.set_axis_off()
    half=.5/PX
    ax.imshow(rgb,origin="lower",extent=(-half,WIDTH-half,-half,HEIGHT-half),interpolation="antialiased")
    ax.set_xlim(0,WIDTH);ax.set_ylim(0,HEIGHT)
    ax.add_patch(Rectangle((5,4.5),54 if legible_scale else 44,11.5 if legible_scale else 10.5,
                           facecolor="white",edgecolor="none",alpha=.97))
    ax.plot([8.5,28.5],[9.6,9.6],c="black",lw=1.5,solid_capstyle="butt")
    ax.text(30.2,9.7,"20 µm",ha="left",va="center",fontsize=7.2 if legible_scale else 5.1)
    # Scale background covers both the exact 20-µm segment and its annotation.
    bx=fig.add_axes([.528,.407,.425,.335]);bx.set_axis_off()
    bx.set_aspect("equal")
    levels=np.arange(0,151,15)
    cmap=LinearSegmentedColormap.from_list("distance",["#EEBD77","#F1E1B9","#C5DDD4","#7AACB0","#487E9A"])
    bx.contourf(xx[::6,::6],yy[::6,::6],dist[::6,::6],levels=levels,cmap=cmap,alpha=.71,antialiased=True)
    bx.contour(xx[::6,::6],yy[::6,::6],dist[::6,::6],levels=levels[1:-1],colors="#798C87",linewidths=.45,alpha=.75)
    a=np.linspace(-np.pi,np.pi,400);rr=radius(a)
    bx.fill(CENTER[0]+rr*np.cos(a),CENTER[1]+.84*rr*np.sin(a),fc="#DF9D4D",ec="#70502B",lw=.7)
    bx.scatter(pts[:,0],pts[:,1],s=4.1,facecolor="#405356",edgecolor="white",lw=.28)
    bx.set_xlim(0,WIDTH);bx.set_ylim(0,HEIGHT)
    annotate_key(fig,.061,.375,COLS["nuclei"],"Nuclei")
    annotate_key(fig,.164,.375,COLS["target"],"Target T")
    annotate_key(fig,.287,.375,COLS["plaque"],"Plaque channel")
    annotate_key(fig,.540,.375,"#435859","Cell centroid","o")
    label(fig,.704,.375,"Contours every 15 µm",7)
    panel(fig,.052,.311,"C","Target-to-reference contrast across distance bands")
    cx=fig.add_axes([.117,.142,.505,.120])
    mid=np.array([r["start_um"]+7.5 for r in profile])
    styles=[("T_vs_flat",VIOLET,"o","-"),("T_vs_tracking",TEAL,"s","--")]
    for key,color,marker,ls in styles:
        cx.plot(mid,[r[key] for r in profile],color=color,lw=1.6,marker=marker,ms=4,
                mfc=color if marker=="o" else "white",mec=color,ls=ls)
    cx.axhline(0,color="#82898B",lw=.55,zorder=0)
    cx.set_xlim(0,90);cx.set_xticks([0,15,30,45,60,75,90])
    cx.set_ylim(-.20,1.28);cx.set_yticks([0,.5,1])
    cx.set_ylabel("Mean log₂ ratio",labelpad=2);cx.set_xlabel("Distance from edge (µm)",labelpad=2)
    for y,(key,color,marker,ls),title in zip([.231,.182],styles,["Flat reference","Co-enriched reference"]):
        from matplotlib.lines import Line2D
        fig.add_artist(Line2D([.667,.708],[y,y],transform=fig.transFigure,c=color,lw=1.45,ls=ls))
        fig.add_artist(Line2D([.687],[y],transform=fig.transFigure,ls="none",marker=marker,ms=4,
                              mfc=color if marker=="o" else "white",mec=color))
        label(fig,.720,y,title,7.5,"bold")
    label(fig,.667,.126,"Centered on cells 75–90 µm\nfrom the plaque edge.",7.0,linespacing=1.35)
    footer(fig,"All coordinates and channel intensities are simulated. Each point in C is a mean across cells in one virtual field.")
    common.save(fig,"01_virtual_plaque")
