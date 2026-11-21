"""Draw the decision workflow for the v9 article."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def save(fig, folder, name):
    for t in fig.findobj(matplotlib.text.Text):
        t.set_color("black")
    for ext in ("png","pdf","tiff"):
        kw={"dpi":400,"facecolor":"white"}
        if ext=="tiff":kw["pil_kwargs"]={"compression":"tiff_lzw"}
        fig.savefig(folder/(name+"."+ext),**kw)
    plt.close(fig)


def box(ax,x,y,w,h,lines,face):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.007,rounding_size=.018",
                                fc=face,ec="#485A61",lw=.75))
    ax.text(x+w/2,y+h/2,lines,va="center",ha="center",fontsize=8.0,linespacing=1.23)


def arr(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color="#454D50",
                                 shrinkA=0,shrinkB=0))


def workflow(folder, shared_wording=False):
    fig=plt.figure(figsize=(6.55,4.05));ax=fig.add_axes([.02,.02,.96,.96]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,.044,.824,.91,.125,"Define tissue pair, target, primary reference, other panels and margin δ","#E4EEF1")
    text = ("The same ≥5 donors have paired data\nfor all targets and reference members?" if shared_wording else
            "At least five donors complete for every\ntarget and reference member?")
    box(ax,.045,.606,.478,.113,text,"#ECF3F3")
    box(ax,.65,.606,.303,.113,"Report shared coverage and\npanel-specific estimates","#F7F0E9")
    box(ax,.045,.397,.478,.113,"Regional simultaneous lower bound\nabove zero?","#ECF3F3")
    box(ax,.65,.397,.303,.113,"Report the regional estimate\nand its interval","#F7F0E9")
    box(ax,.045,.188,.478,.113,"Every relative simultaneous lower bound\nabove the chosen margin?","#ECF3F3")
    box(ax,.65,.188,.303,.113,"Report which named panels\nclear the margin","#F7F0E9")
    box(ax,.045,.015,.478,.079,"Enriched above every named reference","#D5E8DE")
    arr(ax,(.282,.817),(.282,.727))
    for center in (.663,.454,.245):
        arr(ax,(.533,center),(.638,center));ax.text(.582,center+.025,"No",ha="center",fontsize=7.1)
    for ya,yb in ((.598,.518),(.389,.309),(.179,.100)):
        arr(ax,(.282,ya),(.282,yb));ax.text(.305,(ya+yb)/2,"Yes",ha="left",va="center",fontsize=7.1)
    save(fig, folder, "Fig8_decision")


