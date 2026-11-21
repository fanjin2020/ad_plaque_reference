"""Common typography and export settings for the four standalone plates."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
matplotlib.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#767d82", "axes.linewidth": 0.6,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    "figure.facecolor": "white", "legend.frameon": False, "axes.unicode_minus": True,
    "path.simplify": True, "image.composite_image": False,
})
TEAL = "#167773"
GOLD = "#D88725"
BLUE = "#3065A3"
VIOLET = "#805181"
PALE = "#D8E0E0"
EDGE = "#A9B3B3"

def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def write(name, rows):
    with (DATA / name).open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def label(fig, x, y, s, size=8, weight="normal", **kwargs):
    return fig.text(x,y,s,fontsize=size,fontweight=weight,ha=kwargs.pop("ha","left"),
                    va=kwargs.pop("va","center"),color="black",**kwargs)

def rule(fig, y, x1=.052, x2=.953, color=EDGE, lw=.68):
    fig.add_artist(Line2D([x1,x2],[y,y],transform=fig.transFigure,color=color,lw=lw,
                         solid_capstyle="butt",clip_on=False))

def page(head, title, meta, height):
    fig=plt.figure(figsize=(7.0,height))
    label(fig,.052,.958,head,8.1,"bold")
    label(fig,.952,.958,meta,7.3,ha="right")
    label(fig,.050,.888,title,16.2,"bold")
    rule(fig,.831)
    return fig

def panel(fig,x,y,letter,title):
    label(fig,x,y,letter,10.5,"bold")
    label(fig,x+.033,y,title,9.0,"bold")

def footer(fig,line):
    rule(fig,.067)
    label(fig,.052,.036,line,6.45)

def save(fig,stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/(stem+".pdf"),dpi=450,metadata={"Title":fig.texts[2].get_text(),"Creator":"Paper 4 figure study"})
    fig.savefig(FIG/(stem+".png"),dpi=260)
    fig.savefig(FIG/(stem+".tiff"),dpi=450,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

