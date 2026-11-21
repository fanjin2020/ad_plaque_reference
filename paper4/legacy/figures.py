"""Rebuild the four figures and original TOC graphic using tables/; no network needed."""
import numpy as np,pandas as pd,matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black','axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
blue='#226C93'; teal='#11786D'; orange='#BD6428'; purple='#805789'; gray='#788690'; cols=[blue,teal,orange,purple];marks=['o','s','^','D']; groups=[('Leitner','AD'),('Marta-Ariza','EOAD'),('Marta-Ariza','LOAD'),('Drummond 2022','eoAD')]; gl=['Leitner AD','Martá-Ariza EOAD','Martá-Ariza LOAD','Drummond EOAD']
def text(f,x,y,s,size=8,bold=False,**kw):return f.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',va='top',**kw)
def head(f,x,y,l,t):text(f,x,y,l,11,True);text(f,x+.044,y-.002,t,9,True)
def axis(f,r):return f.add_axes(r)
def zero(ax):ax.axvline(0,color='#84939B',ls='--',lw=.7,zorder=0)
def err(ax,x,lo,hi,y,c,m='o'):
    ax.plot([lo,hi],[y,y],color=c,lw=1.25);ax.plot([lo,hi],[y,y],linestyle='none',marker='|',color=c,ms=4);ax.plot(x,y,marker=m,color=c,ls='',mec='black',mew=.45,ms=5)
def key(f,x,y,labels,colors,markers,n=4,step=.24):
    for j,(l,c,m) in enumerate(zip(labels,colors,markers)):
        u=x+step*(j%n);v=y-.052*(j//n);f.add_artist(Line2D([u],[v],transform=f.transFigure,marker=m,ms=5,mfc=c,mec='black',mew=.5,ls=''));f.text(u+.016,v,l,va='center',fontsize=7.1)

def render(tables, out):
    rng = np.random.default_rng(215)
    def read(number):
        return pd.read_csv(next(tables.glob(number + "_*")))
    def save(fig, stem):
        for ext in ("pdf", "png", "tiff"):
            kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
            fig.savefig(out / (stem + "." + ext), dpi={"pdf": 450, "png": 340, "tiff": 600}[ext], **kwargs)
        plt.close(fig)
    # Figure 3: actual donors, never cells. Shared scale across all heatmap strips.
    f=plt.figure(figsize=(6.65,5.1));head(f,.025,.985,'A','Identical three-to-five reference change');head(f,.025,.55,'B','Paired regional differences for each measured donor');sw=read('21');ds=read('22');

    a=axis(f,[.24,.67,.71,.23]);zero(a); 
    for j,(s,g) in enumerate(groups[:3]):
        row=sw.query("study==@s and group==@g and swap=='3to5'").iloc[0];vals=ds.query("study==@s and group==@g and swap=='3to5'").change.to_numpy(); yy=2-j;a.scatter(vals,yy+rng.uniform(-.12,.12,len(vals)),s=10,alpha=.35,c=cols[j],marker=marks[j],edgecolors='none');err(a,row.change,row.change_low,row.change_high,yy,cols[j],marks[j])
    a.set(yticks=[2,1,0],yticklabels=[gl[0]+'  (53)',gl[1]+'  (19)',gl[2]+'  (20)'],ylim=(-.47,2.5),xlabel='Five-control score minus three-control score (log$_2$)');a.xaxis.labelpad=1
    h=read('23'); cm=LinearSegmentedColormap.from_list('region',['#356896','#98B9D0','#FAFAF7','#E5B788','#B1612F']);genes=['CTSD','TPP1','ASAH1','GAA','PLD3','CTSB','PPT1'];left=.10;space=.021;usable=.83
    for j,(s,g) in enumerate(groups[:3]):
        v=h.query('study==@s and group==@g').sort_values('case');w=usable*len(v)/len(h);a=axis(f,[left,.158,w,.274]);im=a.imshow(v[genes].to_numpy().T,aspect='auto',interpolation='nearest',vmin=-1.5,vmax=1.5,cmap=cm);a.set(xlabel=gl[j],xticks=[],yticks=range(7) if j==0 else [],yticklabels=genes if j==0 else []);a.tick_params(length=0);a.axhline(1.5,c='black',lw=.7);a.axhline(4.5,c='black',lw=.7);a.spines[:].set_visible(False);left+=w+space
    cb=f.colorbar(im,cax=axis(f,[.345,.046,.33,.017]),orientation='horizontal',extend='both',ticks=[-1.5,0,1.5]);text(f,.08,.062,'Plaque − adjacent (log$_2$)',7.5);save(f, 'FigS11_fixed_donor_membership')
    # Figure 4: independent display of two target scores plus matched-donor sensitivity.
    f=plt.figure(figsize=(6.65,3.55));head(f,.025,.97,'A','Individual enzymes vs. EC 3 lumen');head(f,.56,.97,'B','Expanding the Leitner reference');a=axis(f,[.19,.26,.32,.56]);zero(a);rr=read('03');
    for j,(s,g) in enumerate(groups):
        for k,gene in enumerate(['CTSD','TPP1']):
            c=gene+' vs controls';v=rr.query("dataset==@s and group==@g and reference_class=='ec3_lysosomal_lumen' and comparison==@c").iloc[0];err(a,v.estimate_log2,v.ci95_low,v.ci95_high,3-j+(.12 if k==0 else -.12),[blue,orange][k],marks[k]);
    a.set(yticks=[3,2,1,0],yticklabels=['Leitner AD','Martá-Ariza EOAD','Martá-Ariza LOAD','Drummond EOAD'],ylim=(-.55,3.55),xlabel='Enzyme − reference (log$_2$)');a=axis(f,[.72,.26,.24,.56]);zero(a)
    for j,swp in enumerate(['5to7','5to8']):
        v=sw.query('swap==@swp').iloc[0];err(a,v.change,v.change_low,v.change_high,1-j,[teal,purple][j],marks[j]);
    a.set(ylim=(-.53,1.52),yticks=[1,0],yticklabels=['5 to 7 (n = 40)','5 to 8 (n = 33)'],xlabel='Score change (log$_2$)',xlim=(-.015,.19),xticks=[0,.08,.16]);key(f,.17,.095,['CTSD','TPP1'],[blue,orange],marks[:2],step=.12);save(f, 'FigS1_enzymes')
