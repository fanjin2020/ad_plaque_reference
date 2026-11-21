"""Render C-alpha geometry from the deposited human V-ATPase coordinate model."""
import gzip
import re
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from . import common
from .common import page, panel, label, footer, read, write

ACCESSION6WM2="bff31536a739100360288ae5058a16b29cea68e15d8ad412a052edc4fd771a32"
SECTOR_ONE="#238883"
SECTOR_ZERO="#D9963D"
ISOFORM="#8B78A0"
NEUTRAL="#C2CDD0"

def table_lines(text, header):
    start=text.index("\n"+header)+1
    block=text[start:text.index("\n#",start)]
    lines=block.splitlines()
    columns=[line.strip().split(".")[-1] for line in lines if line.startswith("_")]
    return columns,lines[len(columns):]

def parse():
    gz=(common.DATA/"6WM2.cif.gz").read_bytes()
    assert hashlib.sha256(gz).hexdigest()==ACCESSION6WM2
    cif=gzip.decompress(gz).decode("utf-8")
    _,refs=table_lines(cif,"_struct_ref.id")
    # In this wwPDB record sequence strings are multiline; UNP rows start with the fields below.
    entities={}
    for line in refs:
        m=re.match(r"^\s*\d+\s+UNP\s+(\S+)\s+(\S+)\s+\S+\s+(\d+)\s*",line)
        if m:
            code,accession,entity=m.groups()
            if code.endswith("_HUMAN"):entities[int(entity)]=accession
    assert len(entities)==16
    ca=[]; ent=[]; seen=set()
    cols,atoms=table_lines(cif,"_atom_site.group_PDB")
    i={s:cols.index(s) for s in ["label_atom_id","label_entity_id","label_asym_id","label_seq_id","pdbx_PDB_model_num","Cartn_x","Cartn_y","Cartn_z"]}
    for line in atoms:
        if not line.startswith("ATOM"):continue
        v=line.split()
        if v[i["label_atom_id"]]!="CA" or v[i["pdbx_PDB_model_num"]]!="1":continue
        e=int(v[i["label_entity_id"]])
        key=(v[i["label_asym_id"]],v[i["label_seq_id"]])
        if e not in entities or key in seen:continue
        seen.add(key);ent.append(e)
        ca.append([float(v[i[k]]) for k in ("Cartn_x","Cartn_y","Cartn_z")])
    coords=np.asarray(ca)
    ent=np.asarray(ent)
    assert len(ca)>6500 and len(ca)==len(seen)
    panel_acc={r["gene"]:r["accession"] for r in read("pump_panel.csv")}
    exact={g for g,a in panel_acc.items() if a in entities.values()}
    assert exact==set(panel_acc)-{"ATP6V1G2"} and "O75348" in entities.values()
    ent_by_acc={v:e for e,v in entities.items()}
    atlas=[dict(gene=g,UniProt=a,sector="V0" if "V0" in g else "V1",
                exact_match=g in exact,entity=ent_by_acc.get(a,""),retained_CA=int(np.sum(ent==ent_by_acc.get(a,-1))))
           for g,a in panel_acc.items()]
    write("structure_mapping.csv",atlas)
    return coords,ent,entities,panel_acc

def unit(v):
    return v/np.linalg.norm(v)

def render_surface(projected, atom_colors, scale):
    # Spherical C-alpha beads (3.55 A radius) supply a coarse surface. Pixel depth and normals
    # come from the atomic coordinates. Smooth local normals for lighting at the export size.
    radius=3.55; pad=5.5
    x,y,z=projected.T
    xmin,xmax=x.min()-pad,x.max()+pad
    ymin,ymax=y.min()-pad,y.max()+pad
    w,h=int(np.ceil((xmax-xmin)*scale))+1,int(np.ceil((ymax-ymin)*scale))+1
    depth=np.full((h,w),-np.inf,dtype="f")
    owner=np.zeros((h,w),dtype=np.int32)
    for j in np.argsort(z):
        cx,cy=(x[j]-xmin)*scale,(ymax-y[j])*scale
        rp=radius*scale
        x0,x1=max(0,int(cx-rp)),min(w,int(cx+rp)+2)
        y0,y1=max(0,int(cy-rp)),min(h,int(cy+rp)+2)
        dx=(np.arange(x0,x1)-cx)/scale
        dy=(np.arange(y0,y1)-cy)/scale
        d2=radius*radius-dy[:,None]**2-dx[None,:]**2
        surface=np.sqrt(np.maximum(d2,0))+z[j]
        local=depth[y0:y1,x0:x1]
        hit=(d2>=0)&(surface>local)
        local[hit]=surface[hit]
        owner[y0:y1,x0:x1][hit]=j
    mask=np.isfinite(depth);wgt=mask.astype("f")
    depth[~mask]=0
    def smooth(k):
        return gaussian_filter(depth,k)/np.maximum(gaussian_filter(wgt,k),.001)
    ds=smooth(.30*scale)
    dy,dx=np.gradient(ds,1/scale)
    norm=np.sqrt(1+dx**2+dy**2)
    lighting=np.clip((.37*dx+.52*dy+.75)/norm,0,1)
    occlusion=np.clip((smooth(2.3*scale)-ds)/7.,0,.90)
    far=np.clip((ds-np.percentile(ds[mask],5))/(np.ptp(ds[mask])+.01),0,1)
    shade=np.clip(.56+.49*lighting-.26*occlusion+.045*far,.33,1.10)
    col=atom_colors[owner]*shade[...,None]+.026*lighting[...,None]**5
    # One-pixel antialias on silhouette, white paper outside.
    alpha=np.clip(gaussian_filter(wgt,.58),0,1)
    rgb=np.where(mask[...,None],np.clip(col,0,1),1)
    rgb=rgb*alpha[...,None]+(1-alpha[...,None])
    return (255*rgb).astype(np.uint8)

def symbol(fig,x,y,color):
    fig.add_artist(Line2D([x],[y],marker="s",ms=6.7,mfc=color,mec="#555C60",mew=.45,
                          ls="none",transform=fig.transFigure))

def render_views():
    xyz,e,entities,panel_data=parse()
    acc=np.array([entities[i] for i in e])
    exact=set(panel_data.values())
    color=[]
    for a,ei in zip(acc,e):
        if a=="Q93050":c=SECTOR_ZERO
        elif a in exact: c=SECTOR_ONE
        elif a=="O75348": c=ISOFORM
        else:c=NEUTRAL
        # Consistent sector hues, subtle subunit shading for the catalytic head.
        base=np.array(to_rgb(c))
        if ei==7:base=base*.85+np.array([.055,.085,.085])
        color.append(base)
    color=np.asarray(color)
    base=xyz[np.isin(e,[11,12])].mean(axis=0)
    head=xyz[np.isin(e,[6,7])].mean(axis=0)
    vertical=unit(head-base)
    toward=xyz[e==5].mean(axis=0)-base;toward=unit(toward-vertical*np.dot(toward,vertical))
    horiz=unit(np.cross(vertical,toward))
    xyz=xyz-(head+base)/2
    def view(angle,rotation=0):
        az=np.deg2rad(rotation);tilt=np.deg2rad(angle)
        front=unit(toward*np.cos(az)+horiz*np.sin(az))
        right=unit(np.cross(vertical,front))
        camera=unit(np.cos(tilt)*front+np.sin(tilt)*vertical)
        up=unit(np.cross(camera,right))
        return np.column_stack((xyz@right,xyz@up,xyz@camera))
    side=render_surface(view(4,12),color,7.6)
    top=render_surface(view(89,12),color,6.)
    return side, top, len(e), len(entities), len(exact & set(entities.values()))


def run():
    side, top, atoms, entity_count, matches = render_views()
    fig=page("STRUCTURAL REFERENCE","Placing the pump reference on human V-ATPase","Cryo-EM coordinates   /   PDB 6WM2",5.52)
    panel(fig,.052,.775,"A","Side view")
    ax=fig.add_axes([.049,.146,.49,.576]);ax.imshow(side,interpolation="antialiased");ax.set_axis_off()
    panel(fig,.555,.775,"B","View from the cytosol")
    bx=fig.add_axes([.639,.521,.225,.204]);bx.imshow(top,interpolation="antialiased");bx.set_axis_off()
    panel(fig,.555,.458,"C","Which panel members map?")
    symbol(fig,.570,.391,SECTOR_ONE);label(fig,.590,.391,"Cytosolic V1 sector",8.5,"bold")
    label(fig,.590,.355,"ATP6V1A, B2, C1, D, E1 and H",7.6)
    symbol(fig,.570,.290,SECTOR_ZERO);label(fig,.590,.290,"Membrane V0 sector",8.5,"bold")
    label(fig,.590,.253,"ATP6V0A1",7.6)
    symbol(fig,.570,.191,ISOFORM);label(fig,.590,.191,"G-subunit isoform",8.5,"bold")
    label(fig,.590,.145,"G2 in the plaque panel\nG1 in the deposited structure",7.45,linespacing=1.32)
    symbol(fig,.066,.105,NEUTRAL);label(fig,.083,.105,"Other human subunits in this structure",7.15)
    footer(fig,"Cα bead representation of deposited human protein coordinates. Seven of the eight panel proteins match by UniProt accession.")
    common.save(fig,"04_vatpase_structure")
    print("Human C-alpha atoms:",atoms,"; human polymer entities:",entity_count,"; exact panel matches:",matches)
