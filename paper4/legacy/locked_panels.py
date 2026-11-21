"""Common-donor swaps; run python locked_panels.py --repo ../ad_lysosome_plaque --out tables (paths relative to working directory)."""
import csv, hashlib, json
import numpy as np

def save(file, rows):
    with file.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def run(repo, out, data_dir=None):
    data_dir = data_dir or repo / 'data'
    from paper4.four_part.study_tables import read_study, paired_group, LEITNER, MARTA
    rng=np.random.default_rng(20261118); m=json.loads((repo/'data_manifest.json').read_bytes()); sums=[];details=[];heat=[]
    base=['ASAH1','GAA','PLD3']; five=base+['CTSB','PPT1']; seven=five+['ARSA','GBA1']; eight=seven+['SGSH']; targ=['CTSD','TPP1']; ids={'CTSD':'P07339','TPP1':'O14773','ASAH1':'Q13510','GAA':'P10253','PLD3':'Q8IV08','CTSB':'P07858','PPT1':'P50897','ARSA':'P15289','GBA1':'P04062','SGSH':'P51688'}
    for name,fn,groups in [('Leitner',LEITNER,[('AD',53)]),('Marta-Ariza',MARTA,[('EOAD',20),('LOAD',20)])]:
        raw=(data_dir/fn).read_bytes();assert hashlib.sha256(raw).hexdigest()==next(r['sha256'] for r in m if r['name']==fn); meta,labels,prot=read_study(name,raw)
        for grp,n in groups:
            cases,paired,single=paired_group(labels,prot,grp,n)
            for g in (targ+ (eight if name=='Leitner' else five)):assert prot[ids[g]][0]==g
            ts=np.mean([paired[ids[g]][1] for g in targ],axis=0)
            for k,(old,new) in enumerate([(base,five)]+([(five,seven),(five,eight)] if name=='Leitner' else [])):
                dif=np.array([paired[ids[g]][1] for g in new]); mask=np.isfinite(ts)&np.isfinite(dif).all(0); c=np.array(cases)[mask]; a=ts[mask]-dif[:len(old),mask].mean(0);b=ts[mask]-dif[:,mask].mean(0);h=b-a
                np.testing.assert_allclose(h,dif[:len(old),mask].mean(0)-dif[:,mask].mean(0),atol=1e-12)
                z=np.array([a,b,h]).T; bs=z[rng.integers(len(c),size=(10000,len(c)))].mean(axis=1);ci=np.quantile(bs,[.025,.975],axis=0); swap=f'{len(old)}to{len(new)}'; key=dict(study=name,group=grp,swap=swap)
                row=dict(**key,eligible=n,complete=len(c),old='|'.join(old),new='|'.join(new));
                for j,t in enumerate(['old_score','new_score','change']):row[t]=z.mean(0)[j];row[t+'_low']=ci[0,j];row[t+'_high']=ci[1,j]
                sums.append(row)
                for i,case in enumerate(c):
                    details.append(dict(**key,case=case,old_score=a[i],new_score=b[i],change=h[i]));
                    if k==0:heat.append(dict(study=name,group=grp,case=case,**{g:paired[ids[g]][1][cases.index(case)] for g in targ+five}))
                print(name,grp,swap,len(c),'mean',np.round(z.mean(0),5),'CI change',np.round(ci[:,2],5),flush=True)
    out.mkdir(parents=True,exist_ok=True);save(out/'21_locked_swaps.csv',sums);save(out/'22_locked_donors.csv',details);save(out/'23_protein_map.csv',heat)
