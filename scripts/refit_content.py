"""Recompute the historical exploratory content-coefficient search."""
import gzip,json
from pathlib import Path
import numpy as np
from review_features import extract_paper_features
P=Path(__file__).resolve().parents[1];out={}
for year in [2024,2025,2026]:
 raw=json.load(gzip.open(P/'data'/f'iclr{year}_numeric.json.gz','rt'));ratings=[r['rating'] for p in raw for r in p['reviews'] if r.get('rating') is not None]
 pp=[extract_paper_features(p,min(ratings),max(ratings)) for p in raw];pp=[p for p in pp if p];rr=[r for p in pp for r in p['reviews']];n=np.array([len(p['reviews']) for p in pp]);pid=np.repeat(np.arange(len(pp)),n)
 lookup={r['id']:r for p in raw for r in p['reviews'] if r.get('id')};ft=np.array([lookup[r['review_id']]['engagement_features'] for r in rr]);r=np.array([r['rating'] for r in rr]);y=np.array([p['decision_binary'] for p in pp]);z=((ft-ft.mean(0))/ft.std(0,ddof=1)).sum(1);slope=np.cov(r,z,ddof=0)[0,1]/r.var();resid=z-(z.mean()+slope*(r-r.mean()));out[str(year)]={}
 for key,x in [('raw_text_z_CONTAMINATED',z),('residualized_text_z_HONEST_TEST',resid)]:
  def mse(a):
   w=1+a*x;valid=np.bincount(pid,weights=(w<=0),minlength=len(pp))==0;den=np.bincount(pid,weights=w);num=np.bincount(pid,weights=w*r)
   return float(np.mean((num[valid]/den[valid]-y[valid])**2)) if valid.any() else float('inf')
  best=0.;loss=mse(0)
  for a in [i*.1 for i in range(-20,21)]:
   m=mse(a)
   if m<loss:loss=m;best=a
  lo,hi=best-.1,best+.1
  for _ in range(20):
   m1=lo+(hi-lo)/3;m2=hi-(hi-lo)/3
   if mse(m1)<mse(m2):hi=m2
   else:lo=m1
  out[str(year)][key]={'fit':{'a_star':round((lo+hi)/2,6)}}
 print(year,out[str(year)],flush=True)
dest=P/'results/recomputed/content_refit.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2))
reference=json.loads((P/'data/content_coefficients.json').read_text());assert out==reference,(out,reference)
print('All content coefficients match the reference.')
