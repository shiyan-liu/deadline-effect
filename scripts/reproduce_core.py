"""Recompute manuscript quantities from the original, unchanged review snapshots."""
import json,sys,statistics,re,gzip
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm,chi2,ttest_rel,ttest_ind
from review_features import extract_paper_features,compute_global_buckets
PROJECT=Path(__file__).resolve().parents[1]; ROOT=PROJECT; OUT=PROJECT/'results/recomputed'; OUT.mkdir(parents=True,exist_ok=True)
allsets={}; result={}; B=500

def reg(outcome,bucket,pid,extra=None):
 X=np.column_stack([np.ones(len(outcome)),bucket==0,bucket==2]+([] if extra is None else [extra]))
 ix=np.linalg.pinv(X.T@X); beta=ix@X.T@outcome;res=outcome-X@beta
 iid=np.sqrt(np.diag(ix)*(res@res)/(len(outcome)-X.shape[1]))
 scores=np.column_stack([np.bincount(pid,weights=X[:,k]*res) for k in range(X.shape[1])]); G=len(scores)
 cov=ix@(scores.T@scores)@ix*(G/(G-1))*((len(outcome)-1)/(len(outcome)-X.shape[1]))
 se=np.sqrt(np.diag(cov))
 return {'beta':beta.tolist(),'se_cluster':se.tolist(),'p_cluster':(2*norm.sf(abs(beta/se))).tolist(),'t_iid':(beta/iid).tolist(),'p_iid':(2*norm.sf(abs(beta/iid))).tolist()}
def mc(x,y):
 b=int(np.sum(x&~y));c=int(np.sum(~x&y));stat=max(abs(b-c)-1,0)**2/(b+c) if b+c else 0
 return {'chi2':stat,'p':float(chi2.sf(stat,1)),'b':b,'c':c}
for year in [2024,2025,2026]:
 raw=json.load(gzip.open(ROOT/'data'/f'iclr{year}_numeric.json.gz','rt'))
 rr=[r['rating'] for p in raw for r in p.get('reviews',[]) if r.get('rating') is not None];lo,hi=min(rr),max(rr)
 pp=[extract_paper_features(p,lo,hi) for p in raw]; pp=[p for p in pp if p]; pp,q1,q2=compute_global_buckets(pp)
 reviews=[r for p in pp for r in p['reviews']]; n=np.array([p['n_reviewers'] for p in pp]);pid=np.repeat(np.arange(len(pp)),n);y=np.array([p['decision_binary'] for p in pp])
 r=np.array([v['rating'] for v in reviews]); rp=np.array([v['relative_pos'] for v in reviews]);c=np.array([max(v['confidence'],1e-3) for v in reviews]);bucket=np.array([{'Early':0,'Mid':1,'Late':2}[v['bucket']] for v in reviews]);means=np.bincount(pid,weights=r)/n
 dev=r-means[pid];loo=dev*n[pid]/(n[pid]-1)
 sumr=np.bincount(pid,weights=r);sumrp=np.bincount(pid,weights=rp);sumpr=np.bincount(pid,weights=r*rp)
 def score(a):return (sumr+a*sumpr)/(n+a*sumrp)
 fit=minimize_scalar(lambda a:np.mean((score(a)-y)**2),bounds=(-.999,2.1),method='bounded').x
 ht=np.array([1-loo[bucket==0].mean(),1,1-loo[bucket==2].mean()]);weights={'Uniform':np.ones(len(r)),'Confidence':c,'Heuristic':ht[bucket],'OTCA':1+fit*rp,'Joint':(1+fit*rp)*c}
 agg={k:np.bincount(pid,weights=w*r)/np.bincount(pid,weights=w) for k,w in weights.items()}
 threshold=float(np.median(r));bl=abs(means-threshold)<=1; correct={k:(s[bl]>=threshold-1e-12)==y[bl] for k,s in agg.items()}
 briers={k:((s-lo)/(hi-lo)-y)**2 for k,s in agg.items()}
 specs={'Global tertiles':reg(loo,bucket,pid)}
 for cutoff in [24,48,72]:
  gap=np.array([v['gap_hours'] for v in reviews]);bb=np.where(gap>cutoff*2,0,np.where(gap<cutoff/2,2,1));specs[f'Gap {cutoff}h']=reg(dev,bb,pid)
 sound=np.array([v['soundness'] for v in reviews]);sd=sound-np.bincount(pid,weights=sound)[pid]/n[pid];specs['Soundness']=reg(sd,bucket,pid)
 # Paper bootstrap of both outcomes; fixed buckets and paper-level blocks.
 rng=np.random.default_rng(20261107+year);sums=np.zeros((len(pp),3));cnt=np.zeros_like(sums);ls=np.zeros_like(sums)
 for b in range(3):
  mask=bucket==b;sums[:,b]=np.bincount(pid[mask],weights=dev[mask],minlength=len(pp));ls[:,b]=np.bincount(pid[mask],weights=loo[mask],minlength=len(pp));cnt[:,b]=np.bincount(pid[mask],minlength=len(pp))
 boot=[];lb=[];bucket_boot=[]
 for _ in range(B):
  ix=rng.integers(0,len(pp),len(pp));den=cnt[ix].sum(axis=0);v=sums[ix].sum(axis=0)/den;v2=ls[ix].sum(axis=0)/den;boot.append(v[2]-v[1]);lb.append(v2[2]-v2[1]);bucket_boot.append(v2)
 quart=np.sort(means)[[len(pp)//4,len(pp)//2,3*len(pp)//4]];group=np.searchsorted(quart,means,side='left');het=[]
 for g in range(4):
  mask=group[pid]==g;gg=reg(dev[mask],bucket[mask],np.unique(pid[mask],return_inverse=True)[1]);het.append({'n':int(sum(group==g)),'accept':float(y[group==g].mean()),'mean':float(means[group==g].mean()),'beta':gg['beta'][2],'p_iid':gg['p_iid'][2],'se_cluster':gg['se_cluster'][2],'p_cluster':gg['p_cluster'][2]})
 result[str(year)]={'papers':len(pp),'reviews':len(r),'scale':[lo,hi],'rating_values':sorted(set(r.tolist())),'threshold':threshold,'n_bl':int(bl.sum()),'q33':q1,'q66':q2,'slope':fit,'heuristic_weights':ht.tolist(),'specifications':specs,'mean_centred_bootstrap_CI':np.quantile(boot,[.025,.975]).tolist(),'loo_bootstrap_CI':np.quantile(lb,[.025,.975]).tolist(),'bucket_ci95':np.quantile(bucket_boot,[.025,.975],axis=0).T.tolist(),'bucket_means':[float(loo[bucket==b].mean()) for b in range(3)],'bucket_brier':[float(np.mean(((r[bucket==b]-lo)/(hi-lo)-y[pid[bucket==b]])**2)) for b in range(3)],'accuracy':{k:float(v.mean()) for k,v in correct.items()},'gain_pp':{k:float(100*(v.mean()-correct['Uniform'].mean())) for k,v in correct.items()},'mcnemar_vs_uniform':{k:mc(v,correct['Uniform']) for k,v in correct.items()},'mcnemar_otca_heuristic':mc(correct['OTCA'],correct['Heuristic']),'mcnemar_joint_confidence':mc(correct['Joint'],correct['Confidence']),'brier':{k:float(v.mean()) for k,v in briers.items()},'brier_p_vs_uniform':{k:float(ttest_rel(briers['Uniform'],v).pvalue) if k!='Uniform' else None for k,v in briers.items()},'heterogeneity':{'cutoffs':quart.tolist(),'groups':het}}
 # Re-evaluate stored content-weight coefficients under the same threshold rule.
 raw_reviews={v['id']:v for p in raw for v in p.get('reviews',[]) if v.get('id')}
 ft=[]
 for v in reviews:
  ft.append(raw_reviews[v['review_id']]['engagement_features'])
 ft=np.asarray(ft,float);engagement={}
 for fidx,fname in enumerate(['Total words','Weakness words','Weakness/strength ratio','Numerical density','Question count']):
  ee=ft[bucket==0,fidx];ll=ft[bucket==2,fidx];pooled=np.sqrt(((len(ee)-1)*ee.var(ddof=1)+(len(ll)-1)*ll.var(ddof=1))/(len(ee)+len(ll)-2));engagement[fname]={'early':float(ee.mean()),'late':float(ll.mean()),'delta':float(ll.mean()-ee.mean()),'cohens_d':float((ll.mean()-ee.mean())/pooled),'p_welch':float(ttest_ind(ll,ee,equal_var=False).pvalue)}
 result[str(year)]['engagement']=engagement
 z=((ft-ft.mean(axis=0))/ft.std(axis=0,ddof=1)).sum(axis=1);slope=np.cov(r,z,ddof=0)[0,1]/np.var(r);resid=z-(z.mean()+slope*(r-r.mean()))
 cf=json.loads((ROOT/'data/content_coefficients.json').read_text())[str(year)];content={}
 for key,feature in [('raw_text_z_CONTAMINATED',z),('residualized_text_z_HONEST_TEST',resid)]:
  a=cf[key]['fit']['a_star'];w=np.maximum(1+a*feature,.01);sc=np.bincount(pid,weights=w*r)/np.bincount(pid,weights=w);co=(sc[bl]>=threshold-1e-12)==y[bl];content[key]={'accuracy':float(co.mean()),'gain_pp':100*float(co.mean()-correct['Uniform'].mean()),'mcnemar':mc(co,correct['Uniform'])}
 result[str(year)]['content']={'rating_to_composite_slope':float(slope),'evaluations':content}
 allsets[year]={'ids':[p['paper_id'] for p in pp],'agg':agg,'n':n,'sumr':sumr,'sumrp':sumrp,'sumpr':sumpr,'y':y,'bl':bl,'threshold':threshold,'correct_uniform':correct['Uniform']}
 print(year, len(pp), 'papers,', len(reviews), 'reviews', flush=True)
for y in [2024,2025]:
 d=allsets[y];a=result[str(2025 if y==2024 else 2024)]['slope'];s=(d['sumr']+a*d['sumpr'])/(d['n']+a*d['sumrp']);corr=(s[d['bl']]>=d['threshold']-1e-12)==d['y'][d['bl']]
 result[str(y)]['transfer']={'fit_year':2025 if y==2024 else 2024,'accuracy':float(corr.mean()),'gain_pp':100*float(corr.mean()-d['correct_uniform'].mean()),'mcnemar':mc(corr,d['correct_uniform'])}
meta=json.loads((ROOT/'data/metareview_labels.json').read_text()); mr=[]
for v in meta:
 if v.get('llm_enthusiasm') is None:continue
 yy=int(v['year']);ds=allsets[yy];lookup={v:i for i,v in enumerate(ds['ids'])};ix=lookup[v['paper_id']]
 if ds['bl'][ix]:mr.append({'year':yy,'enthusiasm':v['llm_enthusiasm'],'decision':ds['y'][ix],'scores':{k:sc[ix] for k,sc in ds['agg'].items()},'threshold':ds['threshold']})
meta_result={'n_all':sum(v.get('llm_enthusiasm') is not None for v in meta),'n_borderline':len(mr),'cutoffs':{}}
for cutoff in [3,4]:
 yy=np.array([v['enthusiasm']>=cutoff for v in mr]);cc={k:np.array([v['scores'][k]>=v['threshold']-1e-12 for v in mr])==yy for k in ['Uniform','OTCA','Joint']}
 meta_result['cutoffs'][str(cutoff)]={'accuracy':{k:float(v.mean()) for k,v in cc.items()},'mc_joint_uniform':mc(cc['Joint'],cc['Uniform'])}
(OUT/'metareview_audit_20261107.json').write_text(json.dumps(meta_result,indent=2))
(OUT/'manuscript_audit_20261107.json').write_text(json.dumps(result,indent=2))
