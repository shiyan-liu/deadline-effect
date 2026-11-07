"""Deadline, confidence, cross-venue and lexical checks from released numeric data."""
import gzip,json,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import norm,pearsonr,spearmanr
from review_features import extract_paper_features,compute_global_buckets
P=Path(__file__).resolve().parents[1];OUT=P/'results/recomputed';OUT.mkdir(parents=True,exist_ok=True)

def load(venue,year):
 raw=json.load(gzip.open(P/'data'/f'{venue}{year}_numeric.json.gz','rt'))
 vals=[r['rating'] for p in raw for r in p.get('reviews',[]) if r.get('rating') is not None];lo,hi=min(vals),max(vals)
 papers=[extract_paper_features(p,lo,hi) for p in raw];papers=[p for p in papers if p];papers,_,_=compute_global_buckets(papers)
 reviews=[r for p in papers for r in p['reviews']];n=np.array([len(p['reviews']) for p in papers]);pid=np.repeat(np.arange(len(papers)),n)
 r=np.array([v['rating'] for v in reviews]);means=np.bincount(pid,weights=r)/n
 return raw,papers,reviews,pid,n,r,(r-means[pid])*n[pid]/(n[pid]-1),lo,hi

def ols(X,y,pid):
 inv=np.linalg.pinv(X.T@X);b=inv@X.T@y;u=y-X@b;g=int(pid.max())+1
 scores=np.column_stack([np.bincount(pid,weights=X[:,j]*u,minlength=g) for j in range(X.shape[1])]);v=inv@(scores.T@scores)@inv*g/(g-1)*(len(y)-1)/(len(y)-X.shape[1]);se=np.sqrt(np.diag(v));iid=np.sqrt(np.diag(inv)*(u@u)/(len(y)-X.shape[1]))
 return {'beta':b.tolist(),'se_cluster':se.tolist(),'p_cluster':(2*norm.sf(abs(b/se))).tolist(),'p_iid':(2*norm.sf(abs(b/iid))).tolist()}

def design(bucket):return np.column_stack([np.ones(len(bucket)),bucket==0,bucket==2])
def bucket(values):
 s=np.sort(values);a,b=s[len(s)//3],s[2*len(s)//3]
 return np.where(values>=b,0,np.where(values>=a,1,2))
def epoch(*args):return datetime(*args,tzinfo=timezone.utc).timestamp()*1000
anchors={2024:epoch(2023,11,1),2025:epoch(2024,11,4,12),2026:epoch(2025,11,2)}
all_X=[];all_y=[];all_pid=[];offset=0;report={'iclr':{}}
for year in [2024,2025,2026]:
 raw,pp,rr,pid,n,r,loo,lo,hi=load('iclr',year)
 b=np.array([{'Early':0,'Mid':1,'Late':2}[v['bucket']] for v in rr]);X=design(b);c=np.array([v['confidence'] for v in rr]);c-=c.mean()
 confidence={'baseline':ols(X,loo,pid),'adjusted':ols(np.column_stack([X,c]),loo,pid),'interaction':ols(np.column_stack([X,c,c*(b==0),c*(b==2)]),loo,pid)}
 rp=np.array([v['relative_pos'] for v in rr]);gap=(anchors[year]-np.array([v['cdate'] for v in rr]))/3600000;D=design(bucket(gap))
 deadline={'pearson':float(pearsonr(rp,gap).statistic),'spearman':float(spearmanr(rp,gap).statistic),'regression':ols(D,loo,pid),'anchor_utc_ms':anchors[year]}
 all_X.append(np.column_stack([D,np.full(len(r),year==2025),np.full(len(r),year==2026)]));all_y.append(loo);all_pid.append(pid+offset);offset+=len(pp)
 raw_by_id={v['id']:v for p in raw for v in p['reviews'] if v.get('id')};f=np.array([raw_by_id[v['review_id']]['ai_features'] for v in rr]);valid=f[:,0]>=20;e=f[(b==0)&valid];l=f[(b==2)&valid]
 pooled=np.sqrt(((len(e)-1)*e[:,2].var(ddof=1)+(len(l)-1)*l[:,2].var(ddof=1))/(len(e)+len(l)-2))
 ai={'density_cohens_d':float((l[:,2].mean()-e[:,2].mean())/pooled),'presence_difference_pp':float(100*((l[:,1]>0).mean()-(e[:,1]>0).mean())),'early_n':len(e),'late_n':len(l)}
 report['iclr'][str(year)]={'confidence':confidence,'deadline':deadline,'ai':ai};print('Validated',year,flush=True)
report['pooled_deadline']=ols(np.concatenate(all_X),np.concatenate(all_y),np.concatenate(all_pid))
raw,pp,rr,pid,n,r,loo,lo,hi=load('neurips',2025);rp=np.array([v['relative_pos'] for v in rr]);b=np.array([{'Early':0,'Mid':1,'Late':2}[v['bucket']] for v in rr]);gap=(epoch(2025,7,3,12)-np.array([v['cdate'] for v in rr]))/3600000
# Match the legacy relative-timing normalisation before leave-one-out subtraction.
rnorm=np.round((r-lo)/(hi-lo),6);means=np.bincount(pid,weights=rnorm)/n;ynorm=(rnorm-means[pid])*n[pid]/(n[pid]-1)
byid={v['id']:v for p in raw for v in p['reviews'] if v.get('id')};ft=np.array([byid[v['review_id']]['neurips_features']+[v['confidence']] for v in rr]);e=ft[b==0];l=ft[b==2]
report['neurips2025']={'papers':len(pp),'reviews':len(rr),'acceptance_rate':float(np.mean([p['decision_binary'] for p in pp])),'relative':ols(design(b),ynorm,pid),'deadline':ols(design(bucket(gap)),loo/(hi-lo),pid),'pearson':float(pearsonr(rp,gap).statistic),'spearman':float(spearmanr(rp,gap).statistic),'engagement_late_minus_early':(l.mean(axis=0)-e.mean(axis=0)).tolist(),'legacy_engagement_d':((l.mean(axis=0)-e.mean(axis=0))/ft[np.isin(b,[0,2])].std(axis=0,ddof=1)).tolist()}
(OUT/'validation.json').write_text(json.dumps(report,indent=2));print('Validation complete.')
