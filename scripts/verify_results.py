"""Verify independently recomputed statistics against the manuscript reference outputs."""
from pathlib import Path
import json,math,hashlib
P=Path(__file__).resolve().parents[1];checks=0

def compare(a,b,path=''):
 global checks
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(path,a,b)
  checks+=1
 else:assert a==b,(path,a,b)
for name in ['manuscript_audit_20261107.json','metareview_audit_20261107.json']:
 compare(json.loads((P/'results/reference'/name).read_text()),json.loads((P/'results/recomputed'/name).read_text()),name)
manifest=json.loads((P/'data/manifest.json').read_text())
for d in manifest['sources']:assert hashlib.sha256((P/d['file']).read_bytes()).hexdigest()==d['sha256']
v=json.loads((P/'results/recomputed/validation.json').read_text())
assert v['neurips2025']['papers']==5541 and v['neurips2025']['reviews']==22373
assert abs(v['pooled_deadline']['beta'][1]+.097843)<1e-6 and abs(v['pooled_deadline']['beta'][2]-.033349)<1e-6
for year,want in [('2024',[-.06,-1.73]),('2025',[.01,.03]),('2026',[-.01,-.71])]:
 z=v['iclr'][year]['ai'];assert abs(z['density_cohens_d']-want[0])<.006;assert abs(z['presence_difference_pp']-want[1])<.006
summary={'passed':True,'numerical_values_checked':checks,'input_hashes_verified':len(manifest['sources']),'validation':'deadline, confidence, AI markers and selected NeurIPS sample','scope':'Computational replay of statistical tables; not fresh paid LLM calls or a claim of ATD invariance.'}
(P/'results/recomputed/verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
