"""Extract numeric research inputs from separately obtained OpenReview snapshots."""
import argparse,gzip,json,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
words=json.loads((Path(__file__).resolve().parents[1]/'data/ai_marker_words.json').read_text());marker_re=re.compile(r'\b('+'|'.join(words)+r')\b',re.I)
raw=json.loads(args.source.read_text());out=[]
for paper in raw:
 reviews=[]
 for r in paper.get('reviews',[]):
  rr={k:r[k] for k in ['id','cdate','rating','confidence','soundness'] if k in r}
  ss=[r.get(k) or '' for k in ['summary','strengths','weaknesses','questions']];full=' '.join(ss);nw=len(full.split());weak=len(ss[2].split());strong=len(ss[1].split())
  rr['engagement_features']=[nw,weak,weak/strong if strong else 0,len(re.findall(r'\d+(\.\d+)?',full))/nw if nw else 0,ss[3].count('?')]
  nf=' '.join([ss[0],ss[2],ss[3]]);nn=len(nf.split());rr['neurips_features']=[nn,len(re.findall(r'\d+(\.\d+)?',nf))/nn if nn else 0,ss[3].count('?')]
  marker_text=' '.join(x.strip() for x in ss if x.strip());num=len(marker_text.split());hits=len(marker_re.findall(marker_text));rr['ai_features']=[num,hits,hits/num*1000 if num else 0]
  reviews.append(rr)
 out.append({'id':paper.get('id',''),'forum':paper.get('forum',paper.get('id','')),'decision':paper.get('decision',''),'reviews':reviews})
args.out.parent.mkdir(parents=True,exist_ok=True)
with gzip.GzipFile(filename=str(args.out),mode='wb',mtime=0) as f:f.write(json.dumps(out,separators=(',',':'),ensure_ascii=True).encode())
print(len(out),'papers exported')
