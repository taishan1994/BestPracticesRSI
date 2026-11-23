"""Offline integrity and numerical checks. No CUDA, network or host installs."""
import argparse,csv,hashlib,json,math,os,re,sys
from pathlib import Path
from html.parser import HTMLParser
R=Path(__file__).resolve().parents[1]
SKIP={'_sources','_work','__pycache__'}
def files():
 for root,dirs,names in os.walk(R):
  dirs[:]=[d for d in dirs if d not in SKIP]
  for name in names:
   p=Path(root)/name
   if p.relative_to(R).as_posix()!='MANIFEST.json':yield p

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gm(x):return math.exp(sum(math.log(v) for v in x)/len(x))
def check_target(base,target):
 target=target.split('#',1)[0]
 if not target or target.startswith(('http:','https:','mailto:','data:')):return
 assert not target.startswith('/'),('absolute link',base,target)
 p=(base.parent/target).resolve();assert p.is_relative_to(R) and p.exists(),('broken link',base,target)
class Links(HTMLParser):
 def __init__(self,path):super().__init__();self.path=path
 def handle_starttag(self,tag,attrs):
  if tag=='a' and dict(attrs).get('href'):check_target(self.path,dict(attrs)['href'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
 if a.write_manifest:
  entries=[dict(path=x.relative_to(R).as_posix(),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(files())]
  (R/'MANIFEST.json').write_text(json.dumps(dict(schema=1,archive_date='2026-11-23',entries=entries),indent=2)+'\n')
 manifest=json.loads((R/'MANIFEST.json').read_text());seen=set()
 for x in manifest['entries']:
  path=Path(x['path']);assert not path.is_absolute() and '..' not in path.parts
  q=R/path;assert q.is_file() and q.stat().st_size==x['bytes'] and sha(q)==x['sha256'],path;seen.add(path.as_posix())
 assert seen=={p.relative_to(R).as_posix() for p in files()},'Unmanifested or missing artifact'
 cases=json.loads((R/'cases/full285.json').read_text());assert len(cases)==len({x['id'] for x in cases})==285;assert sum(x['kind']=='forward' for x in cases)==165
 for label,name,col in [('same_precision','pr3132-vs-pr3109-bf16-full-285.csv','speedup_pr3109_over_pr3132'),('fp8_error_run','pr3132-fp8-vs-pr3109-bf16-full-285-with-error.csv','speedup')]:
  rr=list(csv.DictReader((R/'reports/pr-comparison'/name).open()));assert len(rr)==285
  summary=json.loads((R/'reports/pr-comparison/recomputed-summary.json').read_text())[label]
  for phase in ['forward','decode']:
   q=[x for x in rr if x['phase']==phase];assert len(q)==summary[phase]['cases'];assert abs(gm([float(x[col]) for x in q])-summary[phase]['geomean_speedup'])<1e-12
 sm=R/'reports/sm80';rows=json.loads((sm/'analysis/comparison.json').read_text());kernels=json.loads((sm/'analysis/kernels.json').read_text());assert len(rows)==285 and len(kernels)==1375
 assert all(x['fixed_all_kernel_resources_equal'] for x in rows)
 assert all(x['full188_spill_count']==x['fixed80_spill_count']==x['auto80_spill_count'] for x in rows)
 assert sum(x['full188_spill_count']>0 for x in rows)==17
 assert sum('registers' in x['full188_limiter'] for x in rows)==205
 groups=json.loads((sm/'analysis/groups.json').read_text())
 for kind in ['forward','decode']:
  rr=[x for x in rows if x['kind']==kind];g=next(x for x in groups if x['group']==kind)
  for pfx in ['fixed80','auto80']:assert abs(gm([x[pfx+'_relative_throughput'] for x in rr])-g[pfx+'_geomean_relative_throughput'])<1e-12
 for x in kernels:
  assert x['launch__sm_count']==(188 if x['arm']=='full188_auto' else 80)
  assert x['RF_at_max_residency_pct']<=100.001
 for phase in ['forward','decode']:
  for suffix in ['-SM-proof.json','-profile-SM-proof.json']:
   proof=json.loads((sm/(phase+suffix)).read_text());assert [x['observed_SM'] for x in proof]==[188,80]
 graph=json.loads((sm/'graph-SM-proof.json').read_text());assert [x['graph_observed_SM'] for x in graph]==[188,80]
 for path in [R/'README.md',R/'reports/pr-comparison/README.md',R/'reproduce/README.md',R/'knowledge/measurement-and-sm120.md',R/'papers/README.md',sm/'RAW_EVIDENCE.md']:
  for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^\s)]+)\)',path.read_text()):check_target(path,target)
 for path in sm.glob('*.html'):Links(path).feed(path.read_text())
 for path in (R/'reproduce').rglob('*.py'):compile(path.read_text(),str(path),'exec')
 for x in json.loads((R/'provenance/runner-audit.json').read_text()):assert sha(R/'provenance/historical-runners'/x['runner'])==x['retained_file_sha256']
 print(json.dumps(dict(verified=True,files=len(seen),archive_bytes=sum(x['bytes'] for x in manifest['entries']),cases=285,kernel_records=1375,baseline_spill_cases=17,RF_limited_cases=205,verification='hashes, coverage, numerical aggregates, SM proofs, primary links, runner fingerprints, Python syntax'),ensure_ascii=False))
if __name__=='__main__':main()
