"""Operator-only PR3109/KAD decode coverage over the frozen 120-case domain."""
import argparse, hashlib, importlib, importlib.machinery, json, os, random, statistics, sys, types, traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
PR_ROOT=Path(os.environ['FA4_SOURCE_ROOT'])
CASES=ROOT/'cases/decode.json'
DECODE_POLICY=ROOT/'configs/kad-pr3109-decode-policy.json'
_DECODE_CONFIGS={}

def bootstrap():
 p=types.ModuleType('flash_attn');p.__path__=[str(PR_ROOT)];p.__package__='flash_attn';p.__spec__=importlib.machinery.ModuleSpec('flash_attn',None,is_package=True);p.__spec__.submodule_search_locations=p.__path__;sys.modules['flash_attn']=p;return importlib.import_module('flash_attn.cute.interface')
def timed(fn):
 for _ in range(2):fn()
 torch.cuda.synchronize();v=[]
 for _ in range(5):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();v.append(a.elapsed_time(b))
 return statistics.mean(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();interface=bootstrap();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 cases=json.loads(CASES.read_text())['cases'];random.Random(3109).shuffle(cases);cases=cases[:a.limit] if a.limit else cases
 report={'status':'running','arm':'bf16-source-vs-source-decode','case_count':len(cases),'rows':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
 for c in cases:
  row={'case':c,'status':'running','owner':'pr-fallback'};report['rows'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
  try:
   b,h,hk,d,cap=[c[k] for k in ('batch','h','hk','d','capacity')];page=c['page'];lengths=c['lengths'];dt=torch.bfloat16;gen=torch.Generator(device='cuda').manual_seed(c['seed'])
   q=torch.randn(b,h,d,device='cuda',dtype=dt,generator=gen);lens=torch.tensor(lengths,device='cuda',dtype=torch.int32);cu=torch.arange(b+1,device='cuda',dtype=torch.int32)
   if page:
    k=torch.randn(b*cap//page,page,hk,d,device='cuda',dtype=dt,generator=gen);v=torch.randn_like(k);table=torch.randperm(b*cap//page,device='cuda',generator=gen).int().reshape(b,cap//page)
   else:
    k=torch.randn(b,cap,hk,d,device='cuda',dtype=dt,generator=gen);v=torch.randn_like(k);table=None
   out_t=torch.empty_like(q)
   cfg=None
   def pr_call():return interface._flash_attn_fwd(q,k,v,cu_seqlens_q=cu,seqused_k=lens,page_table=table,max_seqlen_q=1,max_seqlen_k=cap,causal=True,num_splits=0,pack_gqa=None,out=out_t,return_lse=True)[0]
   def kad_call():return interface._flash_attn_fwd(q,k,v,cu_seqlens_q=cu,seqused_k=lens,page_table=table,max_seqlen_q=1,max_seqlen_k=cap,causal=True,num_splits=cfg.get('num_splits', 0) if cfg else 0,pack_gqa=cfg.get('pack_gqa') if cfg else None,tile_mn=((cfg['tile_m'], cfg['tile_n']) if cfg and cfg.get('tile_m') is not None else None),mma_pv_is_rs=cfg.get('mma_pv_is_rs') if cfg else None,intra_wg_overlap=cfg.get('intra_wg_overlap') if cfg else None,out=out_t,return_lse=True)[0]
   pr=pr_call();torch.cuda.synchronize();kad=kad_call();torch.cuda.synchronize();diff=(kad.float()-pr.float()).abs();err=float(diff.max());assert bool(torch.isfinite(kad).all()) and bool(torch.all(diff<=.02+.02*pr.abs()))
   pms,kms=timed(pr_call),timed(kad_call)
   row.update(status='completed',correct=True,max_abs_error=err,owner='source',source_config=None,mean_ms=pms)
  except Exception:row.update(status='failed',error=traceback.format_exc())
  out.write_text(json.dumps(report,indent=2)+'\n');print(c['id'],row['status'],flush=True)
 report['status']='completed';report['completed_rows']=sum(r['status']=='completed' for r in report['rows']);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
