import argparse, hashlib, importlib, importlib.machinery, json, os, random, statistics, sys, types, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PR_ROOT=Path(os.environ['FA4_SOURCE_ROOT'])
CASES=ROOT/'optimization/pr3109-forward-grid-v48/cases.json'

def bootstrap():
 p=types.ModuleType('flash_attn');p.__path__=[str(PR_ROOT)];p.__package__='flash_attn';p.__spec__=importlib.machinery.ModuleSpec('flash_attn',None,is_package=True);p.__spec__.submodule_search_locations=p.__path__;sys.modules['flash_attn']=p;return importlib.import_module('flash_attn.cute.interface')
def timed(fn,torch):
 for _ in range(2):fn()
 torch.cuda.synchronize(); vals=[]
 for _ in range(5):
  a,b=torch.cuda.Event(True),torch.cuda.Event(True);a.record();out=fn();b.record();b.synchronize();vals.append(a.elapsed_time(b))
 return statistics.mean(vals),out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--case-id');ap.add_argument('--ncu',action='store_true');ap.add_argument('--bf16',action='store_true');a=ap.parse_args();import torch
 torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');interface=bootstrap()
 cases=json.loads(CASES.read_text())['cases'];random.Random(3109).shuffle(cases);cases=[c for c in cases if not a.case_id or c['id']==a.case_id];cases=cases[:a.limit] if a.limit else cases
 report={'status':'running','arm':'pr3132-fp8-forward','case_count':len(cases),'rows':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
 for case in cases:
  row={'case':case,'status':'running','dtype':'float8_e4m3fn'};report['rows'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
  try:
   dtype=torch.bfloat16 if a.bf16 else torch.float8_e4m3fn;b,s,h,hk,d=[case[k] for k in ('batch','length','h','hk','d')];lengths=case.get('lengths',[s]*b);packed=case['layout']=='varlen';total=sum(lengths);gen=torch.Generator(device='cuda').manual_seed(case['seed'])
   q=torch.randn((total,h,d) if packed else (b,s,h,d),device='cuda',dtype=torch.float32,generator=gen).to(dtype)
   k=torch.randn((total,hk,d) if packed else (b,s,hk,d),device='cuda',dtype=torch.float32,generator=gen).to(dtype);v=torch.randn(k.shape,device='cuda',dtype=torch.float32,generator=gen).to(dtype)
   out_t=torch.empty((*q.shape[:-1],d),device='cuda',dtype=torch.bfloat16)
   cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32) if packed else None
   kwargs=dict(cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=s,max_seqlen_k=s) if packed else {}
   scale=torch.ones((b,hk),device='cuda',dtype=torch.float32)
   q_ref=q.float().to(torch.bfloat16); k_ref=k.float().to(torch.bfloat16); v_ref=v.float().to(torch.bfloat16); ref_out=torch.empty_like(out_t)
   def call_ref():
    r=interface._flash_attn_fwd(q_ref,k_ref,v_ref,causal=case['causal'],window_size_left=case['left'],window_size_right=0 if case['left'] is not None else None,out=ref_out,return_lse=True,**kwargs);return r[0]
   def call():
    kw=dict(q_descale=scale,k_descale=scale,v_descale=scale) if not a.bf16 else {};r=interface._flash_attn_fwd(q,k,v,causal=case['causal'],window_size_left=case['left'],window_size_right=0 if case['left'] is not None else None,out=out_t,return_lse=True,**kw,**kwargs);return r[0]
   if a.ncu:
    torch.cuda.synchronize();torch.cuda.nvtx.range_push('pr3132_fp8_ncu');result=call();torch.cuda.synchronize();torch.cuda.nvtx.range_pop();row.update(status='ncu_profiled',correct_finite=bool(torch.isfinite(result).all()));out.write_text(json.dumps(report,indent=2)+'\n');print(case['id'],row['status'],flush=True);continue
   result=call();reference=call_ref();torch.cuda.synchronize(); assert bool(torch.isfinite(result).all()),'nonfinite output'
   err=float((result.float()-reference.float()).abs().max());ms,result=timed(call,torch);ref_ms,_=timed(call_ref,torch);row.update(status='completed',correct_finite=True,max_abs_error=err,max_abs_output=float(result.float().abs().max()),fp8_mean_ms=ms,reference_bf16_mean_ms=ref_ms)
  except Exception: row.update(status='failed',error=traceback.format_exc())
  out.write_text(json.dumps(report,indent=2)+'\n');print(case['id'],row['status'],row.get('fp8_mean_ms'),flush=True)
 report['status']='completed';report['completed_rows']=sum(r['status']=='completed' for r in report['rows']);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
