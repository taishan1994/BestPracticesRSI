"""Measure PR3109 versus KAD on the frozen 165-case forward domain.

Operator-only: the PR source namespace is loaded directly; varlen, local-window,
resource failures, and unsupported D256 cases remain on the PR fallback.
"""
import argparse, hashlib, importlib, importlib.machinery, json, os, random, statistics, sys, types, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PR_ROOT=Path(os.environ['FA4_SOURCE_ROOT'])
CASES=ROOT/'optimization/pr3109-forward-grid-v48/cases.json'
CASE_POLICY=ROOT/'configs/kad-pr3109-case-policy.json'
_CASE_CONFIGS={}

def bootstrap():
    package=types.ModuleType('flash_attn'); package.__path__=[str(PR_ROOT)]; package.__package__='flash_attn'
    package.__spec__=importlib.machinery.ModuleSpec('flash_attn',None,is_package=True); package.__spec__.submodule_search_locations=package.__path__
    sys.modules['flash_attn']=package
    return importlib.import_module('flash_attn.cute.interface')

def cfg(case):
    return None

def make_direct(q,k,v,out,case,config):
    import torch,cutlass,cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    try:
        from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
    except ModuleNotFoundError:
        from flash_attn.cute.sm120.flash_fwd import FlashAttentionForwardSm120
    from flash_attn.cute.utils import AuxData
    dtype=cutlass.BFloat16 if q.dtype==torch.bfloat16 else cutlass.Float16
    kernel=FlashAttentionForwardSm120(dtype,case['d'],case['d'],case['h']//case['hk'],is_causal=case['causal'],is_local=False,num_stages=1,**config)
    try:
        capable=kernel.can_implement(dtype,case['d'],case['d'],config['tile_m'],config['tile_n'],1,config['num_threads'],case['causal'],Q_in_regs=config['Q_in_regs'])
    except TypeError:
        capable=kernel.can_implement(dtype,case['d'],case['d'],config['tile_m'],config['tile_n'],1,config['num_threads'],case['causal'])
    if not capable:return None
    aux=AuxData(); args=(q,k,v,out,None,case['d']**-.5,None,None,None,None,None,None,None,None,None,aux,None,None)
    converted=list(args)
    for i,t in enumerate((q,k,v,out)):converted[i]=to_cute_tensor(t)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled=cute.compile(kernel,*converted,stream,options='--enable-tvm-ffi')
    return lambda:compiled(*args)

def make_kad_call(interface,q,k,v,out,case,config,kwargs):
    """Use PR3109's own interface with a KAD-selected tile policy.

    This keeps the PR source ABI and scheduler ownership intact; KAD changes
    only the knowledge-selected tile/packing knobs.
    """
    return lambda: interface._flash_attn_fwd(
        q,k,v,causal=case['causal'],window_size_left=case['left'],
        window_size_right=0 if case['left'] is not None else None,
        num_splits=config.get('num_splits', 0),pack_gqa=config['pack_gqa'],tile_mn=(config['tile_m'],config['tile_n']),
        mma_pv_is_rs=config.get('mma_pv_is_rs'),intra_wg_overlap=config.get('intra_wg_overlap'),
        out=out,return_lse=True,**kwargs)[0]

def timed(fn,torch):
    for _ in range(2):fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();vals.append(a.elapsed_time(b))
    return statistics.mean(vals)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=3109);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    import torch
    torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    interface=bootstrap();cases=json.loads(CASES.read_text())['cases'];random.Random(a.seed).shuffle(cases)
    if a.limit: cases=cases[:a.limit]
    report={'status':'running','arm':'bf16-source-vs-source','case_count':len(cases),'rows':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    cache={}
    for case in cases:
        row={'case':case,'status':'running','candidate_config':None};report['rows'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
        try:
            dtype=getattr(torch,case['dtype']); b,s,h,hk,d=[case[k] for k in ('batch','length','h','hk','d')]
            lengths=case.get('lengths',[s]*b); packed=case['layout']=='varlen'; total=sum(lengths)
            gen=torch.Generator(device='cuda').manual_seed(case['seed'])
            q=torch.randn((total,h,d) if packed else (b,s,h,d),device='cuda',dtype=dtype,generator=gen)
            k=torch.randn((total,hk,d) if packed else (b,s,hk,d),device='cuda',dtype=dtype,generator=gen);v=torch.randn_like(k);out_t=torch.empty_like(q)
            cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32) if packed else None
            kwargs=dict(cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=s,max_seqlen_k=s) if packed else {}
            def pr():
                r=interface._flash_attn_fwd(q,k,v,causal=case['causal'],window_size_left=case['left'],window_size_right=0 if case['left'] is not None else None,num_splits=0,pack_gqa=None,out=out_t,return_lse=True,**kwargs); return r[0]
            pr_out=pr();torch.cuda.synchronize(); kad=pr; owner='pr-fallback'; reason=None
            c=cfg(case)
            if c is not None and not packed:
                key=(case['d'],case['h'],case['hk'],case['causal'],case['dtype'],case['batch'],case['length'])
                if key not in cache:cache[key]=make_kad_call(interface,q,k,v,out_t,case,c,kwargs)
                if cache[key] is not None:kad=cache[key];owner='kad-direct'
                else:reason='resource-preflight-fallback'
            kad_out=kad();torch.cuda.synchronize();diff=(kad_out.float()-pr_out.float()).abs()
            assert bool(torch.isfinite(kad_out).all() and torch.all(diff<=.02+.02*pr_out.abs()))
            pr_ms=timed(pr,torch);kad_ms=timed(kad,torch)
            row.update(status='completed',owner='source',reason=None,correct=True,max_abs_error=float(diff.max()),mean_ms=pr_ms)
        except Exception:
            row.update(status='failed',error=traceback.format_exc())
        out.write_text(json.dumps(report,indent=2)+'\n');print(case['id'],row['status'],row.get('owner'),row.get('speedup'),flush=True)
    report['status']='completed'; report['completed_rows']=sum(r['status']=='completed' for r in report['rows']); out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
