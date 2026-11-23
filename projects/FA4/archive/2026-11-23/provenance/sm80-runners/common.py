import sys,json,types,importlib,importlib.machinery,itertools,traceback,argparse,gc
from pathlib import Path
import torch
ROOT=Path('/workspace/projects/FA4')
p=types.ModuleType('flash_attn');p.__path__=['/workspace/projects/FA4/profile/pr3109-registers-20261123/source/flash_attn'];p.__package__='flash_attn';p.__spec__=importlib.machinery.ModuleSpec('flash_attn',None,is_package=True);p.__spec__.submodule_search_locations=p.__path__;sys.modules['flash_attn']=p
interface=importlib.import_module('flash_attn.cute.interface')

torch.set_grad_enabled(False)
def make_call(c):
 b,h,hk,d=[c[k] for k in ('batch','h','hk','d')];dt=getattr(torch,c['dtype']);g=torch.Generator(device='cuda').manual_seed(c['seed'])
 if c['kind']=='forward':
  s=c['length'];lens=c.get('lengths',[s]*b);packed=c['layout']=='varlen';total=sum(lens)
  q=torch.randn((total,h,d) if packed else (b,s,h,d),device='cuda',dtype=dt,generator=g)
  k=torch.randn((total,hk,d) if packed else (b,s,hk,d),device='cuda',dtype=dt,generator=g);v=torch.randn_like(k);out=torch.empty_like(q)
  cu=torch.tensor([0]+list(itertools.accumulate(lens)),device='cuda',dtype=torch.int32) if packed else None
  kw=dict(cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=s,max_seqlen_k=s) if packed else {}
  def call():return interface._flash_attn_fwd(q,k,v,causal=c['causal'],window_size_left=c['left'],window_size_right=0 if c['left'] is not None else None,num_splits=0,pack_gqa=None,out=out,return_lse=True,**kw)
 else:
  cap=c['capacity'];page=c['page'];q=torch.randn(b,h,d,device='cuda',dtype=dt,generator=g);lens=torch.tensor(c['lengths'],device='cuda',dtype=torch.int32);cu=torch.arange(b+1,device='cuda',dtype=torch.int32)
  k=torch.randn((b*cap//page,page,hk,d) if page else (b,cap,hk,d),device='cuda',dtype=dt,generator=g);v=torch.randn_like(k);table=torch.randperm(b*cap//page,device='cuda',generator=g).int().reshape(b,cap//page) if page else None;out=torch.empty_like(q)
  def call():return interface._flash_attn_fwd(q,k,v,cu_seqlens_q=cu,seqused_k=lens,page_table=table,max_seqlen_q=1,max_seqlen_k=cap,causal=True,num_splits=0,pack_gqa=None,out=out,return_lse=True)
 return call
