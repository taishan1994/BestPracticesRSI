"""Shape-qualified FA4 SM120 inference launcher, pinned to an audited source ABI.

Combines upstream SM120 kernels, measured tile/thread choices inspired by qu0b,
and the direct-launch idea demonstrated by Nebius. No installed file is patched.
Compilation is lazy; each invocation binds current input/output pointers and the
current CUDA stream. Unsupported layouts/shapes take the native FA4 API.
"""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import torch
from flash_attn.cute import flash_attn_func, interface
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120


def shape_key(q,k,v,causal):
    return json.dumps([list(q.shape),list(k.shape),list(v.shape),str(q.dtype).removeprefix('torch.'),causal],separators=(',',':'))


def make_attention(policy_path, *, mode='tuned', device=0):
    if mode not in ('tuned','default','no_qregs'):
        raise ValueError('unknown ablation mode')
    policy=json.loads(Path(policy_path).read_text())
    if metadata.version('flash-attn-4')!=policy['software']['fa4'] or metadata.version('nvidia-cutlass-dsl')!=policy['software']['cutlass']:
        raise RuntimeError('launcher software pin mismatch')
    if torch.__version__!=policy['software']['torch']:
        raise RuntimeError('launcher Torch pin mismatch')
    source=Path(inspect.getfile(interface)).parent
    for name,digest in policy['software']['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('FA4 source ABI changed: '+name)
    bound=torch.device('cuda',device)
    if torch.cuda.get_device_capability(bound)!=(12,0):
        raise ValueError('this policy is qualified only on SM120')
    compiled={};readbacks=[]
    from flash_attn.cute.utils import AuxData
    aux=AuxData()

    def args(q,k,v,out):
        # ABI is pinned by the complete FA4 Python source fingerprint above.
        return (q,k,v,out,None,q.shape[-1]**-.5,None,None,None,None,None,None,None,None,None,aux,None,None)

    def forward(q,k,v,*,causal=False):
        if not isinstance(causal,bool):raise TypeError('causal must be bool')
        if torch.is_grad_enabled() or any(t.requires_grad for t in (q,k,v)):
            raise ValueError('inference launcher requires no_grad and non-gradient inputs')
        if any(t.device!=bound for t in (q,k,v)):
            raise ValueError('input device differs from bound device')
        key=shape_key(q,k,v,causal)
        row=policy['shapes'].get(key)
        eligible=row is not None and all(t.is_contiguous() and t.data_ptr()%16==0 for t in (q,k,v))
        if not eligible:
            result=flash_attn_func(q,k,v,causal=causal)
            return result[0] if isinstance(result,tuple) else result
        cfg=dict(row['default'] if mode=='default' else row['tuned'])
        if mode=='no_qregs':cfg['Q_in_regs']=False
        out=torch.empty_like(q,memory_format=torch.contiguous_format)
        if key not in compiled:
            import cutlass
            import cutlass.cute as cute
            from flash_attn.cute.cute_dsl_utils import to_cute_tensor
            dtype=cutlass.BFloat16 if q.dtype==torch.bfloat16 else cutlass.Float16
            kernel=FlashAttentionForwardSm120(dtype,q.shape[-1],v.shape[-1],q.shape[-2]//k.shape[-2],is_causal=causal,is_local=False,num_stages=1,**cfg)
            if not kernel.can_implement(dtype,q.shape[-1],v.shape[-1],cfg['tile_m'],cfg['tile_n'],1,cfg['num_threads'],causal,Q_in_regs=cfg['Q_in_regs']):
                raise ValueError('configuration exceeds SM120 resource/layout preflight')
            aa=list(args(q,k,v,out))
            for i,t in enumerate((q,k,v,out)):aa[i]=to_cute_tensor(t)
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            compiled[key]=cute.compile(kernel,*aa,stream,options='--enable-tvm-ffi')
            readbacks.append(dict(key=key,tile_m=kernel.tile_m,tile_n=kernel.tile_n,num_threads=kernel.num_threads,Q_in_regs=kernel.Q_in_regs,pack_gqa=kernel.pack_gqa,mode=mode))
        # FFI EnvStream uses the caller's stream; no stored tensor or stream pointers.
        compiled[key](*args(q,k,v,out))
        return out
    forward.readbacks=readbacks
    forward.cache=compiled
    return forward
