"""Shape-qualified FA4 SM120 inference launcher, pinned to an audited source ABI.

Combines upstream SM120 kernels, measured tile/thread choices inspired by qu0b,
and the direct-launch idea demonstrated by Nebius. No installed file is patched.
Revision 2 removes per-call JSON serialization while retaining the same guards.
Compilation is lazy; each invocation binds current input/output pointers and the
current CUDA stream. Unsupported layouts/shapes take the native FA4 API.
"""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import torch
from flash_attn.cute import flash_attn_func, interface
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120


def shape_key(q,k,v,causal):
    return json.dumps([list(q.shape),list(k.shape),list(v.shape),str(q.dtype).removeprefix('torch.'),causal],separators=(',',':'))


def make_attention(policy_path, *, mode='tuned', device=0, kad_policy_path=None):
    if mode not in ('tuned','default','no_qregs','eat','kad','debug_timing'):
        raise ValueError('unknown ablation mode')
    policy=json.loads(Path(policy_path).read_text())
    kad_policy = None
    if mode == 'kad':
        try:
            from .kad_dispatch import load_policy
        except ImportError:  # direct file loading used by the release harness
            from kad_dispatch import load_policy
        kad_policy = load_policy(kad_policy_path or Path(policy_path).with_name('kad-policy.json'))
    if metadata.version('flash-attn-4')!=policy['software']['fa4'] or metadata.version('nvidia-cutlass-dsl')!=policy['software']['cutlass']:
        raise RuntimeError('launcher software pin mismatch')
    if torch.__version__!=policy['software']['torch']:
        raise RuntimeError('launcher Torch pin mismatch')
    source=Path(inspect.getfile(interface)).parent
    if mode != 'debug_timing':
        for name,digest in policy['software']['source_sha256'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('FA4 source ABI changed: '+name)
    bound=torch.device('cuda',device)
    if torch.cuda.get_device_name(bound)!=policy['software']['gpu']:
        raise ValueError('GPU model differs from the qualified policy')
    if torch.cuda.get_device_capability(bound)!=(12,0):
        raise ValueError('this policy is qualified only on SM120')
    qualified={}
    for raw,row in policy['shapes'].items():
        qs,ks,vs,dtype,causal=json.loads(raw)
        qualified[(tuple(qs),tuple(ks),tuple(vs),getattr(torch,dtype),causal)]=row
    compiled={};readbacks=[]
    from flash_attn.cute.utils import AuxData
    aux=AuxData()

    def args(q,k,v,out,timing):
        # ABI is pinned by the complete FA4 Python source fingerprint above.
        return (q,k,v,out,None,q.shape[-1]**-.5,None,None,None,None,None,None,None,None,None,aux,None,None,timing)

    def forward(q,k,v,*,causal=False,timing=None):
        if not isinstance(causal,bool):raise TypeError('causal must be bool')
        if torch.is_grad_enabled() or any(t.requires_grad for t in (q,k,v)):
            raise ValueError('inference launcher requires no_grad and non-gradient inputs')
        if any(t.device!=bound for t in (q,k,v)):
            raise ValueError('input device differs from bound device')
        key=(q.shape,k.shape,v.shape,q.dtype,causal)
        row=qualified.get(key)
        eligible=row is not None and k.dtype==q.dtype and v.dtype==q.dtype and all(t.is_contiguous() and t.data_ptr()%16==0 for t in (q,k,v))
        if not eligible:
            result=flash_attn_func(q,k,v,causal=causal)
            return result[0] if isinstance(result,tuple) else result
        # ``kad`` is an evidence-gated configuration.  It is never inferred
        # from timing at runtime: the policy must carry the complete config
        # and its lower bound must pass the KAD threshold.
        decision = None
        if mode == 'kad':
            try:
                from .kad_dispatch import choose
            except ImportError:
                from kad_dispatch import choose
            shape = dict(b=q.shape[0], sq=q.shape[1], sk=k.shape[1], hq=q.shape[2],
                         hk=k.shape[2], d=q.shape[3], dtype=str(q.dtype).removeprefix('torch.'),
                         causal=causal)
            decision = choose(kad_policy, shape)
            if decision is None:
                result=flash_attn_func(q,k,v,causal=causal)
                return result[0] if isinstance(result,tuple) else result
            selected = decision.config
        # ``eat`` is an offline-promoted configuration.  It is never inferred
        # from a case id at runtime: the policy must carry the complete config
        # and the EAT receipt must have passed numerical/resource gates.
        if mode == 'default':
            selected = row['default']
        elif mode == 'eat':
            selected = row.get('eat', row['tuned'])
        elif mode != 'kad':
            selected = row['tuned']
        cfg=dict(selected)
        if mode == 'debug_timing':
            # Debug-only tuning knobs; production policies remain immutable.
            for key, env_name in [('tile_m','FA4_TUNE_TILE_M'),('tile_n','FA4_TUNE_TILE_N'),('num_threads','FA4_TUNE_THREADS')]:
                if os.environ.get(env_name): cfg[key]=int(os.environ[env_name])
            if os.environ.get('FA4_TUNE_Q_IN_REGS') is not None:
                cfg['Q_in_regs']=os.environ['FA4_TUNE_Q_IN_REGS']=='1'
            debug_stages=int(os.environ.get('FA4_TUNE_STAGES','1'))
        else:
            debug_stages=1
        if mode=='no_qregs':cfg['Q_in_regs']=False
        out=torch.empty_like(q,memory_format=torch.contiguous_format)
        if mode == 'debug_timing' and timing is None:
            raise ValueError('debug_timing requires a flat int64 timing tensor')
        if key not in compiled:
            import cutlass
            import cutlass.cute as cute
            from flash_attn.cute.cute_dsl_utils import to_cute_tensor
            dtype=cutlass.BFloat16 if q.dtype==torch.bfloat16 else cutlass.Float16
            kernel=FlashAttentionForwardSm120(dtype,q.shape[-1],v.shape[-1],q.shape[-2]//k.shape[-2],is_causal=causal,is_local=False,num_stages=debug_stages,**cfg)
            if not kernel.can_implement(dtype,q.shape[-1],v.shape[-1],cfg['tile_m'],cfg['tile_n'],debug_stages,cfg['num_threads'],causal,Q_in_regs=cfg['Q_in_regs']):
                raise ValueError('configuration exceeds SM120 resource/layout preflight')
            aa=list(args(q,k,v,out,timing))
            for i,t in enumerate((q,k,v,out)):aa[i]=to_cute_tensor(t)
            if timing is not None: aa[-1]=to_cute_tensor(timing)
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            # Keep CUDA line tables in the debug binary so NCU SourceCounters can
            # correlate PC samples back to the QK/softmax/PV/correction anchors.
            compiled[key]=cute.compile(kernel,*aa,stream,options='--enable-tvm-ffi --generate-line-info')
            readbacks.append(dict(key=shape_key(q,k,v,causal),tile_m=kernel.tile_m,tile_n=kernel.tile_n,num_threads=kernel.num_threads,Q_in_regs=kernel.Q_in_regs,pack_gqa=kernel.pack_gqa,mode=mode,kad_lower_bound=(decision.lower_bound if decision else None)))
        # FFI EnvStream uses the caller's stream; no stored tensor or stream pointers.
        compiled[key](*args(q,k,v,out,timing))
        return out
    forward.readbacks=readbacks
    forward.cache=compiled
    return forward
