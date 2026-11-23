"""Qualified BF16 FA4 GEMV+combine direct launcher for one fixed SM120 decode profile.

Uses the pinned PR2634 kernels and the separate return-ABI patch; no quantization
and no replacement attention library. The source API is used once to select and
compile its own plan. Current pointers, fresh scratch/output and EnvStream are
bound on every call. No mutable scratch is shared between calls or CUDA streams.
"""
import hashlib,json,math,os,importlib.metadata as metadata
from pathlib import Path
import torch
from flash_attn.cute import interface,flash_attn_func


def make_decode(source_root, patch_manifest, device=0, policy_path=None):
    source=Path(source_root);manifest=json.loads(Path(patch_manifest).read_text())
    if Path(interface.__file__).resolve()!=(source/'interface.py').resolve():
        raise RuntimeError('decode requires the separate pinned FA4 source profile')
    if hashlib.sha256((source/'interface.py').read_bytes()).hexdigest()!=manifest['after_sha256']:
        raise RuntimeError('decode source/patch changed')
    policy=json.loads(Path(policy_path or Path(patch_manifest).with_name('decode-policy.json')).read_text())
    if torch.__version__!=policy['software']['torch'] or metadata.version('nvidia-cutlass-dsl')!=policy['software']['cutlass']:
        raise RuntimeError('decode software pin mismatch')
    for name,digest in policy['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('decode source changed: '+name)
    bound=torch.device('cuda',device)
    if torch.cuda.get_device_name(bound)!=policy['software']['gpu']:
        raise ValueError('decode GPU model differs from qualified profile')
    if torch.cuda.get_device_capability(bound)!=(12,0) or 'RTX PRO 6000' not in torch.cuda.get_device_name(bound):
        raise ValueError('decode profile is qualified only on RTX PRO 6000 SM120')
    if torch.is_grad_enabled():raise ValueError('decode is inference-only')
    q=torch.zeros(1,1,16,128,device=bound,dtype=torch.bfloat16)
    k=torch.zeros(1,8192,4,128,device=bound,dtype=torch.bfloat16);v=torch.zeros_like(k)
    original=os.environ.get('FLASH_ATTENTION_SM120_DECODE_KERNEL')
    try:
        os.environ['FLASH_ATTENTION_SM120_DECODE_KERNEL']='1'
        flash_attn_func(q,k,v);torch.cuda.synchronize(bound)
    finally:
        if original is None:os.environ.pop('FLASH_ATTENTION_SM120_DECODE_KERNEL',None)
        else:os.environ['FLASH_ATTENTION_SM120_DECODE_KERNEL']=original
    import cutlass
    candidates=[key for key in interface._flash_attn_fwd.decode_compile_cache if key[0]==cutlass.BFloat16 and key[1]==cutlass.BFloat16 and key[2]==128 and key[3]==4 and key[5]==64 and not key[6] and not key[7]]
    if len(candidates)!=1:raise RuntimeError('ambiguous decode plan; initialize in an isolated profile before other decode shapes')
    dkey=candidates[0];splits=dkey[4]
    if splits!=policy['num_splits']:raise RuntimeError('source-selected split plan changed')
    decode=interface._flash_attn_fwd.decode_compile_cache[dkey]
    ckey=(120,cutlass.BFloat16,cutlass.Float32,128,16,8,128,max(5,math.ceil(math.log2(splits))),False,False,False,False,False,False)
    combine=interface._flash_attn_fwd_combine.compile_cache[ckey]
    scale=cutlass.Float32(128**-.5)
    qshape=(1,1,16,128);kvshape=(1,8192,4,128)

    def forward(q,k,v,*,causal=False):
        if not isinstance(causal,bool):raise TypeError('causal must be bool')
        if torch.is_grad_enabled() or any(t.requires_grad for t in (q,k,v)):raise ValueError('inference-only decode')
        if q.shape!=qshape or k.shape!=kvshape or v.shape!=kvshape:raise ValueError('shape outside qualified decode profile')
        if any(t.device!=bound or t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.data_ptr()%16 for t in (q,k,v)):raise ValueError('decode input contract mismatch')
        output=torch.empty_like(q)
        partial=torch.empty(splits,1,1,16,128,device=bound,dtype=torch.float32)
        lse=torch.empty(splits,1,16,1,device=bound,dtype=torch.float32)
        decode(q,k,v,partial,lse,scale)
        combine(partial,lse.transpose(-1,-2),output,None,None,None,None,None,None)
        return output
    forward.readback=dict(decode_key=repr(dkey),combine_key=repr(ckey),num_splits=splits,partial_dtype='float32',input_dtype='bfloat16',kv_quantization=False,source=manifest)
    return forward
