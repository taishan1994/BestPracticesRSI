import os,ctypes,json,statistics,random,time,argparse,gc,traceback
from pathlib import Path
from common import torch,make_call
from flash_attn.cute.cute_dsl_utils import get_num_sms_for_selection
ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--case-id');ap.add_argument('--tag',default='');ap.add_argument('--kind',choices=['forward','decode']);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
lib=ctypes.CDLL('/out/harness/green.so');torch.cuda.init();st=ctypes.c_ulonglong();info=(ctypes.c_int*4)();assert lib.make_green(80,ctypes.byref(st),info)==0 and info[3]==80
lib.stream_sms.argtypes=[ctypes.c_ulonglong];lib.probe_sms.argtypes=[ctypes.c_ulonglong,ctypes.c_void_p]
full=torch.cuda.Stream();green=torch.cuda.ExternalStream(st.value)
probes=[]
for label,s in [('full188',full),('green80',green)]:
 seen=torch.zeros(1024,device='cuda',dtype=torch.int32);torch.cuda.synchronize()
 with torch.cuda.stream(s):assert lib.probe_sms(s.cuda_stream,seen.data_ptr())==0
 s.synchronize();ids=seen.nonzero().flatten().tolist();probes.append(dict(stream=label,resource_SM=lib.stream_sms(s.cuda_stream),observed_SM=len(ids),SM_ids=ids))
Path('/out/'+args.tag+('profile-' if args.profile else '')+'SM-proof.json').write_text(json.dumps(probes,indent=2))
assert [x['observed_SM'] for x in probes]==[188,80],probes
cases=json.loads(Path('/out/cases.json').read_text());cases=cases[:args.limit] if args.limit else cases
if args.case_id:cases=[c for c in cases if c['id']==args.case_id]
if args.kind:cases=[c for c in cases if c['kind']==args.kind]
arms=[('full188_auto',full,None),('green80_fixed188',green,188),('green80_auto',green,None)]
result_path=Path('/out/'+args.tag+('profile-progress.json' if args.profile else 'timings.json'))
rows=json.loads(result_path.read_text()) if args.resume and result_path.exists() else []
completed={x['case']['id'] for x in rows if x['status']=='completed'}
for c in cases:
 if c['id'] in completed:continue
 try:
  call=make_call(c);torch.cuda.synchronize();graphs={};outs={};meta={}
  for name,s,override in arms:
   if override:os.environ['FLASH_ATTENTION_NUM_SMS']=str(override)
   else:os.environ.pop('FLASH_ATTENTION_NUM_SMS',None)
   with torch.cuda.stream(s):
    planner=get_num_sms_for_selection(0,120)
    for _ in range(3):ret=call()
   s.synchronize()
   with torch.cuda.stream(s):outs[name]=ret[0].clone()
   s.synchronize();meta[name]=dict(actual_SM=lib.stream_sms(s.cuda_stream),planner_SM=planner)
   if args.profile:
    with torch.cuda.stream(s):
     torch.cuda.nvtx.range_push(name+':'+c['id']);torch.cuda.cudart().cudaProfilerStart()
     try:call();s.synchronize()
     finally:torch.cuda.cudart().cudaProfilerStop();torch.cuda.nvtx.range_pop()
   else:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=s):
     for _ in range(10):call()
    graphs[name]=graph
  ref=outs['full188_auto'];err={}
  for name in outs:
   diff=(outs[name].float()-ref.float()).abs();err[name]=float(diff.max());assert bool(torch.isfinite(outs[name]).all()) and bool(torch.all(diff<=.02+.02*ref.abs())),(name,err)
  samples={name:[] for name,_,_ in arms}
  if not args.profile:
   for rnd in range(9):
    order=list(arms);random.Random(3109+rnd).shuffle(order)
    for name,s,_ in order:
     with torch.cuda.stream(s):
      g=graphs[name];g.replay();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
      for _ in range(10):g.replay()
      b.record()
     b.synchronize();samples[name].append(a.elapsed_time(b)/100)
  row=dict(case=c,status='completed',arms={name:dict(meta[name],max_abs_error=err[name],samples_ms=samples[name],median_ms=statistics.median(samples[name]) if samples[name] else None) for name,_,_ in arms})
  rows.append(row);print(c['id'],'completed', {k:v['median_ms'] for k,v in row['arms'].items()},flush=True)
  del call,graphs,outs,ret,ref
  if not args.profile:del graph,g
  gc.collect();torch.cuda.empty_cache()
 except Exception:
  rows.append(dict(case=c,status='failed',error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
 tmp=result_path.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(result_path)
