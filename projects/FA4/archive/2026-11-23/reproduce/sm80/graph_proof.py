import ctypes,json,torch
from pathlib import Path
lib=ctypes.CDLL('/out/green.so');torch.cuda.init();st=ctypes.c_ulonglong();info=(ctypes.c_int*4)();assert lib.make_green(80,ctypes.byref(st),info)==0
lib.stream_sms.argtypes=[ctypes.c_ulonglong];lib.probe_sms.argtypes=[ctypes.c_ulonglong,ctypes.c_void_p]
rows=[]
for label,s in [('full188',torch.cuda.Stream()),('green80',torch.cuda.ExternalStream(st.value))]:
 seen=torch.zeros(1024,device='cuda',dtype=torch.int32);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):assert lib.probe_sms(s.cuda_stream,seen.data_ptr())==0
 with torch.cuda.stream(s):g.replay()
 s.synchronize();ids=seen.nonzero().flatten().tolist();rows.append(dict(label=label,resource_SM=lib.stream_sms(s.cuda_stream),graph_observed_SM=len(ids),SM_ids=ids))
assert [x['graph_observed_SM'] for x in rows]==[188,80],rows
Path('/out/graph-SM-proof.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
