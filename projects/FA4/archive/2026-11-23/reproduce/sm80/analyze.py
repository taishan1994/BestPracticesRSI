import os
import sys,json,csv,collections,hashlib,math,datetime,argparse
from pathlib import Path
sys.path.insert(0,'/usr/local/cuda-13.1/nsight-compute-2025.4.1/extras/python');import ncu_report
ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
R=Path(os.environ['FA4_RESULTS_ROOT']);cases=json.loads((R/'cases.json').read_text());case_map={x['id']:x for x in cases};timings=[];profile_done=set()
for phase in ['forward','decode']:
 p=R/(phase+'-timings.json')
 if p.exists():timings+=json.loads(p.read_text())
 p=R/(phase+'-profile-progress.json')
 if p.exists():profile_done.update(x['case']['id'] for x in json.loads(p.read_text()) if x['status']=='completed')
keys=['launch__sm_count','launch__uses_green_context','launch__registers_per_thread','launch__registers_per_thread_allocated','launch__block_size','launch__grid_size','launch__shared_mem_config_size','launch__shared_mem_per_block_allocated','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','launch__occupancy_limit_warps','launch__occupancy_limit_blocks','launch__occupancy_limit_barriers','sm__maximum_warps_avg_per_active_cycle','sm__warps_active.avg.pct_of_peak_sustained_active','sass__inst_executed_register_spilling','sass__inst_executed_register_spilling_mem_local_op_read','sass__inst_executed_register_spilling_mem_local_op_write','sass__inst_executed_register_spilling_mem_shared_op_read','sass__inst_executed_register_spilling_mem_shared_op_write','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
rows=[];report_paths=[]
for phase in ['forward','decode']:
 rp=R/'reports'/(phase+'.ncu-rep')
 if not rp.exists():continue
 report_paths.append(rp);r=ncu_report.load_report(str(rp))
 for ri in range(r.num_ranges()):
  rg=r.range_by_idx(ri)
  for ai in range(rg.num_actions()):
   a=rg.action_by_idx(ai);state=a.nvtx_state();tags=[]
   if state:
    for domain in state.domains():tags+=list(state.domain_by_id(domain).push_pop_ranges())
   tags=[t for t in tags if ':' in t and t.split(':',1)[1] in case_map];assert len(tags)==1,tags
   arm,cid=tags[0].split(':',1)
   if cid not in profile_done:continue
   c=case_map[cid];low=a.name().lower();role='combine' if 'combine' in low else ('prepare' if 'prepare' in low else ('attention' if 'flash_fwd' in low or 'flashattention' in low else 'auxiliary'))
   x=dict(case_id=cid,kind=c['kind'],dtype=c['dtype'],d=c['d'],h=c['h'],hk=c['hk'],batch=c['batch'],layout=c['layout'],length=c.get('length',c.get('capacity')),arm=arm,role=role,kernel=a.name())
   for k in keys:
    m=a.metric_by_name(k);x[k]=m.value() if m is not None else None
   x['RF_KiB_per_CTA']=x['launch__registers_per_thread_allocated']*x['launch__block_size']/256
   lim={k:x['launch__occupancy_limit_'+k] for k in ['registers','shared_mem','warps','blocks','barriers']};lim={k:v for k,v in lim.items() if v is not None and v>0};res=min(lim.values());x['max_CTAs_per_SM']=res;x['limiter']=','.join(k for k,v in lim.items() if v==res);x['RF_pct_per_CTA']=x['RF_KiB_per_CTA']/256*100;x['RF_at_max_residency_pct']=x['RF_pct_per_CTA']*res
   assert x['RF_at_max_residency_pct']<=100.001
   assert x['launch__sm_count']==(188 if arm=='full188_auto' else 80)
   assert x['launch__uses_green_context']==(0 if arm=='full188_auto' else 1)
   theo=x['sm__maximum_warps_avg_per_active_cycle'];assert abs(theo-res*math.ceil(x['launch__block_size']/32))<.01
   for counter in ['sass__inst_executed_register_spilling','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']:assert x[counter] is not None and math.isfinite(x[counter]) and x[counter]>=0,(cid,arm,counter,x[counter])
   rows.append(x)
def savecsv(p,rr):
 if not rr:return
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(R/'analysis/kernels.json').write_text(json.dumps(rows,indent=2));savecsv(R/'analysis/kernels.csv',rows)
bycase=collections.defaultdict(dict);allcase=collections.defaultdict(lambda:collections.defaultdict(list))
for x in rows:
 allcase[x['case_id']][x['arm']].append(x)
 if x['role']=='attention':assert x['arm'] not in bycase[x['case_id']];bycase[x['case_id']][x['arm']]=x
fixed_fields=['kernel','launch__registers_per_thread','launch__registers_per_thread_allocated','launch__block_size','launch__grid_size','launch__shared_mem_config_size','launch__shared_mem_per_block_allocated','RF_KiB_per_CTA','max_CTAs_per_SM','limiter']
summ=[]
for t in timings:
 cid=t['case']['id'];c=case_map[cid]
 if t['status']!='completed' or cid not in profile_done:continue
 p=bycase[cid]
 if len(p)!=3:
  if args.partial:continue
  raise RuntimeError((cid,len(p)))
 f,z,a=[p[k] for k in ['full188_auto','green80_fixed188','green80_auto']];ts=t['arms'];b=ts['full188_auto']['median_ms'];zms=ts['green80_fixed188']['median_ms'];ams=ts['green80_auto']['median_ms']
 assert all(math.isfinite(t) and t>0 for t in [b,zms,ams])
 for arm,v in ts.items():assert v['actual_SM']==(188 if arm=='full188_auto' else 80) and v['planner_SM']==(80 if arm=='green80_auto' else 188);assert len(v['samples_ms'])==9
 x=dict(case_id=cid,kind=c['kind'],dtype=c['dtype'],d=c['d'],h=c['h'],hk=c['hk'],batch=c['batch'],layout=c['layout'],length=c.get('length',c.get('capacity')),causal=c['causal'],left=c.get('left'),distribution=c.get('distribution'),page=c.get('page'),full188_ms=b,green80_fixed188_ms=zms,green80_auto_ms=ams,fixed80_slowdown=zms/b,auto80_slowdown=ams/b,fixed80_relative_throughput=b/zms,auto80_relative_throughput=b/ams,fixed_main_control_differences=','.join(k for k in fixed_fields if f[k]!=z[k]),auto_main_changes=','.join(k for k in fixed_fields if f[k]!=a[k]))
 for label,arm in [('full188','full188_auto'),('fixed80','green80_fixed188'),('auto80','green80_auto')]:
  main=p[arm];rr=allcase[cid][arm]
  for dest,src in [('regs_per_thread','launch__registers_per_thread'),('allocated_regs_per_thread','launch__registers_per_thread_allocated'),('threads_per_CTA','launch__block_size'),('grid_blocks','launch__grid_size'),('RF_KiB_per_CTA','RF_KiB_per_CTA'),('RF_pct_per_CTA','RF_pct_per_CTA'),('max_CTAs_per_SM','max_CTAs_per_SM'),('RF_at_max_residency_pct','RF_at_max_residency_pct'),('limiter','limiter')]:x[label+'_'+dest]=main[src]
  x[label+'_kernel_count']=len(rr)
  for dest,src in [('spill_count','sass__inst_executed_register_spilling'),('local_L1_load_sectors','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),('local_L1_store_sectors','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')]:x[label+'_'+dest]=sum(k[src] for k in rr)
  x[label+'_max_abs_error']=ts[arm]['max_abs_error']
 def fingerprint(arm):return [[k['role']]+[k[f] for f in fixed_fields] for k in allcase[cid][arm]]
 x['fixed_all_kernel_resources_equal']=fingerprint('full188_auto')==fingerprint('green80_fixed188')
 x['fixed_new_spill']=x['full188_spill_count']==0 and x['fixed80_spill_count']>0;x['auto_new_spill']=x['full188_spill_count']==0 and x['auto80_spill_count']>0
 summ.append(x)
if not args.partial:
 assert len(timings)==len(profile_done)==len(summ)==285,(len(timings),len(profile_done),len(summ))
 assert len(set(x['case_id'] for x in summ))==285
 assert all(t['status']=='completed' for t in timings)
savecsv(R/'analysis/comparison.csv',summ);(R/'analysis/comparison.json').write_text(json.dumps(summ,indent=2))
def gm(vals):return math.exp(sum(math.log(v) for v in vals)/len(vals))
groups=[]
if summ:
 for name,ss in [('all',summ),('forward',[x for x in summ if x['kind']=='forward']),('decode',[x for x in summ if x['kind']=='decode'])]+[(kind+'-'+layout,[x for x in summ if x['kind']==kind and x['layout']==layout]) for kind,layout in [('forward','dense'),('forward','varlen'),('decode','dense'),('decode','paged256')]]+[('forward-'+dtype,[x for x in summ if x['kind']=='forward' and x['dtype']==dtype]) for dtype in ['bfloat16','float16']]+[(kind+'-d'+str(d),[x for x in summ if x['kind']==kind and x['d']==d]) for kind in ['forward','decode'] for d in sorted(set(x['d'] for x in summ if x['kind']==kind))]:
  if not ss:continue
  groups.append(dict(group=name,cases=len(ss),fixed80_geomean_slowdown=gm([x['fixed80_slowdown'] for x in ss]),auto80_geomean_slowdown=gm([x['auto80_slowdown'] for x in ss]),fixed80_geomean_relative_throughput=gm([x['fixed80_relative_throughput'] for x in ss]),auto80_geomean_relative_throughput=gm([x['auto80_relative_throughput'] for x in ss]),fixed80_min_slowdown=min(x['fixed80_slowdown'] for x in ss),fixed80_max_slowdown=max(x['fixed80_slowdown'] for x in ss),auto80_min_slowdown=min(x['auto80_slowdown'] for x in ss),auto80_max_slowdown=max(x['auto80_slowdown'] for x in ss),baseline_RF_limited_main_cases=sum('registers' in x['full188_limiter'] for x in ss),fixed80_RF_limited_main_cases=sum('registers' in x['fixed80_limiter'] for x in ss),auto80_RF_limited_main_cases=sum('registers' in x['auto80_limiter'] for x in ss),baseline_spill_cases=sum(x['full188_spill_count']>0 for x in ss),fixed_new_spill_cases=sum(x['fixed_new_spill'] for x in ss),auto_new_spill_cases=sum(x['auto_new_spill'] for x in ss)))
(R/'analysis/groups.json').write_text(json.dumps(groups,indent=2));savecsv(R/'analysis/groups.csv',groups)
validation=dict(status='partial' if args.partial else 'completed',completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),timing_cases=len(timings),profile_cases=len(profile_done),joined_cases=len(summ),kernel_launches=len(rows),main_launches=sum(x['role']=='attention' for x in rows),fixed_resource_differences=[x['case_id'] for x in summ if not x['fixed_all_kernel_resources_equal']],fixed_main_differences=[x['case_id'] for x in summ if x['fixed_main_control_differences']],fixed_spill_count_differences=[x['case_id'] for x in summ if x['fixed80_spill_count']!=x['full188_spill_count']],auto_new_spill_cases=[x['case_id'] for x in summ if x['auto_new_spill']],fixed_new_spill_cases=[x['case_id'] for x in summ if x['fixed_new_spill']],missing_requested_metrics=[k for k in keys if rows and all(x[k] is None for x in rows)],source_commit='064643897f27c3d7b2058743d9e4dbdbf0d66325')
if not args.partial:
 validation['ncu_report_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in report_paths}
 validation['hardware_proofs']={}
 graph_proof=json.loads((R/'graph-SM-proof.json').read_text());assert [x['graph_observed_SM'] for x in graph_proof]==[188,80];validation['CUDA_graph_hardware_SM_proof']=[188,80]
 for phase in ['forward','decode']:
  for prefix in [phase+'-',phase+'-profile-']:
   proof=json.loads((R/(prefix+'SM-proof.json')).read_text());assert [x['resource_SM'] for x in proof]==[188,80];assert [x['observed_SM'] for x in proof]==[188,80];validation['hardware_proofs'][prefix]=[188,80]
 source_base=Path(os.environ['FA4_SOURCE_ROOT']).parent;manifest=json.loads((R/'source-manifest.json').read_text());assert all(hashlib.sha256((source_base/f).read_bytes()).hexdigest()==v for f,v in manifest['files'].items());validation['source_files_reverified']=len(manifest['files'])
 validation['cross_GPU_previous_10_case_resource_differences']='Not performed for this new replay'
(R/'analysis/validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2));print(json.dumps(groups[:3],indent=2))
