"""Disposable process groups keep 2/3/4-card failures independent and visible."""
from datetime import timedelta
from pathlib import Path
import gc
import os
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from . import campaign as c, data, kernel as k, submission


def gather(value,world):
    if world==1: return [value]
    values=[None]*world; dist.all_gather_object(values,value); return values


def parity(spec,old,batches,rank,world,device):
    root=Path(spec['root']); comparisons={}; refs={}
    for i,case in enumerate(spec['parity_cases']):
        model,opt=k.build(old,spec['maximum'],rank,world,device,parity=True)
        loss,_=k.update(model,opt,batches[f'parity_{i}'],rank,world,device,case['position'])
        if rank==0:
            snap=k.snapshot(model,opt,loss)
            if world==1:
                path=root/f'parity_{i}.pt'; save_state(path,snap); refs[str(i)]=file_ref(path,root=root)
            else:
                m=load_json(root/'parity_reference.json'); c.validate(m,'PARITY',dict(spec=spec['content_hash']))
                ref=m['snapshots'][str(i)]
                if ref['path']!=f'parity_{i}.pt': raise ValueError('Parity snapshot path differs')
                expected=torch.load(validate_file_ref(ref,root=root),map_location='cpu',weights_only=True)
                comparisons[str(i)]=k.compare(expected,snap,spec['tolerances']); del expected
            del snap
        if world>1: dist.barrier()
        del model,opt; gc.collect(); torch.cuda.empty_cache()
    if rank==0 and world==1:
        write_immutable_json(root/'parity_reference.json',c.artifact('PARITY',parents=dict(spec=spec['content_hash']),
            snapshots=refs,dropout_disabled=True,training_batchnorm=True,cases=spec['parity_cases']))
    return comparisons


def measure(spec,old,batches,m,rank,world,device):
    began=time.perf_counter(); model,opt=k.build(old,spec['maximum'],rank,world,device)
    torch.cuda.synchronize(device); initialization=time.perf_counter()-began
    torch.cuda.reset_peak_memory_stats(device)
    call=lambda batch,**kw:k.update(model,opt,batch,rank,world,device,spec['position'],**kw)
    for i in range(spec['warmup_batches']): call(batches[f'train_{i:02}'])
    samples=[]
    for i in range(spec['warmup_batches'],spec['warmup_batches']+spec['measured_batches']):
        elapsed,(loss,_)=k.clock(lambda:call(batches[f'train_{i:02}']),device,world); samples.append(elapsed)
        if rank==0: print(f'R24_BENCH world={world} batch={i+1} seconds={elapsed:.3f} loss={loss:.6f}',flush=True)
    # Diagnostic shuffles are local to each rank. This measures a parallel
    # implementation's cost, not equality to the original global-shuffle probe.
    _,(_,detail)=k.clock(lambda:call(batches['train_00'],probe=True,measure=True),device,world)
    details=gather(detail,world); diagnostic=max(x['probe_seconds'] for x in details)
    model.eval(); model.capture=False
    def validation(batch):
        (raw,_,_),_=k.shard(batch,rank,world)
        with torch.inference_mode():
            logits=k.native(model,raw,device)
            probabilities=logits.softmax(-1).cpu().numpy()
            if not np.isfinite(probabilities).all(): raise ValueError('Nonfinite native validation')
    validation(batches['val_00']); val_seconds=[]
    for i in range(spec['validation_batches']):
        elapsed,_=k.clock(lambda:validation(batches[f'val_{i:02}']),device,world); val_seconds.append(elapsed)
    # If any rank OOMs, torchrun terminates that group; the parent records it
    # and starts the next world size with fresh processes and CUDA contexts.
    stress={}
    for name in ('partial11','tail','longest_hlt','largest_offline'):
        elapsed,_=k.clock(lambda:call(batches[name]),device,world)
        stress[name]=dict(passed=True,seconds=elapsed,rows=len(batches[name][0]['features']))
    import resource
    memory=gather(dict(gpu_peak_bytes=torch.cuda.max_memory_allocated(device),
        gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(device),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),world)
    startup=max(gather(initialization,world))+m['preparation_seconds']
    projection=c.projection(spec,mean_step=float(np.mean(samples)),
        mean_validation_batch=float(np.mean(val_seconds))+m['mean_validation_input_seconds'],
        probe_seconds=diagnostic,input_seconds=m['mean_input_seconds'],startup_seconds=startup)
    return dict(step_seconds=samples,mean_step_seconds=float(np.mean(samples)),
        validation_batch_seconds=val_seconds,stress=stress,memory_by_rank=memory,
        summed_rank_peak_rss_bytes=sum(x['peak_rss_bytes'] for x in memory),
        initialization_seconds=startup-m['preparation_seconds'],diagnostic_seconds=diagnostic,
        diagnostic_scope='per_rank_shuffles_not_original_global_shuffles',profile_by_rank=details,
        projection=projection,active_gpu_hours=world*projection['estimated_hours'],
        idealized_fits_per_gpu_day=24/(world*projection['estimated_hours']))


def rank_main(spec):
    rank=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    if world not in (1,2,3,4) or not 0<=rank<world or int(os.environ.get('RANK','0'))!=rank:
        raise PermissionError('Only local 1/2/3/4-card benchmark phases')
    if world>1 and (int(os.environ.get('LOCAL_WORLD_SIZE','-1'))!=world or int(os.environ.get('GROUP_RANK','-1'))!=0):
        raise PermissionError('One node only')
    submission.require_job(spec); old=c.check(spec)
    claim=load_json(Path(spec['root'])/'claim/intent.json'); c.validate(claim,'CLAIM',dict(spec=spec['content_hash']))
    if claim['job_id']!=os.environ['SLURM_JOB_ID']: raise PermissionError('Wrong worker claim')
    m,batches=data.read(spec); device=torch.device('cuda',rank); torch.cuda.set_device(device); k.configure(world,spec['cpus'])
    require_backend()
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=3))
    try:
        checks=parity(spec,old,batches,rank,world,device)
        if world==1: return dict(reference_written=True)
        row=measure(spec,old,batches,m,rank,world,device)
        if rank==0:
            result=c.artifact('PHASE',parents=dict(spec=spec['content_hash'],batches=m['content_hash']),
                world_size=world,precision='fp32',global_batch=128,parity=checks,
                qualified=all(x['passed'] for x in checks.values()),**row)
            write_immutable_json(Path(spec['root'])/f'world{world}.json',result)
            return result
    finally:
        if world>1: dist.destroy_process_group()


def accounting(job):
    try:
        result=subprocess.run(['sacct','-X','-n','-P','-j',str(job),'-o','JobIDRaw,State,ElapsedRaw'],
            capture_output=True,text=True,timeout=30,check=True)
        for line in result.stdout.splitlines():
            fields=line.split('|')
            if fields[0]==str(job) and fields[1]=='COMPLETED' and fields[2].isdigit():
                return dict(job_id=str(job),state=fields[1],elapsed_seconds=int(fields[2]))
    except (OSError,subprocess.SubprocessError): pass
    return None


def child(spec,name,args,timeout):
    path=Path(spec['root'])/(name+'.log'); began=time.perf_counter()
    print(f'R24_BENCH starting {name}; live output: {path}',flush=True)
    with path.open('x') as log:
        try:
            p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # torchrun owns descendants; kill only this newly created session,
            # then reap it before the next world size obtains CUDA memory.
            os.killpg(p.pid,signal.SIGKILL); p.wait()
            code=124; log.write('\nBenchmark phase time budget exceeded.\n')
    elapsed=time.perf_counter()-began
    print(f'R24_BENCH finished {name}: exit={code}, seconds={elapsed:.1f}',flush=True)
    return dict(name=name,exit_code=code,elapsed_seconds=elapsed,log=file_ref(path,root=spec['root']))


def run(spec):
    began=time.perf_counter(); old=c.check(spec,full=True); job,devices=submission.allocation(spec)
    root=Path(spec['root']); claim=root/'claim'; claim.mkdir(exist_ok=False)
    write_immutable_json(claim/'intent.json',c.artifact('CLAIM',parents=dict(spec=spec['content_hash']),job_id=job,devices=devices))
    cli=str(Path(spec['project_dir'])/'scripts/luka_r24_gpu_benchmark.py'); sp=str(root/'benchmark_spec.json')
    prep=child(spec,'prepare',[sys.executable,'-u','-B','-s',cli,'prepare','--spec',sp],1800)
    if prep['exit_code']: raise RuntimeError('Batch preparation failed; inspect prepare.log')
    reference=child(spec,'reference',[sys.executable,'-u','-B','-s',cli,'rank','--spec',sp],600)
    phases=[]; rows={}; files=[]
    # A missing numerical reference blocks qualification, but does not prevent
    # obtaining per-world failure diagnostics in fresh child groups.
    for world in spec['worlds']:
        phase=child(spec,f'world{world}',[sys.executable,'-u','-B','-s','-m','torch.distributed.run',
            '--standalone','--nnodes=1','--max-restarts=0',f'--nproc-per-node={world}',cli,'rank','--spec',sp],900)
        phases.append(phase); path=root/f'world{world}.json'
        if phase['exit_code']==0 and path.is_file():
            value=load_json(path); c.validate(value,'PHASE',dict(spec=spec['content_hash'],batches=load_json(root/'batches/manifest.json')['content_hash']))
            if value['world_size']!=world: raise ValueError('Phase identity differs')
            rows[str(world)]=value; files.append(file_ref(path,root=root))
    elapsed=time.perf_counter()-began
    report=c.artifact('REPORT',parents=dict(spec=spec['content_hash']),job_id=job,devices=devices,
        preparation=prep,reference_phase=reference,phases=phases,measurements=rows,
        a6000=spec['reference'],a6000_accounting=accounting(spec['reference']['job_id']),
        measured_allocation_seconds=elapsed,allocated_gpu_hours=4*elapsed/3600,
        comparison='practical_FP32_Pascal_vs_historical_BF16_native_A6000_not_equal_precision',
        science_release=False,automatic_retry=False)
    c.check(spec); write_immutable_json(root/'benchmark_report.json',report)
    outputs=[file_ref(root/'benchmark_report.json',root=root),file_ref(root/'batches/manifest.json',root=root),
        prep['log'],reference['log'],*[p['log'] for p in phases],*files]
    if (root/'parity_reference.json').exists():
        outputs.append(file_ref(root/'parity_reference.json',root=root))
        outputs.extend(load_json(root/'parity_reference.json')['snapshots'].values())
    write_immutable_json(root/'receipt.json',c.artifact('RECEIPT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),job_id=job,outputs=outputs))
    return results(spec)


def results(spec):
    c.check(spec); root=Path(spec['root']); report=load_json(root/'benchmark_report.json')
    c.validate(report,'REPORT',dict(spec=spec['content_hash'])); receipt=load_json(root/'receipt.json')
    c.validate(receipt,'RECEIPT',dict(spec=spec['content_hash'],report=report['content_hash']))
    if receipt['job_id']!=submission.jobs(spec).get('probe') or report['job_id']!=receipt['job_id']:
        raise ValueError('Benchmark report ownership differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=root)
    baseline=spec['reference']; training=baseline['train_seconds']/3600
    total=(baseline['train_seconds']+baseline['validation_seconds'])/3600
    print(f'A6000 R24 observed: training={training:.2f}h; training + validation={total:.2f}h')
    if report['a6000_accounting']: print(f'A6000 Slurm elapsed: {report["a6000_accounting"]["elapsed_seconds"]/3600:.2f}h')
    print('FP32 Pascal projections; no queue wait, metric calculation, checkpoint I/O or extra audits. No scientific AUC.')
    summary={}
    for phase in report['phases']:
        world=phase['name'].removeprefix('world'); row=report['measurements'].get(world)
        if row is None:
            print(f'{world} GPUs: FAILED/INCOMPLETE (exit={phase["exit_code"]}); inspect {phase["log"]["path"]}')
            continue
        h=row['projection']['estimated_hours']; parts=row['projection']['component_seconds']
        th=sum(parts[key] for key in ('training','input_construction','diagnostics'))/3600
        speed=training/th; summary[world]=dict(qualified=row['qualified'],projected_hours=h,train_speedup_vs_A6000=speed,
            active_gpu_hours=row['active_gpu_hours'],peak_gpu_gib=[x['gpu_peak_bytes']/2**30 for x in row['memory_by_rank']])
        print(f'{world} GPUs: {row["mean_step_seconds"]:.3f}s/update; training={th:.2f}h; total estimate={h:.2f}h; '
              f'training speed vs A6000={speed:.2f}x; parity={row["qualified"]}; active GPU-hours={row["active_gpu_hours"]:.2f}')
    print(f'This four-GPU benchmark used at least {report["allocated_gpu_hours"]:.2f} allocated GPU-hours (including idle reserved cards).')
    return dict(summary=summary,report=str(root/'benchmark_report.json'))
