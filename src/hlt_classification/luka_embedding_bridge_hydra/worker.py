"""Exact Slurm jobs, disposable qualification, and frozen producer publication."""
from datetime import timedelta
from pathlib import Path
import os
import math
import subprocess
import sys
import time
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import artifact,validate,recipe
from .registry import FITS,PRODUCERS,cell
from . import campaign,submission,execution,inputs,banks,resources,qualification,training,recovery,distributed as d


def require_job(spec,task):
    job=os.environ.get('SLURM_JOB_ID','');stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if job and submission.accepted_jobs(spec,stage).get(task)==job:return job
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted campaign task/job')


def receipt(spec,task,*,full=True):
    folder=Path(spec['root'])/task;r=load_json(folder/'receipt.json');validate(r,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if (r['task']!=task or r['job_id']!=submission.accepted_jobs(spec,stage).get(task)
            or r['environment']!=spec['reference']['environment'] or r['allocated_gpus']!=(0 if task=='report' else 2)
            or not math.isfinite(r['execution_seconds']) or r['execution_seconds']<=0):raise ValueError('Task ownership/environment differs')
    required={'results.json'} if task=='report' else {'acceptance.json','resource_diagnostic.json','parity.json','replay.json',
        'replay_predictions.npz','u100_replay.json','preparation.json','data/manifest.json','data/u100_train/manifest.json',
        'data/u100_validation/manifest.json','mini_population.json','cpu_resources.json'}|{'mini_'+t+'/receipt.json' for t in FITS} if task=='setup' else {
        'startup/receipt.json','startup/training_report.json','startup_memory.json','fit/receipt.json','fit/training_report.json',
        'fit/selected.pt','fit/selected_full.pt','fit/validation_predictions.npz'}
    if task in PRODUCERS:required|={'bank/train/manifest.json','bank/validation/manifest.json'}
    if task=='setup':required.add('preparation_reuse.json' if spec['preparation_reuse'] is not None else 'u100_input_probe.json')
    names=[v['path'] for v in r['outputs']]
    if len(names)!=len(set(names)) or not required<=set(names):raise ValueError('Incomplete task inventory')
    from .storage import regular
    for ref in r['outputs']:
        regular(folder,ref['path'])
        if full or ref['path'] in required:validate_file_ref(ref,root=folder)
    return r


def completed_setup(spec,*,full=True):
    receipt(spec,'setup',full=full)
    resources.check_cpu(spec,load_json(Path(spec['root'])/'setup/cpu_resources.json'))
    return verify_setup_payloads(spec,full=full)


def verify_setup_payloads(spec,*,full=True):
    qualification.check_replay(spec);ds=inputs.Dataset(spec)
    if spec['preparation_reuse'] is not None:
        saved=load_json(Path(spec['root'])/'setup/preparation_reuse.json')
        if saved!=spec['preparation_reuse']:raise ValueError('Preparation reuse receipt differs')
        recovery.check(saved,spec)
    else:
        qualification.check_u100_witness(load_json(Path(spec['root'])/'setup/u100_input_probe.json'),spec)
    ds.u100('train');ds.u100('validation')
    if full:
        p=load_json(Path(spec['root'])/'setup/mini_population.json');validate(p,'MINI_POPULATION',dict(spec=spec['content_hash']))
        import numpy as np
        selected=resources.selected_indices(ds,wide=True)
        if p['indices']!={k:v.tolist() for k,v in selected.items()} or p['technical']is not True:raise ValueError('Miniature population differs')
        small=inputs.SubsetDataset(ds,selected)
        for task in FITS:
            target=resources.technical_target(spec,task,small,Path(spec['root'])/'setup')
            r,_=training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task),spec,task,technical=True)
            if r['parents']!=dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],**target.parents) or r['world_size']!=2:
                raise ValueError('Production miniature lineage/execution differs')
    return resources.accepted(spec)


def qualify_cpu(spec):
    """Disposable CPU process: measure authentication and full-size report kernels."""
    from hlt_classification.luka_subclass_ddp_hydra.distributed import host_usage,launcher_rss
    from .reporting import measure_cpu
    started=time.perf_counter();campaign.check(spec);require_job(spec,'setup');execution.allocation(spec)
    verify_setup_payloads(spec)
    # Release authenticates the complete setup inventory. Measure those exact
    # hashes here before the final TASK receipt is published by the launcher.
    folder=Path(spec['root'])/'setup'
    for path in sorted(folder.rglob('*')):
        if path.is_file():file_ref(path,root=folder)
    release_seconds=time.perf_counter()-started;release_peak=host_usage()+launcher_rss()
    measured=measure_cpu(spec)
    value=artifact('CPU_RESOURCES',parents=dict(spec=spec['content_hash']),release_seconds=release_seconds,
        release_peak_bytes=release_peak,report_peak_bytes=host_usage()+launcher_rss(),**measured)
    write_immutable_json(folder/'cpu_resources.json',value)
    resources.check_cpu(spec,value)
    return value


def fit_parents(spec,task):
    """Read-only lineage without opening particle arrays or running inference."""
    root=Path(spec['root']);u=load_json(root/'setup/data/u100_train/manifest.json')
    validate(u,'U100_BANK',dict(**inputs.data_parents(spec),original=spec['embedding_hash']))
    if u['role']!='train':raise PermissionError('Wrong U100 target role')
    parents=dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],u100=u['content_hash'])
    predecessor=cell(task)['parent']
    if predecessor:
        m=load_json(root/predecessor/'bank/train/manifest.json');checkpoint=file_ref(root/predecessor/'fit/selected_full.pt')
        validate(m,'TARGET_BANK',banks.parents(spec,predecessor,checkpoint))
        if m['technical']is not False or m['role']!='train' or m['task']!=predecessor or m['meaning']!=banks.meaning(predecessor):
            raise PermissionError('Scientific parent target type differs')
        parents['predecessor']=m['content_hash']
    return parents


def scientific_target(spec,task,ds):
    predecessor=cell(task)['parent'];b=None
    if predecessor:
        receipt(spec,predecessor)
        root=Path(spec['root'])/predecessor;r,_=training.validate_fit(root/'fit',spec,predecessor)
        if r['parents']!=fit_parents(spec,predecessor):raise ValueError('Producer fit lineage differs')
        b=banks.Bank(root/'bank',spec,predecessor,ds.view('train','D000'),file_ref(root/'fit/selected_full.pt'))
    target=banks.Targets(spec,task,ds,parent_bank=b)
    if dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],**target.parents)!=fit_parents(spec,task):
        raise ValueError('Actual optimization target lineage differs')
    return target


def prepare(spec):
    campaign.check(spec);require_job(spec,'setup');execution.allocation(spec)
    started=time.perf_counter()
    if spec['preparation_reuse'] is not None:value=recovery.materialize(spec)
    else:
        inputs.Bundle(spec['bundle_root']).verify_all();value=inputs.prepare(spec)
    import resource
    from hlt_classification.luka_subclass_ddp_hydra.distributed import launcher_rss
    proof=artifact('PREPARATION',parents=dict(spec=spec['content_hash']),seconds=time.perf_counter()-started,
        host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+launcher_rss(),data=value['content_hash'])
    write_immutable_json(Path(spec['root'])/'setup/preparation.json',proof)
    return proof


def run(spec,task):
    started=time.perf_counter()
    if task not in ('setup','release','report',*FITS):raise ValueError('Unregistered task')
    campaign.check(spec);cpu=task in ('release','report');devices=None if cpu else execution.allocation(spec)
    if cpu:execution.cpu_allocation(spec,task)
    job=require_job(spec,task)
    if task=='release':return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task;claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    cli=str(Path(spec['project_dir'])/'scripts/luka_embedding_bridge_hydra.py');path=str(Path(spec['root'])/'campaign_spec.json')
    if task=='setup':
        if spec['preparation_reuse'] is None:
            print('EMBEDDING_BRIDGE checking frozen U100 before full particle preparation',flush=True)
            subprocess.run([sys.executable,'-u','-B','-s','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
                cli,'rank','--spec',path,'--task','input_probe'],check=True)
        subprocess.run([sys.executable,'-u','-B','-s',cli,'prepare','--spec',path],check=True)
    else:completed_setup(spec,full=False)
    if task=='report':
        from .reporting import results
        write_immutable_json(folder/'results.json',results(spec))
    else:
        print(f'EMBEDDING_BRIDGE allocation: {task}; two GTX1080Ti; FP32; global batch128',flush=True)
        subprocess.run([sys.executable,'-u','-B','-s','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
            cli,'rank','--spec',path,'--task',task],check=True)
        if task=='setup':subprocess.run([sys.executable,'-u','-B','-s',cli,'qualify-cpu','--spec',path],check=True)
    campaign.check(spec)
    outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()]
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,environment=installed_environment(),devices=devices,
        outputs=outputs,execution_seconds=time.perf_counter()-started,allocated_gpus=0 if cpu else 2)
    write_immutable_json(folder/'receipt.json',value)
    return value


def rank(spec,task):
    if task not in ('setup','input_probe',*FITS):raise ValueError('Unregistered GPU task')
    owner='setup' if task=='input_probe' else task
    rank=int(os.environ.get('RANK','-1'))
    if (rank not in (0,1) or os.environ.get('LOCAL_RANK')!=str(rank) or os.environ.get('WORLD_SIZE')!='2'
            or os.environ.get('LOCAL_WORLD_SIZE')!='2' or os.environ.get('GROUP_RANK')!='0'):raise PermissionError('Exactly two local ranks required')
    require_job(spec,owner);devices=execution.allocation(spec)
    if campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:raise ValueError('Source changed')
    expected=artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=owner,job_id=os.environ['SLURM_JOB_ID'])
    if load_json(Path(spec['root'])/'claims'/owner/'intent.json')!=expected:raise PermissionError('Missing exact launcher claim')
    torch.cuda.set_device(rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1';dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    ctx=d.Context(rank,2,torch.device('cuda',rank))
    try:
        if task=='input_probe':banks.input_probe(spec,ctx);return
        started=time.perf_counter();ds=inputs.Dataset(spec);ds.view('train','D000');ds.view('validation','D000');ds.u100('train')
        load_seconds=max(ctx.all(time.perf_counter()-started))
        if task=='setup':resources.measure(spec,ds,ctx,devices,load_seconds)
        else:
            acceptance=resources.accepted(spec)
            if devices!=acceptance['devices']:raise PermissionError('Devices differ from setup')
            folder=Path(spec['root'])/task;torch.cuda.reset_peak_memory_stats(ctx.device);target=scientific_target(spec,task,ds)
            startup=resources.miniature(folder/'startup',spec,task,ds,target,ctx)
            resources.check_memory(spec,startup['fit_memory'],devices)
            ctx.main(lambda:write_immutable_json(folder/'startup_memory.json',artifact('STARTUP',parents=fit_parents(spec,task),
                memory=startup['fit_memory'],real_targets=True,technical_weights_discarded=True)))
            d.release_memory(ctx);m,report=training.fit(folder/'fit',spec,task,ds,target,ctx)
            if task in PRODUCERS:
                banks.export(folder/'bank',spec,task,m,ds,ctx,file_ref(folder/'fit/selected_full.pt'))
                banks.Bank(folder/'bank',spec,task,ds.view('train','D000'),file_ref(folder/'fit/selected_full.pt'))
            del m;d.release_memory(ctx)
    finally:dist.destroy_process_group()
