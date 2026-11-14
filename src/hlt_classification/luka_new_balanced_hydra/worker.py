"""Hydra production worker: authenticate, build one view, qualify, fit and export."""
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_new_fullstat_hydra import lineage
from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
from hlt_classification.luka_fullstat_balanced_ladder import contrastive
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_u100_multiview.training import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from . import contracts as c, campaign, population, views, startup, training, banks, artifacts, reporting


def allocation(spec,task):
    job=os.environ.get('SLURM_JOB_ID','')
    if task not in c.tasks() or not re.fullmatch('[1-9][0-9]*',job):raise PermissionError('Registered Hydra job required')
    campaign.binding(spec,task,job)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Wrong cluster')
    out=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(x.split('=',1) for x in out.split() if '=' in x)
    expected=dict(JobId=job,JobState='RUNNING',Account='default',QOS='normal',Partition='gpus',
        JobName='lukanbal_'+task,Comment=spec['content_hash']+':'+task,WorkDir=spec['project'])
    if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('Scheduler identity differs')
    for key,want in [('NumCPUs',6),('NumNodes',1),('NumTasks',1),('CPUs/Task',6)]:
        if fields.get(key) not in (str(want),f'{want}-{want}'):raise PermissionError('Allocation differs: '+key)
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    mem=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    if mem is None or Decimal(mem[1])*{'K':Decimal(1)/1024,'':1,'M':1,'G':1024,'T':1024**2}[mem[2]]!=spec['memory_mb']:
        raise PermissionError('Memory allocation differs')
    if allocation_minutes(fields.get('TimeLimit',''))!=spec['minutes'][task]:raise PermissionError('Walltime differs')
    if (os.environ.get('PYTHONNOUSERSITE')!='1' or Path(sys.prefix).resolve()!=Path(spec['environment_dir']).resolve()
            or campaign.runtime()!=spec['environment']):raise ValueError('Runtime/environment drift')
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported()
            or torch.cuda.get_device_name(0)!=c.GPUS[spec['gpu']][1]):raise PermissionError('Expected one registered BF16 GPU')
    return job


def run(spec,task,*,device='cuda:0'):
    campaign.check(spec);job=allocation(spec,task);stage=task.removeprefix('train_');started=time.monotonic()
    folder=Path(spec['root'])/task/'runs'/job;folder.mkdir(parents=True,exist_ok=False)
    if (Path(spec['root'])/task/'result.json').exists():raise FileExistsError('Already published stage')
    torch.set_num_threads(1);phases=PhaseLog(folder,lambda:memory_snapshot(torch));device=torch.device(device)
    try:
        with phases.phase('authenticate_population_and_teachers'):
            campaign.storage(spec['dataset'],spec['root'],spec['storage']['natural_baseline']['spec_path'],
                spec['storage']['declared_available_gib'],resume=True)
            if lineage.verify(spec['dataset'],spec['project'],c.PARENT_HASH)!=spec['foundation_producer']:
                raise ValueError('Original foundation producer differs')
            loaded=population.load(spec['dataset'],spec['population_root'])
            p,_,_,_,arrays,sampler=loaded
            if p!=spec['population']:raise ValueError('Registered population differs')
            memo={}
            dependencies={t:artifacts.completed(spec,t,memo=memo) for t in c.TEACHERS[stage]}
            parents=artifacts.fit_parents(spec,stage,dependencies)
            write_immutable_json(folder/'fit_binding.json',c.artifact('FIT_BINDING',parents=parents,stage=stage,job_id=job))
        with phases.phase('resident_inputs'):
            caches,proof=views.load(spec,stage,loaded)
            tr,val,outside=(caches[k] for k in ('train','validation','outside_validation'))
            if stage=='DE050':
                previous=load_json(dependencies['UE100']['receipt']['files']['view']['path'])
                if previous['mapping']!=proof['mapping']:raise ValueError('UE100/D050 pairing replay differs')
            write_immutable_json(folder/'view_build.json',proof)
            del loaded
        with phases.phase('teacher_banks'):
            targets={t:banks.Bank(r['folder']/'bank',tr,t,r['refs'],spec) for t,r in dependencies.items()}
            matching=contrastive.Sampler(targets['OFFLINE'].keys(),tr.identities,tr.labels,device=device) if stage=='DE000' else None
            training.join(tr,val,targets,sampler,stage)
            resume=artifacts.restore(spec,stage,parents,sampler)
        with phases.phase('integrated_acceptance'):
            proof=startup.run(spec,stage,tr,val,targets,sampler,matching,device,folder,started,
                remaining_epochs=c.EPOCHS-(len(resume['history']) if resume else 0),outside=outside)
            write_immutable_json(folder/'startup.json',proof)
        with phases.phase('fit_or_finalize'):
            def save(epoch,row,state):artifacts.save_epoch(spec['root'],folder,stage,epoch,row,state,parents,job)
            context=contrastive.CPUWorkers(matching,workers=spec['candidate_workers']) if matching else nullcontext()
            with context:
                net,report,best=training.train(tr,val,banks=targets,sampler=sampler,stage=stage,parents=parents,
                    device=device,save_epoch=save,matching=matching,resume=resume)
            training.validate_report(report,stage,parents,len(val),sampler)
            write_immutable_json(folder/'training_report.json',report)
            native=net.deployed().to(device).eval();del net,best,resume,targets,matching
            artifacts.save_state(folder/'selected.pt',{k:v.cpu() for k,v in native.state_dict().items()})
        with phases.phase('selected_validation'):
            native.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True),strict=True)
            probabilities=predict(native,val,device)
            measured=evaluate_probabilities(val.labels,probabilities)
            if measured!=report['validation']:raise ValueError('Selected checkpoint evaluation changed')
            out_prob=predict(native,outside,device)
            for name,cache,prob in [('validation',val,probabilities),('outside_validation',outside,out_prob)]:
                atomic_publish_bytes(folder/(name+'_predictions.npz'),deterministic_npz_bytes(dict(
                    identities=cache.identities,labels=cache.labels,probabilities=prob,pt=arrays[name+'_pt'])))
            metric=reporting.metrics(val,probabilities,outside,out_prob,arrays,dict(report=report['content_hash']))
            write_immutable_json(folder/'validation_metrics.json',metric)
        refs={k:file_ref(folder/name) for k,name in dict(checkpoint='selected.pt',report='training_report.json').items()}
        if stage!='DE000':
            with phases.phase('selected_train_export'):
                banks.export(folder/'bank',native,tr,stage,refs,spec,device)
                banks.Bank(folder/'bank',tr,stage,refs,spec)
        with phases.phase('publication'):
            if campaign.source(spec['project'],spec['commit'])!=spec['source']:raise ValueError('Source drift during fit')
            files=dict(refs,predictions=file_ref(folder/'validation_predictions.npz'),
                outside_predictions=file_ref(folder/'outside_validation_predictions.npz'),
                metrics=file_ref(folder/'validation_metrics.json'),startup=file_ref(folder/'startup.json'),
                view=file_ref(folder/'view_build.json'))
            if stage!='DE000':files['bank']=file_ref(folder/'bank/manifest.json')
            receipt=c.artifact('RESULT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
                stage=stage,job_id=job,fit_parents=parents,files=files,selected_epoch=report['selected_pass'])
            write_immutable_json(Path(spec['root'])/task/'result.json',receipt)
            print(f'HYDRA_BALANCED PUBLISHED stage={stage} epoch={report["selected_pass"]} AUC={measured["macro_ovr_auc"]:.8f}',flush=True)
        return receipt
    except BaseException as exc:
        write_immutable_json(folder/'failure.json',c.artifact('FAILURE',parents=dict(spec=spec['content_hash']),
            stage=stage,job_id=job,error_type=type(exc).__name__,message=str(exc)))
        raise
