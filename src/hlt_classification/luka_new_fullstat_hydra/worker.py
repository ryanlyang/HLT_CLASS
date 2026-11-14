"""GPU allocation checks, measured startup and completion-last result export."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from decimal import Decimal
import numpy as np
from hlt_classification.data.cache_contracts import (atomic_publish_bytes,deterministic_npz_bytes,
    load_json,write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from . import contracts as c,campaign,population,prepare


def allocation(spec,task):
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job):raise PermissionError('Real Hydra Slurm allocation required')
    campaign.job_binding(spec,task,job)
    output=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(x.split('=',1) for x in output.split() if '=' in x)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Wrong cluster')
    expected=dict(JobId=job,JobState='RUNNING',Account='default',QOS='normal',Partition='compute' if task=='prepare' else 'gpus')
    if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('Scheduler allocation differs')
    cpus=spec['prepare_cpus'] if task=='prepare' else spec['cpus']
    if fields.get('NumCPUs') not in (str(cpus),f'{cpus}-{cpus}') or fields.get('NumNodes') not in ('1','1-1'):
        raise PermissionError('CPU/node allocation differs')
    for key,envkey,want in [('NumTasks','SLURM_NTASKS',1),('CPUs/Task','SLURM_CPUS_PER_TASK',cpus)]:
        observed=[v for v in (fields.get(key),os.environ.get(envkey)) if v not in (None,'N/A','(null)')]
        if not observed or any(v not in (str(want),f'{want}-{want}') for v in observed):
            raise PermissionError('Task allocation differs: '+key)
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    mem=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    expected_mem=spec['prepare_memory_mb'] if task=='prepare' else spec['memory_mb']
    if mem is None or Decimal(mem[1])*{'K':Decimal(1)/1024,'':1,'M':1,'G':1024,'T':1024**2}[mem[2]]!=expected_mem:
        raise PermissionError('Memory allocation differs')
    if os.environ.get('PYTHONNOUSERSITE')!='1' or Path(sys.prefix).resolve()!=Path(spec['environment_dir']).resolve():
        raise PermissionError('Wrong environment')
    if campaign.runtime()!=spec['environment']:raise ValueError('Installed runtime drift')
    if task=='train':
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
            raise PermissionError('One BF16-capable registered GPU required')
        if torch.cuda.get_device_name(0)!=c.GPUS[spec['gpu']][1]:raise PermissionError('GPU model differs')
    return job


def run(spec,task):
    started=time.monotonic();campaign.check(spec);job=allocation(spec,task)
    from . import lineage
    foundation=load_json(Path(spec['dataset'])/'foundation/foundation.json');c.validate(foundation,'FOUNDATION')
    if foundation['content_hash']!=spec['foundation_sha256']:raise ValueError('Registered foundation changed')
    if lineage.verify(spec['dataset'],spec['project'],foundation['content_hash'])!=spec['foundation_producer']:
        raise ValueError('Foundation producer binding changed')
    folder=Path(spec['root'])/'runs'/job;folder.mkdir(parents=True,exist_ok=False)
    if task=='prepare':
        result=prepare.build(spec['dataset'],spec['cache'],workers=spec['prepare_cpus'],available_gib=spec['available_gib'])
        return result
    if task!='train':raise ValueError('No other training/test task')
    import torch
    from . import training
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.luka_fullstat_balanced_ladder.artifacts import save_state
    torch.set_num_threads(1)
    f,members=population.load(Path(spec['dataset'])/'foundation')
    if f['content_hash']!=spec['foundation_sha256']:raise ValueError('Foundation changed')
    budget=int(spec['memory_mb']*1024**2*.65)
    train=prepare.load(spec['cache'],f,members,role='train',max_ram_bytes=budget)
    val=prepare.load(spec['cache'],f,members,role='validation',max_ram_bytes=budget-train.nbytes)
    del members
    history,_=training.committed(spec['root'],spec['content_hash'],(len(train)+127)//128)
    training.qualify(train,val,folder,spec=spec,remaining_epochs=20-len(history),started=started)
    net,history,selected=training.run(train,val,spec['root'],job,spec_hash=spec['content_hash'])
    save_state(folder/'selected.pt',{k:v.detach().cpu() for k,v in net.state_dict().items()})
    net.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    probabilities=predict(net,val,device='cuda',batch_size=128)
    metrics=evaluate_probabilities(val.labels,probabilities)
    if metrics!=selected['validation']:raise ValueError('Selected validation replay differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        probabilities=probabilities,labels=val.labels,identities=val.identities)))
    report=c.artifact('TRAINING_REPORT',spec=spec['content_hash'],node=c.node(),passes=20,
        selected_pass=selected['pass_number'],validation=metrics,validation_history=history,
        rejection30=training.rejection(val.labels,probabilities,.3),rejection50=training.rejection(val.labels,probabilities,.5),
        recipe=c.recipe(),deployed_graph='privileged_OFFLINE_oracle',scientific_fit=True,
        selected_weights_restored=True,source_commit=spec['commit'],environment=spec['environment'])
    write_immutable_json(folder/'training_report.json',report)
    campaign.check(spec)
    result=c.artifact('RESULT',spec=spec['content_hash'],job=job,
        report=file_ref(folder/'training_report.json',root=Path(spec['root'])),
        checkpoint=file_ref(folder/'selected.pt',root=Path(spec['root'])),
        predictions=file_ref(folder/'validation_predictions.npz',root=Path(spec['root'])))
    write_immutable_json(Path(spec['root'])/'result.json',result)
    return result
