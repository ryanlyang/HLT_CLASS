"""Production setup, scheduler-owned release and exact-job scientific workers."""
from pathlib import Path
import json
import os
import re
import subprocess
import time
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_cross_message_hydra import execution
from .contracts import TASKS,ARMS,artifact,validate
from . import campaign,inputs,resources,training,submission


def receipt(spec,task):
    folder=Path(spec['root'])/task; value=load_json(folder/'receipt.json')
    validate(value,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if (value['task']!=task or value['job_id']!=submission.accepted_jobs(spec,stage).get(task)
            or len({r['path'] for r in value['outputs']})!=len(value['outputs'])):
        raise ValueError('Exact job/output ownership differs')
    for ref in value['outputs']: validate_file_ref(ref,root=folder)
    return value


def completed_setup(spec):
    record=receipt(spec,'setup')
    required={'acceptance.json','targets.json'}|{f'{prefix}_{task}/receipt.json' for task in TASKS for prefix in ('warm','mini')}
    if not required<={r['path'] for r in record['outputs']}: raise ValueError('Incomplete setup outputs')
    return resources.accepted(spec)


def cpu_allocation(spec):
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job): raise PermissionError('Slurm CPU allocation required')
    output=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in output.split() if '=' in t)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    observed={k:fields.get(k) for k in ('JobId','JobState','Partition','Account','QOS',
        'NumCPUs','NumNodes','NumTasks','CPUs/Task','ReqTRES','AllocTRES')}
    environment={k:os.environ.get(k) for k in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')}
    print('Particle-teacher CPU allocation evidence: '+json.dumps(dict(controller=observed,
        environment=environment),sort_keys=True),flush=True)
    expected=dict(JobId=job,JobState='RUNNING',Partition=spec['cpu_site']['partition'],
        Account=spec['cpu_site']['account'],QOS=spec['cpu_site']['qos'])
    errors=[]
    def require(name,actual,wanted):
        if actual!=wanted: errors.append(f'{name}: expected {wanted!r}, observed {actual!r}')
    for key,value in expected.items(): require(key,fields.get(key),value)
    require('NumNodes',fields.get('NumNodes'),'1')
    # Hydra job146337 requested one CPU but received two. Authenticate the
    # request separately, then allow only this observed scheduler rounding.
    requested=dict(x.split('=',1) for x in fields.get('ReqTRES','').split(',') if '=' in x)
    allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    require('ReqTRES cpu',requested.get('cpu'),str(spec['cpu_site']['cpus']))
    for label,tres in (('ReqTRES',requested),('AllocTRES',allocated)):
        require(label+' mem',tres.get('mem'),str(spec['cpu_site']['memory_mb'])+'M')
        require(label+' node',tres.get('node'),'1')
        if any(k.startswith('gres/gpu') for k in tres): errors.append(label+' contains GPU resources')
    if allocated.get('cpu') not in ('1','2'):
        errors.append('AllocTRES cpu: expected one CPU or two after scheduler rounding')
    require('NumCPUs versus AllocTRES',fields.get('NumCPUs'),allocated.get('cpu'))
    for key,env_key in (('NumTasks','SLURM_NTASKS'),('CPUs/Task','SLURM_CPUS_PER_TASK')):
        values=[value for value in (fields.get(key),environment[env_key]) if value not in (None,'','(null)','N/A')]
        if not values or any(value!='1' for value in values):
            errors.append(f'{key}/{env_key}: expected one, observed {values!r}')
    if environment['SLURM_MEM_PER_NODE'] is not None:
        require('SLURM_MEM_PER_NODE',environment['SLURM_MEM_PER_NODE'],str(spec['cpu_site']['memory_mb']))
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):
        errors.append('Controller ClusterName must be hydra')
    if errors: raise PermissionError('CPU release/report allocation differs:\n  '+'\n  '.join(errors))
    return job


def setup(spec):
    root=Path(spec['root'])/'setup'; root.mkdir(exist_ok=False); started=time.perf_counter()
    inputs.donor.Bundle(spec['bundle_root']).verify_all()
    inputs.donor.delta(spec).verify_all(); inputs.donor.encoder_delta(spec).verify_all()
    train=inputs.load_hlt(spec,'train'); val=inputs.load_hlt(spec,'validation')
    targets=inputs.targets(spec,train); bank=inputs.bank(spec,train)
    write_immutable_json(root/'targets.json',targets.metadata)
    resources.measure(spec,train,val,targets,bank,time.perf_counter()-started)


def run(spec,task):
    if task not in (*TASKS,'setup','release','report'): raise ValueError('Unregistered task')
    parent=campaign.check(spec)
    if task in ('release','report'): job=cpu_allocation(spec)
    else:
        require_backend(); job=execution.allocation(parent)
        if gpu_identity()!=spec['qualification']['acceptance']['gpu']: raise PermissionError('Parent GPU identity differs')
    if installed_environment()!=spec['reference']['environment']: raise PermissionError('Installed environment differs')
    stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact job')
    if task=='release': return submission.finish(spec)
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    if task=='setup': setup(spec)
    else:
        acceptance=completed_setup(spec)
        if task!='report' and acceptance['gpu']!=gpu_identity(): raise PermissionError('Measured GPU differs')
        root=Path(spec['root'])/task; root.mkdir(exist_ok=False)
        if task=='report':
            from .reporting import results
            write_immutable_json(root/'results.json',results(spec))
        else:
            train=inputs.load_hlt(spec,'train'); val=inputs.load_hlt(spec,'validation'); bank=inputs.bank(spec,train)
            targets=inputs.targets(spec,train) if ARMS[task]['depths'] else None
            training.fit(root/'fit',spec,task,train,val,bank,targets=targets)
    campaign.check(spec); root=Path(spec['root'])/task
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()])
    write_immutable_json(root/'receipt.json',value)
    return value
