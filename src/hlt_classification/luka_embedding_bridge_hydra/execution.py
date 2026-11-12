"""TP/setup on two A6000s; all other fits on two Pascal GPUs, always FP32."""
from hlt_classification.luka_subclass_ddp_hydra.execution import configuration as pascal_configuration
from hlt_classification.luka_u100_pair_hydra.site import _scheduler_errors,environment
from pathlib import Path
import os
import platform
import re
import subprocess
import sys
import torch

TP_PROFILE='a6000_2_fp32'


def configuration(name='1080ti_2_fp32'):
    if name=='1080ti_2_fp32':return pascal_configuration(name)
    if name!=TP_PROFILE:raise ValueError('Only registered pairs of 1080Ti or RTX A6000 GPUs are supported')
    return dict(gpu_profile=name,world_size=2,precision='fp32',global_batch=128,
        execution_site=dict(cluster='hydra',partition='gpus',account='default',qos='normal',
            gres='gpu:nvidia_rtx_a6000:2',gpu_name='NVIDIA RTX A6000',architecture='x86_64'))


def profile(spec,task):
    from .registry import FITS
    if task not in ('setup','input_probe',*FITS):raise ValueError('Unknown GPU task')
    name=spec['tp_gpu_profile'] if task in ('setup','input_probe','TP') else spec['gpu_profile']
    if name!=(TP_PROFILE if task in ('setup','input_probe','TP') else '1080ti_2_fp32'):
        raise ValueError('Task hardware registration changed')
    return name


def check_devices(devices,name=None):
    if len(devices)!=2:raise ValueError('Two GPUs required')
    if name is None:
        name=TP_PROFILE if devices[0]['name']=='NVIDIA RTX A6000' else '1080ti_2_fp32'
    configuration(name)
    names,capability,minimum=(('NVIDIA RTX A6000',),[8,6],45*2**30) if name==TP_PROFILE else (
        ('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti'),[6,1],10*2**30)
    for row in devices:
        if (row['name'] not in names or row['compute_capability']!=capability
                or type(row['total_memory_bytes'])is not int or row['total_memory_bytes']<minimum):
            raise ValueError('GPU family/capacity differs from '+name)


def allocation(spec,task='setup'):
    name=profile(spec,task);site=configuration(name)['execution_site']
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job):raise PermissionError('Real Slurm GPU allocation required')
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    cluster=re.search(r'(?m)^\s*ClusterName\s*=\s*(\S+)\s*$',config)
    output=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in output.split() if '=' in t)
    errors=_scheduler_errors(spec,job,fields,os.environ,cluster[1] if cluster else None)
    kind=site['gres'].split(':')[1]
    for field in ('ReqTRES','AllocTRES'):
        tres=dict(t.split('=',1) for t in fields.get(field,'').split(',') if '=' in t)
        if tres.get('gres/gpu')!='2' and tres.get('gres/gpu:'+kind)!='2':
            errors.append(field+': exactly two registered GPUs required')
        if any(k.startswith('gres/gpu:') and (k!='gres/gpu:'+kind or v!='2') for k,v in tres.items()):
            errors.append(field+': wrong typed GPU allocation')
        if 'gres/gpu' in tres and tres['gres/gpu']!='2':errors.append(field+': wrong total GPU count')
    if errors:raise PermissionError('Task GPU allocation differs: '+'; '.join(errors))
    if (platform.machine()!='x86_64' or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or os.environ.get('PYTHONNOUSERSITE')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise PermissionError('Use the isolated pinned FP32 environment')
    environment(spec['reference']['environment'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise PermissionError('Exactly two visible GPUs required')
    devices=[dict(name=torch.cuda.get_device_name(i),compute_capability=list(torch.cuda.get_device_capability(i)),
                  total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(2)]
    check_devices(devices,name)
    return devices


def cpu_allocation(spec,task):
    if task not in ('release','report'):raise PermissionError('Unknown CPU task')
    job=os.environ.get('SLURM_JOB_ID','');site=spec['report_site'] if task=='report' else spec['cpu_site']
    if not re.fullmatch('[1-9][0-9]*',job):raise PermissionError('Slurm CPU allocation required')
    output=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in output.split() if '=' in t)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    expected=dict(JobId=job,JobState='RUNNING',Partition=site['partition'],Account=site['account'],QOS=site['qos'],NumNodes='1')
    if any(fields.get(k)!=v for k,v in expected.items()):raise PermissionError('CPU allocation identity/site differs')
    requested=dict(x.split('=',1) for x in fields.get('ReqTRES','').split(',') if '=' in x)
    allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    if requested.get('cpu')!=str(site['cpus']):raise PermissionError('Requested CPU count differs')
    for tres in (requested,allocated):
        if tres.get('mem')!=str(site['memory_mb'])+'M' or tres.get('node')!='1' or any(k.startswith('gres/gpu') for k in tres):
            raise PermissionError('CPU memory/node/GPU resources differ')
    if allocated.get('cpu') not in (str(site['cpus']),str(2*site['cpus'])) or fields.get('NumCPUs')!=allocated.get('cpu'):
        raise PermissionError('Allocated CPU count differs beyond thread rounding')
    for key,env_key,wanted in (('NumTasks','SLURM_NTASKS','1'),('CPUs/Task','SLURM_CPUS_PER_TASK',str(site['cpus']))):
        values=[v for v in (fields.get(key),os.environ.get(env_key)) if v not in (None,'','(null)','N/A')]
        if not values or any(v!=wanted for v in values):raise PermissionError('CPU task layout differs')
    if os.environ.get('SLURM_MEM_PER_NODE') not in (None,str(site['memory_mb'])):raise PermissionError('CPU memory environment differs')
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):raise PermissionError('Hydra controller required')
    return job
