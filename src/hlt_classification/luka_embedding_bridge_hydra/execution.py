"""The reviewed two-Pascal profile, with separate release/report CPU requests."""
from hlt_classification.luka_subclass_ddp_hydra.execution import configuration,check_devices,allocation
import os
import re
import subprocess


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
