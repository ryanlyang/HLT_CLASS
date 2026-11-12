"""TP-only Ampere scheduling and measurement ownership; Pascal students stay put."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.luka_embedding_bridge_hydra import execution,submission,resources,registry
from test_luka_embedding_bridge_hydra import execution_spec,resource_diagnostic


def test_exact_hardware_routes_and_unchanged_training_math(tmp_path):
    s=execution_spec(tmp_path)
    boot=submission.plan(s,'bootstrap');science=submission.plan(s,'envelope')
    assert '--gres=gpu:nvidia_rtx_a6000:2' in boot['commands'][0]['command']
    assert not any(arg.startswith('--gres=') for arg in boot['commands'][1]['command'])
    a6000=[row['task_id'] for row in science['commands'] if '--gres=gpu:nvidia_rtx_a6000:2' in row['command']]
    pascal=[row['task_id'] for row in science['commands'] if '--gres=gpu:geforce_gtx_1080_ti:2' in row['command']]
    assert a6000==['TP'] and set(pascal)==set(registry.FITS)-{'TP'}
    assert (s['precision'],s['world_size'],s['global_batch'])==('fp32',2,128)
    bad=copy.deepcopy(science)
    bad['commands'][0]['command']=[arg.replace('geforce_gtx_1080_ti','nvidia_rtx_a6000') for arg in bad['commands'][0]['command']]
    with pytest.raises(PermissionError):submission.within_envelope(science,bad)


@pytest.mark.parametrize('bad',['tp_on_pascal','student_on_ampere','mixed_pair','unmeasured','small_gpu'])
def test_resource_measurements_cannot_cross_profiles(tmp_path,bad):
    s=execution_spec(tmp_path);v=resource_diagnostic(s)
    accepted=resources.accept(s,v)
    assert accepted['task_devices']==v['task_devices']
    if bad=='tp_on_pascal':v['task_devices']['TP']=v['task_devices']['C0']
    elif bad=='student_on_ampere':v['task_devices']['P5']=v['devices']
    elif bad=='mixed_pair':v['task_devices']['P5']=[v['task_devices']['C0'][0],v['devices'][0]]
    elif bad=='unmeasured':del v['task_devices']['TR']
    else:v['devices'][0]['total_memory_bytes']=10*2**30
    with pytest.raises(ValueError):resources.accept(s,with_content_hash(v))


@pytest.mark.parametrize('task',['setup','TP','C0','P5'])
def test_controller_and_visible_devices_must_match_task(tmp_path,monkeypatch,task):
    s=execution_spec(tmp_path);s['environment_dir']=str(Path(sys.prefix).resolve())
    name=execution.profile(s,task);a6000=name==execution.TP_PROFILE
    kind='nvidia_rtx_a6000' if a6000 else 'geforce_gtx_1080_ti'
    monkeypatch.setenv('SLURM_JOB_ID','165999');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('SLURM_CLUSTER_NAME','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE'):
        monkeypatch.delenv(key,raising=False)
    tres=f'cpu=6,mem=48000M,gres/gpu=2,gres/gpu:{kind}=2'
    fields=dict(JobId='165999',JobState='RUNNING',Account='default',Partition='gpus',QOS='normal',
        NumNodes='1',NumCPUs='6',NumTasks='1',**{'CPUs/Task':'6'},ReqTRES=tres,AllocTRES=tres)
    def controller(cmd,**kwargs):return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[-1]=='config'
        else ' '.join(k+'='+v for k,v in fields.items()))
    monkeypatch.setattr(execution.subprocess,'run',controller)
    monkeypatch.setattr(execution.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(execution,'environment',lambda e:None)
    monkeypatch.setattr(execution.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(execution.torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(execution.torch.cuda,'get_device_name',lambda i:'NVIDIA RTX A6000' if a6000 else 'GeForce GTX 1080 Ti')
    monkeypatch.setattr(execution.torch.cuda,'get_device_capability',lambda i:(8,6) if a6000 else (6,1))
    monkeypatch.setattr(execution.torch.cuda,'get_device_properties',lambda i:SimpleNamespace(total_memory=(48 if a6000 else 11)*2**30))
    assert len(execution.allocation(s,task))==2
    for key in ('ReqTRES','AllocTRES'):
        fields[key]=tres.replace('gres/gpu=2','gres/gpu=1')
        with pytest.raises(PermissionError):execution.allocation(s,task)
        fields[key]=tres.replace(kind,'wrong_card')
        with pytest.raises(PermissionError):execution.allocation(s,task)
        fields[key]=tres
    monkeypatch.setattr(execution.torch.cuda,'get_device_name',lambda i:'NVIDIA L40')
    with pytest.raises(ValueError):execution.allocation(s,task)
