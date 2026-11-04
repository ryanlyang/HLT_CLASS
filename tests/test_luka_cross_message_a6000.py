"""A6000 execution profile, numerical qualification and immutable boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from test_luka_cross_message_hydra import compact_native, caches, spec_for
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.luka_cross_message_hydra import campaign, contracts as c, execution as e, submission


def test_profile_changes_resources_only_and_versions_spec():
    a, b = campaign.registration(), campaign.registration('a6000')
    assert b['memory_mb'] == 32000 and b['cpus'] == 6
    assert b['execution_site'] == e.A6000_SITE
    assert b['setup_minutes'] == 120 and b['report_minutes'] == 30
    assert a['memory_mb'] == 120000 and a['setup_minutes'] == 720
    for key in ('recipe', 'arms', 'teachers', 'counts', 'fit_max_minutes', 'storage_budget_bytes'):
        assert a[key] == b[key]
    v2 = c.artifact('SPEC', version=2, **b)
    c.validate(v2, 'SPEC', version=2)
    with pytest.raises(ValueError): c.validate(v2, 'SPEC')
    with pytest.raises(ValueError): campaign.registration('any_gpu')


def test_a6000_all_stages_request_same_site_and_preserve_dag(tmp_path):
    spec = dict(campaign.registration('a6000'), root=str(tmp_path), project_dir='/project', content_hash='a'*64)
    measured = c.artifact('ACCEPTANCE', minutes={t:120 for t in c.TASKS})
    for stage in ('setup', 'envelope', 'science'):
        plan = submission.render(spec, stage, acceptance=measured)
        for row in plan['commands']:
            cmd = row['command']
            assert '--gres=gpu:nvidia_rtx_a6000:1' in cmd
            assert '--mem=32000M' in cmd and '--cpus-per-task=6' in cmd
            assert row['dependencies'] == c.dependencies(row['task_id'])
    assert len(submission.render(spec, 'science', acceptance=measured)['commands']) == 15


def test_scheduler_test_only_and_mixed_gpu_plan_rejected(tmp_path, monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout='ClusterName = hydra\n' if cmd[0]=='scontrol' else 'accepted', stderr='', returncode=0)
    monkeypatch.setattr(e.subprocess, 'run', run)
    spec = dict(campaign.registration('a6000'), root=str(tmp_path), project_dir='/project', content_hash='a'*64)
    plan = submission.render(spec, 'setup')
    e.check_site(plan)
    assert '--test-only' in calls[-1] and calls[-1][-1] == '--wrap=true'
    bad = deepcopy(plan)
    bad['commands'][0]['command'].append('--gres=gpu:1')
    with pytest.raises(ValueError): e.check_site(bad)


def test_allocation_accepts_placeholder_cluster_but_rejects_wrong_gpu_and_ram(monkeypatch):
    spec = dict(campaign.registration('a6000'), schema_version=2, environment_dir=str(Path(sys.prefix).resolve()),
                reference=dict(environment={}))
    fields = ['JobId=123 JobState=RUNNING Account=default Partition=gpus QOS=normal NumNodes=1-1 '
              'NumCPUs=6 NumTasks=1 CPUs/Task=6 AllocTRES=cpu=6,mem=32000M,node=1,gres/gpu=1']
    monkeypatch.setattr(e.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='ClusterName = hydra\n' if cmd[-1]=='config' else fields[0]))
    for key, value in dict(SLURM_JOB_ID='123', SLURM_CLUSTER_NAME='(null)', SLURM_CPUS_PER_TASK='6',
                           SLURM_NTASKS='1', SLURM_MEM_PER_NODE='32000', PYTHONNOUSERSITE='1').items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(e.platform, 'machine', lambda:'x86_64')
    monkeypatch.setattr(e.site, 'environment', lambda _:None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda:1)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda:True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _:(8,6))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _:'NVIDIA RTX A6000')
    assert e.allocation(spec)=='123'
    fields[0] = fields[0].replace('mem=32000M', 'mem=64000M')
    with pytest.raises(PermissionError, match='mem'): e.allocation(spec)
    fields[0] = fields[0].replace('mem=64000M', 'mem=32000M')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _:'NVIDIA L40')
    with pytest.raises(PermissionError, match='A6000'): e.allocation(spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for diagnostic kernels')
def test_real_control_replay_and_probe_authentication(tmp_path, monkeypatch):
    from hlt_classification.context_fusion.worker import save_state
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    from hlt_classification.jetclass2_delphes import model as native, execution as native_execution
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.luka_u100_pair_hydra import precision, precision_campaign
    from hlt_classification.luka_cross_message_hydra import inputs
    tr, va, _, _, _ = caches()
    root = tmp_path/'run'; (root/'setup').mkdir(parents=True)
    spec = dict(campaign.registration('a6000'), content_hash='a'*64, root=str(root), bundle_root='/bundle',
                counts=dict(train=len(tr),validation=len(va)), portability_evidence=dict(content_hash='p'*64),
                reference=dict(environment=native.installed_environment()))
    model = native.DelphesParticleTransformer().cuda().eval()
    checkpoint, saved = tmp_path/'checkpoint.pt', tmp_path/'predictions.npz'
    save_state(checkpoint, {k:v.cpu() for k,v in model.state_dict().items()})
    actual = predict(model, va, device='cuda', batch_size=128)
    atomic_publish_bytes(saved, deterministic_npz_bytes(dict(identities=va.identities,labels=va.labels,probabilities=actual)))
    bundle = SimpleNamespace(path=lambda p:checkpoint if p.endswith('checkpoint.pt') else saved)
    monkeypatch.setattr(inputs, 'Bundle', lambda _:bundle)
    monkeypatch.setattr(precision_campaign.inputs, 'Bundle', lambda _:bundle)
    monkeypatch.setattr(precision_campaign.inputs, 'load_view', lambda *a:va)
    historical = dict(full_validation=dict(settings=precision.backend_settings()))
    monkeypatch.setattr(e.portability, 'validate_evidence', lambda *a:(historical,{}))
    # Actual kernels run on local CUDA; device identity is simulated. This does
    # not constitute real A6000 acceptance, which remains in remote shared setup.
    monkeypatch.setattr(native_execution, 'gpu_identity', lambda:dict(name='NVIDIA RTX A6000',
        compute_capability=[8,6],total_memory_bytes=48*2**30))
    value = e.setup_record(spec, va)
    assert not value['reference_retrained'] and not value['control_inference_reused']
    assert e.validate_setup(spec, value)==value
    assert e.control_predictions(spec).is_file()
    bad = with_content_hash(dict(value, gpu=dict(value['gpu'],name='NVIDIA L40')))
    with pytest.raises(ValueError): e.validate_setup(spec, bad)
    path = root/'setup/control_precision/fixed_batch_predictions.npz'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): e.validate_setup(spec, value)


def test_a6000_reference_path_cannot_reuse_l40_predictions():
    spec = dict(gpu_profile='a6000',root='/new',portability_evidence=dict(refs=dict(predictions=dict(path='/old/l40.npz'))))
    assert e.control_predictions(spec)==Path('/new/setup/control_precision/hydra_validation_predictions.npz')
    spec.pop('gpu_profile')
    assert e.control_predictions(spec)=='/old/l40.npz'


def test_setup_only_dispatch_never_submits_fits_and_finish_resumes(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    from hlt_classification.luka_cross_message_hydra import worker
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=str(8000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess, 'run', run)
    monkeypatch.setattr(campaign, 'check', lambda *a, **kw:None)
    monkeypatch.setattr(submission, 'check_site', lambda *a:None)
    monkeypatch.setattr(submission.resources, 'accepted', lambda s:c.artifact('ACCEPTANCE',minutes={t:120 for t in c.TASKS}))
    monkeypatch.setattr(worker, 'completed_setup', lambda s:None)
    spec = dict(campaign.registration('a6000'),root=str(tmp_path),project_dir='/project',content_hash='a'*64)
    review = submission.review(spec)
    monkeypatch.setattr(submission, 'accounting', lambda _:pytest.fail('Setup-only must not wait'))
    result = submission.dispatch(spec,review['content_hash'],c.AUTHORIZATION,setup_only=True)
    assert result==dict(setup_job='8001',automatic_science=False) and len(calls)==1
    assert not (tmp_path/'submission_science').exists()
    monkeypatch.setattr(submission, 'accounting', lambda _:['COMPLETED','0:0'])
    ledger = submission.finish(spec,review['content_hash'],c.AUTHORIZATION)
    assert len(ledger['jobs'])==15 and len(calls)==16
    assert submission.finish(spec,review['content_hash'],c.AUTHORIZATION)==ledger and len(calls)==16
