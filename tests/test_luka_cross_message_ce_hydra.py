"""CE-only science, no-target boundaries, publication and independent Hydra jobs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_cross_message_hydra import compact_native
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes import model as native
from hlt_classification.luka_cross_message_hydra import model as old_model, contracts as old_c, training as old_t
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_cross_message_ce_hydra import contracts as c, model as m, training as t
from hlt_classification.luka_cross_message_ce_hydra import campaign, submission, worker, reporting

FULL_CONFIG = native.model_config()


def evidence():
    return dict(content_hash='e'*64,acceptance=dict(content_hash='q'*64,
        minutes={'m32_ce_s0':400,'m16_ce_s0':350},environment=native.installed_environment(),
        gpu=dict(name='NVIDIA RTX A6000',total_memory_bytes=48*2**30,compute_capability=[8,6])))


def fixture_spec(root, rows=22):
    spec = campaign.registration(evidence())
    spec.update(content_hash='s'*64,root=str(root),bundle_root=str(Path(root)/'bundle'),project_dir='/project',environment_dir=sys.prefix,
        qualification=evidence(),imported=dict(node=parent_nodes()[0]),anchor=dict(evidence=dict(prepared='a'*64)),
        counts=dict(train=rows,validation=rows),reference=dict(environment=native.installed_environment()))
    return spec


def timing_report(spec, task, seconds=.3, *, identity='m'):
    return dict(content_hash=identity*64,task=task,technical=True,epochs=1,
        parents=dict(spec=spec['content_hash']),counts=dict(train=267,validation=267),
        validation_history=[dict(train_seconds=seconds,validation_seconds=.3,
            validation=dict(macro_ovr_auc=.01))])


@pytest.mark.parametrize('task',c.TASKS)
def test_exact_matched_initialization_ce_gradient_and_deployment(task):
    imp = dict(node=parent_nodes()[0]); cfg = old_c.settings(c.matched_task(task))
    old = old_model.new_model(imp,cfg); old_rng = torch.get_rng_state().clone()
    new = m.new_model(imp,task)
    assert torch.equal(torch.get_rng_state(),old_rng)
    assert not hasattr(new,'predictor') and new.embedding == 0
    assert set(new.state_dict()) == {k for k in old.state_dict() if not k.startswith('predictor.')}
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in new.state_dict().items())
    raw = cache(coordinate='D000').batch(np.arange(3)); opt = old_t.optimizer(new)
    for _ in range(3):
        values = m.step(new,opt,raw,'cpu',lr=.001)
        assert values[0] == values[5] and (values[1:5] == 0).all()
    assert new.expansion.weight.grad.abs().sum() > 0
    assert new.message.net[-1].weight.grad.abs().sum() > 0
    assert new.mod.blocks[0].attn.in_proj.weight.grad.abs().sum() > 0
    restored = m.deployment(task,new.deployment_state(),'cpu')
    new.eval()
    torch.testing.assert_close(old_t.forward(new,raw,'cpu'),old_t.forward(restored,raw,'cpu'),rtol=0,atol=0)
    assert not any(k.startswith('predictor.') for k in restored.state_dict())


@pytest.mark.parametrize('extra',['u_tokens','u_logits','embedding','assignment','teacher_message'])
def test_training_rejects_privileged_fields(extra):
    model = m.new_model(dict(node=parent_nodes()[0]),c.TASKS[0])
    raw = cache(coordinate='D000').batch(np.arange(3)); raw[extra] = np.zeros(3)
    with pytest.raises(PermissionError): m.step(model,old_t.optimizer(model),raw,'cpu',lr=.001)


def test_ce_objective_matches_direct_cross_entropy():
    model = m.new_model(dict(node=parent_nodes()[0]),c.TASKS[0])
    raw = cache(coordinate='D000').batch(np.arange(3)); opt = old_t.optimizer(model)
    model.train(); state = torch.get_rng_state().clone()
    expected = torch.nn.functional.cross_entropy(old_t.forward(model,raw,'cpu').float(),torch.from_numpy(raw['labels']))
    torch.set_rng_state(state)
    values = m.step(model,opt,raw,'cpu',lr=.001)
    assert values[0] == pytest.approx(expected.item(),abs=0,rel=0)


@pytest.mark.parametrize('task',c.TASKS)
def test_full30_ce_only_publication_and_tamper_rejection(tmp_path,task):
    tr = cache(coordinate='D000'); va = cache(role='validation',coordinate='D000')
    spec = fixture_spec(tmp_path); report = t.fit(tmp_path/task,spec,task,tr,va,device='cpu')
    checked,arrays = t.validate_fit(tmp_path/task,spec,task)
    assert checked == report and report['epochs'] == 30 and report['privileged_supervision'] is False
    assert set(report['parents']) == {'spec','recipe','prepared'}
    assert arrays['probabilities'].shape == (22,11)
    assert report['settings']['embedding'] == 0
    epoch = tmp_path/task/'epoch_01.json'; bad = load_json(epoch); bad['embedding_loss'] = 1
    epoch.write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises(ValueError,match='Epoch publication'): t.validate_fit(tmp_path/task,spec,task)


def test_nonhlt_cache_and_nonfinite_step_fail(tmp_path):
    tr = cache(coordinate='U100'); va = cache(role='validation',coordinate='D000'); spec = fixture_spec(tmp_path)
    with pytest.raises(PermissionError): t.fit(tmp_path/'bad',spec,c.TASKS[0],tr,va,device='cpu')
    model = m.new_model(spec['imported'],c.TASKS[0]); raw = va.batch(np.arange(3)); raw['features'][:] = np.nan
    with pytest.raises(ValueError): m.step(model,old_t.optimizer(model),raw,'cpu',lr=.001)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_full_installed_weaver_cuda_ce_only_and_reload(monkeypatch):
    monkeypatch.setattr(native,'model_config',lambda:deepcopy(FULL_CONFIG))
    tr = cache(coordinate='D000'); model = m.new_model(dict(node=parent_nodes()[0]),c.TASKS[0]).cuda()
    raw = tr.batch(np.arange(3)); opt = old_t.optimizer(model)
    for _ in range(3): assert np.isfinite(m.step(model,opt,raw,'cuda',lr=.001)).all()
    model.eval(); restored = m.deployment(c.TASKS[0],model.deployment_state(),'cuda')
    torch.testing.assert_close(old_t.forward(model,raw,'cuda'),old_t.forward(restored,raw,'cuda'),rtol=0,atol=0)


def test_registration_and_two_independent_jobs(tmp_path):
    spec = fixture_spec(tmp_path); value = submission.plan(spec)
    assert len(value['commands']) == 2 and spec['minutes'] == {c.TASKS[0]:410,c.TASKS[1]:360}
    assert spec['privileged_supervision'] is False and spec['storage_budget_bytes'] == 2*2**30
    assert spec['resource_policy'] == c.resource_policy()
    for row in value['commands']:
        assert row['dependencies'] == []
        assert '--gres=gpu:nvidia_rtx_a6000:1' in row['command']
        assert '--mem=32000M' in row['command'] and '--cpus-per-task=6' in row['command']
        assert not any(x.startswith('--dependency=') for x in row['command'])
        assert row['command'][-1] in c.TASKS
    bad = evidence(); bad['acceptance']['minutes']['m32_ce_s0'] = 2880
    with pytest.raises(ValueError): campaign.registration(bad)


def test_exact_submission_dry_auth_idempotence_and_partial_claim(tmp_path,monkeypatch):
    spec = fixture_spec(tmp_path); monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd); return SimpleNamespace(stdout=str(9100+len(calls)),returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    dry = submission.submit(spec); assert dry['dry_run'] and not calls
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='bad',authorization=c.AUTHORIZATION)
    result = submission.submit(spec,execute=True,plan_hash=submission.plan(spec)['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls) == 2 and result['jobs'] == {c.TASKS[0]:'9101',c.TASKS[1]:'9102'}
    assert submission.accepted_jobs(spec) == result['jobs']
    assert submission.submit(spec,execute=True,plan_hash=submission.plan(spec)['content_hash'],authorization=c.AUTHORIZATION) == result
    assert len(calls) == 2
    other = tmp_path/'partial'; other.mkdir(); s = fixture_spec(other); submission.submit(s)
    (other/'submission/live_claim').mkdir()
    with pytest.raises(RuntimeError,match='active or partial'):
        submission.submit(s,execute=True,plan_hash=submission.plan(s)['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls) == 2


def test_resource_record_checks_time_memory_and_not_quality(tmp_path):
    spec = fixture_spec(tmp_path); gpu = evidence()['acceptance']['gpu']
    report = timing_report(spec,c.TASKS[0]); warmup = timing_report(spec,c.TASKS[0],10.,identity='w')
    mem = dict(gpu_peak_bytes=6*2**30,peak_rss_bytes=14*2**30)
    row = worker.resource_record(spec,c.TASKS[0],report,mem,gpu,10,warmup=warmup)
    assert row['passed'] and row['quality_gate'] is False
    bad = deepcopy(report); bad['validation_history'][0]['train_seconds'] = 1e7
    row = worker.resource_record(spec,c.TASKS[0],bad,mem,gpu,10,warmup=warmup)
    assert not row['passed'] and row['violations'][0]['resource'] == 'minutes'
    row = worker.resource_record(spec,c.TASKS[0],report,dict(gpu_peak_bytes=48*2**30,peak_rss_bytes=32*2**30),gpu,10,warmup=warmup)
    assert {v['resource'] for v in row['violations']} == {'gpu_peak_bytes','peak_rss_bytes'}


@pytest.mark.parametrize('task,projection,limit',[(c.TASKS[0],924,421),(c.TASKS[1],896,425)])
def test_first_process_startup_is_not_projected_per_batch(tmp_path,task,projection,limit):
    spec = fixture_spec(tmp_path); spec['counts'] = campaign.COUNTS; spec['minutes'][task] = limit
    gpu = evidence()['acceptance']['gpu']; mem = dict(gpu_peak_bytes=6*2**30,peak_rss_bytes=14*2**30)
    # Reproduce the reported bounds with synthetic durations. The remote logs
    # establish the projected totals, not individual timing components.
    warm = timing_report(spec,task,seconds=.3)
    cold = timing_report(spec,task,identity='w')
    cold['validation_history'][0]['train_seconds'] = ((projection-.5-17)*3-.3*586)/2338
    wrongly_projected = worker.resource_record(spec,task,cold,mem,gpu,120,warmup=warm)
    assert wrongly_projected['projected_minutes'] == projection and not wrongly_projected['passed']
    measured = worker.resource_record(spec,task,warm,mem,gpu,120,warmup=cold)
    assert measured['passed'] and measured['projected_minutes'] == 310
    assert measured['limits']['minutes'] == limit and measured['safety_multiplier'] == 2
    assert measured['parents']['warmup'] == cold['content_hash']
    # A long actual preparation is still charged once; genuine slow warmed
    # execution above is still rejected. No time estimate is clamped.
    extra = worker.resource_record(spec,task,warm,mem,gpu,3720,warmup=cold)
    assert extra['projected_minutes'] == measured['projected_minutes']+60


def test_resource_versions_policy_and_invalid_timings(tmp_path):
    spec = fixture_spec(tmp_path); task = c.TASKS[0]; report = timing_report(spec,task)
    warmup = timing_report(spec,task,identity='w'); gpu = evidence()['acceptance']['gpu']
    mem = dict(gpu_peak_bytes=6*2**30,peak_rss_bytes=14*2**30)
    assert c.artifact('SPEC')['schema_version'] == c.artifact('RESOURCE_CHECK')['schema_version'] == 2
    assert c.recipe()['schema_version'] == 1
    old = c.artifact('SPEC'); old.update(contract='LUKA_CROSS_MESSAGE_CE_HYDRA_SPEC/v1',schema_version=1)
    with pytest.raises(ValueError): c.validate(with_content_hash(old),'SPEC')
    for seconds in (float('nan'),float('inf'),-1.,0.):
        bad = deepcopy(report); bad['validation_history'][0]['train_seconds'] = seconds
        with pytest.raises(ValueError,match='timing'):
            worker.resource_record(spec,task,bad,mem,gpu,120,warmup=warmup)
    for elapsed in (float('nan'),float('inf'),-1.):
        with pytest.raises(ValueError,match='elapsed'):
            worker.resource_record(spec,task,report,mem,gpu,elapsed,warmup=warmup)
    bad = deepcopy(spec); bad['resource_policy']['safety_multiplier'] = 1
    with pytest.raises(ValueError,match='policy'):
        worker.resource_record(bad,task,report,mem,gpu,120,warmup=warmup)
    warmup['task'] = c.TASKS[1]
    with pytest.raises(ValueError,match='miniatures'):
        worker.resource_record(spec,task,report,mem,gpu,120,warmup=warmup)


def test_parent_authentication_runs_original_source_without_source_swap(tmp_path,monkeypatch):
    root = tmp_path/'parent'; root.mkdir(); (root/'setup').mkdir(); project = tmp_path/'source'; project.mkdir()
    parent = dict(source_commit=c.BASE,gpu_profile='a6000',schema_version=3,counts=campaign.COUNTS,
        execution_site=A6000_SITE,environment_dir=sys.prefix,project_dir=str(project),root=str(root),
        content_hash='p'*64,reference=dict(environment=native.installed_environment()))
    path = root/'campaign_spec.json'; path.write_text(json.dumps(parent))
    for name in ('receipt.json','acceptance.json','portability.json'): (root/'setup'/name).write_text('{}')
    a = evidence()['acceptance']; a['parents'] = dict(spec=parent['content_hash'])
    proof = dict(spec=parent['content_hash'],receipt='r'*64,job_id='144840',acceptance=a)
    def run(cmd,**kw):
        assert cmd[:4] == [sys.executable,'-B','-s','-c']
        assert 'sys.path.insert' in cmd[4] and 'worker.completed_setup' in cmd[4]
        assert cmd[-2:] == [str(path.resolve()),str(project)]
        assert kw['cwd'] == str(project) and kw['env']['PYTHONPATH'] == str(project/'src')
        return SimpleNamespace(returncode=0,stdout='CE_PARENT_PROOF='+json.dumps(proof)+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    found,e = campaign.authenticate_parent(path)
    assert found == parent and e['setup_job'] == '144840' and len(e['files']) == 4
    parent['source_commit'] = 'f'*40; path.write_text(json.dumps(parent))
    with pytest.raises(ValueError): campaign.authenticate_parent(path)


def test_create_binds_parent_and_rejects_drift(tmp_path,monkeypatch):
    roots = {k:tmp_path/k for k in ('parent','project','bundle','delta','encoder','source')}
    for p in roots.values(): p.mkdir()
    parent = dict(content_hash='p'*64,root=str(roots['parent']),bundle_root=str(roots['bundle']),
        delta_root=str(roots['delta']),encoder_root=str(roots['encoder']),project_dir=str(roots['source']),
        environment_dir=sys.prefix,reference=dict(environment=native.installed_environment()),
        imported=dict(node=parent_nodes()[0]),anchor=dict(evidence=dict(prepared='a'*64)))
    path = roots['parent']/'campaign_spec.json'; path.write_text(json.dumps(parent))
    e = evidence()
    monkeypatch.setattr(campaign,'authenticate_parent',lambda p:(parent,e))
    monkeypatch.setattr(campaign,'current',lambda *a:dict(content_hash='s'*64))
    spec = campaign.create(tmp_path/'new',path,roots['project'],'f'*40,2)
    assert spec['qualification'] == e and campaign.check(spec) == parent
    assert spec['schema_version'] == 2
    with pytest.raises(ValueError): campaign.create(tmp_path/'new',path,roots['project'],'f'*40,2)
    with pytest.raises(ValueError): campaign.create(roots['parent']/'bad',path,roots['project'],'f'*40,2)
    bad = deepcopy(spec); bad['recipe']['embedding_weight'] = 5
    with pytest.raises(ValueError): campaign.check(bad)


def test_worker_uses_hlt_only_and_fresh_miniature_and_full_fit(tmp_path,monkeypatch):
    spec = fixture_spec(tmp_path); task = c.TASKS[0]; rawfit = t.fit; real_check = worker.integrated_check
    initial_states = []; original_new = t.new_model
    def observed_new(*args):
        value = original_new(*args)
        initial_states.append({k:v.clone() for k,v in value.state_dict().items()})
        return value
    monkeypatch.setattr(t,'new_model',observed_new)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:spec)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.execution,'allocation',lambda s:'9123')
    monkeypatch.setattr(submission,'accepted_jobs',lambda s:{task:'9123'})
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    calls=[]
    def load(s,role):
        calls.append(role); return cache(role=role,coordinate='D000')
    monkeypatch.setattr(campaign,'load_hlt',load)
    def cpu_fit(*args,**kw):
        kw['device']='cpu'; return rawfit(*args,**kw)
    monkeypatch.setattr(t,'fit',cpu_fit)
    monkeypatch.setattr(worker,'integrated_check',lambda *a:real_check(*a,device='cpu'))
    receipt = worker.run(spec,task)
    assert calls == ['train','validation'] and receipt['job_id'] == '9123'
    assert worker.receipt(spec,task) == receipt
    full = load_json(tmp_path/task/'fit/training_report.json')
    mini = load_json(tmp_path/task/'miniature/training_report.json')
    warmup = load_json(tmp_path/task/'warmup/training_report.json')
    assert full['epochs'] == 30 and mini['epochs'] == warmup['epochs'] == 1
    # Miniature rows are deliberately reordered by multiplicity. Compare the
    # actual starting weights, not the resulting epoch metrics on different orders.
    assert len(initial_states) == 3
    assert all(torch.equal(v,state[k]) for state in initial_states[1:] for k,v in initial_states[0].items())
    assert full['sampler_seed'] == mini['sampler_seed']
    with pytest.raises(FileExistsError): worker.run(spec,task)
    # Warmup provenance is required for a completed v2 resource check.
    (tmp_path/task/'warmup/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): worker.receipt(spec,task)


def test_hlt_loader_rejects_test_and_never_requests_u100(monkeypatch):
    seen=[]; monkeypatch.setattr(campaign.inputs,'load_view',lambda s,r,v:seen.append((r,v)))
    campaign.load_hlt({},'train'); campaign.load_hlt({},'validation')
    assert seen == [('train','D000'),('validation','D000')]
    with pytest.raises(PermissionError): campaign.load_hlt({},'test')


def test_epoch_reader_authenticates_task_and_population(tmp_path):
    spec = fixture_spec(tmp_path); task = c.TASKS[0]; folder = tmp_path/task/'fit'; folder.mkdir(parents=True)
    row = c.artifact('EPOCH',parents=dict(spec=spec['content_hash']),task=task,pass_number=1,
                     validation=dict(rows=22,macro_ovr_auc=.9))
    write_immutable_json(folder/'epoch_01.json',row)
    assert reporting.epochs(tmp_path,task,spec) == {1:.9}
    row['task'] = c.TASKS[1]; (folder/'epoch_01.json').write_text(json.dumps(with_content_hash(row)))
    with pytest.raises(ValueError): reporting.epochs(tmp_path,task,spec)


def test_source_pin_rejects_donor_changes(monkeypatch):
    project = Path(campaign.__file__).resolve().parents[3]
    monkeypatch.setattr(campaign,'_source',lambda *a:dict(content_hash='s'*64))
    changes = ['A\t'+p for p in campaign.RUNTIME]
    def run(cmd,**kw):
        output = ('\n'.join(changes) if 'diff' in cmd else '\n'.join(campaign.REQUIRED) if 'ls-files' in cmd else '')
        return SimpleNamespace(stdout=output,returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    assert campaign.current(project,'f'*40)['content_hash'] == 's'*64
    changes.append('M\tsrc/hlt_classification/luka_cross_message_hydra/model.py')
    with pytest.raises(ValueError,match='preserve all donor'): campaign.current(project,'f'*40)


def test_results_and_progress_are_read_only_and_use_correct_anchors(tmp_path,monkeypatch,capsys):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    root = tmp_path/'run'; root.mkdir(); spec = fixture_spec(root); spec['original50k_rows'] = 22
    parent_root = tmp_path/'parent'; parent_root.mkdir()
    parent = dict(spec,root=str(parent_root),content_hash='p'*64)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:parent)
    tr = cache(coordinate='D000'); va = cache(role='validation',coordinate='D000')
    task = c.TASKS[0]; report = t.fit(root/task/'fit',spec,task,tr,va,device='cpu')
    # Completed task ownership is tested by the real worker test; reporting must
    # still run the actual fit/checksum/metrics validation for this fixture.
    (root/task/'receipt.json').write_text('{}')
    monkeypatch.setattr(worker,'receipt',lambda *a:None)
    bundle = tmp_path/'bundle'; bundle.mkdir(); rng = np.random.default_rng(83)
    metrics = {}
    for name,boost in (('M0HLT',0.),('OFFLINE',.8),('matched',.1)):
        p = rng.uniform(.1,1,(22,11)); p[np.arange(22),va.labels] += boost; p = (p/p.sum(1)[:,None]).astype(np.float32)
        target = bundle/f'reference/{name}/predictions.npz'; target.parent.mkdir(parents=True)
        atomic_publish_bytes(target,deterministic_npz_bytes(dict(identities=va.identities,labels=va.labels,probabilities=p)))
        metrics[name] = t.evaluate_probabilities(va.labels,p)
    subset = bundle/'reference/historical/subset.npz'; subset.parent.mkdir()
    atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(22,dtype=np.int64))))
    replay = parent_root/'setup/control_precision/hydra_validation_predictions.npz'; replay.parent.mkdir(parents=True)
    replay.write_bytes((bundle/'reference/matched/predictions.npz').read_bytes())
    lo = metrics['M0HLT']['macro_ovr_auc']; hi = metrics['OFFLINE']['macro_ovr_auc']
    class FakeBundle:
        def path(self,name): return bundle/name
        def json(self,name):
            if name == 'origin/campaign_spec.json': return dict(recovery_anchors=dict(M0HLT=lo,OFFLINE=hi))
            if name == 'reference/matched/report.json': return dict(validation_history=[dict(pass_number=1,validation=metrics['matched'])])
            raise AssertionError(name)
    monkeypatch.setattr(reporting,'Bundle',lambda *a:FakeBundle())
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = reporting.results(spec); reporting.progress(spec)
    after = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    row = next(r for r in result['rows'] if r['model'] == task and r['role'] == 'validation')
    assert row['recovery'] == pytest.approx(100*(report['validation']['macro_ovr_auc']-lo)/(hi-lo))
    assert c.TASKS[1] in result['pending'] and 'm32_ce_s0' in result['pending']
    assert 'no inference' in capsys.readouterr().out
    # Corruption must fail rather than silently omit a supposedly complete fit.
    (root/task/'fit/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): reporting.results(spec)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_real_full_and_partial_miniature_checks(tmp_path,monkeypatch,device):
    if device == 'cuda': monkeypatch.setattr(native,'model_config',lambda:deepcopy(FULL_CONFIG))
    rows = 275; spec = fixture_spec(tmp_path,rows); task = c.TASKS[1]
    root = tmp_path/task; root.mkdir(); tr = cache(coordinate='D000',rows=rows); va = cache(role='validation',coordinate='D000',rows=rows)
    monkeypatch.setattr(worker,'gpu_identity',lambda:spec['qualification']['acceptance']['gpu'])
    record = worker.integrated_check(spec,task,tr,va,time.perf_counter(),device=device)
    report,_ = t.validate_fit(root/'miniature',spec,task,technical=True)
    warmup,_ = t.validate_fit(root/'warmup',spec,task,technical=True)
    assert record['passed'] and report['counts']['train'] > 128
    assert report['counts']['train'] % 128 != 0
    assert report['counts'] == warmup['counts'] and record['parents']['warmup'] == warmup['content_hash']
    assert record['schema_version'] == 2


def test_runtime_source_scope_is_additive_and_launchers_have_no_parent_mutations():
    assert c.BASE == '199928da991f8a4194dcf42df009130cfcf84118'
    assert all('cross_message_ce_hydra' in p for p in campaign.RUNTIME)
    assert len(old_c.ARMS) == 12 and all(v['embedding'] == 5 for v in old_c.ARMS.values())
    queue = Path('scripts/queue_luka_cross_message_ce_hydra.sh').read_text()
    worker_script = Path('sbatch/run_luka_cross_message_ce_hydra.sh').read_text()
    assert 'scancel' not in queue and 'scontrol update' not in queue and 'nohup setsid' in queue
    assert 'PYTHONNOUSERSITE=1' in worker_script and '${ENV_DIR}/lib' in worker_script
    assert '${PROJECT_DIR}/scripts/luka_cross_message_ce_hydra.py' in worker_script
