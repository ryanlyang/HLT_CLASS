"""Twenty exact objectives, ordered debug DAG, reusable banks and installed native workers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_u100_pretrain import cache
from test_luka_embedding_aux import tiny
from test_luka_trial_hybrid import bare_spec as parent_spec, science_fixture as parent_fixture
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_trial_hybrid import contracts as old, worker as old_worker, banks as old_banks
from hlt_classification.luka_auxiliary_grid import contracts as c, model, training, banks, campaign, source, submission, worker, startup, reference


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def bare(path):
    p = parent_spec(Path('/parent'))
    return dict(root=str(path), project_dir='/project', source_commit='a'*40, source_snapshot=dict(content_hash='b'*64),
        content_hash='c'*64, anchor=p['anchor'], reuse=dict(acceptance=p['acceptance'], teachers={}),
        available_quota_gib=4, **campaign.policy(p))


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'new'; root.mkdir(); spec = bare(root); env = installed_environment()
    spec['anchor']['evidence']['environment'] = env
    spec['reuse']['acceptance']['preflight']['environment'] = env
    spec['counts'] = dict(train=22, validation=22); spec['original50k_rows'] = 22
    tr = {co: cache('train', co) for co in ('OFFLINE', 'U100', 'D066', 'D033', 'D000')}
    val = cache('validation', 'D000')
    for i, t in enumerate(c.TEACHERS):
        folder = tmp_path/t; folder.mkdir(); torch.manual_seed(200+i)
        n = DelphesParticleTransformer(); save_state(folder/'selected.pt', n.state_dict())
        write_immutable_json(folder/'report.json', dict(fixture=True))
        files = dict(checkpoint=file_ref(folder/'selected.pt'), report=file_ref(folder/'report.json'))
        spec['reuse']['teachers'][t] = old.artifact('TEACHER', parents=dict(spec='f'*64,
            prepared=spec['anchor']['evidence']['prepared'], **{k: r['sha256'] for k, r in files.items()}),
            scope='shared' if t == 'OFFLINE' else 'c50', teacher=t,
            coordinate=dict(OFFLINE='OFFLINE', UE100='U100', DE066='D066', DE033='D033')[t], files=files)
    references = tmp_path/'references'; references.mkdir(); spec['reuse']['references'] = {}
    uniform = np.full((len(val), 11), 1/11, np.float32)
    for name in c.REFERENCES[:-1]:
        path = references/(name+'.npz')
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(identities=val.identities, labels=val.labels, probabilities=uniform)))
        spec['reuse']['references'][name] = file_ref(path)
    atomic_publish_bytes(references/'subset.npz', deterministic_npz_bytes(dict(indices=np.arange(len(val), dtype=np.int64))))
    spec['reuse']['subset'] = file_ref(references/'subset.npz')
    reference_model = DelphesParticleTransformer()
    with torch.no_grad():
        for parameter in reference_model.mod.fc.parameters(): parameter.zero_()
    save_state(references/'k5.pt', reference_model.state_dict())
    spec['reference'] = dict(content_hash='9'*64, spec=dict(path='/old/campaign_spec.json', sha256='8'*64),
        predictions=spec['reuse']['references']['M0HLT'],checkpoint=file_ref(references/'k5.pt'),
        validation=worker.evaluate_probabilities(val.labels,uniform),
        replay=dict(mode='completed',accounting=dict(k5=dict(state='COMPLETED',exit_code='0:0'))))
    spec = with_content_hash(spec); write_immutable_json(root/'campaign_spec.json', spec)
    jobs = {t['task_id']: str(101+i) for i, t in enumerate(c.tasks())}
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(campaign, 'validate_spec', lambda *a, **kw: None)
    monkeypatch.setattr(source, 'current', lambda *a: spec['source_snapshot'])
    monkeypatch.setattr(worker, 'execution', lambda _, t: jobs[t])
    monkeypatch.setattr(worker, 'storage', lambda _: 0)
    monkeypatch.setattr(worker.regular_cache, 'metadata', lambda _: None)
    monkeypatch.setattr(worker, 'data', lambda _, r, co, meta: tr[co] if r == 'train' else val)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: jobs)
    monkeypatch.setattr(submission, 'successful', lambda *a: None)
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(peak_rss_bytes=2**30, gpu_peak_bytes=2**30))
    monkeypatch.setattr(startup, 'memory_snapshot', worker.memory_snapshot)
    # Tiny population only: production sampler is separately checked on >=187/class.
    class TinyAnchors:
        def __init__(self, labels): self.rows=np.arange(len(labels),dtype=np.int64)
        def sample(self, *a): return self.rows
    monkeypatch.setattr(training, 'AnchorSampler', TinyAnchors)
    monkeypatch.setattr(startup, 'AnchorSampler', TinyAnchors)
    monkeypatch.setattr(reference, 'acquire', lambda s: c.artifact('ACQUISITION',
        parents=dict(spec=s['content_hash'],reference=s['reference']['content_hash']),
        mode='export_after_authenticated_retirement',source_retired=True,
        source_spec=s['reference']['spec'],source_banks={t:'f'*64 for t in c.TEACHERS}))
    return spec, tr, val


def tensors():
    x = torch.randn(4, 11, requires_grad=True); labels = torch.arange(4)
    p = {t: torch.randn(4, 11, requires_grad=True) for t in c.TEACHERS}
    z = {t: torch.randn(4, 128, requires_grad=True) for t in c.TEACHERS}
    q = {t: torch.softmax(torch.randn(4, 11)/2, -1).requires_grad_(True) for t in c.TEACHERS}
    targets = {t: torch.randn(4, 128, requires_grad=True) for t in c.TEACHERS}
    return x, labels, p, z, q, targets


@pytest.mark.parametrize('arm', c.WEIGHTS)
def test_exact_objective_main_gradient_detached_targets_and_zero_control(arm):
    x,y,p,z,q,targets=tensors(); r=c.recipe(arm)
    anchors={t:torch.randn(7,128,requires_grad=True) for t in c.TEACHERS}
    valid=torch.ones(4,7,dtype=torch.bool); valid[0,0]=False
    kw=dict(beta=.7,anchors=anchors,valid=valid) if r['relation']['enabled'] else {}
    loss,ce,kd,embed,rel=training.objective(x,y,p,z,q,targets,arm=arm,**kw)
    expected=.5*F.cross_entropy(x,y)+sum(r['embedding_weights'][t]*F.smooth_l1_loss(z[t],targets[t].detach()) for t in c.TEACHERS)
    expected+=sum(r['logit_weights'][t]*4*F.kl_div((p[t]/2).log_softmax(-1),q[t].detach(),reduction='batchmean') for t in c.TEACHERS)
    if kw: expected+=.7*sum(training.reference_kl(z[t],targets[t],anchors[t],valid) for t in c.TEACHERS)/4
    torch.testing.assert_close(loss,expected,rtol=1e-7,atol=1e-7)
    actual=torch.autograd.grad(loss,x,retain_graph=True)[0]
    torch.testing.assert_close(actual,torch.autograd.grad(.5*ce,x,retain_graph=True)[0],rtol=0,atol=0)
    loss.backward()
    assert all(v.grad is None for v in [*q.values(),*targets.values(),*anchors.values()])
    assert all(v.grad is not None and v.grad.abs().sum()>0 for v in [*z.values(),*p.values()])

@pytest.mark.parametrize('bad', ['nan', 'negative', 'normalization', 'inventory', 'shape'])
def test_bad_targets_rejected(bad):
    x, y, p, z, q, targets = tensors()
    if bad == 'nan': targets['OFFLINE'] = torch.full((4, 128), float('nan'))
    if bad == 'negative': q['DE033'] = -q['DE033']
    if bad == 'normalization': q['DE066'] = q['DE066']*2
    if bad == 'inventory': del q['UE100']
    if bad == 'shape': z['DE033'] = torch.zeros(4, 127)
    with pytest.raises(ValueError): training.objective(x, y, p, z, q, targets, arm='k5_e10_plain')


def test_graph_weights_seeds_resource_allowance_and_native_nodes():
    spec=bare(Path('/new')); plan=submission.render(spec)
    assert (plan['jobs'],plan['scientific_fits'],plan['preflight_jobs'])==(22,20,0)
    assert spec['startup_arms']==list(c.PROBES) and spec['technical_epochs']==0
    assert [r['task_id'] for r in plan['commands']]==['targets',*c.ARMS,'report']
    assert c.WAVES==((5.,10.),(5.,20.),(2.5,10.),(2.5,20.),(1.,10.),(1.,20.),
                     (5.,40.),(2.5,40.),(1.,40.),(10.,20.))
    previous=[]
    for wave,(k,e) in enumerate(c.WAVES):
        pair=[c.arm_name(k,e,rel) for rel in (True,False)]
        for a in pair:
            r=c.recipe(a); n=spec['nodes'][a]
            assert (r['ce_coefficient'],r['logit_total'],r['embedding_total'])==(.5,k,e)
            assert sum(r['logit_weights'].values())==k and sum(r['embedding_weights'].values())==e
            assert n['teacher'] is None and n['coordinate']=='D000' and n['deployable']
            assert r['embedding_seeds']==old.recipe('c50','DE000')['predictor_seeds']
            assert next(x for x in plan['commands'] if x['task_id']==a)['dependencies']==['targets',*previous]
        previous=pair
    assert plan['commands'][-1]['dependencies']==['targets',*c.ARMS]
    assert all('--partition=debug' in x['command'] for x in plan['commands'])
    assert all('--kill-on-invalid-dep=yes' in x['command'] for x in plan['commands'])
    assert not any(x.startswith('--gres=') for x in plan['commands'][-1]['command'])
    bad=parent_spec(Path('/old')); bad['task_minutes']['c50_DE000']=1441
    with pytest.raises(ValueError,match='24h'): campaign.policy(bad)
    with pytest.raises(ValueError): c.recipe('unregistered')

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bf16', [False, True])
def test_full_native_architecture_parity_gradient_isolation_and_all_arm_rng(device, bf16):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    spec = bare(Path('/new')); raw = cache().batch(np.arange(4))
    torch.manual_seed(spec['nodes']['k5_e10_plain']['initialization_seed']); native = DelphesParticleTransformer().to(device).eval()
    reference_rng = torch.random.get_rng_state(); previous = None
    for arm in c.WEIGHTS:
        student = model.new_model(spec['nodes'][arm]).to(device).eval(); student.capture = True
        assert torch.equal(torch.random.get_rng_state(), reference_rng)
        assert all(torch.equal(v, native.state_dict()[k].cpu()) for k, v in student.native_state().items())
        state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if previous is not None: assert all(torch.equal(v, previous[k]) for k, v in state.items())
        previous = state
        with parity_backend(device):
            x = _forward(native, raw, device, bf16=bf16); actual = _forward(student, raw, device, bf16=bf16)
            torch.testing.assert_close(x, actual, rtol=0, atol=0)
            assert all(v.dtype == torch.float32 for v in student.auxiliary_logits.values())
            aux = sum(v.square().mean() for v in [*student.auxiliary_logits.values(), *student.predictions.values()])
            aux.backward()
            assert all(p.grad is None for p in student.mod.fc.parameters())
            assert all(p.grad is not None for p in student.logit_heads.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in student.named_parameters() if n.startswith('mod.'))
            student.zero_grad(set_to_none=True)
            actual = _forward(student, raw, device, bf16=bf16); F.cross_entropy(actual.float(), torch.from_numpy(raw['labels']).to(device)).backward()
            assert all(p.grad is None for p in [*student.predictors.parameters(), *student.logit_heads.parameters()])
        assert not any(k.startswith(('predictors.', 'logit_heads.')) for k in student.native_state())


def test_native_targets_normalization_exact_joins_and_no_wrong_branch(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch)
    worker.run(spec, 'targets', device='cpu')
    bundle = banks.Bundle(spec, tr['D000']); banks.validate_bundle(bundle.manifest, spec)
    for teacher in c.TEACHERS:
        record = banks.record(spec, teacher)
        m, a = old_banks.load(banks.directory(spec, teacher), tr['D000'], record)
        np.testing.assert_allclose(a['targets'].mean(0), 0, atol=1e-6)
        assert a['mean'].dtype == a['scale'].dtype == np.float64 and min(a['scale']) >= 1e-3
        n = worker.native(record['files']['checkpoint'], 'cpu')
        np.testing.assert_allclose(a['probabilities'], predict(n, tr[record['coordinate']], device='cpu', temperature=2.), rtol=1e-5, atol=1e-6)
    with pytest.raises(PermissionError): banks.Bundle(spec, val)
    bad = deepcopy(tr['D000']); bad.identities = bad.identities[::-1].copy()
    with pytest.raises(ValueError): banks.Bundle(spec, bad)
    wrong = deepcopy(spec); r = wrong['reuse']['teachers']['DE033']; r['scope'] = 'c25'
    wrong['reuse']['teachers']['DE033'] = with_content_hash(r)
    with pytest.raises(ValueError): banks.record(wrong, 'DE033')


@pytest.mark.parametrize('damage', ['probability', 'target', 'label', 'moment'])
def test_rehashed_bad_banks_fail_semantic_validation(tmp_path, monkeypatch, damage):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch)
    teacher = 'OFFLINE'; record = banks.record(spec, teacher)
    banks.export(spec, teacher, worker.native(record['files']['checkpoint'], 'cpu'), tr['OFFLINE'], device='cpu')
    folder = banks.directory(spec, teacher); a = load_npz_arrays(folder/'targets.npz')
    if damage == 'probability': a['probabilities'][0] *= 2
    if damage == 'target': a['targets'][0, 0] = np.nan
    if damage == 'label': a['labels'][0] = (a['labels'][0]+1) % 11
    if damage == 'moment': a['mean'][0] += 1
    (folder/'targets.npz').write_bytes(deterministic_npz_bytes(a))
    m = load_json(folder/'manifest.json'); m['payload'] = file_ref(folder/'targets.npz', root=folder)
    (folder/'manifest.json').write_text(json.dumps(with_content_hash(m)))
    with pytest.raises(ValueError): old_banks.load(folder, tr['D000'], record)


def test_startup_failure_does_not_publish_targets_or_allow_science(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, _, _ = fixture(tmp_path, monkeypatch)
    def fail(*a, **kw): raise ValueError('injected startup failure')
    monkeypatch.setattr(startup, 'run', fail)
    with pytest.raises(ValueError, match='startup'): worker.run(spec, 'targets', device='cpu')
    assert not (Path(spec['root'])/'targets/receipt.json').exists()
    with pytest.raises(FileNotFoundError): worker.run(spec, 'k5_e10_plain', device='cpu')
    assert not (Path(spec['root'])/'k5_e10_plain').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual installed-Weaver CUDA required')
def test_full_architecture_twenty_fixed30_workers_native_checkpoints_and_retirement(tmp_path, monkeypatch):
    spec, tr, val = fixture(tmp_path, monkeypatch)
    teacher_refs = [r['files']['checkpoint'] for r in spec['reuse']['teachers'].values()]
    worker.run(spec, 'targets')
    assert worker.results(spec)['pending'] == list(c.WEIGHTS)
    startup.check(load_json(Path(spec['root'])/'targets/startup.json'), spec,
                  load_json(Path(spec['root'])/'targets/bundle.json')['content_hash'])
    for arm in c.WEIGHTS:
        worker.run(spec, arm)
        report, arrays = worker.validate_fit(spec, arm)
        assert report['passes'] == 30 and len(report['validation_history']) == 30
        selected = torch.load(Path(spec['root'])/arm/'fit/selected.pt', weights_only=True)
        assert not any(k.startswith(('predictors.', 'logit_heads.')) for k in selected)
    before = worker.results(spec); assert not before['pending'] and len(before['rows']) == 26
    worker.run(spec, 'report')
    assert before == worker.results(spec)
    assert all(not (Path(spec['root'])/p).exists() for p in worker.target_payloads())
    for ref in teacher_refs: validate_file_ref(ref)
    for role in ('validation', 'original50k'):
        assert all(row[role]['rows'] == 22 for row in before['rows'])
    # Retained results need neither old teacher weights nor retired target arrays.
    monkeypatch.setattr(source, 'validate_reuse', lambda *a: (_ for _ in ()).throw(AssertionError('teacher access')))
    assert before == worker.results(spec)
    path = Path(spec['root'])/'k10_e20_plain/fit/validation_predictions.npz'; path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): worker.results(spec)


def test_poor_science_finishes_high_weight_heads_update_and_loss_tampering(tmp_path, monkeypatch):
    tiny(monkeypatch); spec, tr, val = fixture(tmp_path, monkeypatch); worker.run(spec, 'targets', device='cpu')
    bank = banks.Bundle(spec, tr['D000']); reference = model.new_model(spec['nodes']['k5_e10_plain'])
    initial = {k: v.clone() for k, v in reference.logit_heads.state_dict().items()}
    monkeypatch.setattr(training, 'predict', lambda *a, **kw: np.full((22, 11), 1/11, np.float32))
    net, report, state = training.train(tr['D000'], val, bank=bank, node=spec['nodes']['k5_e10_plain'], device='cpu')
    training.validate_report(report, spec, 'k5_e10_plain', bank.manifest['content_hash'])
    training.validate_report(json.loads(json.dumps(report,sort_keys=True)), spec, 'k5_e10_plain', bank.manifest['content_hash'])
    assert report['passes'] == 30 and report['selected_pass'] == 1
    assert any(not torch.equal(initial[k], v) for k, v in net.logit_heads.state_dict().items())
    bad = deepcopy(report); bad['validation_history'][0]['weighted_logit_losses']['OFFLINE'] = .1
    with pytest.raises(ValueError): training.validate_report(with_content_hash(bad), spec, 'k5_e10_plain', bank.manifest['content_hash'])


def test_twenty_two_job_dry_live_exact_dependencies_and_duplicate_refusal(tmp_path,monkeypatch):
    spec=bare(tmp_path); calls=[]
    monkeypatch.setattr(campaign,'check',lambda _:None); monkeypatch.setattr(submission,'check_site',lambda _:None)
    def sbatch(cmd,**kw):
        calls.append(cmd); return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    plan=submission.submit(spec); assert calls==[]
    with pytest.raises(PermissionError): submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    jobs=submission.accepted_jobs(spec)
    assert len(calls)==len(jobs)==22
    assert [next(x for x in cmd if x.startswith('--job-name=')) for cmd in calls]==['--job-name=lukaax_'+r['task_id'] for r in c.tasks()]
    for row,cmd in zip(c.tasks(),calls):
        if row['dependencies']:
            assert '--dependency=afterok:'+':'.join(jobs[t] for t in row['dependencies']) in cmd
    assert not any('${JOB_' in word for cmd in calls for word in cmd)
    with pytest.raises(FileExistsError): submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)

def test_ambiguous_submission_no_retry(tmp_path, monkeypatch):
    spec = bare(tmp_path)
    monkeypatch.setattr(campaign, 'check', lambda _: None); monkeypatch.setattr(submission, 'check_site', lambda _: None)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='ambiguous', stderr=''))
    plan = submission.submit(spec)
    with pytest.raises(ValueError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_parent_replay_stays_identical_to_accepted_strong_import():
    from hlt_classification.luka_auxiliary_logits_strong import source as previous
    assert source.REPLAY.replace('AUXGRID_REUSE=','AUXLOGIT_STRONG_REUSE=')==previous.REPLAY

def test_create_exact_policy_and_reporting_without_old_payload_access(tmp_path,monkeypatch):
    project=tmp_path/'project'; project.mkdir(); parent_root=tmp_path/'parent'; parent_root.mkdir()
    p=parent_spec(parent_root); p['project_dir']=str(tmp_path/'parent_source')
    p['anchor']['evidence'].update(prepared_root=str(tmp_path/'prepared'),raw_container=str(tmp_path/'raw'))
    p=with_content_hash(p); path=parent_root/'campaign_spec.json'; write_immutable_json(path,p)
    reused=c.artifact('REUSE',parents=dict(spec=p['content_hash']),spec=file_ref(path),anchor=p['anchor'])
    imported=c.artifact('REFERENCE',parents=dict(reuse=reused['content_hash'],spec='d'*64),
        spec=dict(sha256='d'*64),root=str(tmp_path/'strong'),project_dir=str(tmp_path/'strong_source'))
    snapshot=dict(content_hash='b'*64)
    monkeypatch.setattr(source,'current',lambda *a:snapshot)
    monkeypatch.setattr(source,'import_parent',lambda *a:(reused,p))
    monkeypatch.setattr(source,'validate_reuse',lambda _:p)
    monkeypatch.setattr(reference,'import_reference',lambda *a:imported)
    monkeypatch.setattr(reference,'validate_reference',lambda *a:None)
    monkeypatch.setattr(campaign,'storage',lambda _:0)
    def create(root,quota=4): return campaign.create(root,path,project,'a'*40,available_quota_gib=quota,reference_path='/old')
    for root,quota,error in ((parent_root/'nested',4,PermissionError),(project,4,PermissionError),
                             (tmp_path/'strong'/'nested',4,PermissionError),(tmp_path/'bad_quota',3,ValueError)):
        with pytest.raises(error): create(root,quota)
    spec=create(tmp_path/'new'); campaign.check(spec)
    with pytest.raises(FileExistsError): create(tmp_path/'new')
    monkeypatch.setattr(source,'validate_reuse',lambda _:(_ for _ in ()).throw(AssertionError('historical access')))
    campaign.validate_spec(spec,imports=False)
    for field,value in (('preflight_jobs',1),('cpus',8),('available_quota_gib',3),('scientific_fits',5)):
        bad=with_content_hash(dict(spec,**{field:value}))
        (tmp_path/'new/campaign_spec.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): campaign.validate_spec(bad,imports=False)

@pytest.mark.parametrize('bad', ['environment', 'source', 'teacher'])
def test_bad_execution_prerequisites_cannot_publish_any_outputs(tmp_path, monkeypatch, bad):
    tiny(monkeypatch); spec, _, _ = fixture(tmp_path, monkeypatch)
    if bad == 'environment':
        monkeypatch.setattr(worker, 'installed_environment', lambda: {})
    else:
        def fail(_): raise ValueError('Invalid '+bad)
        monkeypatch.setattr(campaign, 'check', fail)
    with pytest.raises(ValueError): worker.run(spec, 'targets', device='cpu')
    assert not (Path(spec['root'])/'targets').exists()


def test_retirement_pending_escape_and_partial_journal_fail_closed(tmp_path, monkeypatch):
    root = tmp_path/'new'; root.mkdir(); (root/'report').mkdir(); spec = bare(root)
    result = dict(pending=['k10_e20_plain'])
    with pytest.raises(ValueError): worker.retire(spec, result, '106')
    assert not (root/'report/results.json').exists()
    outside = tmp_path/'protected'; outside.write_bytes(b'keep')
    monkeypatch.setattr(worker, 'target_payloads', lambda: ['../protected'])
    with pytest.raises(PermissionError): worker.retire(spec, dict(pending=[]), '106')
    assert outside.read_bytes() == b'keep'
    assert not (root/'report/retirement.json').exists()
    write_immutable_json(root/'report/retirement.json', dict(partial=True))
    with pytest.raises(ValueError, match='Partial target retirement'): worker.retired_evidence(spec)


def test_source_guard_requires_complete_files_and_no_legacy_runtime_edits(monkeypatch):
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in args else 'M\tsrc/old.py'))
    with pytest.raises(ValueError, match='runtime'): source.current(Path(__file__).resolve().parents[1], 'b'*40)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError, match='Commit'): source.current(Path(__file__).resolve().parents[1], 'b'*40)


def test_exact_ancestor_accounting_rejects_failure_or_wrong_job(monkeypatch):
    spec = bare(Path('/new'))
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(targets='123'))
    monkeypatch.setattr(submission.getpass, 'getuser', lambda: 'ryreu')
    for line in ('', '123|lukaax_targets|ryreu|reu-aisocial|FAILED|1:0', '123|other|ryreu|reu-aisocial|COMPLETED|0:0'):
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=line))
        with pytest.raises(ValueError): submission.successful(spec, 'targets', '123')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='123|lukaax_targets|ryreu|reu-aisocial|COMPLETED|0:0'))
    submission.successful(spec, 'targets', '123')


def test_registered_shells_and_docs_complete():
    project = Path(__file__).resolve().parents[1]
    assert all((project/p).is_file() for p in source.REQUIRED)
    shell = (project/'sbatch/run_luka_auxiliary_grid.sh').read_text()
    assert '${PROJECT_DIR}/sbatch/common.sh' in shell and 'PYTHONNOUSERSITE=1' in shell and '${CONDA_PREFIX}/lib' in shell
    helper = (project/'scripts/queue_luka_auxiliary_grid.sh').read_text()
    assert c.AUTHORIZATION in helper and '</dev/tty' in helper and '--plan-hash' in helper
    assert 'scancel' not in helper and 'scontrol update' not in helper


def test_twenty_unique_new_recipes_and_no_old_contract_reinterpretation():
    from hlt_classification.luka_auxiliary_logits_strong import contracts as previous
    assert len(c.ARMS)==len({c.recipe(a)['content_hash'] for a in c.ARMS})==20
    assert previous.WEIGHTS=={'k5':5.,'k10':10.}
    for a in c.ARMS:
        with pytest.raises(ValueError): previous.recipe(a)
        with pytest.raises(ValueError): previous.validate(c.recipe(a),'RECIPE')
        with pytest.raises(ValueError): c.validate(previous.recipe('k5'),'RECIPE')

def test_grid_rng_and_plain_gradient_scaling_against_original_k1():
    from hlt_classification.luka_auxiliary_logits import model as pm,training as pt
    spec=bare(Path('/new')); original=pm.new_model(dict(spec['nodes']['k5_e10_plain'],arm='k1'))
    for a,config in c.ARMS.items():
        student=model.new_model(spec['nodes'][a])
        assert all(torch.equal(v,original.state_dict()[k]) for k,v in student.state_dict().items())
        x,y,p,z,q,targets=tensors()
        old_loss,*_=pt.objective(x,y,p,z,q,targets,arm='k1')
        new_loss,*_=training.objective(x,y,p,z,q,targets,arm=a)
        parameters=[x,*p.values(),*z.values()]
        before=torch.autograd.grad(old_loss,parameters,retain_graph=True)
        after=torch.autograd.grad(new_loss,parameters)
        for index,(grad,old_grad) in enumerate(zip(after,before)):
            multiplier=config['kd'] if 1<=index<=4 else config['embedding']/5 if index>=5 else 1.
            torch.testing.assert_close(grad,multiplier*old_grad,rtol=2e-6,atol=1e-7)


def test_real_balanced_sampler_and_relation_self_mask_gradient():
    from hlt_classification.luka_multiteacher_trial.training import AnchorSampler, reference_kl
    assert training.AnchorSampler is AnchorSampler and training.reference_kl is reference_kl
    labels=np.tile(np.arange(11,dtype=np.int64),200)
    sampler=AnchorSampler(labels); before=np.random.get_state(); torch_before=torch.get_rng_state()
    for step in range(11):
        rows=sampler.sample(2,step,step)
        assert len(rows)==len(np.unique(rows))==2048
        np.testing.assert_array_equal(rows,sampler.sample(2,step,step))
        counts=np.bincount(labels[rows]); assert set(counts)=={186,187}
        assert set(np.flatnonzero(counts==187))=={(2*step)%11,(2*step+1)%11}
    assert torch.equal(torch_before,torch.get_rng_state())
    after=np.random.get_state(); np.testing.assert_array_equal(before[1],after[1]); assert before[2:]==after[2:]
    with pytest.raises(ValueError,match='187'): AnchorSampler(np.tile(np.arange(11,dtype=np.int64),186))
    x=torch.randn(4,128,requires_grad=True); y=torch.randn(4,128,requires_grad=True)
    a=torch.randn(2048,128,requires_grad=True); mask=torch.ones(4,2048,dtype=torch.bool); mask[0,0]=False
    loss=reference_kl(x,y,a,mask)
    order=torch.randperm(2048)
    torch.testing.assert_close(loss,reference_kl(x,y,a[order],mask[:,order]))
    loss.backward(); assert x.grad is not None and y.grad is None and a.grad is None
    mask[1]=False
    with pytest.raises(ValueError): reference_kl(x,y,a,mask)


@pytest.mark.parametrize('case',['plain_relation','nan_beta','large_beta','missing_anchors','invalid_mask'])
def test_relation_misconfiguration_fails(case):
    args=tensors(); arm='k5_e10_rel'; kw=dict(beta=1.)
    if case=='plain_relation': arm='k5_e10_plain'
    if case=='nan_beta': kw['beta']=float('nan')
    if case=='large_beta': kw['beta']=2.
    if case=='invalid_mask':
        kw.update(anchors={t:torch.randn(8,128) for t in c.TEACHERS},valid=torch.zeros(4,8,dtype=torch.bool))
    with pytest.raises(ValueError): training.objective(*args,arm=arm,**kw)


@pytest.mark.parametrize('retired',[False,True])
def test_original_strong_replay_private_copy_or_registered_retirement(tmp_path,monkeypatch,retired):
    from test_luka_auxiliary_logits_strong import fixture as strong_fixture
    from hlt_classification.luka_auxiliary_logits_strong import contracts as pc, worker as pw
    tiny(monkeypatch); original=tmp_path/'old'; original.mkdir()
    prior,tr,val=strong_fixture(original,monkeypatch)
    parent_path=original/'parent.json'; write_immutable_json(parent_path,dict(parent=True))
    prior['reuse']['spec']=file_ref(parent_path)
    prior['reuse']['anchor']=prior['anchor']
    prior=pc.artifact('SPEC',**{k:v for k,v in prior.items() if k!='content_hash'})
    oldpath=Path(prior['root'])/'campaign_spec.json'; oldpath.write_text(json.dumps(prior))
    pw.run(prior,'targets',device='cpu'); pw.run(prior,'k5',device='cpu')
    if retired:
        pw.run(prior,'k10',device='cpu'); pw.run(prior,'report',device='cpu')
    spec=bare(tmp_path/'grid'); Path(spec['root']).mkdir()
    spec['reuse']=c.artifact('REUSE',**deepcopy(prior['reuse'])); spec['anchor']=prior['anchor']; spec['counts']=prior['counts']
    monkeypatch.setattr(reference,'validate_source_snapshot',lambda *a,**kw:None)
    def replay(path,project,program,marker):
        monkeypatch.setattr(sys,'argv',['replay',str(path)])
        output=io.StringIO()
        with redirect_stdout(output): exec(program,{})
        return file_ref(path),json.loads(output.getvalue().split(marker)[1])
    monkeypatch.setattr(reference,'replay',replay)
    real_run=subprocess.run
    def accounting_run(args,**kwargs):
        if args[0]=='sacct':
            job=args[args.index('-j')+1]; task={'101':'targets','102':'k5'}[job]
            import getpass
            return SimpleNamespace(stdout=f'{job}|lukaas_{task}|{getpass.getuser()}|{prior["execution_site"]["account"]}|COMPLETED|0:0\n')
        return real_run(args,**kwargs)
    monkeypatch.setattr(subprocess,'run',accounting_run)
    spec['reference']=reference.import_reference(oldpath,spec['reuse']); spec=with_content_hash(spec)
    acquired=reference.acquire(spec); reference.check_acquisition(acquired,spec)
    assert acquired['source_retired']==retired
    if retired:
        assert acquired['mode']=='export_after_authenticated_retirement'
        assert not (Path(spec['root'])/'targets').exists()
    else:
        assert acquired['mode']=='verified_private_copy'
        bundle=banks.Bundle(spec,tr['D000']); banks.validate_bundle(bundle.manifest,spec)
        for t in c.TEACHERS:
            original=Path(prior['root'])/'targets/banks'/t/'targets.npz'
            copied=banks.directory(spec,t)/'targets.npz'
            assert copied.read_bytes()==original.read_bytes()
            assert not copied.is_symlink() and not original.samefile(copied)
        # A corrupted source must not silently fall back to regeneration.
        original.write_bytes(original.read_bytes()+b'corrupt')
        with pytest.raises(ValueError): reference.acquire(spec)
        # Independent copies still authenticate; no input/model array was changed.
        banks.validate_bundle(bundle.manifest,spec)
    wrong=deepcopy(spec['reuse']); wrong['teachers']['DE033']['scope']='c25'
    with pytest.raises(ValueError,match='Wrong strong'): reference.inspect(oldpath,wrong)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual installed-Weaver CUDA required')
def test_real_physical128_startup_all_corners_with_2048_anchor_sampler(monkeypatch):
    from test_luka_attention import cache as sized_cache
    tr=sized_cache('train','D000',rows=2068); val=cache('validation','D000')
    rng=np.random.default_rng(17); spec=bare(Path('/new'))
    bank=SimpleNamespace(labels=tr.labels,manifest=dict(content_hash='f'*64),
        probabilities={t:np.full((len(tr),11),1/11,np.float32) for t in c.TEACHERS},
        targets={t:rng.normal(size=(len(tr),128)).astype(np.float32) for t in c.TEACHERS})
    monkeypatch.setattr(startup,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    value=startup.run(spec,tr,val,bank,'cuda'); startup.check(value,spec,bank.manifest['content_hash'])
    assert value['batch_size']==128 and value['arms']==list(c.PROBES)


def test_shell_and_document_blocks_syntax_and_links():
    import re
    project=Path(__file__).resolve().parents[1]; bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        import shutil
        bash=shutil.which('bash')
    if not bash: pytest.skip('Bash unavailable')
    for path in ('scripts/queue_luka_auxiliary_grid.sh','sbatch/run_luka_auxiliary_grid.sh'):
        subprocess.run([str(bash),'-n',path],cwd=project,check=True,capture_output=True,timeout=30)
    for path in source.REQUIRED:
        text=(project/path).read_text()
        assert all(line==line.rstrip() for line in text.splitlines()),path
        assert not text.endswith('\n\n'),path
        if path.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/path).parent.joinpath(link).is_file(),(path,link)
            for block in re.findall(r'^```bash\n(.*?)^```',text,re.M|re.S):
                subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
