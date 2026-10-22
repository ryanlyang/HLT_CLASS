"""Two-stage scientific semantics, exact dependency workflow and CUDA miniature."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import io
import subprocess
import sys
from contextlib import redirect_stdout
import numpy as np
import pytest
import torch
from test_luka_attention import cache as old_cache, parent_nodes
from test_luka_embedding_aux import tiny
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux import contracts as anchor_contracts
from hlt_classification.luka_d066_embedding import contracts as reference_contracts
from hlt_classification.luka_u100_pretrain import contracts as c, targets, training, worker, source, campaign, submission


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cache(role='train', coordinate='D000'):
    result = old_cache(role, coordinate)
    if role == 'validation':
        result.identities[:, -1] = 1
        for block in result.blocks:
            block.identities[:, -1] = 1
    return result


def node():
    return c.node(dict(nodes=parent_nodes()))


def bank_fixture(root, device='cpu'):
    root.mkdir(exist_ok=True)
    teacher = DelphesParticleTransformer()
    save_state(root/'teacher.pt', teacher.state_dict())
    record = c.artifact('TEACHER', coordinate='U100', target_roles=['train', 'validation'],
        prepared='a'*64, files=dict(checkpoint=file_ref(root/'teacher.pt')))
    targets.export(root/'targets', teacher, cache(coordinate='U100'), cache('validation', 'U100'), record, device)
    train, val = cache(), cache('validation')
    bank = targets.Bank(root/'targets', train, val, record)
    return train, val, bank, record


def spec_fixture(root):
    root.mkdir(exist_ok=True)
    anchor = anchor_contracts.artifact('ANCHOR', spec=dict(path='/parent/spec.json'), evidence=dict(environment={'fixture': True},
        prepared='a'*64, cache='b'*64, prepared_root='/prepared', raw_container='/raw', models={}))
    teacher = c.artifact('TEACHER', parents=dict(anchor=anchor['content_hash']), prepared='a'*64,
        coordinate='U100', target_roles=['train', 'validation'], spec=anchor['spec'],
        node=dict(node_id='COARSE_U100', coordinate='U100', context_coordinate=None),
        files={key: {} for key in ('checkpoint', 'report', 'predictions', 'receipt')})
    reference = reference_contracts.artifact('REFERENCE', parents=dict(anchor=anchor['content_hash']),
        spec=dict(path='/reference/spec.json'), node=node(), files={})
    snapshot = dict(content_hash='c'*64)
    spec = c.artifact('SPEC', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        teacher=teacher['content_hash'], reference=reference['content_hash']), root=str(root),
        project_dir='/project', source_commit='f'*40, source_snapshot=snapshot, anchor=anchor,
        teacher=teacher, reference=reference, available_quota_gib=2,
        **dict(campaign.registration(dict(nodes=parent_nodes())), counts=dict(train=22, validation=22), original50k_rows=22))
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def test_registered_schedule_and_science():
    r = c.recipe()
    assert (r['teacher'], r['pretrain_epochs'], r['finetune_epochs']) == ('COARSE_U100', 20, 30)
    assert (r['pretrain_ce'], r['pretrain_aux'], r['finetune_ce'], r['finetune_aux']) == (0, 1, 1, 5)
    assert not r['logit_kd'] and not r['attention_kd']
    assert node()['coordinate'] == 'D000' and node()['teacher'] is None
    assert c.learning_rate(3, 'pretrain') == pytest.approx(3e-4)
    assert c.learning_rate(20, 'pretrain') == pytest.approx(1.5e-5)
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
    for p in np.linspace(.01, 30, 100):
        assert c.learning_rate(p, 'finetune') == learning_rate(p)
    for p in (0, 21, float('nan')):
        with pytest.raises(ValueError): c.learning_rate(p, 'pretrain')


def test_loss_pretrain_no_ce_and_finetune_weight5():
    pred = torch.randn(4, 128, requires_grad=True); target = torch.randn(4, 128, requires_grad=True)
    loss, ce, aux = training.objective(pred, target, stage='pretrain')
    assert ce is None; torch.testing.assert_close(loss, aux)
    loss.backward(); assert pred.grad is not None and target.grad is None
    logits = torch.randn(4, 11, requires_grad=True); labels = torch.arange(4)
    with pytest.raises(PermissionError): training.objective(pred, target, stage='pretrain', logits=logits, labels=labels)
    loss, ce, aux = training.objective(pred, target, stage='finetune', logits=logits, labels=labels)
    torch.testing.assert_close(loss, ce+5*aux)
    loss.backward(); assert logits.grad is not None and target.grad is None
    with pytest.raises(ValueError): training.objective(pred*float('nan'), target, stage='pretrain')


def test_train_only_normalization_and_role_separation(tmp_path, monkeypatch):
    tiny(monkeypatch)
    tr, val, bank, teacher = bank_fixture(tmp_path)
    assert bank.for_cache(tr).shape == bank.for_cache(val).shape == (22, 128)
    np.testing.assert_allclose(bank.for_cache(tr).mean(0), 0, atol=1e-6)
    # Strongly shift validation raw vectors: TRAIN moments must not change.
    original = targets.extract
    monkeypatch.setattr(targets, 'extract', lambda t, x, d: original(t, x, d)+(100 if x.role == 'validation' else 0))
    model = DelphesParticleTransformer(); model.load_state_dict(torch.load(tmp_path/'teacher.pt', weights_only=True))
    targets.export(tmp_path/'shift', model, cache(coordinate='U100'), cache('validation', 'U100'), teacher, 'cpu')
    old = load_npz_arrays(tmp_path/'targets/normalization.npz'); new = load_npz_arrays(tmp_path/'shift/normalization.npz')
    for name in old: np.testing.assert_array_equal(old[name], new[name])
    changed = deepcopy(tr); changed.identities = changed.identities[::-1].copy()
    with pytest.raises(ValueError): bank.for_cache(changed)
    changed = deepcopy(val); changed.role = 'final_test'
    with pytest.raises(PermissionError): bank.for_cache(changed)
    with pytest.raises(PermissionError): bank.for_cache(cache(coordinate='U100'))
    overlapping = cache('validation'); overlapping.identities = tr.identities.copy()
    with pytest.raises(ValueError, match='overlap'): targets.Bank(tmp_path/'targets', tr, overlapping, teacher)
    (tmp_path/'targets/validation.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets', tr, val, teacher)


@pytest.mark.parametrize('field,value', [('coordinate','OFFLINE'), ('normalization_fit_role','validation'), ('width',127)])
def test_resigned_bank_semantics_rejected(tmp_path, monkeypatch, field, value):
    tiny(monkeypatch); tr, val, _, teacher = bank_fixture(tmp_path)
    path = tmp_path/'targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path), **{field: value}))))
    with pytest.raises(ValueError): targets.Bank(tmp_path/'targets', tr, val, teacher)


def test_pretrain_updates_backbone_head_not_classifier_and_selects_best(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, val, bank, _ = bank_fixture(tmp_path)
    model = new_model(node()); initial = training.full_state(model)
    # Select an early epoch, irrespective of later training or class labels.
    losses = [float(x) for x in [4, 2, 1, 1]+[5]*16]
    monkeypatch.setattr(training, 'validation_loss', lambda *a: losses.pop(0))
    monkeypatch.setattr(training, 'predict', lambda *a, **kw: pytest.fail('No classifier validation in stage1'))
    report, state = training.train(model, tr, val, bank, node(), 'pretrain', 'cpu')
    assert report['passes'] == 20 and report['selected_pass'] == 3 and report['classifier_unchanged']
    assert all(torch.equal(v, state[k]) for k, v in initial.items() if k.startswith('mod.fc.'))
    assert any(not torch.equal(v, state[k]) for k, v in initial.items() if k.startswith('predictor.'))
    assert any(not torch.equal(v, state[k]) for k, v in initial.items() if k.startswith('mod.') and not k.startswith('mod.fc.'))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    spec = dict(node=node(), counts=dict(train=22, validation=22))
    training.validate_report(report, spec, bank.manifest['content_hash'], 'pretrain')
    changed = with_content_hash(dict(report, selected_pass=20, selected=report['history'][-1]))
    with pytest.raises(ValueError, match='selected'): training.validate_report(changed, spec, bank.manifest['content_hash'], 'pretrain')


def test_transition_restores_full_state_before_fresh_optimizer(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, val, bank, _ = bank_fixture(tmp_path)
    first = new_model(node())
    _, selected = training.train(first, tr, val, bank, node(), 'pretrain', 'cpu', technical=True)
    actual = torch.optim.AdamW; calls = []
    second = new_model(node())
    def optimizer(*a, **kw):
        assert all(torch.equal(v, second.state_dict()[k].cpu()) for k, v in selected.items())
        opt = actual(*a, **kw); assert not opt.state
        calls.append(opt); return opt
    monkeypatch.setattr(training.torch.optim, 'AdamW', optimizer)
    report, _ = training.train(second, tr, val, bank, node(), 'finetune', 'cpu', initial_state=selected, technical=True)
    assert len(calls) == 1 and report['optimizer_reset'] and report['history'][0]['update'] == 1
    assert all(p.requires_grad for p in second.mod.fc.parameters())
    with pytest.raises(ValueError, match='restore'): training.train(new_model(node()), tr, val, bank, node(), 'finetune', 'cpu')
    native = {k: v for k, v in selected.items() if not k.startswith('predictor.')}
    with pytest.raises(ValueError, match='predictor'): training.train(new_model(node()), tr, val, bank, node(), 'finetune', 'cpu', initial_state=native)


def test_stage2_update_matches_original_weight5_kernel(tmp_path, monkeypatch):
    from hlt_classification.luka_d066_embedding import training as donor
    tiny(monkeypatch); train, val, bank, _ = bank_fixture(tmp_path)
    fresh = new_model(node()); initial = training.full_state(fresh)
    report, state = training.train(fresh, train, val, bank, node(), 'finetune', 'cpu',
                                   initial_state=initial, technical=True)
    # Same synthetic numbers, adapted only to the donor's target contract, to
    # compare the mathematical update rather than substitute a real teacher.
    old_bank = SimpleNamespace(identities=train.identities, labels=train.labels, targets=bank.for_cache(train),
        manifest=reference_contracts.artifact('BANK', role='train', coordinate='D066'))
    old_report, expected = donor.train(new_model(node()), train, val, bank=old_bank,
                                       node=node(), device='cpu', technical=True)
    assert report['selected']['validation'] == old_report['validation']
    for name in expected:
        torch.testing.assert_close(state[name], expected[name], rtol=0, atol=0)


def test_epoch_measurement_does_not_use_low_resolution_clock(tmp_path, monkeypatch):
    tiny(monkeypatch); train, val, bank, _ = bank_fixture(tmp_path)
    monkeypatch.setattr(training.time, 'monotonic', lambda: 1.)
    report, _ = training.train(new_model(node()), train, val, bank, node(), 'pretrain', 'cpu', technical=True)
    assert report['history'][0]['train_seconds'] > 0
    assert report['history'][0]['validation_seconds'] > 0


def test_exact_debug_dag_and_dry_submit(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec = spec_fixture(tmp_path); monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def submit(cmd, **kw):
        calls.append(cmd); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(exact.subprocess, 'run', submit)
    plan = submission.submit(spec)
    assert not calls and plan['jobs'] == 3
    assert [r['dependencies'] for r in plan['commands']] == [[], ['setup'], ['setup', 'pretrain']]
    for r in plan['commands']:
        assert '--partition=debug' in r['command'] and '--mem=90000M' in r['command']
    with pytest.raises(PermissionError): submission.submit(spec, execute=True, plan_hash='wrong', authorization=c.AUTHORIZATION)
    ledger = submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert ledger['jobs'] == dict(setup='101', pretrain='102', finetune='103')
    assert '--dependency=afterok:101' in calls[1]
    assert '--dependency=afterok:101:102' in calls[2]
    assert submission.accepted_jobs(spec) == ledger['jobs']
    assert submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION) == ledger
    assert len(calls) == 3


def test_ambiguous_submit_not_retried(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as exact
    spec = spec_fixture(tmp_path); monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    plan = submission.submit(spec); calls = []
    def lost(*a, **kw): calls.append(1); raise RuntimeError('lost response')
    monkeypatch.setattr(exact.subprocess, 'run', lost)
    with pytest.raises(RuntimeError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1


@pytest.mark.parametrize('key,value', [('peak_rss_bytes',90*2**30), ('gpu_peak_bytes',40*2**30), ('peak_rss_bytes',0)])
def test_resource_failures(tmp_path, key, value):
    spec = spec_fixture(tmp_path)
    m = dict(environment=spec['anchor']['evidence']['environment'], gpu=dict(total_memory_bytes=40*2**30),
        peak_rss_bytes=2**30, gpu_peak_bytes=2**30,
        timings=dict(authentication=10., load_HLT=10., pretrain_train=20., pretrain_validation=10.,
                     finetune_train=20., finetune_validation=10., readback=10.))
    assert set(worker.admission(spec, m)) == {'pretrain', 'finetune'}
    with pytest.raises((ValueError, MemoryError)): worker.admission(spec, dict(m, **{key: value}))
    m['timings']['finetune_train'] = 100000
    with pytest.raises(ValueError, match='walltime'): worker.admission(spec, m)


def test_science_refuses_failed_setup_before_training(tmp_path, monkeypatch):
    spec = spec_fixture(tmp_path)
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *a, **k: dict(setup='1', pretrain='2', finetune='3'))
    monkeypatch.setattr(worker, 'validate_task', lambda *a: dict(job_id='1', content_hash='a'*64))
    def failed(*a): raise ValueError('setup FAILED')
    monkeypatch.setattr(worker, 'preflight_accounting', failed)
    monkeypatch.setattr(worker, 'fit', lambda *a, **kw: pytest.fail('Must not train'))
    with pytest.raises(ValueError, match='setup FAILED'): worker.run(spec, 'pretrain')
    assert not (tmp_path/'pretrain').exists()


def test_source_rejects_shared_modification_and_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    monkeypatch.setattr(source, '__file__', str(tmp_path/'src/hlt_classification/luka_u100_pretrain/source.py'))
    def git(cmd, **kw):
        return SimpleNamespace(stdout='\n'.join(sorted(source.REQUIRED)) if 'ls-files' in cmd else 'M\tsrc/shared.py\n')
    monkeypatch.setattr(source.subprocess, 'run', git)
    with pytest.raises(ValueError, match='donor source'): source.current(tmp_path, 'f'*40)
    monkeypatch.setattr(source.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError, match='required'): source.current(tmp_path, 'f'*40)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires installed-Weaver CUDA')
def test_real_cuda_complete_setup_pretrain20_finetune30(tmp_path, monkeypatch):
    tiny(monkeypatch)
    from hlt_classification.jetclass2_delphes import acceptance, model as native_model
    monkeypatch.setattr(acceptance, 'model_config', native_model.model_config)
    spec = spec_fixture(tmp_path/'campaign')
    registered = campaign.registration
    monkeypatch.setattr(campaign, 'registration', lambda p: dict(registered(p),
        counts=dict(train=22, validation=22), original50k_rows=22))
    teacher = DelphesParticleTransformer().to('cuda')
    save_state(tmp_path/'teacher.pt', teacher.state_dict())
    val = cache('validation', 'U100')
    a = dict(identities=val.identities, labels=val.labels,
             probabilities=predict(teacher, val, device='cuda', batch_size=128))
    atomic_publish_bytes(tmp_path/'reference.npz', deterministic_npz_bytes(a))
    spec['teacher']['files'] = dict(checkpoint=file_ref(tmp_path/'teacher.pt'), predictions=file_ref(tmp_path/'reference.npz'),
                                   report=file_ref(tmp_path/'reference.npz'), receipt=file_ref(tmp_path/'reference.npz'))
    spec['reference']['files'] = dict(predictions=file_ref(tmp_path/'reference.npz'))
    spec['anchor']['evidence']['models'] = {n: dict(predictions=file_ref(tmp_path/'reference.npz')) for n in ('M0HLT', 'OFFLINE')}
    spec['anchor']['evidence']['environment'] = worker.installed_environment()
    spec['anchor'] = with_content_hash(spec['anchor'])
    for key in ('teacher', 'reference'):
        spec[key]['parents']['anchor'] = spec['anchor']['content_hash']
        spec[key] = with_content_hash(spec[key])
    for k in ('teacher','reference','anchor'): spec['parents'][k] = spec[k]['content_hash']
    spec = with_content_hash(spec)
    (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(campaign, 'check', lambda _: None)
    monkeypatch.setattr(source, 'current', lambda *a: spec['source_snapshot'])
    jobs = dict(setup='1', pretrain='2', finetune='3')
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *a, **kw: jobs)
    monkeypatch.setattr(worker, 'execution', lambda *a: '1')
    monkeypatch.setattr(worker, 'preflight_accounting', lambda *a: None)
    monkeypatch.setattr(worker, 'data', lambda s, r, co, meta: cache(r, co))
    monkeypatch.setattr(worker.regular_cache, 'metadata', lambda _: ({}, dict(summaries={
        r: dict(D000=dict(maximum=4)) for r in ('train', 'validation')}), {}, {}))
    monkeypatch.setattr(worker.alljets_data, 'comparison_indices', lambda *a: np.arange(22, dtype=np.int64))
    monkeypatch.setattr(worker, 'memory_snapshot', lambda _: dict(rss_bytes=2**30, peak_rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    worker.run(spec, 'setup')
    monkeypatch.setattr(worker, 'execution', lambda *a: '2')
    worker.run(spec, 'pretrain')
    first = load_json(Path(spec['root'])/'pretrain/fit/training_report.json')
    assert first['report']['passes'] == 20 and not first['report']['technical']
    checkpoint = torch.load(first['checkpoint']['path'], weights_only=True)
    assert any(k.startswith('predictor.') for k in checkpoint)
    monkeypatch.setattr(worker, 'execution', lambda *a: '3')
    worker.run(spec, 'finetune')
    second = load_json(Path(spec['root'])/'finetune/fit/training_report.json')
    assert second['report']['passes'] == 30 and not second['report']['technical']
    assert second['initial_checkpoint'] == first['checkpoint']
    native = torch.load(second['checkpoint']['path'], weights_only=True)
    assert not any(k.startswith('predictor.') for k in native)
    with pytest.raises(FileExistsError): worker.run(spec, 'finetune')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Reporting must not load a model'))
    result = worker.results(spec)
    assert len(result['rows']) == 5 and result['rows'][-1]['model'] == 'HLT_U100_PRETRAIN_W5'
    worker.print_results(result)
    (Path(spec['root'])/'pretrain/fit/selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): worker.results(spec)


def test_create_and_recheck_exact_registration(tmp_path, monkeypatch):
    template = spec_fixture(tmp_path/'template')
    parent = dict(nodes=parent_nodes(), project_dir=str(tmp_path/'old_source'), root=str(tmp_path/'old/science'))
    # Real publication and semantic validation, with source replay isolated.
    monkeypatch.setattr(source, 'current', lambda *a: template['source_snapshot'])
    monkeypatch.setattr(source, 'authenticate', lambda *a: (template['anchor'], parent))
    monkeypatch.setattr(source, 'teacher', lambda *a: template['teacher'])
    monkeypatch.setattr(source, 'reference', lambda *a: template['reference'])
    monkeypatch.setattr(source, 'validate_file_ref', lambda *a: None)
    args = (tmp_path/'new', tmp_path/'parent.json', tmp_path/'ref/science/campaign_spec.json',
            tmp_path/'project', 'f'*40, 2)
    spec = campaign.create(*args)
    assert campaign.check(spec) == spec
    assert spec['counts'] == dict(train=299234, validation=74895)
    assert spec['execution_site']['partition'] == 'debug'
    with pytest.raises(FileExistsError): campaign.create(*args)
    with pytest.raises(ValueError, match='space'): campaign.create(tmp_path/'other', *args[1:-1], 1)
    with pytest.raises(PermissionError, match='overlaps'): campaign.create(tmp_path/'project/output', *args[1:])
    changed = deepcopy(spec); changed['node']['auxiliary_weight'] = 10
    with pytest.raises(ValueError, match='Registered'): campaign.validate_spec(with_content_hash(changed))
    monkeypatch.setattr(source, 'current', lambda *a: dict(content_hash='d'*64))
    with pytest.raises(ValueError, match='source'): campaign.check(spec)


@pytest.mark.parametrize('case', ['teacher_coordinate', 'teacher_node', 'teacher_fusion', 'teacher_population',
                                 'teacher_roles', 'teacher_spec', 'reference_parent', 'reference_seed'])
def test_resigned_import_mismatches_rejected(tmp_path, case):
    spec = spec_fixture(tmp_path)
    if case == 'teacher_coordinate': spec['teacher']['coordinate'] = 'D066'
    elif case == 'teacher_node': spec['teacher']['node']['node_id'] = 'FUSION_U100'
    elif case == 'teacher_fusion': spec['teacher']['node']['context_coordinate'] = 'U050'
    elif case == 'teacher_population': spec['teacher']['prepared'] = 'z'*64
    elif case == 'teacher_roles': spec['teacher']['target_roles'] = ['train', 'final_test']
    elif case == 'teacher_spec': spec['teacher']['spec'] = dict(path='/wrong')
    elif case == 'reference_parent': spec['reference']['parents']['anchor'] = 'z'*64
    else: spec['reference']['node']['sampler_seed'] += 1
    for k in ('teacher', 'reference'): spec[k] = with_content_hash(spec[k])
    with pytest.raises(ValueError): source.validate_imports(spec, payloads=False)


@pytest.mark.parametrize('case', ['valid', 'incomplete', 'fusion', 'coordinate', 'population'])
def test_original_source_u100_replay_snippet(tmp_path, monkeypatch, case):
    from hlt_classification.luka_fullsim import alljets_fusion_campaign as old_campaign, alljets_fusion_worker as old_worker
    parent = dict(root=str(tmp_path), project_dir=str(tmp_path/'old_project'), prepared_sha256='a'*64)
    path = tmp_path/'campaign_spec.json'; write_immutable_json(path, parent)
    directory = tmp_path/'train_COARSE_U100'; directory.mkdir()
    for name in ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json'):
        (directory/name).write_bytes(b'fixture')
    n = dict(node_id='COARSE_U100', coordinate='U100', context_coordinate=None)
    if case == 'fusion': n['context_coordinate'] = 'U050'
    elif case == 'coordinate': n['coordinate'] = 'D066'
    monkeypatch.setattr(old_campaign, 'validate_spec', lambda s: None)
    monkeypatch.setattr(old_campaign, 'scoped_nodes', lambda s: [n])
    calls = []
    def completed(s, task):
        calls.append(task); return None if case == 'incomplete' else dict(passed=True)
    monkeypatch.setattr(old_worker, 'completed', completed)
    def replay(path, project, code, marker):
        monkeypatch.setattr(sys, 'argv', ['replay', str(path)])
        output = io.StringIO()
        with redirect_stdout(output): exec(compile(code, '<original-source-U100>', 'exec'), {})
        return file_ref(path), json.loads(output.getvalue().split(marker)[1])
    monkeypatch.setattr(source, 'replay', replay)
    anchor = dict(spec=file_ref(path), content_hash='b'*64,
                  evidence=dict(prepared='c'*64 if case == 'population' else 'a'*64))
    if case != 'valid':
        with pytest.raises(ValueError): source.teacher(anchor)
    else:
        result = source.teacher(anchor)
        c.validate(result, 'TEACHER', dict(anchor=anchor['content_hash']))
        assert calls == ['train_COARSE_U100'] and result['node'] == n
        assert result['files']['checkpoint'] == file_ref(directory/'selected.pt')


def test_poor_embedding_quality_never_truncates_epochs(tmp_path, monkeypatch):
    tiny(monkeypatch); train, val, bank, _ = bank_fixture(tmp_path)
    losses = iter(range(100, 120))
    monkeypatch.setattr(training, 'validation_loss', lambda *a: float(next(losses)))
    record, _ = training.train(new_model(node()), train, val, bank, node(), 'pretrain', 'cpu')
    assert record['passes'] == 20 and record['selected_pass'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires installed-Weaver CUDA')
def test_production_width_depth_installed_weaver_cuda_parity():
    # No tiny-model patch: authoritative installed forward at production config.
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    with parity_backend('cuda'):
        assert installed_parity(cache(), device='cuda')['passed']


def test_scripts_and_required_paths():
    root = Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    script = (root/'sbatch/run_luka_u100_pretrain.sh').read_text()
    for token in ('${PROJECT_DIR}/sbatch/common.sh', 'PYTHONNOUSERSITE=1', '${CONDA_PREFIX}/lib', 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
        assert token in script
    queue = (root/'scripts/queue_luka_u100_pretrain.sh').read_text()
    assert 'scancel' not in queue and 'scontrol update' not in queue and 'PLAN_HASH' in queue
    assert queue.count('</dev/tty') == 3
