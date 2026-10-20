"""Matched weight fixtures, real model tests and fail-closed submission evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny, bank_fixture
from hlt_classification.data.cache_contracts import (with_content_hash, write_immutable_json,
    load_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_aux import contracts as old
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_ladder.contracts import artifact as ladder_artifact
from hlt_classification.luka_embedding_weight import (
    contracts as c, training, worker, campaign, preflight, submission, source)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_gate(root):
    anchor = old.artifact('ANCHOR', spec=dict(path='/parent/campaign_spec.json'), evidence=dict(
        prepared='a'*64, cache='b'*64, prepared_root='/prepared', raw_container='/raw',
        environment=dict(fixture=True), models={}))
    imported = ladder_artifact('EMBEDDING_IMPORT', parents=dict(anchor=anchor['content_hash'], spec='d'*64),
        spec=dict(path='/embedding/science/campaign_spec.json'))
    snapshot = dict(content_hash='c'*64)
    return c.artifact('GATE', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        embedding=imported['content_hash']), root=str(root), project_dir='/project', source_commit='f'*40,
        source_snapshot=snapshot, anchor=anchor, embedding=imported, available_quota_gib=2,
        **campaign.registration(dict(nodes=parent_nodes())))


def small_counts(monkeypatch, rows=22):
    actual = campaign.registration
    monkeypatch.setattr(campaign, 'registration', lambda parent: {
        **actual(parent), 'counts':dict(train=rows, validation=rows), 'original50k_rows':rows})


def test_explicit_three_weights_and_unchanged_old_recipe():
    assert c.WEIGHTS == dict(w075=.75, w15=1.5, w10=10.)
    assert old.recipe()['auxiliary_coefficient'] == .3
    gate = fixture_gate(Path('/run'))
    campaign.validate_gate_record(gate)
    for arm, weight in c.WEIGHTS.items():
        scoped = c.arm_gate(gate, arm)
        r = scoped['recipe']
        assert r['ce_coefficient'] == 1 and r['auxiliary_coefficient'] == weight
        assert not r['logit_kd'] and not r['attention_kd'] and not r['warm_start']
        assert scoped['node']['coordinate'] == 'D000' and scoped['node']['teacher'] is None
        assert scoped['node']['initialization_seed'] == parent_nodes()[0]['initialization_seed']
        assert r['base'] == old.recipe()['base'] and r['epochs'] == 30
    assert gate['execution_site']['partition'] == 'tier3'
    for invalid in (.3, .6, -1, True, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            c.recipe(invalid)
    bad = deepcopy(gate)
    bad['nodes']['w10']['auxiliary_weight'] = .3
    with pytest.raises(ValueError):
        campaign.validate_gate_record(with_content_hash(bad))


@pytest.mark.parametrize('weight', [.75, 1.5, 10.])
def test_loss_gradient_scaling_detached_teacher_and_no_kd(weight):
    torch.manual_seed(37)
    logits = torch.randn(4, 11, requires_grad=True)
    predicted = torch.randn(4, 128, requires_grad=True)
    target = torch.randn(4, 128, requires_grad=True)
    loss, ce, aux = training.objective(logits, torch.arange(4), predicted, target, coefficient=weight)
    torch.testing.assert_close(loss, ce + weight*aux)
    expected = torch.autograd.grad(aux, predicted, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(predicted.grad, weight*expected)
    assert target.grad is None and torch.count_nonzero(logits.grad)
    with pytest.raises(ValueError):
        training.objective(logits, torch.arange(4), predicted, target, coefficient=.3)
    target.data[0, 0] = float('nan')
    with pytest.raises(ValueError):
        training.objective(logits, torch.arange(4), predicted, target, coefficient=weight)


@pytest.mark.parametrize('arm', list(c.WEIGHTS))
def test_full_fixed30_no_kd_checkpoint_and_selection(tmp_path, monkeypatch, arm):
    tiny(monkeypatch); small_counts(monkeypatch)
    train, bank, _ = bank_fixture(tmp_path)
    gate = fixture_gate(tmp_path)
    scoped = c.arm_gate(gate, arm)
    val = cache('validation', coordinate='D000')
    result = worker.fit(tmp_path/arm, scoped, train, val, bank, device='cpu', technical=False)
    report, arrays = worker.validate_fit(tmp_path/arm, scoped, bank.manifest['content_hash'], technical=False)
    assert len(report['validation_history']) == report['passes'] == 30
    assert report['node']['auxiliary_weight'] == c.WEIGHTS[arm]
    for epoch in report['validation_history']:
        assert epoch['weighted_auxiliary_loss'] == c.WEIGHTS[arm]*epoch['auxiliary_loss']
        assert epoch['total_loss'] == epoch['ce_loss']+epoch['weighted_auxiliary_loss']
    assert report['parents']['recipe'] == c.recipe(c.WEIGHTS[arm])['content_hash']
    assert result['checkpoint_probability_roundtrip'] and arrays['probabilities'].shape == (22, 11)
    state = torch.load(tmp_path/arm/'selected.pt', weights_only=True)
    assert not any(k.startswith('predictor.') for k in state)
    with pytest.raises(FileExistsError):
        worker.fit(tmp_path/arm, scoped, train, val, bank, device='cpu', technical=False)
    with pytest.raises(ValueError):
        worker.report_valid(with_content_hash(dict(report, passes=29)), scoped,
                            bank.manifest['content_hash'], technical=False)
    bad = deepcopy(train); bad.identities = train.identities[::-1].copy()
    with pytest.raises(ValueError, match='alignment'):
        training.train(new_model(scoped['node']), bad, val, bank=bank, node=scoped['node'], device='cpu')


def test_all_weights_longest_support_stress(tmp_path, monkeypatch):
    tiny(monkeypatch)
    gate = fixture_gate(tmp_path)
    for arm, weight in c.WEIGHTS.items():
        node = gate['nodes'][arm]
        for backward in (True, False):
            role = 'train' if backward else 'validation'
            row = preflight.stress(cache(role, coordinate='D000'), node, 'cpu', backward=backward)
            assert row['weight'] == weight and row['maximum'] == 4 and row['batch'] == 128


def test_shared_bank_read_only_and_corruption(tmp_path, monkeypatch):
    tiny(monkeypatch)
    train, bank, teacher = bank_fixture(tmp_path)
    gate = fixture_gate(tmp_path)
    gate['embedding'].update(bank=file_ref(tmp_path/'bank/manifest.json'), payload=file_ref(tmp_path/'bank/targets.npz'))
    gate['anchor']['evidence']['models'] = dict(OFFLINE=dict(checkpoint=teacher))
    before = file_ref(tmp_path/'bank/targets.npz')
    for arm in c.WEIGHTS:
        current = worker.load_bank(gate, train)
        np.testing.assert_array_equal(current.targets, bank.targets)
    assert file_ref(tmp_path/'bank/targets.npz') == before
    assert not (tmp_path/'targets').exists()
    (tmp_path/'bank/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        worker.load_bank(gate, train)


def test_source_guard_and_required_files(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source, '_source', lambda *a: dict(content_hash='a'*64))
    def git(cmd, **kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
                               'src/hlt_classification/luka_embedding_aux/training.py\n')
    monkeypatch.setattr(source.subprocess, 'run', git)
    with pytest.raises(ValueError, match='additive'):
        source.current(project, 'a'*40)
    monkeypatch.setattr(source.subprocess, 'run', lambda cmd, **kw: SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else '\n'.join(source.ADDITIONS)))
    assert source.current(project, 'a'*40)['content_hash'] == 'a'*64
    for name in source.REQUIRED:
        assert (project/name).is_file(), name


def test_failed_or_mismatched_gate_accounting_blocks_science(tmp_path, monkeypatch):
    gate = fixture_gate(tmp_path)
    (tmp_path/'preflight').mkdir()
    write_immutable_json(tmp_path/'preflight/receipt.json', dict(job_id='77'))
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(preflight, 'validate_report', lambda *a: None)
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='77'))
    monkeypatch.setattr(campaign, 'preflight_accounting',
                        lambda _: (_ for _ in ()).throw(ValueError('FAILED accounting')))
    with pytest.raises(ValueError, match='accounting'):
        campaign.science(gate)
    assert not (tmp_path/'science').exists()
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='78'))
    with pytest.raises(ValueError, match='journal'):
        campaign.science(gate)


def test_fresh_root_quota_and_pinned_import_guards(tmp_path, monkeypatch):
    template = fixture_gate(tmp_path/'new')
    anchor, imported, snapshot = template['anchor'], template['embedding'], template['source_snapshot']
    anchor['evidence'].update(prepared_root=str(tmp_path/'prepared'), raw_container=str(tmp_path/'raw'))
    anchor = with_content_hash(anchor)
    imported['parents']['anchor'] = anchor['content_hash']
    imported = with_content_hash(imported)
    parent = dict(nodes=parent_nodes(), project_dir=str(tmp_path/'oldsource'), root=str(tmp_path/'old/science'))
    monkeypatch.setattr(source, 'current', lambda *a: snapshot)
    monkeypatch.setattr(source, 'authenticate', lambda *a: (anchor, parent))
    monkeypatch.setattr(source, 'import_embedding', lambda *a: imported)
    args = dict(parent_path='parent', embedding_path=tmp_path/'embed/science/campaign_spec.json',
                project=tmp_path/'newsource', commit='a'*40, available_quota_gib=2)
    for path in ('prepared/child', 'raw/child', 'old/child', 'newsource/child', 'embed/child'):
        with pytest.raises(PermissionError):
            campaign.create(tmp_path/path, **args)
        assert not (tmp_path/path).exists()
    with pytest.raises(ValueError):
        campaign.create(tmp_path/'lowquota', **{**args, 'available_quota_gib':1})
    gate = campaign.create(tmp_path/'new', **args)
    assert campaign.check_gate(gate) == parent
    with pytest.raises(FileExistsError):
        campaign.create(tmp_path/'new', **args)
    monkeypatch.setattr(source, 'current', lambda *a: dict(content_hash='e'*64))
    with pytest.raises(ValueError, match='source'):
        campaign.check_gate(gate)


def test_three_exact_tier3_jobs_after_gate_and_idempotence(tmp_path, monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate = fixture_gate(tmp_path)
    write_immutable_json(tmp_path/'gate_spec.json', gate)
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(campaign, 'check_spec', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    calls = []
    def scheduler(cmd, **kw):
        calls.append(cmd)
        assert '--partition=tier3' in cmd and '--partition=debug' not in cmd
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in kw['env'])
        return SimpleNamespace(stdout=str(700+len(calls)))
    monkeypatch.setattr(exact.subprocess, 'run', scheduler)
    plan = submission.submit(gate)
    assert plan['jobs'] == 1 and not calls
    submission.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(gate) == {'preflight':'701'}
    science_root = tmp_path/'science'; science_root.mkdir()
    spec = c.artifact('SPEC', root=str(science_root), gate_path=str(tmp_path/'gate_spec.json'),
        project_dir='/project', train_minutes={arm:120 for arm in c.WEIGHTS})
    dry = submission.submit(spec)
    assert dry['jobs'] == 3 and len(calls) == 1
    assert [r['task_id'] for r in dry['commands']] == list(c.WEIGHTS)
    assert all(not r['dependencies'] for r in dry['commands'])
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True, plan_hash='bad', authorization=c.AUTHORIZATION)
    submission.submit(spec, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    submission.submit(spec, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 4


def test_old_debug_registration_cannot_be_silently_reused(tmp_path):
    from hlt_classification.jetclass2_delphes.execution import execution_site
    gate = fixture_gate(tmp_path)
    old_debug = with_content_hash(dict(gate, execution_site=execution_site('sporc_a100_debug')))
    assert old_debug['content_hash'] != gate['content_hash']
    assert submission.render(old_debug)['content_hash'] != submission.render(gate)['content_hash']
    with pytest.raises(ValueError, match='resources'):
        campaign.validate_gate_record(old_debug)


def test_tier3_source_base_preserves_existing_scientific_runtime():
    project = Path(__file__).resolve().parents[1]
    changed = subprocess.run(['git', 'diff', '--name-status',
        '5ed50964085e9bf364030764bbbc755179c445ae', source.BASE_COMMIT, '--', 'src'],
        cwd=project, check=True, capture_output=True, text=True).stdout.splitlines()
    assert changed
    assert all(row.startswith('A\tsrc/hlt_classification/luka_fullstat_embedding/') for row in changed)


def test_uncertain_submission_never_resubmits(tmp_path, monkeypatch):
    import hlt_classification.scouting.hcwdl_exact_dag_submission as exact
    gate = fixture_gate(tmp_path)
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(submission, 'check_site', lambda _: None)
    dry = submission.submit(gate)
    calls = []
    def uncertain(*a, **kw):
        calls.append(a)
        raise RuntimeError('lost reply after possible submission')
    monkeypatch.setattr(exact.subprocess, 'run', uncertain)
    with pytest.raises(RuntimeError):
        submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    with pytest.raises((FileExistsError, PermissionError, ValueError)):
        submission.submit(gate, execute=True, plan_hash=dry['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 1


def test_complete_real_cuda_preflight_and_all_science_workers(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Real local CUDA required for production architecture fixture')
    # Production ParT architecture; only populations, scheduler and remote-source fixtures are small.
    small_counts(monkeypatch)
    train, bank, teacher = bank_fixture(tmp_path)
    val = cache('validation', coordinate='D000')
    root = tmp_path/'run'; root.mkdir()
    gate = fixture_gate(root)
    arrays = dict(identities=val.identities, labels=val.labels,
                  probabilities=np.full((len(val), 11), 1/11, np.float32))
    atomic_publish_bytes(tmp_path/'parent_predictions.npz', deterministic_npz_bytes(arrays))
    ref = file_ref(tmp_path/'parent_predictions.npz')
    gate['anchor']['evidence']['models'] = {n: dict(checkpoint=teacher, predictions=ref) for n in ('M0HLT', 'OFFLINE')}
    gate['anchor'] = with_content_hash(gate['anchor'])
    gate['embedding'].update(bank=file_ref(tmp_path/'bank/manifest.json'), payload=file_ref(tmp_path/'bank/targets.npz'),
                             predictions=ref)
    gate['embedding']['parents']['anchor'] = gate['anchor']['content_hash']
    gate['embedding'] = with_content_hash(gate['embedding'])
    gate['parents'].update(anchor=gate['anchor']['content_hash'], embedding=gate['embedding']['content_hash'])
    gate = with_content_hash(gate)
    write_immutable_json(root/'gate_spec.json', gate)
    meta = ({}, dict(summaries={r:dict(D000=dict(maximum=4)) for r in ('train', 'validation')}), {}, {})
    monkeypatch.setattr(campaign, 'check_gate', lambda _: None)
    monkeypatch.setattr(worker, 'execution', lambda *a: '77')
    monkeypatch.setattr(worker, 'data', lambda g, r, co, meta: train if r == 'train' else val)
    monkeypatch.setattr(preflight, 'installed_environment', lambda: gate['anchor']['evidence']['environment'])
    monkeypatch.setattr(source, 'current', lambda *a: gate['source_snapshot'])
    monkeypatch.setattr(preflight.regular_cache, 'metadata', lambda _: meta)
    monkeypatch.setattr(worker.alljets_data, 'comparison_indices', lambda *a: np.arange(len(val), dtype=np.int64))
    def memory(_):
        return dict(rss_bytes=2**30, peak_rss_bytes=2**30, gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight, 'memory_snapshot', memory)
    monkeypatch.setattr(worker, 'memory_snapshot', memory)
    report = preflight.run(gate)
    assert report['technical_checks_passed'] and set(report['train_minutes']) == set(c.WEIGHTS)
    assert not (root/'targets').exists()
    monkeypatch.setattr(submission, 'accepted_jobs', lambda _: dict(preflight='77'))
    monkeypatch.setattr(campaign, 'preflight_accounting', lambda _: None)
    spec = campaign.science(gate)
    assert worker.results(spec)['pending'] == list(c.WEIGHTS)
    for i, arm in enumerate(c.WEIGHTS):
        result = worker.run(spec, arm)
        assert result['weight'] == c.WEIGHTS[arm]
        partial = worker.results(spec)
        assert len(partial['rows']) == 4+i
    assert worker.results(spec)['pending'] == []
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Results must not load any model'))
    monkeypatch.setattr(source, 'authenticate', lambda *a: pytest.fail('Compact result replay only'))
    assert len(worker.results(spec)['rows']) == 6
    bad = deepcopy(report)
    bad['outputs'] = [r for r in bad['outputs'] if r['path'] != 'preflight/w10/receipt.json']
    with pytest.raises(ValueError, match='coverage'):
        preflight.validate_report(gate, with_content_hash(bad))
    bad = with_content_hash(dict(report, technical_checks_passed=False))
    with pytest.raises(ValueError, match='acceptance'):
        preflight.validate_report(gate, bad)
    bad = deepcopy(report); bad['stress']['w10'][0]['maximum'] += 1
    with pytest.raises(ValueError, match='stress'):
        preflight.validate_report(gate, with_content_hash(bad))
    path = root/'science/w10/fit/validation_predictions.npz'
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError):
        worker.results(spec)


def test_shell_syntax_and_required_environment():
    project = Path(__file__).resolve().parents[1]
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    import shutil
    executable = str(bash) if bash.is_file() else shutil.which('bash')
    if executable is None:
        pytest.skip('Bash syntax check unavailable')
    for name in ('scripts/queue_luka_embedding_weight.sh', 'sbatch/run_luka_embedding_weight.sh'):
        subprocess.run([executable, '-n', name], cwd=project, check=True, capture_output=True)
        text = (project/name).read_text()
        for token in ('PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
                      '${CONDA_PREFIX}/lib', 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
            assert token in text
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in (project/'sbatch/run_luka_embedding_weight.sh').read_text()
    text = (project/'docs/LUKA_EMBEDDING_WEIGHT_RUNBOOK.md').read_text()
    import re
    for block in re.findall(r'```bash\n(.*?)\n```', text, re.S):
        subprocess.run([executable, '-n'], input=block, text=True, cwd=project, check=True, capture_output=True)
    for relative in re.findall(r'\]\(([^)]+)\)', text):
        assert (project/'docs'/relative).resolve().is_file()
