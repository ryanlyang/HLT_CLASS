"""Longer validation support and immutable single-preflight recovery regression."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.luka_fullsim import regular_campaign as c, regular_preflight as pre
from hlt_classification.luka_fullsim import regular_recovery as recovery, regular_submission as sub
from test_luka_fullsim_regular import evidence_fixture


def test_original_gate_plan_replays_pinned_builder_exactly(tmp_path, monkeypatch):
    # The original per-job journal must still authenticate after changing the
    # active submitter. Execute the trusted pinned repository donor in isolation.
    repository = Path(__file__).resolve().parents[1]
    code = subprocess.run(['git', '-C', str(repository), 'show',
        recovery.ORIGINAL_COMMIT+':src/hlt_classification/luka_fullsim/regular_submission.py'],
        check=True, capture_output=True, text=True).stdout
    namespace = dict(__package__='hlt_classification.luka_fullsim', __name__='pinned_submission_fixture')
    exec(compile(code, 'pinned_submission_fixture', 'exec'), namespace)
    monkeypatch.setattr(c, 'validate_gate', lambda _: None)
    original = c.make_artifact('GATE', root=str(tmp_path), project_dir='/fixture/project', **c.gate_policy())
    assert namespace['plan'](original) == sub._plan(original)


def test_larger_validation_envelope_accepted_but_missing_role_rejected():
    r, gate, p, manifest = evidence_fixture()
    assert r['stress_maxima']['validation']['U000'] > r['stress_maxima']['train']['U000']
    pre.validate_evidence(r, gate, p, manifest)
    for mutation in ('missing', 'wrong', 'train_validation', 'old_version'):
        bad = deepcopy(r)
        if mutation == 'missing':
            del bad['stress_maxima']['validation']
        elif mutation == 'wrong':
            bad['stress_maxima']['validation']['D000'] -= 1
        elif mutation == 'train_validation':
            bad['measurements'][0]['validation_stress_inference_only'] = False
        else:
            bad['contract'], bad['schema_version'] = 'LUKA_FULLSIM_REGULAR_PREFLIGHT/v1', 1
        with pytest.raises(ValueError):
            pre.validate_evidence(with_content_hash(bad), gate, p, manifest)


def test_validation_stress_longest_without_labels_gradients_or_parameter_changes():
    import torch
    class Cache:
        role = 'validation'
        blocks = [SimpleNamespace(offsets=np.array([0, 3, 40, 48]))]
        def batch(self, indices):
            assert np.array_equal(indices, np.full(256, 1))  # longest is 37, not first/last
            return dict(features=np.ones((256, 17, 37), np.float32), vectors=np.ones((256, 4, 37), np.float32),
                        mask=np.ones((256, 1, 37), bool))  # intentionally no labels/identities
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(11))
            self.calls = 0
        def forward(self, *, features, vectors, mask):
            assert not self.training and not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
            assert features.shape[-1] == 37 and mask.all()
            self.calls += 1
            return self.weight[None, :].expand(256, 11)
    model = Model()
    prior = model.weight.detach().clone()
    pre.validation_stress(model, Cache(), device='cpu')
    assert model.calls == 3 and model.weight.grad is None and torch.equal(model.weight, prior)
    bad = Cache()
    bad.role = 'final_test'
    with pytest.raises(PermissionError):
        pre.validation_stress(model, bad, device='cpu')
    with torch.no_grad():
        model.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        pre.validation_stress(model, Cache(), device='cpu')


def test_source_reuse_keeps_all_cache_and_science_modules_identical(tmp_path):
    from test_luka_fullsim_preflight_v2 import _git, _commit
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    _git(old, 'init')
    paths = recovery.CHANGES | {'src/hlt_classification/luka_fullsim/regular_cache.py',
                               'src/hlt_classification/luka_fullsim/regular_worker.py'}
    for name in paths:
        path = old/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    frozen = '\n'.join(f'{name} = 1' for name in sorted(recovery.FROZEN_DECLARATIONS))+'\n'
    campaign_path = 'src/hlt_classification/luka_fullsim/regular_campaign.py'
    (old/campaign_path).write_text(frozen)
    before = _commit(old)
    _git(tmp_path, 'clone', str(old), str(new))
    for name in recovery.ADDITIONS | recovery.CHANGES:
        path = new/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('new operational implementation')
    (new/campaign_path).write_text(frozen+'# new operational implementation\n')
    active = _commit(new)
    source = dict(project_dir=str(old), source_snapshot=before)
    result = recovery.compatible_source(source, active, new)
    assert result['modules']['src/hlt_classification/luka_fullsim/regular_cache.py']['preparation'] == result['modules']['src/hlt_classification/luka_fullsim/regular_cache.py']['execution']
    (new/campaign_path).write_text(frozen.replace('nodes = 1', 'nodes = 2'))
    active = _commit(new)
    with pytest.raises(ValueError, match='science/population'):
        recovery.compatible_source(source, active, new)
    (new/campaign_path).write_text(frozen)
    active = _commit(new)
    path = new/'src/hlt_classification/luka_fullsim/regular_cache.py'
    path.write_text('changed cached input semantics')
    with pytest.raises(ValueError, match='dirty'):
        recovery.compatible_source(source, active, new)
    active = _commit(new)
    with pytest.raises(ValueError, match='scientific source changed'):
        recovery.compatible_source(source, active, new)


@pytest.fixture
def recovered(tmp_path, monkeypatch):
    oldroot = tmp_path/'original'
    oldroot.mkdir()
    oldsource = dict(content_hash='b'*64, git_commit=recovery.ORIGINAL_COMMIT)
    newsource = dict(content_hash='c'*64, git_commit='c'*40)
    old = c.make_artifact('GATE', parents=dict(source=oldsource['content_hash']), root=str(oldroot), project_dir=str(tmp_path/'old_project'),
        source_commit=recovery.ORIGINAL_COMMIT, source_snapshot=oldsource,
        input_container=str(tmp_path/'raw'), **c.gate_policy())
    write_immutable_json(oldroot/'gate_spec.json', old)
    monkeypatch.setattr(sub, 'check_site', lambda _: None)
    real_validate = c.validate_gate
    monkeypatch.setattr(c, 'validate_gate', lambda g: None if g == old else real_validate(g))
    calls = []
    def scheduler(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(900+len(calls))+';sporc', '')
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    plan = sub.submit(old)
    sub.submit(old, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    (oldroot/'preflight-903.out').write_text('ValueError: Validation support exceeds TRAIN stress; register a new envelope')
    source_reuse = c.make_artifact('SOURCE_REUSE', parents=dict(preparation='b'*64, execution='c'*64))
    monkeypatch.setattr(recovery, 'compatible_source', lambda *a: source_reuse)
    monkeypatch.setattr(c, 'source', lambda *a: newsource)
    p = dict(content_hash='d'*64, foundation_root=str(oldroot/'foundation'), source_snapshot=oldsource,
             audit=dict(content_hash='e'*64), conventions=c.conventions('e'*64))
    manifest = dict(content_hash='f'*64, parents=dict(prepared=p['content_hash'], gate=old['content_hash']))
    monkeypatch.setattr(recovery.cache, 'metadata', lambda _: (manifest, p, {}, {}))
    monkeypatch.setattr(c, 'check_population', lambda *a: None)
    monkeypatch.setattr(c, 'load_foundation', lambda _: (dict(source_snapshot=oldsource, input_container=old['input_container']), {}, {}))
    monkeypatch.setattr(c, 'preflight_accounting', lambda j: dict(job_id=str(j), state='COMPLETED', exit_code='0:0'))
    monkeypatch.setattr(recovery, 'failed_accounting', lambda j: dict(job_id=str(j), state='FAILED', exit_code='1:0'))
    gate = recovery.create(original_gate=oldroot/'gate_spec.json', root=tmp_path/'recovery',
                           project=tmp_path/'new_project', commit='c'*40)
    return gate, old, p, manifest, calls


def test_recovery_only_one_job_then_science_uses_original_preparation(recovered, monkeypatch):
    gate, old, p, manifest, calls = recovered
    oldpath = Path(old['root'])
    oldbytes = {str(f.relative_to(oldpath)): f.read_bytes() for f in oldpath.rglob('*') if f.is_file()}
    c.validate_gate(gate)
    plan = sub.submit(gate)
    assert len(calls) == 3 and plan['jobs'] == 1
    assert plan['commands'][0]['task_id'] == 'preflight' and not plan['commands'][0]['dependencies']
    assert '--mem=90000M' in plan['commands'][0]['command'] and '--time=08:00:00' in plan['commands'][0]['command']
    ledger = sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert ledger['jobs'] == {'preflight':'904'}
    sub.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 4
    assert c.gate_jobs(gate) == dict(foundation='901', prepare='902', preflight='904')
    # Synthetic technical evidence with genuine artifact/file hashing. This
    # exercises the recovered-source admission path, not actual GPU admission.
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    r, _, reference_p, _ = evidence_fixture()
    p['summaries'] = reference_p['summaries']
    r.update(job_id='904', source_snapshot=gate['source_snapshot'], preparation_source_snapshot=p['source_snapshot'],
             environment=dict(mocked=True), parents=dict(gate=gate['content_hash'], prepared=p['content_hash'], cache=manifest['content_hash']))
    for row in r['measurements']:
        row['training'] = with_content_hash(dict(row['training'], foundation_sha256=p['content_hash']))
    preflight = Path(gate['root'])/'preflight'
    preflight.mkdir()
    for record in r['outputs']:
        path = preflight/record['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if record['path'] == 'runtime.json':
            write_immutable_json(path, dict(job_id='904', environment=r['environment'], gpu=r['gpu'],
                                            cublas_workspace_config=':4096:8'))
        else:
            path.write_bytes(b'technical unit-test fixture')
    r['outputs'] = [file_ref(preflight/record['path'], root=preflight) for record in r['outputs']]
    r = with_content_hash(r)
    write_immutable_json(preflight/'preflight.json', r)
    assert c.evidence(gate) == (manifest, p, r)
    spec = c.create_science(gate)
    assert spec['admission']['prepared_root'] == str(oldpath/'prepared')
    assert spec['source_snapshot'] != p['source_snapshot']
    c.validate_spec(spec)
    science_plan = sub.submit(spec)
    assert science_plan['jobs'] == 16
    assert {str(f.relative_to(oldpath)): f.read_bytes() for f in oldpath.rglob('*') if f.is_file()} == oldbytes
    assert not (Path(gate['root'])/'prepared').exists()
    # Corrupting the original terminal log breaks the recovery binding, not silently reused.
    (oldpath/'preflight-903.out').write_text('tampered')
    with pytest.raises(ValueError, match='bindings'):
        c.validate_gate(gate)


def test_recovery_rejects_wrong_parent_and_overlapping_destination(recovered, tmp_path):
    gate, old, p, manifest, calls = recovered
    bad = deepcopy(gate)
    bad['reuse']['cache_sha256'] = '0'*64
    bad = with_content_hash(bad)
    # Canonical immutable gate mismatch must reject before any submission.
    with pytest.raises(ValueError):
        c.validate_gate(bad)
    for root in (old['root'], str(Path(old['root'])/'retry'), str(Path(old['root']).parent)):
        with pytest.raises(PermissionError):
            recovery.check_location(root, old, gate['project_dir'])
    manifest['parents']['gate'] = '0'*64
    with pytest.raises(ValueError, match='preparation lineage'):
        c.validate_gate(gate)


@pytest.mark.parametrize('state,exit_code', [('RUNNING','0:0'), ('PENDING','0:0'), ('COMPLETED','0:0'), ('FAILED','0:0')])
def test_recovery_refuses_live_or_successful_original_job(monkeypatch, state, exit_code):
    monkeypatch.setattr(recovery.subprocess, 'run', lambda args, **kw:
        subprocess.CompletedProcess(args, 0, f'123|{state}|{exit_code}\n', ''))
    with pytest.raises(PermissionError):
        recovery.failed_accounting('123')


def test_expected_failure_accounting_requires_exact_unique_job(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, 'run', lambda args, **kw:
        subprocess.CompletedProcess(args, 0, '124|COMPLETED|0:0\n123|FAILED|1:0\n', ''))
    assert recovery.failed_accounting('123')['state'] == 'FAILED'
    with pytest.raises(ValueError):
        recovery.failed_accounting('123;bad')
