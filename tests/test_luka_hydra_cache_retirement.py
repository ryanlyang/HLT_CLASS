"""Destructive-operation scope, interrupted retirement, and unchanged science DAG."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('retirement_under_test',
    ROOT/'src/hlt_classification/operations/hydra_cache_retirement.py')
o = importlib.util.module_from_spec(loader)
loader.loader.exec_module(o)
RAW_REBUILD = o.raw_rebuild_evidence
BASELINE_EVIDENCE = o.baseline_evidence


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    base = tmp_path.resolve()
    baseline = base/'runs/baseline/campaign_spec.json'
    balanced = base/'runs/balanced/campaign_spec.json'
    cache = base/'cache'
    for k, v in dict(BASE=base, BASELINE=baseline, BALANCED=balanced, CACHE=cache).items():
        monkeypatch.setattr(o, k, v)
    baseline.parent.mkdir(parents=True); balanced.parent.mkdir()
    cache.mkdir(); (cache/'blocks').mkdir()
    s = o.record('TEST_BASELINE', cache=str(cache))
    o.publish(baseline, s)
    outputs = {}
    for key in ('checkpoint', 'report', 'predictions'):
        path = baseline.parent/(key+'.payload'); path.write_bytes(('kept-'+key).encode())
        outputs[key] = dict(o.file_ref(path), path=path.name)
    r = dict(contract='LUKA_NEW_FULLSTAT_HYDRA_RESULT/v1', spec=s['content_hash'],
             job='174643', **outputs)
    r = dict(r, content_hash=o.digest(r)); o.publish(baseline.parent/'result.json', r)
    refs = []
    for i, role in enumerate(('train', 'validation')):
        path = cache/'blocks'/f'{i:05d}_00000_{role}.npz'
        path.write_bytes(('derived-'+role).encode())
        refs.append(dict(o.file_ref(path), path=path.relative_to(cache).as_posix()))
    op = o.record('SPEC', root=str(o.operation_path().parent), project=str(base/'operator'), commit='a'*40,
        payloads=refs, available_gib=14, cache=str(cache),
        runtime_overrides=['campaign.storage', 'campaign.commands'],
        baseline=dict(spec=o.file_ref(baseline), result=o.file_ref(baseline.parent/'result.json'),
            outputs={k: o.file_ref(baseline.parent/v['path']) for k, v in outputs.items()}))
    (cache/'cache.json').write_text('metadata retained')
    (op_root := Path(op['root'])).mkdir(parents=True)
    spec = dict(root=str(balanced.parent), dataset=str(base/'data'), content_hash='s')
    events = []
    campaign = SimpleNamespace(remaining=lambda *a: (['train_M0HLT'], {'OFFLINE':'receipt'}),
        committed_bytes=lambda root: 2*2**30)
    monkeypatch.setattr(o, 'other_consumers', lambda *a: events.append('consumers'))
    monkeypatch.setattr(o, 'baseline_evidence', lambda *a: events.append('baseline'))
    monkeypatch.setattr(o, 'raw_rebuild_evidence', lambda *a: events.append('raw'))
    return SimpleNamespace(base=base, cache=cache, baseline=baseline, op=op, spec=spec,
                           campaign=campaign, events=events, op_root=op_root)


def test_retirement_preserves_science_and_metadata_and_is_idempotent(fixture):
    f = fixture
    result = o.retire(f.op, f.campaign, f.spec)
    assert result['raw_sources_verified'] and o.retired(f.op) == result
    assert f.events[:3] == ['consumers', 'baseline', 'raw']
    assert not list((f.cache/'blocks').iterdir())
    assert (f.cache/'cache.json').read_text() == 'metadata retained'
    o.preserved_baseline(f.op)
    assert o.retire(f.op, f.campaign, f.spec) == result


@pytest.mark.parametrize('failure', ['corrupt', 'missing', 'extra', 'outside', 'raw', 'live'])
def test_invalid_inputs_never_delete_any_remaining_blocks(fixture, monkeypatch, failure):
    f = fixture; op = json.loads(json.dumps(f.op))
    first = f.cache/op['payloads'][0]['path']
    if failure == 'corrupt': first.write_bytes(b'corrupt')
    if failure == 'missing': first.unlink()
    if failure == 'extra': (f.cache/'blocks/unknown.bin').write_bytes(b'extra')
    if failure == 'outside': op['payloads'][0]['path'] = '../selected.pt'
    if failure == 'raw': monkeypatch.setattr(o, 'raw_rebuild_evidence', lambda *a: (_ for _ in ()).throw(ValueError('raw')))
    if failure == 'live': f.campaign.remaining = lambda *a: (_ for _ in ()).throw(ValueError('active'))
    before = {p: p.read_bytes() for p in (f.cache/'blocks').iterdir()}
    with pytest.raises((ValueError, FileNotFoundError)): o.retire(op, f.campaign, f.spec)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (f.op_root/'retirement_intent.json').exists()


def test_interrupted_deletion_resumes_only_after_bound_intent(fixture, monkeypatch):
    f = fixture; original = Path.unlink
    second = f.cache/f.op['payloads'][1]['path']
    def interrupted(path, *a, **kw):
        if path == second: raise OSError('simulated interruption')
        return original(path, *a, **kw)
    with monkeypatch.context() as m:
        m.setattr(Path, 'unlink', interrupted)
        with pytest.raises(OSError, match='interruption'): o.retire(f.op, f.campaign, f.spec)
    assert (f.op_root/'retirement_intent.json').exists() and second.exists()
    assert not (f.op_root/'retirement_complete.json').exists()
    o.retire(f.op, f.campaign, f.spec)
    assert not second.exists()
    o.preserved_baseline(f.op)


def test_hardlinks_rejected_without_unlink(fixture):
    import os
    f = fixture; path = f.cache/f.op['payloads'][0]['path']
    alias = f.base/'external_link'; os.link(path, alias)
    with pytest.raises(ValueError, match='linked'): o.payload_state(f.op)
    assert path.exists() and alias.exists()


def test_reappeared_retired_payload_blocks_quota_credit(fixture, monkeypatch):
    f = fixture; o.retire(f.op, f.campaign, f.spec)
    path = f.cache/f.op['payloads'][0]['path']; path.write_bytes(b'reappeared')
    monkeypatch.setattr(o.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=100*2**30))
    with pytest.raises(ValueError, match='reappeared'):
        o.quota(f.op, f.spec, f.campaign, f.spec['dataset'], f.spec['root'], f.baseline, 29, resume=True)


def test_completed_baseline_no_longer_reserves_regeneration_space(fixture, monkeypatch):
    f = fixture; o.retire(f.op, f.campaign, f.spec)
    monkeypatch.setattr(o.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=30*2**30))
    args = (f.op, f.spec, f.campaign, f.spec['dataset'], f.spec['root'], f.baseline)
    quota = o.quota(*args, 29, resume=True)
    assert quota['required_free_bytes'] == 12*2**30
    assert quota['natural_baseline']['remaining_bytes'] == 0
    assert quota['declared_available_gib'] == 14
    with pytest.raises(OSError): o.quota(*args, 11, resume=True)
    with pytest.raises(ValueError): o.quota(*args, 29, resume=False)
    (f.baseline.parent/'checkpoint.payload').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='Payload differs'): o.quota(*args, 29, resume=True)


def test_rehashed_protected_output_redirection_rejected(fixture):
    f = fixture; op = json.loads(json.dumps(f.op))
    op['baseline']['outputs']['checkpoint'] = op['baseline']['outputs']['predictions']
    with pytest.raises(ValueError, match='reference differs'): o.preserved_baseline(op)


def test_cache_manifest_scope_uses_original_contracts(fixture):
    from hlt_classification.luka_new_fullstat_hydra import contracts as c
    f = fixture
    blocks = [dict(r, role=role, rows=count) for r, role, count in
        zip(f.op['payloads'], ('train', 'validation'), (3000000, 500000))]
    (f.cache/'cache.json').unlink()
    o.publish(f.cache/'cache.json', c.artifact('CACHE', parent='foundation', coordinate='OFFLINE',
        counts=dict(train=3000000, validation=500000), blocks=blocks))
    o.publish(f.cache/'preparation_intent.json',
              c.artifact('PREPARATION_INTENT', parent='foundation', coordinate='OFFLINE'))
    assert o.cache_inventory({'foundation_sha256': 'foundation'}) == f.op['payloads']
    with pytest.raises(ValueError, match='Wrong prepared'):
        o.cache_inventory({'foundation_sha256': 'other'})


def test_raw_and_membership_bytes_required_for_rebuild(fixture):
    f = fixture
    dataset = f.base/'data'; foundation_root = dataset/'foundation'
    foundation_root.mkdir(parents=True)
    raw = dataset/'tiny.root'; raw.write_bytes(b'opaque raw bytes, never decoded')
    member = foundation_root/'membership.npz'; member.write_bytes(b'opaque membership')
    foundation = o.record('FOUNDATION_TEST', files=[dict(path='tiny.root', sha256=o.sha(raw),
        membership=dict(o.file_ref(member), path=member.name))])
    o.publish(foundation_root/'foundation.json', foundation)
    f.baseline.unlink()
    o.publish(f.baseline, o.record('TEST_BASELINE', foundation_sha256=foundation['content_hash']))
    op = dict(f.op, dataset=str(dataset), foundation=o.file_ref(foundation_root/'foundation.json'))
    RAW_REBUILD(op)
    raw.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Raw rebuild'): RAW_REBUILD(op)
    raw.write_bytes(b'opaque raw bytes, never decoded'); member.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Payload differs'): RAW_REBUILD(op)


def test_baseline_evidence_matches_original_completion_and_selection_schema(fixture, monkeypatch):
    from hlt_classification import provenance
    from hlt_classification.luka_new_fullstat_hydra import campaign, contracts as c, training
    f = fixture
    s = c.artifact('SPEC', commit=o.BASELINE_COMMIT, root=str(f.baseline.parent), cache=str(f.cache),
                   project=str(f.base/'old_source'), source={}, environment={'test': True})
    f.baseline.unlink(); o.publish(f.baseline, s)
    rows = [dict(pass_number=i, update=i*23438, validation=dict(rows=500000,
        macro_ovr_auc=0.9+i/1000, cross_entropy=1., macro_mean_log_qcd_rejection_at_50pct_signal=2.))
        for i in range(1, 21)]
    report_path = f.baseline.parent/'report.payload'; report_path.unlink()
    report = c.artifact('TRAINING_REPORT', spec=s['content_hash'], passes=20, validation_history=rows,
        selected_weights_restored=True, selected_pass=20, validation=rows[-1]['validation'],
        source_commit=s['commit'], environment=s['environment'], recipe=c.recipe(), node=c.node())
    o.publish(report_path, report)
    result_path = f.baseline.parent/'result.json'; result_path.unlink()
    result = c.artifact('RESULT', spec=s['content_hash'], job='174643', **{
        k: dict(o.file_ref(f.baseline.parent/(k+'.payload')), path=k+'.payload')
        for k in ('checkpoint', 'report', 'predictions')})
    o.publish(result_path, result)
    folder = f.baseline.parent/'submission/attempt_001'; folder.mkdir(parents=True)
    fields = dict(spec=s['content_hash'], command=['sbatch', 'original'])
    o.publish(folder/'train.intent.json', c.artifact('SUBMISSION_INTENT', **fields, plan='plan'))
    o.publish(folder/'train.submitted.json', c.artifact('SUBMITTED', **fields, task='train', job='174643'))
    monkeypatch.setattr(campaign, 'check', lambda *a, **k: None)
    monkeypatch.setattr(provenance, 'validate_source_snapshot', lambda *a, **k: None)
    monkeypatch.setattr(training, 'committed', lambda *a: (rows, []))
    f.campaign.terminal = lambda job: [job, 'COMPLETED', '0:0']
    baseline, evidence = BASELINE_EVIDENCE(f.campaign)
    assert baseline == s and evidence['outputs']['checkpoint']['sha256'] == result['checkpoint']['sha256']
    f.campaign.terminal = lambda job: [job, 'FAILED', '1:0']
    with pytest.raises(ValueError, match='did not complete'): BASELINE_EVIDENCE(f.campaign)


def test_protected_baseline_corruption_before_unlink_preserves_inputs(fixture):
    f = fixture
    (f.baseline.parent/'report.payload').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Payload differs'): o.retire(f.op, f.campaign, f.spec)
    assert all((f.cache/r['path']).is_file() for r in f.op['payloads'])
    assert not (f.op_root/'retirement_intent.json').exists()


def test_missing_completed_offline_blocks_recovery_before_deletion(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(o, 'prepare', lambda *a: f.op)
    f.campaign.remaining = lambda *a: (['train_OFFLINE'], {})
    with pytest.raises(ValueError, match='refit'):
        o.recover_locked('project', 'a'*40, 14, f.campaign, f.spec)
    assert all((f.cache/r['path']).is_file() for r in f.op['payloads'])


def test_recovery_orders_review_retirement_dry_plan_then_original_submission(fixture, monkeypatch):
    f = fixture; events = []
    monkeypatch.setattr(o, 'prepare', lambda *a: f.op)
    original_retire = o.retire
    def retire(*args):
        events.append('retire'); return original_retire(*args)
    monkeypatch.setattr(o, 'retire', retire)
    f.campaign.storage = lambda *a: pytest.fail('unmodified quota guard must not run')
    f.campaign.commands = lambda *a: [dict(task='train_M0HLT', dependencies=[], command=['sbatch', 'old', 'p', 's', 't'])]
    f.campaign.site = lambda rows: events.append('resources')
    f.campaign.c = SimpleNamespace(AUTH='original authorization')
    def queue(spec, *, execute=False, authorization=None, resume=False, available_gib=None):
        assert resume and available_gib == 14
        assert (f.op_root/'retirement_complete.json').exists()
        assert f.campaign.commands(spec, ['train_M0HLT'])[0]['command'][-3] == f.op['project']
        if execute:
            assert authorization == f.campaign.c.AUTH
            events.append('submit'); return {'train_M0HLT': '123'}
        events.append('dry'); return {'commands': f.campaign.commands(spec, ['train_M0HLT'])}
    f.campaign.queue = queue
    jobs = o.recover_locked('project', 'a'*40, 14, f.campaign, f.spec)
    assert jobs == {'train_M0HLT': '123'} and events == ['resources', 'retire', 'dry', 'submit']


def test_wrapper_preserves_real_original_resources_dependencies_and_restores_functions(fixture):
    from hlt_classification.luka_new_balanced_hydra import campaign
    from test_luka_new_balanced_hydra import spec_fixture
    f = fixture; spec = spec_fixture(f.base/'fixture')
    tasks = [t for t in campaign.c.tasks() if t != 'train_OFFLINE']
    original = campaign.commands(spec, tasks)
    old_commands, old_storage = campaign.commands, campaign.storage
    with pytest.raises(RuntimeError, match='test'):
        with o.execution(f.op, spec, campaign):
            altered = campaign.commands(spec, tasks)
            assert len(altered) == 5 and altered[0]['task'] == 'train_M0HLT'
            for a, b in zip(original, altered):
                assert a['dependencies'] == b['dependencies']
                assert a['command'][:-4] == b['command'][:-4]
                assert b['command'][-4:] == [str(Path(f.op['project'])/o.WORKER), f.op['project'],
                                             str(o.operation_path()), b['task']]
            raise RuntimeError('test')
    assert campaign.commands is old_commands and campaign.storage is old_storage


def test_no_scientific_call_or_deletion_without_authorization(monkeypatch):
    monkeypatch.setattr(o, 'bootstrap', lambda: pytest.fail('must not start'))
    with pytest.raises(ValueError, match='authorization'): o.recover('.', 'a'*40, 14, 'wrong')
    with pytest.raises(ValueError, match='authorization'): o.cleanup('.', 'a'*40, 14, o.AUTH)


def test_cleanup_only_never_calls_queue_or_scheduler_review(fixture, monkeypatch, capsys):
    from contextlib import nullcontext
    f = fixture
    monkeypatch.setattr(o, 'bootstrap', lambda: (f.spec, f.campaign))
    monkeypatch.setattr(o, 'operational_lock', nullcontext)
    monkeypatch.setattr(o, 'prepare', lambda *a: f.op)
    for name in ('queue', 'commands', 'site'):
        setattr(f.campaign, name, lambda *a, **k: pytest.fail('cleanup must not reach submission'))
    completion = o.cleanup('project', 'a'*40, 14, o.CLEANUP_AUTH)
    assert completion == o.retired(f.op)
    assert 'CLEANUP_COMPLETE' in capsys.readouterr().out
    assert not (Path(f.spec['root'])/'submission').exists()
    o.preserved_baseline(f.op)
    with pytest.raises(FileExistsError): o.cleanup('project', 'a'*40, 14, o.CLEANUP_AUTH)
    assert o.cleanup('project', 'a'*40, 14, o.CLEANUP_AUTH, continue_existing=True) == completion


def test_consumer_scan_catches_nested_external_specs_and_active_baseline(fixture, monkeypatch):
    f = fixture
    # Restore the real function replaced by the synthetic fixture.
    module_loader = importlib.util.spec_from_file_location('consumer_module', ROOT/o.MODULE)
    other = importlib.util.module_from_spec(module_loader); module_loader.loader.exec_module(other)
    for key in ('BASE', 'BASELINE', 'BALANCED', 'CACHE'): setattr(other, key, getattr(o, key))
    monkeypatch.setattr(other.subprocess, 'check_output', lambda *a, **k: '')
    nested = f.base/'runs/other/nested/recovery_spec.json'; nested.parent.mkdir(parents=True)
    nested.write_text(json.dumps({'cache': str(f.cache)}))
    with pytest.raises(ValueError, match='Another saved'): other.other_consumers(f.cache)
    nested.unlink()
    values = iter(['123', 'WorkDir=code_new_fullstat_a865e5b6'])
    monkeypatch.setattr(other.subprocess, 'check_output', lambda *a, **k: next(values))
    with pytest.raises(ValueError, match='still active'): other.other_consumers(f.cache)


def test_bootstrap_only_loads_original_scientific_checkout(tmp_path):
    project = tmp_path/'original'; package = project/'src/hlt_classification/luka_new_balanced_hydra'
    package.mkdir(parents=True)
    (package.parent/'__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    (package/'campaign.py').write_text('def check(s): pass\ndef runtime(): return {"test": True}\n')
    path = tmp_path/'runs/campaign_spec.json'; path.parent.mkdir()
    s = o.record('BOOT_TEST', root=str(path.parent), project=str(project), commit=o.OLD_COMMIT,
                 environment_dir=sys.prefix, environment={'test': True})
    o.publish(path, s)
    code = '''import importlib.util,sys
from pathlib import Path
l=importlib.util.spec_from_file_location('operation',sys.argv[1]);m=importlib.util.module_from_spec(l);l.loader.exec_module(m)
m.BALANCED=Path(sys.argv[2]);m.OLD_HASH=sys.argv[3]
s,c=m.bootstrap()
assert Path(c.__file__).resolve()==Path(s['project'])/'src/hlt_classification/luka_new_balanced_hydra/campaign.py'
print('ORIGINAL_ONLY')
'''
    run = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT/o.MODULE), str(path), s['content_hash']],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'ORIGINAL_ONLY' in run.stdout


def test_shell_syntax_absolute_worker_and_detached_launch():
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if sys.platform == 'win32' and git_bash.is_file() else shutil.which('bash')
    if bash is None: pytest.skip('Bash unavailable')
    if not Path(bash).is_file(): pytest.skip('Bash unavailable')
    for name in (o.WORKER, o.LAUNCHER):
        path = ROOT/name
        assert b'\r' not in path.read_bytes()
        result = subprocess.run([bash, '-n', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    launcher = (ROOT/o.LAUNCHER).read_text()
    assert 'nohup bash "$PROJECT/scripts/' in launcher and '</dev/null &' in launcher
    assert 'PYTHONNOUSERSITE=1' in launcher and 'LD_LIBRARY_PATH=' in launcher
    assert 'exec "$ENV_DIR/bin/python"' in (ROOT/o.WORKER).read_text()
    assert 'detach-cleanup' in launcher and 'ACTION=cleanup' in launcher
    assert o.CLEANUP_AUTH in launcher
