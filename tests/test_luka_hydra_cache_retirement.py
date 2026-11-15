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
AUTHENTICATED_RETIREMENT = o.authenticated_retirement


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
    with pytest.raises(ValueError, match='authorization'): o.resume_campaign('.', 'a'*40, 28, o.CLEANUP_AUTH)


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
    for name in (o.WORKER, o.RESUME_WORKER, o.LAUNCHER):
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
    assert 'detach-resume' in launcher and 'ACTION=resume' in launcher and o.RESUME_AUTH in launcher


@pytest.fixture
def post_cleanup(fixture, monkeypatch):
    from hlt_classification.luka_new_balanced_hydra import contracts as c
    f = fixture
    fields = {k: v for k, v in f.op.items() if k not in ('content_hash', 'contract', 'schema_version', *o.FLAGS)}
    fields.update(commit=o.RETIREMENT_COMMIT, available_gib=1, source={'git_commit': o.RETIREMENT_COMMIT})
    f.op = o.record('SPEC', **fields)
    o.publish(o.operation_path(), f.op)
    complete = o.retire(f.op, f.campaign, f.spec)
    saved = o.record('FIXTURE_BALANCED', project=str(f.base/'old_science'),
                     **{k: v for k, v in f.spec.items() if k != 'content_hash'})
    o.publish(o.BALANCED, saved); f.spec = saved
    receipt = o.record('FIXTURE_RESULT', stage='OFFLINE')
    o.publish(o.BALANCED.parent/'train_OFFLINE/result.json', receipt)
    f.completed = {'OFFLINE': receipt['content_hash']}
    f.tasks = [t for t in c.tasks() if t != 'train_OFFLINE']
    f.campaign.c = c
    f.campaign.remaining = lambda *a: (f.tasks, f.completed)
    monkeypatch.setattr(o, 'operator_source', lambda *a: {'git_commit': 'b'*40})
    monkeypatch.setattr(o, 'authenticated_retirement', lambda *a: (f.op, o.retired(f.op)))
    monkeypatch.setattr(o.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=int(28.30*2**30)))
    f.post = o.prepare_resume(f.base/'new_operator', 'b'*40, 28, f.campaign, f.spec)
    # Nothing in continuation may re-enter destructive retirement or hash raw data.
    monkeypatch.setattr(o, 'retire', lambda *a: pytest.fail('no deletion on resume'))
    monkeypatch.setattr(o, 'raw_rebuild_evidence', lambda *a: pytest.fail('no repeated raw scan'))
    return f


def test_post_cleanup_28_gib_is_independent_of_original_quota_and_preserves_records(post_cleanup):
    f = post_cleanup
    old_bytes = o.operation_path().read_bytes()
    quota = o.resume_quota(f.post, f.spec, f.campaign, f.spec['dataset'], f.spec['root'], f.baseline, 29, resume=True)
    assert quota['declared_available_gib'] == 28 and f.op['available_gib'] == 1
    assert quota['natural_baseline']['remaining_bytes'] == 0
    assert quota['required_free_bytes'] == 12*2**30
    assert o.operation_path().read_bytes() == old_bytes
    assert o.load_resume(o.resume_path(), f.post['project'], f.campaign, f.spec) == f.post
    assert o.prepare_resume(f.post['project'], 'b'*40, 28, f.campaign, f.spec) == f.post
    with pytest.raises(ValueError, match='registration differs'):
        o.prepare_resume(f.post['project'], 'b'*40, 27, f.campaign, f.spec)


@pytest.mark.parametrize('failure', ['disk', 'declaration', 'receipt', 'reappeared', 'model'])
def test_post_cleanup_fails_closed_before_submission(post_cleanup, monkeypatch, failure):
    f = post_cleanup
    declared = 29
    if failure == 'disk': monkeypatch.setattr(o.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=11*2**30))
    if failure == 'declaration': declared = 11
    if failure == 'receipt': (f.op_root/'retirement_complete.json').unlink()
    if failure == 'reappeared': (f.cache/f.op['payloads'][0]['path']).write_bytes(b'reappeared')
    if failure == 'model': (f.baseline.parent/'checkpoint.payload').write_bytes(b'corrupt')
    with pytest.raises((ValueError, OSError)):
        o.resume_quota(f.post, f.spec, f.campaign, f.spec['dataset'], f.spec['root'], f.baseline, declared, resume=True)


def test_post_cleanup_requires_frozen_cleanup_source_before_import(post_cleanup, monkeypatch):
    from hlt_classification import provenance
    f = post_cleanup
    def reject(*a, **kw): raise ValueError('source changed')
    monkeypatch.setattr(provenance, 'validate_source_snapshot', reject)
    # The real function, replaced only for the synthetic fixture above.
    with pytest.raises(ValueError, match='source changed'):
        AUTHENTICATED_RETIREMENT(f.campaign, f.spec)


def test_post_cleanup_real_submission_preserves_five_fit_dag(post_cleanup, monkeypatch):
    from hlt_classification.luka_new_balanced_hydra import campaign
    from test_luka_new_balanced_hydra import spec_fixture
    f = post_cleanup
    spec = spec_fixture(f.base/'command_fixture'); spec['root'] = f.spec['root']; spec['dataset'] = f.spec['dataset']
    spec['storage']['natural_baseline']['spec_path'] = str(f.baseline)
    monkeypatch.setattr(o, 'prepare_resume', lambda *a: f.post)
    monkeypatch.setattr(campaign, 'check', lambda *a, **kw: None)
    def remaining(*a):
        campaign.accepted_jobs(spec)  # Preserve ambiguous-intent rejection.
        return f.tasks, f.completed
    monkeypatch.setattr(campaign, 'remaining', remaining)
    monkeypatch.setattr(campaign, 'committed_bytes', lambda *a: 2*2**30)
    reviews = []; calls = []
    monkeypatch.setattr(campaign, 'site', lambda rows: reviews.append(rows))
    def accepted(args, **kw):
        calls.append(args); return SimpleNamespace(returncode=0, stdout=str(200+len(calls)), stderr='')
    monkeypatch.setattr(campaign.subprocess, 'run', accepted)
    jobs = o.resume_locked(f.post['project'], 'b'*40, 28, campaign, spec)
    assert list(jobs) == f.tasks and len(calls) == 5 and len(reviews) == 2
    assert '--dependency=afterok:202:201' in calls[2]
    assert '--dependency=afterok:202:203:201' in calls[3]
    assert '--dependency=afterok:202:203:204' in calls[4]
    original = campaign.commands(spec, f.tasks)
    for args, row in zip(calls, original):
        assert [x for x in args[:-4] if not x.startswith('--dependency=')] == [x for x in row['command'][:-4] if not x.startswith('--dependency=')]
        assert args[-4:] == [str(Path(f.post['project'])/o.RESUME_WORKER), f.post['project'], str(o.resume_path()), row['task']]
    assert len(campaign.accepted_jobs(spec)) == 5
    with o.execution(f.post, spec, campaign, post_cleanup=True):
        with pytest.raises(ValueError, match='unfinished stages'):
            campaign.commands(spec, ['train_OFFLINE'])
    with pytest.raises(FileExistsError): o.resume_locked(f.post['project'], 'b'*40, 28, campaign, spec)
    # A live predecessor stops explicit continuation before another sbatch.
    monkeypatch.setattr(campaign, 'remaining', lambda *a: (_ for _ in ()).throw(ValueError('live prior job')))
    with pytest.raises(ValueError, match='live'):
        o.resume_locked(f.post['project'], 'b'*40, 28, campaign, spec, continue_existing=True)
    assert len(calls) == 5


def test_post_cleanup_worker_installs_guard_and_publishes_execution(post_cleanup, monkeypatch):
    from hlt_classification.luka_new_balanced_hydra import worker
    f = post_cleanup; job = '123'; task = 'train_M0HLT'
    monkeypatch.setenv('SLURM_JOB_ID', job)
    f.campaign.binding = lambda *a: {'command': ['sbatch', str(Path(f.post['project'])/o.RESUME_WORKER),
        f.post['project'], str(o.resume_path()), task]}
    original = lambda *a, **kw: pytest.fail('old storage reservation was used')
    f.campaign.storage = original; f.campaign.commands = lambda *a: []
    def run(spec, task):
        quota = f.campaign.storage(spec['dataset'], spec['root'], f.baseline, 29, resume=True)
        assert quota['declared_available_gib'] == 28 and quota['natural_baseline']['remaining_bytes'] == 0
        result = o.record('FIXTURE_RESULT', stage='M0HLT')
        o.publish(Path(spec['root'])/task/'result.json', result)
        return result
    monkeypatch.setattr(worker, 'run', run)
    result = o.run_task(f.post, f.spec, f.campaign, task, post_cleanup=True)
    assert result['stage'] == 'M0HLT' and f.campaign.storage is original
    complete = o.read(Path(f.post['root'])/'executions/123.complete.json', 'RESUME_EXECUTION_COMPLETE')
    assert complete['parents']['operation'] == f.post['content_hash']


def test_post_cleanup_detached_shell_finishes_after_tail_exits(tmp_path):
    import os
    import time
    bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
    def shellpath(path):
        p = Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform == 'win32' else str(p)
    project = tmp_path/'project'; (project/'scripts').mkdir(parents=True)
    env_dir = tmp_path/'environment'; (env_dir/'bin').mkdir(parents=True)
    root = tmp_path/'campaign'; root.mkdir()
    script = project/o.LAUNCHER
    text = (ROOT/o.LAUNCHER).read_text().replace(
        'ROOT=/cs/home/rlyang/hlt_u100_pairs/runs/new_balanced_4a61a34c_r1',
        'ROOT="'+shellpath(root)+'"')
    script.write_bytes(text.encode())
    binpath = tmp_path/'bin'; binpath.mkdir()
    bodies = {
        binpath/'git': 'printf "%s\\n" "'+('b'*40)+'"\n',
        binpath/'python3': 'printf "%s\\n" "$FIXTURE_ENV"\n',
        binpath/'tail': 'printf "TAIL_VIEW\\n"\n',
        env_dir/'bin/python': 'sleep 1\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n',
    }
    for path, body in bodies.items():
        path.write_bytes(('#!/usr/bin/env bash\n'+body).encode()); path.chmod(0o755)
    bash_env = tmp_path/'environment.sh'
    bash_env.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    calls = tmp_path/'calls'
    env = dict(os.environ, BASH_ENV=shellpath(bash_env), FIXTURE_BIN=shellpath(binpath),
               FIXTURE_CALLS=shellpath(calls), FIXTURE_ENV=shellpath(env_dir))
    args = [bash, str(script), 'detach-resume', shellpath(project), 'b'*40, '28']
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'DETACHED resume' in result.stdout and 'TAIL_VIEW' in result.stdout
    log = Path(str(root)+'.post-cache-resume-launch')/'launcher.log'
    deadline = time.monotonic()+10
    while (not log.exists() or 'LAUNCHER_EXIT=0' not in log.read_text()) and time.monotonic() < deadline:
        time.sleep(.05)
    assert 'LAUNCHER_EXIT=0' in log.read_text()
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 and ' resume ' in lines[0] and '--available-gib 28' in lines[0]
    assert '--authorize '+o.RESUME_AUTH in lines[0]
    assert subprocess.run(args, env=env, capture_output=True, timeout=5).returncode != 0
