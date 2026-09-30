"""Source-pinned repair and one-shot pilot -> bulk handoff, no real scheduler."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hlt_classification.cms2jc2_production import (
    campaign as c, recovery as r, recovery_controller as rc, submission as sub,
    output as out, worker as w, contracts as k,
)
from hlt_classification.cms2jc2_response.contracts import with_content_hash
from test_cms2jc2_proxy_production import (
    study, population, bundle, native_threads, terminal_fixture, fixture_generate,
    pairs_for, LocalMeasurement,
)


@pytest.fixture
def setup(study, bundle, monkeypatch, tmp_path, request):
    if getattr(request, 'param', False):
        study = with_content_hash(dict(study,
            contract='CMS2JC2_PROXY_STUDY_REDUCED_CONFIRMATION/v1', allow_reduced_confirmation=True,
            confirmation_status='inconclusive_incomplete_population', confirmation_scope=dict(
                kind='reduced_36_of_57', full_population_status='inconclusive_incomplete_population',
                subset_decision='rejected', coverage=dict(included_indices=list(range(36)), excluded_indices=list(range(36, 57))))))
        (Path(study['root'])/'study_spec.json').write_text(json.dumps(study))
        preflight = k.load_json(Path(study['root'])/'preflight.json')
        preflight = with_content_hash(dict(preflight, parents={'study': study['content_hash']}))
        (Path(study['root'])/'preflight.json').write_text(json.dumps(preflight))
    original = terminal_fixture(study)
    receipt = fixture_generate(monkeypatch, study, study['shards'][0], bundle)
    receipt = with_content_hash(dict(receipt, allocation=dict(job_id='124', array_job_id='123', index=0)))
    receipt_path = Path(study['root'])/'shards/s00000.json'
    receipt_path.write_text(json.dumps(receipt))  # disposable fixture only
    states = [dict(job='120', state='COMPLETED', exit_code='0:0', name='c2jp_preflight'),
              dict(job='123_0', state='COMPLETED', exit_code='0:0', name='c2jp_generate'),
              dict(job='123_1', state='FAILED', exit_code='1:0', name='c2jp_generate')]
    monkeypatch.setattr(sub, 'accounting', lambda ids: [s for s in states if s['job'].split('_')[0] in ids])
    def scheduler(argv, **kwargs):
        assert argv[0] == 'squeue', 'Dry preparation must never submit'
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr(sub.subprocess, 'run', scheduler)
    project = tmp_path/'replacement'; project.mkdir()
    source = k.artifact('SOURCE', commit='b'*40, executable=True, files={p: 'b'*64 for p in r.ADDED})
    monkeypatch.setattr(r, 'source_snapshot', lambda *a: source)
    monkeypatch.setattr(r, 'active_project', lambda: project)
    monkeypatch.setattr(c, 'numerical_environment', lambda: study['numerical_environment'])
    path = Path(study['root'])/'study_spec.json'
    return SimpleNamespace(study=study, source=source, project=project, path=path,
                           states=states, original=original, receipt=receipt, bundle=bundle)


def prepare(setup):
    return r.create(setup.path, project_dir=setup.project, source_commit='b'*40)


def test_dry_recovery_preserves_original_and_only_retries_missing_pilot(setup):
    root = Path(setup.study['root'])
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan = prepare(setup)
    repair, attempt, saved = r.read_plan(setup.study)
    assert saved == plan and prepare(setup) == plan
    assert attempt['shards'] == ['s00001'] and attempt['kind'] == 'pilot'
    assert attempt['contract'] == 'CMS2JC2_PROXY_ATTEMPT_EXECUTION_REPAIR/v1'
    commands = sub.get_plan(k.checked(plan['initial_attempt']))[2]['commands']
    assert len(commands) == 1 and commands[0]['task'] == 'generate'
    assert commands[0]['dependency'] is None  # reuse successful preflight
    assert '--array=0-0%1' in commands[0]['argv']
    assert '--chdir='+str(setup.project) in commands[0]['argv']
    assert set(repair['retained']) == {'s00000'}
    assert all(p.read_bytes() == value for p, value in original.items())
    assert list(root.rglob('*_response.json'))  # original journal retained
    assert not (k.checked(plan['initial_attempt']).parent/'submission').exists()


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'UNKNOWN'])
def test_recovery_refuses_live_or_unknown_original(setup, state):
    setup.states[-1]['state'] = state
    with pytest.raises(PermissionError): prepare(setup)
    assert not (Path(setup.study['root'])/r.REPAIR_PATH).exists()


def test_success_without_missing_receipt_is_not_silently_retried(setup):
    setup.states[-1].update(state='COMPLETED', exit_code='0:0')
    with pytest.raises(PermissionError, match='unsuccessful'): prepare(setup)


def test_corrupt_retained_block_stops_preparation(setup):
    path = Path(setup.study['root'])/setup.receipt['blocks'][0]['relative']
    path.write_bytes(b'broken fixture')
    with pytest.raises(ValueError, match='checksum'): prepare(setup)
    assert not (Path(setup.study['root'])/r.REPAIR_PATH).exists()


def test_changed_source_or_wrong_loaded_project_stops_execution(setup, monkeypatch):
    plan = prepare(setup)
    changed = k.artifact('SOURCE', commit='c'*40, files={}, executable=True)
    with monkeypatch.context() as context:
        context.setattr(r, 'source_snapshot', lambda *a: changed)
        with pytest.raises(ValueError, match='source changed'):
            r.validate_execution(setup.study, plan['execution_repair'])
    monkeypatch.setattr(r, 'active_project', lambda: setup.project.parent)
    with pytest.raises(PermissionError, match='exact replacement'):
        r.validate_execution(setup.study, plan['execution_repair'])


def test_scientific_equivalence_for_actual_changed_modules(tmp_path):
    project = Path(__file__).resolve().parents[1]
    old = tmp_path/'old'
    for name, functions in r.OPERATIONAL.items():
        payload = subprocess.check_output(['git', '-C', str(project), 'show',
            'b6f88defce6f357969b2e6ded80585db439540a3:'+name])
        path = old/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        assert r.scientific_skeleton(path, functions) == r.scientific_skeleton(project/name, functions)


def test_scientific_change_or_added_file_cannot_be_source_repair(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    name = r.PREFIX+'worker.py'
    for root, constant in ((old, 1), (new, 2)):
        path = root/name; path.parent.mkdir(parents=True)
        path.write_text(f'def generate(): return {constant}\ndef run(): return None\n')
    study = dict(project_dir=str(old), source=dict(files={name: 'a'*64}))
    source = dict(files={name: 'b'*64, **{p: 'b'*64 for p in r.ADDED}})
    with pytest.raises(ValueError, match='Scientific source'): r.compatible_source(study, new, source)
    source['files']['injected.py'] = 'a'*64
    with pytest.raises(ValueError, match='file set'): r.compatible_source(study, new, source)


def install_worker_fixture(setup, monkeypatch):
    from hlt_classification.cms2jc2_production import population as pop
    original_generate = w.generate
    monkeypatch.setattr(w, 'generate', lambda *a, **kw: original_generate(*a, **kw, measurement_factory=LocalMeasurement))
    def iterate(data, population, shard, review, **kwargs):
        yield from pairs_for(setup.study, shard)
    monkeypatch.setattr(pop, 'iterate', iterate)
    monkeypatch.setattr(sub, 'worker_identity', lambda *a: dict(job_id='200', array_job_id='200', index=0,
                         cpus=36, gpus=0, partition='tigris'))


def test_repaired_worker_publishes_truthful_lineage_and_reuses_old_shard(setup, monkeypatch):
    plan = prepare(setup)
    install_worker_fixture(setup, monkeypatch)
    result = w.run(k.checked(plan['initial_attempt']), 'generate', index=0)
    assert result['contract'] == 'CMS2JC2_PROXY_SHARD_EXECUTION_REPAIR/v1'
    assert result['source'] == setup.source['content_hash']
    assert result['scientific_source'] == setup.study['source']['content_hash']
    assert result['serial_process_exact'] and result['execution_repair'] == plan['execution_repair']
    assert out.completed(setup.study, physical=True)['s00000'] == setup.receipt
    disguised = with_content_hash(dict(result, contract='CMS2JC2_PROXY_SHARD/v1'))
    with pytest.raises(ValueError, match='Original shard'): out.verify_shard(setup.study, disguised)
    with pytest.raises(FileExistsError): w.run(k.checked(plan['initial_attempt']), 'generate', index=0)


def scheduler_execution(setup, monkeypatch):
    calls = []
    states = setup.states
    def run(argv, **kwargs):
        if argv[0] == 'squeue':
            return SimpleNamespace(stdout='', stderr='', returncode=0)
        assert argv[0] == 'sbatch'
        if '--test-only' in argv:
            return SimpleNamespace(stdout='test accepted', stderr='', returncode=0)
        number = str(200+len(calls))
        calls.append(argv)
        task = argv[-1]
        ids = [number+'_0'] if task == 'generate' and len(calls) == 1 else [number]
        states.extend(dict(job=j, state='COMPLETED', exit_code='0:0', name='c2jp_'+task) for j in ids)
        return SimpleNamespace(stdout=number+'\n', stderr='', returncode=0)
    monkeypatch.setattr(sub.subprocess, 'run', run)
    monkeypatch.setattr(rc, 'controller_lock', lambda root: nullcontext())
    # Never use synthetic timing as genuine admission: explicitly stub in tests.
    monkeypatch.setattr(c, 'resources', lambda rows: dict(memory_gib=32, hours=1, bytes_per_jet=4096))
    return calls


def test_controller_end_to_end_and_restart_do_not_duplicate(setup, monkeypatch):
    plan = prepare(setup)
    calls = scheduler_execution(setup, monkeypatch)
    install_worker_fixture(setup, monkeypatch)
    real_wait = rc.wait_pilot
    def finish(study, attempt, submitted):
        w.run(k.checked(plan['initial_attempt']), 'generate', index=0)
        return real_wait(study, attempt, submitted, sleep=lambda _: pytest.fail('already complete'))
    monkeypatch.setattr(rc, 'wait_pilot', finish)
    result = rc.run(setup.path, plan_hash=plan['content_hash'], authorize_test=True)
    assert len(calls) == 3  # missing pilot, bulk array, finalizer; no new preflight
    assert result['jobs'] == dict(generate='201', finalize='202')
    bulk = rc.continuation(setup.study, plan)
    assert bulk['shards'] == [s['shard_id'] for s in setup.study['shards'] if not s['pilot']]
    assert '--dependency=afterany:201' in calls[-1]
    assert '--cpus-per-task=36' in calls[1] and '--partition=tigris' in calls[1]
    assert result == rc.run(setup.path, plan_hash=plan['content_hash'], authorize_test=True)
    assert len(calls) == 3
    assert not result['dataset_complete'] and c.test_lock(setup.study)['evaluate'] is False


@pytest.mark.parametrize('setup', [False, True], indirect=True)
def test_complete_repaired_bank_and_sealed_reader(setup, monkeypatch):
    plan = prepare(setup)
    install_worker_fixture(setup, monkeypatch)
    w.run(k.checked(plan['initial_attempt']), 'generate', index=0)
    monkeypatch.setattr(sub, 'require_terminal', lambda *a, **k: None)
    monkeypatch.setattr(c, 'resources', lambda rows: dict(memory_gib=32, hours=1, bytes_per_jet=4096))
    bulk = c.advance(setup.study, authorize_test=True, execution_repair=plan['execution_repair'])
    path = Path(setup.study['root'])/f'attempts/{bulk["name"]}/attempt_spec.json'
    for index in range(len(bulk['shards'])):
        w.run(path, 'generate', index=index)
    manifest = w.run(path, 'finalize')
    reduced = c.study_kind(setup.study) == 'STUDY_REDUCED_CONFIRMATION'
    kind = 'DATASET_REDUCED_CONFIRMATION' if reduced else 'DATASET'
    assert manifest['contract'] == f'CMS2JC2_PROXY_{kind}_EXECUTION_REPAIR/v1'
    if reduced:
        assert manifest['confirmation_scope'] == setup.study['confirmation_scope']
        assert manifest['confirmation_status'] == 'inconclusive_incomplete_population'
    assert manifest['source'] == setup.study['source']
    assert manifest['execution_source'] == setup.source
    assert manifest['final_test_materialized'] and not manifest['final_test_evaluated']
    assert len(list(out.read_role(setup.study['root'], 'train'))) == setup.study['counts']['train']
    with pytest.raises(PermissionError, match='sealed'): list(out.read_role(setup.study['root'], 'final_test'))
    assert out.completed(setup.study)['s00000'] == setup.receipt
    disguised = with_content_hash(dict(manifest, contract=f'CMS2JC2_PROXY_{kind}/v1'))
    with pytest.raises(ValueError, match='Original manifest'): r.validate_manifest_execution(setup.study, disguised)


def test_reduced_manifest_keeps_reduced_evidence_kind():
    study = dict(contract='CMS2JC2_PROXY_STUDY_REDUCED_CONFIRMATION/v1')
    assert r.manifest_execution(study, [])[0] == 'DATASET_REDUCED_CONFIRMATION'


def test_failed_pilot_never_queues_bulk(setup, monkeypatch):
    plan = prepare(setup)
    calls = scheduler_execution(setup, monkeypatch)
    original = sub.accounting
    def failed(ids):
        return [dict(row, state='FAILED', exit_code='1:0') if row['job'].startswith('200_') else row
                for row in original(ids)]
    monkeypatch.setattr(sub, 'accounting', failed)
    with pytest.raises(RuntimeError, match='no bulk submitted'):
        rc.run(setup.path, plan_hash=plan['content_hash'], authorize_test=True)
    assert len(calls) == 1
    assert not (Path(setup.study['root'])/'test_build_lock.json').exists()


def test_bad_authorization_and_ambiguous_submission_are_not_retried(setup, monkeypatch):
    plan = prepare(setup)
    calls = scheduler_execution(setup, monkeypatch)
    with pytest.raises(PermissionError, match='materialization'):
        rc.run(setup.path, plan_hash=plan['content_hash'])
    with pytest.raises(PermissionError, match='reviewed'):
        rc.run(setup.path, plan_hash='0'*64, authorize_test=True)
    directory = k.checked(plan['initial_attempt']).parent
    (directory/'submission').mkdir()
    with pytest.raises(PermissionError, match='already attempted'):
        rc.run(setup.path, plan_hash=plan['content_hash'], authorize_test=True)
    assert not calls


def test_different_study_reference_rejected(setup, tmp_path):
    plan = prepare(setup)
    path = tmp_path/'copied_repair.json'
    path.write_bytes(k.checked(plan['execution_repair']).read_bytes())
    with pytest.raises(ValueError, match='exact dataset root'): r.load_repair(setup.study, k.ref(path))


def test_unjournaled_attempt_stops_controller_before_submission(setup, monkeypatch):
    plan = prepare(setup)
    calls = scheduler_execution(setup, monkeypatch)
    c.make_attempt(setup.study, 'pilot', setup.study['shards'][1:2], memory_gib=64, hours=4,
                   bytes_per_jet=4096, execution_repair=plan['execution_repair'])
    with pytest.raises(PermissionError, match='Unjournaled'):
        rc.run(setup.path, plan_hash=plan['content_hash'], authorize_test=True)
    assert not calls


def test_missing_accounting_waits_then_times_out_without_resubmission(setup, monkeypatch):
    plan = prepare(setup)
    attempt = k.load_json(k.checked(plan['initial_attempt']))
    monkeypatch.setattr(sub, 'accounting', lambda ids: [])
    times = iter([0, 1, 15*86400])
    sleeps = []
    with pytest.raises(TimeoutError, match='jobs preserved'):
        rc.wait_pilot(setup.study, attempt, {'jobs': {'generate': '200'}},
                      sleep=sleeps.append, monotonic=lambda: next(times))
    assert sleeps == [60]


def test_helper_detaches_and_has_no_destructive_commands():
    project = Path(__file__).resolve().parents[1]
    script = (project/'scripts/queue_cms2jc2_proxy_recovery.sh').read_text()
    assert 'nohup bash' in script and '</dev/null &' in script
    assert 'RECOVERY_HELPER_EXIT=' in script
    assert all(command not in script for command in ('scancel', 'rm -', 'git reset', 'git checkout'))
    assert 'PYTHONNOUSERSITE=1' in script and 'LD_LIBRARY_PATH=' in script
