"""Pending-only migration, bounded concurrency and unchanged production science."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import getpass
import subprocess
import sys
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_supervision_depth import contracts as science, worker as old
from hlt_classification.luka_depth_recovery import contracts as c, source, scheduler, campaign, submission, worker, scratch
from test_luka_supervision_depth import bare, science_fixture


@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def pair(tmp_path, monkeypatch, reused=('targets', 'baseline', 'set8_p3')):
    original = bare(tmp_path); root = Path(original['root']); root.mkdir()
    write_immutable_json(root/'campaign_spec.json', original)
    spec = dict(root=str(tmp_path/'recovery'), project_dir='/new_project', source_commit='d'*40,
        source_snapshot=dict(content_hash='f'*64), original=file_ref(root/'campaign_spec.json'),
        content_hash='e'*64, reused={t:{} for t in reused}, tasks=c.schedule(reused))
    Path(spec['root']).mkdir()
    monkeypatch.setattr(source, 'original', lambda _:original)
    return original, spec


def test_two_lanes_early_ce_and_only_real_afterok_edges(tmp_path, monkeypatch):
    original, spec = pair(tmp_path, monkeypatch)
    plan = submission.render(spec); rows = spec['tasks']
    assert [r['task_id'] for r in rows[:4]] == ['teacher_ce', 'duplicate_offline', 'ce_diverse', 'teacher_supcon']
    assert plan['jobs'] == 21 and plan['scientific_fits'] == 18 and plan['teacher_fits'] == 2
    assert plan['preflight_jobs'] == plan['target_exports'] == 0
    assert not ({r['task_id'] for r in rows} & {'targets', 'baseline', 'set8_p3', 'cleanup'})
    for lane in (0, 1):
        prior = set()
        for r in rows:
            if r['lane'] != lane:
                continue
            assert prior <= set(r['dependencies'])
            assert r['success_dependencies'] == [t for t in c.scientific_dependencies(r['task_id']) if t not in spec['reused']]
            prior.add(r['task_id'])
    by = {r['task_id']:r for r in rows}
    assert by['ce_diverse']['success_dependencies'] == ['teacher_ce']
    assert by['teacher_supcon']['success_dependencies'] == []
    assert by['teacher_supcon']['slot_dependencies'] == ['duplicate_offline']
    assert by['set8_p10']['success_dependencies'] == []
    for row in plan['commands']:
        cmd = row['command']
        assert '--partition=debug' in cmd and '--no-requeue' in cmd
        assert not any('219739' in v for v in cmd)
        if row['task_id'] == 'teacher_supcon':
            assert '--dependency=afterany:${JOB_duplicate_offline}' in cmd
    assert set(rows[-1]['dependencies']) == c.ALL_FITS - set(spec['reused'])
    # Scientific recipes/nodes were neither rewritten nor downweighted.
    assert original['recipes'] == {a:science.recipe(a) for a in science.ARMS}


@pytest.mark.parametrize('prefix', range(2, 24))
def test_all_completed_prefixes_reuse_every_finished_fit(prefix):
    old_tasks = [r['task_id'] for r in science.tasks()][:prefix]
    reused = [t for t in old_tasks if t in c.ALL_FITS or t == 'targets']
    rows = c.schedule(reused)
    new = {r['task_id'] for r in rows} - {'report'}
    assert new.isdisjoint(reused) and new | (set(reused)-{'targets'}) == c.ALL_FITS


@pytest.mark.parametrize('reused', [[], ['targets'], ['baseline'], ['targets', 'baseline', 'invented']])
def test_no_retraining_controls_or_unknown_reuse(reused):
    with pytest.raises(ValueError): c.schedule(reused)


def test_accounting_identity_is_exact(tmp_path, monkeypatch):
    original = bare(tmp_path)
    text = f"123|lukasd_set8_p10|{getpass.getuser()}|reu-aisocial|CANCELLED by 123|0:0|0|Unknown|\n"
    monkeypatch.setattr(scheduler.subprocess, 'run', lambda *a, **kw:SimpleNamespace(stdout=text))
    assert scheduler.accounting(original, {'set8_p10':'123'})['set8_p10']['state'] == 'CANCELLED'
    with pytest.raises(PermissionError): scheduler.accounting(original, {'teacher_ce':'123'})
    with pytest.raises(ValueError): scheduler.accounting(original, {'teacher_ce':'123;scancel'})


def stop_fixture(tmp_path, monkeypatch):
    original = bare(tmp_path); jobs = {'report':'203', 'set8_p10':'202', 'set8_p3':'201'}
    original['tasks'] = [dict(task_id=t) for t in ('set8_p3', 'set8_p10', 'report')]
    ref = {'path':'original'}
    states = {t:dict(state='PENDING', job_id=j) for t,j in jobs.items()}
    value = c.artifact('STOP_PLAN', original=ref, jobs=jobs, snapshot=states,
        cancel_candidates=['report', 'set8_p10', 'set8_p3'], running_jobs_preserved=True)
    path = tmp_path/'stop.json'; write_immutable_json(path, value)
    monkeypatch.setattr(source, 'authenticate', lambda *a:(ref, original, jobs))
    monkeypatch.setattr(scheduler, 'accounting', lambda s,j:{t:states[t] for t in j})
    return path, value, original, states


def test_stop_freezes_only_pending_and_preserves_dispatch_race(tmp_path, monkeypatch):
    path, value, original, states = stop_fixture(tmp_path, monkeypatch)
    states['set8_p3']['state'] = 'RUNNING'
    commands = []; probes = {}
    def live(s,t,j):
        probes[t] = probes.get(t,0) + 1
        # This one starts after first read but before hold.
        state = 'RUNNING' if t == 'set8_p10' and probes[t] > 1 else 'PENDING'
        return dict(JobState=state, Priority='0', Reason='JobHeldUser')
    monkeypatch.setattr(scheduler, 'live_job', live)
    monkeypatch.setattr(scheduler.subprocess, 'run', lambda cmd, **kw:(commands.append(cmd) or SimpleNamespace(returncode=0)))
    result = scheduler.stop(path, '/project', value['content_hash'])
    assert result['cancelled'] == ['report']
    assert result['preserved'] == ['set8_p10', 'set8_p3']
    assert [cmd for cmd in commands if cmd[0] == 'scancel'] == [['scancel','203']]
    with pytest.raises(FileExistsError): scheduler.stop(path, '/project', value['content_hash'])


def test_stop_wrong_hash_and_failed_hold_never_cancel(tmp_path, monkeypatch):
    path, value, _, _ = stop_fixture(tmp_path, monkeypatch)
    with pytest.raises(PermissionError): scheduler.stop(path, '/p', 'wrong')
    commands = []
    monkeypatch.setattr(scheduler, 'live_job', lambda *a:dict(JobState='PENDING', Priority='100', Reason='Priority'))
    monkeypatch.setattr(scheduler.subprocess, 'run', lambda cmd, **kw:(commands.append(cmd) or SimpleNamespace(returncode=1)))
    with pytest.raises(RuntimeError): scheduler.stop(path, '/p', value['content_hash'])
    assert commands == [['scontrol','hold','203']]


@pytest.mark.parametrize('fault', ['running', 'partial', 'failed', 'started', 'retired', 'bad_receipt'])
def test_inventory_never_silently_restarts_work(tmp_path, monkeypatch, fault):
    original = bare(tmp_path); root = Path(original['root']); root.mkdir()
    jobs = {t['task_id']:str(100+i) for i,t in enumerate(original['tasks'])}
    states = {t:dict(state='CANCELLED', exit_code='0:0', elapsed=0, start='Unknown') for t in jobs}
    for t in ('targets','baseline'):
        states[t] = dict(state='COMPLETED', exit_code='0:0', elapsed=100, start='date')
        (root/t).mkdir(); write_immutable_json(root/t/'receipt.json', dict(task=t))
    monkeypatch.setattr(scheduler, 'accounting', lambda *a:states)
    monkeypatch.setattr(old, 'validate_task', lambda *a:None)
    monkeypatch.setattr(old.submission, 'successful', lambda *a:None)
    if fault == 'running': states['set8_p3']['state'] = 'RUNNING'
    if fault == 'partial': (root/'set8_p3').mkdir()
    if fault == 'failed': states['set8_p3']['state'] = 'FAILED'
    if fault == 'started': states['set8_p3']['start'] = 'date'
    if fault == 'retired':
        (root/'cleanup').mkdir(); write_immutable_json(root/'cleanup/intent.json', {})
    if fault == 'bad_receipt':
        monkeypatch.setattr(old, 'validate_task', lambda *a:(_ for _ in ()).throw(ValueError('corrupt')))
    with pytest.raises(ValueError): campaign.inventory(original, jobs)


def test_submission_full_review_and_real_exact_journal(tmp_path, monkeypatch):
    original, spec = pair(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, 'check', lambda s:original)
    monkeypatch.setattr(submission, 'check_site', lambda p:None)
    commands = []
    def run(cmd, **kw):
        commands.append(cmd); return SimpleNamespace(stdout=str(700+len(commands)))
    with pytest.raises(PermissionError): submission.submit(spec, execute=True, plan_hash='bad', authorization=c.AUTHORIZATION)
    plan = submission.submit(spec)
    monkeypatch.setattr(submission.subprocess, 'run', run)
    ledger = submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    assert len(commands) == 21 and not any('${JOB_' in v for cmd in commands for v in cmd)
    assert submission.accepted_jobs(spec) == ledger['jobs']
    assert '--dependency=afterok:701' in commands[2]
    with pytest.raises(FileExistsError): submission.submit(spec, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)


def test_existing_outputs_block_before_any_submission(tmp_path, monkeypatch):
    original, spec = pair(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, 'check', lambda _:original)
    (Path(original['root'])/'teacher_ce').mkdir()
    with pytest.raises(FileExistsError): submission.submit(spec)


def test_create_check_exclusive_owner_and_no_scientific_edits(tmp_path, monkeypatch):
    original, _ = pair(tmp_path, monkeypatch)
    root = Path(original['root']); jobs = {r['task_id']:str(100+i) for i,r in enumerate(original['tasks'])}
    states = {t:dict(job_id=j, state='CANCELLED', exit_code='0:0', elapsed=0, start='Unknown') for t,j in jobs.items()}
    for task in ('targets','baseline'):
        states[task].update(state='COMPLETED', elapsed=100, start='date')
        (root/task).mkdir(); write_immutable_json(root/task/'receipt.json', dict(task=task))
    ref = file_ref(root/'campaign_spec.json'); snap = dict(content_hash='f'*64)
    monkeypatch.setattr(source, 'current', lambda *a:snap)
    monkeypatch.setattr(source, 'authenticate', lambda *a:(ref, original, jobs))
    monkeypatch.setattr(scheduler, 'accounting', lambda *a:deepcopy(states))
    monkeypatch.setattr(old, 'validate_task', lambda *a:None)
    monkeypatch.setattr(old.submission, 'successful', lambda *a:None)
    monkeypatch.setattr(old.submission, 'accepted_jobs', lambda *a:jobs)
    monkeypatch.setattr(campaign, 'storage', lambda *a:None)
    before = deepcopy(original)
    value = campaign.create(tmp_path/'corrected', ref['path'], tmp_path/'project', 'd'*40, available_quota_gib=8)
    assert campaign.check(value) == original == before
    assert set(value['reused']) == {'targets','baseline'}
    with pytest.raises(FileExistsError):
        campaign.create(tmp_path/'different', ref['path'], tmp_path/'project', 'd'*40, available_quota_gib=8)
    states['set8_p3']['state'] = 'RUNNING'
    with pytest.raises(ValueError, match='active'): campaign.check(value)
    states['set8_p3']['state'] = 'CANCELLED'
    with (root/'baseline/receipt.json').open('r+b') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError): campaign.check(value, live=False)


@pytest.mark.parametrize('free,reserved,required,ok', [(100,20,80,True),(100,20,81,False),(100,101,1,False),(100,0,0,False)])
def test_scratch_reservations_prevent_double_booking(free, reserved, required, ok):
    if ok: scratch.budget(free, reserved, required)
    else:
        with pytest.raises(OSError): scratch.budget(free, reserved, required)


def test_full_production_recovery_reuses_targets_and_controls(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    original, tr, va = science_fixture(tmp_path, monkeypatch)
    for task in ('targets', 'baseline', 'set8_p3'):
        old.run(original, task, device='cuda')
    root = Path(original['root']); reused = {t:file_ref(root/t/'receipt.json') for t in ('targets','baseline','set8_p3')}
    protected = {p:sha256_file(p) for t in reused for p in (root/t).rglob('*') if p.is_file()}
    spec = dict(root=str(tmp_path/'recovery'), reused=reused, tasks=c.schedule(reused), content_hash='f'*64,
        original=file_ref(root/'campaign_spec.json'), project_dir='/recovery', source_commit='d'*40,
        source_snapshot=dict(content_hash='e'*64))
    Path(spec['root']).mkdir(); jobs = {r['task_id']:str(800+i) for i,r in enumerate(spec['tasks'])}
    monkeypatch.setattr(campaign, 'check', lambda *a, **kw:original)
    monkeypatch.setattr(source, 'current', lambda *a:spec['source_snapshot'])
    monkeypatch.setattr(source, 'original', lambda *a:original)
    monkeypatch.setattr(worker, 'execution', lambda s,o,t:jobs[t])
    monkeypatch.setattr(submission, 'accepted_jobs', lambda *a:jobs)
    monkeypatch.setattr(submission, 'successful', lambda *a:None)
    monkeypatch.setattr(scratch, 'reserve', lambda *a:nullcontext())
    # Real model, bank, startup, 30-epoch training, publication and validators.
    for row in spec['tasks']:
        worker.run(spec, row['task_id'], device='cuda')
    result = worker.results(spec)
    assert not result['pending'] and len(result['rows']) == 28
    assert all(sha256_file(p) == h for p,h in protected.items())
    assert all((root/n).is_file() for n in old.target_payloads())
    assert not list((tmp_path/'scratch').iterdir())
    for row in spec['tasks'][:-1]:
        task = row['task_id']
        assert not (root/task/'receipt.json').exists()  # never impersonate old job receipts
        report = load_json(root/task/'fit/training_report.json')
        assert report['passes'] == 30 and report['node'] == original['nodes'][task]
    with pytest.raises(ValueError): worker.run(spec, 'baseline', device='cuda')
    with pytest.raises(FileExistsError): worker.run(spec, 'teacher_ce', device='cuda')
    assert result == load_json(Path(spec['root'])/'report/results.json')


def test_cli_inventory_bash_and_original_source_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    subprocess.run([sys.executable, 'scripts/luka_depth_recovery.py', '--help'], check=True, capture_output=True)
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if bash.exists():
        for name in ('scripts/queue_luka_depth_recovery.sh','sbatch/run_luka_depth_recovery.sh'):
            subprocess.run([str(bash), '-n', name], check=True, capture_output=True)
    changed = subprocess.run(['git','diff','--name-only',source.BASE,'--','src/hlt_classification/luka_supervision_depth'],
        check=True, capture_output=True, text=True).stdout
    assert changed == ''
