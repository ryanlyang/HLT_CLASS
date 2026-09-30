"""Time-budget amendment, exact retirement, and honest production provenance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from hlt_classification.cms2jc2_response import (
    reduced_confirmation as r, dev_campaign as dev, dev_submission as sub,
    dev_worker, frozen_joint_campaign as full,
)
from hlt_classification.cms2jc2_response.contracts import artifact, with_content_hash, load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
from hlt_classification.cms2jc2_production import contracts as k, campaign as production
from test_cms2jc2_frozen_joint import synthetic_result
from test_cms2jc2_bounded import LocalMeasurement


@pytest.fixture
def amendment(tmp_path, monkeypatch, request):
    original_root = tmp_path/'original'; original_root.mkdir()
    project = tmp_path/'project'; project.mkdir()
    raw = tmp_path/'raw'; raw.mkdir()
    prep = tmp_path/'preparation.json'; k.write(prep, {})
    old_study = dict(project_dir=str(project), root=str(original_root),
        source=dict(files={}, commit='a'*40), imported=dict(cms_root=str(raw), preparation_spec=file_ref(prep)),
        review={}, numerical_environment=artifact('NUMERICAL_ENVIRONMENT', synthetic=True),
        site=dict(partition='debug', account='reu-aisocial', qos='qos_tier3'))
    old_study = artifact('DEV_STUDY', **old_study)
    k.write(original_root/'study_spec.json', old_study)
    files = [dict(path=f'{i}.root', sha256=f'{i+1:064x}', source='early' if i < 4 else 'late',
                  jets=10 if i < 5 else 7) for i in range(6)]
    shards = [dict(index=i, path=files[i//10]['path'], sha256=files[i//10]['sha256'],
                   jets=1, ordered_identities=f'{i+100:064x}') for i in range(57)]
    membership = artifact('FROZEN_MEMBERSHIP', files=files, shards=shards, jets=57)
    original = artifact('FROZEN_STAGE', stage='frozen_confirm', name='frozen_confirm_r1',
        root=str(original_root), study=file_ref(original_root/'study_spec.json'),
        membership=membership, protocol=full.protocol(), reuse=artifact('FROZEN_REUSE'),
        tasks=full.tasks('frozen_confirm', membership), policy='SEARCH')
    original_path = dev.stage_dir(original)/'stage_spec.json'
    k.write(original_path, original)
    old_plan = dev.command_plan(original, old_study)
    k.write(dev.stage_dir(original)/'command_plan.json', old_plan)
    jobids = {}
    for i, command in enumerate(old_plan['commands']):
        task = command['task_id']
        intent = artifact('DEV_SUBMIT_INTENT', parents={'stage': original['content_hash'], 'plan': old_plan['content_hash']},
            task_id=task, argv=sub.argv_for(command, jobids))
        jobids[task] = str(1000+i)
        receipt = artifact('DEV_SUBMIT_RECEIPT', parents={'intent': intent['content_hash']},
            task_id=task, job_id=jobids[task], reconciled=False)
        k.write(sub.journal(original, task)/'intent.json', intent)
        k.write(sub.journal(original, task)/'receipt.json', receipt)
    ledger = artifact('DEV_LEDGER', parents={'stage': original['content_hash'], 'plan': old_plan['content_hash']},
        jobs=jobids, dry_run=False)
    k.write(dev.stage_dir(original)/'submission_ledger.json', ledger)
    needs_statistics = request.node.name in (
        'test_report_uses_actual_saved_statistics_and_never_claims_full_support',
        'test_completed_omitted_shard_is_preserved_but_not_added',
    )
    base = synthetic_result() if needs_statistics else {key: {'file0': {}}
        for key in ('by_file', 'tracking', 'audit')}
    for i in range(36):
        f = shards[i]['sha256']
        payload = {key: {f: deepcopy(base[key]['file0'])} for key in ('by_file', 'tracking', 'audit')}
        row = artifact('FROZEN_SHARD', parents=full.parents(original), **payload,
            jets=1, shard=i, ordered_identities=shards[i]['ordered_identities'], source_groups=[f],
            flags={}, corrections={}, selected='JOINT', reference='B_DZ', replicas=[0, 1, 2], confirmation_accessed=True)
        path = original_root/'reports'/f'{i}.json'; k.write(path, row)
        ref = dict(relative=path.relative_to(original_root).as_posix(), sha256=k.sha256_file(path))
        receipt = artifact('DEV_OUTPUTS', parents={'stage': original['content_hash']},
            owner=f'jf_eval_{i:04d}', outputs={'result': ref})
        k.write(dev.stage_dir(original)/'receipts'/f'jf_eval_{i:04d}.json', receipt)
    status = {job: ('COMPLETED', '0:0') if i < 36 else ('RUNNING' if i < 38 else 'PENDING', '0:0')
              for i, job in enumerate(jobids.values())}
    calls = []
    def scheduler(argv):
        calls.append(argv)
        if argv[0] == 'scancel':
            for job in argv:
                if job in status:
                    status[job] = ('CANCELLED', '0:0')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(sub, 'scheduler', scheduler)
    monkeypatch.setattr(sub, 'states', lambda jobs: {v: status.get(v, ('UNKNOWN', None)) for v in jobs.values()})
    monkeypatch.setattr(sub, 'scheduler_identity', lambda spec, study, task, job, **kwargs:
        dict(JobName='c2jd_'+task, JobState=status[job][0]))
    monkeypatch.setattr(full, 'validate_stage', lambda *a, **kw: old_study)
    monkeypatch.setattr(dev, 'validate_study', lambda *a, **kw: None)
    def create_study(**kwargs):
        new = with_content_hash(dict(old_study, root=str(kwargs['root']), project_dir=str(kwargs['project_dir']),
            site=dict(old_study['site'], partition=kwargs['partition'])))
        Path(new['root']).mkdir()
        k.write(Path(new['root'])/'study_spec.json', new)
        return new
    monkeypatch.setattr(dev, 'create_study', create_study)
    spec = r.create(subject_spec=original_path, project_dir=project, source_commit='a'*40,
        root=tmp_path/'reduced', partition='debug')
    return spec, original, status, calls


def retire(spec):
    plan = load_json(dev.stage_dir(spec)/'command_plan.json')
    return r.retire(spec, execute=True, phrase=r.RETIRE_PHRASE, plan_hash=plan['content_hash'])


def test_exact_cutoff_and_missing_sources(amendment):
    spec, original, _, calls = amendment
    assert spec['coverage']['included_jets'] == 36
    assert spec['coverage']['planned_jets'] == 57
    assert spec['coverage']['missing_sources'] == ['late']
    assert spec['coverage']['included_source_files'] == 4
    assert spec['coverage']['files'][3]['included_jets'] == 6
    assert spec['coverage']['representative_random_sample'] is False
    assert len(spec['tasks']) == 1 and spec['tasks'][0]['cpus'] == 1
    plan = sub.submit(spec)
    assert plan['gpus'] == 0 and plan['cpu_upper_bound'] == 1
    assert calls == []
    changed = deepcopy(original); changed['membership']['shards'].pop()
    with pytest.raises(PermissionError): r.coverage(changed)


def test_dry_retirement_then_exact_pending_and_running_cancel(amendment):
    spec, original, status, calls = amendment
    before = k.sha256_file(dev.stage_dir(original)/'stage_spec.json')
    assert r.retire(spec)['read_only'] and not calls
    with pytest.raises(PermissionError): r.retire(spec, execute=True, phrase='wrong', plan_hash='a'*64)
    result = retire(spec)
    cancel = next(c for c in calls if c[0] == 'scancel')
    assert cancel[4:] == [spec['subject_jobs'][t] for t in r.TARGETS]
    assert all(status[spec['subject_jobs'][f'jf_eval_{i:04d}']] == ('COMPLETED', '0:0') for i in range(36))
    assert all(v['state'] == 'CANCELLED' for v in result['tasks'].values())
    assert k.sha256_file(dev.stage_dir(original)/'stage_spec.json') == before
    retire(spec)
    assert len([c for c in calls if c[0] == 'scancel']) == 1


@pytest.mark.parametrize('fault', ['included_failed', 'unknown_target', 'foreign_target', 'changed_cutoff'])
def test_cancel_refusals_are_nonmutating(amendment, monkeypatch, fault):
    spec, _, status, calls = amendment
    if fault == 'included_failed': status[spec['subject_jobs']['jf_eval_0000']] = ('FAILED', '1:0')
    if fault == 'unknown_target': status.pop(spec['subject_jobs']['jf_eval_0050'])
    if fault == 'foreign_target':
        monkeypatch.setattr(sub, 'scheduler_identity', lambda *a, **kw: dict(JobName='other', JobState='PENDING'))
    if fault == 'changed_cutoff':
        spec = deepcopy(spec); spec['coverage']['included_indices'][-1] = 36; spec = with_content_hash(spec)
    with pytest.raises((ValueError, PermissionError)): retire(spec)
    assert calls == []


def test_completed_omitted_shard_is_preserved_but_not_added(amendment):
    spec, _, status, calls = amendment
    job = spec['subject_jobs']['jf_eval_0036']; status[job] = ('COMPLETED', '0:0')
    receipt = retire(spec)
    assert receipt['tasks']['jf_eval_0036']['state'] == 'COMPLETED'
    assert job not in next(c for c in calls if c[0] == 'scancel')
    assert r.aggregate(spec)[0]['jets'] == 36


@pytest.mark.parametrize('fault', ['bytes', 'identity', 'index', 'file_count'])
def test_included_artifact_corruption(amendment, fault):
    spec, original, _, _ = amendment
    row = load_json(spec['included_outputs'][0]['result']['path'])
    if fault == 'bytes':
        Path(spec['included_outputs'][0]['result']['path']).write_text('{}')
        with pytest.raises(ValueError): r.aggregate(spec)
    else:
        if fault == 'identity': row['ordered_identities'] = 'a'*64
        if fault == 'index': row['shard'] = 36
        if fault == 'file_count': row['source_groups'] = ['b'*64]
        with pytest.raises(ValueError): r.check_shard(original, 0, with_content_hash(row))


def test_report_uses_actual_saved_statistics_and_never_claims_full_support(amendment, monkeypatch):
    spec, original, _, _ = amendment
    retire(spec)
    total, _ = r.aggregate(spec)
    expected = r.metrics.assess(total['by_file'], total['audit'])
    monkeypatch.setattr(dev_worker, 'context', lambda *a: {})
    study = load_json(spec['study']['path'])
    monkeypatch.setattr(dev_worker, 'allocation', lambda *a: dict(job_id='123'))
    monkeypatch.setattr(dev_worker, 'Measurement', LocalMeasurement)
    monkeypatch.setattr(dev_worker, 'numerical_environment', lambda: study['numerical_environment'])
    # Run the same registered dispatcher/publication path used by the scheduled report.
    monkeypatch.setattr(sub, 'scheduler_identity', lambda *a, **kw: {})
    dev_worker.run(spec, 'jr_report')
    row = r.read(spec)
    assert row['decision'] == expected and row['jets'] == 36
    assert row['full_population_status'] == r.FULL_STATUS
    assert row['raw_particles_read'] is False and row['production_qualified'] is False
    assert row['decision_scope'] == 'included_36_shards_only'
    assert row['all_registered_jets_included'] is False
    assert len(row['charts']['JOINT']) > 10
    assert not (dev.stage_dir(original)/'receipts/jf_report.json').exists()
    with pytest.raises(ValueError): r.validate(row, 'FROZEN_REPORT')


def test_report_and_live_submission_require_retirement(amendment):
    spec, _, _, calls = amendment
    with pytest.raises(FileNotFoundError): r.report(spec)
    plan = sub.submit(spec)
    with pytest.raises(FileNotFoundError): sub.submit(spec, execute=True,
        authorization_phrase=dev.PHRASES['frozen_reduced'], reviewed_plan_hash=plan['content_hash'])
    assert not calls


def test_reduced_production_import_requires_explicit_opt_in(monkeypatch):
    spec = dict(contract=r.CONTRACT, stage='frozen_reduced')
    monkeypatch.setattr(r, 'read', lambda *a: {'selected': 'JOINT'})
    monkeypatch.setattr(r, 'donor', lambda *a: {'content_hash': 'a'*64})
    with pytest.raises(PermissionError, match='explicit'): production.confirmation_evidence(spec)
    assert production.confirmation_evidence(spec, allow_reduced=True) == ({'selected': 'JOINT'}, 'a'*64)


def test_helper_syntax_and_dry_default():
    root = Path(__file__).parents[1]
    helper = root/'scripts/queue_cms2jc2_reduced_confirmation.sh'
    checked = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(helper)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    text = helper.read_text()
    assert '[[ $# -eq 4 ]]' in text and 'No jobs changed.' in text
    assert '--plan-hash "$6"' in text and 'scancel' not in text
