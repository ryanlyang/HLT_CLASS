"""Reuse completed work; require reviewed archive evidence for interrupted fits."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_supervision_depth import worker as old_worker, submission as old_submission
from hlt_classification.luka_embedding_weight.worker import storage
from .contracts import artifact, validate, schedule, ALL_FITS
from . import source, scheduler, interrupted


def inventory(original, jobs):
    states = scheduler.accounting(original, jobs)
    active = {t: r for t, r in states.items() if r['state'] in scheduler.ACTIVE}
    if active:
        raise ValueError('Leave running originals to finish before creating recovery: ' + repr(active))
    if (Path(original['root'])/'cleanup/intent.json').exists():
        raise ValueError('Shared target retirement has started; do not regenerate silently')
    restarted = interrupted.evidence(original, jobs, states)
    reused = {}; seen = {}
    for task in ('targets', *[t['task_id'] for t in original['tasks'] if t['task_id'] in ALL_FITS]):
        state = states[task]
        folder = Path(original['root'])/task
        if state['state'] == 'COMPLETED' and state['exit_code'] == '0:0':
            old_worker.validate_task(original, task, seen)
            old_submission.successful(original, task, jobs[task])
            reused[task] = file_ref(folder/'receipt.json')
        elif (task not in ('targets', 'baseline') and state['state'] == 'CANCELLED'
                and state['elapsed'] == 0 and state['start'] in ('Unknown', 'None', 'N/A', '')
                and not folder.exists() and not folder.is_symlink()):
            continue
        elif (task in restarted and state['state'] == 'CANCELLED' and state['elapsed'] > 0
                and state['start'] not in ('Unknown', 'None', 'N/A', '')
                and not folder.exists() and not folder.is_symlink()):
            continue
        else:
            raise ValueError('Not completed/reusable or approved for restart: ' + task
                + '; preserve partial output; interrupted set8_p3 needs reviewed restart-plan/restart')
    schedule(reused, restarted)
    return states, reused, restarted


def create(root, parent, project, commit, *, available_quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    ref, original, jobs = source.authenticate(parent, project)
    oldroot = Path(original['root'])
    protected = [oldroot, project, Path(original['project_dir']), Path(original['anchor']['evidence']['prepared_root']),
        Path(original['anchor']['evidence']['raw_container'])]
    if oldroot != oldroot.resolve(strict=True) or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Recovery metadata must be separate from source, data and original campaign')
    if type(available_quota_gib) != int or available_quota_gib < 8:
        raise ValueError('Reserve eight personally available GiB for the remaining fits')
    states, reused, restarted = inventory(original, jobs)
    storage(original)
    value = artifact('SPEC', parents=dict(original=original['content_hash'], source=snapshot['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        original=ref, original_jobs=jobs, stopped_accounting=states, reused=reused, restarted=restarted,
        tasks=schedule(reused, restarted), available_quota_gib=available_quota_gib,
        scientific_identity='original_spec_unchanged', max_gpu_jobs=2, preflight_jobs=0,
        shared_targets_reused_in_place=True, automatic_retry=False, borrowed_payload_cleanup=False)
    # One recovery owner for the old output namespace, even across new commits.
    claim = oldroot/'parallel_recovery_claim'; claim.mkdir(exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('CLAIM', parents=dict(recovery=value['content_hash']),
        root=str(root), original=ref))
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root/'campaign_spec.json', value)
    return value


def check(value, *, live=True):
    original = source.original(value['original'])
    validate(value, 'SPEC', dict(original=original['content_hash'], source=value['source_snapshot']['content_hash']))
    if (value['tasks'] != schedule(value['reused'], value['restarted']) or value['max_gpu_jobs'] != 2
            or value['preflight_jobs'] != 0 or value['borrowed_payload_cleanup'] is not False
            or value['shared_targets_reused_in_place'] is not True or value['automatic_retry'] is not False
            or value['scientific_identity'] != 'original_spec_unchanged'
            or type(value['available_quota_gib']) != int or value['available_quota_gib'] < 8
            or load_json(Path(value['root'])/'campaign_spec.json') != value):
        raise ValueError('Recovery policy differs')
    if interrupted.evidence(original, value['original_jobs'], value['stopped_accounting']) != value['restarted']:
        raise ValueError('Reviewed interrupted-attempt evidence differs')
    claim = load_json(Path(original['root'])/'parallel_recovery_claim/intent.json')
    expected = artifact('CLAIM', parents=dict(recovery=value['content_hash']), root=value['root'], original=value['original'])
    if claim != expected or old_submission.accepted_jobs(original) != value['original_jobs']:
        raise ValueError('Original namespace owner/journal differs')
    for task, ref in value['reused'].items():
        if Path(ref['path']) != Path(original['root'])/task/'receipt.json':
            raise ValueError('Reused receipt path differs')
        validate_file_ref(ref)
    if live:
        if source.current(value['project_dir'], value['source_commit']) != value['source_snapshot']:
            raise ValueError('Recovery source changed')
        ref, _, jobs = source.authenticate(value['original']['path'], value['project_dir'])
        if ref != value['original'] or jobs != value['original_jobs']:
            raise ValueError('Original source changed')
        now = scheduler.accounting(original, jobs)
        if any(r['state'] in scheduler.ACTIVE for r in now.values()):
            raise ValueError('Original campaign unexpectedly active')
        for task in ALL_FITS | {'targets'}:
            prior = value['stopped_accounting'][task]
            if now[task] != prior:
                raise ValueError('Original job accounting changed: ' + task)
    return original
