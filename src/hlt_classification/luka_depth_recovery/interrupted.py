"""Explicit, reviewed restart-zero of the user-cancelled set8_p3 attempt.

No deletion, scheduler mutation, checkpoint salvage, or automatic crash retry.
The entire old task directory is renamed inside the authenticated campaign.
"""
from pathlib import Path
import os
import re
import stat

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_supervision_depth import worker as old
from .contracts import artifact, validate
from . import source, scheduler

TASK = 'set8_p3'
AUTHORIZATION = 'ARCHIVE_CANCELLED_SET8_P3_RESTART_ZERO_V2'
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def checked(path, *, directory):
    """Reject aliases, junctions, special files and hardlinked payloads."""
    path = Path(path)
    info = path.lstat()
    if (path != path.resolve(strict=True) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)):
        raise PermissionError('Noncanonical archive path: ' + str(path))
    return path


def paths(original, jobs):
    root = checked(Path(original['root']), directory=True)
    job = jobs[TASK]
    if not re.fullmatch('[1-9][0-9]*', job):
        raise ValueError('Exact original job ID required')
    home = root/'interrupted_attempts'
    folder = home/(TASK+'-'+job)
    for path in (home, folder):
        if path.exists() or path.is_symlink():
            checked(path, directory=True)
    return root/TASK, folder, root/(TASK+'-'+job+'.out')


def tree(path):
    """Complete relative inventory, including empty dirs; never follow links."""
    root = checked(path, directory=True)
    device = root.stat().st_dev
    rows = []
    def visit(folder):
        for child in sorted(folder.iterdir()):
            info = child.lstat()
            directory = stat.S_ISDIR(info.st_mode)
            checked(child, directory=directory)
            if info.st_dev != device:
                raise PermissionError('Archive may not cross a mount: ' + str(child))
            name = child.relative_to(root).as_posix()
            if directory:
                rows.append(dict(path=name, kind='directory'))
                visit(child)
            else:
                rows.append(dict(kind='file', **file_ref(child, root=root)))
    visit(root)
    return rows


def eligible(original, jobs, states):
    if set(states) != set(jobs) or any(r['state'] not in TERMINAL for r in states.values()):
        raise ValueError('All original jobs must be terminal before archiving')
    row = states[TASK]
    if (row['job_id'] != jobs[TASK] or row['state'] != 'CANCELLED'
            or row['elapsed'] <= 0 or row['start'] in ('Unknown', 'None', 'N/A', '')):
        raise ValueError('Only an explicitly cancelled, started set8_p3 is eligible')
    root = Path(original['root'])
    if (root/'parallel_recovery_claim').exists() or (root/'parallel_recovery_claim').is_symlink():
        raise FileExistsError('Recovery already owns the namespace; preserve its claim')
    if (root/'cleanup/intent.json').exists():
        raise ValueError('Target retirement has started')
    for task in ('targets', 'baseline'):
        if states[task]['state'] != 'COMPLETED' or states[task]['exit_code'] != '0:0':
            raise ValueError('Completed targets and baseline required')
        old.validate_task(original, task)
        old.submission.successful(original, task, jobs[task])


def preview(ref, original, jobs, snapshot, project, commit):
    states = scheduler.accounting(original, jobs)
    eligible(original, jobs, states)
    attempt, folder, log = paths(original, jobs)
    if folder.exists() or folder.is_symlink():
        raise FileExistsError('Archive/intent already exists; inspect it, do not retry')
    files = tree(attempt)
    # A late cancellation may have a complete fit worth salvaging separately.
    if any(r['path'] in ('receipt.json', 'fit/receipt.json', 'fit/selected.pt',
                         'fit/training_report.json') for r in files):
        raise ValueError('Saved completion/checkpoint evidence needs separate review; do not restart')
    checked(log, directory=False)
    return artifact('RESTART_PLAN', parents=dict(original=original['content_hash'], source=snapshot['content_hash']),
        original=ref, original_jobs=jobs, stopped_accounting=states, task=TASK,
        project_dir=str(Path(project).resolve()), source_commit=commit, source_snapshot=snapshot,
        attempt=str(attempt), archive=str(folder/'output'), files=files, log=file_ref(log),
        authorization=AUTHORIZATION, restart_epoch=1, epochs=30, defer_to_lane_end=True,
        delete_payloads=False, reuse_partial_checkpoint=False)


def plan(parent, project, commit, output):
    snapshot = source.current(project, commit)
    ref, original, jobs = source.authenticate(parent, project)
    value = preview(ref, original, jobs, snapshot, project, commit)
    output = Path(output).resolve()
    protected = [Path(original['root']), Path(project).resolve(), Path(original['project_dir']),
                 Path(original['anchor']['evidence']['prepared_root']), Path(original['anchor']['evidence']['raw_container'])]
    if any(output.is_relative_to(p) for p in protected):
        raise PermissionError('Write review metadata outside original/source/data directories')
    write_immutable_json(output, value)
    return value


def execute(path, project, plan_hash, authorization):
    value = load_json(path); validate(value, 'RESTART_PLAN')
    if value['content_hash'] != plan_hash or authorization != AUTHORIZATION:
        raise PermissionError('Exact reviewed restart plan hash and authorization required')
    snapshot = source.current(project, value['source_commit'])
    ref, original, jobs = source.authenticate(value['original']['path'], project)
    if preview(ref, original, jobs, snapshot, project, value['source_commit']) != value:
        raise ValueError('Restart plan is stale; nothing moved')
    attempt, folder, log = paths(original, jobs)
    # Exclusive durable intent precedes the same-filesystem rename. A crash leaves
    # inspectable evidence, never an automatic retry/overwrite of either attempt.
    folder.parent.mkdir(exist_ok=True)
    checked(folder.parent, directory=True)
    folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json', value)
    if scheduler.accounting(original, jobs) != value['stopped_accounting']:
        raise ValueError('Original accounting changed; preserve archive intent')
    if tree(attempt) != value['files']:
        raise ValueError('Partial outputs changed; preserve archive intent')
    checked(log, directory=False); validate_file_ref(value['log'])
    archive = folder/'output'
    if archive.exists() or archive.is_symlink() or attempt.stat().st_dev != folder.stat().st_dev:
        raise FileExistsError('Archive must be absent and on the same filesystem')
    # Both resolved absolute paths have now been checked within the exact old root.
    os.rename(attempt, archive)
    if tree(archive) != value['files']:
        raise ValueError('Archived bytes differ; preserve both paths and intent')
    result = artifact('RESTART', parents=dict(original=original['content_hash'], plan=value['content_hash']),
        plan=file_ref(folder/'intent.json'), task=TASK, job_id=jobs[TASK], archive=str(archive),
        restart_epoch=1, epochs=30, delete_payloads=False)
    write_immutable_json(folder/'receipt.json', result)
    return result


def evidence(original, jobs, states):
    """Authenticate retirement of the failed attempt, not a scientific result."""
    _, folder, log = paths(original, jobs)
    if not folder.exists():
        return {}
    path = checked(folder/'receipt.json', directory=False)
    result = load_json(path); validate(result, 'RESTART')
    intent = checked(folder/'intent.json', directory=False)
    value = load_json(intent)
    validate(value, 'RESTART_PLAN', dict(original=original['content_hash'], source=value['source_snapshot']['content_hash']))
    validate(result, 'RESTART', dict(original=original['content_hash'], plan=value['content_hash']))
    expected = artifact('RESTART', parents=result['parents'], plan=file_ref(intent), task=TASK,
        job_id=jobs[TASK], archive=str(folder/'output'), restart_epoch=1, epochs=30, delete_payloads=False)
    row = states[TASK]
    if (row['state'] != 'CANCELLED' or row['job_id'] != jobs[TASK] or row['elapsed'] <= 0
            or row['start'] in ('Unknown', 'None', 'N/A', '')
            or result != expected or value['original_jobs'] != jobs
            or value['task'] != TASK or value['archive'] != str(folder/'output')
            or value['attempt'] != str(Path(original['root'])/TASK)
            or value['stopped_accounting'][TASK] != states[TASK]
            or value['authorization'] != AUTHORIZATION or value['restart_epoch'] != 1
            or value['epochs'] != 30 or value['defer_to_lane_end'] is not True
            or value['delete_payloads'] is not False or value['reuse_partial_checkpoint'] is not False
            or value['log']['path'] != str(log)):
        raise ValueError('Interrupted-attempt lineage/policy differs')
    if source.original(value['original']) != original:
        raise ValueError('Archive belongs to another campaign')
    if tree(folder/'output') != value['files']:
        raise ValueError('Archived partial output changed')
    checked(log, directory=False); validate_file_ref(value['log'])
    return {TASK: file_ref(path)}
