"""Immutable sidecar registration; the c891da0d campaign stays read-only."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, with_content_hash, validate_content_hash, write_immutable_json,
)
from hlt_classification.jetclass2_delphes import concat_k2_runtime as legacy
from hlt_classification.jetclass2_delphes.concat_k2_campaign import validate_campaign
from hlt_classification.jetclass2_delphes.concat_k2_submit import plan as old_plan, _time_seconds
from hlt_classification.jetclass2_delphes.production import _source
from hlt_classification.jetclass2_delphes.execution import execution_site, slurm_options, allocation
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, _resolved, _journal
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from hlt_classification.jetclass2_delphes.submission import _guarded_exact_submission

DONOR_COMMIT = 'c891da0d45dd3251dea9ea72df975bb96bae3570'
AUTHORIZE = 'AUTHORIZE K2 SEGMENTED DEBUG CONTINUATION EXACT SPEC'
RETIRE = 'CANCEL ONLY REGISTERED PENDING K2 REMAINDER'
NODES = ('CONCAT_K2_D025', 'CONCAT_K2_D000', 'HLT_X1_COMPRESSED')
REMAINING = ('train_CONCAT_K2_D025', 'reduce_CONCAT_K2_D025',
             'train_CONCAT_K2_D000', 'reduce_CONCAT_K2_D000',
             'train_HLT_X1_COMPRESSED', 'aggregate', 'complete')
RESOURCES = dict(segment=dict(cpus=4, memory_mb=320000, minutes=1380, gpu=True),
                 preflight=dict(cpus=4, memory_mb=320000, minutes=360, gpu=True),
                 reduce=dict(cpus=4, memory_mb=320000, minutes=360, gpu=True),
                 metadata=dict(cpus=1, memory_mb=8192, minutes=240, gpu=False))


def artifact(kind, **fields):
    return with_content_hash(dict(contract='K2_SEGMENTED_' + kind + '/v1', schema_version=1, **fields))


def validate(value, kind):
    return validate_content_hash(value, expected_contract='K2_SEGMENTED_' + kind + '/v1', expected_schema_version=1)


def graph():
    rows = [dict(task_id='preflight', kind='preflight', dependencies=[], resource='preflight')]
    parent = 'preflight'
    for node in NODES:
        for segment in (1, 2, 3):
            name = f'train_{node}_part{segment}'
            rows.append(dict(task_id=name, kind='segment', node_id=node, segment=segment,
                             dependencies=[parent], resource='segment'))
            parent = name
        if node != NODES[-1]:
            name = 'reduce_' + node
            rows.append(dict(task_id=name, kind='reduce', node_id=node, dependencies=[parent], resource='reduce'))
            parent = name
    rows.extend([dict(task_id='aggregate', kind='aggregate', dependencies=[parent], resource='metadata'),
                 dict(task_id='complete', kind='complete', dependencies=['aggregate'], resource='metadata')])
    return rows


def original(spec):
    path = Path(spec['source_spec']['path'])
    if sha256_file(path) != spec['source_spec']['sha256']:
        raise ValueError('Original spec bytes changed')
    source = load_json(path)
    validate_campaign(source)
    if (source['source_commit'] != DONOR_COMMIT or source['content_hash'] != spec['source_spec']['content_hash']
            or source.get('experiment_profile') is not None):
        raise ValueError('Only the completed-prefix c891da0d full K2 campaign is supported')
    return source


def source_ledger(source):
    path = Path(source['campaign_root']) / 'submissions_science/submission_ledger.json'
    ledger = load_json(path)
    validate_submission_ledger(ledger)
    expected = old_plan(source, 'science')
    if (ledger['dry_run'] or ledger['campaign_spec_sha256'] != source['content_hash']
            or set(ledger['jobs']) != {r['task_id'] for r in expected['commands']}
            or len(set(ledger['jobs'].values())) != len(ledger['jobs'])):
        raise ValueError('Original science ledger identity differs')
    for row in expected['commands']:
        if ledger['commands'][row['task_id']] != _resolved(row, ledger['jobs']):
            raise ValueError('Original command differs from the pinned DAG')
    return ledger


def verify_prefix(source, receipt_hashes=None):
    legacy.science_gate(source)
    expected = [r['task_id'] for r in source['tasks'] if r['task_id'] not in REMAINING]
    receipts = {}
    for name in expected:
        value = legacy.completed(source, name)
        if value is None:
            raise PermissionError('Required completed source task is missing: ' + name)
        receipts[name] = value['content_hash']
    # Do not silently redo a task that completed after the snapshot.
    for name in REMAINING:
        if (Path(source['campaign_root']) / 'tasks' / (name + '.json')).exists():
            raise PermissionError('Source remainder advanced; register a reviewed new continuation: ' + name)
    if receipt_hashes is not None and receipts != receipt_hashes:
        raise ValueError('Imported completed-prefix receipts changed')
    return receipts


def create(*, source_spec, campaign_root, project_dir, source_commit):
    source_path = Path(source_spec).resolve()
    source = load_json(source_path)
    validate_campaign(source)
    if source['source_commit'] != DONOR_COMMIT:
        raise ValueError('Unexpected original K2 source pin')
    project = Path(project_dir).resolve()
    _source(project, source_commit)
    root = Path(campaign_root).resolve()
    protected = [Path(source[k]).resolve() for k in ('campaign_root', 'project_dir', 'data_root')]
    protected.append(project)
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Continuation root overlaps protected source/data/project')
    if root.exists():
        raise FileExistsError('Use a fresh continuation root')
    ledger = source_ledger(source)
    receipts = verify_prefix(source)
    spec = artifact('CAMPAIGN_SPEC', project_dir=str(project), source_commit=source_commit,
        campaign_root=str(root), source_spec=dict(path=str(source_path), sha256=sha256_file(source_path),
        content_hash=source['content_hash']), original_project_dir=source['project_dir'],
        original_commit=DONOR_COMMIT, source_ledger_sha256=ledger['content_hash'], imported_receipts=receipts,
        old_jobs={name: ledger['jobs'][name] for name in REMAINING}, tasks=graph(), resources=RESOURCES,
        segments_per_fit=3, segment_minutes=1380, reserve_seconds=1800,
        maximum_checkpoint_bytes=32 * 1024**3, minimum_free_bytes=4 * 1024**3,
        initial_partition='debug', allowed_partitions=['debug', 'tier3'],
        training=dict(source['training']), role_counts=source['role_counts'],
        final_test_accessed=False, original_artifacts_read_only=True)
    root.mkdir(parents=True)
    write_immutable_json(root / 'campaign_spec.json', spec)
    for stage in ('full', 'gate', 'science'):
        submit(spec, stage=stage)
    return spec


def validate_spec(spec, *, deep=True):
    validate(spec, 'CAMPAIGN_SPEC')
    source = original(spec)
    _source(Path(spec['project_dir']), spec['source_commit'])
    fixed = dict(tasks=graph(), resources=RESOURCES, segments_per_fit=3, segment_minutes=1380,
        reserve_seconds=1800, maximum_checkpoint_bytes=32 * 1024**3, minimum_free_bytes=4 * 1024**3,
        initial_partition='debug', allowed_partitions=['debug', 'tier3'],
        training=dict(source['training']), role_counts=source['role_counts'],
        final_test_accessed=False, original_artifacts_read_only=True,
        original_project_dir=source['project_dir'], original_commit=DONOR_COMMIT)
    if any(spec.get(k) != v for k, v in fixed.items()):
        raise ValueError('Segmented registration differs')
    ledger = source_ledger(source)
    if (ledger['content_hash'] != spec['source_ledger_sha256']
            or spec['old_jobs'] != {name: ledger['jobs'][name] for name in REMAINING}):
        raise ValueError('Source job mapping differs')
    if deep:
        verify_prefix(source, spec['imported_receipts'])
    return source


def command(spec, name, resource, dependencies=()):
    project, root = Path(spec['project_dir']), Path(spec['campaign_root'])
    cmd = slurm_options(execution_site('sporc_a100_debug')) + [
        f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory_mb']}M",
        f"--time={resource['minutes']}", '--job-name=jc2k2s_' + name,
        '--chdir=' + str(project), '--output=' + str(root / 'slurm-%j.out')]
    if resource['gpu']:
        cmd.append('--gres=gpu:a100:1')
    if dependencies:
        cmd.append('--dependency=afterok:' + ':'.join(dependencies))
    return cmd + [str(project / 'sbatch/run_jetclass2_k2_segmented.sh'), str(project),
        str(root / 'campaign_spec.json'), name]


def plan(spec, stage):
    if stage not in ('full', 'gate', 'science'):
        raise ValueError('Unknown stage')
    rows = [r for r in graph() if stage == 'full' or (r['kind'] == 'preflight') == (stage == 'gate')]
    names = {r['task_id'] for r in rows}
    commands = []
    for row in rows:
        parents = [p for p in row['dependencies'] if p in names]
        commands.append(dict(task_id=row['task_id'], dependencies=parents,
            command=command(spec, row['task_id'], RESOURCES[row['resource']], ['${JOB_' + p + '}' for p in parents])))
    if stage in ('full', 'gate'):
        commands.insert(1, dict(task_id='after_gate', dependencies=['preflight'],
            command=command(spec, 'after_gate', RESOURCES['metadata'], ['${JOB_preflight}'])))
    return artifact('COMMAND_PLAN', campaign_sha256=spec['content_hash'], stage=stage, commands=commands)


def accounting(ids):
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(ids),
        '--format=JobIDRaw,State%40,ExitCode'], check=True, text=True, capture_output=True)
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) >= 3 and parts[0].strip() in ids:
            rows[parts[0].strip()] = (parts[1].strip().split()[0], parts[2].strip())
    return rows


def require_retired(spec):
    states = accounting(list(spec['old_jobs'].values()))
    if any(states.get(job, ('UNKNOWN',))[0] != 'CANCELLED' for job in spec['old_jobs'].values()):
        raise PermissionError('All seven exact old pending jobs must be retired before replacement submission')


def retire(spec, *, execute=False, authorization=None):
    validate_spec(spec)
    ids = list(spec['old_jobs'].values())
    states = accounting(ids)
    if any(states.get(job, ('UNKNOWN',))[0] not in ('PENDING', 'CANCELLED') for job in ids):
        raise PermissionError('Retirement allows only exact PENDING or already CANCELLED source jobs')
    pending = [job for job in ids if states[job][0] == 'PENDING']
    if execute:
        if authorization != RETIRE:
            raise PermissionError('Exact pending-K2 retirement authorization required')
        # State filter protects a job that starts between inspection and cancellation.
        if pending:
            subprocess.run(['scancel', '--state=PENDING', *pending], check=True)
        for _ in range(10):
            try:
                require_retired(spec)
                break
            except PermissionError:
                time.sleep(1)
        require_retired(spec)
        write_immutable_json(Path(spec['campaign_root']) / 'retirement.json', artifact('RETIREMENT',
            campaign_sha256=spec['content_hash'], old_jobs=spec['old_jobs'], state='CANCELLED'))
    return dict(execute=execute, exact_pending_ids=pending, other_jobs_changed=False)


def gate(spec):
    from .runtime import completed
    receipt = completed(spec, 'preflight')
    if receipt is None:
        raise PermissionError('Fresh native resume gate must pass before science')
    evidence = load_json(Path(spec['campaign_root']) / receipt['result']['acceptance'])
    validate(evidence, 'ACCEPTANCE')
    if (evidence['campaign_sha256'] != spec['content_hash'] or not evidence['passed']
            or evidence['final_test_accessed'] or not evidence['uninterrupted_resumed_parity']
            or not evidence['legacy_kernel_parity']):
        raise ValueError('Segmented acceptance differs')
    return evidence


def authorization_record(spec):
    return artifact('AUTHORIZATION', campaign_sha256=spec['content_hash'],
        authorization_phrase=AUTHORIZE, debug_policy_confirmed=True,
        automatic_science_after_native_gate=True, final_test_accessed=False)


def require_authorization(spec):
    path = Path(spec['campaign_root']) / 'authorization.json'
    if not path.is_file() or load_json(path) != authorization_record(spec):
        raise PermissionError('Automatic continuation lacks durable operator authorization')


def submit(spec, *, stage, execute=False, authorization=None, debug_policy_confirmed=False):
    validate_spec(spec)
    root = Path(spec['campaign_root']) / ('submissions_' + stage)
    commands = plan(spec, stage)
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / 'command_plan.json', commands)
    dry = root / 'dry_run_submission_ledger.json'
    dry_result = submit_exact_dag(identity=spec['content_hash'], plan=commands, output=dry,
        canonical_dry_run=dry, execute=False)
    if not execute:
        return dry_result
    if stage == 'full' or authorization != AUTHORIZE:
        raise PermissionError('Use authorized staged submission, never live full-DAG')
    if not debug_policy_confirmed:
        raise PermissionError('Operator must confirm RC permits segmented production on debug')
    require_retired(spec)
    if stage == 'science':
        gate(spec)
    write_immutable_json(Path(spec['campaign_root']) / 'authorization.json', authorization_record(spec))
    full_dry = Path(spec['campaign_root']) / 'submissions_full/dry_run_submission_ledger.json'
    submit_exact_dag(identity=spec['content_hash'], plan=plan(spec, 'full'), output=full_dry,
        canonical_dry_run=full_dry, execute=False)
    claim = root / 'submission_in_progress.claim'
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        # Prevent inherited SBATCH overrides from changing the reviewed resources.
        for key in list(os.environ):
            if key.startswith('SBATCH_'):
                del os.environ[key]
        return _guarded_exact_submission(spec, commands, root)
    finally:
        claim.unlink()


def authenticate_job(spec, name):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real exact Slurm job is required')
    stage = 'gate' if name in ('preflight', 'after_gate') else 'science'
    directory = Path(spec['campaign_root']) / ('submissions_' + stage)
    commands = plan(spec, stage)
    for _ in range(120):
        events, jobs = _journal(directory / 'submission_ledger_journal', identity=spec['content_hash'], plan=commands)
        if name in jobs:
            break
        time.sleep(.25)
    if jobs.get(name) != job:
        raise PermissionError('Worker does not match its exact submission journal')
    resource = RESOURCES['metadata'] if name == 'after_gate' else RESOURCES[next(r['resource'] for r in graph() if r['task_id'] == name)]
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], text=True, capture_output=True, check=True)
    fields = dict(token.split('=', 1) for token in result.stdout.split() if '=' in token)
    if (fields.get('JobId') != job or fields.get('Partition') not in spec['allowed_partitions']
            or fields.get('Account') != 'reu-aisocial' or fields.get('QOS') != 'qos_tier3'
            or fields.get('NumCPUs') != str(resource['cpus']) or fields.get('NumNodes') != '1'
            or fields.get('NumTasks') != '1' or _time_seconds(fields.get('TimeLimit', '')) != resource['minutes'] * 60
            or os.environ.get('SLURM_CLUSTER_NAME') != 'sporc'
            or os.environ.get('SLURM_JOB_PARTITION') != fields['Partition']
            or int(os.environ.get('SLURM_MEM_PER_NODE', 0)) != resource['memory_mb']
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or sys.prefix != '/home/ryreu/miniconda3/envs/atlas_kd_sporc'):
        raise PermissionError('Segmented worker allocation/environment differs')
    if resource['gpu']:
        allocation(execution_site('sporc_a100_debug' if fields['Partition'] == 'debug' else 'sporc_a100'))
    elapsed = _time_seconds(fields['RunTime'])
    evidence = artifact('EXECUTION', campaign_sha256=spec['content_hash'], task_id=name, job_id=job,
        actual_partition=fields['Partition'], resource=resource, nodes=fields.get('NodeList'),
        original_scientific_source=DONOR_COMMIT, executor_source=spec['source_commit'])
    write_immutable_json(Path(spec['campaign_root']) / 'execution' / (name + '.json'), evidence)
    return time.monotonic() + resource['minutes'] * 60 - elapsed
