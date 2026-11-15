"""Retire one completed baseline cache; resume the unchanged pinned Hydra workers.

This module is deliberately stdlib-only until bootstrap selects the ORIGINAL
scientific checkout. The only runtime overrides are storage accounting and the
submission wrapper. Every execution records the separate operational identity.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

BASE = Path('/cs/home/rlyang/hlt_u100_pairs')
BALANCED = BASE/'runs/new_balanced_4a61a34c_r1/campaign_spec.json'
BASELINE = BASE/'runs/new_fullstat_offline_a865e5b6_r1/campaign_spec.json'
CACHE = BASE/'cache_new_fullstat_offline_a865e5b6_r1'
OLD_COMMIT = '4a61a34c97888d3ad5f479e29a5b94f9b6afa22a'
BASELINE_COMMIT = 'a865e5b65bc0271d1359644b15956d98f94caca3'
OLD_HASH = '3b6fdd3e0dab78645fc7630575282bf7aa15c69aee5f66b31283abfcafc9b6d9'
AUTH = 'AUTHORIZE_HYDRA_BASELINE_CACHE_RETIREMENT_AND_BALANCED_RESUME_V1'
CLEANUP_AUTH = 'AUTHORIZE_HYDRA_COMPLETED_BASELINE_CACHE_RETIREMENT_ONLY_V1'
RESUME_AUTH = 'AUTHORIZE_HYDRA_BALANCED_POST_CLEANUP_RESUME_V1'
RETIREMENT_COMMIT = '68dc0cb6eeb0f1f527bbeba11f4753b84789bac7'
MODULE = 'src/hlt_classification/operations/hydra_cache_retirement.py'
CLI = 'scripts/luka_hydra_cache_retirement.py'
WORKER = 'sbatch/run_luka_hydra_cache_recovery.sh'
RESUME_WORKER = 'sbatch/run_luka_hydra_balanced_resume.sh'
LAUNCHER = 'scripts/launch_luka_hydra_cache_recovery.sh'
PREFIX = 'LUKA_HYDRA_CACHE_RETIREMENT_'
FLAGS = dict(final_test_accessed=False, final_test_sealed=True,
             evaluation_scope='development_validation', scientific_recipe_changed=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def record(kind, **fields):
    value = dict(contract=PREFIX+kind+'/v1', schema_version=1, **FLAGS, **fields)
    return dict(value, content_hash=digest(value))


def read(path, kind=None):
    value = json.loads(Path(path).read_bytes())
    body = {k: v for k, v in value.items() if k != 'content_hash'}
    require(value.get('content_hash') == digest(body), 'Metadata hash differs: '+str(path))
    if kind:
        require(value.get('contract') == PREFIX+kind+'/v1' and value.get('schema_version') == 1
            and all(value.get(k) == v for k, v in FLAGS.items()), 'Operational contract differs')
    return value


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
            handle.flush(); os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    require(path == path.resolve() and root == root.resolve(), 'Symlink/noncanonical path: '+str(path))
    require(path.is_relative_to(root) and path != root, 'Path outside protected boundary: '+str(path))
    return path


def file_ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(ref, root=None):
    path = Path(ref['path']) if root is None else Path(root)/ref['path']
    if root is not None:
        safe(path, root)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == ref['bytes']
            and sha(path) == ref['sha256'], 'Payload differs: '+str(path))
    return path


def bootstrap():
    """Never import science from the operational checkout, even accidentally."""
    spec = read(BALANCED)
    require(spec['content_hash'] == OLD_HASH and spec['commit'] == OLD_COMMIT
            and Path(spec['root']) == BALANCED.parent, 'Unexpected balanced campaign')
    old_src = Path(spec['project'])/'src'
    for name, module in list(sys.modules.items()):
        if name == 'hlt_classification' or name.startswith('hlt_classification.'):
            filename = getattr(module, '__file__', None)
            require(filename is None or Path(filename).resolve().is_relative_to(old_src),
                    'Scientific module already imported from another checkout: '+name)
    sys.path.insert(0, str(old_src))
    campaign = importlib.import_module('hlt_classification.luka_new_balanced_hydra.campaign')
    require(Path(campaign.__file__).resolve() == old_src/'hlt_classification/luka_new_balanced_hydra/campaign.py',
            'Wrong scientific runtime')
    campaign.check(spec)
    require(Path(sys.prefix).resolve() == Path(spec['environment_dir']).resolve()
            and campaign.runtime() == spec['environment'], 'Installed environment drift')
    return spec, campaign


def operator_source(project, commit):
    from hlt_classification.provenance import capture_source_snapshot
    project = Path(project).resolve(strict=True)
    require(Path(__file__).resolve() == project/MODULE, 'Wrong operational source locator')
    require(re.fullmatch('[0-9a-f]{40}', commit), 'Full pushed operational commit required')
    snapshot = capture_source_snapshot(project, require_clean=True)
    require(snapshot['git_commit'] == commit, 'Operational source commit differs')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
                   check=True, capture_output=True)
    tracked = subprocess.check_output(['git', '-C', str(project), 'ls-files'], text=True).splitlines()
    require({MODULE, CLI, WORKER, RESUME_WORKER, LAUNCHER, 'tests/test_luka_hydra_cache_retirement.py',
             'docs/LUKA_HYDRA_CACHE_RETIREMENT_RUNBOOK.md',
             'docs/contracts/LUKA_HYDRA_CACHE_RETIREMENT.md'} <= set(tracked), 'Commit all operational files')
    return snapshot


def baseline_evidence(campaign):
    from hlt_classification.provenance import validate_source_snapshot
    from hlt_classification.luka_new_fullstat_hydra import campaign as bc, contracts as c, training
    s = read(BASELINE)
    bc.check(s, live=False)
    require(s['commit'] == BASELINE_COMMIT and Path(s['root']) == BASELINE.parent
            and Path(s['cache']) == CACHE, 'Unexpected baseline/cache')
    validate_source_snapshot(s['source'], repository=s['project'], require_clean=True)
    result = read(BASELINE.parent/'result.json'); c.validate(result, 'RESULT')
    require(result['spec'] == s['content_hash'] and result['job'] == '174643', 'Baseline completion binding differs')
    jobs = []
    for intent in sorted((BASELINE.parent/'submission').glob('attempt_*/*.intent.json')):
        i = read(intent); c.validate(i, 'SUBMISSION_INTENT')
        receipt = read(intent.with_name(intent.name.replace('.intent.json', '.submitted.json')))
        c.validate(receipt, 'SUBMITTED')
        require(receipt['spec'] == i['spec'] == s['content_hash'] and receipt['command'] == i['command'],
                'Baseline submission binding differs')
        row = campaign.terminal(receipt['job'])
        if receipt['task'] == 'train' and receipt['job'] == result['job']:
            require(row[1] == 'COMPLETED' and row[2] == '0:0', 'Baseline train did not complete')
        jobs.append((receipt['task'], receipt['job']))
    require(('train', result['job']) in jobs, 'Missing completed baseline submission')
    refs = {k: file_ref(verify(result[k], BASELINE.parent)) for k in ('checkpoint', 'predictions', 'report')}
    report = read(refs['report']['path']); c.validate(report, 'TRAINING_REPORT')
    rows, _ = training.committed(BASELINE.parent, s['content_hash'], math.ceil(3000000/128))
    require(len(rows) == 20 and report['validation_history'] == rows and report['passes'] == 20
            and report['spec'] == s['content_hash'] and report['validation']['rows'] == 500000
            and report['selected_weights_restored'] is True, 'Incomplete baseline history')
    selected = max(rows, key=lambda row: training.selection_key(row['validation'], row['update']))
    require(report['selected_pass'] == selected['pass_number'] and report['validation'] == selected['validation']
            and report['source_commit'] == s['commit'] and report['environment'] == s['environment']
            and report['recipe'] == c.recipe() and report['node'] == c.node(), 'Baseline selection/recipe differs')
    return s, dict(spec=file_ref(BASELINE), result=file_ref(BASELINE.parent/'result.json'), outputs=refs)


def other_consumers(cache):
    """Refuse direct external cache consumers and newly active original jobs."""
    def references(value):
        if isinstance(value, str): return str(cache).replace('\\', '/') in value.replace('\\', '/')
        if isinstance(value, dict): return any(references(v) for v in value.values())
        if isinstance(value, list): return any(references(v) for v in value)
        return False
    for path in (BASE/'runs').rglob('*spec*.json'):
        if path in (BASELINE, BALANCED) or path.is_relative_to(operation_path().parent):
            continue
        require(not references(json.loads(path.read_bytes())),
                'Another saved specification references the cache: '+str(path))
    ids = subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split()
    for job in ids:
        text = subprocess.check_output(['scontrol', 'show', 'job', job], text=True)
        require(str(cache) not in text and 'code_new_fullstat_a865e5b6' not in text
                and 'new_fullstat_offline_a865e5b6_r1' not in text,
                'Baseline/cache consumer is still active: '+job)


def cache_inventory(s):
    from hlt_classification.luka_new_fullstat_hydra import contracts as c
    require(CACHE.resolve(strict=True) == CACHE, 'Cache symlink')
    m = read(CACHE/'cache.json'); c.validate(m, 'CACHE')
    intent = read(CACHE/'preparation_intent.json'); c.validate(intent, 'PREPARATION_INTENT')
    require(m['parent'] == intent['parent'] == s['foundation_sha256']
            and m['coordinate'] == intent['coordinate'] == 'OFFLINE'
            and m['counts'] == {'train': 3000000, 'validation': 500000}, 'Wrong prepared cache')
    refs = []
    for block in m['blocks']:
        require(re.fullmatch(r'blocks/[0-9]{5}_[0-9]{5}_(train|validation)\.npz', block['path'])
                and block['role'] in ('train', 'validation'), 'Unexpected cache payload name/role')
        safe(CACHE/block['path'], CACHE)
        refs.append({k: block[k] for k in ('path', 'bytes', 'sha256')})
    require(len({r['path'] for r in refs}) == len(refs) and refs, 'Duplicate/empty cache inventory')
    require(all(sum(b['rows'] for b in m['blocks'] if b['role'] == role) == count
                for role, count in m['counts'].items()), 'Cache row coverage differs')
    return refs


def operation_path():
    return BALANCED.parent/'operations/baseline_cache_retirement_v1/spec.json'


def prepare(project, commit, available_gib, campaign, spec):
    require(type(available_gib) is int and available_gib >= 0, 'Declare current free quota in whole GiB')
    path = operation_path()
    if path.exists():
        op = load_operation(path, project, campaign, spec)
        require(op['commit'] == commit and op['available_gib'] == available_gib, 'Existing operation differs')
        return op
    source = operator_source(project, commit)
    tasks, completed = campaign.remaining(spec, True)
    require('OFFLINE' in completed, 'Published balanced OFFLINE must be reusable before cleanup')
    baseline, evidence = baseline_evidence(campaign)
    require(baseline['dataset'] == spec['dataset'], 'Baseline/balanced dataset differs')
    other_consumers(CACHE)
    payloads = cache_inventory(baseline)
    op = record('SPEC', root=str(path.parent), project=str(Path(project).resolve()), commit=commit, source=source,
        parent_spec=file_ref(BALANCED), baseline=evidence, cache=str(CACHE),
        cache_manifest=file_ref(CACHE/'cache.json'), cache_intent=file_ref(CACHE/'preparation_intent.json'),
        dataset=spec['dataset'], foundation=file_ref(Path(spec['dataset'])/'foundation/foundation.json'),
        payloads=payloads, available_gib=available_gib, balanced_completed=completed,
        storage_policy='completed_baseline_zero_future_reserve_keep_balanced_14GiB_v1',
        runtime_overrides=['campaign.storage', 'campaign.commands'], requested_tasks=tasks)
    publish(path, op)
    return op


def load_operation(path, project, campaign, spec):
    require(Path(path).resolve() == operation_path(), 'Noncanonical operation locator')
    op = read(path, 'SPEC')
    require(op['root'] == str(Path(path).parent) and Path(op['project']) == Path(project).resolve()
            and op['cache'] == str(CACHE) and op['dataset'] == spec['dataset']
            and op['runtime_overrides'] == ['campaign.storage', 'campaign.commands']
            and op['storage_policy'] == 'completed_baseline_zero_future_reserve_keep_balanced_14GiB_v1',
            'Unexpected operational scope')
    require(operator_source(project, op['commit']) == op['source'], 'Operational source drift')
    require(verify(op['parent_spec']) == BALANCED and read(BALANCED) == spec, 'Parent spec differs')
    require(verify(op['baseline']['spec']) == BASELINE and Path(read(BASELINE)['cache']) == CACHE,
            'Baseline locator differs')
    require(verify(op['cache_manifest']) == CACHE/'cache.json'
            and verify(op['cache_intent']) == CACHE/'preparation_intent.json', 'Cache metadata differs')
    require(op['payloads'] == cache_inventory(read(BASELINE)), 'Retirement payload scope differs')
    preserved_baseline(op)
    return op


def preserved_baseline(op):
    baseline = read(verify(op['baseline']['spec']))
    result_path = verify(op['baseline']['result'])
    require(result_path == BASELINE.parent/'result.json', 'Baseline result locator differs')
    result = read(result_path)
    require(result['contract'] == 'LUKA_NEW_FULLSTAT_HYDRA_RESULT/v1'
            and result['spec'] == baseline['content_hash'] and result['job'] == '174643',
            'Retired baseline result binding differs')
    require(set(op['baseline']['outputs']) == {'checkpoint', 'predictions', 'report'},
            'Protected baseline output inventory differs')
    for name, ref in op['baseline']['outputs'].items():
        expected = dict(result[name], path=str(safe(BASELINE.parent/result[name]['path'], BASELINE.parent)))
        require(ref == expected, 'Protected output reference differs: '+name)
        verify(ref)


def payload_state(op, missing_allowed=False):
    """Hash only declared regular cache blocks. Never glob-delete or follow links."""
    allowed = {str(safe(CACHE/r['path'], CACHE)) for r in op['payloads']}
    actual = set()
    for p in (CACHE/'blocks').rglob('*'):
        require(not p.is_symlink(), 'Symlink in cache block tree')
        if not p.is_dir(): actual.add(str(p))
    require(actual <= allowed, 'Unregistered file in cache block tree')
    for i, ref in enumerate(op['payloads'], 1):
        path = safe(CACHE/ref['path'], CACHE)
        if not path.exists():
            require(missing_allowed, 'Missing block before retirement intent: '+str(path))
            continue
        st = path.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                and (not hasattr(os, 'getuid') or st.st_uid == os.getuid()), 'Unowned/linked cache payload')
        verify(ref, CACHE)
        if i % 500 == 0: print(f'CACHE RETIREMENT verified {i}/{len(op["payloads"])} blocks', flush=True)


def raw_rebuild_evidence(op):
    foundation = read(verify(op['foundation']))
    require(foundation['content_hash'] == read(BASELINE)['foundation_sha256'], 'Raw foundation differs')
    for i, row in enumerate(foundation['files'], 1):
        path = safe(Path(op['dataset'])/row['path'], op['dataset'])
        require(path.is_file() and sha(path) == row['sha256'], 'Raw rebuild source differs: '+str(path))
        verify(row['membership'], Path(op['dataset'])/'foundation')
        if i % 500 == 0: print(f'RAW REBUILD verified {i}/{len(foundation["files"])} source files', flush=True)


def retired(op):
    root = Path(op['root'])
    intent = read(root/'retirement_intent.json', 'INTENT')
    complete = read(root/'retirement_complete.json', 'COMPLETE')
    require(intent['parents'] == dict(operation=op['content_hash']) and intent['payloads'] == op['payloads']
            and complete['parents'] == dict(operation=op['content_hash'], intent=intent['content_hash'])
            and complete['retired_bytes'] == sum(r['bytes'] for r in op['payloads'])
            and all(complete.get(k) is True for k in ('raw_sources_verified',
                'baseline_outputs_preserved', 'metadata_preserved')), 'Retirement binding differs')
    require(not any((CACHE/r['path']).exists() or (CACHE/r['path']).is_symlink() for r in op['payloads']),
            'Retired cache blocks have reappeared')
    return complete


def retire(op, campaign, spec):
    root = Path(op['root']); marker = root/'retirement_intent.json'
    if (root/'retirement_complete.json').exists(): return retired(op)
    campaign.remaining(spec, True)
    other_consumers(CACHE)
    baseline_evidence(campaign)
    raw_rebuild_evidence(op)
    previous = marker.exists()
    intent = record('INTENT', parents=dict(operation=op['content_hash']), payloads=op['payloads'])
    if previous: require(read(marker, 'INTENT') == intent, 'Existing retirement intent differs')
    payload_state(op, missing_allowed=previous)
    # After authentication, recheck original jobs/consumers immediately before deletion.
    campaign.remaining(spec, True); other_consumers(CACHE)
    preserved_baseline(op)
    if not previous: publish(marker, intent)
    for ref in op['payloads']:
        path = safe(CACHE/ref['path'], CACHE)
        if not path.exists(): continue
        st = path.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'Block changed before retirement')
        verify(ref, CACHE)
        path.unlink()
    result = record('COMPLETE', parents=dict(operation=op['content_hash'], intent=intent['content_hash']),
        retired_bytes=sum(r['bytes'] for r in op['payloads']), raw_sources_verified=True,
        baseline_outputs_preserved=True, metadata_preserved=True)
    publish(root/'retirement_complete.json', result)
    print(f'RETIRED {result["retired_bytes"]/2**30:.2f} GiB cache payload; raw data/models/reports retained.', flush=True)
    return result


def quota(op, spec, campaign, dataset, root, baseline_spec, available_gib, *, resume=False):
    require(str(dataset) == spec['dataset'] and str(root) == spec['root']
            and Path(baseline_spec) == BASELINE and resume is True, 'Storage override outside exact resumed campaign')
    completion = retired(op)
    preserved_baseline(op)
    own = campaign.committed_bytes(root)
    needed = max(0, 14*2**30-own)
    declared = min(available_gib, op['available_gib'])
    free = shutil.disk_usage(root).free
    require(type(declared) is int and declared >= 0, 'Invalid quota declaration')
    if min(free, declared*2**30) < needed:
        raise OSError(f'Need {math.ceil(needed/2**30)} GiB for remaining balanced outputs; baseline is retired')
    return dict(new_results_budget_bytes=14*2**30, natural_baseline=dict(remaining_bytes=0,
        retirement=completion['content_hash'], result=op['baseline']['result']),
        declared_available_gib=declared, filesystem_free_bytes=free, required_free_bytes=needed,
        committed_output_bytes_credited=own, operation=op['content_hash'])


@contextmanager
def execution(op, spec, campaign, *, post_cleanup=False):
    """Explicit operational adapter; original science/source validation stays active."""
    original_storage, original_commands = campaign.storage, campaign.commands
    script = RESUME_WORKER if post_cleanup else WORKER
    locator = resume_path() if post_cleanup else operation_path()
    guard = resume_quota if post_cleanup else quota
    def commands(s, tasks):
        require(s == spec, 'Commands outside registered parent')
        if post_cleanup:
            require(set(tasks) <= set(op['requested_tasks']) and 'train_OFFLINE' not in tasks,
                    'Commands outside registered unfinished stages')
        rows = original_commands(s, tasks)
        return [dict(row, command=row['command'][:-4]+[str(Path(op['project'])/script),
            op['project'], str(locator), row['task']]) for row in rows]
    campaign.storage = lambda *a, **kw: guard(op, spec, campaign, *a, **kw)
    campaign.commands = commands
    try: yield
    finally: campaign.storage, campaign.commands = original_storage, original_commands


def run_task(op, spec, campaign, task, *, post_cleanup=False):
    job = os.environ.get('SLURM_JOB_ID', '')
    binding = campaign.binding(spec, task, job)
    script = RESUME_WORKER if post_cleanup else WORKER
    locator = resume_path() if post_cleanup else operation_path()
    kind = 'RESUME_EXECUTION' if post_cleanup else 'EXECUTION'
    require(binding['command'][-4:] == [str(Path(op['project'])/script), op['project'],
        str(locator), task], 'Job not bound to this operational wrapper')
    if post_cleanup:
        require(task in op['requested_tasks'] and task != 'train_OFFLINE', 'Unregistered recovery task')
        resume_retirement(op)
    else:
        retired(op)
    path = Path(op['root'])/'executions'/f'{job}.json'
    publish(path, record(kind, parents=dict(operation=op['content_hash'], spec=spec['content_hash']),
        job=job, task=task, scientific_project=spec['project'], operational_project=op['project'],
        overrides=op['runtime_overrides']))
    with execution(op, spec, campaign, post_cleanup=post_cleanup):
        from hlt_classification.luka_new_balanced_hydra import worker
        result = worker.run(spec, task)
    publish(path.with_name(job+'.complete.json'), record(kind+'_COMPLETE',
        parents=dict(operation=op['content_hash'], execution=read(path)['content_hash']),
        job=job, task=task, result=file_ref(Path(spec['root'])/task/'result.json')))
    return result


@contextmanager
def operational_lock():
    # Unix advisory lock survives neither crashes nor disconnecting processes;
    # its file remains as evidence. It prevents concurrent explicit continuations.
    import fcntl
    root = operation_path().parent
    safe(root, BALANCED.parent)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root/'execution.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def recover(project, commit, available_gib, authorization, *, continue_existing=False):
    require(authorization == AUTH, 'Explicit cache-retirement and campaign-resume authorization required')
    spec, campaign = bootstrap()
    with operational_lock():
        return recover_locked(project, commit, available_gib, campaign, spec,
                              continue_existing=continue_existing)


def cleanup(project, commit, available_gib, authorization, *, continue_existing=False):
    """Retire the declared cache only; no scheduler review or submission calls."""
    require(authorization == CLEANUP_AUTH, 'Explicit cleanup-only authorization required')
    spec, campaign = bootstrap()
    with operational_lock():
        return cleanup_locked(project, commit, available_gib, campaign, spec,
                              continue_existing=continue_existing)


def cleanup_locked(project, commit, available_gib, campaign, spec, *, continue_existing=False):
    claim = operation_path().parent/'live_claim'
    if continue_existing:
        require(claim.is_dir(), 'No attempted operation to continue')
    else:
        claim.mkdir(exist_ok=False)
    op = prepare(project, commit, available_gib, campaign, spec)
    result = retire(op, campaign, spec)
    free = shutil.disk_usage(BASE).free/2**30
    print(f'CLEANUP_COMPLETE: {result["retired_bytes"]/2**30:.2f} GiB retired; '
          f'filesystem reports {free:.2f} GiB free. No jobs submitted or changed.', flush=True)
    return result


def recover_locked(project, commit, available_gib, campaign, spec, *, continue_existing=False):
    path = operation_path()
    claim = path.parent/'live_claim'
    if continue_existing:
        require(claim.is_dir(), 'No attempted operation to continue')
    else:
        claim.mkdir(exist_ok=False)
    op = prepare(project, commit, available_gib, campaign, spec)
    # Dry resource review of the exact replacement workers precedes any removal.
    tasks, completed = campaign.remaining(spec, True)
    require('OFFLINE' in completed and 'train_OFFLINE' not in tasks,
            'Completed balanced OFFLINE must remain reusable; refusing a refit')
    with execution(op, spec, campaign): campaign.site(campaign.commands(spec, tasks))
    retire(op, campaign, spec)
    with execution(op, spec, campaign):
        plan = campaign.queue(spec, resume=True, available_gib=available_gib)
        require('train_OFFLINE' not in [r['task'] for r in plan['commands']], 'Unexpected OFFLINE refit')
        jobs = campaign.queue(spec, execute=True, authorization=campaign.c.AUTH, resume=True,
                              available_gib=available_gib)
    print(json.dumps(jobs, indent=2), flush=True)
    return jobs


def resume_path():
    return BALANCED.parent/'operations/post_cache_resume_v1/spec.json'


def authenticated_retirement(campaign, spec):
    """Read the completed cleanup using its immutable, independently checked source."""
    from hlt_classification.provenance import validate_source_snapshot
    path = operation_path()
    old = read(path, 'SPEC')
    require(old['commit'] == RETIREMENT_COMMIT
            and old['source']['git_commit'] == RETIREMENT_COMMIT, 'Unexpected cleanup source')
    validate_source_snapshot(old['source'], repository=old['project'], require_clean=True)
    source = Path(old['project'])/MODULE
    loader = importlib.util.spec_from_file_location('frozen_hydra_retirement_68dc0cb6', source)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    require(module.BALANCED == BALANCED and module.CACHE == CACHE, 'Cleanup scope differs')
    checked = module.load_operation(path, old['project'], campaign, spec)
    require(checked == old, 'Cleanup changed during authentication')
    return old, module.retired(old)


def resume_retirement(op):
    """Validate the exact completed receipt without repeating deletion or decoding data."""
    require(verify(op['retirement']['spec']) == operation_path()
            and verify(op['retirement']['complete']) == operation_path().parent/'retirement_complete.json',
            'Cleanup dependency locator differs')
    old = read(operation_path(), 'SPEC')
    require(old['root'] == str(operation_path().parent) and old['cache'] == str(CACHE)
            and old['commit'] == RETIREMENT_COMMIT, 'Cleanup dependency scope differs')
    complete = retired(old)
    require(op['parents']['retirement'] == old['content_hash']
            and op['parents']['retirement_complete'] == complete['content_hash'], 'Cleanup dependency hash differs')
    preserved_baseline(old)
    return old, complete


def resume_quota(op, spec, campaign, dataset, root, baseline_spec, available_gib, *, resume=False):
    require(str(dataset) == spec['dataset'] and str(root) == spec['root']
            and Path(baseline_spec) == BASELINE and resume is True, 'Resume storage scope differs')
    require(type(available_gib) is int and available_gib >= 0
            and type(op['available_gib']) is int and op['available_gib'] >= 0, 'Invalid current quota declaration')
    old, completion = resume_retirement(op)
    own = campaign.committed_bytes(root)
    needed = max(0, 14*2**30-own)
    declared = min(available_gib, op['available_gib'])
    free = shutil.disk_usage(root).free
    if min(free, declared*2**30) < needed:
        raise OSError(f'Need {math.ceil(needed/2**30)} GiB for remaining balanced outputs; current free space is insufficient')
    return dict(new_results_budget_bytes=14*2**30, natural_baseline=dict(remaining_bytes=0,
        retirement=completion['content_hash'], result=old['baseline']['result']),
        declared_available_gib=declared, filesystem_free_bytes=free, required_free_bytes=needed,
        committed_output_bytes_credited=own, operation=op['content_hash'])


def load_resume(path, project, campaign, spec):
    require(Path(path).resolve() == resume_path(), 'Noncanonical resume locator')
    op = read(path, 'RESUME_SPEC')
    require(op['root'] == str(resume_path().parent) and Path(op['project']) == Path(project).resolve()
            and op['runtime_overrides'] == ['campaign.storage', 'campaign.commands']
            and op['storage_policy'] == 'completed_baseline_zero_future_reserve_post_cleanup_quota_v1'
            and type(op['available_gib']) is int and op['available_gib'] >= 0,
            'Unexpected post-cleanup scope')
    require(operator_source(project, op['commit']) == op['source'], 'Recovery source drift')
    require(verify(op['parent_spec']) == BALANCED and read(BALANCED) == spec
            and op['parents']['spec'] == spec['content_hash'], 'Recovery parent differs')
    require(op['requested_tasks'] and len(set(op['requested_tasks'])) == len(op['requested_tasks'])
            and set(op['requested_tasks']) <= set(campaign.c.tasks())-{'train_OFFLINE'}
            and 'OFFLINE' in op['completed'], 'Recovery must reuse completed OFFLINE')
    for stage, saved_hash in op['completed'].items():
        require(stage in campaign.c.STAGES
                and read(BALANCED.parent/('train_'+stage)/'result.json')['content_hash'] == saved_hash,
                'Previously completed stage receipt changed')
    old, completion = authenticated_retirement(campaign, spec)
    require(resume_retirement(op) == (old, completion), 'Retirement authentication differs')
    return op


def prepare_resume(project, commit, available_gib, campaign, spec):
    require(type(available_gib) is int and available_gib >= 0, 'Declare current free quota in whole GiB')
    path = resume_path()
    if path.exists():
        op = load_resume(path, project, campaign, spec)
        require(op['commit'] == commit and op['available_gib'] == available_gib, 'Existing recovery registration differs')
        return op
    source = operator_source(project, commit)
    old, completion = authenticated_retirement(campaign, spec)
    tasks, completed = campaign.remaining(spec, True)
    require('OFFLINE' in completed and 'train_OFFLINE' not in tasks, 'Completed OFFLINE must be reused')
    op = record('RESUME_SPEC', root=str(path.parent), project=str(Path(project).resolve()),
        commit=commit, source=source, parent_spec=file_ref(BALANCED),
        parents=dict(spec=spec['content_hash'], retirement=old['content_hash'], retirement_complete=completion['content_hash']),
        retirement=dict(spec=file_ref(operation_path()),
            complete=file_ref(operation_path().parent/'retirement_complete.json')),
        available_gib=available_gib, completed=completed, requested_tasks=tasks,
        storage_policy='completed_baseline_zero_future_reserve_post_cleanup_quota_v1',
        runtime_overrides=['campaign.storage', 'campaign.commands'])
    resume_quota(op, spec, campaign, spec['dataset'], spec['root'], BASELINE, available_gib, resume=True)
    publish(path, op)
    return op


def resume_campaign(project, commit, available_gib, authorization, *, continue_existing=False):
    require(authorization == RESUME_AUTH, 'Explicit post-cleanup resume authorization required')
    spec, campaign = bootstrap()
    with operational_lock():
        return resume_locked(project, commit, available_gib, campaign, spec,
                             continue_existing=continue_existing)


def resume_locked(project, commit, available_gib, campaign, spec, *, continue_existing=False):
    root = resume_path().parent
    safe(root, BALANCED.parent); root.mkdir(exist_ok=True)
    claim = root/'live_claim'
    if continue_existing:
        require(claim.is_dir(), 'No attempted post-cleanup recovery to continue')
    else:
        claim.mkdir(exist_ok=False)
    op = prepare_resume(project, commit, available_gib, campaign, spec)
    # Original remaining() refuses live/ambiguous attempts before any submission.
    tasks, completed = campaign.remaining(spec, True)
    require(set(tasks) <= set(op['requested_tasks']) and 'OFFLINE' in completed,
            'Remaining stages differ from registered recovery')
    with execution(op, spec, campaign, post_cleanup=True):
        plan = campaign.queue(spec, resume=True, available_gib=available_gib)
        require([r['task'] for r in plan['commands']] == tasks and 'train_OFFLINE' not in tasks,
                'Unexpected recovery plan/refit')
        jobs = campaign.queue(spec, execute=True, authorization=campaign.c.AUTH, resume=True,
                              available_gib=available_gib)
    publish(root/f'submission_{len(list(root.glob("submission_*.json")))+1:03d}.json',
        record('RESUME_SUBMISSION', parents=dict(operation=op['content_hash']), jobs=jobs))
    print(json.dumps(jobs, indent=2), flush=True)
    return jobs
