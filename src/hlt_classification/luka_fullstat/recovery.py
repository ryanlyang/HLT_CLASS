"""Explicit sparse recovery and fail-closed Oscar science continuation.

The original gate/cache are never relabelled. A new gate and fresh GPU acceptance
bind the reused foundation and thirteen completed preparation receipts.
"""
from pathlib import Path
import getpass
import json
import re
import subprocess
import time

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal, _resolved
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from .contracts import artifact, validate
from . import campaign as c, cache, data, submission as s

PRODUCER = '592a6dace8adc662903d3da0d409bcab5fefb9a7'
PRIOR = 'bf04bb89a95d928faf008d899481ace05a69a5f3'
ORIGINAL_JOBS = dict(foundation='7486667', prepare='7486668', assemble='7486669', preflight='7486670')
RETRY = [12, 13, 14]
REUSED = [i for i in range(16) if i not in RETRY]
EXCLUDED_NODE = 'node1804'
AUTHORIZE = 'AUTHORIZE FULLSTAT SPARSE RECOVERY AND AUTOMATIC SCIENCE'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}
# Only orchestration changes are allowed relative to the reviewed parent. The
# parent contains the separate trial-30 runner dispatch; it receives NEW GPU
# acceptance here. Every other tracked source, especially preparation, is pinned.
OPERATIONAL = {
    'src/hlt_classification/luka_fullstat/'+n+'.py'
    for n in ('campaign', 'submission', 'worker', 'recovery')
} | {'scripts/luka_fullstat.py', 'scripts/luka_fullstat_recovery.py',
     'scripts/queue_luka_fullstat_recovery.sh'}


def git(project, *args):
    return subprocess.run(['git', '-C', str(project), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def compatibility(project):
    """Authenticate producer->reviewed parent->recovery source, not mtime."""
    git(project, 'merge-base', '--is-ancestor', PRIOR, 'HEAD')
    changed = set(git(project, 'diff', '--name-only', PRIOR, 'HEAD', '--', 'src', 'scripts', 'sbatch').splitlines())
    if not changed <= OPERATIONAL:
        raise ValueError('Recovery source has unreviewed runtime changes: '+repr(sorted(changed-OPERATIONAL)))
    # Explicitly verify the durable cache producers in addition to the full-tree
    # parent check. No changed split, views, features or assignment code is allowed.
    paths = ['src/hlt_classification/'+p for p in (
        'luka_fullstat/data.py', 'luka_fullstat/cache.py', 'luka_fullstat/contracts.py',
        'luka_fullstat/recipe.py', 'luka_fullsim/particles.py', 'luka_fullsim/stage2.py',
        'luka_fullsim/preflight_cache.py', 'literature_context', 'cms_proxy_ladder',
        'data/cache_contracts.py', 'jetclass2_delphes/cache.py', 'jetclass2_delphes/schema.py')]
    if git(project, 'diff', '--name-only', PRODUCER, 'HEAD', '--', *paths):
        raise ValueError('Frozen preparation semantics changed; cached files cannot be reused')
    return dict(producer=PRODUCER, reviewed_parent=PRIOR, operational_changes=sorted(changed))


def exact_ledger(spec):
    plan = s.render(spec)
    directory = Path(spec['root'])/('gate_submission' if 'GATE/' in spec['contract'] else 'submission')
    if load_json(directory/'command_plan.json') != plan:
        raise ValueError('Registered command plan differs')
    events, jobs = load_exact_dag_journal(directory/'submission_ledger_journal',
        identity=spec['content_hash'], plan=plan)
    if len(events) != len(plan['commands']) or len(set(jobs.values())) != len(jobs):
        raise ValueError('Incomplete/duplicate exact submission journal; do not retry blindly')
    expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'], jobs=jobs,
        commands={r['task_id']: _resolved(r, jobs) for r in plan['commands']}, dry_run=False)
    if load_json(directory/'submission_ledger.json') != expected:
        raise ValueError('Submission ledger/journal differs')
    return expected


def original(gate_path):
    path = Path(gate_path).resolve(strict=True)
    value = load_json(path)
    validate(value, 'GATE', dict(source=value['source_snapshot']['content_hash']))
    if (path != Path(value['root'])/'gate_spec.json' or value['source_commit'] != PRODUCER
            or value['acknowledged_row_split'] is not True
            or any(value.get(k) != v for k,v in c.policy(128).items())
            or capture_source_snapshot(value['project_dir'], require_clean=True) != value['source_snapshot']
            or data.transfer(value['container']) != (value['transfer'], value['source_records'])):
        raise ValueError('Original source, population, recipe or transfer metadata differs')
    ledger = exact_ledger(value)
    if ledger['jobs'] != ORIGINAL_JOBS:
        raise ValueError('This recovery is registered only for the reviewed 7486668 gate')
    return value, ledger


def expectations(jobs, indices, *, foundation=False):
    value = {jobs[k]: 'lukafs_'+k for k in ('assemble', 'preflight')}
    if foundation:
        value[jobs['foundation']] = 'lukafs_foundation'
    value.update({jobs['prepare']+'_'+str(i): 'lukafs_prepare' for i in indices})
    return value


def read_states(text, expected, user):
    states = {}
    for line in text.splitlines():
        fields = line.strip().rstrip('|').split('|')
        if not line.strip() or fields[0] not in expected:
            continue  # sacct can also return the array parent; never count it.
        if len(fields) != 6 or fields[0] in states:
            raise ValueError('Duplicate/malformed scheduler accounting')
        job, name, owner, account, state, code = fields
        if (name != expected[job] or owner != user or account != 'default'):
            raise PermissionError('Scheduler identity differs: '+job)
        states[job] = (state.split()[0].rstrip('+'), code)
    return states


def accounting(expected):
    output = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(expected),
        '-o', 'JobID,JobName%100,User,Account,State%40,ExitCode'], check=True,
        capture_output=True, text=True, env=s.environment()).stdout
    return read_states(output, expected, getpass.getuser())


def check_original():
    expected = expectations(ORIGINAL_JOBS, range(16), foundation=True)
    states = accounting(expected)
    cancelled = {ORIGINAL_JOBS[k] for k in ('assemble', 'preflight')} | {
        ORIGINAL_JOBS['prepare']+'_'+str(i) for i in RETRY}
    for job in expected:
        wanted = ('CANCELLED', '0:0') if job in cancelled else ('COMPLETED', '0:0')
        if states.get(job) != wanted:
            raise PermissionError('Original gate is active, incomplete or changed: '+job)
    return states


def missing_outputs(origin):
    root = Path(origin['cache_root'])
    for i in RETRY:
        for path in (root/'files'/f'{i:02d}', root/'chunks'/f'f{i:02d}'):
            if path.exists() or path.is_symlink():
                raise FileExistsError('Preserve partial/unexpected retry output: '+str(path))
    for path in (root/'cache_manifest.json', Path(origin['root'])/'preflight', Path(origin['root'])/'science'):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Original downstream output exists; inspect before recovery: '+str(path))


def claim_original(gate):
    """One recovery writer per original campaign, even across recovery roots."""
    origin = c.foundation_gate(gate)
    directory = Path(origin['root'])/'fullstat_recovery_claim'
    value = artifact('RECOVERY_CLAIM', parents=dict(original_gate=origin['content_hash'],
        recovery_gate=gate['content_hash']), recovery_root=gate['root'])
    if directory.is_symlink():
        raise PermissionError('Recovery claim cannot be a symlink')
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        if not (directory/'intent.json').is_file() or load_json(directory/'intent.json') != value:
            raise PermissionError('Another/ambiguous recovery owns this original campaign; preserve claim')
    else:
        write_immutable_json(directory/'intent.json', value)


def create(gate_path, output, *, project, commit):
    project, output = Path(project).resolve(), Path(output).resolve()
    snapshot = c.source(project, commit)
    compatible = compatibility(project)
    origin, ledger = original(gate_path)
    c._separate(output, origin['root'], origin['cache_root'], origin['container'], project)
    if output.exists():
        raise FileExistsError('Recovery root exists; review it, do not recreate it')
    check_original()
    missing_outputs(origin)
    foundation, members = data.load(Path(origin['root'])/'foundation', expected_gate=origin['content_hash'])
    c.validate_foundation(origin, foundation)
    reused = []
    for i in REUSED:
        path = Path(origin['cache_root'])/'files'/f'{i:02d}'/'receipt.json'
        row = load_json(path)
        if row['file_index'] != i:
            raise ValueError('Reused receipt index differs')
        cache.file_metadata(row, foundation, members[i])
        reused.append(file_ref(path))
    value = artifact('RECOVERY_GATE', parents=dict(source=snapshot['content_hash'],
        original_gate=origin['content_hash'], original_ledger=ledger['content_hash'], foundation=foundation['content_hash']),
        root=str(output), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        original_gate=file_ref(Path(origin['root'])/'gate_spec.json'),
        original_ledger=file_ref(Path(origin['root'])/'gate_submission/submission_ledger.json'),
        foundation=file_ref(Path(origin['root'])/'foundation/foundation.json'),
        cache_root=origin['cache_root'], container=origin['container'], compatibility=compatible,
        retry_indices=RETRY, reused_receipts=reused, excluded_node=EXCLUDED_NODE,
        **{**c.policy(128), 'automatic_followup': True})
    output.mkdir(parents=True, exist_ok=False)
    write_immutable_json(output/'gate_spec.json', value)
    return value


def validate_gate(gate):
    validate(gate, 'RECOVERY_GATE')
    origin, ledger = original(validate_file_ref(gate['original_gate']))
    if (c.source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or compatibility(gate['project_dir']) != gate['compatibility']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate
            or any(gate.get(k) != v for k,v in {**c.policy(128), 'automatic_followup': True}.items())
            or gate['cache_root'] != origin['cache_root'] or gate['container'] != origin['container']
            or gate['retry_indices'] != RETRY or gate['excluded_node'] != EXCLUDED_NODE):
        raise ValueError('Recovery source/policy/locator differs')
    c._separate(gate['root'], origin['root'], gate['cache_root'], gate['container'], gate['project_dir'])
    refs = [('original_ledger', Path(origin['root'])/'gate_submission/submission_ledger.json'),
            ('foundation', Path(origin['root'])/'foundation/foundation.json')]
    for key, path in refs:
        if Path(validate_file_ref(gate[key])) != path:
            raise ValueError('Recovery evidence locator differs')
    foundation = load_json(validate_file_ref(gate['foundation']))
    c.validate_foundation(origin, foundation)
    expected = dict(source=gate['source_snapshot']['content_hash'], original_gate=origin['content_hash'],
                    original_ledger=ledger['content_hash'], foundation=foundation['content_hash'])
    if gate['parents'] != expected or len(gate['reused_receipts']) != len(REUSED):
        raise ValueError('Recovery parent/receipt coverage differs')
    for i, ref in zip(REUSED, gate['reused_receipts']):
        path = validate_file_ref(ref)
        if Path(path) != Path(gate['cache_root'])/'files'/f'{i:02d}'/'receipt.json':
            raise ValueError('Reused preparation locator differs')
        row = load_json(path)
        validate(row, 'FILE_PREPARED', dict(foundation=foundation['content_hash']))
        if row['file_index'] != i or row['final_test_accessed'] is not False:
            raise ValueError('Reused preparation identity differs')
    return origin


def render(gate):
    # Reuse the exact resource renderer; no synthetic gate is ever published.
    plan = s.render({**gate, 'contract': 'LUKA_FULLSTAT_GATE/v1'})
    commands = []
    for row in plan['commands'][1:]:
        row = dict(row, command=list(row['command']))
        if row['task_id'] == 'prepare':
            row['dependencies'] = []
            row['command'] = [a for a in row['command'] if not a.startswith(('--array=', '--dependency='))]
            row['command'][1:1] = ['--array=12-14%3', '--exclude='+EXCLUDED_NODE]
        commands.append(row)
    return artifact('RECOVERY_PLAN', parents=dict(spec=gate['content_hash']), commands=commands,
        mode='gate', jobs=3, scheduled_tasks=5, automatic_followup=True, authorization=c.AUTHORIZATION,
        final_test_accessed=False)


def gate_task(gate, task, file_index):
    origin = validate_gate(gate)
    if task == 'prepare':
        if type(file_index) is not int or file_index not in RETRY:
            raise PermissionError('Only the three registered missing files may be prepared')
        result = cache.prepare_file(origin['container'], Path(origin['root'])/'foundation',
            origin['cache_root'], file_index, workers=gate['preparation_workers'])
    elif task == 'assemble':
        # Full physical authentication of ALL reused and replacement payloads.
        result = cache.assemble(Path(origin['root'])/'foundation', origin['cache_root'], gate_hash=origin['content_hash'])
    elif task == 'preflight':
        from .preflight import run
        result = run(gate)
    else:
        raise PermissionError('No foundation rebuild or test task in recovery')
    validate_gate(gate)
    return result


def gate_ready(states, expected):
    for job in expected:
        state = states.get(job)
        if state is not None and state != ('COMPLETED', '0:0') and state[0] not in ACTIVE:
            raise PermissionError(f'Gate failed: {job} {state}; NO science submitted')
    return all(states.get(job) == ('COMPLETED', '0:0') for job in expected)


def completed_gate(gate):
    check_original()
    ledger = exact_ledger(gate)
    expected = expectations(ledger['jobs'], RETRY)
    if not gate_ready(accounting(expected), expected):
        raise PermissionError('Every replacement array element, assembly AND GPU preflight must pass')
    return ledger


def scope(gate):
    return dict(counts=gate['counts'], training_recipe=gate['training_recipe'], nodes=c.nodes(), tasks=c.tasks(),
        jobs=16, fits=9, reducers=5, fusion=False, dense=False, final_test_accessed=False,
        site=gate['execution_site'], cpus=6, memory_mb=180000, max_minutes=2880)


def review(gate):
    validate_gate(gate)
    value = s.submit(gate)  # Complete dry gate, scheduler resource probes only.
    # Full future DAG at the authorized ceiling. Real requests are determined by
    # acceptance, never silently enlarged beyond this reviewed template.
    future = dict(gate, contract='LUKA_FULLSTAT_RECOVERY_SPEC/v1', root=str(Path(gate['root'])/'science'),
                  train_minutes=2880, reduce_minutes=2880)
    template = s.render(future)
    s.check_site(template)
    value = artifact('FOLLOWUP_REVIEW', parents=dict(gate=gate['content_hash'], gate_plan=value['content_hash']),
        scope=scope(gate), recovery_plan=value, science_template=template, placeholder_science_identity=True,
        authorization=AUTHORIZE, final_test_accessed=False)
    write_immutable_json(Path(gate['root'])/'followup_review.json', value)
    return value


def verify_science(gate, spec, plan, reviewed):
    c.validate_spec(spec)
    if (scope(gate) != reviewed['scope'] or plan != s.render(spec)
            or spec['parents']['gate'] != gate['content_hash'] or plan['jobs'] != 16
            or not 0 < spec['train_minutes'] <= 2880 or not 0 < spec['reduce_minutes'] <= 2880):
        raise PermissionError('Measured science plan exceeds authorized scope')
    # Require equality of every argument except the explicitly measured time
    # and the science identity in the comment. Paths, task IDs and deps are exact.
    template = reviewed['science_template']['commands']
    for actual, bound in zip(plan['commands'], template):
        def normalize(row):
            return dict(row, command=[re.sub(r'lukafs:[0-9a-f]{64}:', 'lukafs:IDENTITY:', a)
                for a in row['command'] if not a.startswith('--time=')])
        if normalize(actual) != normalize(bound):
            raise PermissionError('Actual science DAG differs from the reviewed template')


def run(gate, *, review_hash, authorization, timeout_seconds=14*86400, poll_seconds=60):
    from hlt_classification.literature_ladder_followup import lock
    validate_gate(gate)
    root = Path(gate['root'])
    reviewed = load_json(root/'followup_review.json')
    validate(reviewed, 'FOLLOWUP_REVIEW')
    # Recompute the entire review before allowing any external write.
    if reviewed != review(gate) or review_hash != reviewed['content_hash'] or authorization != AUTHORIZE:
        raise PermissionError('Exact reviewed recovery + automatic-science authorization required')
    with lock(root/'followup'):
        write_immutable_json(root/'followup/authorization.json', artifact('FOLLOWUP_AUTHORIZATION',
            parents=dict(review=review_hash, gate=gate['content_hash']), authorization=AUTHORIZE, scope=scope(gate)))
        live = root/'gate_submission/submission_ledger.json'
        if not live.exists():
            check_original()
            missing_outputs(c.foundation_gate(gate))
        claim_original(gate)
        plan = s.render(gate)
        ledger = s.submit(gate, execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
        if ledger != exact_ledger(gate):
            raise ValueError('Recovery submission journal differs')
        print('ARMED: recovery queued; all 16 science jobs will submit ONLY after every replacement prepare, assembly and GPU preflight passes.', flush=True)
        print('Recovery jobs: '+json.dumps(ledger['jobs'], sort_keys=True), flush=True)
        expected = expectations(ledger['jobs'], RETRY)
        deadline = time.monotonic()+timeout_seconds
        while True:
            states = accounting(expected)
            print('Gate accounting: '+json.dumps(states, sort_keys=True), flush=True)
            if gate_ready(states, expected):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Gate wait expired; NO science submitted')
            time.sleep(min(60, poll_seconds))
        validate_gate(gate)
        completed_gate(gate)
        path = root/'science/campaign_spec.json'
        spec = load_json(path) if path.exists() else c.admit(gate)
        science = s.submit(spec)
        verify_science(gate, spec, science, reviewed)
        write_immutable_json(root/'followup/science_review.json', artifact('FOLLOWUP_SCIENCE_REVIEW',
            parents=dict(review=review_hash, spec=spec['content_hash'], plan=science['content_hash']), plan=science))
        # Repeat accounting immediately before external submission. A partial
        # sbatch reply is never automatically retried (exclusive live_claim).
        completed_gate(gate)
        ledger = s.submit(spec, execute=True, plan_hash=science['content_hash'], authorization=c.AUTHORIZATION)
        if ledger != exact_ledger(spec):
            raise ValueError('Science journal differs')
        write_immutable_json(root/'followup/submitted.json', artifact('FOLLOWUP_SUBMITTED',
            parents=dict(review=review_hash, ledger=ledger['content_hash']), jobs=ledger['jobs'], training_complete=False))
        print('ALL 16 SCIENCE JOBS QUEUED: '+json.dumps(ledger['jobs'], sort_keys=True), flush=True)
        return ledger
