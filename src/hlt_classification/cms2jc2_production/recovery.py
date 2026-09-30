"""Explicit execution-only repair; original scientific study and shards stay frozen."""
import ast
from pathlib import Path
import subprocess

from . import campaign as c, storage
from .contracts import artifact, validate, checked, ref, write, safe, load_json, sha256_file

REPAIR_PATH = 'recovery/repair_spec.json'
PLAN_PATH = 'recovery/recovery_plan.json'
PREFIX = 'src/hlt_classification/cms2jc2_production/'
OPERATIONAL = {
    PREFIX+'campaign.py': {'make_attempt', 'validate_attempt', 'advance', 'require_admission'},
    PREFIX+'submission.py': {'fields', 'worker_identity', 'task_specs', 'plan', 'submit'},
    PREFIX+'worker.py': {'run'},
    PREFIX+'output.py': {'verify_shard', 'finalize', 'read_role'},
}
DOCUMENTS = {'docs/plans/CMS2JC2_PROXY_DATASET_PRODUCTION_PLAN.md',
             'docs/contracts/CMS2JC2_PROXY_DATASET_PRODUCTION.md'}
EXTRA = ('scripts/cms2jc2_proxy_recovery.py', 'scripts/queue_cms2jc2_proxy_recovery.sh',
         'docs/plans/CMS2JC2_PROXY_EXECUTION_RECOVERY_PLAN.md',
         'docs/contracts/CMS2JC2_PROXY_EXECUTION_RECOVERY.md')
ADDED = {PREFIX+'recovery.py', PREFIX+'recovery_controller.py', *EXTRA}
POLICY = dict(cpus=36, max_concurrent=16, memory_gib=[32, 64, 128], max_hours=8,
              resources='unchanged campaign.resources from both retained pilots',
              storage='original frozen budget including failed reservations',
              finalizer='afterany:generate', materialize_only=True, evaluate=False,
              automatic_failure_retry=False, poll_seconds=60, max_wait_days=14)


def active_project():
    return Path(__file__).resolve().parents[3]


def source_snapshot(project, commit):
    source = c.source_snapshot(project, commit)
    files = dict(source['files'])
    for name in EXTRA:
        subprocess.run(['git', '-C', str(project), 'ls-files', '--error-unmatch', name],
                       check=True, capture_output=True)
        files[name] = sha256_file(Path(project)/name)
    return artifact('SOURCE', commit=source['commit'], files=files, executable=True)


def scientific_skeleton(path, functions):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not functions <= names:
        raise ValueError('Missing registered operational function')
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions)]
    return ast.dump(tree, include_attributes=False)


def compatible_source(study, project, source):
    """All old bytes match, except the enumerated execution-only functions/docs."""
    old = study['source']['files']
    new = source['files']
    if set(old)-set(new) or set(new)-set(old) != ADDED:
        raise ValueError('Unexpected replacement source file set')
    changed = []
    for name, digest in old.items():
        if new[name] == digest:
            continue
        changed.append(name)
        if name in DOCUMENTS:
            continue
        if name not in OPERATIONAL or scientific_skeleton(
                Path(study['project_dir'])/name, OPERATIONAL[name]) != scientific_skeleton(
                Path(project)/name, OPERATIONAL[name]):
            raise ValueError('Scientific source change is not authorized by scheduler repair: '+name)
    return dict(changed=sorted(changed), added=sorted(ADDED), scientific_code_unchanged=True)


def load_repair(study, record, *, source=False):
    c.validate_study(study)
    if checked(record).resolve() != safe(study['root'], REPAIR_PATH):
        raise ValueError('Repair must belong to this exact dataset root')
    repair = load_json(checked(record))
    validate(repair, 'EXECUTION_REPAIR', parents={'study': study['content_hash'],
        'original_source': study['source']['content_hash'], 'execution_source': repair['source']['content_hash']}, test=False)
    validate(repair['source'], 'SOURCE', test=False)
    if (load_json(checked(repair['study'])) != study or repair['policy'] != POLICY
            or repair['source']['executable'] is not True
            or repair['source']['commit'] == study['source']['commit']
            or Path(repair['project_dir']).resolve() == Path(study['project_dir']).resolve()
            or repair['environment'] != study['numerical_environment']['content_hash']):
        raise ValueError('Execution repair scope/source differs')
    if (checked(repair['preflight']).resolve() != safe(study['root'], 'preflight.json')
            or load_json(checked(repair['preflight'])) != c.require_preflight(study)):
        raise ValueError('Original preflight changed')
    if len(repair['original_attempts']) != 1:
        raise ValueError('Repair requires the original single pilot attempt')
    for row in repair['original_attempts']:
        attempt = load_json(checked(row['attempt']))
        c.validate_attempt(attempt, study)
        if 'execution_repair' in attempt or attempt['kind'] != 'pilot':
            raise ValueError('Repair cannot recursively replace another repair or bulk')
        directory = safe(study['root'], f'attempts/{attempt["name"]}')
        if (checked(row['attempt']).resolve() != directory/'attempt_spec.json'
                or checked(row['ledger']).resolve() != directory/'submission_ledger.json'):
            raise ValueError('Original attempt path differs')
    pilots = {s['shard_id'] for s in study['shards'] if s['pilot']}
    if (len(pilots) != 2 or len(repair['retained']) != 1 or len(repair['missing_pilots']) != 1
            or set(repair['retained']) | set(repair['missing_pilots']) != pilots
            or set(repair['retained']) & set(repair['missing_pilots'])):
        raise ValueError('Repair must retain one pilot and retry the other')
    for shard, receipt in repair['retained'].items():
        if checked(receipt).resolve() != safe(study['root'], f'shards/{shard}.json'):
            raise ValueError('Retained shard receipt path differs')
        value = load_json(checked(receipt))
        validate(value, 'SHARD', parents={'study': study['content_hash'],
            'population': study['population']['content_hash']}, test=False)
        if value['source'] != study['source']['content_hash']:
            raise ValueError('Retained source differs')
    if source:
        c.validate_study(study, source=True)
        if source_snapshot(repair['project_dir'], repair['source']['commit']) != repair['source']:
            raise ValueError('Replacement execution source changed')
        if compatible_source(study, repair['project_dir'], repair['source']) != repair['compatibility']:
            raise ValueError('Execution-only compatibility changed')
    return repair


def validate_execution(study, record=None):
    if record is None:
        c.validate_study(study, source=True)
        return None
    repair = load_repair(study, record, source=True)
    if active_project() != Path(repair['project_dir']).resolve():
        raise PermissionError('Run recovery with the exact replacement worktree')
    return repair


def execution_project(study, attempt):
    if 'execution_repair' not in attempt:
        return study['project_dir']
    return load_repair(study, attempt['execution_repair'])['project_dir']


def validate_shard_lineage(study, receipt):
    repaired = receipt.get('contract') == 'CMS2JC2_PROXY_SHARD_EXECUTION_REPAIR/v1'
    validate(receipt, 'SHARD_EXECUTION_REPAIR' if repaired else 'SHARD',
        parents={'study': study['content_hash'], 'population': study['population']['content_hash']},
        test=receipt['role'] == 'final_test')
    if not repaired:
        if ('execution_repair' in receipt or 'scientific_source' in receipt
                or receipt['source'] != study['source']['content_hash']):
            raise ValueError('Original shard source/contract differs')
        return
    repair = load_repair(study, receipt['execution_repair'])
    attempt = load_json(safe(study['root'], f'attempts/{receipt["attempt"]}/attempt_spec.json'))
    c.validate_attempt(attempt, study)
    if (attempt.get('execution_repair') != receipt['execution_repair']
            or receipt['shard_id'] not in attempt['shards']
            or receipt['source'] != repair['source']['content_hash']
            or receipt['scientific_source'] != study['source']['content_hash']):
        raise ValueError('Repaired shard execution lineage differs')


def manifest_execution(study, receipts):
    kind = 'DATASET_REDUCED_CONFIRMATION' if c.study_kind(study) == 'STUDY_REDUCED_CONFIRMATION' else 'DATASET'
    records = [r['execution_repair'] for r in receipts if 'execution_repair' in r]
    if not records:
        return kind, {}
    if any(r != records[0] for r in records):
        raise ValueError('Mixed execution repairs')
    repair = load_repair(study, records[0])
    return kind+'_EXECUTION_REPAIR', dict(execution_repair=records[0], execution_source=repair['source'])


def validate_manifest_execution(study, manifest):
    kind = 'DATASET_REDUCED_CONFIRMATION' if c.study_kind(study) == 'STUDY_REDUCED_CONFIRMATION' else 'DATASET'
    if manifest.get('contract') == f'CMS2JC2_PROXY_{kind}_EXECUTION_REPAIR/v1':
        repair = load_repair(study, manifest['execution_repair'])
        if manifest['execution_source'] != repair['source'] or manifest['source'] != study['source']:
            raise ValueError('Manifest execution source differs')
        return kind+'_EXECUTION_REPAIR'
    if 'execution_repair' in manifest or 'execution_source' in manifest:
        raise ValueError('Original manifest cannot carry repaired execution')
    return kind


def original_pilot_evidence(study, attempts, retained):
    from . import submission
    if len(attempts) != 1:
        raise PermissionError('Expected one original pilot attempt')
    attempt = load_json(checked(attempts[0]['attempt']))
    if attempt['kind'] != 'pilot' or 'execution_repair' in attempt or len(attempt['shards']) != 2:
        raise PermissionError('Not the original two-element pilot')
    jobs = load_json(checked(attempts[0]['ledger']))['jobs']
    states = {row['job']: row for row in submission.accounting(list(jobs.values()))}
    preflight = states.get(jobs['preflight'], {})
    if preflight.get('state') != 'COMPLETED' or preflight.get('exit_code') != '0:0':
        raise PermissionError('Original preflight did not complete successfully')
    for index, sid in enumerate(attempt['shards']):
        state = states.get(f'{jobs["generate"]}_{index}', {})
        if sid in retained:
            allocation = retained[sid]['allocation']
            if (state.get('state') != 'COMPLETED' or state.get('exit_code') != '0:0'
                    or retained[sid]['attempt'] != attempt['name']
                    or allocation.get('array_job_id') != jobs['generate'] or allocation.get('index') != index):
                raise PermissionError('Retained pilot does not match successful original allocation')
        elif state.get('state') not in submission.TERMINAL or state['state'] == 'COMPLETED':
            raise PermissionError('Missing pilot must have a terminal unsuccessful allocation')


def create(study_path, *, project_dir, source_commit):
    from . import output, submission
    study = load_json(study_path)
    root = Path(study['root'])
    if Path(study_path).resolve() != safe(root, 'study_spec.json'):
        raise ValueError('Wrong original study path')
    project = Path(project_dir).resolve(strict=True)
    with storage.lock(root, 'repair_preparation'):
        if safe(root, REPAIR_PATH).exists():
            saved = load_repair(study, ref(safe(root, REPAIR_PATH)), source=True)
            if saved['project_dir'] != str(project) or saved['source']['commit'] != source_commit:
                raise ValueError('A different repair is already frozen')
            return read_plan(study)[2]
        print('CMS2JC2-RECOVERY phase=authenticate_original', flush=True)
        c.validate_study(study, source=True)
        submission.require_terminal(study)
        c.require_preflight(study)
        if any(safe(root, p).exists() for p in ('test_build_lock.json', 'pilot_admission.json', 'dataset_manifest.json')):
            raise PermissionError('This scheduler repair is restricted to pre-bulk pilot failure')
        source = source_snapshot(project, source_commit)
        compatibility = compatible_source(study, project, source)
        if c.numerical_environment() != study['numerical_environment']:
            raise ValueError('Recovery must use the pinned Tigris environment')
        done = output.completed(study, physical=True)
        pilots = [s for s in study['shards'] if s['pilot']]
        missing = [s for s in pilots if s['shard_id'] not in done]
        if len(done) != 1 or len(missing) != 1 or any(s not in {p['shard_id'] for p in pilots} for s in done):
            raise PermissionError('Require one completed pilot and one missing pilot')
        original_attempts = [dict(attempt=ref(p), ledger=ref(p.parent/'submission_ledger.json'))
            for p in sorted(safe(root, 'attempts').glob('*/attempt_spec.json'))]
        original_pilot_evidence(study, original_attempts, done)
        repair = artifact('EXECUTION_REPAIR', parents={'study': study['content_hash'],
            'original_source': study['source']['content_hash'], 'execution_source': source['content_hash']},
            study=ref(study_path), preflight=ref(safe(root, 'preflight.json')),
            project_dir=str(project), source=source, environment=study['numerical_environment']['content_hash'],
            compatibility=compatibility, original_attempts=original_attempts,
            retained={sid: ref(safe(root, f'shards/{sid}.json')) for sid in done},
            missing_pilots=[s['shard_id'] for s in missing], policy=POLICY)
        record = write(safe(root, REPAIR_PATH), repair)
        load_repair(study, record, source=True)
        attempt = c.make_attempt(study, 'pilot', missing, memory_gib=64, hours=4,
            bytes_per_jet=max(4096, c.imported(study, 'tigris_gate')['serial']['output_bytes']/64*8),
            execution_repair=record)
        directory = safe(root, f'attempts/{attempt["name"]}')
        plan = artifact('RECOVERY_PLAN', parents={'study': study['content_hash'], 'repair': repair['content_hash']},
            execution_repair=record, initial_attempt=ref(directory/'attempt_spec.json'),
            initial_plan=ref(directory/'command_plan.json'), policy=POLICY,
            authorization_phrase='AUTHORIZE CMS2JC2 EXECUTION RECOVERY AND REMAINING DATASET')
        write(safe(root, PLAN_PATH), plan)
        return plan


def read_plan(study):
    from . import submission
    plan = load_json(safe(study['root'], PLAN_PATH))
    repair = load_repair(study, plan['execution_repair'])
    validate(plan, 'RECOVERY_PLAN', parents={'study': study['content_hash'], 'repair': repair['content_hash']}, test=False)
    if plan['policy'] != POLICY or plan['authorization_phrase'] != 'AUTHORIZE CMS2JC2 EXECUTION RECOVERY AND REMAINING DATASET':
        raise ValueError('Recovery policy differs')
    original, attempt, commands = submission.get_plan(checked(plan['initial_attempt']))
    if (original != study or attempt.get('execution_repair') != plan['execution_repair']
            or attempt['kind'] != 'pilot' or attempt['shards'] != repair['missing_pilots']
            or attempt['resources']['memory_gib'] != 64 or attempt['resources']['hours'] != 4
            or load_json(checked(plan['initial_plan'])) != commands):
        raise ValueError('Recovery initial plan differs')
    return repair, attempt, plan
