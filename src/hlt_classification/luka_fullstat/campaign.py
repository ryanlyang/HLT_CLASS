"""Source-pinned Oscar gate and nine-fit regular scientific registration."""
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.regular_campaign import nodes, tasks
from .contracts import COUNTS, artifact, validate, split_policy
from .recipe import recipe
from . import data, cache

AUTHORIZATION = 'AUTHORIZE LUKA FULLSTAT OSCAR20 EXACT PLAN'


def policy(batch_size=128):
    return dict(counts=COUNTS, training_recipe=recipe(batch_size), nodes=nodes(), tasks=tasks(),
        split_policy=split_policy(), final_test_accessed=False, fusion=False, dense=False,
        automatic_followup=False, execution_site=execution_site('oscar_l40s'),
        cpus=6, memory_mb=180000, max_minutes=2880, cpu_partition='batch',
        preparation_workers=4, preparation_concurrency=4, gate_minutes=480,
        cpu_minutes=1440, expected_files=16, expected_raw_rows=7036584)


def source(project, commit):
    project = Path(project).resolve(strict=True)
    if (re.fullmatch('[0-9a-f]{40}', commit) is None
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat/campaign.py'):
        raise ValueError('FullStat imports/commit escape pinned source')
    value = capture_source_snapshot(project, require_clean=True)
    if value['git_commit'] != commit:
        raise ValueError('Wrong pinned commit')
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                             text=True, capture_output=True).stdout.splitlines()
    required = [f'src/hlt_classification/luka_fullstat/{name}.py' for name in
                ('__init__', 'recipe', 'contracts', 'data', 'cache', 'campaign', 'preflight', 'worker', 'submission')]
    required += ['scripts/luka_fullstat.py', 'scripts/queue_luka_fullstat_oscar.sh',
                 'sbatch/run_luka_fullstat_oscar.sh', 'docs/plans/LUKA_FULLSTAT_OSCAR20_PLAN.md',
                 'docs/contracts/LUKA_FULLSTAT_OSCAR20.md', 'docs/LUKA_FULLSTAT_OSCAR20_RUNBOOK.md']
    if not set(required) <= set(tracked):
        raise ValueError('Commit all FullStat implementation/contracts before execution')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'refs/remotes/origin/main'], check=True)
    return value


def _separate(*paths):
    values = [Path(p).resolve() for p in paths]
    for i, a in enumerate(values):
        for b in values[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError('Source/raw/cache/campaign paths must not overlap')


def create(container, output, cache_root, *, project, commit, acknowledge_row_split=False,
           quota_gib, batch_size=128):
    if acknowledge_row_split is not True:
        raise PermissionError('Explicitly acknowledge row split with unverified event independence')
    if type(quota_gib) is not int or quota_gib <= 0:
        raise ValueError('Declare available scratch quota in GiB, not filesystem-wide free space')
    container, output, cache_root, project = [Path(p).resolve() for p in (container, output, cache_root, project)]
    _separate(container, output, cache_root, project)
    if output.exists() or cache_root.exists():
        raise FileExistsError('Preserve existing campaign/cache; select fresh explicit paths')
    pinned = source(project, commit)
    transfer, records = data.transfer(container)
    value = artifact('GATE', parents=dict(source=pinned['content_hash']), root=str(output),
        cache_root=str(cache_root), container=str(container), project_dir=str(project), source_commit=commit,
        source_snapshot=pinned, transfer=transfer, source_records=records,
        acknowledged_row_split=True, declared_available_cache_bytes=quota_gib*2**30, **policy(batch_size))
    output.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(output/'gate_spec.json', value)
    return value


def validate_gate(gate):
    if gate.get('contract') == 'LUKA_FULLSTAT_RECOVERY_GATE/v1':
        from .recovery import validate_gate as validate_recovery
        return validate_recovery(gate)
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash']))
    expected = policy(gate['training_recipe']['batch_size'])
    if (any(gate.get(k) != v for k, v in expected.items()) or gate['acknowledged_row_split'] is not True
            or type(gate['declared_available_cache_bytes']) is not int or gate['declared_available_cache_bytes'] <= 0):
        raise ValueError('Gate population/recipe/site/acknowledgement differs')
    _separate(gate['container'], gate['root'], gate['cache_root'], gate['project_dir'])
    if load_json(Path(gate['root'])/'gate_spec.json') != gate:
        raise ValueError('Gate locator differs')
    if source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Active source differs')
    if data.transfer(gate['container']) != (gate['transfer'], gate['source_records']):
        raise ValueError('Transferred snapshot metadata differs')


def gate_tasks():
    return [dict(task_id='foundation', kind='foundation', dependencies=[]),
            dict(task_id='prepare', kind='prepare', dependencies=['foundation']),
            dict(task_id='assemble', kind='assemble', dependencies=['prepare']),
            dict(task_id='preflight', kind='preflight', dependencies=['assemble'])]


def validate_foundation(gate, foundation):
    validate(foundation, 'FOUNDATION', dict(gate=gate['content_hash']))
    if (foundation['counts'] != gate['counts'] or foundation['transfer'] != gate['transfer']
            or foundation['raw_entries'] != gate['expected_raw_rows']
            or [{k:r[k] for k in ('path','sha256','category')} for r in foundation['files']] != gate['source_records']):
        raise ValueError('Foundation differs from the exact frozen snapshot/population')


def authenticated_cache(gate):
    origin = foundation_gate(gate)
    result = cache.metadata(Path(origin['root'])/'foundation', gate['cache_root'], gate_hash=origin['content_hash'])
    validate_foundation(origin, result[1])
    return result


def foundation_gate(gate):
    if gate.get('contract') == 'LUKA_FULLSTAT_RECOVERY_GATE/v1':
        return load_json(validate_file_ref(gate['original_gate']))
    return gate


def gate_ledger(gate):
    if gate.get('contract') == 'LUKA_FULLSTAT_RECOVERY_GATE/v1':
        from .recovery import exact_ledger
        return exact_ledger(gate)
    from .submission import render
    from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
    ledger = load_json(Path(gate['root'])/'gate_submission'/'submission_ledger.json')
    validate_submission_ledger(ledger)
    plan = render(gate)
    if (ledger['campaign_spec_sha256'] != gate['content_hash'] or ledger['dry_run']
            or set(ledger['jobs']) != {t['task_id'] for t in gate_tasks()}):
        raise ValueError('Gate submission lineage differs')
    from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
    expected = {r['task_id']: _resolved(r, ledger['jobs']) for r in plan['commands']}
    if ledger['commands'] != expected:
        raise ValueError('Gate submission commands differ')
    return ledger


def accounting(gate):
    if gate.get('contract') == 'LUKA_FULLSTAT_RECOVERY_GATE/v1':
        from .recovery import completed_gate
        return completed_gate(gate)
    ledger = gate_ledger(gate)
    jobs = ledger['jobs']
    requested = [jobs['foundation'], jobs['assemble'], jobs['preflight']]
    requested += [jobs['prepare']+'_'+str(i) for i in range(gate['expected_files'])]
    output = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs.values()),
                             '-o', 'JobID,State%40,ExitCode'], check=True, text=True, capture_output=True).stdout
    states = {}
    for line in output.splitlines():
        parts = line.strip().split('|')
        if len(parts) >= 3:
            states[parts[0]] = parts[1:3]
    if any(states.get(job) != ['COMPLETED', '0:0'] for job in requested):
        raise PermissionError('Every exact gate job/array element must be COMPLETED/0:0')
    return ledger


def admit(gate):
    validate_gate(gate)
    ledger = accounting(gate)
    authenticated = authenticated_cache(gate)
    manifest, foundation, _ = authenticated
    from .preflight import validate_evidence
    evidence_path = Path(gate['root'])/'preflight'/'preflight.json'
    evidence = load_json(evidence_path)
    validate_evidence(evidence, gate, manifest)
    if evidence['job_id'] != ledger['jobs']['preflight']:
        raise ValueError('Preflight report belongs to another job')
    kind = 'RECOVERY_SPEC' if gate['contract'] == 'LUKA_FULLSTAT_RECOVERY_GATE/v1' else 'SPEC'
    value = artifact(kind, parents=dict(gate=gate['content_hash'], cache=manifest['content_hash'], preflight=evidence['content_hash']),
        gate=file_ref(Path(gate['root'])/'gate_spec.json'), root=str(Path(gate['root'])/'science'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'], source_snapshot=gate['source_snapshot'],
        cache_root=gate['cache_root'], foundation_root=str(Path(foundation_gate(gate)['root'])/'foundation'),
        prepared_sha256=manifest['content_hash'], preflight=file_ref(evidence_path),
        train_minutes=evidence['train_minutes'], reduce_minutes=evidence['reduce_minutes'],
        gate_ledger=file_ref(Path(gate['root'])/'gate_submission'/'submission_ledger.json'), **policy(gate['training_recipe']['batch_size']))
    root = Path(value['root'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', value)
    return value


def validate_spec(spec):
    recovered = spec.get('contract') == 'LUKA_FULLSTAT_RECOVERY_SPEC/v1'
    validate(spec, 'RECOVERY_SPEC' if recovered else 'SPEC')
    gate = load_json(validate_file_ref(spec['gate']))
    validate_gate(gate)
    if (recovered != (gate['contract'] == 'LUKA_FULLSTAT_RECOVERY_GATE/v1')
            or spec['root'] != str(Path(gate['root'])/'science')
            or spec['foundation_root'] != str(Path(foundation_gate(gate)['root'])/'foundation')
            or any(spec[k] != gate[k] for k in ('cache_root', 'project_dir', 'source_commit', 'source_snapshot'))
            or any(spec.get(k) != v for k,v in policy(gate['training_recipe']['batch_size']).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Scientific registration differs')
    for field, relative in (('gate_ledger','gate_submission/submission_ledger.json'), ('preflight','preflight/preflight.json')):
        if spec[field]['path'] != str(Path(gate['root'])/relative):
            raise ValueError('Admission evidence locator differs')
    validate_file_ref(spec['gate_ledger'])
    ledger = gate_ledger(gate)
    authenticated = authenticated_cache(gate)
    evidence = load_json(validate_file_ref(spec['preflight']))
    from .preflight import validate_evidence
    validate_evidence(evidence, gate, authenticated[0])
    if (evidence['job_id'] != ledger['jobs']['preflight']
            or spec['parents'] != dict(gate=gate['content_hash'], cache=authenticated[0]['content_hash'], preflight=evidence['content_hash'])
            or spec['prepared_sha256'] != authenticated[0]['content_hash']
            or (spec['train_minutes'], spec['reduce_minutes']) != (evidence['train_minutes'], evidence['reduce_minutes'])):
        raise ValueError('Science admission evidence differs')
    return gate, authenticated, evidence
