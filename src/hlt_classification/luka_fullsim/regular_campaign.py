"""New ordinary-view campaign; no dependency on deleted historical artifacts."""
from pathlib import Path
import subprocess

from hlt_classification.cms_proxy_ladder.campaign import coordinate, paired_seed
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.execution import execution_site
from .foundation import _source, load_foundation
from .science_campaign import preflight_accounting
from . import stage2 as s

COUNTS = dict(train=100000, validation=50000)
INVENTORY = '0832a577f7bcd0b9a9b719a7f62ef2a8e1b5b24108beec5bd7b3946a09642d17'
SPLITS = '79ac6e577b92a5d2ed00d848c996812b388cab57f9d97d6b519f2bc10060c3b2'
AUTHORIZATION = 'AUTHORIZE LUKA REGULAR EXACT PLAN'
REQUIRED = [f'src/hlt_classification/luka_fullsim/regular_{name}.py'
            for name in ('cache', 'campaign', 'preflight', 'worker', 'submission', 'recovery')]
REQUIRED += ['scripts/luka_fullsim_regular.py', 'scripts/queue_luka_fullsim_regular.sh',
             'sbatch/run_luka_fullsim_regular.sh', 'tests/test_luka_fullsim_regular.py',
             'docs/plans/LUKA_FULLSIM_REGULAR_PLAN.md', 'docs/contracts/LUKA_FULLSIM_REGULAR.md',
             'docs/LUKA_FULLSIM_REGULAR_RUNBOOK.md', 'scripts/queue_luka_fullsim_regular_recovery.sh',
             'tests/test_luka_fullsim_regular_recovery.py']


def make_artifact(kind, *, parents=None, version=1, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSIM_REGULAR_{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), final_test_accessed=False, historical_test_evaluated=True, **fields))


def validate(value, kind, parents=None, *, version=1):
    digest = validate_content_hash(value, expected_contract=f'LUKA_FULLSIM_REGULAR_{kind}/v{version}', expected_schema_version=version)
    if value.get('final_test_accessed') is not False or value.get('historical_test_evaluated') is not True:
        raise PermissionError('New test access forbidden; prior holdout is consumed')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Regular parent lineage differs')
    return digest


def source(project, commit):
    value = _source(Path(project), commit)
    if Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/regular_campaign.py':
        raise ValueError('Regular import escapes pinned project')
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit all new regular implementation/tests/contracts before queueing')
    return value


def conventions(audit_hash):
    return s.conventions(audit_sha256=audit_hash, offline_unit='mm', hlt_unit='mm',
        zero_error='error_only_unavailable', authority='operator_provisional',
        evidence='User agreed both sides provisional mm and error-only zero uncertainty masking; producer units unconfirmed.')


def check_population(inventory, splits):
    if (inventory['content_hash'] != INVENTORY or splits['content_hash'] != SPLITS
            or {r: splits['selected_counts'][r] for r in COUNTS} != COUNTS):
        raise ValueError('Original frozen Luka inventory/split differs; do not create a new split')


def nodes():
    rows = [('M0HLT', 'D000', None, 'CONTROL'), ('OFFLINE', 'OFFLINE', None, 'CONTROL'),
            ('U000', 'U000', None, 'CONTROL'), ('DIRECT_D000', 'D000', 'U000', 'DIRECT'),
            ('COARSE_U050', 'U050', 'U000', 'COARSE'),
            ('COARSE_U100', 'U100', 'COARSE_U050', 'COARSE'),
            ('COARSE_D066', 'D066', 'COARSE_U100', 'COARSE'),
            ('COARSE_D033', 'D033', 'COARSE_D066', 'COARSE'),
            ('COARSE_D000', 'D000', 'COARSE_D033', 'COARSE')]
    result = []
    for name, coord, teacher, branch in rows:
        u, f = coordinate(coord)
        result.append(dict(node_id=name, coordinate=coord, context_coordinate=None, teacher=teacher,
            branch=branch, u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(coord, 'initialization'), sampler_seed=paired_seed(coord, 'sampler'),
            context_initialization_seed=None, deployable=coord == 'D000'))
    return result


def tasks():
    rows, teachers = [], {n['teacher'] for n in nodes()}-{None}
    for node in nodes():
        name = node['node_id']
        rows.append(dict(task_id='train_'+name, kind='train', node_id=name,
                         dependencies=[] if node['teacher'] is None else ['reduce_'+node['teacher']]))
        if name in teachers:
            rows.append(dict(task_id='reduce_'+name, kind='reduce', node_id=name, dependencies=['train_'+name]))
    rows.append(dict(task_id='aggregate', kind='aggregate', node_id=None, dependencies=[r['task_id'] for r in rows]))
    rows.append(dict(task_id='complete', kind='complete', node_id=None, dependencies=['aggregate']))
    return rows


def registration():
    return dict(counts=COUNTS, nodes=nodes(), tasks=tasks(), training_recipe=recipe(), fits=9, reducers=5, jobs=16,
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=90000,
        all_models_cold_start=True, imported_models=[], disk_view_cache=True, automatic_retry=False,
        final_test_capability=False, class_balance='unchanged_natural_stage1_proportions',
        validation_policy='entire_50k_checkpoint_selection_and_exploratory_reporting',
        reference_zero='M0HLT', reference_hundred='OFFLINE')


def gate_policy():
    return dict(counts=COUNTS, inventory_sha256=INVENTORY, split_sha256=SPLITS,
                workers=8, cpu_memory_mb=64000, gpu_memory_mb=90000, gpu_cpus=6,
                gate_minutes=480, foundation_minutes=30, execution_site=execution_site('sporc_a100_debug'),
                automatic_followup=False, science_authorized=False)


def create_gate(*, root, container, project, commit):
    pinned = source(project, commit)
    root, container, project = Path(root).resolve(), Path(container).resolve(strict=True), Path(project).resolve()
    for protected in (container, project):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('Gate root overlaps protected data/source; use a sibling worktree')
    value = make_artifact('GATE', parents=dict(source=pinned['content_hash']), root=str(root),
        input_container=str(container), project_dir=str(project), source_commit=commit,
        source_snapshot=pinned, **gate_policy())
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', value)
    return value


def validate_gate(gate):
    if is_recovery(gate):
        from .regular_recovery import validate_gate as validate_recovery
        return validate_recovery(gate)
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash']))
    if (any(gate.get(k) != v for k, v in gate_policy().items())
            or source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Pinned gate/source/policy differs')
    for protected in (gate['input_container'], gate['project_dir']):
        if Path(gate['root']).resolve().is_relative_to(Path(protected).resolve()) or Path(protected).resolve().is_relative_to(Path(gate['root']).resolve()):
            raise PermissionError('Gate root overlaps protected input')
    return gate


def is_recovery(gate):
    return gate.get('contract') == 'LUKA_FULLSIM_REGULAR_RECOVERY_GATE/v1'


def is_gate(spec):
    return spec.get('contract') == 'LUKA_FULLSIM_REGULAR_GATE/v1' or is_recovery(spec)


def prepared_root(gate):
    return Path(gate['reuse']['prepared_root']) if is_recovery(gate) else Path(gate['root'])/'prepared'


def preparation_gate(gate):
    return load_json(gate['reuse']['original_gate']['path']) if is_recovery(gate) else gate


def evidence(gate):
    from .regular_cache import metadata
    from .regular_preflight import validate_evidence
    validate_gate(gate)
    root = Path(gate['root'])
    original = preparation_gate(gate)
    f, inventory, splits = load_foundation(Path(original['root'])/'foundation')
    check_population(inventory, splits)
    if f['source_snapshot'] != original['source_snapshot'] or f['input_container'] != original['input_container']:
        raise ValueError('Foundation source/container differs')
    manifest, p, _, _ = metadata(prepared_root(gate))
    if (manifest['parents'] != dict(prepared=p['content_hash'], gate=original['content_hash'])
            or p['source_snapshot'] != original['source_snapshot']
            or p['foundation_root'] != str(Path(original['root'])/'foundation')
            or p['conventions'] != conventions(p['audit']['content_hash'])):
        raise ValueError('Prepared source/policy differs')
    r = load_json(root/'preflight/preflight.json')
    validate_evidence(r, gate, p, manifest)
    for ref in r['outputs']:
        validate_file_ref(ref, root=root/'preflight')
    runtime = load_json(root/'preflight/runtime.json')
    if (runtime['job_id'] != r['job_id'] or runtime['environment'] != r['environment']
            or runtime['gpu'] != r['gpu'] or runtime['cublas_workspace_config'] != ':4096:8'):
        raise ValueError('Preflight runtime identity/environment differs')
    return manifest, p, r


def gate_ledger(gate, *, historical=False):
    # Authenticate both the final ledger and its resolved per-job journal.
    ledger = load_json(Path(gate['root'])/'submission/submission_ledger.json')
    from .regular_submission import plan, _plan
    from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
    from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
    events, _ = load_exact_dag_journal(Path(gate['root'])/'submission/submission_ledger_journal',
        identity=gate['content_hash'], plan=_plan(gate) if historical else plan(gate))
    expected = assemble_submission_ledger(campaign_spec_sha256=gate['content_hash'], events=events)
    if ledger != expected or set(ledger['jobs']) != ({'preflight'} if is_recovery(gate) else {'foundation', 'prepare', 'preflight'}):
        raise ValueError('Gate submission journal/IDs differ')
    return ledger


def gate_jobs(gate):
    jobs = dict(gate_ledger(gate)['jobs'])
    if is_recovery(gate):
        jobs.update({k: gate['reuse']['accounting'][k]['job_id'] for k in ('foundation', 'prepare')})
    return jobs


def create_science(gate):
    manifest, p, r = evidence(gate)
    jobs = gate_jobs(gate)
    if jobs['preflight'] != str(r['job_id']):
        raise ValueError('Accepted preflight job differs from gate journal')
    accounting = {name: preflight_accounting(job) for name, job in jobs.items()}
    root = Path(gate['root'])/'science'
    spec = make_artifact('SPEC', parents=dict(gate=gate['content_hash'], cache=manifest['content_hash'], preflight=r['content_hash']),
        root=str(root), project_dir=gate['project_dir'], source_commit=gate['source_commit'], source_snapshot=gate['source_snapshot'],
        gate_path=str(Path(gate['root'])/'gate_spec.json'), gate_accounting=accounting,
        admission=dict(prepared_root=str(prepared_root(gate)), preflight=file_ref(Path(gate['root'])/'preflight/preflight.json')),
        prepared_sha256=p['content_hash'], preflight_sha256=r['content_hash'],
        train_minutes=r['projected_train_minutes'], reduce_minutes=r['projected_reduce_minutes'], **registration())
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec):
    gate = load_json(spec['gate_path'])
    manifest, p, r = evidence(gate)
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], cache=manifest['content_hash'], preflight=r['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items())
            or spec['prepared_sha256'] != p['content_hash'] or spec['preflight_sha256'] != r['content_hash']
            or spec['source_snapshot'] != gate['source_snapshot'] or spec['source_commit'] != gate['source_commit']
            or spec['project_dir'] != gate['project_dir'] or Path(spec['root']) != Path(gate['root'])/'science'
            or spec['train_minutes'] != r['projected_train_minutes'] or spec['reduce_minutes'] != r['projected_reduce_minutes']
            or spec['admission'] != dict(prepared_root=str(prepared_root(gate)), preflight=file_ref(Path(gate['root'])/'preflight/preflight.json'))
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Registered science/source/resources differ')
    if set(spec['gate_accounting']) != {'foundation', 'prepare', 'preflight'} or any(
            row.get('state') != 'COMPLETED' or row.get('exit_code') != '0:0' for row in spec['gate_accounting'].values()):
        raise ValueError('Successful full gate accounting required')
    jobs = gate_jobs(gate)
    if ({k: v['job_id'] for k, v in spec['gate_accounting'].items()} != jobs
            or jobs['preflight'] != str(r['job_id'])):
        raise ValueError('Science gate-accounting identities differ')
    return p, r
