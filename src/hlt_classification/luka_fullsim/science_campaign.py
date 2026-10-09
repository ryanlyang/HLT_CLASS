"""Fresh controls and asymmetric fusion DAG over the accepted Luka population."""
from pathlib import Path
import hashlib
import subprocess

from hlt_classification.cms_proxy_ladder.campaign import paired_seed, coordinate
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, validate_content_hash
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact
from . import science_admission as a

AUTHORIZATION = 'AUTHORIZE LUKA FULLSIM FUSION SCIENCE EXACT PLAN'


def make_artifact(kind, *, parents=None, **fields):
    return artifact('SCIENCE_'+kind, parents=dict(parents or {}), final_test_accessed=False, **fields)


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract='LUKA_FULLSIM_SCIENCE_'+kind+'/v1', expected_schema_version=1)
    if value.get('final_test_accessed') is not False:
        raise PermissionError('Luka final test is sealed')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Science parent lineage differs')
    return digest


def nodes():
    rows = [('M0HLT', 'D000', None, None, 'CONTROL'), ('OFFLINE', 'OFFLINE', None, None, 'CONTROL'),
        ('U000', 'U000', None, None, 'CONTROL'), ('DIRECT_D000', 'D000', None, 'U000', 'DIRECT'),
        ('FUSION_U050', 'U050', 'U000', 'U000', 'FUSION'),
        ('FUSION_U100', 'U100', 'U050', 'FUSION_U050', 'FUSION'),
        ('FUSION_D066', 'D066', 'U100', 'FUSION_U100', 'FUSION'),
        ('FUSION_D033', 'D033', 'D066', 'FUSION_D066', 'FUSION'),
        ('FUSION_D000', 'D000', 'D033', 'FUSION_D033', 'FUSION'),
        ('FINAL_DIRECT_D000', 'D000', None, 'FUSION_D000', 'FUSION'),
        ('FUSION_D000_D000', 'D000', 'D000', 'FUSION_D000', 'FUSION'),
        ('FINAL_BRIDGE_D000', 'D000', None, 'FUSION_D000_D000', 'FUSION')]
    result = []
    for name, primary, other, teacher, branch in rows:
        u, f = coordinate(primary)
        result.append(dict(node_id=name, coordinate=primary, context_coordinate=other, teacher=teacher, branch=branch,
            u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(primary, 'initialization'), sampler_seed=paired_seed(primary, 'sampler'),
            context_initialization_seed=None if other is None else int.from_bytes(
                hashlib.sha256(('CONTEXT_FUSION/v1/context/'+name).encode()).digest()[:4], 'big'),
            deployable=primary == 'D000' and other in (None, 'D000')))
    return result


def tasks():
    rows, teachers = [], {n['teacher'] for n in nodes()} - {None}
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
    return dict(counts=dict(a.COUNTS), nodes=nodes(), tasks=tasks(), training_recipe=recipe(),
        fits=12, reducers=7, jobs=21, class_balance='unchanged_natural_stage1_proportions',
        validation_policy='entire_50k_checkpoint_selection_and_exploratory_reporting',
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=a.MEMORY_MB,
        maximum_minutes=1440, all_models_cold_start=True, imported_models=[],
        alpha=1., fusion_injections=[2, 4, 6, 8], direction='context_to_primary',
        rolling_resume=False, disk_view_cache=False, automatic_retry=False,
        final_test_capability=False, reference_zero='M0HLT', reference_hundred='OFFLINE')


def check_location(root, project, admission, p, f):
    root = Path(root).resolve()
    protected = (project, admission['prepared_root'], admission['prepared_project'], admission['preflight_project'],
        Path(admission['preflight']['path']).parent, p['foundation_root'], f['input_container'])
    if any(root.is_relative_to(Path(x).resolve()) or Path(x).resolve().is_relative_to(root) for x in protected):
        raise PermissionError('Science root overlaps protected data/source/preparation')


def preflight_accounting(job):
    result = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '-o', 'JobIDRaw,State,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    matches = [r for r in rows if r[0] == str(job)]
    if len(matches) != 1 or matches[0][1:3] != ['COMPLETED', '0:0']:
        raise ValueError('Accepted preflight must be COMPLETED with exit 0:0 in Slurm accounting')
    return dict(job_id=str(job), state='COMPLETED', exit_code='0:0')


def create(*, prepared_root, preflight_path, prepared_project, preflight_project, project, commit, root):
    print('LUKA authenticating preparation, accepted report payloads and three source snapshots...', flush=True)
    active = a.source(project, commit)
    admitted, p, f, r = a.authenticate(prepared_root=prepared_root, preflight_path=preflight_path,
        prepared_project=prepared_project, preflight_project=preflight_project, project=project, active=active)
    root, project = Path(root).resolve(), Path(project).resolve()
    check_location(root, project, admitted, p, f)
    if root.exists():
        raise FileExistsError('Fresh science campaign root required; preserve any existing campaign')
    spec = make_artifact('SPEC', parents=dict(admission=admitted['content_hash'], source=active['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        admission=admitted, preflight_accounting=preflight_accounting(r['job_id']),
        prepared_sha256=p['content_hash'], preflight_sha256=r['content_hash'],
        train_minutes=r['projected_train_minutes'], reduce_minutes=r['projected_reduce_minutes'], **registration())
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec):
    validate(spec, 'SPEC', dict(admission=spec['admission']['content_hash'], source=spec['source_snapshot']['content_hash']))
    if any(spec.get(k) != v for k, v in registration().items()):
        raise ValueError('Frozen scientific registration differs')
    if a.source(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Science source differs')
    old = spec['admission']
    admitted, p, f, r = a.authenticate(prepared_root=old['prepared_root'], preflight_path=old['preflight']['path'],
        prepared_project=old['prepared_project'], preflight_project=old['preflight_project'],
        project=spec['project_dir'], active=spec['source_snapshot'])
    if (admitted != old or spec['prepared_sha256'] != p['content_hash'] or spec['preflight_sha256'] != r['content_hash']
            or spec['train_minutes'] != r['projected_train_minutes'] or spec['reduce_minutes'] != r['projected_reduce_minutes']
            or spec['preflight_accounting'] != dict(job_id=str(r['job_id']), state='COMPLETED', exit_code='0:0')):
        raise ValueError('Accepted inputs/resources/accounting differ')
    check_location(spec['root'], spec['project_dir'], admitted, p, f)
    if load_json(Path(spec['root'])/'campaign_spec.json') != spec:
        raise ValueError('Canonical science specification differs')
    return p, r
