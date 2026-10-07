"""CORR_HIGH_TOPO ordinary dataset and six-fit SPORC direct/coarse campaign."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.correlated_topology import dataset as d
from . import correlated as old
from .contracts import artifact, validate, write_json, sha256_file

KIND = d.KIND
SCIENCE_VERSION = 9
GATE_AUTHORIZATION = 'AUTHORIZE CORR HIGH TOPO 100K 50K SPORC DEBUG GATE'
SCIENCE_AUTHORIZATION = 'AUTHORIZE CORR HIGH TOPO 100K 50K DIRECT COARSE SCIENCE'
SOURCE_FILES = ('scripts/jetclass2_correlated_topology.py',
    'scripts/queue_jetclass2_correlated_topology.sh', 'tests/test_correlated_topology.py',
    'docs/plans/JETCLASS2_CORR_HIGH_TOPO_PLAN.md', 'docs/contracts/JETCLASS2_CORR_HIGH_TOPO.md')


def source_lock(project, commit):
    from .gate import source_lock as base
    root = Path(project).resolve(strict=True)
    lock = base(root, commit, literature=True, correlated=True)
    files = dict(lock['files'])
    for path in [root/p for p in SOURCE_FILES] + sorted((root/'src/hlt_classification/correlated_topology').glob('*.py')):
        relative = path.relative_to(root).as_posix()
        subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', relative], check=True, capture_output=True)
        files[relative] = sha256_file(path)
    return artifact('SOURCE', commit=commit, files=files)


def gate_tasks():
    return [dict(task_id='authenticate_release', kind='cpu', dependencies=[], cpus=6, memory_mb=90000, minutes=480),
        dict(task_id='build_foundation', kind='cpu', dependencies=['authenticate_release'], cpus=6, memory_mb=90000, minutes=480),
        dict(task_id='preflight', kind='gpu', dependencies=['build_foundation'], cpus=6, memory_mb=90000, minutes=480)]


def create_gate(*, original_release, noise_spec, gate_root, project_dir, source_commit, available_quota_gib, persistent):
    import math
    if (persistent is not True or type(available_quota_gib) not in (int, float)
            or not math.isfinite(available_quota_gib) or available_quota_gib < 10):
        raise PermissionError('Attest persistent destination with at least 10 GiB available quota')
    root, project = Path(gate_root).resolve(), Path(project_dir).resolve(strict=True)
    if root.exists():
        raise FileExistsError('Fresh topology gate required')
    request = d.request(original_release, noise_spec)
    for protected in (project, Path(request['study_root']), Path(request['offline_root']),
                      Path(original_release).resolve().parent, Path(noise_spec).resolve().parent):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('Gate overlaps protected input/worktree')
    source = source_lock(project, source_commit)
    spec = artifact('GATE_SPEC', version=12,
        parents=dict(source=source['content_hash'], request=request['content_hash']), source=source,
        request=request, gate_root=str(root), project_dir=str(project), source_commit=source_commit,
        capacity=512, execution_site=execution_site('sporc_a100_debug'), tasks=gate_tasks(),
        workers=6, foundation_workers=6, scientific_branches=['DIRECT', 'COARSE'], dataset_kind=KIND,
        full_views_persisted=False, site_transfer_policy=None,
        storage=dict(persistent=True, available_quota_gib=available_quota_gib),
        admission='ordinary_materialization_replay_full_Weaver_resource_preflight')
    validate_gate(spec, check_source=True)
    root.mkdir(parents=True)
    write_json(root/'gate_spec.json', spec)
    return spec


def validate_gate(spec, *, check_source=False):
    digest = validate(spec, 'GATE_SPEC', version=12, parents=dict(
        source=spec['source']['content_hash'], request=spec['request']['content_hash']))
    validate(spec['source'], 'SOURCE')
    d.validate_request(spec['request'])
    root = Path(spec['gate_root']).resolve()
    for protected in (Path(spec['project_dir']), Path(spec['request']['study_root']),
                      Path(spec['request']['offline_root']),
                      Path(spec['request']['original_release']['path']).parent,
                      Path(spec['request']['noise_spec']['path']).parent):
        protected = protected.resolve()
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('Gate overlaps protected input/worktree')
    if (spec['source_commit'] != spec['source']['commit'] or spec['execution_site'] != execution_site('sporc_a100_debug')
            or spec['tasks'] != gate_tasks() or spec['workers'] != 6 or spec['foundation_workers'] != 6
            or spec['capacity'] != 512 or spec['dataset_kind'] != KIND or spec['scientific_branches'] != ['DIRECT', 'COARSE']
            or spec['full_views_persisted'] is not False or spec['site_transfer_policy'] is not None
            or 'measurement_site' in spec or spec['storage']['persistent'] is not True
            or not 10 <= spec['storage']['available_quota_gib'] < float('inf')
            or spec['admission'] != 'ordinary_materialization_replay_full_Weaver_resource_preflight'):
        raise ValueError('Topology gate differs')
    if check_source and source_lock(Path(spec['project_dir']), spec['source_commit']) != spec['source']:
        raise ValueError('Topology source drift')
    return digest


def scientific_plan(foundation, *, foundation_root=None):
    return old.scientific_plan(foundation, foundation_root=foundation_root,
                               _version=9, _foundation_version=7, _prefix='CORRHT')


def validate_profile(profile, *, foundation, spec):
    return old.validate_profile(profile, foundation=foundation, spec=spec, _version=12, _plan_builder=scientific_plan)


def create_campaign(**kwargs):
    return old.create_campaign(**kwargs, _adapter=sys.modules[__name__])


def validate_campaign(spec, *, check_source=False):
    return old.validate_campaign(spec, check_source=check_source, _adapter=sys.modules[__name__])
