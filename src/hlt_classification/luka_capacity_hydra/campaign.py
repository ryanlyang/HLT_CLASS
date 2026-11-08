"""Immutable, additive registration of twelve fresh capacity fits."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from .contracts import BASE, COUNTS, FITS, artifact, validate, recipe, cell
from . import reuse

MODULES = ('__init__', 'contracts', 'campaign', 'reuse', 'inputs', 'model', 'training',
           'resources', 'worker', 'submission', 'reporting', 'complementarity')
RUNTIME = {f'src/hlt_classification/luka_capacity_hydra/{n}.py' for n in MODULES} | {
    'scripts/luka_capacity_hydra.py', 'scripts/queue_luka_capacity_hydra.sh', 'sbatch/run_luka_capacity_hydra.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_capacity_hydra.py', 'docs/plans/LUKA_CAPACITY_HYDRA_PLAN.md',
    'docs/contracts/LUKA_CAPACITY_HYDRA.md', 'docs/LUKA_CAPACITY_HYDRA_RUNBOOK.md', 'docs/LUKA_CAPACITY_HYDRA_VALIDATION.md'}
COPIED = ('bundle_root', 'reference', 'imported', 'anchor', 'embedding_hash', 'qualification', 'recovery_anchors')


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if (not REQUIRED <= set(git('ls-files')) or any(r[:2] != 'A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_capacity_hydra/campaign.py'):
        raise ValueError('Commit complete additive capacity campaign; preserve donor runtime')
    return snapshot


def registration():
    return dict(recipe=recipe(), tasks=list(FITS), counts=COUNTS, execution_site=A6000_SITE,
        gpu_profile='a6000', cpus=6, memory_mb=32000, setup_minutes=720, fit_max_minutes=2880,
        analysis_minutes=120, release_minutes=60, report_minutes=60, original50k_rows=50000,
        cpu_site=dict(partition='compute', account='default', qos='normal', cpus=1, memory_mb=4000),
        storage_budget_bytes=8*2**30, resource_policy=dict(safety_multiplier=2., fixed_overhead_seconds=900, memory_fraction=.85),
        science_dependencies='none', teacher_fits=0, student_fits=12, automatic_retry=False)


def nodes(old):
    base = old['imported']['node']
    return {t: dict(base, node_id=t, coordinate='D000', deployable=True, teacher=None,
                   initialization_seed=base['initialization_seed']+cell(t)['seed']*recipe()['seed_offset'],
                   sampler_seed=base['sampler_seed']+cell(t)['seed']*recipe()['seed_offset'],
                   capacity=cell(t)) for t in FITS}


def create(root, selective_spec, project, commit, quota_gib):
    project = Path(project).resolve(); root = Path(root).absolute()
    if type(quota_gib) is not int or quota_gib < 9:
        raise ValueError('Reserve eight GiB outputs plus one GiB headroom')
    source = current(project, commit)
    print('Authenticating completed selective parent in its pinned worktree...', flush=True)
    old = reuse.original(selective_spec)
    protected = [Path(old[k]).resolve() for k in ('root', 'project_dir', 'environment_dir', 'bundle_root')] + [project]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root
            or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate capacity output root')
    site.environment(old['reference']['environment'])
    if Path(sys.prefix).resolve() != Path(old['environment_dir']).resolve():
        raise ValueError('Use the original pinned Brown environment')
    value = artifact('SPEC', parents=dict(source=source['content_hash'], selective=old['content_hash'], recipe=recipe()['content_hash']),
        **registration(), root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()), selective_spec=file_ref(Path(selective_spec).resolve()),
        parent_root=old['root'], parent_hash=old['content_hash'], reused=reuse.references(old),
        available_quota_gib=quota_gib, nodes=nodes(old), **{k: old[k] for k in COPIED})
    storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json', value)
    return value


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], selective=spec['parent_hash'], recipe=recipe()['content_hash']))
    if any(spec.get(k) != v for k, v in registration().items()):
        raise ValueError('Registration changed')
    validate_file_ref(spec['selective_spec']); old = reuse.original(spec['selective_spec']['path'])
    if (old['content_hash'] != spec['parent_hash'] or old['root'] != spec['parent_root']
            or any(spec[k] != old[k] for k in COPIED) or spec['nodes'] != nodes(old)
            or spec['reused'] != reuse.references(old) or spec['available_quota_gib'] < 9
            or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable campaign/source/parent changed')
    site.environment(spec['reference']['environment'])
    if check_storage:
        storage(spec)
