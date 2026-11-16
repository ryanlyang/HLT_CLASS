"""Immutable nine-fit registration beside the original scientific worker."""
import importlib
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from .contracts import PARENT_COMMIT, FITS, LANES, artifact, validate, recipe, QUALIFICATION_PROTOCOL

MODULES = ('__init__', 'contracts', 'campaign', 'kernel', 'training', 'resources', 'worker',
    'submission', 'reporting', 'identity', 'optimization', 'decoder', 'readouts', 'storage')
REQUIRED = tuple(f'src/hlt_classification/luka_ema_control_hydra/{n}.py' for n in MODULES) + (
    'scripts/luka_ema_control_hydra.py', 'scripts/queue_luka_ema_control_hydra.sh',
    'sbatch/run_luka_ema_control_hydra.sh', 'tests/test_luka_ema_control_hydra.py',
    'docs/plans/LUKA_EMA_CONTROL_HYDRA_PLAN.md', 'docs/contracts/LUKA_EMA_CONTROL_HYDRA.md',
    'docs/LUKA_EMA_CONTROL_HYDRA_RUNBOOK.md', 'docs/LUKA_EMA_CONTROL_HYDRA_VALIDATION.md')
INHERITED = ('original_spec', 'environment_dir', 'environment', 'counts', 'maximum', 'imported',
    'recovery_anchors', 'embedding_hash', 'foundation_hash')


def hardware(lane):
    if lane != 'a6000':
        raise ValueError('Only single-A6000 execution is registered')
    return dict(world_size=1, cpus=6, memory_mb=32000,
        execution_site=dict(cluster='hydra', partition='gpus', account='default', qos='normal',
            gres='gpu:nvidia_rtx_a6000:1'))


def profile(spec, lane):
    return dict(spec, **hardware(lane))


def registration(lanes=LANES):
    if tuple(lanes) != LANES:
        raise ValueError('Only the A6000 lane is registered')
    return dict(recipe=recipe(), tasks=list(FITS), enabled_lanes=list(LANES),
        qualification_protocol=QUALIFICATION_PROTOCOL, global_batch=128, hardware={'a6000': hardware('a6000')},
        cpu_site=dict(partition='compute', account='default', qos='normal', cpus=1, memory_mb=4000),
        setup_minutes=360, fit_max_minutes=2880, release_minutes=60, report_minutes=60,
        automatic_retry=False, science_dependencies='none',
        resource_policy=dict(safety_multiplier=2., fixed_overhead_seconds=1800., host_fraction=.85, gpu_fraction=.88))


def source(project, commit):
    project = Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or Path(__file__).resolve() != project / REQUIRED[2]:
        raise ValueError('Use this driver from its exact pushed worktree')
    snap = capture_source_snapshot(project, require_clean=True)
    names = subprocess.check_output(['git', '-C', str(project), 'ls-files'], text=True).splitlines()
    if snap['git_commit'] != commit or not set(REQUIRED) <= set(names):
        raise ValueError('Commit the complete EMA control implementation')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
        check=True, capture_output=True)
    return snap


def parent_runtime(parent):
    folder = Path(parent['project_dir']) / 'src/hlt_classification/luka_reconstruction_stability_hydra'
    name = 'ema_control_stability_parent'
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != folder / '__init__.py':
            raise ValueError('Ambiguous parent runtime')
    else:
        loader = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(loader)
        sys.modules[name] = module
        loader.loader.exec_module(module)
    return lambda part: importlib.import_module(name + '.' + part)


def parent(ref, *, full=False):
    validate_file_ref(ref)
    p = load_json(ref['path'])
    if p['source_commit'] != PARENT_COMMIT:
        raise ValueError('Use the original 3ccc35ae stability spec')
    runtime = parent_runtime(p)
    old = runtime('campaign').check(p, full=full, check_storage=False)
    bn = load_json(Path(p['root']) / 'setup_pascal/bn_membership.json')
    runtime('contracts').validate(bn, 'BN_MEMBERSHIP', dict(spec=p['content_hash']))
    return p, old


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = max(0, spec['output_budget_bytes'] - used)
    if used > spec['output_budget_bytes'] or shutil.disk_usage(root if root.exists() else root.parent).free < remaining + 2**30:
        raise OSError('Insufficient remaining EMA output allowance plus one GiB reserve')


def create(root, parent_spec, project, commit, quota_gib):
    ref = file_ref(Path(parent_spec).resolve())
    print('Authenticating frozen inputs and parent source; no new data export is needed.', flush=True)
    p, old = parent(ref, full=True)
    snap = source(project, commit)
    root = Path(root).absolute()
    protected = [Path(p['root']), Path(p['project_dir']), Path(project)] + [Path(old[k]) for k in
        ('root', 'project_dir', 'environment_dir', 'bundle_root', 'raw_root', 'delta_root')]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root or
        any(x.is_symlink() for x in (root, *root.parents)) or
        any(root.is_relative_to(x.resolve()) or x.resolve().is_relative_to(root) for x in protected)):
        raise ValueError('Use a fresh separate EMA output root')
    from .storage import inventory
    footprint = inventory(p['maximum'], p['imported'], p['counts'])
    required = footprint['output_budget_bytes'] // 2**30 + 1
    if type(quota_gib) is not int or quota_gib < required:
        raise ValueError(f'Reserve at least {required} personally available GiB, including one GiB reserve')
    spec = artifact('SPEC', parents=dict(original=old['content_hash'], stability=p['content_hash'],
        source=snap['content_hash'], recipe=recipe()['content_hash']), **registration(),
        **{k: p[k] for k in INHERITED}, root=str(root), project_dir=str(Path(project).resolve()),
        source_commit=commit, source_snapshot=snap, parent_spec=ref,
        reused_membership=file_ref(Path(p['root']) / 'setup_pascal/bn_membership.json'),
        quota_gib=quota_gib, storage_inventory=footprint, output_budget_bytes=footprint['output_budget_bytes'])
    storage(spec)
    root.mkdir()
    write_immutable_json(root / 'campaign_spec.json', spec)
    return spec


def check(spec, *, full=False, check_storage=True):
    validate(spec, 'SPEC')
    p, old = parent(spec['parent_spec'], full=full)
    snap = source(spec['project_dir'], spec['source_commit'])
    validate_file_ref(spec['reused_membership'])
    expected = dict(original=old['content_hash'], stability=p['content_hash'], source=snap['content_hash'], recipe=recipe()['content_hash'])
    if (spec['parents'] != expected or spec['source_snapshot'] != snap or
        any(spec.get(k) != v for k, v in registration().items()) or any(spec[k] != p[k] for k in INHERITED) or
        spec['reused_membership'] != file_ref(Path(p['root']) / 'setup_pascal/bn_membership.json') or
        load_json(Path(spec['root']) / 'campaign_spec.json') != spec):
        raise ValueError('Immutable EMA source/input/registration identity differs')
    if check_storage:
        from .storage import validate_inventory
        validate_inventory(spec['storage_inventory'], spec)
        storage(spec)
    return old
