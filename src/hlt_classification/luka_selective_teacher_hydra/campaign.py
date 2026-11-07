"""New immutable Hydra execution; all historical sources/artifacts stay pinned."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from .reuse import original, references
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_next_supervision_hydra import reuse
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import BASE, COUNTS, FITS, TEACHERS, STUDENTS, recipe, artifact, validate
from .transfer import selected

MODULES = ('__init__', 'contracts', 'campaign', 'transfer', 'inputs', 'targets', 'model',
           'training', 'resources', 'submission', 'worker', 'reporting', 'site', 'views', 'reuse', 'validation_cache')
RUNTIME = {f'src/hlt_classification/luka_selective_teacher_hydra/{n}.py' for n in MODULES} | {
    'scripts/luka_selective_teacher_hydra.py', 'scripts/queue_luka_selective_teacher_hydra.sh', 'sbatch/run_luka_selective_teacher_hydra.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_selective_teacher_hydra.py', 'docs/plans/LUKA_SELECTIVE_TEACHER_HYDRA_PLAN.md',
    'docs/contracts/LUKA_SELECTIVE_TEACHER_HYDRA.md', 'docs/LUKA_SELECTIVE_TEACHER_HYDRA_RUNBOOK.md', 'docs/LUKA_SELECTIVE_TEACHER_HYDRA_VALIDATION.md'}
COPIED = ('bundle_root', 'reference', 'imported', 'anchor', 'embedding_hash', 'qualification')


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if (not REQUIRED <= set(git('ls-files')) or any(r[:2] != 'A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_selective_teacher_hydra/campaign.py'):
        raise ValueError('Commit the complete additive selective campaign; preserve donor runtime')
    return snapshot


def registration():
    return dict(recipe=recipe(), tasks=list(FITS), counts=COUNTS, execution_site=A6000_SITE,
        gpu_profile='a6000', cpus=6, memory_mb=32000, setup_minutes=720, fit_max_minutes=2880,
        release_minutes=60, report_minutes=60, original50k_rows=50000,
        cpu_site=dict(partition='compute', account='default', qos='normal', cpus=1, memory_mb=4000),
        storage_budget_bytes=8*2**30, resource_policy=dict(safety_multiplier=2., fixed_overhead_seconds=900, memory_fraction=.85),
        science_dependencies='student_after_required_teachers_only', teacher_fits=4, student_fits=8,
        automatic_retry=False, reference_retrained=False)


def nodes(old):
    original_node = Bundle(old['bundle_root']).json('reference/OFFLINE/report.json')['node']
    result = {t: dict(original_node, node_id=t, coordinate=t, teacher=None, context_coordinate=None,
        deployable=False, initialization_seed=original_node['initialization_seed']+200000,
        sampler_seed=original_node['sampler_seed']+200000) for t in TEACHERS}
    result.update({s: dict(old['imported']['node'], node_id=s) for s in STUDENTS})
    return result


def anchors(old):
    bundle = Bundle(old['bundle_root']); result = {}
    for n in ('M0HLT', 'OFFLINE'):
        a = load_npz_arrays(bundle.path(f'reference/{n}/predictions.npz'))
        result[n] = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    return result


def validation_source(old, root=None, digest=None):
    if (root is None) != (digest is None): raise ValueError('Provide both validation root/hash or neither')
    if root is not None: return 'selective_export', str(Path(root).resolve(strict=True)), digest
    # original() has already authenticated the teacher-sweep and its Group A
    # parent using their pinned implementations. Bind its exact file again.
    validate_file_ref(old['group_a_spec'])
    group_a = load_json(old['group_a_spec']['path'])
    return 'existing_frozen_adapter', group_a['delta_root'], group_a['delta_hash']


def create(root, teacher_sweep_spec, validation_root, validation_hash, project, commit, quota_gib):
    old = original(teacher_sweep_spec); project = Path(project).resolve(); root = Path(root).absolute()
    if type(quota_gib) is not int or quota_gib < 9:
        raise ValueError('Reserve eight GiB outputs plus one GiB headroom after transfers/environment')
    kind, validation_root, validation_hash = validation_source(old, validation_root, validation_hash)
    d = selected(kind, validation_root, validation_hash, old['bundle_root']); d.verify_all()
    protected = [Path(old[k]).resolve() for k in ('root', 'project_dir', 'environment_dir', 'bundle_root')]+[project, d.root]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate selective-teacher root')
    source = current(project, commit); site.environment(old['reference']['environment'])
    reuse.check(old['control'], imported=old['imported'])
    value = artifact('SPEC', parents=dict(source=source['content_hash'], teacher_sweep=old['content_hash'], recipe=recipe()['content_hash'], validation=validation_hash),
        **registration(), root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()), teacher_sweep_spec=file_ref(Path(teacher_sweep_spec).resolve()),
        validation_root=str(d.root), validation_hash=validation_hash, validation_kind=kind, available_quota_gib=quota_gib,
        control=old['control'], reused=references(old), nodes=nodes(old), recovery_anchors=anchors(old), **{k: old[k] for k in COPIED})
    storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json', value)
    return value


def check(spec, *, check_storage=True):
    validate_file_ref(spec['teacher_sweep_spec']); old = original(spec['teacher_sweep_spec']['path'])
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], teacher_sweep=old['content_hash'], recipe=recipe()['content_hash'], validation=spec['validation_hash']))
    if spec['validation_kind'] == 'existing_frozen_adapter':
        kind, root, digest = validation_source(old)
        if spec['validation_root'] != str(Path(root).resolve(strict=True)) or spec['validation_hash'] != digest:
            raise ValueError('Existing validation source differs from original Group A')
    if (any(spec.get(k) != v for k, v in registration().items()) or any(spec[k] != old[k] for k in COPIED)
            or spec['reused'] != references(old) or spec['control'] != old['control'] or spec['nodes'] != nodes(old) or spec['recovery_anchors'] != anchors(old)
            or spec['available_quota_gib'] < 9 or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable selective-teacher registration differs')
    site.environment(spec['reference']['environment']); reuse.check(spec['control'], imported=spec['imported'])
    selected(spec['validation_kind'], spec['validation_root'], spec['validation_hash'], spec['bundle_root'])
    if check_storage: storage(spec)
