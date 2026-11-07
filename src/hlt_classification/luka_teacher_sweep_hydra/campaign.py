"""New immutable Hydra execution; all historical sources/artifacts stay pinned."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_next_supervision_hydra.group_b import original
from hlt_classification.luka_next_supervision_hydra.campaign import storage
from hlt_classification.luka_next_supervision_hydra import reuse
from hlt_classification.luka_cross_message_hydra.execution import A6000_SITE
from hlt_classification.luka_u100_pair_hydra import site
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import BASE, COUNTS, FITS, TEACHERS, STUDENTS, recipe, artifact, validate
from .transfer import Delta

MODULES = ('__init__', 'contracts', 'campaign', 'transfer', 'inputs', 'targets', 'model',
           'training', 'resources', 'submission', 'worker', 'reporting', 'site')
RUNTIME = {f'src/hlt_classification/luka_teacher_sweep_hydra/{n}.py' for n in MODULES} | {
    'scripts/luka_teacher_sweep_hydra.py', 'scripts/queue_luka_teacher_sweep_hydra.sh', 'sbatch/run_luka_teacher_sweep_hydra.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_teacher_sweep_hydra.py', 'docs/plans/LUKA_TEACHER_SWEEP_HYDRA_PLAN.md',
    'docs/contracts/LUKA_TEACHER_SWEEP_HYDRA.md', 'docs/LUKA_TEACHER_SWEEP_HYDRA_RUNBOOK.md', 'docs/LUKA_TEACHER_SWEEP_HYDRA_VALIDATION.md'}
COPIED = ('bundle_root', 'reference', 'imported', 'anchor', 'embedding_hash', 'qualification')


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if (not REQUIRED <= set(git('ls-files')) or any(r[:2] != 'A\t' or r[2:] not in RUNTIME for r in changes)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_teacher_sweep_hydra/campaign.py'):
        raise ValueError('Commit the complete additive teacher sweep; preserve donor runtime')
    return snapshot


def registration():
    return dict(recipe=recipe(), tasks=list(FITS), counts=COUNTS, execution_site=A6000_SITE,
        gpu_profile='a6000', cpus=6, memory_mb=32000, setup_minutes=720, fit_max_minutes=2880,
        release_minutes=60, report_minutes=60, original50k_rows=50000,
        cpu_site=dict(partition='compute', account='default', qos='normal', cpus=1, memory_mb=4000),
        storage_budget_bytes=8*2**30, resource_policy=dict(safety_multiplier=2., fixed_overhead_seconds=900, memory_fraction=.85),
        science_dependencies='student_after_own_teacher_only', teacher_fits=12, student_fits=12,
        automatic_retry=False, reference_retrained=False)


def nodes(old):
    original_node = Bundle(old['bundle_root']).json('reference/OFFLINE/report.json')['node']
    result = {t: dict(original_node, node_id=t, coordinate='OFFLINE', teacher=None, context_coordinate=None,
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


def create(root, group_a_spec, offline_root, offline_hash, project, commit, quota_gib):
    old = original(group_a_spec); project = Path(project).resolve(); root = Path(root).absolute()
    if type(quota_gib) is not int or quota_gib < 9:
        raise ValueError('Reserve eight GiB outputs plus one GiB headroom after transfers/environment')
    d = Delta(offline_root, offline_hash, old['bundle_root']); d.verify_all()
    protected = [Path(old[k]).resolve() for k in ('root', 'project_dir', 'environment_dir', 'bundle_root')]+[project, d.root]
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate teacher-sweep root')
    source = current(project, commit); site.environment(old['reference']['environment'])
    reuse.check(old['reused']['C0'], imported=old['imported'])
    value = artifact('SPEC', parents=dict(source=source['content_hash'], group_a=old['content_hash'], recipe=recipe()['content_hash'], offline=offline_hash),
        **registration(), root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()), group_a_spec=file_ref(Path(group_a_spec).resolve()),
        offline_root=str(d.root), offline_hash=offline_hash, available_quota_gib=quota_gib,
        control=old['reused']['C0'], nodes=nodes(old), recovery_anchors=anchors(old), **{k: old[k] for k in COPIED})
    storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json', value)
    return value


def check(spec, *, check_storage=True):
    validate_file_ref(spec['group_a_spec']); old = original(spec['group_a_spec']['path'])
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], group_a=old['content_hash'], recipe=recipe()['content_hash'], offline=spec['offline_hash']))
    if (any(spec.get(k) != v for k, v in registration().items()) or any(spec[k] != old[k] for k in COPIED)
            or spec['control'] != old['reused']['C0'] or spec['nodes'] != nodes(old) or spec['recovery_anchors'] != anchors(old)
            or spec['available_quota_gib'] < 9 or Path(sys.prefix).resolve() != Path(spec['environment_dir'])
            or current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable teacher-sweep registration differs')
    site.environment(spec['reference']['environment']); reuse.check(spec['control'], imported=spec['imported'])
    Delta(spec['offline_root'], spec['offline_hash'], spec['bundle_root'])
    if check_storage: storage(spec)
