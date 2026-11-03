"""Portable execution spec bound to immutable original science and transfer."""
from pathlib import Path
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_pair_correction.campaign import storage
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import BASE, SITE, PARITY, ARMS, COUNTS, recipe, artifact, validate
from .inputs import Bundle
from . import site

MODULES = ('__init__', 'contracts', 'inputs', 'site', 'campaign', 'submission', 'worker',
           'precision', 'precision_campaign')
RUNTIME = {'src/hlt_classification/luka_u100_pair_hydra/'+n+'.py' for n in MODULES} | {
    'scripts/luka_u100_pair_hydra.py', 'scripts/setup_luka_u100_pair_hydra.sh',
    'scripts/queue_luka_u100_pair_hydra.sh', 'scripts/queue_luka_u100_pair_hydra_precision.sh',
    'sbatch/run_luka_u100_pair_hydra.sh'}
REQUIRED = RUNTIME | {'docs/plans/LUKA_U100_PAIR_HYDRA_PLAN.md',
    'docs/contracts/LUKA_U100_PAIR_HYDRA.md', 'docs/LUKA_U100_PAIR_HYDRA_RUNBOOK.md',
    'tests/test_luka_u100_pair_hydra.py', 'tests/test_luka_u100_pair_hydra_precision.py',
    'docs/LUKA_U100_PAIR_HYDRA_PRECISION_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    rows = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if not REQUIRED <= set(git('ls-files')) or any(r[:2] not in ('A\t', 'M\t') or r[2:] not in RUNTIME for r in rows):
        raise ValueError('Commit the complete additive Hydra port; preserve original scientific runtime')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_pair_hydra/campaign.py':
        raise ValueError('Runtime escapes pinned source')
    return snapshot


def registration():
    return dict(recipe=recipe(), arms=ARMS, counts=COUNTS, original50k_rows=50000,
        execution_site=SITE, cpus=6, memory_mb=120000, storage_budget_bytes=8*2**30,
        setup_minutes=720, fit_max_minutes=1440, report_minutes=120, max_concurrent_fits=2,
        scientific_fits=4, reference_fits=0, new_teacher_fits=0, standalone_preflight=False,
        admission='shared_setup_cross_site_parity_and_four_real_miniatures', parity_tolerances=PARITY,
        automatic_retry=False)


def create(root, bundle_root, project, commit, quota_gib):
    root = Path(root).absolute(); project = Path(project).resolve()
    bundle = Bundle(bundle_root); bundle.verify_all()
    origin = bundle.json('origin/campaign_spec.json')
    proof = bundle.manifest['origin']
    if (origin['content_hash'] != proof['campaign'] or origin['reference']['content_hash'] != proof['reference']
            or origin['reference']['environment'] != proof['environment'] or origin['counts'] != COUNTS
            or origin['source_commit'] != proof['source_commit'] or origin['recipe'] != recipe()
            or origin['reference']['job_id'] != '21993056'):
        raise ValueError('Transferred origin does not bind this experiment')
    if type(quota_gib) is not int or quota_gib < 8:
        raise ValueError('Reserve at least eight GiB for outputs, after inputs and environment')
    environment_dir = Path(sys.prefix).resolve()
    protected = (bundle.root, project, environment_dir)
    if (root.resolve() != root or root.exists() or not root.parent.is_dir()
            or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise PermissionError('Use a fresh output root separate from source, inputs and environment')
    site.environment(proof['environment'])
    snapshot = current(project, commit)
    anchors = {}
    for name in ('M0HLT', 'OFFLINE'):
        a = load_npz_arrays(bundle.path(f'reference/{name}/predictions.npz'))
        anchors[name] = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], origin=origin['content_hash'],
        transfer=bundle.manifest['content_hash']), root=str(root), project_dir=str(project), source_commit=commit,
        source_snapshot=snapshot, bundle_root=str(bundle.root), environment_dir=str(environment_dir),
        reference=origin['reference'], imported=origin['imported'], anchor=origin['anchor'],
        recovery_anchors=anchors, available_quota_gib=quota_gib, **registration())
    storage(spec); root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    bundle = Bundle(spec['bundle_root']); origin = bundle.json('origin/campaign_spec.json')
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], origin=origin['content_hash'],
        transfer=bundle.manifest['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items()) or spec['available_quota_gib'] < 8
            or any(spec[k] != origin[k] for k in ('reference', 'imported', 'anchor'))
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']):
        raise ValueError('Immutable Hydra registration/source differs')
    site.environment(spec['reference']['environment'])
    if Path(sys.prefix).resolve() != Path(spec['environment_dir']):
        raise ValueError('Use the registered isolated Python environment')
    if check_storage:
        storage(spec)
    return spec
