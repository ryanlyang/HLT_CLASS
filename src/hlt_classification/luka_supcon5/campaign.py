"""Pinned small debug campaign, separate from every historical execution."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_pair_correction import source as old_source
from hlt_classification.luka_u100_pair_correction.campaign import storage
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from .contracts import artifact, validate, recipe, COUNTS, TEACHERS, STUDENTS

DONOR = '00fbb92641553e8817b577b8765796e6618afc13'
MODULES = ('__init__', 'contracts', 'campaign', 'model', 'training', 'inputs', 'resources', 'worker', 'submission')
RUNTIME = {'src/hlt_classification/luka_supcon5/'+x+'.py' for x in MODULES} | {
    'scripts/luka_supcon5.py', 'scripts/queue_luka_supcon5.sh', 'sbatch/run_luka_supcon5.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_supcon5.py', 'docs/plans/LUKA_SUPCON5_DEBUG_PLAN.md',
    'docs/contracts/LUKA_SUPCON5.md', 'docs/LUKA_SUPCON5_DEBUG_RUNBOOK.md', 'docs/LUKA_SUPCON5_VALIDATION.md'}


def current(project, commit):
    project = Path(project).resolve(); result = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', DONOR, commit)
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit the complete SupCon5 implementation')
    changes = git('diff', '--name-status', '--no-renames', DONOR, commit, '--', 'src', 'scripts', 'sbatch')
    if any(row[:2] not in ('A\t', 'M\t') or row[2:] not in RUNTIME for row in changes):
        raise ValueError('Unexpected runtime changes since reviewed donor')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_supcon5/campaign.py':
        raise ValueError('Runtime outside pinned project')
    return result


def policy():
    return dict(recipe=recipe(), counts=COUNTS, execution_site=execution_site('sporc_a100_debug'),
        cpus=6, memory_mb=90000, storage_budget_bytes=8*2**30, setup_minutes=720,
        fit_max_minutes=1440, report_minutes=120, original50k_rows=50000,
        teacher_fits=1, student_fits=1, reference_fits=0)


def nodes(reference):
    ref = reference['imported']['anchor']['evidence']['models']['OFFLINE']['report']
    validate_file_ref(ref); original = load_json(ref['path'])['node']
    result = {t: dict(original, node_id=t, coordinate='OFFLINE', teacher=None, context_coordinate=None,
        deployable=False, initialization_seed=original['initialization_seed']+200000,
        sampler_seed=original['sampler_seed']+200000) for t in TEACHERS}
    result.update({s: dict(reference['imported']['node'], node_id=s) for s in STUDENTS})
    return result


def create(root, reference, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve() != root or root.is_symlink() or type(quota_gib) is not int or quota_gib < 8:
        raise ValueError('Fresh unredirected root and eight available GiB required')
    snapshot = current(project, commit); control = old_source.import_control(reference)
    imported = control['imported']; anchor = imported['anchor']
    protected = (project, Path(reference).resolve().parent, Path(anchor['evidence']['prepared_root']),
                 Path(anchor['evidence']['raw_container']), Path(control['u100_manifest']['path']).parent)
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Output overlaps protected artifacts')
    anchors = {}
    for name in ('M0HLT', 'OFFLINE'):
        ref = anchor['evidence']['models'][name]['predictions']; validate_file_ref(ref)
        a = load_npz_arrays(ref['path']); anchors[name] = evaluate_probabilities(a['labels'], a['probabilities'])['macro_ovr_auc']
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], reference=control['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        reference=control, imported=imported, anchor=anchor, nodes=nodes(control), recovery_anchors=anchors,
        available_quota_gib=quota_gib, **policy())
    storage(spec); root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reference=spec['reference']['content_hash']))
    if (any(spec.get(k) != v for k, v in policy().items()) or spec['available_quota_gib'] < 8
            or spec['imported'] != spec['reference']['imported'] or spec['anchor'] != spec['imported']['anchor']
            or spec['nodes'] != nodes(spec['reference']) or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Immutable scientific registration differs')
    old_source.validate_reference(spec['reference'])
    if current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source snapshot changed')
    if check_storage:
        storage(spec)
    return spec
