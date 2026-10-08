"""Exact-source baseline reuse and separately authorized preflight/science DAGs."""
from pathlib import Path
import math
import shlex
import subprocess

from hlt_classification.cms_proxy_ladder import correlated_topology as parent_adapter, production
from hlt_classification.cms_proxy_ladder.contracts import sha256_file
from hlt_classification.cms_proxy_ladder.correlated import check_submission_site, submit_claimed
from hlt_classification.cms_proxy_ladder.gate import _source
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.jetclass2_delphes.campaign import recipe as training_recipe
from hlt_classification.jetclass2_delphes.contracts import validate as validate_training
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .contracts import artifact, validate, file_ref, validate_file_ref, write_json, load_json, safe
from .kernel import recipe, CANDIDATES

BASE_COMMIT = '8c5b47029ed15f88aa852b6f38d06e98bbbfcd9b'
COUNTS = dict(train=100000, validation=50000)
EXTRA_FILES = ('scripts/jetclass2_gap_sweep.py', 'scripts/queue_jetclass2_gap_sweep.sh',
    'docs/plans/JETCLASS2_GAP_SWEEP_PLAN.md', 'docs/contracts/JETCLASS2_GAP_SWEEP.md',
    'tests/test_gap_sweep.py')
AUTHORIZATION = 'AUTHORIZE CORR HIGH TOPO BASELINE GAP SWEEP EXACT PLAN'


def selection_policy():
    return dict(metric='accuracy', gap='OFFLINE_minus_candidate_absolute_fraction',
        lower_exclusive=.01, upper_inclusive=.02, choose='mildest_registered_in_band',
        all_candidates_required=True, kd_results_used=False, development_only=True,
        heldout_confirmation_required=True, automatic_followup=False)


def protected_roots(parent, project):
    release = parent['foundation']['release']
    return [Path(p).resolve() for p in (project, parent['campaign_root'], parent['gate_root'],
        parent['foundation_root'], parent['foundation']['release_root'], release['offline_root'], release['study_root'])]


def parent_controls(path):
    parent = load_json(path)
    parent_adapter.validate_campaign(parent, check_source=True)
    if (parent['source_commit'] != BASE_COMMIT or parent['foundation']['role_counts'] != COUNTS
            or parent['scientific_plan']['recipe'] != training_recipe()):
        raise ValueError('Expected original CORR_HIGH_TOPO 100k/50k baseline campaign')
    controls = {}
    for name in ('OFFLINE', 'M0HLT'):
        pointer = production.completed_task(parent, 'train_'+name)
        if pointer is None:
            raise ValueError('Required original baseline is not committed: '+name)
        report_path = safe(Path(parent['campaign_root']), pointer['result']['training_report'])
        report = load_json(report_path)
        validate_training(report, 'KERNEL_TRAINING_REPORT')
        node = next(n for n in parent['scientific_plan']['nodes'] if n['node_id'] == name)
        if (report['content_hash'] != pointer['result']['training_report_sha256']
                or report['node'] != node or node['teacher'] is not None
                or report['foundation_sha256'] != parent['foundation']['content_hash']
                or report['recipe_sha256'] != training_recipe()['content_hash']
                or report['scientific_fit'] is not True or report['acceptance_only'] is not False
                or report['final_test_accessed'] is not False):
            raise ValueError('Original CE baseline lineage differs')
        controls[name] = file_ref(report_path)
    # No DIRECT/COARSE task reports or logits are read here.
    return parent, controls


def source_lock(parent, project, commit):
    root = Path(project).resolve(strict=True)
    _source(root, commit)
    # Reused training/reader/response files must be byte-identical, not just named alike.
    files = dict(parent['source']['files'])
    for name, digest in files.items():
        if sha256_file(root/name) != digest:
            raise ValueError('Parent scientific source changed: '+name)
    names = [*files, *EXTRA_FILES, *(
        p.relative_to(root).as_posix() for p in sorted((root/'src/hlt_classification/gap_sweep').glob('*.py')))]
    for name in names:
        subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', name], check=True, capture_output=True)
        files[name] = sha256_file(root/name)
    return artifact('SOURCE', commit=commit, files=files,
        unchanged_parent_source=parent['source']['content_hash'])


def create(*, parent_spec, project_dir, source_commit, root):
    parent_path, project, root = Path(parent_spec).resolve(strict=True), Path(project_dir).resolve(strict=True), Path(root).resolve()
    parent, controls = parent_controls(parent_path)
    protected = protected_roots(parent, project)
    if root.exists() or any(root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root) for p in protected):
        raise FileExistsError('Fresh non-overlapping sweep root required; preserve existing output')
    source = source_lock(parent, project, source_commit)
    spec = artifact('SPEC', parents=dict(parent=parent['content_hash'], source=source['content_hash'], recipe=recipe()['content_hash']),
        root=str(root), parent_spec=file_ref(parent_path), project_dir=str(project), source_commit=source_commit,
        source=source, recipe=recipe(), controls=controls, counts=dict(COUNTS), selection=selection_policy(),
        training_recipe=training_recipe(), execution_site=parent['runtime_profile']['execution_site'],
        workers=6, cpus=6, memory_mb=90000, new_scientific_fits=3, final_dataset_export=False)
    validate_spec(spec)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'study_spec.json', spec)
    return spec


def validate_spec(spec):
    validate(spec, 'SPEC')
    parent, controls = parent_controls(validate_file_ref(spec['parent_spec']))
    source = source_lock(parent, spec['project_dir'], spec['source_commit'])
    validate(spec, 'SPEC', parents=dict(parent=parent['content_hash'], source=source['content_hash'], recipe=recipe()['content_hash']))
    if (spec['source'] != source or spec['recipe'] != recipe() or spec['counts'] != COUNTS
            or spec['controls'] != controls or spec['selection'] != selection_policy()
            or spec['training_recipe'] != training_recipe()
            or spec['execution_site'] != parent['runtime_profile']['execution_site']
            or (spec['workers'], spec['cpus'], spec['memory_mb'], spec['new_scientific_fits']) != (6, 6, 90000, 3)
            or spec['final_dataset_export'] is not False):
        raise ValueError('Frozen gap-screen specification differs')
    root = Path(spec['root']).resolve()
    for protected in protected_roots(parent, spec['project_dir']):
        p = Path(protected).resolve()
        if root.is_relative_to(p) or p.is_relative_to(root):
            raise ValueError('Sweep overlaps protected source')
    return parent


def candidate_identity(spec, parent, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Unregistered candidate')
    return artifact('INPUT', parents=dict(foundation=parent['foundation']['content_hash'], recipe=spec['recipe']['content_hash']),
                    candidate=candidate)['content_hash']


def node(parent, candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Unregistered candidate')
    baseline = next(n for n in parent['scientific_plan']['nodes'] if n['node_id'] == 'M0HLT')
    return dict(baseline, node_id='GAP_'+candidate, branch='BASELINE_SCREEN')


def validate_preflight(value, spec, parent):
    validate(value, 'PREFLIGHT', parents=dict(spec=spec['content_hash']))
    if (value['passed'] is not True or value['counts'] != spec['counts']
            or value['replayed_candidates'] != list(CANDIDATES) or value['exact_process_replay'] is not True
            or value['source_commit'] != spec['source_commit'] or value['site'] != spec['execution_site']
            or value['installed_parity']['passed'] is not True
            or value['installed_parity']['forward_and_feature_and_parameter_gradients'] is not True
            or value['installed_parity']['device'] != 'cuda'
            or not 60 <= value['train_minutes'] <= 1440
            or not 0 < value['gpu_peak_bytes'] <= .85*value['gpu']['total_memory_bytes']
            or not 0 < value['cache_bytes'] <= .75*spec['memory_mb']*1024**2):
        raise ValueError('Measured screen preflight differs')
    validate_training(value['installed_parity'], 'WEAVER_PARITY')
    validate_training(value['acceptance'], 'KERNEL_TRAINING_REPORT')
    if (value['acceptance']['acceptance_only'] is not True or value['acceptance']['scientific_fit'] is not False
            or value['acceptance']['passes'] != 1 or value['acceptance']['final_test_accessed'] is not False
            or value['acceptance']['node'] != node(parent, 'S1')
            or value['acceptance']['foundation_sha256'] != candidate_identity(spec, parent, 'S1')
            or value['acceptance']['recipe_sha256'] != spec['training_recipe']['content_hash']
            or value['gpu'] != parent['runtime_profile']['gpu']
            or value['installed_environment'] != parent['runtime_profile']['installed_environment']
            or value['installed_parity']['model'] != parent['model']
            or value['replay']['candidate_order'] != list(CANDIDATES)
            or value['replay']['jets_per_candidate'] != min(32, spec['counts']['train'])
            or len(value['replay']['hashes']) != 3*value['replay']['jets_per_candidate']):
        raise ValueError('Technical acceptance report differs')
    timing = (value['cache_seconds'], value['acceptance']['runtime_seconds'])
    if any(not math.isfinite(v) or v <= 0 for v in timing):
        raise ValueError('Invalid preflight timing')
    expected_minutes = max(60, parent['runtime_profile']['train_minutes'], math.ceil((timing[0]+100*timing[1])*1.75/60))
    if value['train_minutes'] != expected_minutes:
        raise ValueError('Measured walltime projection differs')
    return value


def preflight(spec):
    parent = load_json(validate_file_ref(spec['parent_spec']))
    return validate_preflight(load_json(Path(spec['root'])/'preflight.json'), spec, parent)


def plan(spec, mode):
    validate_spec(spec)
    if mode == 'gate':
        tasks = [('preflight', [], True, 120)]
        parents = dict(spec=spec['content_hash'])
    elif mode == 'science':
        measured = preflight(spec)
        tasks = [('fit_'+n, [], True, measured['train_minutes']) for n in CANDIDATES]
        tasks += [('summary', ['fit_'+n for n in CANDIDATES], False, 60)]
        parents = dict(spec=spec['content_hash'], preflight=measured['content_hash'])
    else:
        raise ValueError('Unknown submission stage')
    commands = []
    site, project, root = spec['execution_site'], spec['project_dir'], Path(spec['root'])
    for task, dependencies, gpu, minutes in tasks:
        wrap = '\n'.join(('set -euo pipefail', 'export PROJECT_DIR='+shlex.quote(project),
            'export JC2_SITE='+shlex.quote(site['name']),
            'source '+shlex.quote(project+'/sbatch/jetclass2_delphes_common.sh'),
            'python -u -s '+shlex.quote(project+'/scripts/jetclass2_gap_sweep.py')+
            ' run --spec '+shlex.quote(str(root/'study_spec.json'))+' --task '+shlex.quote(task)))
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus'] if gpu else 1), '--mem='+str(spec['memory_mb'] if gpu else 16000)+'M',
            '--time='+_walltime(minutes), '--job-name=jc2gap_'+task,
            '--comment=jc2gap:'+spec['content_hash']+':'+task,
            '--chdir='+project, '--output='+str(root/'slurm-%j.out')]
        if gpu:
            command.append('--gres='+site['gres'])
        if dependencies:
            command.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in dependencies))
        commands.append(dict(task_id=task, dependencies=dependencies, command=command+['--wrap='+wrap]))
    return artifact('PLAN', parents=parents, mode=mode, commands=commands, automatic_followup=False)


def submit(spec, *, mode, execute=False, plan_hash=None, authorization=None):
    value = plan(spec, mode)
    root = Path(spec['root'])/('submission_'+mode)
    root.mkdir(exist_ok=True)
    path = root/'command_plan.json'
    if path.exists() and load_json(path) != value:
        raise ValueError('Saved exact plan changed')
    if not path.exists():
        write_json(path, value)
    if not execute:
        submit_exact_dag(identity=spec['content_hash'], plan=value,
            output=root/'dry_run_submission_ledger.json', canonical_dry_run=root/'dry_run_submission_ledger.json', execute=False)
        return value
    if authorization != AUTHORIZATION or plan_hash != value['content_hash']:
        raise PermissionError('Explicit authorization and reviewed exact hash required')
    check_submission_site(value)
    return submit_claimed(spec, value, root)


def completed_fit(spec, parent, candidate):
    root = Path(spec['root'])/('fit_'+candidate)
    report = load_json(root/'receipt.json')
    validate(report, 'FIT', parents=dict(spec=spec['content_hash'], preflight=preflight(spec)['content_hash']))
    expected = candidate_identity(spec, parent, candidate)
    training = load_json(validate_file_ref(report['training'], root=root))
    validate_training(training, 'KERNEL_TRAINING_REPORT')
    validate_file_ref(report['checkpoint'], root=root)
    diagnostics = load_json(validate_file_ref(report['diagnostics'], root=root))
    validate(diagnostics, 'DIAGNOSTICS', parents=dict(spec=spec['content_hash'], input=expected))
    if (report['candidate'] != candidate or report['input_identity'] != expected or report['counts'] != COUNTS
            or report['source_commit'] != spec['source_commit']
            or training['foundation_sha256'] != expected or training['node'] != node(parent, candidate)
            or training['recipe_sha256'] != training_recipe()['content_hash']
            or training['scientific_fit'] is not True or training['acceptance_only'] is not False
            or training['final_test_accessed'] is not False
            or diagnostics['role'] != 'train' or diagnostics['candidate'] != candidate
            or diagnostics['development_only'] is not True
            or diagnostics['mechanisms']['jets'] != spec['counts']['train']
            or not spec['training_recipe']['minimum_passes'] <= training['passes'] <= spec['training_recipe']['maximum_passes']
            or not 1 <= training['selected_pass'] <= training['passes']
            or len(training['validation_history']) != training['passes']
            or training['selected_weights_restored'] is not True):
        raise ValueError('Candidate baseline report differs')
    return report, training


def choose(offline_accuracy, metrics):
    if set(metrics) != set(CANDIDATES):
        raise ValueError('All three baseline results are required; no partial selection')
    scores = [offline_accuracy, *(m['accuracy'] for m in metrics.values())]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('Invalid baseline accuracy')
    gaps = {name: offline_accuracy-metrics[name]['accuracy'] for name in CANDIDATES}
    # Compare in percentage points with tiny arithmetic tolerance only (not statistical).
    selected = next((name for name, gap in gaps.items() if gap > .01+1e-12 and gap <= .02+1e-12), None)
    return dict(selected=selected, status='development_selection_only' if selected else 'no_candidate_in_band',
                gap_percentage_points={n: 100*g for n, g in gaps.items()})


def summarize(spec):
    parent = validate_spec(spec)
    controls = {name: load_json(validate_file_ref(ref))['validation'] for name, ref in spec['controls'].items()}
    reports, metrics = {}, {}
    for name in CANDIDATES:
        receipt, training = completed_fit(spec, parent, name)
        reports[name], metrics[name] = receipt['content_hash'], training['validation']
    return artifact('SUMMARY', parents=dict(spec=spec['content_hash'], **reports), controls=controls,
        metrics=metrics, selection_policy=selection_policy(), **choose(controls['OFFLINE']['accuracy'], metrics))
