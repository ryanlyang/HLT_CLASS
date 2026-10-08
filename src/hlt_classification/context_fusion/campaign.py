"""Pinned parent import, frozen fusion graph, and separate dry/live admission."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import shlex
import subprocess

from hlt_classification.cms_proxy_ladder import context, production
from hlt_classification.cms_proxy_ladder.campaign import paired_seed, coordinate
from hlt_classification.cms_proxy_ladder.gate import _source
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.contracts import validate as validate_kernel
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import (
    PAIR_OFFLOAD_POLICY, PARITY_BACKEND, PARITY_CHECKS, PARITY_TOLERANCES, validate_offload_stats)
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .contracts import artifact, validate, file_ref, validate_file_ref, load_json, write_json, safe

PARENT_COMMIT = 'f889f359cdddca1beb3d6e817332d2b42a42ddf3'
COUNTS = dict(train=100000, validation=50000)
AUTHORIZATION = 'AUTHORIZE CONTEXT V1 OSCAR FUSION EXACT PLAN'
EXTRA_FILES = ('scripts/jetclass2_context_fusion.py', 'scripts/queue_jetclass2_context_fusion.sh',
    'docs/plans/JETCLASS2_CONTEXT_FUSION_PLAN.md', 'docs/contracts/JETCLASS2_CONTEXT_FUSION.md',
    'tests/test_context_fusion.py')


def nodes():
    pairs = [('FUSION_U050', 'U050', 'U000', 'U000'),
        ('FUSION_U100', 'U100', 'U050', 'FUSION_U050'),
        ('FUSION_D066', 'D066', 'U100', 'FUSION_U100'),
        ('FUSION_D033', 'D033', 'D066', 'FUSION_D066'),
        ('FUSION_D000', 'D000', 'D033', 'FUSION_D033'),
        ('FINAL_DIRECT_D000', 'D000', None, 'FUSION_D000'),
        ('FUSION_D000_D000', 'D000', 'D000', 'FUSION_D000'),
        ('FINAL_BRIDGE_D000', 'D000', None, 'FUSION_D000_D000')]
    result = []
    for name, primary, other, teacher in pairs:
        u, f = coordinate(primary)
        result.append(dict(node_id=name, coordinate=primary, context_coordinate=other,
            teacher=teacher, branch='FUSION', u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(primary, 'initialization'), sampler_seed=paired_seed(primary, 'sampler'),
            context_initialization_seed=None if other is None else int.from_bytes(
                hashlib.sha256(('CONTEXT_FUSION/v1/context/'+name).encode()).digest()[:4], 'big'),
            deployable=primary == 'D000' and other in (None, 'D000')))
    return result


def tasks():
    result = []
    teachers = {n['teacher'] for n in nodes()} - {'U000'}
    for node in nodes():
        name = node['node_id']
        deps = [] if node['teacher'] == 'U000' else ['reduce_'+node['teacher']]
        result.append(dict(task_id='train_'+name, kind='train', node_id=name, dependencies=deps))
        if name in teachers:
            result.append(dict(task_id='reduce_'+name, kind='reduce', node_id=name, dependencies=['train_'+name]))
    result.append(dict(task_id='aggregate', kind='aggregate', node_id=None,
        dependencies=[r['task_id'] for r in result]))
    result.append(dict(task_id='complete', kind='complete', node_id=None, dependencies=['aggregate']))
    return result


def registration():
    return dict(counts=dict(COUNTS), nodes=nodes(), tasks=tasks(), training_recipe=recipe(),
        validation_policy='same_entire_50k_for_checkpoint_selection_and_development_reporting',
        execution_site=execution_site('oscar_l40s'), cpus=6, workers=6, memory_mb=180000,
        gate_minutes=720, maximum_train_minutes=2880, maximum_reduce_minutes=1440,
        new_fits=8, new_reducers=6, science_jobs=16, all_new_models_cold_start=True,
        alpha=1., fusion_injections=[2, 4, 6, 8], direction='context_to_primary',
        saved_tensor_storage=deepcopy(PAIR_OFFLOAD_POLICY), parity_backend=deepcopy(PARITY_BACKEND),
        reused_parent_rows='all_9_controls_direct_coarse', rolling_resume=False,
        full_views_persisted=False, existing_campaign_mutations=False, automatic_followup=False)


def import_parent(path):
    parent = load_json(path)
    context.validate_campaign(parent, check_source=True)
    if (parent['source_commit'] != PARENT_COMMIT or parent['schema_version'] != 4
            or parent['foundation']['role_counts'] != COUNTS
            or parent['scientific_plan']['recipe'] != recipe()):
        raise ValueError('Expected pinned CONTEXT_V1 100k/50k parent')
    rows = production.result_rows(parent)
    if len(rows) != 9 or any(r['state'] != 'COMPLETE' for r in rows):
        raise ValueError('All original comparison fits must be committed')
    references = {}
    for task in ['train_'+r['node_id'] for r in rows] + ['reduce_U000']:
        report = production.completed_task(parent, task)
        if report is None:
            raise ValueError('Missing parent teacher/comparison: '+task)
        references[task] = file_ref(Path(parent['campaign_root'])/'tasks'/f'{task}.json')
        if task.startswith('train_'):
            report = load_json(safe(Path(parent['campaign_root']), report['result']['training_report']))
            validate_kernel(report, 'KERNEL_TRAINING_REPORT')
            if (report['scientific_fit'] is not True or report['acceptance_only'] is not False
                    or report['recipe_sha256'] != recipe()['content_hash']
                    or report['final_test_accessed'] is not False):
                raise ValueError('Parent comparison is not a scientific fit')
    bank = production.completed_task(parent, 'reduce_U000')['result']
    if bank['teacher_report_sha256'] != production.completed_task(parent, 'train_U000')['result']['training_report_sha256']:
        raise ValueError('Imported U000 bank names another teacher')
    return parent, artifact('PARENT', parents=dict(campaign=parent['content_hash']),
        spec=file_ref(Path(path)), receipts=references, rows=rows,
        foundation_sha256=parent['foundation']['content_hash'], u000_bank=bank)


def source_lock(project, commit):
    root = Path(project).resolve(strict=True)
    _source(root, commit)
    # Pin the whole reusable Python package, including all transitive scientific
    # donors. Do not depend on whichever scripts happen to be on PYTHONPATH.
    names = subprocess.run(['git', '-C', str(root), 'ls-files', 'src/hlt_classification',
        'sbatch/common.sh', 'sbatch/jetclass2_delphes_common.sh', *EXTRA_FILES],
        capture_output=True, text=True, check=True).stdout.splitlines()
    if not set(EXTRA_FILES).issubset(names) or not any(n.endswith('context_fusion/worker.py') for n in names):
        raise ValueError('Commit all context-fusion implementation files first')
    return artifact('SOURCE', commit=commit, files={n: sha256_file(root/n) for n in names})


def protected_roots(parent, project):
    r = parent['foundation']['release']
    return [Path(p).resolve() for p in (project, parent['project_dir'], parent['campaign_root'],
        parent['gate_root'], parent['foundation_root'], parent['foundation']['release_root'],
        r['study_root'], r['offline_root'], r['request']['provenance_root'])]


def validate_location(root, parent, project):
    root = Path(root).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected_roots(parent, project)):
        raise PermissionError('Output overlaps protected parent/data/source')


def create(*, parent_spec, project_dir, source_commit, root):
    parent, imported = import_parent(Path(parent_spec).resolve(strict=True))
    root, project = Path(root).resolve(), Path(project_dir).resolve(strict=True)
    validate_location(root, parent, project)
    if root.exists():
        raise FileExistsError('Fresh separate fusion root required')
    source = source_lock(project, source_commit)
    spec = artifact('SPEC', parents=dict(parent=imported['content_hash'], source=source['content_hash']),
        root=str(root), project_dir=str(project), source_commit=source_commit,
        source=source, parent_import=imported, **registration())
    from .inputs import budgets
    budgets(parent, workers=spec['workers'], memory_mb=spec['memory_mb'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'study_spec.json', spec)
    return spec


def validate_spec(spec):
    validate(spec, 'SPEC', parents=dict(parent=spec['parent_import']['content_hash'], source=spec['source']['content_hash']))
    parent, imported = import_parent(validate_file_ref(spec['parent_import']['spec']))
    if (imported != spec['parent_import'] or any(spec.get(k) != v for k, v in registration().items())
            or spec['source'] != source_lock(spec['project_dir'], spec['source_commit'])):
        raise ValueError('Frozen fusion scientific/source registration differs')
    validate_location(spec['root'], parent, spec['project_dir'])
    return parent


def validate_preflight(value, spec, parent):
    validate(value, 'PREFLIGHT', parents=dict(spec=spec['content_hash']))
    from hlt_classification.jetclass2_delphes.model import model_contract
    p = value['single_parity']
    validate_kernel(p, 'WEAVER_PARITY')
    validate_kernel(value['environment'], 'INSTALLED_ENVIRONMENT', version=2)
    if (p['passed'] is not True or p['device'] != 'cuda' or p['model'] != model_contract()
            or p['forward_and_feature_and_parameter_gradients'] is not True
            or value['counts'] != COUNTS or value['site'] != spec['execution_site']
            or value['source_commit'] != spec['source_commit'] or value['passed'] is not True
            or value['native_mask_parity'] is not True or value['checkpoint_bank_roundtrip'] is not True
            or value['environment']['architecture'] != 'x86_64' or 'L40S' not in value['gpu']['name']
            or not 0 < value['gpu_peak_bytes'] <= .85*value['gpu']['total_memory_bytes']
            or not 0 < value['peak_rss_bytes'] <= .85*spec['memory_mb']*1024**2
            or not 0 < value['cache_bytes'] <= .65*spec['memory_mb']*1024**2):
        raise ValueError('Fusion technical acceptance differs')
    if set(value['offload_parity']) != {'adjacent_fp32', 'adjacent_bf16', 'endpoint_fp32', 'endpoint_bf16'}:
        raise ValueError('Missing paired precision/view parity')
    for key, row in value['offload_parity'].items():
        precision = key.rsplit('_', 1)[1]
        if (row['passed'] is not True or row['device_type'] != 'cuda' or row['precision'] != precision
                or row['steps'] != 3 or row['checks'] != PARITY_CHECKS
                or row['parity_backend'] != PARITY_BACKEND or row['saved_tensor_storage'] != PAIR_OFFLOAD_POLICY
                or row['tolerance'] != PARITY_TOLERANCES[precision]):
            raise ValueError('Fusion offload parity differs')
        validate_offload_stats(row['offload_stats'], calls=3)
    expected = [nodes()[0], nodes()[6], nodes()[5]]
    if [row['node'] for row in value['measurements']] != expected:
        raise ValueError('Missing single/adjacent/HLT-pair acceptance')
    for row, node in zip(value['measurements'], expected):
        report = row['training']
        validate_kernel(report, 'KERNEL_TRAINING_REPORT')
        if (report['node'] != node or report['foundation_sha256'] != parent['foundation']['content_hash']
                or report['recipe_sha256'] != recipe()['content_hash'] or report['acceptance_only'] is not True
                or report['scientific_fit'] is not False or report['passes'] != 1
                or report['final_test_accessed'] is not False or row['stress_steps'] != 3
                or row['stress_batch'] != 256):
            raise ValueError('Fusion full-population acceptance training differs')
        if node['context_coordinate']:
            validate_offload_stats(row['stress_offload'], calls=3)
        for seconds in (row['cache_seconds'], report['runtime_seconds'], row['inference_seconds']):
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid resource timing')
    train = max(60, math.ceil(max((r['cache_seconds']+100*r['training']['runtime_seconds'])*1.75
        for r in value['measurements'])/60))
    reduce = max(30, math.ceil(max((r['cache_seconds']+r['inference_seconds'])*2
        for r in value['measurements'])/60))
    if (value['train_minutes'] != train or value['reduce_minutes'] != reduce
            or train > spec['maximum_train_minutes'] or reduce > spec['maximum_reduce_minutes']):
        raise ValueError('Measured full-budget runtime exceeds execution envelope')
    return value


def preflight(spec, parent):
    return validate_preflight(load_json(Path(spec['root'])/'preflight'/'result.json'), spec, parent)


def plan(spec, mode):
    parent = validate_spec(spec)
    if mode == 'gate':
        rows = [dict(task_id='preflight', kind='gate', dependencies=[])]
        evidence, parents = None, dict(spec=spec['content_hash'])
    elif mode == 'science':
        evidence = preflight(spec, parent)
        rows, parents = tasks(), dict(spec=spec['content_hash'], preflight=evidence['content_hash'])
    else:
        raise ValueError('Unknown phase')
    commands = []
    root, project, site = Path(spec['root']), spec['project_dir'], spec['execution_site']
    for row in rows:
        task, kind, deps = row['task_id'], row['kind'], row['dependencies']
        gpu = kind in {'train', 'reduce', 'gate'}
        minutes = spec['gate_minutes'] if kind == 'gate' else (
            evidence[kind+'_minutes'] if gpu else 60)
        lines = ['set -euo pipefail', 'export PROJECT_DIR='+shlex.quote(project),
            'export JC2_SITE=oscar_l40s', 'source '+shlex.quote(project+'/sbatch/jetclass2_delphes_common.sh')]
        if kind == 'gate':
            lines.append('export CUBLAS_WORKSPACE_CONFIG=:4096:8')
        lines.append('python -u -s '+shlex.quote(project+'/scripts/jetclass2_context_fusion.py')+
            ' run --spec '+shlex.quote(str(root/'study_spec.json'))+' --task '+shlex.quote(task))
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus'] if gpu else 1),
            '--mem='+str(spec['memory_mb'] if gpu else 16000)+'M', '--time='+_walltime(minutes),
            '--job-name=jc2cxf_'+task, '--comment=jc2cxf:'+spec['content_hash']+':'+task,
            '--chdir='+project, '--output='+str(root/'slurm-%j.out')]
        if gpu:
            args.append('--gres='+site['gres'])
        if deps:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps))
        commands.append(dict(task_id=task, dependencies=deps, command=args+['--wrap='+'\n'.join(lines)]))
    return artifact('PLAN', parents=parents, mode=mode, commands=commands, automatic_followup=False)


def submit(spec, *, mode, execute=False, plan_hash=None, authorization=None):
    value = plan(spec, mode)
    root = Path(spec['root'])/('submission_'+mode)
    root.mkdir(exist_ok=True)
    path = root/'command_plan.json'
    if path.exists():
        if load_json(path) != value:
            raise ValueError('Saved exact plan differs')
    else:
        write_json(path, value)
    if not execute:
        submit_exact_dag(identity=spec['content_hash'], plan=value,
            output=root/'dry_run_submission_ledger.json', canonical_dry_run=root/'dry_run_submission_ledger.json', execute=False)
        return value
    if authorization != AUTHORIZATION or plan_hash != value['content_hash']:
        raise PermissionError('Reviewed plan hash and explicit phase authorization required')
    context.check_submission_site(value)
    return context.submit_claimed(spec, value, root)
