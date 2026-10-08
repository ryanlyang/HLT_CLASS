"""Frozen selection and teacher reuse; isolated direct/coarse source and DAG."""
import math
from pathlib import Path
import shlex
import subprocess

from hlt_classification.gap_sweep import campaign as screen
from hlt_classification.cms_proxy_ladder import production
from hlt_classification.cms_proxy_ladder.gate import _source
from hlt_classification.cms_proxy_ladder.contracts import sha256_file
from hlt_classification.cms_proxy_ladder.correlated import check_submission_site, submit_claimed
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.jetclass2_delphes.banks import load_bank
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.contracts import validate as validate_training
from hlt_classification.jetclass2_delphes.reporting import recovery
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from . import cache, views
from .contracts import artifact, validate, file_ref, validate_file_ref, write_json, load_json, safe

SCREEN_COMMIT = '8abdc943c02d9844f98c91028f68e82e02b62fd4'
AUTHORIZATION = 'AUTHORIZE FROZEN S3 DIRECT COARSE EXACT PLAN'
EXTRA_FILES = ('scripts/jetclass2_s3_ladder.py', 'scripts/queue_jetclass2_s3_ladder.sh',
    'docs/plans/JETCLASS2_S3_LADDER_PLAN.md', 'docs/contracts/JETCLASS2_S3_LADDER.md',
    'tests/test_s3_ladder.py')


def screen_inputs(path):
    s = load_json(path)
    parent = screen.validate_spec(s)
    if s['source_commit'] != SCREEN_COMMIT:
        raise ValueError('Expected completed frozen S3 screen source')
    root = Path(s['root'])
    summary = load_json(root/'summary.json')
    if summary != screen.summarize(s) or summary['selected'] != 'S3':
        raise ValueError('Authenticated complete screen must select S3')
    receipt, _ = screen.completed_fit(s, parent, 'S3')
    teacher = production.completed_task(parent, 'reduce_OFFLINE')
    if teacher is None:
        raise ValueError('Original OFFLINE reducer must be completed')
    offline = load_json(validate_file_ref(s['controls']['OFFLINE']))
    result = teacher['result']
    bank_root = safe(Path(parent['campaign_root']), result['train_bank'])
    if result['teacher_report_sha256'] != offline['content_hash']:
        raise ValueError('OFFLINE reducer report differs')
    load_bank(bank_root, foundation_sha256=parent['foundation']['content_hash'],
        teacher_report_sha256=offline['content_hash'], teacher_node='OFFLINE', role='train',
        expected_identities=cache.identities(parent, 'train'))
    frozen = dict(summary=file_ref(root/'summary.json'), s3_receipt=file_ref(root/'fit_S3/receipt.json'),
        s3_report=file_ref(validate_file_ref(receipt['training'], root=root/'fit_S3')),
        offline_report=s['controls']['OFFLINE'],
        offline_reducer=file_ref(Path(parent['campaign_root'])/'tasks/reduce_OFFLINE.json'),
        offline_bank=file_ref(bank_root/'manifest.json'), endpoint_sha256=receipt['endpoint_sha256'])
    return s, parent, frozen


def source_lock(sweep, project, commit):
    root = Path(project).resolve(strict=True)
    _source(root, commit)
    files = dict(sweep['source']['files'])
    for name, digest in files.items():
        if sha256_file(root/name) != digest:
            raise ValueError('Inherited scientific source changed: '+name)
    names = [*files, *EXTRA_FILES, *(
        p.relative_to(root).as_posix() for p in sorted((root/'src/hlt_classification/s3_ladder').glob('*.py')))]
    for name in names:
        subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', name], check=True, capture_output=True)
        files[name] = sha256_file(root/name)
    return artifact('SOURCE', commit=commit, files=files, unchanged_screen_source=sweep['source']['content_hash'])


def graph(parent):
    nodes = []
    for n in parent['scientific_plan']['nodes']:
        if n['branch'] not in ('DIRECT', 'COARSE'):
            continue
        nodes.append(dict(n, node_id=n['node_id'].replace('CORRHT_', 'S3_', 1),
                          teacher=n['teacher'].replace('CORRHT_', 'S3_', 1)))
    expected = [('DIRECT', 'D000'), ('COARSE', 'D066'), ('COARSE', 'D033'), ('COARSE', 'D000')]
    if [(n['branch'], n['coordinate']) for n in nodes] != expected:
        raise ValueError('Original descending graph differs')
    return artifact('GRAPH', nodes=nodes, recipe=recipe(), reused_controls=['OFFLINE', 'S3'],
        reused_teacher='OFFLINE', reducers=[n['node_id'] for n in nodes if n['coordinate'] != 'D000'],
        fresh_fits=4, final_comparison='COARSE_D000_minus_DIRECT_D000',
        inference='D000_HLT_only_intermediates_oracle', no_weight_continuation=True)


def node(spec, name):
    return next(n for n in spec['graph']['nodes'] if n['node_id'] == name)


def input_identity(sweep, parent):
    return artifact('INPUT', parents=dict(foundation=parent['foundation']['content_hash'],
        endpoint=screen.candidate_identity(sweep, parent, 'S3'), views=views.contract()['content_hash']))['content_hash']


def _protect(root, parent, sweep, project):
    for p in [*screen.protected_roots(parent, project), Path(sweep['root']), Path(sweep['project_dir'])]:
        p = p.resolve()
        if root.is_relative_to(p) or p.is_relative_to(root):
            raise PermissionError('S3 output overlaps a protected input or source')


def create(*, sweep_spec, project_dir, source_commit, root):
    path, project, root = Path(sweep_spec).resolve(strict=True), Path(project_dir).resolve(strict=True), Path(root).resolve()
    if root.exists():
        raise FileExistsError('Fresh S3 campaign root required')
    sweep, parent, frozen = screen_inputs(path)
    _protect(root, parent, sweep, project)
    source, g = source_lock(sweep, project, source_commit), graph(parent)
    spec = artifact('SPEC', parents=dict(sweep=sweep['content_hash'], source=source['content_hash'], graph=g['content_hash']),
        root=str(root), sweep_spec=file_ref(path), project_dir=str(project), source_commit=source_commit,
        source=source, frozen=frozen, graph=g, views=views.contract(), input_identity=input_identity(sweep, parent),
        counts=sweep['counts'], execution_site=sweep['execution_site'], workers=6, cpus=6, memory_mb=90000,
        automatic_followup=False, development_only=True, dataset_export=False)
    validate_spec(spec)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'study_spec.json', spec)
    return spec


def validate_spec(spec):
    validate(spec, 'SPEC')
    sweep, parent, frozen = screen_inputs(validate_file_ref(spec['sweep_spec']))
    source, g = source_lock(sweep, spec['project_dir'], spec['source_commit']), graph(parent)
    validate(spec, 'SPEC', parents=dict(sweep=sweep['content_hash'], source=source['content_hash'], graph=g['content_hash']))
    if (spec['source'] != source or spec['frozen'] != frozen or spec['graph'] != g
            or spec['views'] != views.contract() or spec['input_identity'] != input_identity(sweep, parent)
            or spec['counts'] != sweep['counts'] or spec['execution_site'] != sweep['execution_site']
            or (spec['workers'], spec['cpus'], spec['memory_mb']) != (6, 6, 90000)
            or spec['automatic_followup'] is not False or spec['development_only'] is not True
            or spec['dataset_export'] is not False):
        raise ValueError('Frozen S3 campaign differs')
    _protect(Path(spec['root']).resolve(), parent, sweep, spec['project_dir'])
    return parent


def walltimes(parent, cache_seconds, kd_seconds, inference_seconds):
    return dict(train_minutes=max(60, parent['runtime_profile']['train_minutes'],
                    math.ceil((cache_seconds+100*kd_seconds)*1.75/60)),
                reduce_minutes=max(30, parent['runtime_profile']['reduce_minutes'],
                    math.ceil((cache_seconds+inference_seconds)*2/60)))


def validate_preflight(value, spec, parent):
    validate(value, 'PREFLIGHT', parents=dict(spec=spec['content_hash']))
    parity, kd = value['installed_parity'], value['acceptance']
    validate_training(parity, 'WEAVER_PARITY'); validate_training(kd, 'KERNEL_TRAINING_REPORT')
    first = next(n for n in spec['graph']['nodes'] if n['coordinate'] == 'D066')
    if (value['passed'] is not True or value['source_commit'] != spec['source_commit']
            or value['counts'] != spec['counts'] or value['site'] != spec['execution_site']
            or value['gpu'] != parent['runtime_profile']['gpu']
            or value['installed_environment'] != parent['runtime_profile']['installed_environment']
            or parity['model'] != parent['model'] or parity['device'] != 'cuda'
            or parity['passed'] is not True or parity['forward_and_feature_and_parameter_gradients'] is not True
            or kd['node'] != first or kd['foundation_sha256'] != spec['input_identity']
            or kd['recipe_sha256'] != recipe()['content_hash'] or kd['passes'] != 1
            or kd['acceptance_only'] is not True or kd['scientific_fit'] is not False
            or kd['final_test_accessed'] is not False
            or value['offline_bank'] != spec['frozen']['offline_bank']
            or value['endpoint_sha256'] != spec['frozen']['endpoint_sha256']
            or value['replay']['exact'] is not True or value['replay']['coordinates'] != list(views.COORDINATES)
            or value['replay']['jets'] != min(32, spec['counts']['train'])
            or len(value['replay']['hashes']) != 3*value['replay']['jets']
            or not 0 < value['gpu_peak_bytes'] <= .85*value['gpu']['total_memory_bytes']
            or not 0 < value['cache_bytes'] <= .75*spec['memory_mb']*1024**2):
        raise ValueError('S3 technical acceptance differs')
    times = (value['cache_seconds'], kd['runtime_seconds'], value['inference_seconds'])
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Invalid S3 timing')
    expected = walltimes(parent, *times)
    if any(value[k] != v or not 30 <= v <= 1440 for k, v in expected.items()):
        raise ValueError('S3 resource envelope differs')
    return value


def preflight(spec, parent):
    return validate_preflight(load_json(Path(spec['root'])/'preflight.json'), spec, parent)


def tasks(spec):
    rows = []
    for n in spec['graph']['nodes']:
        deps = [] if n['teacher'] == 'OFFLINE' else ['reduce_'+n['teacher']]
        rows.append(dict(task_id='fit_'+n['node_id'], dependencies=deps, kind='fit', node_id=n['node_id']))
        if n['node_id'] in spec['graph']['reducers']:
            rows.append(dict(task_id='reduce_'+n['node_id'], dependencies=['fit_'+n['node_id']],
                             kind='reduce', node_id=n['node_id']))
    rows.append(dict(task_id='summary', dependencies=['fit_'+n['node_id'] for n in spec['graph']['nodes']], kind='summary'))
    return rows


def plan(spec, mode):
    parent = validate_spec(spec)
    if mode == 'gate':
        rows = [dict(task_id='preflight', dependencies=[], kind='preflight')]
        measured, parents = {}, dict(spec=spec['content_hash'])
    elif mode == 'science':
        measured = preflight(spec, parent)
        rows, parents = tasks(spec), dict(spec=spec['content_hash'], preflight=measured['content_hash'])
    else:
        raise ValueError('Unknown S3 stage')
    commands = []
    site, project, root = spec['execution_site'], spec['project_dir'], Path(spec['root'])
    for row in rows:
        task, deps, kind = row['task_id'], row['dependencies'], row['kind']
        gpu = kind != 'summary'
        minutes = 120 if kind == 'preflight' else 60 if kind == 'summary' else measured[kind+'_minutes' if kind == 'reduce' else 'train_minutes']
        wrap = '\n'.join(('set -euo pipefail', 'export PROJECT_DIR='+shlex.quote(project),
            'export JC2_SITE='+shlex.quote(site['name']),
            'source '+shlex.quote(project+'/sbatch/jetclass2_delphes_common.sh'),
            'python -u -s '+shlex.quote(project+'/scripts/jetclass2_s3_ladder.py')+
            ' run --spec '+shlex.quote(str(root/'study_spec.json'))+' --task '+shlex.quote(task)))
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus'] if gpu else 1), '--mem='+str(spec['memory_mb'] if gpu else 16000)+'M',
            '--time='+_walltime(minutes), '--job-name=jc2s3_'+task,
            '--comment=jc2s3:'+spec['content_hash']+':'+task, '--chdir='+project,
            '--output='+str(root/'slurm-%j.out')]
        if gpu:
            command.append('--gres='+site['gres'])
        if deps:
            command.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps))
        commands.append(dict(task_id=task, dependencies=deps, command=command+['--wrap='+wrap]))
    return artifact('PLAN', parents=parents, mode=mode, commands=commands, automatic_followup=False)


def submit(spec, *, mode, execute=False, plan_hash=None, authorization=None):
    value = plan(spec, mode)
    root = Path(spec['root'])/('submission_'+mode)
    root.mkdir(exist_ok=True)
    path = root/'command_plan.json'
    if path.exists() and load_json(path) != value:
        raise ValueError('Saved S3 plan changed')
    if not path.exists():
        write_json(path, value)
    if not execute:
        submit_exact_dag(identity=spec['content_hash'], plan=value, output=root/'dry_run_submission_ledger.json',
                         canonical_dry_run=root/'dry_run_submission_ledger.json', execute=False)
        return value
    if authorization != AUTHORIZATION or plan_hash != value['content_hash']:
        raise PermissionError('Explicit S3 authority and reviewed exact hash required')
    check_submission_site(value)
    return submit_claimed(spec, value, root)


def teacher_binding(spec, parent, name):
    if name == 'OFFLINE':
        receipt = load_json(validate_file_ref(spec['frozen']['offline_reducer']))
        report = load_json(validate_file_ref(spec['frozen']['offline_report']))
        path = validate_file_ref(spec['frozen']['offline_bank']).parent
        return dict(receipt=receipt['content_hash'], path=path, foundation=parent['foundation']['content_hash'],
                    report=report['content_hash'], node='OFFLINE')
    receipt = completed_reducer(spec, parent, name)
    report = load_json(validate_file_ref(receipt['bank'], root=Path(spec['root'])/('reduce_'+name)))
    return dict(receipt=receipt['content_hash'], path=Path(spec['root'])/('reduce_'+name)/'train',
                foundation=spec['input_identity'], report=report['teacher_report_sha256'], node=name)


def teacher_probabilities(spec, parent, name, ids):
    binding = teacher_binding(spec, parent, name)
    p = load_bank(binding['path'], foundation_sha256=binding['foundation'],
        teacher_report_sha256=binding['report'], teacher_node=binding['node'], role='train', expected_identities=ids)
    return p, binding['receipt']


def completed_fit(spec, parent, name):
    n = node(spec, name)
    root = Path(spec['root'])/('fit_'+name)
    value = load_json(root/'receipt.json')
    teacher = teacher_binding(spec, parent, n['teacher'])
    validate(value, 'FIT', parents=dict(spec=spec['content_hash'], preflight=preflight(spec, parent)['content_hash'],
                                       teacher=teacher['receipt']))
    validate_file_ref(value['checkpoint'], root=root)
    training = load_json(validate_file_ref(value['training'], root=root))
    validate_training(training, 'KERNEL_TRAINING_REPORT')
    if (value['node'] != n or value['source_commit'] != spec['source_commit']
            or value['counts'] != spec['counts'] or training['node'] != n
            or training['foundation_sha256'] != spec['input_identity']
            or training['recipe_sha256'] != recipe()['content_hash']
            or training['scientific_fit'] is not True or training['acceptance_only'] is not False
            or training['final_test_accessed'] is not False or training['selected_weights_restored'] is not True
            or not recipe()['minimum_passes'] <= training['passes'] <= recipe()['maximum_passes']
            or not 1 <= training['selected_pass'] <= training['passes']
            or len(training['validation_history']) != training['passes']):
        raise ValueError('S3 complete fit differs')
    if n['coordinate'] == 'D000' and value['endpoint_sha256'] != spec['frozen']['endpoint_sha256']:
        raise ValueError('S3 final fit endpoint drift')
    return value, training


def completed_reducer(spec, parent, name):
    if name not in spec['graph']['reducers']:
        raise ValueError('Unregistered S3 reducer')
    fit, training = completed_fit(spec, parent, name)
    root = Path(spec['root'])/('reduce_'+name)
    value = load_json(root/'receipt.json')
    validate(value, 'REDUCER', parents=dict(spec=spec['content_hash'], fit=fit['content_hash']))
    path = validate_file_ref(value['bank'], root=root)
    if (path != (root/'train/manifest.json').resolve() or value['node_id'] != name
            or value['source_commit'] != spec['source_commit']
            or value['train_endpoint_sha256'] != fit['endpoint_sha256']['train']):
        raise ValueError('S3 reducer lineage differs')
    load_bank(path.parent, foundation_sha256=spec['input_identity'], teacher_report_sha256=training['content_hash'],
              teacher_node=name, role='train', expected_identities=cache.identities(parent, 'train'))
    return value


def result_rows(spec, parent, *, complete=False):
    reports = {n: load_json(validate_file_ref(spec['frozen'][key]))
               for n, key in (('OFFLINE', 'offline_report'), ('S3', 's3_report'))}
    for n in spec['graph']['nodes']:
        name = n['node_id']
        if (Path(spec['root'])/('fit_'+name)/'receipt.json').exists():
            reports[name] = completed_fit(spec, parent, name)[1]
        elif complete:
            raise ValueError('All four S3 fits are required')
    rows = []
    for name in ['OFFLINE', 'S3', *(n['node_id'] for n in spec['graph']['nodes'])]:
        report = reports.get(name)
        m = report['validation'] if report else None
        rows.append(dict(node_id=name, state='COMPLETE' if report else 'NOT_COMMITTED',
            passes=report['passes'] if report else None, selected_pass=report['selected_pass'] if report else None,
            validation=m, recovery=None if m is None else recovery(m, reports['S3']['validation'], reports['OFFLINE']['validation'])))
    return rows


def summarize(spec, parent):
    rows = result_rows(spec, parent, complete=True)
    direct, coarse = rows[2]['validation'], rows[-1]['validation']
    return artifact('SUMMARY', parents=dict(spec=spec['content_hash'], **{
        n['node_id']: completed_fit(spec, parent, n['node_id'])[0]['content_hash'] for n in spec['graph']['nodes']}),
        rows=rows, final_delta={k: coarse[k]-direct[k] for k in ('accuracy', 'macro_ovr_auc')},
        recovery_reference=dict(zero='S3', hundred='OFFLINE'), development_only=True,
        repeated_validation=True, significance_established=False, compute_matched=False)
