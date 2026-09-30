"""Exact-plan CPU arrays, durable submission journals and no-overlap recovery."""
import getpass
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from . import campaign as c, storage
from .contracts import artifact, validate, safe, write, load_json, checked

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}


def task_specs(attempt):
    r = attempt['resources']
    tasks = []
    if attempt['kind'] == 'pilot':
        tasks.append(dict(task='preflight', cpus=4, memory_gib=32, hours=4, dependency=None))
    if attempt['kind'] != 'finalize':
        tasks.append(dict(task='generate', cpus=36, memory_gib=r['memory_gib'], hours=r['hours'],
            dependency='afterok:preflight' if attempt['kind'] == 'pilot' else None,
            array=f'0-{len(attempt["shards"])-1}%{r["concurrent"]}'))
    if attempt['kind'] != 'pilot':
        tasks.append(dict(task='finalize', cpus=1, memory_gib=32, hours=4,
            dependency=None if attempt['kind'] == 'finalize' else 'afterany:generate'))
    return tasks


def plan(study, attempt):
    commands = []
    directory = safe(study['root'], f'attempts/{attempt["name"]}')
    for task in task_specs(attempt):
        name = task['task']
        argv = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition=tigris', '--account=reu-aisocial', f'--cpus-per-task={task["cpus"]}',
            f'--mem={task["memory_gib"]}G', f'--time={task["hours"]:02d}:00:00',
            f'--job-name=c2jp_{name}', f'--comment=c2jp:{attempt["content_hash"]}:{name}',
            f'--chdir={study["project_dir"]}',
            f'--output={directory}/slurm-{name}-%A_%a.out']
        if task.get('array'):
            argv.append('--array='+task['array'])
        argv += [str(Path(study['project_dir'])/'sbatch/run_cms2jc2_proxy_dataset_cpu.sh'),
                 study['project_dir'], str(directory/'attempt_spec.json'), name]
        commands.append(dict(**task, argv=argv))
    return artifact('PLAN', parents={'attempt': attempt['content_hash']}, commands=commands,
        gpus=0, generation_cpu_cap=576, maximum_concurrent_generation_tasks=16,
        authorization_phrase=f'AUTHORIZE CMS2JC2 PROXY {attempt["kind"].upper()} EXACT PLAN')


def get_plan(attempt_path):
    attempt = load_json(attempt_path)
    study = load_json(checked(attempt['study']))
    c.validate_study(study)
    c.validate_attempt(attempt, study)
    saved = load_json(Path(attempt_path).parent/'command_plan.json')
    if saved != plan(study, attempt):
        raise ValueError('Reviewed command plan differs')
    return study, attempt, saved


def fields(job):
    result = subprocess.run(['scontrol', 'show', 'job', '-o', str(job)],
                            check=True, capture_output=True, text=True)
    return dict(re.findall(r'(?:^|\s)([\w/]+)=([^\s]+)', result.stdout))


def submitted_identity(study, attempt, task, *, wait_for_response=False):
    """Bind a worker to its exact durable submission, including array parent ID."""
    saved = load_json(safe(study['root'], f'attempts/{attempt["name"]}/command_plan.json'))
    if saved != plan(study, attempt):
        raise ValueError('Worker command plan differs')
    number = next(i for i, r in enumerate(saved['commands']) if r['task'] == task)
    row = saved['commands'][number]
    prefix = f'attempts/{attempt["name"]}/submission/{number:02d}_{task}'
    response_path = safe(study['root'], prefix+'_response.json')
    # sbatch may start a small job before its client has published the response.
    for _ in range(60 if wait_for_response else 1):
        if response_path.is_file():
            break
        if wait_for_response:
            time.sleep(1)
    else:
        raise PermissionError('No durable scheduler response; ambiguous submission cannot execute')
    parents = {'attempt': attempt['content_hash'], 'plan': saved['content_hash']}
    response = load_json(response_path)
    validate(response, 'SUBMIT_RESPONSE', parents=parents, test=False)
    if response['returncode'] != 0 or not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', response['stdout'].strip()):
        raise PermissionError('Submission response is not a successful exact job ID')
    intent = load_json(safe(study['root'], prefix+'_intent.json'))
    validate(intent, 'SUBMIT_INTENT', parents=parents, test=False)
    argv = list(row['argv'])
    if row['dependency']:
        kind, parent = row['dependency'].split(':')
        argv.insert(1, '--dependency='+kind+':'+submitted_identity(study, attempt, parent))
    if argv != intent['argv']:
        raise PermissionError('Worker command differs from submitted exact plan')
    return response['stdout'].strip().split(';')[0]


def worker_identity(study, attempt, task, index):
    row = next((r for r in task_specs(attempt) if r['task'] == task), None)
    env = os.environ
    if row is None or platform.system() != 'Linux':
        raise PermissionError('A registered Linux Slurm CPU worker is required')
    if (env.get('CONDA_PREFIX') != c.ENVIRONMENT or env.get('SLURM_JOB_PARTITION') != 'tigris'
            or env.get('SLURM_JOB_ACCOUNT') != 'reu-aisocial'
            or env.get('SLURM_CPUS_PER_TASK') != str(row['cpus'])
            or env.get('SLURM_JOB_NUM_NODES') != '1'
            or env.get('PYTHONNOUSERSITE') != '1' or env.get('PYTHONDONTWRITEBYTECODE') != '1'
            or not env.get('LD_LIBRARY_PATH', '').startswith(c.ENVIRONMENT+'/lib')
            or any(env.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            or any(env.get(k, '') not in ('', '0', 'NoDevFiles') for k in
                ('SLURM_GPUS', 'SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE'))):
        raise PermissionError('Allocated CPU-only Tigris environment/resources differ')
    job = env.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise PermissionError('Missing numeric scheduler job ID')
    expected_job = submitted_identity(study, attempt, task, wait_for_response=True)
    if (env.get('SLURM_ARRAY_JOB_ID') if task == 'generate' else job) != expected_job:
        raise PermissionError('Job does not belong to this registered submission')
    f = fields(job)
    if (f.get('Comment') != f'c2jp:{attempt["content_hash"]}:{task}'
            or f.get('JobName') != 'c2jp_'+task or f.get('Partition') != 'tigris'
            or f.get('Account') != 'reu-aisocial' or f.get('NumCPUs') != str(row['cpus'])
            or f.get('CPUs/Task') != str(row['cpus']) or f.get('NumNodes') != '1'
            or f.get('NumTasks') != '1'
            or f.get('TimeLimit') != f'{row["hours"]:02d}:00:00'
            or f.get('JobState') != 'RUNNING' or f.get('WorkDir') != study['project_dir']
            or f.get('UserId', '').split('(')[0] != getpass.getuser()
            or 'gres/gpu' in f.get('ReqTRES', '')):
        raise PermissionError('Scheduler identity/resources differ')
    memory = re.fullmatch(r'(\d+)([KMGTP]?)', f.get('MinMemoryNode', ''))
    if (memory is None or int(memory.group(1))*1024**({'': 2, 'K': 1, 'M': 2, 'G': 3, 'T': 4, 'P': 5}[memory.group(2)])
            != row['memory_gib']*2**30):
        raise PermissionError('Scheduler memory reservation differs')
    if task == 'generate':
        if (type(index) is not int or not 0 <= index < len(attempt['shards'])
                or env.get('SLURM_ARRAY_TASK_ID') != str(index)
                or f.get('ArrayTaskId') != str(index)
                or f.get('ArrayJobId') != env.get('SLURM_ARRAY_JOB_ID')):
            raise PermissionError('Array element escaped exact shard registration')
    elif index is not None:
        raise PermissionError('Non-array task received an array index')
    return dict(job_id=job, array_job_id=env.get('SLURM_ARRAY_JOB_ID'), index=index,
                partition='tigris', account='reu-aisocial', cpus=row['cpus'], gpus=0,
                scheduler=f, host=platform.node())


def accounting(job_ids):
    if not job_ids or any(not str(j).isdigit() for j in job_ids):
        raise ValueError('Exact numeric scheduler IDs required')
    result = subprocess.run(['sacct', '--array', '-j', ','.join(map(str, job_ids)), '-X', '-n', '-P',
        '-o', 'JobID%80,State,ExitCode,User,Account,JobName%80'],
        check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if len(parts) < 6 or not parts[0]:
            continue
        job, state, exit_code, user, account, name = parts[:6]
        if not any(job == j or job.startswith(j+'_') for j in job_ids):
            continue
        if user != getpass.getuser() or account != 'reu-aisocial' or not name.startswith('c2jp_'):
            raise PermissionError('Accounting job ownership/campaign differs')
        rows.append(dict(job=job, state=state.split()[0].rstrip('+'), exit_code=exit_code, name=name))
    return rows


def require_terminal(study, *, exclude=None):
    for path in sorted(safe(study['root'], 'attempts').glob('*/attempt_spec.json')):
        attempt = load_json(path)
        if attempt['name'] == exclude:
            continue
        ledger_path = path.parent/'submission_ledger.json'
        if not ledger_path.exists():
            raise PermissionError('Prior attempt unsubmitted/ambiguous; reconcile before creating another')
        ledger = load_json(ledger_path)
        validate(ledger, 'LEDGER', parents={'attempt': attempt['content_hash']}, test=False)
        ids = list(ledger['jobs'].values())
        expected_tasks = {r['task'] for r in task_specs(attempt)}
        if set(ledger['jobs']) != expected_tasks or ledger['dry_run'] is not False:
            raise ValueError('Previous live ledger differs')
        if ledger['plan'] != plan(study, attempt)['content_hash'] or any(
                submitted_identity(study, attempt, task) != job for task, job in ledger['jobs'].items()):
            raise ValueError('Previous ledger does not match its exact submission journal')
        rows = accounting(ids)
        for task, job in ledger['jobs'].items():
            bound = [r for r in rows if r['job'] == job or r['job'].startswith(job+'_')]
            if not bound or any(r['state'] not in TERMINAL for r in bound):
                raise PermissionError(f'Preserve prior active/unknown job {job}; no overlapping attempts')
            expected_ids = {f'{job}_{i}' for i in range(len(attempt['shards']))} if task == 'generate' else {job}
            if not expected_ids <= {r['job'] for r in bound} or any(r['name'] != 'c2jp_'+task for r in bound):
                raise PermissionError('Incomplete or wrong accounting records; do not overlap attempts')
        # Explicit current-queue check closes sacct lag and compressed-array ambiguity.
        result = subprocess.run(['squeue', '-h', '-j', ','.join(ids), '-o', '%i|%T'],
                                capture_output=True, text=True)
        if result.stdout.strip():
            raise PermissionError('Prior attempt still present in live queue')
        if result.returncode and 'Invalid job id' not in result.stderr:
            raise RuntimeError('Could not verify absence from live scheduler')


def submit(attempt_path, *, execute=False, plan_hash=None, phrase=None):
    study, attempt, saved = get_plan(attempt_path)
    if not execute:
        return saved
    if saved['content_hash'] != plan_hash or saved['authorization_phrase'] != phrase:
        raise PermissionError('Exact reviewed plan hash and authorization phrase required')
    c.validate_study(study, source=True)
    require_terminal(study, exclude=attempt['name'])
    if attempt['kind'] != 'pilot':
        c.require_preflight(study)
        c.require_admission(study, attempt)
        c.test_lock(study)
    root = Path(study['root'])
    directory = Path(attempt_path).resolve().parent
    if directory != safe(root, f'attempts/{attempt["name"]}'):
        raise PermissionError('Wrong immutable attempt path')
    with storage.lock(root, 'submission'):
        if (directory/'submission_ledger.json').exists() or (directory/'submission').exists():
            raise PermissionError('Submission already attempted; inspect journal, never submit twice')
        # Test exact shapes, no dependencies and no jobs actually submitted here.
        for row in saved['commands']:
            result = subprocess.run([row['argv'][0], '--test-only', *row['argv'][1:]],
                                    env=clean_env(), capture_output=True, text=True)
            if result.returncode:
                raise PermissionError('Site rejected exact CPU shape: '+result.stdout+result.stderr)
        (directory/'submission').mkdir()
        jobs = {}
        for number, row in enumerate(saved['commands']):
            argv = list(row['argv'])
            if row['dependency']:
                kind, parent = row['dependency'].split(':')
                argv.insert(1, f'--dependency={kind}:{jobs[parent]}')
            prefix = directory/'submission'/f'{number:02d}_{row["task"]}'
            write(str(prefix)+'_intent.json', artifact('SUBMIT_INTENT',
                parents={'attempt': attempt['content_hash'], 'plan': saved['content_hash']}, argv=argv))
            result = subprocess.run(argv, env=clean_env(), capture_output=True, text=True)
            text = result.stdout.strip()
            write(str(prefix)+'_response.json', artifact('SUBMIT_RESPONSE',
                parents={'attempt': attempt['content_hash'], 'plan': saved['content_hash']},
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            if result.returncode or not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?', text):
                raise RuntimeError('Ambiguous/failed submission; journal preserved, no automatic retry')
            jobs[row['task']] = text.split(';')[0]
        ledger = artifact('LEDGER', parents={'attempt': attempt['content_hash']},
                          plan=saved['content_hash'], jobs=jobs, dry_run=False)
        write(directory/'submission_ledger.json', ledger)
    return ledger
