"""One authorized Oscar workflow: generation -> runtime checks -> science DAG."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from .contracts import artifact, validate, require, write, checked, load_json, sha256_file, ref

AUTHORIZATION = 'AUTHORIZE CONTEXT V2 DATASET AND DIRECT COARSE OSCAR WORKFLOW'
COUNTS = dict(train=100000, validation=50000)
EXTRA_SOURCE = (
    'docs/plans/JETCLASS2_CONTEXT_V2_OSCAR_WORKFLOW_PLAN.md',
    'docs/contracts/JETCLASS2_CONTEXT_V2_OSCAR_WORKFLOW.md',
    'scripts/jetclass2_context_v2_workflow.py',
    'scripts/queue_jetclass2_context_v2_workflow.sh',
    'tests/test_context_v2.py',
)


def policy():
    return dict(counts=dict(COUNTS), branches=['DIRECT', 'COARSE'], fits=9, reducers=5,
        science_jobs=16, science_site='oscar_l40s', max_train_minutes=2880,
        max_reduce_minutes=1440, cpus=6, memory_mb=90000, max_gpus_per_task=1,
        generation_workers=6, generation_concurrency=8, extra_pilots=0,
        final_test_evaluated=False, strength_selection='one_frozen_change_no_metric_gate')


def source_lock(project, commit):
    from hlt_classification.cms_proxy_ladder.gate import source_lock as base
    root = Path(project).resolve(strict=True)
    lock = base(root, commit, literature=True, context=True)
    files = dict(lock['files'])
    for path in [root/p for p in EXTRA_SOURCE] + sorted((root/'src/hlt_classification/context_v2').glob('*.py')):
        relative = path.relative_to(root).as_posix()
        subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', relative],
                       check=True, capture_output=True)
        files[relative] = sha256_file(path)
    return artifact('SOURCE', commit=commit, files=files)


def tasks(spec):
    return [dict(task_id='generate', dependencies=[], cpus=6, memory_mb=32000, minutes=120,
                 partition='batch', array=f"0-{len(spec['shards'])-1}%8"),
        dict(task_id='finalize', dependencies=['generate'], cpus=1, memory_mb=16000, minutes=120, partition='batch'),
        dict(task_id='prepare', dependencies=['finalize'], cpus=6, memory_mb=90000, minutes=480, partition='batch'),
        dict(task_id='preflight', dependencies=['prepare'], cpus=6, memory_mb=90000, minutes=720, partition='gpu'),
        dict(task_id='launch_science', dependencies=['preflight'], cpus=1, memory_mb=8000, minutes=120, partition='batch')]


def plan(spec):
    from hlt_classification.cms_proxy_ladder.submission import _walltime
    commands = []
    for task in tasks(spec):
        name, project, root = task['task_id'], spec['project_dir'], spec['root']
        wrap = '\n'.join(('set -euo pipefail', f'export PROJECT_DIR={shlex.quote(project)}',
            'export JC2_SITE=oscar_l40s',
            f'source {shlex.quote(project + "/sbatch/jetclass2_delphes_common.sh")}',
            f'python -u -s {shlex.quote(project + "/scripts/jetclass2_context_v2_workflow.py")} '
            f'worker --spec {shlex.quote(root + "/workflow_spec.json")} --task {name}'))
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--account=default', f"--partition={task['partition']}", f"--cpus-per-task={task['cpus']}",
            f"--mem={task['memory_mb']}M", f"--time={_walltime(task['minutes'])}",
            f'--job-name=jc2ctx2_{name}', f"--comment=ctx2:{spec['content_hash']}:{name}",
            f'--chdir={project}', f'--output={root}/slurm-%j.out']
        if 'array' in task:
            args += [f"--array={task['array']}", f'--output={root}/slurm-%A_%a.out']
        if task['partition'] == 'gpu':
            args += ['--qos=norm-gpu', '--gres=gpu:l40s:1']
        if task['dependencies']:
            args += ['--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in task['dependencies'])]
        args += ['--wrap='+wrap]
        commands.append(dict(task_id=name, dependencies=task['dependencies'], command=args))
    return artifact('PLAN', parents=dict(workflow=spec['content_hash']), commands=commands,
        followup_policy=policy(), automatic_science=True, live_authorization_phrase=AUTHORIZATION,
        note='Five initial submissions including generation array; then 16 measured science jobs automatically.')


def environment():
    return {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_'))}


def site_checks(value):
    config = subprocess.run(['scontrol', 'show', 'config'], capture_output=True,
                            text=True, check=True, env=environment()).stdout
    require(re.search(r'(?m)^\s*ClusterName\s*=\s*slurmctld\s*$', config), 'Submit from Oscar only')
    shapes = set()
    for row in value['commands']:
        args = [a for a in row['command'] if not a.startswith(('--dependency=', '--wrap=', '--array='))]
        shape = tuple(a for a in args if a.startswith(('--partition=', '--qos=', '--account=',
            '--cpus-per-task=', '--mem=', '--time=', '--gres=')))
        if shape in shapes:
            continue
        shapes.add(shape)
        args[1:1] = ['--test-only']
        result = subprocess.run(args+['--wrap=true'], capture_output=True, text=True, env=environment())
        require(result.returncode == 0, 'Oscar rejected resource shape: '+result.stdout+result.stderr)


def authorization(spec, value):
    return artifact('AUTHORIZATION', parents=dict(workflow=spec['content_hash'], plan=value['content_hash']),
        phrase=AUTHORIZATION, followup_policy=policy(), authorize_materialization=True, authorize_science=True)


def submit(spec, *, execute=False, reviewed_hash=None, phrase=None):
    from .dataset import validate_spec
    from hlt_classification.cms_proxy_ladder.context import submit_claimed
    from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
    validate_spec(spec, check_source=True)
    root, value = Path(spec['root']), plan(spec)
    if execute:
        require(reviewed_hash == value['content_hash'] and phrase == AUTHORIZATION,
                'Exact reviewed plan and dataset+science authorization required')
        require(load_json(root/'command_plan.json') == value, 'Reviewed plan drift')
        site_checks(value)
        write(root/'authorization.json', authorization(spec, value))
        return submit_claimed(spec, value, root)
    write(root/'command_plan.json', value)
    submit_exact_dag(identity=spec['content_hash'], plan=value,
        output=root/'dry_run_submission_ledger.json', canonical_dry_run=root/'dry_run_submission_ledger.json', execute=False)
    return value


def _bound_job(spec, name):
    """Bind allocation to the accepted exact-ID prefix, including fast array starts."""
    from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
    from hlt_classification.jetclass2_delphes.execution import allocation, execution_site
    require(name in {t['task_id'] for t in tasks(spec)}, 'Unregistered workflow task')
    root, value = Path(spec['root']), plan(spec)
    require(load_json(root/'authorization.json') == authorization(spec, value), 'Workflow authority differs')
    require(load_json(root/'command_plan.json') == value, 'Workflow plan differs')
    job = None
    for _ in range(31):
        _, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=spec['content_hash'], plan=value)
        job = jobs.get(name)
        if job is not None:
            break
        time.sleep(2)
    require(job is not None, 'No accepted submission journal entry for this worker')
    task = next(t for t in tasks(spec) if t['task_id'] == name)
    actual = os.environ.get('SLURM_ARRAY_JOB_ID') if name == 'generate' else os.environ.get('SLURM_JOB_ID')
    require(actual == job and int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) == task['cpus'], 'Allocation/job identity differs')
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1'))
    if name == 'generate':
        require(0 <= index < len(spec['shards']), 'Unregistered array index')
    selector = f'{job}_{index}' if name == 'generate' else job
    result = subprocess.run(['scontrol', 'show', 'job', '-o', selector], capture_output=True, text=True, check=True)
    fields = dict(token.split('=', 1) for token in result.stdout.split() if '=' in token)
    require(fields.get('JobName') == 'jc2ctx2_'+name and fields.get('Comment') == f"ctx2:{spec['content_hash']}:{name}"
        and fields.get('Account') == 'default' and fields.get('Partition') == task['partition']
        and fields.get('UserId', '').split('(')[0] == os.environ.get('USER')
        and fields.get('WorkDir') == spec['project_dir'] and fields.get('JobState') == 'RUNNING'
        and fields.get('CPUs/Task') == str(task['cpus'])
        and fields.get('NumNodes') == '1' and fields.get('NumTasks') == '1'
        and os.environ.get('SLURM_CLUSTER_NAME') == 'slurmctld'
        and int(os.environ.get('SLURM_MEM_PER_NODE', '0')) == task['memory_mb'], 'Scheduler task binding differs')
    if name == 'generate':
        require(fields.get('ArrayJobId') == job and fields.get('ArrayTaskId') == str(index), 'Array allocation differs')
    else:
        require(fields.get('JobId') == job, 'Job ID differs')
    require(os.environ.get('PYTHONNOUSERSITE') == '1' and os.environ.get('JC2_SITE') == 'oscar_l40s'
        and os.environ.get('CONDA_PREFIX') == sys.prefix
        and sys.prefix == '/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar'
        and os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == sys.prefix+'/lib', 'Oscar environment differs')
    if name == 'preflight':
        allocation(execution_site('oscar_l40s'))
    else:
        require('gres/gpu' not in fields.get('ReqTRES', '') and 'gres/gpu' not in fields.get('AllocTRES', ''), 'CPU worker unexpectedly requested a GPU')
    return index


def check_science(spec, campaign, value):
    from hlt_classification.cms_proxy_ladder import context_v2 as adapter, submission
    adapter.validate_campaign(campaign, check_source=True)
    require(value == submission.science_plan(campaign) and len(value['commands']) == 16,
            'Science plan differs from frozen direct/coarse policy')
    require(campaign['gate_root'] == str(Path(spec['root'])/'gate')
        and campaign['campaign_root'] == str(Path(spec['root'])/'science')
        and campaign['foundation']['identity_sha256'] == spec['selection_identities']
        and campaign['foundation']['role_counts'] == policy()['counts']
        and campaign['source'] == spec['source'] and campaign['source_commit'] == spec['source_commit'],
        'Science plan escaped authorized workflow')
    profile = campaign['runtime_profile']
    require(profile['cpus'] == 6 and profile['memory_mb'] == 90000
        and profile['train_minutes'] <= policy()['max_train_minutes']
        and profile['reduce_minutes'] <= policy()['max_reduce_minutes'], 'Measured science exceeds authorized resource bounds')


def run(spec, name):
    from . import dataset
    from hlt_classification.cms_proxy_ladder import context_v2 as adapter, gate, submission
    print(f'CONTEXT_V2 task={name} phase=authenticate source_checks_pending=true', flush=True)
    dataset.validate_spec(spec, check_source=True)
    index = _bound_job(spec, name)
    root = Path(spec['root'])
    print(f'CONTEXT_V2 task={name} phase=start no_metric_admission_gate=true', flush=True)
    if name == 'generate':
        return dataset.generate(spec, index, workers=6)
    if name == 'finalize':
        return dataset.finalize(spec)
    with dataset.claim(root, name):
        if name == 'prepare':
            g = adapter.create_gate(spec)
            gate.run_gate_task(g, 'authenticate_release')
            return gate.run_gate_task(g, 'build_foundation')
        if name == 'preflight':
            return gate.run_gate_task(load_json(root/'gate/gate_spec.json'), 'preflight')
        campaign = adapter.create_campaign(gate_root=root/'gate', campaign_root=root/'science')
        value = submission.science_plan(campaign)
        check_science(spec, campaign, value)
        # Automatic followup was explicitly authorized in the initial plan.
        submission.submit(subject=campaign, mode='science', execute=False, authorization_phrase=None)
        ledger = submission.submit(subject=campaign, mode='science', execute=True,
                                   authorization_phrase=adapter.SCIENCE_AUTHORIZATION)
        receipt = artifact('FOLLOWUP', parents=dict(workflow=spec['content_hash'], campaign=campaign['content_hash'],
            plan=value['content_hash']), jobs=ledger['jobs'], automatic_science=True, policy=policy())
        write(root/'science_submission_receipt.json', receipt)
        print('SCIENCE QUEUED: all 16 direct/coarse jobs; final test untouched.', flush=True)
        return receipt
