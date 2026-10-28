"""Two-lane afterany capacity edges; afterok only for actual teacher inputs."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from .contracts import artifact, AUTHORIZATION
from . import campaign, source


def render(spec):
    original = source.original(spec['original']); site = original['execution_site']; rows = []
    for task in spec['tasks']:
        name = task['task_id']; report = name == 'report'
        cpus = original['report_cpus'] if report else original['cpus']
        memory = original['report_memory_mb'] if report else original['memory_mb']
        cmd = ['sbatch', '--parsable', '--no-requeue', '--kill-on-invalid-dep=yes', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(cpus), '--mem='+str(memory)+'M', '--time='+_walltime(original['task_minutes'][name]),
            '--job-name=lukasr_'+name, '--comment=lukasr:'+spec['content_hash']+':'+name,
            '--chdir='+spec['project_dir'], '--output='+str(Path(spec['root'])/(name+'-%j.out'))]
        if not report:
            cmd.append('--gres='+site['gres'])
        clauses = []
        for kind, key in (('afterok', 'success_dependencies'), ('afterany', 'slot_dependencies')):
            if task[key]:
                clauses.append(kind+':'+':'.join('${JOB_'+d+'}' for d in task[key]))
        if clauses:
            cmd.append('--dependency='+','.join(clauses))
        cmd += [spec['project_dir']+'/sbatch/run_luka_depth_recovery.sh', spec['project_dir'],
            str(Path(spec['root'])/'campaign_spec.json'), name]
        rows.append(dict(task_id=name, dependencies=task['dependencies'], command=cmd))
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=rows, jobs=len(rows),
        reused_tasks=list(spec['reused']), restarted_tasks=list(spec.get('restarted', {})),
        max_gpu_jobs=2, preflight_jobs=0, target_exports=0,
        scientific_fits=sum(r['task_id'] in original['recipes'] for r in rows),
        teacher_fits=sum(r['task_id'] in original['teacher_recipes'] for r in rows),
        authorization=AUTHORIZATION, payload_deletion=False)


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    original = campaign.check(spec); plan = render(spec)
    folder = Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved, dry, live = (folder/n for n in ('command_plan.json', 'dry_run_submission_ledger.json', 'submission_ledger.json'))
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION
            or not saved.is_file() or load_json(saved) != plan or not dry.is_file()):
        raise PermissionError('Exact full dry review and recovery authorization required')
    # Fail before submitting anything if another writer/partial fit appeared.
    for task in spec['tasks']:
        name = task['task_id']
        target = Path(spec['root'])/'report' if name == 'report' else Path(original['root'])/name
        if target.exists() or target.is_symlink():
            raise FileExistsError('Output already exists; preserve it: ' + str(target))
    write_immutable_json(saved, plan); check_site(plan)
    if execute:
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json', artifact('SUBMIT', parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    value = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=live if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return value if execute else plan


def accepted_jobs(spec):
    folder = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(folder/'submission_ledger_journal', identity=spec['content_hash'], plan=render(spec))
    if (folder/'submission_ledger.json').exists() and load_json(folder/'submission_ledger.json') != assemble_submission_ledger(events, campaign_spec_sha256=spec['content_hash']):
        raise ValueError('Recovery ledger differs')
    return jobs


def successful(spec, task, job):
    if accepted_jobs(spec).get(task) != job:
        raise ValueError('Not an accepted recovery job')
    p = subprocess.run(['sacct', '-j', job, '-X', '-n', '-P', '-o', 'JobIDRaw,JobName%80,User,Account,State,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = [s.strip().split('|')[:6] for s in p.stdout.splitlines() if s.strip()]
    original = source.original(spec['original'])
    if [r for r in rows if r[0] == job] != [[job, 'lukasr_'+task, getpass.getuser(), original['execution_site']['account'], 'COMPLETED', '0:0']]:
        raise ValueError('Recovery ancestor must be exact COMPLETED/0:0')
