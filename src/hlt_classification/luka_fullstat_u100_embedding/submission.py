"""One exact export->fit submission; no controller, preflight or old-job mutation."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.luka_fullstat.submission import check_site, environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from .contracts import artifact, AUTHORIZATION
from . import campaign


def render(spec):
    root, site = Path(spec['root']), spec['execution_site']
    commands = []
    for task in spec['tasks']:
        name = task['task_id']
        minutes = spec['export_minutes'] if name == 'export_U100' else spec['train_minutes']
        if type(minutes) is not int or not 1 <= minutes <= 2880:
            raise ValueError('Invalid walltime request')
        args = ['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes',
            '--nodes=1','--ntasks=1','--export=NONE','--partition='+site['partition'],
            '--account='+site['account'],'--qos='+site['qos'],'--gres='+site['gres'],
            '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
            '--time='+_walltime(minutes),'--job-name=lukafsu_'+name,
            '--comment=lukafsu:'+spec['content_hash']+':'+name,
            '--chdir='+spec['project_dir'],'--output='+str(root/(name+'-%j.out'))]
        if task['dependencies']:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in task['dependencies']))
        args += [spec['project_dir']+'/sbatch/run_luka_fullstat_u100_embedding.sh', spec['project_dir'],
                 str(root/'campaign_spec.json'), name]
        commands.append(dict(task_id=name, dependencies=task['dependencies'], command=args))
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=commands,
        jobs=2, mode='science', preflight_jobs=0, authorization=AUTHORIZATION)


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    campaign.check(spec)
    plan = render(spec)
    root = Path(spec['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, live = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION
                    or not saved.is_file() or load_json(saved) != plan or not dry.is_file()):
        raise PermissionError('Exact saved dry review and explicit two-job authorization required')
    write_immutable_json(saved, plan)
    check_site(plan)
    if execute:
        (root/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(root/'live_claim/intent.json', artifact('SUBMISSION_CLAIM',
            parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    value = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=live if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=environment())
    return value if execute else plan


def accepted_jobs(spec):
    root = Path(spec['root'])/'submission'
    plan = render(spec)
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal',
        identity=spec['content_hash'], plan=plan)
    if (root/'submission_ledger.json').is_file() and load_json(root/'submission_ledger.json') != \
            assemble_submission_ledger(events, campaign_spec_sha256=spec['content_hash']):
        raise ValueError('Accepted job ledger differs')
    return jobs


def successful_export(spec, job):
    if accepted_jobs(spec).get('export_U100') != job:
        raise ValueError('Not the exact export job')
    result = subprocess.run(['sacct','-j',job,'-X','-n','-P','-o','JobIDRaw,JobName%80,User,Account,State,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:6] for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == job]
    if matches != [[job,'lukafsu_export_U100',getpass.getuser(),spec['execution_site']['account'],'COMPLETED','0:0']]:
        raise ValueError('Exact export must be COMPLETED/0:0 with registered owner/account')
