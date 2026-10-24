"""Hash-reviewed exact22-job debug DAG; no preflight or login controller."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.luka_fullsim.regular_submission import check_site,clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from .contracts import artifact,AUTHORIZATION
from . import campaign


def render(spec):
    site=spec['execution_site']; root=Path(spec['root']); commands=[]
    for row in spec['tasks']:
        task=row['task_id']; report=task=='report'
        cpus=spec['report_cpus'] if report else spec['cpus']; memory=spec['report_memory_mb'] if report else spec['memory_mb']
        cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
            '--cpus-per-task='+str(cpus),'--mem='+str(memory)+'M','--time='+_walltime(spec['task_minutes'][task]),
            '--job-name=lukahy_'+task,'--comment=lukahy:'+spec['content_hash']+':'+task,'--chdir='+spec['project_dir'],
            '--output='+str(root/(task+'-%j.out'))]
        if not report: cmd.append('--gres='+site['gres'])
        if row['dependencies']: cmd.append('--dependency=afterok:'+':'.join('${JOB_'+t+'}' for t in row['dependencies']))
        cmd += [spec['project_dir']+'/sbatch/run_luka_trial_hybrid.sh',spec['project_dir'],str(root/'campaign_spec.json'),task]
        commands.append(dict(task_id=task,dependencies=row['dependencies'],command=cmd))
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=commands,jobs=22,scientific_fits=20,
        target_export_jobs=1,report_jobs=1,preflight_jobs=0,authorization=AUTHORIZATION)


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec); plan=render(spec); root=Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
    saved,dry,live=root/'command_plan.json',root/'dry_run_submission_ledger.json',root/'submission_ledger.json'
    if execute and (plan_hash!=plan['content_hash'] or authorization!=AUTHORIZATION
            or not saved.is_file() or load_json(saved)!=plan or not dry.is_file()):
        raise PermissionError('Exact full dry review and twenty-fit authorization required')
    write_immutable_json(saved,plan); check_site(plan)
    if execute:
        (root/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(root/'live_claim/intent.json',artifact('CLAIM',parents=dict(spec=spec['content_hash'],plan=plan['content_hash'])))
    value=submit_exact_dag(identity=spec['content_hash'],plan=plan,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    return value if execute else plan


def accepted_jobs(spec):
    root=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(root/'submission_ledger_journal',identity=spec['content_hash'],plan=render(spec))
    if (root/'submission_ledger.json').is_file() and load_json(root/'submission_ledger.json')!=assemble_submission_ledger(
            events,campaign_spec_sha256=spec['content_hash']): raise ValueError('Accepted ledger differs')
    return jobs


def successful(spec,task,job):
    if accepted_jobs(spec).get(task)!=job: raise ValueError('Not an accepted ancestor job')
    run=subprocess.run(['sacct','-j',job,'-X','-n','-P','-o','JobIDRaw,JobName%80,User,Account,State,ExitCode'],
        check=True,capture_output=True,text=True)
    rows=[x.strip().split('|')[:6] for x in run.stdout.splitlines() if x.strip()]
    if [r for r in rows if r[0]==job]!=[[job,'lukahy_'+task,getpass.getuser(),spec['execution_site']['account'],'COMPLETED','0:0']]:
        raise ValueError('Exact ancestor must be COMPLETED/0:0 with registered owner/account/name')
