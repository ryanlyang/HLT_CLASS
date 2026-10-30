"""Three fresh fits, two teachers before the student; no standalone setup or preflight."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site,environment as clean_environment
from .contracts import artifact,TASKS,AUTHORIZATION
from . import campaign


def render(spec):
    site=spec['execution_site']; root=Path(spec['root']); project=spec['project_dir']; commands=[]
    for task,deps in TASKS.items():
        command=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
            '--time='+_walltime(spec['minutes'][task]),'--gres='+site['gres'],
            '--job-name=lukafsmv_'+task,'--comment=lukafsmv:'+spec['content_hash']+':'+task,
            '--chdir='+project,'--output='+str(root/(task+'-%j.out'))]
        if deps:
            kind='afterany' if task=='teacher_FUSION' else 'afterok'
            command+=['--dependency='+kind+':'+':'.join('${JOB_'+d+'}' for d in deps)]
        command+=[project+'/sbatch/run_luka_fullstat_multiview.sh',project,str(root/'campaign_spec.json'),task]
        commands.append(dict(task_id=task,dependencies=deps,command=command))
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),jobs=len(TASKS),authorization=AUTHORIZATION,commands=commands,
        mode='science',setup_jobs=0,scientific_fits=len(TASKS),preflight_jobs=0)


def accepted_jobs(spec,*,partial=False):
    folder=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=render(spec))
    if not jobs or not set(jobs)<=TASKS.keys() or (not partial and set(jobs)!=TASKS.keys()):
        raise ValueError('Exact submission journal incomplete')
    ledger=folder/'submission_ledger.json'
    if ledger.exists():
        if load_json(ledger)!=assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'],events=events):
            raise ValueError('Final ledger differs from journal')
    elif not partial: raise ValueError('Missing completed submission ledger')
    return jobs


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec,full_reuse=True)
    if any((Path(spec['root'])/task).exists() for task in TASKS):
        raise FileExistsError('Preserve existing task outputs; no implicit resubmission')
    plan=render(spec)
    if execute and (plan_hash!=plan['content_hash'] or authorization!=AUTHORIZATION):
        raise PermissionError('Exact reviewed plan hash and authorization required')
    folder=Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry_run_submission_ledger.json'; ledger=folder/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved)!=plan): raise ValueError('Saved unchanged dry review required')
    write_immutable_json(saved,plan); check_site(plan)
    if execute:
        expected=build_submission_ledger(campaign_spec_sha256=spec['content_hash'],jobs={r['task_id']:'1' for r in plan['commands']},
            commands={r['task_id']:r['command'] for r in plan['commands']},dry_run=True)
        if not dry.is_file() or load_json(dry)!=expected: raise ValueError('Canonical dry ledger required')
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('CLAIM',parents=dict(plan=plan['content_hash'])))
    value=submit_exact_dag(identity=spec['content_hash'],plan=plan,output=ledger if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    return value if execute else plan

def successful(spec, task, job):
    import getpass
    import subprocess
    if accepted_jobs(spec).get(task)!=job:
        raise ValueError('Not the exact dependency job')
    run=subprocess.run(['sacct','-j',job,'-X','-n','-P','-o','JobIDRaw,JobName%80,User,Account,State,ExitCode'],
        check=True,capture_output=True,text=True)
    rows=[line.strip().split('|')[:6] for line in run.stdout.splitlines() if line.strip()]
    if [r for r in rows if r[0]==job] != [[job,'lukafsmv_'+task,getpass.getuser(),spec['execution_site']['account'],'COMPLETED','0:0']]:
        raise ValueError('Dependency must have exact registered owner/account and COMPLETED/0:0')

