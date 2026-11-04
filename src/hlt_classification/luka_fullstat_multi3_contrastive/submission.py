"""One reviewed allocation; exact journals make ambiguous submission fail closed."""
from pathlib import Path
import getpass
import subprocess
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger,build_submission_ledger
from hlt_classification.luka_fullstat.submission import check_site,environment
from .contracts import artifact,TASK,AUTHORIZATION
from . import campaign


def render(spec):
    site=spec['execution_site'];root=Path(spec['root']);project=spec['project_dir']
    args=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
        '--cpus-per-task='+str(spec['cpus']),'--mem='+str(spec['memory_mb'])+'M',
        '--time='+_walltime(spec['minutes'][TASK]),'--gres='+site['gres'],
        '--job-name=lukafsmc_'+TASK,'--comment=lukafsmc:'+spec['content_hash']+':'+TASK,
        '--chdir='+project,'--output='+str(root/(TASK+'-%j.out')),
        project+'/sbatch/run_luka_fullstat_multi3_contrastive.sh',project,str(root/'campaign_spec.json')]
    return artifact('PLAN',parents=dict(spec=spec['content_hash']),commands=[dict(task_id=TASK,dependencies=[],command=args)],
        jobs=1,scientific_fits=1,teacher_fits=0,preflight_jobs=0,mode='science',authorization=AUTHORIZATION)


def accepted_jobs(spec,*,partial=False):
    folder=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=render(spec))
    if set(jobs)!={TASK}:raise ValueError('Missing exact student job binding')
    saved=folder/'submission_ledger.json'
    if saved.exists():
        if load_json(saved)!=assemble_submission_ledger(events=events,campaign_spec_sha256=spec['content_hash']):
            raise ValueError('Ledger differs from append-only journal')
    elif not partial:raise ValueError('Missing final submission ledger')
    return jobs


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec);plan=render(spec);root=Path(spec['root'])
    if (root/TASK).exists():raise FileExistsError('Task already attempted; no implicit retry')
    if execute and (plan_hash!=plan['content_hash'] or authorization!=AUTHORIZATION):
        raise PermissionError('Exact reviewed plan and bounded authorization required')
    folder=root/'submission';folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json';dry=folder/'dry_run_submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved)!=plan):raise ValueError('Saved dry review required')
    write_immutable_json(saved,plan);check_site(plan)
    if execute:
        expected=build_submission_ledger(campaign_spec_sha256=spec['content_hash'],jobs={TASK:'1'},
            commands={TASK:plan['commands'][0]['command']},dry_run=True)
        if not dry.is_file() or load_json(dry)!=expected:raise ValueError('Canonical dry ledger missing')
        from .recovery import claim
        claim(spec,plan)
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('CLAIM',parents=dict(plan=plan['content_hash'])))
    result=submit_exact_dag(identity=spec['content_hash'],plan=plan,
        output=folder/'submission_ledger.json' if execute else dry,canonical_dry_run=dry,
        execute=execute,environment=environment())
    return result if execute else plan


def accounting(spec,*,success=True):
    job=accepted_jobs(spec)[TASK]
    out=subprocess.run(['sacct','-S','2026-11-01','-j',job,'-X','-n','-P',
        '-o','JobIDRaw,JobName%80,User,Account,State%40,ExitCode'],check=True,capture_output=True,text=True).stdout
    rows=[line.strip().split('|')[:6] for line in out.splitlines() if line.strip()]
    if (len(rows)!=1 or rows[0][:4]!=[job,'lukafsmc_'+TASK,getpass.getuser(),spec['execution_site']['account']]):
        raise ValueError('Scheduler identity/ownership differs')
    state=rows[0][4].split(' by ')[0]
    if success:
        if state!='COMPLETED' or rows[0][5]!='0:0':raise ValueError('Student not COMPLETED/0:0')
    elif state not in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
        raise ValueError('Recovery requires an owned terminal unsuccessful job')
    return rows[0]
