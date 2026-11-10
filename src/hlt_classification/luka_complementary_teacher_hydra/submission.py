"""Exact-ID bootstrap and durable release of three complementary teachers, three OOF fits and four students."""
from pathlib import Path
import subprocess
import time
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag,load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_particle_teacher_hydra.submission import scheduler_review
from .contracts import FITS,TEACHERS,STUDENTS,DEPENDENCIES,AUTHORIZATION,artifact
from . import campaign


def command(spec,task,minutes,*,cpu=False,dependencies=(),after='afterok'):
    project=spec['project_dir']; root=Path(spec['root']); site=spec['cpu_site'] if cpu else spec['execution_site']
    cmd=['sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes','--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
        '--cpus-per-task='+str(site['cpus'] if cpu else spec['cpus']),
        '--mem='+str(site['memory_mb'] if cpu else spec['memory_mb'])+'M','--time='+_walltime(minutes),
        '--job-name=lukact_'+task,'--comment=lukact:'+spec['content_hash']+':'+task,
        '--chdir='+project,'--output='+str(root/(task+'-%j.out'))]
    if not cpu: cmd.append('--gres='+site['gres'])
    if dependencies: cmd.append('--dependency='+after+':'+':'.join('${JOB_'+t+'}' for t in dependencies))
    cmd += [project+'/sbatch/run_luka_complementary_teacher_hydra.sh',project,str(root/'campaign_spec.json'),task]
    return dict(task_id=task,dependencies=list(dependencies),command=cmd)


def plan(spec,stage,acceptance=None):
    parents=dict(spec=spec['content_hash'])
    if stage=='bootstrap':
        rows=[command(spec,'setup',spec['setup_minutes']),command(spec,'release',spec['release_minutes'],cpu=True,dependencies=['setup'])]
    elif stage in ('science','envelope'):
        if stage=='envelope': acceptance=dict(minutes={t:spec['fit_max_minutes'] for t in FITS},content_hash='upper_bound')
        if acceptance is None or set(acceptance['minutes'])!=set(FITS): raise ValueError('Complete measured acceptance required')
        if any(type(n) is not int or not 0<n<=spec['fit_max_minutes'] for n in acceptance['minutes'].values()): raise ValueError('Invalid measured walltime')
        parents['acceptance']=acceptance['content_hash']
        rows=[command(spec,t,acceptance['minutes'][t],dependencies=DEPENDENCIES[t]) for t in FITS]
        rows.append(command(spec,'report',spec['report_minutes'],cpu=True,dependencies=FITS,after='afterany'))
    else: raise ValueError('Unknown submission stage')
    return artifact('PLAN',parents=parents,stage=stage,commands=rows,authorization=AUTHORIZATION,
        science_tasks=list(FITS),science_max_minutes=spec['fit_max_minutes'],registration=campaign.registration(),
        upper_bound_commands=plan(spec,'envelope')['commands'] if stage=='bootstrap' else None)


def accepted_jobs(spec,stage):
    folder=Path(spec['root'])/('submission_'+stage); saved=folder/'command_plan.json'
    if not saved.exists(): return {}
    acceptance=load_json(Path(spec['root'])/'setup/acceptance.json') if stage=='science' else None
    value=plan(spec,stage,acceptance)
    if load_json(saved)!=value: raise ValueError('Saved exact plan differs')
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=value['content_hash'],plan=value)
    ledger=folder/'submission_ledger.json'
    if ledger.exists() and load_json(ledger)!=assemble_submission_ledger(events,campaign_spec_sha256=value['content_hash']):
        raise ValueError('Submission ledger/journal mismatch')
    return jobs


def submit_stage(spec,value,*,execute):
    folder=Path(spec['root'])/('submission_'+value['stage']); folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry_run_submission_ledger.json'; live=folder/'submission_ledger.json'
    if execute:
        if not saved.exists() or load_json(saved)!=value: raise PermissionError('Exact dry review required')
        if live.exists():
            if set(accepted_jobs(spec,value['stage']))!={r['task_id'] for r in value['commands']}: raise ValueError('Incomplete submission ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'submission_ledger_journal').exists():
            raise RuntimeError('Submission active or ambiguous/partial; preserve and inspect journals')
    write_immutable_json(saved,value)
    if not execute:
        scheduler_review(value)
        if value['stage']=='bootstrap': scheduler_review(plan(spec,'envelope'))
    if execute:
        (folder/'live_claim').mkdir(exist_ok=False)
        write_immutable_json(folder/'live_claim/intent.json',artifact('SUBMIT_CLAIM',parents=dict(plan=value['content_hash']),authorization=AUTHORIZATION))
    return submit_exact_dag(identity=value['content_hash'],plan=value,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())


def launch(spec,*,execute=False,plan_hash=None,authorization=None):
    campaign.check(spec); value=plan(spec,'bootstrap')
    if execute and (plan_hash!=value['content_hash'] or authorization!=AUTHORIZATION): raise PermissionError('Authorize exact bootstrap and bounded ten fits plus report')
    result=submit_stage(spec,value,execute=execute)
    if execute: print('Setup and durable CPU release submitted. Safe to log off after both job IDs are recorded.',flush=True)
    return result


def accounting(job):
    r=subprocess.run(['sacct','-X','-n','-P','-j',job,'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows=[x.split('|')[1:3] for x in r.stdout.splitlines() if x.split('|')[0]==job]
    if len(rows)>1: raise ValueError('Ambiguous accounting')
    return rows[0] if rows else None


def finish(spec,job):
    from .worker import completed_setup
    campaign.check(spec); bootstrap=plan(spec,'bootstrap'); folder=Path(spec['root'])/'submission_bootstrap'
    expected=artifact('SUBMIT_CLAIM',parents=dict(plan=bootstrap['content_hash']),authorization=AUTHORIZATION)
    if load_json(folder/'live_claim/intent.json')!=expected: raise PermissionError('Original authorization differs')
    for _ in range(12):
        jobs=accepted_jobs(spec,'bootstrap')
        if jobs.get('release')==job and jobs.get('setup'): break
        time.sleep(5)
    else: raise PermissionError('Release is not the exact accepted job')
    for _ in range(60):
        result=accounting(jobs['setup'])
        if result==['COMPLETED','0:0']: break
        if result and result[0] not in ('PENDING','RUNNING','COMPLETING'): raise RuntimeError('Exact setup failed; no science submitted: '+str(result))
        print('Waiting for exact successful setup accounting: '+str(result),flush=True); time.sleep(5)
    else: raise RuntimeError('Setup accounting unavailable; no science submitted')
    acceptance=completed_setup(spec); value=plan(spec,'science',acceptance)
    within_envelope(plan(spec,'envelope'),value)
    science=Path(spec['root'])/'submission_science'
    if not (science/'submission_ledger.json').exists():
        if (science/'live_claim').exists() or (science/'submission_ledger_journal').exists(): raise RuntimeError('Ambiguous partial science submission; preserve journals')
        submit_stage(spec,value,execute=False)
    result=submit_stage(spec,value,execute=True)
    print('All ten fits submitted with only the required teacher/OOF dependencies.',flush=True)
    return result


def within_envelope(envelope, measured):
    if len(envelope['commands']) != len(measured['commands']):
        raise ValueError('Science task envelope differs')
    for a, b in zip(envelope['commands'], measured['commands']):
        strip = lambda xs: [x for x in xs if not x.startswith('--time=')]
        if (a['task_id'] != b['task_id'] or a['dependencies'] != b['dependencies']
                or strip(a['command']) != strip(b['command'])):
            raise PermissionError('Measured science changed the authorized commands')
    # Numeric walltime caps were checked by plan(); only --time can differ.
    return True
