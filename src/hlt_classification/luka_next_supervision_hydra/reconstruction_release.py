"""R2/R3 CPU release, adapted from the particle-teacher exact Slurm workflow."""
from pathlib import Path
import json
import os
import re
import subprocess
import time
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from .contracts import artifact, authorization
from . import campaign, submission


def release_command(spec):
    site=spec['cpu_site']; project=spec['project_dir']; root=Path(spec['root'])
    return dict(task_id='release',dependencies=['setup'],command=[
        'sbatch','--parsable','--no-requeue','--kill-on-invalid-dep=yes',
        '--nodes=1','--ntasks=1','--export=NONE',
        '--partition='+site['partition'],'--account='+site['account'],'--qos='+site['qos'],
        '--cpus-per-task='+str(site['cpus']),'--mem='+str(site['memory_mb'])+'M',
        '--time='+_walltime(spec['release_minutes']),'--job-name=lukanr_release',
        '--comment=lukanr:'+spec['content_hash']+':release','--chdir='+project,
        '--output='+str(root/'release-%j.out'),'--dependency=afterok:${JOB_setup}',
        project+'/sbatch/run_luka_next_supervision_hydra.sh',project,str(root/'campaign_spec.json'),'release'])


def scheduler_review(value):
    """Test both registered bootstrap shapes without submitting a job."""
    env=clean_environment()
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True,env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config):
        raise PermissionError('Submit on Brown CS Hydra')
    for row in value['commands']:
        args=[x for x in row['command'][:-4] if not x.startswith('--dependency=')]
        args.insert(1,'--test-only'); args.append('--wrap=true')
        result=subprocess.run(args,env=env,capture_output=True,text=True)
        print((result.stdout+result.stderr).strip(),flush=True)
        if result.returncode: raise RuntimeError('Hydra rejected the bootstrap resource shape')


def cpu_allocation(spec):
    job=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch('[1-9][0-9]*',job): raise PermissionError('Slurm CPU allocation required')
    output=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True).stdout
    fields=dict(t.split('=',1) for t in output.split() if '=' in t)
    config=subprocess.run(['scontrol','show','config'],check=True,capture_output=True,text=True).stdout
    observed={k:fields.get(k) for k in ('JobId','JobState','Partition','Account','QOS',
        'NumCPUs','NumNodes','NumTasks','CPUs/Task','ReqTRES','AllocTRES')}
    environment={k:os.environ.get(k) for k in ('SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')}
    print('Reconstruction CPU release allocation: '+json.dumps(dict(controller=observed,
        environment=environment),sort_keys=True),flush=True)
    expected=dict(JobId=job,JobState='RUNNING',Partition=spec['cpu_site']['partition'],
        Account=spec['cpu_site']['account'],QOS=spec['cpu_site']['qos'])
    errors=[]
    def require(name,actual,wanted):
        if actual!=wanted: errors.append(f'{name}: expected {wanted!r}, observed {actual!r}')
    for key,value in expected.items(): require(key,fields.get(key),value)
    require('NumNodes',fields.get('NumNodes'),'1')
    # Observed Hydra rounding: one requested CPU may allocate two threads.
    requested=dict(x.split('=',1) for x in fields.get('ReqTRES','').split(',') if '=' in x)
    allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    require('ReqTRES cpu',requested.get('cpu'),str(spec['cpu_site']['cpus']))
    for label,tres in (('ReqTRES',requested),('AllocTRES',allocated)):
        require(label+' mem',tres.get('mem'),str(spec['cpu_site']['memory_mb'])+'M')
        require(label+' node',tres.get('node'),'1')
        if any(k.startswith('gres/gpu') for k in tres): errors.append(label+' contains GPU resources')
    if allocated.get('cpu') not in ('1','2'):
        errors.append('AllocTRES cpu: expected one CPU or two after scheduler rounding')
    require('NumCPUs versus AllocTRES',fields.get('NumCPUs'),allocated.get('cpu'))
    for key,env_key in (('NumTasks','SLURM_NTASKS'),('CPUs/Task','SLURM_CPUS_PER_TASK')):
        values=[v for v in (fields.get(key),environment[env_key]) if v not in (None,'','(null)','N/A')]
        if not values or any(v!='1' for v in values): errors.append(f'{key}/{env_key}: expected one, observed {values!r}')
    if environment['SLURM_MEM_PER_NODE'] is not None:
        require('SLURM_MEM_PER_NODE',environment['SLURM_MEM_PER_NODE'],str(spec['cpu_site']['memory_mb']))
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config): errors.append('Controller ClusterName must be hydra')
    if errors: raise PermissionError('CPU release allocation differs:\n  '+'\n  '.join(errors))
    return job


def finish(spec,job):
    """Honor the original bounded authorization; never resume a partial sbatch journal."""
    from .worker import completed_setup
    campaign.check(spec)
    if not submission.automatic(spec): raise PermissionError('Registered automatic reconstruction authorization required')
    bootstrap=submission.plan(spec,'bootstrap'); folder=Path(spec['root'])/'submission_bootstrap'
    claim=load_json(folder/'live_claim/intent.json')
    expected=artifact('SUBMIT_CLAIM',parents=dict(plan=bootstrap['content_hash']),authorization=authorization(spec))
    if claim!=expected: raise PermissionError('Original automatic campaign authorization differs')
    # A queued worker can start before the submitting process finishes its journal.
    for _ in range(12):
        jobs=submission.accepted_jobs(spec,'bootstrap')
        if jobs.get('release')==job and jobs.get('setup'): break
        time.sleep(5)
    else: raise PermissionError('Release is not the exact accepted bootstrap job')
    for _ in range(60):
        record=submission.accounting(jobs['setup'])
        if record==['COMPLETED','0:0']: break
        if record and record[0] not in ('PENDING','RUNNING','COMPLETING'):
            raise RuntimeError(f'Exact setup {jobs["setup"]} ended {record}; no science submitted')
        print(f'Waiting for setup {jobs["setup"]} accounting: {record}',flush=True)
        time.sleep(5)
    else: raise RuntimeError('Exact successful setup accounting is unavailable; no science submitted')
    acceptance=completed_setup(spec)  # Includes full receipt hashes and measured limits.
    value=submission.plan(spec,'science',acceptance)
    science=Path(spec['root'])/'submission_science'
    if not (science/'submission_ledger.json').exists():
        if (science/'live_claim').exists() or (science/'submission_ledger_journal').exists():
            raise RuntimeError('Ambiguous partial science submission; inspect exact-ID journals before recovery')
        submission.submit(spec,'science')  # Persist and dry-review every measured sbatch request.
    result=submission.submit(spec,'science',execute=True,plan_hash=value['content_hash'],authorization=authorization(spec))
    print('All twelve reconstruction fits submitted independently. See submission_science/submission_ledger.json.',flush=True)
    return result


def run(spec):
    campaign.check(spec)
    if installed_environment()!=spec['reference']['environment']: raise PermissionError('Qualified software differs')
    return finish(spec,cpu_allocation(spec))
