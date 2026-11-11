"""One bounded two-GPU probe, with exact-ID journals and no science release."""
import os
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_particle_teacher_hydra.submission import scheduler_review
from . import benchmark as b

AUTHORIZATION='AUTHORIZE_SUBCLASS_TWO_GPU_BENCHMARK_V1'


def plan(spec):
    project=spec['project_dir']; root=Path(spec['root'])
    command=['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
        '--partition=gpus','--account=default','--qos=normal','--cpus-per-task=6','--mem=32000M',
        '--gres=gpu:geforce_gtx_1080_ti:2','--time=02:00:00','--job-name=lukascb_probe',
        '--comment=lukascb:'+spec['content_hash'],'--chdir='+project,'--output='+str(root/'probe-%j.out'),
        project+'/sbatch/run_luka_subclass_gpu_benchmark.sh',project,str(root/'benchmark_spec.json'),'probe']
    return b.artifact('PLAN',parents=dict(spec=spec['content_hash']),authorization=AUTHORIZATION,
        commands=[dict(task_id='probe',dependencies=[],command=command)],science_release=False)


def jobs(spec):
    folder=Path(spec['root'])/'submission'; p=plan(spec)
    if not (folder/'command_plan.json').exists(): return {}
    if load_json(folder/'command_plan.json')!=p: raise ValueError('Benchmark submission plan differs')
    events,ids=load_exact_dag_journal(folder/'ledger_journal',identity=p['content_hash'],plan=p)
    if (folder/'ledger.json').exists() and load_json(folder/'ledger.json')!=assemble_submission_ledger(events,campaign_spec_sha256=p['content_hash']):
        raise ValueError('Benchmark ledger differs')
    return ids


def require_job(spec):
    job=os.environ.get('SLURM_JOB_ID','')
    import time
    for _ in range(12):
        if jobs(spec).get('probe')==job and job: return job
        time.sleep(5)
    raise PermissionError('Worker is not the exact accepted benchmark job')


def submit(spec,*,execute=False,plan_hash=None,authorization=None):
    b.check(spec); p=plan(spec); folder=Path(spec['root'])/'submission'; folder.mkdir(exist_ok=True)
    saved=folder/'command_plan.json'; dry=folder/'dry.json'; live=folder/'ledger.json'
    if execute:
        if authorization!=AUTHORIZATION or plan_hash!=p['content_hash']: raise PermissionError('Exact benchmark authorization required')
        if not saved.exists() or load_json(saved)!=p: raise PermissionError('Full benchmark dry review required')
        if live.exists():
            if set(jobs(spec))!={'probe'}: raise ValueError('Incomplete benchmark ledger')
            return load_json(live)
        if (folder/'live_claim').exists() or (folder/'ledger_journal').exists(): raise RuntimeError('Ambiguous benchmark submission; preserve journals')
    write_immutable_json(saved,p)
    if not execute: scheduler_review(p)
    else: (folder/'live_claim').mkdir()
    # The reusable submitter names its journal from the output stem.
    result=submit_exact_dag(identity=p['content_hash'],plan=p,output=live if execute else dry,
        canonical_dry_run=dry,execute=execute,environment=clean_environment())
    return result
