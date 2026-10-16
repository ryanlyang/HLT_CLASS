"""Exact reviewed Oscar DAG; no automatic followup or ambiguous retry."""
from pathlib import Path
import os
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from .contracts import artifact
from . import campaign as c


def render(spec):
    gate = spec['contract'] == 'LUKA_FULLSTAT_GATE/v1'
    if not gate and spec['contract'] != 'LUKA_FULLSTAT_SPEC/v1':
        raise ValueError('Not a FullStat gate/scientific spec')
    commands = []
    root, project = Path(spec['root']), spec['project_dir']
    for row in c.gate_tasks() if gate else c.tasks():
        name, kind = row['task_id'], row['kind']
        gpu = kind in ('preflight','train','reduce')
        cpus, memory = (spec['cpus'],spec['memory_mb']) if gpu else ((4,32000) if kind=='prepare' else (1,16384))
        minutes = spec['gate_minutes'] if kind=='preflight' else (spec[kind+'_minutes'] if kind in ('train','reduce') else spec['cpu_minutes'])
        site = spec['execution_site']
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
                '--partition='+(site['partition'] if gpu else spec['cpu_partition']), '--account='+site['account'],
                '--cpus-per-task='+str(cpus), '--mem='+str(memory)+'M', '--time='+_walltime(minutes),
                '--job-name=lukafs_'+name, '--comment=lukafs:'+spec['content_hash']+':'+name,
                '--chdir='+project, '--output='+str(root/(name+('-%A_%a.out' if kind=='prepare' else '-%j.out')))]
        if gpu:
            args += ['--gres='+site['gres'], '--qos='+site['qos']]
        if kind == 'prepare':
            args.append('--array=0-'+str(spec['expected_files']-1)+'%'+str(spec['preparation_concurrency']))
        if row['dependencies']:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in row['dependencies']))
        args += [project+'/sbatch/run_luka_fullstat_oscar.sh', project,
                 str(root/('gate_spec.json' if gate else 'campaign_spec.json')), name]
        commands.append(dict(task_id=name, dependencies=row['dependencies'], command=args))
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=commands,
        mode='gate' if gate else 'science', jobs=len(commands),
        scheduled_tasks=19 if gate else 16, automatic_followup=False, authorization=c.AUTHORIZATION,
        final_test_accessed=False)


def environment():
    return {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}


def check_site(plan):
    env = environment()
    config = subprocess.run(['scontrol','show','config'], check=True, capture_output=True, text=True, env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*slurmctld\s*$', config):
        raise PermissionError('Submit on the registered Oscar cluster, not SPORC/Tigris')
    shapes = set()
    for row in plan['commands']:
        args = row['command']
        shape = tuple(a for a in args if a.startswith(('--partition=','--qos=','--mem=','--time=','--cpus-per-task=','--gres=')))
        if shape in shapes:
            continue
        shapes.add(shape)
        test = [a for a in args[:-4] if not a.startswith(('--dependency=','--array='))]
        test.insert(1,'--test-only')
        result = subprocess.run(test+['--wrap=true'], env=env, capture_output=True, text=True)
        print((result.stdout+result.stderr).strip(), flush=True)
        if result.returncode:
            raise PermissionError('Oscar rejected exact resource shape; no jobs submitted')


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    gate = spec['contract'] == 'LUKA_FULLSTAT_GATE/v1'
    c.validate_gate(spec) if gate else c.validate_spec(spec)
    value = render(spec)
    if execute and (plan_hash != value['content_hash'] or authorization != c.AUTHORIZATION):
        raise PermissionError('Exact reviewed plan hash and authorization required')
    root = Path(spec['root'])/('gate_submission' if gate else 'submission')
    root.mkdir(exist_ok=True)
    path, dry, live = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not path.is_file() or load_json(path) != value or not dry.is_file()):
        raise ValueError('Full dry-review plan/ledger required before execution')
    write_immutable_json(path, value)
    check_site(value)
    if not execute:
        submit_exact_dag(identity=spec['content_hash'], plan=value, output=dry, canonical_dry_run=dry, execute=False)
        return value
    if not live.is_file():
        claim = root/'live_claim'
        try:
            claim.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise PermissionError('Submission active/interrupted; preserve journal, no automatic retry') from exc
        write_immutable_json(claim/'intent.json', artifact('SUBMISSION_CLAIM',
            parents=dict(spec=spec['content_hash'], plan=value['content_hash'])))
    return submit_exact_dag(identity=spec['content_hash'], plan=value, output=live,
                            canonical_dry_run=dry, execute=True, environment=environment())
