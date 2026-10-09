"""Reviewed exact DAG, exclusive live claim, and durable per-job journal."""
from pathlib import Path
import os
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from . import science_campaign as c


def plan(spec):
    c.validate_spec(spec)
    root, project, site = Path(spec['root']), spec['project_dir'], spec['execution_site']
    commands = []
    for row in c.tasks():
        name, kind, deps = row['task_id'], row['kind'], row['dependencies']
        gpu = kind in ('train', 'reduce')
        minutes = spec[kind+'_minutes'] if gpu else 60
        args = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus'] if gpu else 1),
            '--mem='+str(spec['memory_mb'] if gpu else 16384)+'M', '--time='+_walltime(minutes),
            '--job-name=luka_'+name, '--comment=luka:'+spec['content_hash']+':'+name,
            '--chdir='+project, '--output='+str(root/(name+'-%j.out'))]
        if gpu:
            args.append('--gres='+site['gres'])
        if deps:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+d+'}' for d in deps))
        args += [project+'/sbatch/run_luka_fullsim_science.sh', project, str(root/'campaign_spec.json'), name]
        commands.append(dict(task_id=name, dependencies=deps, command=args))
    return c.make_artifact('PLAN', parents=dict(spec=spec['content_hash'], preflight=spec['preflight_sha256']),
        commands=commands, jobs=21, authorization=c.AUTHORIZATION)


def clean_environment():
    return {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}


def check_site(value):
    env = clean_environment()
    config = subprocess.run(['scontrol', 'show', 'config'], check=True, capture_output=True, text=True, env=env).stdout
    if not re.search(r'(?m)^\s*ClusterName\s*=\s*sporc\s*$', config):
        raise PermissionError('Submit Luka science on SPORC, not Oscar/Tigris')
    shapes = set()
    for row in value['commands']:
        command = row['command']
        shape = tuple(x for x in command if x.startswith(('--mem=', '--time=', '--cpus-per-task=', '--gres=')))
        if shape in shapes:
            continue
        shapes.add(shape)
        # Worker has three arguments; test the exact request without unresolved dependencies.
        args = [x for x in command[:-4] if not x.startswith('--dependency=')]
        args.insert(1, '--test-only')
        args.append('--wrap=true')
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        print((result.stdout+result.stderr).strip(), flush=True)
        if result.returncode:
            raise PermissionError('SPORC rejected reviewed resource shape')


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    value = plan(spec)
    root = Path(spec['root'])/'submission'
    if execute and (plan_hash != value['content_hash'] or authorization != c.AUTHORIZATION):
        raise PermissionError('Exact reviewed plan hash and science authorization required')
    root.mkdir(exist_ok=True)
    path, dry = root/'command_plan.json', root/'dry_run_submission_ledger.json'
    if execute and (not path.is_file() or load_json(path) != value):
        raise ValueError('Full saved dry plan must exist unchanged before live submission')
    write_immutable_json(path, value)
    check_site(value)
    if not execute:
        submit_exact_dag(identity=spec['content_hash'], plan=value, output=dry, canonical_dry_run=dry, execute=False)
        return value
    expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
        jobs={r['task_id']: '1' for r in value['commands']},
        commands={r['task_id']: r['command'] for r in value['commands']}, dry_run=True)
    if load_json(dry) != expected:
        raise ValueError('Canonical dry ledger differs')
    ledger = root/'submission_ledger.json'
    if not ledger.is_file():
        claim = root/'live_claim'
        try:
            claim.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise PermissionError('Live submitter active/interrupted: preserve journals; no automatic retry') from exc
        write_immutable_json(claim/'intent.json', c.make_artifact('SUBMISSION_CLAIM',
            parents=dict(spec=spec['content_hash'], plan=value['content_hash'])))
    return submit_exact_dag(identity=spec['content_hash'], plan=value, output=ledger, canonical_dry_run=dry,
        execute=True, environment=clean_environment())
