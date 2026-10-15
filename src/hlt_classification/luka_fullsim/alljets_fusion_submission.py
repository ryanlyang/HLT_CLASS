"""Exact reviewed plans; a one-job gate then a separate sixteen-job continuation."""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger, build_submission_ledger
from .regular_submission import check_site, clean_environment
from . import alljets_fusion_campaign as c


def is_gate(spec):
    return spec['contract'] == 'LUKA_ALLJETS_FUSION_GATE/v1'


def render(spec):
    gate = is_gate(spec)
    rows = [dict(task_id='preflight', kind='preflight', dependencies=[])] if gate else c.tasks()
    root, site = Path(spec['root']), spec['execution_site']
    commands = []
    for row in rows:
        name, kind = row['task_id'], row['kind']
        gpu = kind in ('train','reduce','preflight')
        minutes = spec['gate_minutes'] if gate else spec[kind+'_minutes'] if gpu else 60
        args = ['sbatch','--parsable','--no-requeue','--nodes=1','--ntasks=1','--export=NONE',
            '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
            '--cpus-per-task='+str(spec['cpus'] if gpu else 1),
            '--mem='+str(spec['memory_mb'] if gpu else 16384)+'M', '--time='+_walltime(minutes),
            '--job-name=lukaaf_'+name, '--comment=lukaaf:'+spec['content_hash']+':'+name,
            '--chdir='+spec['project_dir'], '--output='+str(root/(name+'-%j.out'))]
        if gpu:
            args.append('--gres='+site['gres'])
        if row['dependencies']:
            args.append('--dependency=afterok:'+':'.join('${JOB_'+dep+'}' for dep in row['dependencies']))
        args += [spec['project_dir']+'/sbatch/run_luka_fullsim_alljets_fusion.sh', spec['project_dir'],
            str(root/('gate_spec.json' if gate else 'campaign_spec.json')), name]
        commands.append(dict(task_id=name, dependencies=row['dependencies'], command=args))
    return c.artifact('PLAN', parents=dict(spec=spec['content_hash']), commands=commands, jobs=len(commands),
        mode='gate' if gate else 'science', authorization=c.AUTHORIZATION)


def accepted_jobs(spec):
    root = Path(spec['root'])/'submission'
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=spec['content_hash'], plan=render(spec))
    if load_json(root/'submission_ledger.json') != assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'], events=events):
        raise ValueError('Completed gate ledger differs from exact journal')
    if set(jobs) != {'preflight'}:
        raise ValueError('Incomplete one-job gate submission')
    return jobs


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    c.validate_gate(spec) if is_gate(spec) else c.validate_spec(spec)
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != c.AUTHORIZATION):
        raise PermissionError('Exact reviewed hash and authorization required')
    root = Path(spec['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Saved unchanged full dry plan required')
    write_immutable_json(saved, plan)
    check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={r['task_id']:'1' for r in plan['commands']}, commands={r['task_id']:r['command'] for r in plan['commands']}, dry_run=True)
        if not dry.is_file() or load_json(dry) != expected:
            raise ValueError('Canonical dry ledger required')
        if not ledger.is_file():
            try:
                (root/'live_claim').mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise PermissionError('Submission active/interrupted; preserve journal, no automatic retry') from exc
            write_immutable_json(root/'live_claim/intent.json', c.artifact('SUBMISSION_CLAIM',
                parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan
