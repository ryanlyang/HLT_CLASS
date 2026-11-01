"""Recover the exact 8177361 timing rejection; preserve the completed CONCAT.

The old source authenticates CONCAT in full. New banks keep their own original
spec/checkpoint parents; no receipt, source tree or target payload is rewritten.
"""
from pathlib import Path
import getpass
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from . import campaign, source, resources
from .contracts import artifact, validate, COUNTS, is_recovery

PARENT_COMMIT = '8ad0b381547134569818b4258f3cd31b509e9efb'
PARENT_HASH = 'aa0c2e10abeb2040441db458b0293d5a2fe837ca23b34164b1ffc1cd93db1420'
BASELINE = '810433f83eb6ab4ed78d18f6ab75ce70f5114350'
JOBS = dict(teacher_CONCAT='8177360', teacher_FUSION='8177361', train_multi3='8177362')
CLAIM = 'multiview_recovery_claim_v1'
# These operational adapters may change. Model, fusion, views, targets and
# training bytes (and every pre-existing external dependency) remain frozen.
OPERATIONS = {f'src/hlt_classification/luka_fullstat_multiview/{name}.py' for name in
    ('campaign','contracts','execution','launch','recovery','resources','source','startup','submission','worker')}
CHANGES = OPERATIONS | {'scripts/luka_fullstat_multiview.py',
                       'scripts/launch_luka_fullstat_multiview_recovery.sh'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout
    git('merge-base','--is-ancestor',BASELINE,commit)
    changed = set(git('diff','--name-only',BASELINE,commit,'--','src','scripts','sbatch').splitlines())
    tracked = set(git('ls-files').splitlines())
    if (not changed <= CHANGES or not source.REQUIRED <= tracked
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_multiview/recovery.py'):
        raise ValueError('Recovery changed frozen scientific runtime or lacks required source')
    return snapshot


def parent(spec):
    return load_json(validate_file_ref(spec['recovery']['parent_spec']))


def accounting(old, jobs):
    if jobs != JOBS:
        raise ValueError('Recovery requires the exact original three job IDs')
    run = subprocess.run(['sacct','-S','2026-10-30','-j',','.join(jobs.values()),'-X','-n','-P',
        '-o','JobIDRaw,JobName%80,User,Account,State%40,ExitCode'],
        check=True,capture_output=True,text=True)
    rows = [line.strip().split('|')[:6] for line in run.stdout.splitlines() if line.strip()]
    if len(rows) != 3 or len({r[0] for r in rows}) != 3:
        raise ValueError('Missing or ambiguous original accounting')
    expected = dict(teacher_CONCAT=('COMPLETED','0:0'), teacher_FUSION=('FAILED','1:0'), train_multi3=('CANCELLED','0:0'))
    for task, job in jobs.items():
        row = next((r for r in rows if r[0] == job), None)
        if (row is None or row[1:4] != ['lukafsmv_'+task,getpass.getuser(),old['execution_site']['account']]
                or row[4].split(' by ')[0] != expected[task][0] or row[5] != expected[task][1]):
            raise ValueError('Original job not in the expected terminal owner/state: '+task)
    return sorted(rows)


def failed_parent(old):
    validate(old,'SPEC')
    if (old['source_commit'] != PARENT_COMMIT or old['content_hash'] != PARENT_HASH
            or old['counts'] != COUNTS or old['runtime_policy'] != resources.policy()):
        raise ValueError('Expected the exact completed-CONCAT / failed-FUSION v2 campaign')
    root = Path(old['root'])
    folder = root/'teacher_FUSION'
    for name in ('startup.json','epochs','completed_training.json','task_receipt.json','bank'):
        if os.path.lexists(folder/name):
            raise ValueError('FUSION has scientific/accepted outputs; restart-zero recovery forbidden')
    if os.path.lexists(root/'train_multi3'):
        raise ValueError('Student already attempted; inspect instead of retrying')
    paths = dict(measurements=folder/'startup_diagnostics/09-measurements.json',
                 failure=folder/'startup_diagnostics/10-failure.json')
    values = {}
    for key, path in paths.items():
        values[key] = load_json(path)
        validate(values[key],'STARTUP_DIAGNOSTIC',dict(spec=old['content_hash']))
        if (values[key]['task_id'] != 'teacher_FUSION' or values[key]['stage'] != key
                or values[key]['accepted'] is not False or values[key]['runtime_policy'] != resources.policy()):
            raise ValueError('Wrong failure diagnostic')
    m, f = values['measurements'], values['failure']
    from .startup import timing
    t = m['timing']
    if (timing(old,'teacher_FUSION',t['train_step'],t['val_step'],t['export_step'],t['preparation'],enforce=False) != t
            or not old['minutes']['teacher_FUSION'] < t['conservative_minutes'] <= 4320
            or m['allocated_minutes'] != old['minutes']['teacher_FUSION']
            or f['failed_stage'] != 'measurements' or f['error_type'] != 'ValueError'
            or f['message'] != f'Measured full20 plus export requires {t["conservative_minutes"]} minutes; allocation is {m["allocated_minutes"]} minutes. See startup_diagnostics; no fallback.'):
        raise ValueError('Failure is not the admitted 48h timing-only rejection')
    resources.check_memory(old,m['measured_memory'])
    return {key:file_ref(path) for key,path in paths.items()}


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_multiview import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s,full_reuse=True)
rec=worker.completed(s,'teacher_CONCAT',accounting=False)
root=Path(s['root'])
print('MULTIVIEW_RECOVERY='+json.dumps(dict(jobs=submission.accepted_jobs(s),
    receipt=file_ref(root/'teacher_CONCAT/task_receipt.json'),
    bank=file_ref(root/'teacher_CONCAT/bank/manifest.json')),sort_keys=True,allow_nan=False))
'''


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    old = load_json(path); diagnostics = failed_parent(old)
    if path != Path(old['root'])/'campaign_spec.json':
        raise ValueError('Wrong original spec locator')
    source.runtime(old['reuse']['u100']['environment'])
    original = Path(old['project_dir'])
    names = subprocess.run(['git','-C',str(original),'ls-files','src','scripts','sbatch'],
                           check=True,capture_output=True,text=True).stdout.splitlines()
    hashes = {name:sha256_file(original/name) for name in names if name.endswith(('.py','.sh'))}
    if not hashes: raise ValueError('Missing original runtime inventory')
    for name, digest in hashes.items():
        if name not in CHANGES and sha256_file(Path(project)/name) != digest:
            raise ValueError('Frozen scientific runtime changed: '+name)
    print('Authenticating completed CONCAT, all selected/epoch outputs and TRAIN bank in original source; no teacher inference.',flush=True)
    env = dict(os.environ,PYTHONPATH=str(original/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)],cwd=original,env=env,capture_output=True,text=True)
    if run.returncode: raise ValueError('Original CONCAT authentication failed:\n'+run.stderr)
    lines = [line.split('=',1)[1] for line in run.stdout.splitlines() if line.startswith('MULTIVIEW_RECOVERY=')]
    if len(lines) != 1: raise ValueError('Missing exact original-source recovery evidence')
    evidence = json.loads(lines[0])
    states = accounting(old,evidence['jobs'])
    return artifact('RECOVERY_REUSE',parents=dict(spec=old['content_hash']),parent_spec=file_ref(path),
                    original_source=hashes,accounting=states,**diagnostics,**evidence)


def validate_reuse(spec):
    r = spec['recovery']; old = parent(spec)
    validate(r,'RECOVERY_REUSE',dict(spec=old['content_hash']))
    if (failed_parent(old) != {k:r[k] for k in ('measurements','failure')} or r['jobs'] != JOBS
            or any(spec[k] != old[k] for k in ('reuse','counts','nodes','recipe','anchor','references','bank_manifest','teacher_refs'))):
        raise ValueError('Recovery changed original science, populations or failure binding')
    root = Path(old['root'])
    for key, name in (('receipt','task_receipt.json'),('bank','bank/manifest.json')):
        if r[key] != file_ref(root/'teacher_CONCAT'/name): raise ValueError('Completed CONCAT reference differs')
    from hlt_classification.luka_fullstat_multiteacher.campaign import protected
    protected(spec['root'],dict(anchor=spec['anchor'],protected_roots=[old['root'],old['project_dir']]),spec['project_dir'])


def check(spec, *, full=False):
    validate_reuse(spec); old = parent(spec)
    for name, digest in spec['recovery']['original_source'].items():
        if sha256_file(Path(old['project_dir'])/name) != digest:
            raise ValueError('Original source changed: '+name)
        if name not in CHANGES and sha256_file(Path(spec['project_dir'])/name) != digest:
            raise ValueError('Frozen scientific runtime changed: '+name)
    if full and authenticate(spec['recovery']['parent_spec']['path'],spec['project_dir']) != spec['recovery']:
        raise ValueError('Original recovery evidence changed')
    claim_dir = Path(old['root'])/CLAIM
    if claim_dir.exists():
        rec = load_json(claim_dir/'intent.json'); validate(rec,'RECOVERY_CLAIM',dict(spec=spec['content_hash']))
        from .submission import render
        if rec['root'] != spec['root'] or rec['plan_hash'] != render(spec)['content_hash']:
            raise ValueError('Original campaign claimed by another recovery/plan')


def claim(spec, plan):
    # Atomic original-root claim prevents two fresh roots racing to retry it.
    old = parent(spec); accounting(old,spec['recovery']['jobs'])
    directory = Path(old['root'])/CLAIM
    directory.mkdir(exist_ok=False)
    write_immutable_json(directory/'intent.json',artifact('RECOVERY_CLAIM',parents=dict(spec=spec['content_hash']),
        root=spec['root'],plan_hash=plan['content_hash'],jobs=spec['recovery']['jobs']))


def completed_concat(spec):
    validate_reuse(spec); old = parent(spec)
    rec = load_json(validate_file_ref(spec['recovery']['receipt']))
    validate(rec,'TASK',dict(spec=old['content_hash']))
    if rec['task_id'] != 'teacher_CONCAT' or rec['job_id'] != JOBS['teacher_CONCAT']:
        raise ValueError('Wrong completed CONCAT receipt')
    from .worker import output_names
    if [r['path'] for r in rec['outputs']] != output_names('teacher_CONCAT'):
        raise ValueError('Original CONCAT output inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=Path(old['root']))
    return rec


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    snapshot = current(project,commit)
    evidence = authenticate(previous_spec,project)
    old = load_json(previous_spec)
    if os.path.lexists(Path(old['root'])/CLAIM): raise FileExistsError('Original already claimed; inspect, do not retry')
    spec = artifact('RECOVERY_SPEC',parents=dict(source=snapshot['content_hash'],reuse=old['reuse']['content_hash'],
        recovery=evidence['content_hash']),root=str(root),project_dir=str(project),source_commit=commit,
        source_snapshot=snapshot,reuse=old['reuse'],recovery=evidence,available_quota_gib=quota_gib,
        **campaign.registration(old['reuse'],recovered=True))
    campaign.validate_spec(spec); campaign.storage(spec)
    root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json',spec)
    return spec
