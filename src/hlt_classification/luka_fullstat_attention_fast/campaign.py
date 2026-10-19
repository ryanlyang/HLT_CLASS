"""Fresh execution registration over an authenticated, already exported bank."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from hlt_classification.luka_fullstat_attention import source as original_source
from hlt_classification.luka_fullstat_attention.contracts import validate as original_validate
from .contracts import artifact, validate, policy

DONOR = 'cfb3a1587a6739b1949be9e65ac80f7ba9f8ead9'
ORIGINAL = '62c9b2df91eb3bda4fd4df4edc75c5f810a48c55'
MODULES = ('__init__', 'contracts', 'banks', 'checkpoints', 'training', 'campaign', 'preflight', 'worker', 'submission')
RUNTIME = {f'src/hlt_classification/luka_fullstat_attention_fast/{n}.py' for n in MODULES} | {
    'scripts/luka_fullstat_attention_fast.py', 'scripts/queue_luka_fullstat_attention_fast.sh',
    'sbatch/run_luka_fullstat_attention_fast.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_fullstat_attention_fast.py',
    'docs/plans/LUKA_FULLSTAT_FAST_ATTENTION_PLAN.md', 'docs/contracts/LUKA_FULLSTAT_FAST_ATTENTION.md',
    'docs/LUKA_FULLSTAT_FAST_ATTENTION_RUNBOOK.md'}

REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_attention import campaign,worker
s=load_json(sys.argv[1]); g,r=campaign.check_spec(s)
export=worker.completed(s,g,'export_D033')
if export is None: raise ValueError('No authenticated completed D033 export')
print('FAST_REUSE='+json.dumps(dict(gate=g,preflight=r,export=export),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    tracked = set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines())
    changes = set(subprocess.run(['git','-C',str(project),'diff','--name-only',DONOR,commit,'--','src','scripts','sbatch'],
                                check=True,capture_output=True,text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not changes <= RUNTIME
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_attention_fast/campaign.py'):
        raise ValueError('Commit the complete additive fast execution; existing runtime must remain frozen')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    original_validate(spec, 'SPEC')
    if spec['source_commit'] != ORIGINAL or path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected the original62c9 attention science spec')
    gate = load_json(spec['gate_path'])
    original_source.compatible(gate, project)
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(spec['project_dir'])/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating original attention registration and exported bank in its pinned worktree...', flush=True)
    result = subprocess.run([sys.executable,'-B','-s','-c',REPLAY,str(path)], cwd=spec['project_dir'],
                            env=env,capture_output=True,text=True)
    if result.returncode:
        raise ValueError('Original-source bank authentication failed:\n'+result.stderr)
    rows = [s.split('=',1)[1] for s in result.stdout.splitlines() if s.startswith('FAST_REUSE=')]
    if len(rows) != 1: raise ValueError('Missing original-source reuse evidence')
    evidence = json.loads(rows[0])
    if evidence['gate'] != gate: raise ValueError('Original gate changed during replay')
    validate_file_ref(ref)
    return artifact('REUSE', parents=dict(spec=spec['content_hash']), spec=ref, **evidence)


def separate(root, reuse, project):
    root = Path(root).resolve()
    old = reuse['gate']; e = old['anchor']['evidence']
    protected = [project, old['project_dir'], old['root'], old['bank_root'],
                 e['foundation_root'], e['cache_root'], e['raw_container']]
    for path in protected:
        other = Path(path).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise PermissionError('New execution root overlaps protected source/data/results')


def create(root, original_spec, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = current(project, commit)
    reuse = authenticate(original_spec, project)
    separate(root, reuse, project)
    resident = 4*reuse['preflight']['audit']['sum_squared_counts'] + 8*(reuse['gate']['counts']['train']+1)
    if resident > policy()['resident_cap_bytes']:
        raise MemoryError(f'Authenticated full-bank targets need {resident/2**30:.2f}GiB, above the resident cap; no GPU job registered')
    print(f'Authenticated resident target requirement: {resident/2**30:.2f} GiB',flush=True)
    if type(quota_gib) is not int or quota_gib*2**30 < policy()['result_headroom_bytes']:
        raise ValueError('Declare at least8 GiB personally available result storage')
    gate = artifact('GATE', parents=dict(source=snapshot['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        reuse=reuse,policy=policy(),resident_estimate_bytes=resident,available_result_bytes=quota_gib*2**30)
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'gate_spec.json',gate)
    return gate


def check_gate(g):
    validate(g,'GATE',dict(source=g['source_snapshot']['content_hash'],reuse=g['reuse']['content_hash']))
    if (current(g['project_dir'],g['source_commit']) != g['source_snapshot'] or g['policy'] != policy()
            or load_json(Path(g['root'])/'gate_spec.json') != g
            or g['available_result_bytes'] < policy()['result_headroom_bytes']
            or g['resident_estimate_bytes'] != 4*g['reuse']['preflight']['audit']['sum_squared_counts']+8*(g['reuse']['gate']['counts']['train']+1)
            or g['resident_estimate_bytes'] > policy()['resident_cap_bytes']):
        raise ValueError('Fast gate source/registration differs')
    if authenticate(g['reuse']['spec']['path'],g['project_dir']) != g['reuse']:
        raise ValueError('Frozen bank/parent lineage changed')
    separate(g['root'],g['reuse'],g['project_dir'])
    return g['reuse']['gate']


def accepted(g):
    check_gate(g)
    from .preflight import validate_report
    from .submission import accepted_jobs
    report = load_json(Path(g['root'])/'preflight/receipt.json')
    validate_report(g, report)
    if accepted_jobs(g) != dict(preflight=report['job_id']):
        raise ValueError('Accepted performance gate differs from submission journal')
    preflight_accounting(report['job_id'])
    return report


def science(g):
    report = accepted(g)
    root = Path(g['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=g['content_hash'],preflight=report['content_hash']),
        root=str(root),gate_path=str(Path(g['root'])/'gate_spec.json'),project_dir=g['project_dir'],
        source_commit=g['source_commit'],train_minutes=report['train_minutes'],
        tasks=[dict(task_id='train_ATTN_D000',dependencies=[]),dict(task_id='aggregate',dependencies=['train_ATTN_D000'])])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check_spec(s):
    g = load_json(s['gate_path']); r = accepted(g)
    validate(s,'SPEC',dict(gate=g['content_hash'],preflight=r['content_hash']))
    if (s['project_dir'] != g['project_dir'] or s['source_commit'] != g['source_commit']
            or s['train_minutes'] != r['train_minutes'] or Path(s['root']) != Path(g['root'])/'science'
            or s['tasks'] != [dict(task_id='train_ATTN_D000',dependencies=[]),dict(task_id='aggregate',dependencies=['train_ATTN_D000'])]
            or load_json(Path(s['root'])/'campaign_spec.json') != s):
        raise ValueError('Science execution differs')
    return g, r


def storage(g):
    if shutil.disk_usage(g['root']).free < policy()['result_headroom_bytes']:
        raise OSError('Insufficient result/checkpoint space; no files deleted')


def binding(g):
    return dict(gate=g['content_hash'],bank=g['reuse']['export']['result']['bank_hash'],
                source=g['source_snapshot']['content_hash'])


def check_stopped(s,exclude=None):
    """No duplicate training/recovery while any earlier science job is active."""
    from .submission import registered_jobs
    specs=[s]
    for path in sorted((Path(s['root'])/'recoveries').glob('*/resume_spec.json')):
        if exclude is not None and path.parent==Path(exclude): continue
        earlier=load_json(path)
        validate(earlier,'RESUME',dict(spec=s['content_hash']))
        specs.append(earlier)
    jobs={j for spec in specs for j in registered_jobs(spec).values()}
    for job in sorted(jobs):
        result=subprocess.run(['sacct','-j',str(job),'-X','-n','-P','-o','JobIDRaw,State,ExitCode'],
                              check=True,capture_output=True,text=True)
        matches=[r.split('|') for r in result.stdout.splitlines() if r.split('|')[0]==str(job)]
        terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
        if len(matches)!=1 or matches[0][1].split()[0] not in terminal:
            raise ValueError(f'Earlier science job {job} is active/unknown; inspect exact job before recovery')


def recovery(s,epoch_path,root):
    from . import checkpoints
    g,r=check_spec(s); check_stopped(s)
    path=Path(epoch_path).resolve(strict=True)
    if not path.is_relative_to(Path(s['root'])/'attempts'):
        raise PermissionError('Resume only an epoch from this exact execution')
    checkpoints.load(path,binding=binding(g))
    root=Path(root).resolve()
    if not root.is_relative_to(Path(s['root'])/'recoveries'):
        raise PermissionError('Recovery ledger must be a fresh child of science/recoveries')
    if (Path(s['root'])/'train_ATTN_D000').exists():
        raise FileExistsError('Scientific output exists; preserve it and inspect before recovery')
    value=artifact('RESUME',parents=dict(spec=s['content_hash']),root=str(root),science=file_ref(Path(s['root'])/'campaign_spec.json'),
        epoch=file_ref(path),gate_path=s['gate_path'],project_dir=s['project_dir'],source_commit=s['source_commit'],
        train_minutes=s['train_minutes'],tasks=s['tasks'])
    root.mkdir(parents=True,exist_ok=False); write_immutable_json(root/'resume_spec.json',value)
    return value


def check_resume(value,*,require_stopped=False):
    from . import checkpoints
    s=load_json(validate_file_ref(value['science'])); g,r=check_spec(s)
    validate(value,'RESUME',dict(spec=s['content_hash']))
    if (any(value[k]!=s[k] for k in ('gate_path','project_dir','source_commit','train_minutes','tasks'))
            or load_json(Path(value['root'])/'resume_spec.json')!=value
            or not Path(value['root']).is_relative_to(Path(s['root'])/'recoveries')
            or not Path(value['epoch']['path']).is_relative_to(Path(s['root'])/'attempts')):
        raise ValueError('Resume registration differs')
    checkpoints.load(validate_file_ref(value['epoch']),binding=binding(g))
    if require_stopped: check_stopped(s,exclude=value['root'])
    return s,g,r
