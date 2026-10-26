"""Replay the completed finalization/retry in its own immutable pinned source."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import artifact, COUNTS

DONOR = BASELINE = '420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6'
RUNTIME = '/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar'
PARENT_CONTRACTS = ('LUKA_FULLSTAT_SPLITHEAD_FINALIZE_SPEC/v1',
                    'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_SPEC/v1')
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'model', 'training', 'worker',
           'submission', 'kd_banks', 'execution', 'launch')
ADDITIONS = {f'src/hlt_classification/luka_fullstat_relational_ladder/{n}.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_fullstat_relational_ladder.py',
    'scripts/queue_luka_fullstat_relational_ladder.sh', 'sbatch/run_luka_fullstat_relational_ladder.sh',
    'scripts/launch_luka_fullstat_relational_ladder.sh'}
# A separately committed finalizer retry may precede this experiment. These
# paths are NOT changed by this package. authenticate additionally requires all
# parent runtime files to match that completed parent's exact pinned bytes.
RECOVERY_MODIFIED = {f'src/hlt_classification/luka_fullstat_splithead/{n}.py'
    for n in ('campaign', 'finalize', 'finalize_source', 'kd_banks', 'submission', 'worker')}
RECOVERY_MODIFIED |= {'scripts/luka_fullstat_splithead_finalize.py'}
RECOVERY_ADDED = {f'src/hlt_classification/luka_fullstat_splithead/{n}.py'
    for n in ('finalize_execution', 'finalize_retry')}
RECOVERY_ADDED |= {'scripts/queue_luka_fullstat_splithead_finalize_retry.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_fullstat_relational_ladder.py',
    'tests/test_luka_fullstat_relational_launch.py',
    'docs/plans/LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_RELATIONAL_LADDER.md',
    'docs/LUKA_FULLSTAT_RELATIONAL_LADDER_RUNBOOK.md'}

def compatible_changes(changed, added):
    new = ADDITIONS | ENTRYPOINTS | RECOVERY_ADDED
    return changed <= new | RECOVERY_MODIFIED and added == changed & new


def runtime(expected):
    if (Path(sys.prefix).resolve() != Path(RUNTIME).resolve(strict=True)
            or os.environ.get('CONDA_PREFIX') != RUNTIME):
        raise PermissionError('Use the verified runtime_repair_v1 environment')
    if installed_environment() != expected:
        raise ValueError('Replacement environment fingerprint differs')


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_splithead import campaign,worker,submission
s=load_json(sys.argv[1]); campaign.check(s)
completed=worker.completed(s,'DE000')
if completed is None:
    raise ValueError('Wait for the FINALIZATION/RETRY DE000 to complete; no new jobs submitted')
def refs(stage):
    d=Path(s['root'])/('train_'+stage)
    if stage in s['external_refs']:
        return dict(s['external_refs'][stage],receipt=file_ref(d/'task_receipt.json'))
    return {k:file_ref(d/f) for k,f in [('checkpoint','selected.pt'),
        ('report','training_report.json'),('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
r={('OLD_'+k if k in ('UE000','UE100') else k):v for k,v in s['references'].items()}
r.update(OLD_SPLITHEAD_DE033=refs('DE033'),OLD_SPLITHEAD_DE000=refs('DE000'))
old=load_json(validate_file_ref(s['finalization']['parent_spec']))
protected=s['reuse']['protected_roots']+[s['root'],s['project_dir'],s['runtime_prefix'],
    old['root'],old['project_dir'],str(Path(old['recovery']['parent_spec']['path']).parent)]
if 'retry' in s:
    protected += [s['retry']['previous_root'],s['retry']['previous_project']]
e=dict(accepted=s['reuse'],anchor=s['anchor'],references=r,
    external_refs={t:s['external_refs'][t] for t in ('OFFLINE','M0HLT')},
    imported_banks={t:s['imported_banks'][t] for t in ('OFFLINE','M0HLT')},
    base_node=s['reuse']['base_node'],execution_site=s['execution_site'],
    environment=s['reuse']['environment'],gpu=s['reuse']['gpu'],summaries=s['reuse']['summaries'],
    completed_parent_receipt=file_ref(Path(s['root'])/'train_DE000/task_receipt.json'),
    jobs=submission.accepted_jobs(s),runtime_prefix=s['runtime_prefix'],
    parent_contract=s['contract'],protected_roots=protected)
print('RELATIONAL_REUSE='+json.dumps(e,sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'],
                   check=True, capture_output=True, text=True)
    tracked = set(subprocess.run(['git', '-C', str(project), 'ls-files'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    changed = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    added = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only', '--diff-filter=A',
        BASELINE, commit, '--', 'src', 'scripts', 'sbatch'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= tracked or not compatible_changes(changed, added)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_relational_ladder/source.py'):
        raise ValueError('Source changes outside relational ladder and separately authenticated finalizer retry')
    return snapshot


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    s = load_json(path)
    original = Path(s['project_dir'])
    if (s['counts'] != COUNTS or s['contract'] not in PARENT_CONTRACTS
            or s.get('runtime_prefix') != RUNTIME
            or path != Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed FullStat finalization/retry specification')
    runtime(s['reuse']['environment'])
    subprocess.run(['git', '-C', str(original), 'merge-base', '--is-ancestor', DONOR, s['source_commit']],
                   check=True, capture_output=True, text=True)
    names = subprocess.run(['git', '-C', str(original), 'ls-files', 'src', 'sbatch', 'scripts'],
        check=True, capture_output=True, text=True).stdout.splitlines()
    hashes = {}
    for name in names:
        if not name.endswith(('.py', '.sh')):
            continue
        before, after = original/name, Path(project)/name
        if not after.is_file() or sha256_file(before) != sha256_file(after):
            raise ValueError('Changed donor runtime: '+name)
        hashes[name] = sha256_file(after)
    if not hashes:
        raise ValueError('Empty donor runtime inventory')
    env = dict(os.environ, PYTHONPATH=str(original/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed finalization/retry DE000 in its original source; no new preflight.', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=original, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    lines = [line.split('=', 1)[1] for line in run.stdout.splitlines() if line.startswith('RELATIONAL_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    evidence = json.loads(lines[0])
    # Original worker validates recovery accounting and all native descendants.
    return artifact('REUSE', parent_spec=file_ref(path), source_commit=s['source_commit'],
        counts=dict(COUNTS), compatible_source=hashes, **evidence)
