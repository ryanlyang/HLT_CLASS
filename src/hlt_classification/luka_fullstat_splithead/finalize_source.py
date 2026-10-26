"""Exact source/runtime boundary for the post-export DE033 recovery."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import validate, COUNTS

PARENT_COMMIT = '0c4a71a55e8bc163ea3c4805b72f06d18e640cb5'
BASELINE = '028c3314394949706787c6d1ef5e632f53ca157c'
RUNTIME = '/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar'
MODIFIED = {f'src/hlt_classification/luka_fullstat_splithead/{n}.py'
            for n in ('campaign', 'worker', 'submission', 'kd_banks')}
ADDED = {f'src/hlt_classification/luka_fullstat_splithead/{n}.py'
         for n in ('finalize', 'finalize_source')}
ADDED |= {'scripts/luka_fullstat_splithead_finalize.py',
          'scripts/queue_luka_fullstat_splithead_finalize.sh',
          'sbatch/run_luka_fullstat_splithead_finalize.sh',
          'sbatch/luka_splithead_finalize_common.sh'}
REQUIRED = MODIFIED | ADDED | {'tests/test_luka_fullstat_splithead_finalize.py',
    'docs/contracts/LUKA_FULLSTAT_SPLITHEAD_FINALIZE.md',
    'docs/LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RUNBOOK.md'}
JOBS = dict(recover_DE066='7938791', train_DE033='7938792', train_DE000='7938793')

REPLAY = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_splithead import campaign,submission,worker
s=load_json(sys.argv[1]); campaign.check(s)
done=worker.completed(s,'DE066')
if done is None: raise ValueError('Missing completed DE066 recovery')
print('FINALIZE_PARENT='+json.dumps(dict(spec=s,jobs=submission.accepted_jobs(s),de066=done),sort_keys=True,allow_nan=False))
'''


def compatible(changed, added):
    return changed <= MODIFIED | ADDED and added == changed & ADDED


def current(project, commit):
    project = Path(project).resolve()
    snap = _source(project, commit)
    def git(*args):
        return set(subprocess.run(['git', '-C', str(project), *args], check=True,
            capture_output=True, text=True).stdout.splitlines())
    changed = git('diff', '--name-only', BASELINE, commit, '--', 'src', 'scripts', 'sbatch')
    added = git('diff', '--name-only', '--diff-filter=A', BASELINE, commit, '--', 'src', 'scripts', 'sbatch')
    if (not compatible(changed, added) or not REQUIRED <= git('ls-files')
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_splithead/finalize_source.py'):
        raise ValueError('Changes outside the reviewed finalization source boundary')
    return snap


def runtime(expected):
    if Path(sys.prefix).resolve() != Path(RUNTIME).resolve(strict=True):
        raise PermissionError('Use the verified runtime_repair_v1 environment')
    if installed_environment() != expected:
        raise ValueError('Replacement environment fingerprint differs')


def original(path):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    validate(old, 'RECOVERY_SPEC')
    if (old['source_commit'] != PARENT_COMMIT or old['counts'] != COUNTS
            or path != Path(old['root']).resolve()/'campaign_spec.json'):
        raise ValueError('Expected the original 0c4a71a5 recovery execution')
    runtime(old['reuse']['environment'])
    project = Path(old['project_dir']).resolve(strict=True)
    env = dict(os.environ, PYTHONPATH=str(project/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating saved DE066 ancestry in original recovery source...', flush=True)
    result = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=project, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original recovery authentication failed:\n'+result.stderr)
    lines = [x.partition('=')[2] for x in result.stdout.splitlines() if x.startswith('FINALIZE_PARENT=')]
    if len(lines) != 1:
        raise ValueError('Missing or ambiguous original recovery replay')
    replay = json.loads(lines[0])
    if replay['spec'] != old or replay['jobs'] != JOBS or not replay['de066']:
        raise ValueError('Original recovery spec/jobs/DE066 evidence differs')
    return old, replay['jobs'], replay['de066']
