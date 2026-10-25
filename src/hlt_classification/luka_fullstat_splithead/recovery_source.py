"""Pinned failed-run replay and a narrow source boundary for export-only repair."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.foundation import _source
from .contracts import validate, COUNTS

FAILED_COMMIT = 'a7efcef6a8399cdcde80b2941e1a21910ba1bf67'
BASELINE = '234b029ec14a8d4861134e098cafbf939d325cd8'
MODIFIED = {f'src/hlt_classification/luka_fullstat_splithead/{name}.py'
            for name in ('training', 'worker', 'campaign', 'submission', 'kd_banks')}
ADDED = {f'src/hlt_classification/luka_fullstat_splithead/{name}.py'
         for name in ('recovery', 'recovery_source')}
ADDED |= {'scripts/luka_fullstat_splithead_recovery.py',
          'scripts/queue_luka_fullstat_splithead_recovery.sh',
          'sbatch/run_luka_fullstat_splithead_recovery.sh'}
REQUIRED = MODIFIED | ADDED | {'tests/test_luka_fullstat_splithead_recovery.py',
    'docs/contracts/LUKA_FULLSTAT_SPLITHEAD_RECOVERY.md',
    'docs/LUKA_FULLSTAT_SPLITHEAD_RECOVERY_RUNBOOK.md'}

REPLAY = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_splithead import campaign,submission
s=load_json(sys.argv[1]); campaign.check(s)
print('FAILED_SPLITHEAD='+json.dumps(dict(spec=s,jobs=submission.accepted_jobs(s)),sort_keys=True,allow_nan=False))
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
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_splithead/recovery_source.py'):
        raise ValueError('Recovery source changes outside the reviewed repair boundary')
    return snap


def original(path):
    path = Path(path).resolve(strict=True)
    old = load_json(path)
    validate(old, 'SPEC')
    if (old['source_commit'] != FAILED_COMMIT or old['counts'] != COUNTS
            or path != Path(old['root']).resolve()/'campaign_spec.json'):
        raise ValueError('Expected the original a7efcef6 FullStat split-head execution')
    project = Path(old['project_dir']).resolve(strict=True)
    env = dict(os.environ, PYTHONPATH=str(project/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating failed campaign and completed upstream teachers in original pinned source...', flush=True)
    result = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=project, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original campaign authentication failed:\n'+result.stderr)
    lines = [line.partition('=')[2] for line in result.stdout.splitlines() if line.startswith('FAILED_SPLITHEAD=')]
    if len(lines) != 1:
        raise ValueError('Missing or ambiguous failed-campaign replay')
    replay = json.loads(lines[0])
    if replay['spec'] != old or set(replay['jobs']) != {'train_DE066', 'train_DE033', 'train_DE000'}:
        raise ValueError('Original spec or exact three-job ledger differs')
    return old, replay['jobs']
