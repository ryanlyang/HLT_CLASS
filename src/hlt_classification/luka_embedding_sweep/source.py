"""Reuse original-source teacher/bank authentication without modifying old code."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_embedding_ladder.source import import_embedding

BASE_COMMIT = 'bf360343974121397a568fdb635a2c1c1011bc90'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'training', 'worker', 'preflight', 'submission', 'followup')
ADDITIONS = {'src/hlt_classification/luka_embedding_sweep/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_embedding_sweep.py', 'scripts/queue_luka_embedding_sweep.sh',
    'sbatch/run_luka_embedding_sweep.sh', 'tests/test_luka_embedding_sweep.py',
    'tests/test_luka_embedding_sweep_followup.py',
    'docs/plans/LUKA_EMBEDDING_SWEEP_PLAN.md', 'docs/contracts/LUKA_EMBEDDING_SWEEP.md',
    'docs/LUKA_EMBEDDING_SWEEP_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return set(subprocess.run(['git', '-C', str(project), *args], check=True,
                                  capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= git('ls-files')
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_embedding_sweep/source.py'
            or not git('diff', '--name-only', BASE_COMMIT, '--', 'src') <= ADDITIONS):
        raise ValueError('Commit complete additive weight implementation; preserve existing runtime')
    return snapshot
