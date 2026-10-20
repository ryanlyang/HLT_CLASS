"""Reuse original-source teacher/bank authentication without modifying old code."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_embedding_ladder.source import import_embedding

BASE_COMMIT = 'ed129fa95a885f0f3b884b2dc861b1776327bc82'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'training', 'worker', 'preflight', 'submission')
ADDITIONS = {'src/hlt_classification/luka_embedding_weight/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_embedding_weight.py', 'scripts/queue_luka_embedding_weight.sh',
    'sbatch/run_luka_embedding_weight.sh', 'tests/test_luka_embedding_weight.py',
    'docs/plans/LUKA_EMBEDDING_WEIGHT_PLAN.md', 'docs/contracts/LUKA_EMBEDDING_WEIGHT.md',
    'docs/LUKA_EMBEDDING_WEIGHT_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return set(subprocess.run(['git', '-C', str(project), *args], check=True,
                                  capture_output=True, text=True).stdout.splitlines())
    if (not REQUIRED <= git('ls-files')
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_embedding_weight/source.py'
            or not git('diff', '--name-only', BASE_COMMIT, '--', 'src') <= ADDITIONS):
        raise ValueError('Commit complete additive weight implementation; preserve existing runtime')
    return snapshot
