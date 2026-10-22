"""Pin unchanged donors and authenticate the existing regular U100, never fusion."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_d066_embedding.source import reference
from hlt_classification.luka_u100_embedding.source import teacher, validate_imports

BASE = 'ed6d0af4be65f059356543f90583b30185702615'
MODULES = ('__init__', 'contracts', 'source', 'targets', 'training', 'worker', 'campaign', 'submission')
ADDITIONS = {'src/hlt_classification/luka_ce_warmstart/'+p+'.py' for p in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_ce_warmstart.py', 'scripts/queue_luka_ce_warmstart.sh',
    'sbatch/run_luka_ce_warmstart.sh', 'tests/test_luka_ce_warmstart.py',
    'docs/plans/LUKA_CE_WARMSTART_PLAN.md', 'docs/contracts/LUKA_CE_WARMSTART.md',
    'docs/LUKA_CE_WARMSTART_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    missing = REQUIRED - set(git('ls-files'))
    if missing:
        raise ValueError('Commit required CE warm-start files:\n'+'\n'.join(sorted(missing)))
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src')
    bad = [row for row in changes if not row.startswith('A\t') or row[2:] not in ADDITIONS]
    if bad:
        raise ValueError('Existing donor source changed or unreviewed additions:\n'+'\n'.join(bad))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_ce_warmstart/source.py':
        raise ValueError('Imported source escapes pinned project')
    return snapshot
