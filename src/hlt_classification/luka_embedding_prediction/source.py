"""Pin unchanged donors and authenticate original U000 plus historical controls."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_ladder.source import authenticate

DONOR_BASE = '118aa375eea423e5e99338cd9a6eda198225ef21'
# Reviewed integration adds only the separate FullStat multiteacher package;
# all pre-existing scientific donors are unchanged from DONOR_BASE.
BASE = 'b21c2d4e01211a31e57d879c5a86ca90aab2ca0b'
MODULES = ('__init__', 'contracts', 'source', 'targets', 'data', 'training', 'worker', 'campaign', 'submission')
ADDITIONS = {'src/hlt_classification/luka_embedding_prediction/'+p+'.py' for p in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_embedding_prediction.py', 'scripts/queue_luka_embedding_prediction.sh',
    'sbatch/run_luka_embedding_prediction.sh', 'tests/test_luka_embedding_prediction.py',
    'docs/plans/LUKA_EMBEDDING_PREDICTION_PLAN.md', 'docs/contracts/LUKA_EMBEDDING_PREDICTION.md',
    'docs/LUKA_EMBEDDING_PREDICTION_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    missing = REQUIRED - set(git('ls-files'))
    if missing:
        raise ValueError('Commit required embedding prediction files:\n'+'\n'.join(sorted(missing)))
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src')
    bad = [row for row in changes if not row.startswith('A\t') or row[2:] not in ADDITIONS]
    if bad:
        raise ValueError('Existing donor source changed or unreviewed additions:\n'+'\n'.join(bad))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_embedding_prediction/source.py':
        raise ValueError('Imported source escapes pinned project')
    return snapshot
