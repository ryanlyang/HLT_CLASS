"""Keep historical runtime unchanged and authenticate original, completed teachers."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_multiview.source import import_parent, validate_import

BASE = '8ad0b381547134569818b4258f3cd31b509e9efb'
MODULES = ('__init__', 'contracts', 'source', 'model', 'targets', 'sampling', 'losses',
           'training', 'resources', 'worker', 'campaign', 'submission')
RUNTIME = {'src/hlt_classification/luka_u100_contrastive/'+n+'.py' for n in MODULES} | {
    'scripts/luka_u100_contrastive.py', 'scripts/queue_luka_u100_contrastive.sh',
    'sbatch/run_luka_u100_contrastive.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_u100_contrastive.py',
    'docs/contracts/LUKA_U100_CONTRASTIVE.md', 'docs/LUKA_U100_CONTRASTIVE_RUNBOOK.md',
    'docs/plans/LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md'}


def compatible_changes(rows):
    return all(r[:2] in ('A\t', 'M\t') and r[2:] in RUNTIME for r in rows)


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit complete contrastive implementation, tests, plan, contract and runbook')
    if not compatible_changes(git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')):
        raise ValueError('Unreviewed runtime change outside the additive contrastive campaign')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_contrastive/source.py':
        raise ValueError('Imported runtime is outside pinned project')
    return snapshot
