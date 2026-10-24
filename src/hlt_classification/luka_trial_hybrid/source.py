"""Reuse authenticated trial native/KD and embedding evidence; never FullStat data."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_joint_prediction.source import import_acceptance, validate_acceptance

BASE = '2376559bc9b1d0ff455bbdb242ae5769a54d6168'
MODULES = ('__init__','contracts','source','model','banks','training','campaign','submission','worker')
REQUIRED = {'src/hlt_classification/luka_trial_hybrid/'+n+'.py' for n in MODULES} | {
    'scripts/luka_trial_hybrid.py','scripts/queue_luka_trial_hybrid.sh','sbatch/run_luka_trial_hybrid.sh',
    'tests/test_luka_trial_hybrid.py','docs/plans/LUKA_TRIAL_HYBRID_PLAN.md',
    'docs/contracts/LUKA_TRIAL_HYBRID.md','docs/LUKA_TRIAL_HYBRID_RUNBOOK.md'}


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit complete trial hybrid implementation, tests and documentation')
    changed=git('diff','--name-status','--no-renames','--diff-filter=MDRT',BASE,commit,'--','src')
    if changed: raise ValueError('Existing runtime differs from donor pin: '+repr(changed))
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_trial_hybrid/source.py':
        raise ValueError('Imported source escapes pinned worktree')
    return snapshot
