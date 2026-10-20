"""Original-source FullStat replay plus additive source compatibility."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat_embedding.source import authenticate as authenticate_d033
from .contracts import artifact

DONOR = '1dff4e5767e66fb5f355cf5d592626ecffcf73ea'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'banks', 'training',
           'checkpoints', 'worker', 'preflight', 'submission')
RUNTIME = {f'src/hlt_classification/luka_fullstat_replace_gate/{n}.py' for n in MODULES} | {
    'scripts/luka_fullstat_replace_gate.py', 'scripts/queue_luka_fullstat_replace_gate.sh',
    'sbatch/run_luka_fullstat_replace_gate.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_fullstat_replace_gate.py',
    'docs/plans/LUKA_FULLSTAT_REPLACE_GATE_PLAN.md',
    'docs/contracts/LUKA_FULLSTAT_REPLACE_GATE.md', 'docs/LUKA_FULLSTAT_REPLACE_GATE_RUNBOOK.md',
    'docs/LUKA_REPLACE_GATE_STRATEGY_HANDOFF.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if (not REQUIRED <= set(git('ls-files'))
            or not set(git('diff', '--name-only', DONOR, commit, '--', 'src', 'scripts', 'sbatch')) <= RUNTIME
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_fullstat_replace_gate/source.py'):
        raise ValueError('Commit complete additive gate source; old runtime must remain frozen')
    return snapshot


def authenticate(path, project):
    checked, parent = authenticate_d033(path, project)
    return artifact('ANCHOR', parents=dict(original=checked['content_hash']),
        original=checked, spec=checked['spec'], evidence=checked['evidence'],
        logits=checked['logits']), parent
