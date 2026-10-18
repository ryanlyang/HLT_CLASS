"""Pinned existing scientific source and original-source anchor authentication."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import authenticate as original_anchor
from .contracts import artifact

BASE_COMMIT = '84fcbf25529091a2f32daf96bf8b910dec50dc3f'
MODULES = ('__init__','contracts','source','storage','training','worker','campaign',
           'preflight','submission','followup')
ADDITIONS = {'src/hlt_classification/luka_gating_cold/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_gating_cold.py','scripts/queue_luka_gating_cold.sh',
    'sbatch/run_luka_gating_cold.sh','tests/test_luka_gating_cold.py',
    'tests/test_luka_gating_cold_followup.py','docs/plans/LUKA_COLD_GATE_TIER3_PLAN.md',
    'docs/contracts/LUKA_COLD_GATE_TIER3.md','docs/LUKA_COLD_GATE_TIER3_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    names = set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,
        capture_output=True,text=True).stdout.splitlines())
    if not REQUIRED <= names or Path(__file__).resolve() != project/'src/hlt_classification/luka_gating_cold/source.py':
        raise ValueError('Commit complete cold-gate source, tests and runbook first')
    changed = subprocess.run(['git','-C',str(project),'diff','--name-only',BASE_COMMIT,'--','src'],
                              check=True,capture_output=True,text=True).stdout.splitlines()
    if not set(changed) <= ADDITIONS:
        raise ValueError('Existing scientific source differs from cold-gate donor pin')
    return snapshot


def authenticate(path):
    anchor, parent = original_anchor(path)
    return artifact('ANCHOR', parents=anchor['parents'], spec=anchor['spec'], evidence=anchor['evidence']), parent
