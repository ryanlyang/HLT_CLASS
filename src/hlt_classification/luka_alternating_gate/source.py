"""Pinned existing scientific source and original-source anchor authentication."""
from pathlib import Path
import subprocess
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import authenticate as original_anchor
from .contracts import artifact

BASE_COMMIT = 'e95c0d83a60b75cfc9acca4aaf377740fde8930e'
MODULES = ('__init__','contracts','source','storage','training','worker','campaign',
           'preflight','submission','banks')
ADDITIONS = {'src/hlt_classification/luka_alternating_gate/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_alternating_gate.py','scripts/queue_luka_alternating_gate.sh',
    'sbatch/run_luka_alternating_gate.sh','tests/test_luka_alternating_gate.py',
    'docs/plans/LUKA_ALTERNATING_GATE_PLAN.md',
    'docs/contracts/LUKA_ALTERNATING_GATE.md','docs/LUKA_ALTERNATING_GATE_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    names = set(subprocess.run(['git','-C',str(project),'ls-files'],check=True,
        capture_output=True,text=True).stdout.splitlines())
    if not REQUIRED <= names or Path(__file__).resolve() != project/'src/hlt_classification/luka_alternating_gate/source.py':
        raise ValueError('Commit complete alternating-gate source, tests and runbook first')
    changed = subprocess.run(['git','-C',str(project),'diff','--name-only',BASE_COMMIT,'--','src'],
                              check=True,capture_output=True,text=True).stdout.splitlines()
    if not set(changed) <= ADDITIONS:
        raise ValueError('Existing scientific source differs from alternating-gate donor pin')
    return snapshot


def authenticate(path):
    anchor, parent = original_anchor(path)
    return artifact('ANCHOR', parents=anchor['parents'], spec=anchor['spec'], evidence=anchor['evidence']), parent
