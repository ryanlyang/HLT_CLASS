"""Exact pushed source and data-only provenance; no historical teacher reuse."""
import subprocess
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.operations.fullstat_pt_audit import resolve_inputs
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullstat import cache
from hlt_classification.luka_fullstat_readout.source import runtime, RUNTIME
from .contracts import artifact, validate

REQUIRED = ('scripts/luka_fullstat_balanced_ladder.py', 'scripts/launch_luka_fullstat_balanced_ladder.sh',
    'scripts/queue_luka_fullstat_balanced_ladder.sh', 'sbatch/run_luka_fullstat_balanced_ladder.sh',
    'docs/contracts/LUKA_FULLSTAT_BALANCED_LADDER.md', 'docs/LUKA_FULLSTAT_BALANCED_LADDER_RUNBOOK.md',
    'docs/LUKA_FULLSTAT_BALANCED_LADDER_VALIDATION.md',
    'docs/plans/LUKA_FULLSTAT_BALANCED_LADDER_PLAN.md', 'tests/test_luka_fullstat_balanced_ladder.py',
    'tests/test_luka_fullstat_balanced_ladder_workflow.py','tests/test_luka_fullstat_balanced_ladder_launch.py')


def current(project, commit):
    project=Path(project).resolve(strict=True)
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_fullstat_balanced_ladder/source.py':
        raise ValueError('Imported balanced implementation is outside pinned project')
    snapshot=_source(project,commit)
    names=subprocess.run(['git','-C',str(project),'ls-files'],check=True,capture_output=True,text=True).stdout.splitlines()
    local={p.relative_to(project).as_posix() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')}
    if not (set(REQUIRED)|local)<=set(names):raise ValueError('Commit the full balanced implementation before queuing')
    # Include non-Python semantics (occupancy.json) and every transitive runtime module.
    files={n:meta.file_ref(project/n,root=project) for n in names
           if n.startswith(('src/','scripts/','sbatch/','configs/')) or n in REQUIRED}
    return artifact('SOURCE',snapshot=snapshot,commit=commit,files=files)


def authenticate(path):
    old,foundation_root,foundation,refs=resolve_inputs(path)
    e=old['anchor']['evidence']
    evidence={k:e[k] for k in ('foundation_root','cache_root','foundation_gate','prepared','raw_container',
                               'environment','cudnn_v8_disabled')}
    runtime(evidence['environment'])
    prepared=cache.metadata(foundation_root,evidence['cache_root'],gate_hash=evidence['foundation_gate'])
    if prepared[0]['content_hash']!=evidence['prepared'] or prepared[1]!=foundation:
        raise ValueError('Prepared population differs')
    record=artifact('INPUTS',evidence=evidence,refs=refs,historical_model_reuse=False)
    return record,prepared


def replay(value):
    validate(value,'INPUTS')
    for ref in value['refs']:meta.validate_file_ref(ref)
    actual,prepared=authenticate(value['refs'][0]['path'])
    if actual!=value:raise ValueError('Registered input provenance differs')
    return prepared
