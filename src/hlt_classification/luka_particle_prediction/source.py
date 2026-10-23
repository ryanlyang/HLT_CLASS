"""Authenticate the exact completed U100-w5 comparator in its original source."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_embedding_aux.source import authenticate, matched_core
from hlt_classification.luka_u100_embedding.source import teacher
from hlt_classification.luka_u100_embedding.contracts import node as donor_node
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact, validate, COUNTS

BASE = '01d45d6057e799b4e262f8c590b51706803ebfd1'
MODULES = ('__init__','contracts','source','model','banks','training','campaign','submission','worker')
REQUIRED = {'src/hlt_classification/luka_particle_prediction/'+n+'.py' for n in MODULES} | {
    'scripts/luka_particle_prediction.py','scripts/queue_luka_particle_prediction.sh',
    'sbatch/run_luka_particle_prediction.sh','tests/test_luka_particle_prediction.py',
    'docs/plans/LUKA_PARTICLE_PREDICTION_PLAN.md','docs/contracts/LUKA_PARTICLE_PREDICTION.md',
    'docs/LUKA_PARTICLE_PREDICTION_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit the complete particle prediction implementation/tests/runbook')
    # Additive unrelated packages need not be imported and do not invalidate
    # these donors. Every previously tracked runtime file remains frozen.
    changed = git('diff','--name-status','--no-renames','--diff-filter=MDRT',BASE,commit,'--','src')
    if changed: raise ValueError('Existing donor runtime changed: '+repr(changed))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_particle_prediction/source.py':
        raise ValueError('Imported runtime escapes pinned project')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_embedding import direct
s=load_json(sys.argv[1]); direct.results(s)
root=Path(s['root'])
print('TOKEN_REFERENCE='+json.dumps(dict(anchor=s['anchor'],teacher=s['teacher'],reference=s['reference'],
    node=s['node'],counts=s['counts'],files={k:file_ref(root/f) for k,f in dict(
    receipt='receipt.json',report='fit/training_report.json',predictions='fit/validation_predictions.npz').items()}),sort_keys=True))
'''


def import_reference(path):
    path = Path(path).resolve(strict=True); saved = load_json(path)
    if (path != Path(saved['root'])/'campaign_spec.json' or saved['counts'] != COUNTS
            or saved['contract'] != 'LUKA_U100_EMBEDDING_DIRECT_SPEC/v1'):
        raise ValueError('Expected completed original direct trial U100-w5')
    validate_source_snapshot(saved['source_snapshot'],repository=saved['project_dir'],require_clean=True)
    ref, data = replay(path,saved['project_dir'],REPLAY,'TOKEN_REFERENCE=')
    anchor,parent = authenticate(data['anchor']['spec']['path'])
    t = teacher(anchor)
    if data['anchor'] != anchor or data['teacher'] != t or data['node'] != donor_node(parent):
        raise ValueError('U100 comparator teacher/population/recipe differs')
    matched_core(saved['project_dir'],Path(__file__).resolve().parents[3])
    for value in data['files'].values(): validate_file_ref(value)
    return artifact('REFERENCE',spec=ref,**data),parent


def validate_reference(ref):
    validate(ref,'REFERENCE'); validate_file_ref(ref['spec'])
    if ref['counts'] != COUNTS or ref['teacher']['coordinate'] != 'U100':
        raise ValueError('Wrong reference scope')
    for value in ref['files'].values(): validate_file_ref(value)
    validate_file_ref(ref['teacher']['files']['checkpoint'])
