"""Original-source CE/teacher authentication; no dependence on attention banks."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from .contracts import artifact, COUNTS, node

BASE_COMMIT = '62c9b2df91eb3bda4fd4df4edc75c5f810a48c55'
PARENT_COMMIT = '74aa58c582ff6f497a2c6d91c503f5489fac3e9a'
MODULES = ('__init__', 'contracts', 'source', 'model', 'targets', 'training', 'campaign',
           'preflight', 'worker', 'submission')
ADDITIONS = {'src/hlt_classification/luka_embedding_aux/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_embedding_aux.py', 'scripts/queue_luka_embedding_aux.sh',
    'sbatch/run_luka_embedding_aux.sh', 'tests/test_luka_embedding_aux.py',
    'docs/plans/LUKA_EMBEDDING_AUX_PLAN.md', 'docs/contracts/LUKA_EMBEDDING_AUX.md',
    'docs/LUKA_EMBEDDING_AUX_RUNBOOK.md'}
MATCHED_CORE = tuple('src/hlt_classification/'+p for p in (
    'models/particle_transformer.py', 'jetclass2_delphes/model.py',
    'jetclass2_delphes/runner.py', 'jetclass2_delphes/batch128_recipe.py',
    'jetclass2_delphes/batch128_30e_recipe.py', 'luka_fullsim/regular_cache.py',
    'luka_fullsim/particles.py', 'luka_fullsim/stage2.py',
    'literature_context/transform.py', 'cms_proxy_ladder/inputs.py'))

PARENT_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_worker as w
from hlt_classification.luka_fullsim.regular_cache import metadata
s=load_json(sys.argv[1]); _,_,accepted=c.validate_spec(s)
if s['schema_version']!=6: raise ValueError('Expected original fixed30 v6')
selected={}; seen={}
for name in ('M0HLT','OFFLINE'):
    r=w.completed(s,'train_'+name,seen)
    if r is None: raise ValueError('Required parent fit is incomplete: '+name)
    directory=Path(s['root'])/('train_'+name)
    selected[name]={key:file_ref(directory/filename) for key,filename in dict(
        receipt='receipt.json',checkpoint='selected.pt',report='training_report.json',
        predictions='validation_predictions.npz',diagnostics='validation_diagnostics.json').items()}
parent=load_json(s['parent_import']['spec']['path'])
root=parent['admission']['prepared_root']; m,p,_,_=metadata(root)
print('EMBED_ANCHOR='+json.dumps(dict(prepared_root=root,prepared=p['content_hash'],
    raw_container=load_json(parent['gate_path'])['input_container'],environment=accepted['environment'],
    cache=m['content_hash'],models=selected),sort_keys=True,allow_nan=False))
'''


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    names = set(subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
        capture_output=True, text=True).stdout.splitlines())
    if not REQUIRED <= names or Path(__file__).resolve() != project/'src/hlt_classification/luka_embedding_aux/source.py':
        raise ValueError('Commit the complete embedding auxiliary implementation, tests and runbook')
    changed = set(subprocess.run(['git', '-C', str(project), 'diff', '--name-only', BASE_COMMIT, '--', 'src'],
        check=True, capture_output=True, text=True).stdout.splitlines())
    if not changed <= ADDITIONS:
        raise ValueError('Existing scientific runtime differs from registered donor pin')
    return snapshot


def authenticate(path):
    path = Path(path).resolve(strict=True)
    parent = load_json(path)
    from hlt_classification.luka_fullsim.alljets_fusion_campaign import validate
    validate(parent, 'SPEC')
    if (parent['schema_version'] != 6 or parent['source_commit'] != PARENT_COMMIT
            or parent['counts'] != COUNTS or parent['training_recipe'] != base_recipe()
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected completed original batch128 fixed30 trial ALLJETS parent')
    node(parent)  # Fail if the reference is not CE-only native HLT.
    validate_source_snapshot(parent['source_snapshot'], repository=parent['project_dir'], require_clean=True)
    core = matched_core(parent['project_dir'], Path(__file__).resolve().parents[3])
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(parent['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating M0HLT CE baseline and frozen OFFLINE teacher in original source...', flush=True)
    run = subprocess.run([sys.executable, '-B', '-s', '-c', PARENT_CHECK, str(path)],
        cwd=parent['project_dir'], env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed; preserve parent artifacts:\n'+run.stderr)
    lines = [x.split('=', 1)[1] for x in run.stdout.splitlines() if x.startswith('EMBED_ANCHOR=')]
    if len(lines) != 1:
        raise ValueError('Missing original-source evidence')
    evidence = json.loads(lines[0])
    if evidence['prepared'] != parent['prepared_sha256'] or set(evidence['models']) != {'M0HLT', 'OFFLINE'}:
        raise ValueError('Parent model/population binding differs')
    validate_file_ref(ref)
    return artifact('ANCHOR', parents=dict(parent=parent['content_hash']), spec=ref,
                    evidence=evidence, matched_core=core), parent


def matched_core(original, current_project):
    """Both scopes use identical native training/input code, not just a recipe label."""
    records = {}
    for name in MATCHED_CORE:
        old, new = sha256_file(Path(original)/name), sha256_file(Path(current_project)/name)
        if old != new:
            raise ValueError('Matched CE/input source differs from original parent: '+name)
        records[name] = new
    return records
