"""Read-only original-source authentication and compact reference snapshots."""
import json
import os
from pathlib import Path
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_gating.source import PARENT_CHECK, PARENT_COMMIT
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe
from .contracts import artifact, COUNTS

BASE_COMMIT = '4e4a38a1a50bb2e227ace9f65657a8c20c585bb5'
REFERENCE_COMMIT = '94f560259dd04253c0aa47b31dec3aedc105ac14'
MODULES = ('__init__', 'contracts', 'source', 'training', 'retention', 'worker',
           'campaign', 'preflight', 'submission')
ADDITIONS = {'src/hlt_classification/luka_attention_weight/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_attention_weight.py', 'scripts/queue_luka_attention_weight.sh',
    'sbatch/run_luka_attention_weight.sh', 'tests/test_luka_attention_weight.py',
    'docs/plans/LUKA_ATTENTION_WEIGHT_SWEEP_PLAN.md', 'docs/contracts/LUKA_ATTENTION_WEIGHT_SWEEP.md',
    'docs/LUKA_ATTENTION_WEIGHT_SWEEP_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    names = set(subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                              capture_output=True, text=True).stdout.splitlines())
    if not REQUIRED <= names or Path(__file__).resolve() != project/'src/hlt_classification/luka_attention_weight/source.py':
        raise ValueError('Commit the entire coefficient implementation and tests first')
    changes = subprocess.run(['git', '-C', str(project), 'diff', '--name-only', BASE_COMMIT,
                              '--', 'src'], check=True, capture_output=True, text=True).stdout.splitlines()
    if not set(changes) <= ADDITIONS:
        raise ValueError('Existing scientific source changed relative to registered base')
    return snapshot


def replay(path, project, program, marker):
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(project)/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-s', '-c', program, str(path)],
        cwd=project, env=env, text=True, capture_output=True)
    if result.returncode:
        raise ValueError('Original-source replay failed:\n'+result.stderr)
    lines = [s[len(marker):] for s in result.stdout.splitlines() if s.startswith(marker)]
    if len(lines) != 1:
        raise ValueError('Missing exact original-source evidence')
    validate_file_ref(ref)
    return ref, json.loads(lines[0])


def authenticate(path):
    path = Path(path).resolve(strict=True)
    parent = load_json(path)
    from hlt_classification.luka_fullsim.alljets_fusion_campaign import validate
    validate(parent, 'SPEC')
    if (parent['schema_version'] != 6 or parent['source_commit'] != PARENT_COMMIT
            or parent['counts'] != COUNTS or parent['training_recipe'] != base_recipe()
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original fixed30 scientific parent')
    validate_source_snapshot(parent['source_snapshot'], repository=parent['project_dir'], require_clean=True)
    print('Authenticating the shared fixed30 U100 and controls...', flush=True)
    ref, evidence = replay(path, parent['project_dir'], PARENT_CHECK, 'LUKA_GATE_ANCHOR=')
    if evidence['prepared'] != parent['prepared_sha256']:
        raise ValueError('Anchor population differs')
    return artifact('ANCHOR', parents=dict(parent=parent['content_hash']), spec=ref, evidence=evidence), parent


REFERENCE_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_attention import campaign,worker
s=load_json(sys.argv[1]); g,_=campaign.check_spec(s)
if worker.completed(s,g,'aggregate') is None: raise ValueError('Reference experiment incomplete')
rows=worker.result_rows(s,g)
predictions={n['node_id']:file_ref(Path(s['root'])/('train_'+n['node_id'])/'validation_predictions.npz') for n in g['nodes']}
print('WEIGHT_REFERENCES='+json.dumps(dict(rows=rows,predictions=predictions,
    parent=g['anchor']['content_hash'],prepared=g['anchor']['evidence']['prepared']),sort_keys=True,allow_nan=False))
'''


def references(path, anchor):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    if spec['source_commit'] != REFERENCE_COMMIT or path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected completed original lambda0/0.1 experiment')
    ref, evidence = replay(path, spec['project_dir'], REFERENCE_CHECK, 'WEIGHT_REFERENCES=')
    # The anchor namespaces differ; exact parent spec and prepared payload are authoritative.
    gate = load_json(spec['gate_path'])
    if gate['anchor']['spec'] != anchor['spec'] or evidence['prepared'] != anchor['evidence']['prepared']:
        raise ValueError('Reference split/anchor differs')
    return artifact('REFERENCES', parents=dict(anchor=anchor['content_hash']), spec=ref, **evidence)
