"""Original-source parent and completed embedding-bank import."""
from pathlib import Path
import json
import os
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux import source as old
from hlt_classification.luka_embedding_aux.contracts import validate as validate_embedding
from .contracts import artifact, COUNTS

BASE_COMMIT = '991c9d44161552c3ff8ed65eafd59933217f08aa'
MODULES = ('__init__', 'contracts', 'source', 'data', 'banks', 'training', 'worker',
           'campaign', 'preflight', 'submission')
ADDITIONS = {'src/hlt_classification/luka_embedding_ladder/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_embedding_ladder.py', 'scripts/queue_luka_embedding_ladder.sh',
    'sbatch/run_luka_embedding_ladder.sh', 'tests/test_luka_embedding_ladder.py',
    'docs/plans/LUKA_EMBEDDING_LADDER_PLAN.md', 'docs/contracts/LUKA_EMBEDDING_LADDER.md',
    'docs/LUKA_EMBEDDING_LADDER_RUNBOOK.md'}
PARENT_CHECK = old.PARENT_CHECK.replace("('M0HLT','OFFLINE')", "('M0HLT','OFFLINE','U000')")


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if (not REQUIRED <= set(git('ls-files'))
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_embedding_ladder/source.py'
            or not set(git('diff', '--name-only', BASE_COMMIT, '--', 'src')) <= ADDITIONS):
        raise ValueError('Commit complete additive implementation; existing runtime must remain unchanged')
    return snapshot


def replay(path, project, code, marker):
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(project)/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-s', '-c', code, str(path)],
        cwd=project, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Original-source authentication failed:\n'+result.stderr)
    rows = [x[len(marker):] for x in result.stdout.splitlines() if x.startswith(marker)]
    if len(rows) != 1:
        raise ValueError('Missing unique original-source evidence')
    validate_file_ref(ref)
    return ref, json.loads(rows[0])


def authenticate(path):
    from hlt_classification.luka_fullsim.alljets_fusion_campaign import validate
    from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe
    path = Path(path).resolve(strict=True)
    parent = load_json(path)
    validate(parent, 'SPEC')
    if (parent['schema_version'] != 6 or parent['source_commit'] != old.PARENT_COMMIT
            or parent['counts'] != COUNTS or parent['training_recipe'] != recipe()
            or path != Path(parent['root'])/'campaign_spec.json'):
        raise ValueError('Expected original fixed30 trial parent')
    validate_source_snapshot(parent['source_snapshot'], repository=parent['project_dir'], require_clean=True)
    core = old.matched_core(parent['project_dir'], Path(__file__).resolve().parents[3])
    print('Authenticating original U000, M0HLT and OFFLINE...', flush=True)
    ref, evidence = replay(path, parent['project_dir'], PARENT_CHECK, 'EMBED_ANCHOR=')
    if evidence['prepared'] != parent['prepared_sha256'] or set(evidence['models']) != {'M0HLT','OFFLINE','U000'}:
        raise ValueError('Parent population/models differ')
    return artifact('ANCHOR', parents=dict(parent=parent['content_hash']), spec=ref,
                    evidence=evidence, matched_core=core), parent


EMBED_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_aux import campaign,worker
s=load_json(sys.argv[1]); g,r=campaign.check_spec(s); result=worker.results(s)
print('EMBED_IMPORT='+json.dumps(dict(gate=g,preflight=r['content_hash'],
    bank=file_ref(Path(g['root'])/'targets/manifest.json'),
    payload=file_ref(Path(g['root'])/'targets/targets.npz'),
    predictions=file_ref(Path(s['root'])/'fit/validation_predictions.npz'),
    result=file_ref(Path(s['root'])/'results.json')),sort_keys=True,allow_nan=False))
'''


def import_embedding(path, anchor):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    validate_embedding(spec, 'SPEC')
    gate = load_json(spec['gate_path'])
    validate_source_snapshot(gate['source_snapshot'], repository=spec['project_dir'], require_clean=True)
    if path != Path(spec['root'])/'campaign_spec.json' or spec['source_commit'] != gate['source_commit']:
        raise ValueError('Embedding spec location/source differs')
    print('Authenticating completed single-stage embedding bank in original source...', flush=True)
    ref, evidence = replay(path, spec['project_dir'], EMBED_CHECK, 'EMBED_IMPORT=')
    imported = evidence.pop('gate')
    old_anchor = imported['anchor']
    if (old_anchor['spec'] != anchor['spec'] or imported['counts'] != COUNTS
            or any(old_anchor['evidence'][k] != anchor['evidence'][k] for k in
                   ('prepared', 'cache', 'prepared_root', 'raw_container', 'environment'))
            or any(old_anchor['evidence']['models'][n] != anchor['evidence']['models'][n]
                   for n in ('M0HLT', 'OFFLINE'))):
        raise ValueError('Embedding bank teacher/population/reference differs')
    return artifact('EMBEDDING_IMPORT', parents=dict(anchor=anchor['content_hash'], spec=spec['content_hash']),
                    spec=ref, **evidence)
