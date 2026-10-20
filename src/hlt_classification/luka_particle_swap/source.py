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
from .contracts import artifact, validate, COUNTS
from hlt_classification.luka_attention_weight.reference_archive import SCOPE
from hlt_classification.luka_attention_weight import source as reference_source

BASE_COMMIT = '901535f026d32868e19c159a073947dfdbbbdd3c'
REFERENCE_COMMIT = '94f560259dd04253c0aa47b31dec3aedc105ac14'
MODULES = ('__init__', 'contracts', 'source', 'views', 'data', 'banks', 'training', 'retention', 'worker',
           'campaign', 'preflight', 'submission')
ADDITIONS = {'src/hlt_classification/luka_particle_swap/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_particle_swap.py', 'scripts/queue_luka_particle_swap.sh',
    'sbatch/run_luka_particle_swap.sh', 'tests/test_luka_particle_swap.py',
    'tests/test_luka_particle_swap_views.py',
    'docs/plans/LUKA_PARTICLE_SWAP_PLAN.md', 'docs/contracts/LUKA_PARTICLE_SWAP.md',
    'docs/LUKA_PARTICLE_SWAP_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    names = set(subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                              capture_output=True, text=True).stdout.splitlines())
    if not REQUIRED <= names or Path(__file__).resolve() != project/'src/hlt_classification/luka_particle_swap/source.py':
        raise ValueError('Commit the entire swap implementation and tests first')
    changes = subprocess.run(['git', '-C', str(project), 'diff', '--name-only', BASE_COMMIT,
                              '--', 'src'], check=True, capture_output=True, text=True).stdout.splitlines()
    if not set(changes) <= ADDITIONS:
        raise ValueError('Existing scientific source changed relative to registered base')
    return snapshot


def replay(path, project, program, marker, *, extra=()):
    ref = file_ref(path)
    env = dict(os.environ, PYTHONPATH=str(Path(project)/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-s', '-c', program, str(path), *map(str, extra)],
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
import json,sys,runpy
read_archive=runpy.run_path(sys.argv[2])['read_archive']
print('WEIGHT_REFERENCES='+json.dumps(read_archive(sys.argv[1]),sort_keys=True,allow_nan=False))
'''


def references(path, anchor):
    path = Path(path).resolve(strict=True)
    spec = load_json(path)
    if spec['source_commit'] != REFERENCE_COMMIT or path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected completed original lambda0/0.1 experiment')
    print('Verifying archived comparison predictions and reports; old banks are NOT reused...', flush=True)
    ref, evidence = replay(path, spec['project_dir'], REFERENCE_CHECK, 'WEIGHT_REFERENCES=',
                           extra=(Path(reference_source.__file__).with_name('reference_archive.py'),))
    # The anchor namespaces differ; exact parent spec and prepared payload are authoritative.
    old_anchor = evidence.pop('anchor')
    if (old_anchor['spec'] != anchor['spec'] or old_anchor['evidence'] != anchor['evidence']
            or evidence['prepared'] != anchor['evidence']['prepared']):
        raise ValueError('Reference split/anchor differs')
    proof = evidence.pop('archive')
    archive = artifact('REFERENCE_ARCHIVE', parents=dict(anchor=anchor['content_hash'],
        spec=spec['content_hash'], results=proof['results']), **proof)
    result = artifact('REFERENCES', parents=dict(anchor=anchor['content_hash']),
                      spec=ref, archive=archive, **evidence)
    validate_archive(result, anchor)
    return result


def validate_archive(imported, anchor):
    """Validate the copied archival scope without reopening discarded parents."""
    validate(imported, 'REFERENCES', dict(anchor=anchor['content_hash']))
    archive = imported['archive']
    validate(archive, 'REFERENCE_ARCHIVE', dict(anchor=anchor['content_hash'],
        spec=archive['spec'], results=archive['results']))
    if any(archive.get(k) != v for k, v in SCOPE.items()) or archive['source_commit'] != REFERENCE_COMMIT:
        raise ValueError('Archival comparison scope differs; not training/preflight acceptance')
    names = {a+'_'+c for a in ('LOGIT','ATTN') for c in ('D066','D033','D000')}
    if (set(imported['predictions']) != names or set(archive['training_reports']) != names
            or set(archive['training_receipts']) != {'train_'+n for n in names}
            or any(archive['recorded_dependencies'].get(n) != h
                   for n, h in archive['training_receipts'].items())
            or imported['spec'] not in archive['verified_files']
            or any(ref not in archive['verified_files'] for ref in imported['predictions'].values())):
        raise ValueError('Incomplete archival comparison evidence')
