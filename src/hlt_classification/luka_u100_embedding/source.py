"""Pin unchanged donors and authenticate the existing regular U100, never fusion."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_embedding_aux.contracts import validate as validate_anchor
from hlt_classification.luka_d066_embedding.source import reference, REFERENCE_COMMIT
from hlt_classification.luka_d066_embedding.contracts import validate as validate_reference
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate

BASE = 'dd51e8e25770482134809dbb34d4766f8dac74b9'
MODULES = ('__init__', 'contracts', 'source', 'targets', 'training', 'worker', 'direct')
ADDITIONS = {'src/hlt_classification/luka_u100_embedding/'+p+'.py' for p in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_u100_embedding.py', 'scripts/queue_luka_u100_embedding.sh',
    'sbatch/run_luka_u100_embedding.sh', 'tests/test_luka_u100_embedding.py',
    'docs/plans/LUKA_U100_EMBEDDING_PLAN.md', 'docs/contracts/LUKA_U100_EMBEDDING.md',
    'docs/LUKA_U100_EMBEDDING_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    missing = REQUIRED - set(git('ls-files'))
    if missing:
        raise ValueError('Commit required U100 embedding files:\n'+'\n'.join(sorted(missing)))
    changes = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src')
    bad = [row for row in changes if not row.startswith('A\t') or row[2:] not in ADDITIONS]
    if bad:
        raise ValueError('Existing donor source changed or unreviewed additions:\n'+'\n'.join(bad))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_embedding/source.py':
        raise ValueError('Imported source escapes pinned project')
    return snapshot


TEACHER_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import alljets_fusion_campaign as c, alljets_fusion_worker as w
s=load_json(sys.argv[1]); c.validate_spec(s)
n=next(n for n in c.scoped_nodes(s) if n['node_id']=='COARSE_U100')
if n['coordinate']!='U100' or n.get('context_coordinate') is not None:
    raise ValueError('Expected regular single-encoder COARSE_U100')
r=w.completed(s,'train_COARSE_U100')
if r is None: raise ValueError('Original regular COARSE_U100 incomplete')
root=Path(s['root'])/'train_COARSE_U100'
print('U100_TEACHER='+json.dumps(dict(node=n,prepared=s['prepared_sha256'],files={
    key:file_ref(root/name) for key,name in dict(checkpoint='selected.pt',report='training_report.json',
    predictions='validation_predictions.npz',receipt='receipt.json').items()}),sort_keys=True,allow_nan=False))
'''


def teacher(anchor):
    path = anchor['spec']['path']; parent = load_json(path)
    ref, evidence = replay(path, parent['project_dir'], TEACHER_CHECK, 'U100_TEACHER=')
    if evidence['prepared'] != anchor['evidence']['prepared']:
        raise ValueError('U100 population differs from HLT anchor')
    for item in evidence['files'].values():
        validate_file_ref(item)
    return artifact('TEACHER', parents=dict(anchor=anchor['content_hash']), spec=ref,
                    coordinate='U100', target_role='train', **evidence)


def validate_imports(spec, *, payloads=True):
    validate_anchor(spec['anchor'], 'ANCHOR')
    validate(spec['teacher'], 'TEACHER', dict(anchor=spec['anchor']['content_hash']))
    validate_reference(spec['reference'], 'REFERENCE')
    teacher = spec['teacher']; reference = spec['reference']; anchor = spec['anchor']
    if (teacher['spec'] != anchor['spec'] or teacher['coordinate'] != 'U100'
            or teacher['target_role'] != 'train'
            or teacher['node']['node_id'] != 'COARSE_U100' or teacher['node']['coordinate'] != 'U100'
            or teacher['node'].get('context_coordinate') is not None
            or teacher['prepared'] != anchor['evidence']['prepared']
            or set(teacher['files']) != {'checkpoint', 'report', 'predictions', 'receipt'}
            or reference['parents']['anchor'] != anchor['content_hash']):
        raise ValueError('Imported teacher/reference binding differs')
    previous, current_node = dict(reference['node']), dict(spec['node'])
    for value in (previous, current_node):
        value.pop('node_id'); value.pop('branch')
    if previous != current_node:
        raise ValueError('Historical w5 student differs')
    if not payloads:
        return
    for key in ('anchor', 'teacher', 'reference'):
        validate_file_ref(spec[key]['spec'])
    for imported in (spec['teacher'], spec['reference']):
        for ref in imported['files'].values():
            validate_file_ref(ref)
