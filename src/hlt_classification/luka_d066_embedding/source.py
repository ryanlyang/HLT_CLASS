"""Authenticate original controls, archived D066 checkpoint and matched w5."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate, COUNTS

BASE_COMMIT = '4a3124e9a3b190754ab195ca7430c00ceebcea78'
TEACHER_COMMIT = '94f560259dd04253c0aa47b31dec3aedc105ac14'
REFERENCE_COMMIT = '855c4f8d40b4db7837238c7ac6a3799597466e85'
MODULES = ('__init__', 'contracts', 'source', 'targets', 'training', 'worker',
           'campaign', 'preflight', 'submission', 'followup', 'priority')
ADDITIONS = {'src/hlt_classification/luka_d066_embedding/'+n+'.py' for n in MODULES}
REQUIRED = ADDITIONS | {'scripts/luka_d066_embedding.py', 'scripts/queue_luka_d066_embedding.sh',
    'sbatch/run_luka_d066_embedding.sh', 'tests/test_luka_d066_embedding.py',
    'tests/test_luka_d066_embedding_priority.py', 'docs/plans/LUKA_D066_EMBEDDING_PLAN.md',
    'docs/contracts/LUKA_D066_EMBEDDING.md', 'docs/LUKA_D066_EMBEDDING_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if (not REQUIRED <= set(git('ls-files'))
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_d066_embedding/source.py'
            or not set(git('diff', '--name-only', BASE_COMMIT, '--', 'src')) <= ADDITIONS):
        raise ValueError('Commit complete additive D066 implementation; preserve existing scientific runtime')
    return snapshot


TEACHER_CHECK = '''
import json,sys,runpy
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_attention.contracts import validate
a=runpy.run_path(sys.argv[2])['read_archive'](sys.argv[1])
s=load_json(sys.argv[1]); g=load_json(s['gate_path']); root=Path(s['root'])
n=next(n for n in g['nodes'] if n['node_id']=='LOGIT_D066')
if (n['coordinate'],n['branch'],n['attention_coefficient']) != ('D066','LOGIT',0.):
    raise ValueError('Expected LOGIT_D066, not another teacher')
task='train_LOGIT_D066'; r=load_json(root/task/'receipt.json')
validate(r,'TASK',dict(spec=s['content_hash']))
if r['content_hash'] != a['archive']['training_receipts'][task]:
    raise ValueError('Archived checkpoint receipt differs')
matches=[x for x in r['outputs'] if x['path']==task+'/selected.pt']
if len(matches)!=1: raise ValueError('Missing unique selected checkpoint binding')
p=validate_file_ref(matches[0],root=root)
refs={key:file_ref(root/task/name) for key,name in dict(checkpoint='selected.pt',
    report='training_report.json',predictions='validation_predictions.npz',receipt='receipt.json').items()}
print('D066_TEACHER='+json.dumps(dict(node=n,anchor=a['anchor'],prepared=a['prepared'],
    archive=a['archive'],files=refs),sort_keys=True,allow_nan=False))
'''


def teacher(path, anchor):
    path = Path(path).resolve(strict=True); spec = load_json(path)
    if spec['source_commit'] != TEACHER_COMMIT or path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected original94f56025 attention experiment')
    print('Authenticating retained LOGIT_D066 lineage and selected checkpoint; no old attention banks...', flush=True)
    helper = Path(__file__).parents[1]/'luka_attention_weight/reference_archive.py'
    ref, data = replay(path, spec['project_dir'], TEACHER_CHECK, 'D066_TEACHER=', extra=(helper,))
    if (data['anchor']['spec'] != anchor['spec'] or data['prepared'] != anchor['evidence']['prepared']
            or any(data['anchor']['evidence']['models'][n][key] != anchor['evidence']['models'][n][key]
                   for n in ('M0HLT','OFFLINE') for key in ('receipt','checkpoint','report','diagnostics'))):
        raise ValueError('D066 and CE anchor populations differ')
    for item in data['files'].values(): validate_file_ref(item)
    return artifact('TEACHER', parents=dict(anchor=anchor['content_hash'], spec=spec['content_hash']),
        spec=ref, node=data['node'], files=data['files'], archive=data['archive'],
        prepared=data['prepared'], coordinate='D066', target_role='train',
        checkpoint_payload_revalidated=True, legacy_attention_banks_reused=False)


REFERENCE_CHECK = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_sweep import debug
s=load_json(sys.argv[1]); result=debug.results(s); g=load_json(s['gate_path'])
if 'w5' in result['pending']: raise ValueError('OFFLINE-target w5 is not complete yet')
root=Path(s['root'])/'w5'
print('D066_REFERENCE='+json.dumps(dict(anchor=g['anchor'],counts=g['counts'],node=g['nodes']['w5'],
    files={k:file_ref(root/f) for k,f in dict(receipt='receipt.json',result='results.json',
    report='fit/training_report.json',predictions='fit/validation_predictions.npz').items()}),
    sort_keys=True,allow_nan=False))
'''


def reference(path, anchor, student):
    path = Path(path).resolve(strict=True); spec = load_json(path)
    if spec['source_commit'] != REFERENCE_COMMIT or path != Path(spec['root'])/'campaign_spec.json':
        raise ValueError('Expected transferred855c4f8d OFFLINE-target w5 reference')
    validate_source_snapshot(spec['source_snapshot'], repository=spec['project_dir'], require_clean=True)
    ref, data = replay(path, spec['project_dir'], REFERENCE_CHECK, 'D066_REFERENCE=')
    old_node = dict(data['node']); new_node = dict(student)
    for d in (old_node, new_node):
        d.pop('node_id'); d.pop('branch')
    if data['anchor'] != anchor or data['counts'] != COUNTS or old_node != new_node:
        raise ValueError('OFFLINE-target w5 split, seeds or student recipe differ')
    for item in data['files'].values(): validate_file_ref(item)
    return artifact('REFERENCE', parents=dict(anchor=anchor['content_hash'], spec=spec['content_hash']),
                    spec=ref, files=data['files'], node=data['node'])


def validate_imports(gate):
    for key, kind in (('teacher', 'TEACHER'), ('reference', 'REFERENCE')):
        value = gate[key]; validate(value, kind)
        if value['parents']['anchor'] != gate['anchor']['content_hash']:
            raise ValueError('Imported teacher/reference anchor differs')
        validate_file_ref(value['spec'])
        for item in value['files'].values(): validate_file_ref(item)
