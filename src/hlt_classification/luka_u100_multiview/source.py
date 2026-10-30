"""Pinned additive source and original-source authentication of historical runs."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_u100_embedding.source import teacher
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate, COUNTS

BASE = 'ac76bbd9866cb00891bc554dac7ae8b31778c643'
MODULES = ('__init__', 'contracts', 'source', 'views', 'teachers', 'model', 'targets',
           'training', 'worker', 'campaign', 'submission', 'acceptance', 'retention', 'offline_cache')
ADDITIONS = {'src/hlt_classification/luka_u100_multiview/'+n+'.py' for n in MODULES}
ENTRYPOINTS = {'scripts/luka_u100_multiview.py', 'scripts/queue_luka_u100_multiview.sh',
              'sbatch/run_luka_u100_multiview.sh'}
REQUIRED = ADDITIONS | ENTRYPOINTS | {'tests/test_luka_u100_multiview.py',
    'tests/test_luka_u100_multiview_offline.py',
    'docs/plans/LUKA_U100_MULTIVIEW_READOUT_PLAN.md', 'docs/contracts/LUKA_U100_MULTIVIEW.md',
    'docs/LUKA_U100_MULTIVIEW_RUNBOOK.md'}


def compatible_changes(changed):
    return all(line[:2] in ('A\t', 'M\t') and line[2:] in ADDITIONS | ENTRYPOINTS
               for line in changed)


def current(project, commit):
    project = Path(project).resolve()
    result = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True,
                              capture_output=True, text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit the complete multiview implementation/tests/docs')
    git('merge-base', '--is-ancestor', BASE, commit)
    changed = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if not compatible_changes(changed):
        raise ValueError('Unexpected donor/runtime changes since reviewed source')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_multiview/source.py':
        raise ValueError('Runtime import escapes pinned project')
    return result


REPLAY = '''
import json,re,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_embedding import direct,worker,contracts as c
s=load_json(sys.argv[1]); direct.check_spec(s)
root=Path(s['root'])
# Result-only archive validation. The obsolete TRAIN target array is never
# reused; its retained receipt digest is checked against its bank manifest.
r=load_json(root/'receipt.json')
c.validate(r,'DIRECT_SCIENCE',dict(spec=s['content_hash'],resources=s['resources']['content_hash']))
required={'teacher_readback.json','targets/manifest.json','targets/targets.npz','results.json',
    'fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz'}
required|={'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100','original50k')}
if (r['new_preflight_performed'] is not False or not re.fullmatch('[1-9][0-9]*',r['job_id'])
        or {x['path'] for x in r['outputs']}!=required
        or len(r['outputs'])!=len(required)): raise ValueError('Original result receipt coverage differs')
for x in r['outputs']:
    if x['path']!='targets/targets.npz': validate_file_ref(x,root=root)
bank=load_json(root/'targets/manifest.json')
c.validate(bank,'BANK',dict(recipe=c.recipe()['content_hash'],prepared=s['anchor']['evidence']['prepared'],teacher=s['teacher']['content_hash']))
payload=next(x for x in r['outputs'] if x['path']=='targets/targets.npz')
if (bank['role']!='train' or bank['coordinate']!='U100' or bank['width']!=128
        or bank['rows']!=s['counts']['train'] or bank['teacher']!=s['teacher']['files']['checkpoint']
        or dict(bank['payload'],path='targets/targets.npz')!=payload):
    raise ValueError('Archived target metadata differs')
proof=load_json(root/'teacher_readback.json'); c.validate(proof,'TEACHER_READBACK',dict(teacher=s['teacher']['content_hash']))
if (proof['native_probability_parity'] is not True or proof['normalization_used'] is not False
        or proof['coordinate']!='U100' or proof['role']!='validation' or proof['rows']!=s['counts']['validation']
        or (proof['rtol'],proof['atol'])!=(1e-5,1e-6)
        or not 0<=proof['maximum_absolute_error']<=1.1e-5): raise ValueError('Archived teacher proof differs')
_,student=worker.validate_fit(root/'fit',s,bank['content_hash'],technical=False)
fit=load_json(root/'fit/receipt.json'); result=load_json(root/'results.json')
c.validate(result,'DIRECT_RESULTS',dict(spec=s['content_hash'],fit=fit['content_hash']))
if (result['counts']!=s['counts'] or result['new_preflight_performed'] is not False
        or result['selection_population']!='entire_validation' or result['original50k_selection'] is not False
        or result['single_training_seed'] is not True
        or result['rows']!=worker.comparison(s,student,folder=root/'references')):
    raise ValueError('Archived result metric replay differs')
print('MULTIVIEW_IMPORT='+json.dumps(dict(anchor=s['anchor'],teacher=s['teacher'],node=s['node'],
    counts=s['counts'],files={k:file_ref(root/v) for k,v in dict(
    receipt='receipt.json',report='fit/training_report.json',predictions='fit/validation_predictions.npz',
    subset='references/original50k.npz',bank_manifest='targets/manifest.json',readback='teacher_readback.json',
    fit_receipt='fit/receipt.json',checkpoint='fit/selected.pt',results='results.json').items()}),sort_keys=True,allow_nan=False))
'''


def import_parent(parent_path, reference_path):
    anchor, parent = authenticate(parent_path)
    u100 = teacher(anchor)
    saved = load_json(reference_path)
    if saved['source_commit'] != 'ed6d0af4be65f059356543f90583b30185702615':
        raise ValueError('Use the historical direct U100-w5 campaign')
    validate_source_snapshot(saved['source_snapshot'], repository=saved['project_dir'], require_clean=True)
    ref, evidence = replay(reference_path, saved['project_dir'], REPLAY, 'MULTIVIEW_IMPORT=')
    if evidence['anchor'] != anchor or evidence['teacher'] != u100 or evidence['counts'] != COUNTS:
        raise ValueError('Historical U100-w5 lineage differs')
    oracle = next(n for n in parent['nodes'] if n['node_id'] == 'OFFLINE')
    result = artifact('IMPORT', parents=dict(anchor=anchor['content_hash'], teacher=u100['content_hash']),
        spec=ref, anchor=anchor, teacher=u100, node=evidence['node'], oracle_node=oracle,
        files=evidence['files'], reference_source=saved['source_snapshot'], reference_project=saved['project_dir'],
        historical_training_payload_reused=False, historical_import_scope='retained_results_only_v1')
    validate_import(result)
    return result


def validate_import(value):
    validate(value, 'IMPORT', dict(anchor=value['anchor']['content_hash'], teacher=value['teacher']['content_hash']))
    if (value['historical_training_payload_reused'] is not False
            or value['historical_import_scope'] != 'retained_results_only_v1'
            or set(value['files']) != {'receipt', 'report', 'predictions', 'subset', 'bank_manifest',
                                       'readback', 'fit_receipt', 'checkpoint', 'results'}):
        raise ValueError('Historical result-only import scope differs')
    validate_file_ref(value['spec']); validate_file_ref(value['anchor']['spec'])
    for spec_ref in (value['spec'], value['anchor']['spec']):
        old = load_json(spec_ref['path'])
        validate_source_snapshot(old['source_snapshot'], repository=old['project_dir'], require_clean=True)
    for ref in (*value['files'].values(), *value['teacher']['files'].values()):
        validate_file_ref(ref)
    for model in value['anchor']['evidence']['models'].values():
        for ref in model.values():
            validate_file_ref(ref)
    if (value['teacher']['coordinate'] != 'U100' or value['teacher']['node']['context_coordinate'] is not None
            or value['node']['coordinate'] != 'D000' or value['node']['teacher'] is not None
            or value['oracle_node']['coordinate'] != 'OFFLINE'):
        raise ValueError('Reference model roles differ')
