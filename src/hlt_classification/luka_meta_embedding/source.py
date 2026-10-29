"""Exact original checkpoint evidence; retired sweep target arrays are not needed."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_embedding_aux.source import authenticate
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate

BASE='830a2afb2b3350c14ae147d915b8d3072701269c'
W5_COMMIT='855c4f8d40b4db7837238c7ac6a3799597466e85'
MODULES=('__init__','contracts','source','model','meta','training','worker','campaign','submission')
ADDITIONS={'src/hlt_classification/luka_meta_embedding/'+p+'.py' for p in MODULES}
REQUIRED=ADDITIONS|{'scripts/luka_meta_embedding.py','scripts/queue_luka_meta_embedding.sh',
    'sbatch/run_luka_meta_embedding.sh','tests/test_luka_meta_embedding.py',
    'docs/plans/LUKA_META_EMBEDDING_PLAN.md','docs/contracts/LUKA_META_EMBEDDING.md',
    'docs/LUKA_META_EMBEDDING_RUNBOOK.md'}


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    missing=REQUIRED-set(git('ls-files'))
    if missing: raise ValueError('Commit complete meta-embedding implementation:\n'+'\n'.join(sorted(missing)))
    changed=git('diff','--name-status','--no-renames',BASE,commit,'--','src')
    if any(not r.startswith('A\t') or r[2:] not in ADDITIONS for r in changed):
        raise ValueError('Existing scientific runtime changed; preserve donor source and use additive commit')
    if Path(__file__).resolve()!=project/'src/hlt_classification/luka_meta_embedding/source.py':
        raise ValueError('Imported code escapes pinned source')
    return snapshot


REPLAY='''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_embedding_sweep import debug,campaign,contracts,worker
s=load_json(sys.argv[1]); g=load_json(s['gate_path'])
campaign.validate_gate_record(g)
p=load_json(Path(g['root'])/'preflight/receipt.json')
contracts.validate(p,'PREFLIGHT',dict(gate=g['content_hash'],bank=p['bank_hash']))
from hlt_classification.luka_embedding_aux.contracts import validate as bank_validate,recipe as bank_recipe
bank_path=Path(g['root'])/'preflight/target_manifest.json'
bank=load_json(bank_path)
bank_validate(bank,'BANK',dict(recipe=bank_recipe()['content_hash'],prepared=g['anchor']['evidence']['prepared'],
    teacher=g['anchor']['evidence']['models']['OFFLINE']['checkpoint']['sha256']))
if (bank['content_hash']!=p['bank_hash'] or bank['role']!='train' or bank['coordinate']!='OFFLINE'
        or bank['teacher']!=g['anchor']['evidence']['models']['OFFLINE']['checkpoint']):
    raise ValueError('Original target identity differs')
debug.validate_spec(s,g,p)
folder=Path(s['root'])/'w5'; receipt=load_json(folder/'receipt.json')
debug.validate(receipt,'SCIENCE',dict(spec=s['content_hash'],preflight=p['content_hash']))
expected=['results.json','fit/receipt.json','fit/selected.pt','fit/training_report.json','fit/validation_predictions.npz']
if receipt['arm']!='w5' or receipt['weight']!=5. or [x['path'] for x in receipt['outputs']]!=expected:
    raise ValueError('Wrong original w5 receipt')
for item in receipt['outputs']: validate_file_ref(item,root=folder)
report,arrays=worker.validate_fit(folder/'fit',contracts.arm_gate(g,'w5'),p['bank_hash'],technical=False)
files={k:file_ref(folder/name) for k,name in dict(receipt='receipt.json',result='results.json',
    fit_receipt='fit/receipt.json',checkpoint='fit/selected.pt',report='fit/training_report.json',
    predictions='fit/validation_predictions.npz').items()}
print('META_W5='+json.dumps(dict(anchor=g['anchor'],node=g['nodes']['w5'],files=files,
    bank_hash=p['bank_hash'],bank_manifest=file_ref(bank_path),gate=file_ref(s['gate_path']),
    preflight=file_ref(Path(g['root'])/'preflight/receipt.json')),
    sort_keys=True,allow_nan=False))
'''


def reference(path,anchor):
    path=Path(path).resolve(strict=True); original=load_json(path)
    if original['source_commit']!=W5_COMMIT or path!=Path(original['root'])/'campaign_spec.json':
        raise ValueError('Expected exact original debug OFFLINE-w5 sweep')
    validate_source_snapshot(original['source_snapshot'],repository=original['project_dir'],require_clean=True)
    print('Authenticating selected OFFLINE-w5 native checkpoint in its original source...',flush=True)
    ref,evidence=replay(path,original['project_dir'],REPLAY,'META_W5=')
    n=evidence['node']
    if (evidence['anchor']!=anchor or n['coordinate']!='D000' or n['teacher'] is not None
            or n.get('context_coordinate') is not None or n['auxiliary_weight']!=5.):
        raise ValueError('Original w5 teacher/population/objective differs')
    for f in [*evidence['files'].values(),evidence['gate'],evidence['preflight'],evidence['bank_manifest']]: validate_file_ref(f)
    return artifact('REFERENCE',parents=dict(anchor=anchor['content_hash']),spec=ref,**evidence)


def validate_imports(spec):
    from hlt_classification.luka_embedding_aux.contracts import validate as old_validate
    anchor=spec['anchor']; ref=spec['reference']
    old_validate(anchor,'ANCHOR'); validate(ref,'REFERENCE',dict(anchor=anchor['content_hash']))
    if ref['anchor']!=anchor or spec['initial_checkpoint']!=ref['files']['checkpoint']:
        raise ValueError('Initializer/teacher binding differs')
    for f in [anchor['spec'],ref['spec'],ref['gate'],ref['preflight'],ref['bank_manifest'],*ref['files'].values(),
              *[v for model in anchor['evidence']['models'].values() for v in model.values()]]:
        validate_file_ref(f)
