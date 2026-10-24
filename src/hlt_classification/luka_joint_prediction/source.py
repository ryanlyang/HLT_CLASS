"""Reuse completed token-study acceptance, without launching a new preflight."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_particle_prediction import source as donor
from hlt_classification.luka_particle_prediction.contracts import validate as old_validate
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact, validate, COUNTS, ARMS

BASE = 'd8dba242c20c65a9c5155f93a68a5eaf12ba5063'
MODULES = ('__init__','contracts','source','model','banks','training','campaign','submission','worker')
REQUIRED = {'src/hlt_classification/luka_joint_prediction/'+n+'.py' for n in MODULES} | {
    'scripts/luka_joint_prediction.py','scripts/queue_luka_joint_prediction.sh',
    'sbatch/run_luka_joint_prediction.sh','tests/test_luka_joint_prediction.py',
    'docs/plans/LUKA_JOINT_PREDICTION_PLAN.md','docs/contracts/LUKA_JOINT_PREDICTION.md',
    'docs/LUKA_JOINT_PREDICTION_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    if not REQUIRED <= set(git('ls-files')):
        raise ValueError('Commit complete joint prediction implementation/tests/runbook')
    changed = git('diff','--name-status','--no-renames','--diff-filter=MDRT',BASE,commit,'--','src')
    if changed: raise ValueError('Existing donor runtime changed: '+repr(changed))
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_joint_prediction/source.py':
        raise ValueError('Imported runtime escapes pinned project')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_particle_prediction import campaign,worker
s=load_json(sys.argv[1]); campaign.check(s)
p=worker.validate_preflight(s); r=worker.results(s)
if r['pending']: raise ValueError('Both original particle fits must be complete')
root=Path(s['root'])
files={'preflight':file_ref(root/'preflight/receipt.json'),
       'bank':file_ref(root/'preflight/bank_manifest.json')}
predictions={}
for arm in ('aligned','set'):
    files[arm]=file_ref(root/arm/'receipt.json')
    predictions['TOKEN_'+arm.upper()]=file_ref(root/arm/'fit/validation_predictions.npz')
print('JOINT_ACCEPTANCE='+json.dumps(dict(reference=s['reference'],anchor=s['anchor'],teacher=s['teacher'],
    counts=s['counts'],resources={k:s[k] for k in ('execution_site','cpus','memory_mb')},
    preflight=p,bank=worker.retained_bank(root/'preflight',s),files=files,predictions=predictions),
    sort_keys=True,allow_nan=False))
'''


def import_acceptance(path):
    path = Path(path).resolve(strict=True); saved = load_json(path)
    if (path != Path(saved['root'])/'campaign_spec.json' or saved['counts'] != COUNTS
            or saved['contract'] != 'LUKA_PARTICLE_PREDICTION_SPEC/v1' or saved['source_commit'] != BASE):
        raise ValueError('Expected completed original debug token study d8dba242')
    validate_source_snapshot(saved['source_snapshot'],repository=saved['project_dir'],require_clean=True)
    ref, evidence = replay(path,saved['project_dir'],REPLAY,'JOINT_ACCEPTANCE=')
    donor.matched_core(saved['project_dir'],Path(__file__).resolve().parents[3])
    value = artifact('ACCEPTANCE',spec=ref,**evidence)
    validate_acceptance(value)
    return value,load_json(value['anchor']['spec']['path'])


def validate_acceptance(value):
    validate(value,'ACCEPTANCE'); validate_file_ref(value['spec'])
    saved = load_json(value['spec']['path']); old_validate(saved,'SPEC')
    if (saved['source_commit'] != BASE or saved['counts'] != COUNTS or value['counts'] != COUNTS
            or any(value[k] != saved[k] for k in ('anchor','reference','teacher'))
            or value['resources'] != {k:saved[k] for k in ('execution_site','cpus','memory_mb')}
            or set(value['files']) != {'preflight','bank',*ARMS}
            or set(value['predictions']) != {'TOKEN_ALIGNED','TOKEN_SET'}):
        raise ValueError('Inherited token study differs')
    donor.validate_reference(value['reference'])
    for ref in (*value['files'].values(),*value['predictions'].values()): validate_file_ref(ref)
    root = Path(saved['root'])
    expected = dict(preflight=root/'preflight/receipt.json',bank=root/'preflight/bank_manifest.json',
                    **{a:root/a/'receipt.json' for a in ARMS})
    if (any(value['files'][k] != file_ref(p) for k,p in expected.items())
            or any(value['predictions']['TOKEN_'+a.upper()] != file_ref(root/a/'fit/validation_predictions.npz') for a in ARMS)):
        raise ValueError('Inherited file paths escape exact original outputs')
    pre = value['preflight']; old_validate(pre,'PREFLIGHT',dict(spec=saved['content_hash']))
    if (pre != load_json(value['files']['preflight']['path']) or pre['technical_checks_passed'] is not True
            or pre['environment'] != value['anchor']['evidence']['environment']
            or set(pre['projections']) != set(ARMS)
            or any(type(v) is not int or not 1<=v<=1440 for v in pre['projections'].values())
            or value['bank'] != load_json(value['files']['bank']['path'])):
        raise ValueError('Prior resource/target evidence differs')
    old_validate(value['bank'],'BANK')
    if (value['bank']['checkpoint'] != value['teacher']['files']['checkpoint']
            or value['bank']['rows'] != COUNTS['train'] or value['bank']['coordinate'] != 'U100'
            or value['bank']['role'] != 'train'):
        raise ValueError('Accepted bank teacher/population differs')
