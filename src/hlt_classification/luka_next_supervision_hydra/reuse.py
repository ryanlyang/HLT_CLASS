"""Authenticate completed controls in their own pinned worktree, never by pathname."""
from pathlib import Path
import json
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from .contracts import artifact,setting,seed,validate

MODULES={'C0':('luka_particle_teacher_hydra','control_s0'),
         'C1':('luka_particle_teacher_hydra','control_s1'),
         'J5':('luka_u100_contrastive_hydra','mix_l0p3_s0')}


def inspect(row,path,imported,prepared,embedding_hash,key_hash=None):
    if row not in MODULES: raise ValueError('Only matched C0/C1/J5 reuse is registered')
    module,task=MODULES[row]; path=Path(path).resolve(strict=True); old=load_json(path)
    code='''import importlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'src'))
from hlt_classification.data.cache_contracts import load_json
s=load_json(sys.argv[1]); name=sys.argv[3]; task=sys.argv[4]
c=importlib.import_module('hlt_classification.'+name+'.campaign')
w=importlib.import_module('hlt_classification.'+name+'.worker')
c.check(s,check_storage=False); receipt=w.receipt(s,task)
if name=='luka_particle_teacher_hydra':
 t=importlib.import_module('hlt_classification.'+name+'.training'); r,a=t.validate_fit(Path(s['root'])/task/'fit',s,task)
else:
 t=importlib.import_module('hlt_classification.luka_u100_contrastive.training'); r,a=t.validate_fit(Path(s['root'])/task/'fit',s,task)
print('GROUP_A_REUSE='+json.dumps(dict(receipt=receipt['content_hash'],report=r),sort_keys=True,allow_nan=False))
'''
    env=clean_environment(); env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(old['project_dir'])/'src'))
    run=subprocess.run([sys.executable,'-B','-s','-c',code,str(path),old['project_dir'],module,task],
        cwd=old['project_dir'],env=env,capture_output=True,text=True)
    if run.returncode: raise RuntimeError('Original-source reference authentication failed:\n'+run.stdout+run.stderr)
    lines=[x.removeprefix('GROUP_A_REUSE=') for x in run.stdout.splitlines() if x.startswith('GROUP_A_REUSE=')]
    if len(lines)!=1: raise ValueError('Missing unique reuse proof')
    proof=json.loads(lines[0]); r=proof['report']; folder=Path(old['root'])/task/'fit'; cfg=setting(row)
    # Old artifact field names differ; scientific seed/recipe/population must not.
    init=r.get('initialization_seed',r.get('resolved_seeds',{}).get('initialization'))
    sampler=r.get('sampler_seed',r.get('resolved_seeds',{}).get('sampler'))
    if (old['counts']!=dict(train=299234,validation=74895) or old['imported']!=imported
            or old['anchor']['evidence']['prepared']!=prepared or r.get('technical',False)
            or init!=seed(imported,cfg,'initialization_seed') or sampler!=seed(imported,cfg,'sampler_seed')
            or len(r['validation_history'])!=30 or r.get('epochs',30)!=30):
        raise ValueError('Reference is not the same population, seeds and full budget')
    if row[0]=='C' and old['embedding_hash']!=embedding_hash: raise ValueError('Control U100 bank differs')
    if row=='J5' and r['parents']['u100']!=embedding_hash: raise ValueError('Dot control U100 bank differs')
    if row=='J5' and (key_hash is None or r['parents']['offline']!=key_hash): raise ValueError('Dot control OFFLINE key bank differs')
    files=[file_ref(path),file_ref(Path(old['root'])/task/'receipt.json')]+[file_ref(folder/n) for n in (
        'receipt.json','training_report.json','validation_predictions.npz','selected.pt')]
    return artifact('REUSE',parents=dict(origin=old['content_hash'],receipt=proof['receipt'],report=r['content_hash']),
        row=row,module=module,task=task,files=files,report=r,original_spec=str(path),source_mode='pinned_original_worktree',
        initialization_seed=init,sampler_seed=sampler,embedding_hash=embedding_hash,key_hash=key_hash,prepared=prepared)


def transferred_dot(spec):
    """Reuse the original fit already authenticated/exported on SPORC; no weights copied."""
    import numpy as np
    from hlt_classification.luka_u100_contrastive_hydra.transfer import Delta,COMPLETED_FILES
    from hlt_classification.luka_u100_contrastive.training import validate_report
    from hlt_classification.luka_u100_contrastive.contracts import recipe as old_recipe
    from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
    from hlt_classification.luka_u100_pair_hydra.portability import same_metrics
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.data.cache_contracts import load_npz_arrays
    d=Delta(spec['delta_root'],spec['delta_hash']); task='mix_l0p3_s0'
    if task not in d.manifest['completed']: return None
    old=d.manifest['origin_spec']; r=load_json(d.path(f'completed/{task}/report.json'))
    validate_report(r,old,task)
    if (old['imported']!=spec['imported'] or r['parents']!=dict(spec=old['content_hash'],recipe=old_recipe()['content_hash'],
            prepared=spec['anchor']['evidence']['prepared'],u100=spec['embedding_hash'],offline=spec['key_hash'])
            or r['resolved_seeds']['initialization']!=seed(spec['imported'],setting('J5'),'initialization_seed')
            or r['resolved_seeds']['sampler']!=seed(spec['imported'],setting('J5'),'sampler_seed')):
        raise ValueError('Transferred dot fit recipe/seeds/targets differ')
    a=load_npz_arrays(d.path(f'completed/{task}/predictions.npz')); validate_predictions(a,old['counts']['validation'])
    b=Bundle(spec['bundle_root']); reference=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if (any(not np.array_equal(a[k],reference[k]) for k in ('identities','labels'))
            or not same_metrics(evaluate_probabilities(a['labels'],a['probabilities']),r['validation'])):
        raise ValueError('Transferred dot predictions differ')
    files=[file_ref(d.root/'manifest.json')]+[file_ref(d.path(f'completed/{task}/{n}')) for n in COMPLETED_FILES]
    binding={k:spec[k] for k in ('delta_root','delta_hash','bundle_root','imported','anchor','embedding_hash','key_hash')}
    proof=d.manifest['original_replay']['completed'][task]
    return artifact('REUSE',parents=dict(origin=old['content_hash'],transfer=d.manifest['content_hash'],report=r['content_hash']),
        row='J5',task=task,source_mode='transferred_original_SPORC_result',files=files,report=r,binding=binding,
        job_id=proof['job_id'],prepared=spec['anchor']['evidence']['prepared'],
        embedding_hash=spec['embedding_hash'],key_hash=spec['key_hash'],selected_checkpoint_transferred=False)


def check(proof,*,replay=False,imported=None):
    validate(proof,'REUSE')
    if proof.get('source_mode')=='transferred_original_SPORC_result':
        for r in proof['files']: validate_file_ref(r)
        if transferred_dot(proof['binding'])!=proof: raise ValueError('Transferred dot result proof changed')
        return Path(next(r['path'] for r in proof['files'] if Path(r['path']).name=='predictions.npz'))
    if proof.get('source_mode')!='pinned_original_worktree': raise ValueError('Unknown reuse capability')
    if proof['row'] not in MODULES or (proof['module'],proof['task'])!=MODULES[proof['row']]: raise ValueError('Wrong reused task')
    for r in proof['files']: validate_file_ref(r)
    from hlt_classification.provenance import validate_source_snapshot
    old=load_json(proof['original_spec'])
    validate_source_snapshot(old['source_snapshot'],repository=old['project_dir'],require_clean=True)
    report_path=next(r['path'] for r in proof['files'] if Path(r['path']).name=='training_report.json')
    if (load_json(report_path)!=proof['report'] or proof['parents']['report']!=proof['report']['content_hash']
            or proof['parents']['origin']!=old['content_hash'] or proof['prepared']!=old['anchor']['evidence']['prepared']):
        raise ValueError('Reused report/spec parent differs')
    if replay and inspect(proof['row'],proof['original_spec'],imported,proof['prepared'],proof['embedding_hash'],proof['key_hash'])!=proof:
        raise ValueError('Reused reference proof changed')
    return Path(next(r['path'] for r in proof['files'] if Path(r['path']).name=='validation_predictions.npz'))
