"""Matched-seed and local-variant follow-up centered on direct R8."""
from pathlib import Path
import subprocess
import sys

from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from .reconstruction_contracts import artifact,validate,recipe as base_recipe
from .reconstruction import COPIED,BASE
from . import campaign,inputs,reuse,reconstruction_targets as targets

KIND='R8_FOLLOWUP_V2'
AUTHORIZATION='AUTHORIZE_DIRECT_RECONSTRUCTION_R8_FOLLOWUP_V2'
# Replications first, then interleave weights and strategy screens.
TASKS=('C2','R13','C3','R14','R15','R23','R16','R24','R17','R25',
       'R18','R26','R19','R27','R20','R28','R21','R22')
ALL_TASKS=('C0','C1',*TASKS)
SETTING_TASKS=frozenset(TASKS)


def setting(task):
    controls={
        'C0':0,'C1':1,'C2':2,'C3':3,
    }
    if task in controls:
        return dict(family='C',memory='global',source='OFFLINE',consistency=False,
            coefficient=0.,seed_index=controls[task],law='exact_set_reconstruction',
            dot=False,group_size=0,independent=False)
    from .reconstruction_contracts import setting as original
    cfg=original('R8').copy()
    if task in ('R13','R14'): cfg['seed_index']=2 if task=='R13' else 3
    elif task in tuple('R'+str(n) for n in range(15,23)):
        cfg['coefficient']=(.15,.2,.5,.7,1.,2.,5.,10.)[int(task[1:])-15]
    elif task=='R23': cfg['ramp_epochs']=6.
    elif task=='R24': cfg['start_after_epochs']=3.
    elif task=='R25': cfg.update(fade_start=15.,fade_end=25.)
    elif task=='R26': cfg.update(consistency=True,consistency_weight=.1)
    elif task=='R27': cfg['field_weights']=[1.,1.,2.]
    elif task=='R28': cfg['token_layers']=[4,8]
    else: raise NotImplementedError(task)
    return cfg


def execution_key(task):
    # Qualify every maximum weight, including 5/10. Scheduling/seeds share
    # compute; track matching and doubled token memory each need a miniature.
    return tuple((k,str(v)) for k,v in setting(task).items() if k not in
        ('seed_index','ramp_epochs','start_after_epochs','fade_start','fade_end'))


def description(task):
    if task.startswith('C'): return 'CE + U100-w5 control, seed '+task[1:]
    if task in ('R13','R14'): return 'exact R8 weight0.3, seed '+str(setting(task)['seed_index'])
    if task in ('R23','R24','R25','R26','R27','R28'):
        return dict(R23='six-epoch ramp',R24='three-epoch delay',R25='fade epochs15-25',
            R26='whole-jet consistency',R27='track emphasis 1:1:2',R28='block4+8 particle memory')[task]+'; weight0.3 seed0'
    return 'R8 weight'+str(setting(task)['coefficient'])+', seed0'


def storage_bytes(sources,rows):
    paths=len({execution_key(t) for t in TASKS})
    return targets.storage_bytes(sources,rows,fits=len(TASKS)+2*paths)


def recipe():
    return artifact('FOLLOWUP_RECIPE',parents=dict(base_reconstruction=base_recipe()['content_hash']),
        campaign_kind=KIND,tasks=list(TASKS),settings={t:setting(t) for t in TASKS},
        controls=['C2','C3'],replication_pairs=[['R13','C2'],['R14','C3']],
        anchor='R8_tokens_OFFLINE_weight0.3_seed0',epochs=30,batch_size=128,
        embedding_weight=5.,target='same_authenticated_complete_OFFLINE_TRAIN_set',
        decoder=base_recipe()['decoder'],
        variants=dict(token_layers='R28 concatenates live blocks 4 and 8 with shared projection and masks; no depth embeddings',
            field_weights='R27 (kinematic+composition+2*track)/4 in BOTH assignment and matched loss',
            fade='R25 lambda 0.3 through position15, linearly to zero at25; zero thereafter',
            consistency='R26 adds 0.1*aggregate consistency inside reconstruction weight0.3'),
        consistency_definition='existing_R3_whole_jet_consistency',
        schedule='lambda_max*clip((epoch_position-start_after_epochs)/ramp_epochs,0,1)',
        selection='native_validation_AUC_then_CE_then_earliest',quality_gate=False,
        inference='unchanged_native_HLT_no_decoder')


def registration():
    from .reconstruction import registration as base_registration
    value=base_registration()
    value.update(campaign_kind=KIND,recipe=recipe(),tasks=list(ALL_TASKS),global_order=list(TASKS),
        remaining=list(TASKS),execution_version='R5',release_policy='slurm_afterok_measured_independent_v1',
        followup_scope='R8_four_replications_eight_weights_six_strategies',report_minutes=60,
        qualification_policy='every_maximum_weight_and_decoder_loss_path_v2')
    return value


def create(root,group_a_spec,project,commit,quota_gib):
    from .group_b import original
    old=original(group_a_spec); project=Path(project).resolve(); root=Path(root).absolute()
    if type(quota_gib) is not int or quota_gib<1:
        raise ValueError('State personally available output GiB after existing inputs/environment')
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate reconstruction follow-up output root')
    snapshot=campaign.current(project,commit)
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',BASE,commit],check=True,capture_output=True)
    reused={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
    if set(reused)!={'C0','C1'}: raise ValueError('Authenticated seed-0/1 C0 controls required; do not retrain them')
    spec=dict(registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()),group_a_spec=file_ref(Path(group_a_spec).resolve()),
        available_quota_gib=quota_gib,reused=reused,**{k:old[k] for k in COPIED})
    for proof in reused.values(): reuse.check(proof,imported=spec['imported'])
    print('R8 follow-up authenticating complete raw TRAIN reconstruction inventory',flush=True)
    spec['target_inventory']=targets.inventory(spec); spec['storage_budget_bytes']=storage_bytes(spec['target_inventory']['sources'],spec['counts']['train'])
    if quota_gib*2**30<spec['storage_budget_bytes']+2**30:
        raise OSError(f"Follow-up needs {spec['storage_budget_bytes']/2**30:.2f} GiB plus one GiB headroom; no root or jobs created")
    value=artifact('FOLLOWUP_SPEC',parents=dict(source=snapshot['content_hash'],group_a=old['content_hash'],raw=old['raw_hash'],
        inventory=spec['target_inventory']['content_hash'],recipe=recipe()['content_hash']),**spec)
    campaign.storage(value); root.mkdir(); write_immutable_json(root/'campaign_spec.json',value)
    return value


def check(spec,*,check_storage=True):
    from .group_b import original
    validate_file_ref(spec['group_a_spec']); old=original(spec['group_a_spec']['path'])
    inv=spec['target_inventory']; validate(inv,'INVENTORY',dict(raw=spec['raw_hash']))
    validate(spec,'FOLLOWUP_SPEC',dict(source=spec['source_snapshot']['content_hash'],group_a=old['content_hash'],
        raw=old['raw_hash'],inventory=inv['content_hash'],recipe=recipe()['content_hash']))
    if (set(inv['sources'])!={'HLT','OFFLINE'} or inv['maximum']!=max(1,*(s['maximum_count'] for s in inv['sources'].values()))
            or any(s['rows']!=spec['counts']['train'] or s['tokens']<s['rows'] or s['maximum_count']<1 for s in inv['sources'].values())
            or inv['storage_budget_bytes']!=targets.storage_bytes(inv['sources'],spec['counts']['train'])
            or spec['storage_budget_bytes']!=storage_bytes(inv['sources'],spec['counts']['train']) or inv['quality_gate'] is not False
            or any(spec.get(k)!=v for k,v in registration().items()) or any(spec[k]!=old[k] for k in COPIED)
            or spec['reused']!={k:v for k,v in old['reused'].items() if k in ('C0','C1')}
            or spec['available_quota_gib']*2**30<spec['storage_budget_bytes']+2**30
            or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']):
        raise ValueError('Immutable R8 follow-up registration differs')
    for proof in spec['reused'].values(): reuse.check(proof,imported=spec['imported'])
    inputs.raw(spec)
    if check_storage: campaign.storage(spec)
    return load_json(spec['parent_spec']['path'])
