"""Keep unaffected evidence under its old identity; never waive TP admission."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_bridge_hydra import qualified,resources,contracts as c,registry as r
from test_luka_embedding_bridge_hydra import execution_spec,resource_diagnostic,synthetic_dataset


def legacy(value):
    v=copy.deepcopy(value)
    cp=v['tp_checkpoint'];cp.pop('execution_math_policy');cp.pop('numerical_settings')
    for check in cp['checks']:
        for row in check['ranks']:row.pop('ordinary_repeat')
        check['ranks']=check['ranks']*2
    cp['world_size']=2
    cp.update(contract='LUKA_EMBEDDING_BRIDGE_HYDRA_TP_CHECKPOINT/v1',schema_version=1)
    v['tp_checkpoint']=with_content_hash(cp)
    v.pop('crop_frontend');v.pop('qualification_reuse')
    v['devices']=v.pop('task_devices')['C0']
    for memory in (v['profiles']['TP']['memory'],v['tp_stress']['memory']):
        memory['ranks']=memory['ranks']*2
    v['tp_stress']=with_content_hash(v['tp_stress'])
    v.update(contract='LUKA_EMBEDDING_BRIDGE_HYDRA_RESOURCES/v2',schema_version=2)
    return with_content_hash(v)


def test_historical_profiles_can_be_inspected_without_accepting_excessive_time(tmp_path):
    s=execution_spec(tmp_path);v=resource_diagnostic(s)
    v['profiles']['TP'].update(step_samples=[10000.]*12,step_seconds=10000.)
    old=legacy(v)
    minutes=resources.profile_minutes(s,old,historical=True)
    assert [t for t in r.FITS if minutes[t]>s['fit_max_minutes']]==['TP']
    with pytest.raises(ValueError,match='contract'):resources.accept(s,old)
    with pytest.raises(RuntimeError,match='TP:'):resources.accept(s,with_content_hash(v))


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    old=execution_spec(tmp_path/'old');root=Path(old['root'])/'setup';root.mkdir(parents=True)
    old.update(recipe=c.recipe(),source_commit=qualified.SOURCE)
    write_immutable_json(Path(old['root'])/'campaign_spec.json',old)
    v=resource_diagnostic(old);v['profiles']['TP'].update(step_samples=[10000.]*12,step_seconds=10000.)
    write_immutable_json(root/'resource_diagnostic.json',legacy(v))
    ds=synthetic_dataset();ds.manifest=dict(content_hash='9'*64)
    selected={role:np.arange(n) for role,n in old['counts'].items()}
    write_immutable_json(root/'mini_population.json',c.artifact('MINI_POPULATION',parents=dict(spec=old['content_hash']),
        indices={k:v.tolist() for k,v in selected.items()},technical=True))
    write_immutable_json(root/'preparation.json',c.artifact('PREPARATION',parents=dict(spec=old['content_hash']),data='9'*64))
    for name in ('parity.json','replay.json','u100_replay.json','replay_predictions.npz'):(root/name).write_bytes(b'fixture')
    for t in r.FITS:
        folder=root/('mini_'+t);folder.mkdir();(folder/'receipt.json').write_bytes(b'original')
    monkeypatch.setattr(qualified,'source_spec',lambda path:old)
    monkeypatch.setattr(qualified.recovery,'terminal_jobs',lambda *a,**kw:'165857')
    monkeypatch.setattr(qualified.qualification,'check_replay',lambda s,**kw:None)
    monkeypatch.setattr(qualified.inputs,'Dataset',lambda s:ds)
    monkeypatch.setattr(resources,'selected_indices',lambda *a,**kw:selected)
    monkeypatch.setattr(resources,'technical_target',lambda *a:SimpleNamespace(parents=dict(u100='b'*64)))
    monkeypatch.setattr(qualified.training,'validate_fit',lambda *a,**kw:(dict(world_size=2,
        parents=dict(spec=old['content_hash'],recipe=c.recipe()['content_hash'],u100='b'*64)),None))
    return old,root


def test_historical_fit_location_and_immutability(evidence,tmp_path):
    old,root=evidence
    reuse=qualified.inspect(Path(old['root'])/'campaign_spec.json')
    new=dict(old,root=str(tmp_path/'new'),content_hash='7'*64,qualification_reuse=reuse)
    assert qualified.check(reuse,new,full=True)==old
    assert len(reuse['reused'])==29 and reuse['remeasure']==['TP']
    assert qualified.miniature(new,'P5')==(root/'mini_P5',old)
    assert qualified.miniature(new,'TP')==(Path(new['root'])/'setup/mini_TP',new)
    assert not Path(new['root']).exists()
    (root/'mini_P5/receipt.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError):qualified.check(reuse,new,full=True)


@pytest.mark.parametrize('bad',['accepted','other_failure','changed_targets','lineage'])
def test_recovery_rejects_different_failures_and_lineage(evidence,tmp_path,bad,monkeypatch):
    old,root=evidence
    if bad=='accepted':(root/'acceptance.json').write_bytes(b'unexpected')
    elif bad=='other_failure':
        v=load_json(root/'resource_diagnostic.json');v['profiles']['TR'].update(step_samples=[10000.]*12,step_seconds=10000.)
        (root/'resource_diagnostic.json').write_text(__import__('json').dumps(with_content_hash(v)))
    elif bad=='lineage':
        monkeypatch.setattr(qualified.training,'validate_fit',lambda *a,**kw:(dict(world_size=2,parents={}),None))
    else:
        reuse=qualified.inspect(Path(old['root'])/'campaign_spec.json')
        new=dict(old,embedding_hash='changed')
        with pytest.raises(ValueError,match='different scientific'):qualified.check(reuse,new)
        return
    with pytest.raises(ValueError):qualified.inspect(Path(old['root'])/'campaign_spec.json')


def test_recovered_profile_is_conservative_and_cannot_be_edited(evidence,tmp_path,monkeypatch):
    old,root=evidence;reuse=qualified.inspect(Path(old['root'])/'campaign_spec.json')
    new=dict(old,root=str(tmp_path/'new'),content_hash='7'*64,qualification_reuse=reuse)
    value=resource_diagnostic(new);value['qualification_reuse']=reuse
    historical=load_json(root/'resource_diagnostic.json')
    for task in qualified.REUSED:value['profiles'][task]=copy.deepcopy(historical['profiles'][task])
    assert set(resources.accept(new,with_content_hash(value))['minutes'])==set(r.FITS)
    value['profiles']['C0']['startup_seconds']+=.1
    with pytest.raises(ValueError,match='profiles changed'):resources.accept(new,with_content_hash(value))


def test_single_rank_parity_cannot_claim_new_two_gpu_validation(evidence,tmp_path):
    import json
    from hlt_classification.luka_subclass_ddp_hydra.resources import TOLERANCES,PROBE_TOLERANCES
    from hlt_classification.luka_embedding_bridge_hydra import qualification
    old,root=evidence
    checks={k:dict.fromkeys(TOLERANCES,0.)|dict(passed=True) for k in qualification.CHECKS}
    probes={k:dict.fromkeys(PROBE_TOLERANCES,0.)|dict(passed=True,rng_equal=True) for k in ('C0','TR','TP','R8','P7')}
    measured=dict(checks=checks,probe_neutrality=probes,tolerances=TOLERANCES,probe_tolerances=PROBE_TOLERANCES,
        factory_fp32=True,dropout_disabled=True,baseline=dict(model_update_equal=True,adamw_equal=True))
    historical=c.artifact('PARITY',parents=dict(spec=old['content_hash']),**measured)
    historical.update(contract='LUKA_EMBEDDING_BRIDGE_HYDRA_PARITY/v1',schema_version=1)
    (root/'parity.json').write_text(json.dumps(with_content_hash(historical)))
    reuse=qualified.inspect(Path(old['root'])/'campaign_spec.json')
    new=dict(old,root=str(tmp_path/'new'),content_hash='7'*64,qualification_reuse=reuse)
    proof=qualified.distributed_evidence(new)
    assert proof['world_size']==2 and proof['tasks']==list(qualified.REUSED) and 'TP' not in proof['tasks']
    assert proof['source_spec_hash']==old['content_hash']
    value=c.artifact('PARITY',parents=dict(spec=new['content_hash']),**measured,reference_world_size=1,world_size=1,
        comparison='native_single_rank_repeat',historical_distributed=proof)
    qualification.check_parity(value,new)
    with pytest.raises(ValueError,match='ownership'):
        qualification.check_parity(with_content_hash(dict(value,world_size=2)),new)
    with pytest.raises(ValueError,match='ownership'):
        qualification.check_parity(with_content_hash(dict(value,historical_distributed=dict(proof,tasks=list(r.FITS)))),new)
    (root/'parity.json').write_text('corrupt')
    with pytest.raises(ValueError):qualification.check_parity(value,new)


def test_reused_technical_bank_keeps_original_owner_in_new_parity_run(tmp_path,monkeypatch):
    import torch
    from test_luka_embedding_bridge_hydra import small
    from hlt_classification.luka_embedding_bridge_hydra import banks,training,distributed as d
    small(monkeypatch);old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        ds=synthetic_dataset();old=ds.spec;old['root']=str(tmp_path/'old')
        ctx=d.Context(0,1,torch.device('cpu'));folder=Path(old['root'])/'setup/mini_TR'
        model,report=training.fit(folder,old,'TR',ds,banks.Targets(old,'TR',ds),ctx,technical=True,val_diagnostics=False)
        checkpoint=file_ref(folder/'selected_full.pt')
        banks.export(folder/'bank',old,'TR',model,ds,ctx,checkpoint,technical=True)
        path=Path(old['root'])/'campaign_spec.json';write_immutable_json(path,old)
        new=dict(old,root=str(tmp_path/'new'),content_hash='7'*64,qualification_reuse=dict(source_spec=file_ref(path)))
        monkeypatch.setattr(qualified,'source_spec',lambda path:old)
        target=resources.technical_target(new,'R5',ds,Path(new['root'])/'setup')
        assert target.parent.manifest['parents']['spec']==old['content_hash']
        assert target.parent.manifest['technical'] is True
        m,w,opt=d.build(new,'R5',ctx)
        values,_=d.step(m,w,opt,'R5',ds,target,np.arange(15),3,0,3.,ctx)
        assert np.isfinite(list(values.values())).all()
        with pytest.raises(PermissionError):banks.Bank(folder/'bank',old,'TR',ds.view('train','D000'),checkpoint)
    finally:torch.set_num_threads(old_threads)


@pytest.mark.parametrize('too_slow',[False,True])
def test_regional_setup_measures_tp_first_and_skips_29_reused_fits(tmp_path,monkeypatch,too_slow):
    """Exercise measure orchestration and its real acceptance, mock only GPU work."""
    s=execution_spec(tmp_path);(tmp_path/'setup').mkdir()
    fixture=resource_diagnostic(s);ds=synthetic_dataset()
    selected={k:np.arange(n) for k,n in ds.spec['counts'].items()}
    class Context:
        world=1;rank=0;device=SimpleNamespace(type='cuda')
        def main(self,fn):return fn()
    monkeypatch.setattr(resources.qualified,'profiles',lambda *a:{t:fixture['profiles'][t] for t in qualified.REUSED})
    monkeypatch.setattr(resources.qualified,'profile_devices',lambda *a:fixture['task_devices']['C0'])
    monkeypatch.setattr(resources.regional_inputs,'qualify',lambda *a:fixture['crop_frontend'])
    monkeypatch.setattr(resources.qualification,'tp_checkpoint',lambda *a:fixture['tp_checkpoint'])
    monkeypatch.setattr(resources,'tp_stress',lambda *a:fixture['tp_stress'])
    monkeypatch.setattr(resources.banks,'prepare_u100',lambda *a:None)
    monkeypatch.setattr(resources.qualification,'replay',lambda *a:None)
    monkeypatch.setattr(resources,'selected_indices',lambda *a,**kw:selected)
    monkeypatch.setattr(resources,'technical_target',lambda *a:None)
    write_immutable_json(tmp_path/'setup/preparation.json',c.artifact('PREPARATION',parents=dict(spec=s['content_hash']),seconds=1.,host_peak_bytes=2**30))
    fitted=[]
    def fit(folder,spec,task,*a,**kw):
        fitted.append(task);folder.mkdir();(folder/'selected_full.pt').write_bytes(b'miniature')
        return object(),dict(fit_memory=fixture['profiles'][task]['memory'],history=[])
    monkeypatch.setattr(resources.training,'fit',fit)
    monkeypatch.setattr(resources.banks,'export',lambda *a,**kw:None)
    monkeypatch.setattr(resources.d,'timed',lambda fn,ctx:(10. if too_slow else .2,fn()))
    monkeypatch.setattr(resources.d,'release_memory',lambda *a:None)
    monkeypatch.setattr(resources.torch.cuda,'reset_peak_memory_stats',lambda *a:None)
    monkeypatch.setattr(resources.d,'build',lambda *a:(None,None,None))
    monkeypatch.setattr(resources.d,'step',lambda *a,**kw:None)
    def evaluate(*a):
        assert not too_slow, 'Full validation ran despite impossible training bound'
    monkeypatch.setattr(resources.training,'evaluate',evaluate)
    monkeypatch.setattr(resources.training,'target_diagnostics',lambda *a:None)
    monkeypatch.setattr(resources.d,'memory',lambda *a:copy.deepcopy(fixture['profiles']['TP']['memory']))
    monkeypatch.setattr(resources.qualification,'qualify',lambda *a:None)
    monkeypatch.setattr(resources,'installed_environment',lambda:s['reference']['environment'])
    ds.physical=lambda role:{src+'_offsets':np.arange(len(ds.view(role,'D000'))+1) for src in ('HLT','OFFLINE')}
    # Production wide miniature has >128 rows. This fixture only needs a shape-
    # compatible iterator; the measured GPU operations above are mocked.
    # Supply a full-size technical view to the timing iterator.
    monkeypatch.setattr(resources.inputs,'SubsetDataset',lambda *a:SimpleNamespace(view=lambda *a:range(256)))
    # Preserve representative global-row selection for stress indexing.
    if too_slow:
        with pytest.raises(RuntimeError,match='training alone'):resources.measure(s,ds,Context(),fixture['devices'],1.)
        assert fitted==['TP']
        assert (tmp_path/'setup/tp_training_timing.json').is_file()
        assert not (tmp_path/'setup/acceptance.json').exists()
        return
    result=resources.measure(s,ds,Context(),fixture['devices'],1.)
    assert fitted==['TP'] and set(result['minutes'])==set(r.FITS)
    assert (tmp_path/'setup/resource_TP.json').is_file()
