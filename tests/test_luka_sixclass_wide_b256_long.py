"""One longer allocation preserves training and supersedes the split option."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash,canonical_sha256
from hlt_classification.luka_sixclass_wide_b256_oscar import (
    contracts as c,campaign,long_recovery,time_recovery,recovery,artifacts,worker,population)
from test_luka_sixclass_wide_b256_segments import third_recovery,segmented_worker,timing
from test_luka_fullstat_balanced_ladder import threads


def registered(tmp_path,monkeypatch):
    old,previous,segmented=third_recovery(tmp_path,monkeypatch)
    fields={k:v for k,v in segmented.items() if k not in
            ('contract','schema_version','content_hash','parents','time_recovery',*c.FLAGS)}
    fields.update(root=str(tmp_path/'long'),project='/new/long_project',commit='e'*40,
        population_root=str(tmp_path/'long/population'),tasks=c.tasks(5),
        minutes={t:(4320 if t=='train_CONCAT' else 2880) for t in c.tasks(5)},
        long_recovery=long_recovery.evidence(previous))
    return old,previous,segmented,c.artifact('SPEC',version=5,**fields)


def test_v5_long_allocation_preserves_recipe_and_reuses_baselines(tmp_path,monkeypatch):
    old,previous,segmented,s=registered(tmp_path,monkeypatch)
    campaign.check(s,live=False)
    assert s['recipe']==old['recipe'] and s['population']==old['population']
    assert recovery.origin(s,'OFFLINE')==old and recovery.origin(s,'M0HLT')==old
    assert recovery.model_options(s,'CONCAT')==dict(pair_storage=True,pointwise_dgrad=True)
    assert s['long_recovery']['full_estimate']==timing(20)
    assert 2880<timing(20)['conservative_minutes']<s['minutes']['train_CONCAT']
    assert c.epoch_window(s,'train_CONCAT')==(1,20)
    with pytest.raises(ValueError):c.epoch_window(s,'train_CONCAT_part1')
    # User-supplied norm-gpu budgets; no runtime or two-GPU speedup assumption.
    assert 4320<=5760 and 6*4320<=34560
    auth=long_recovery.authenticate
    monkeypatch.setattr(long_recovery,'authenticate',lambda p:auth(p,payloads=False))
    monkeypatch.setattr(artifacts,'completed',lambda spec,stage:dict(receipt={'content_hash':stage}))
    tasks,complete=campaign.remaining(s,False)
    assert tasks==['train_CONCAT','train_UE100','train_DE050','train_DE000']
    assert set(complete)=={'OFFLINE','M0HLT'}
    rows=campaign.commands(s,tasks)
    assert rows[0]['dependencies']==[] and rows[1]['dependencies']==['train_CONCAT']
    assert '--time=4320' in rows[0]['command']
    assert all('--time=2880' in r['command'] for r in rows[1:])
    assert all('--gres=gpu:l40s:1' in r['command'] for r in rows)


@pytest.mark.parametrize('first',['long','segmented'])
def test_alternative_recoveries_share_one_claim_and_preserve_parent(tmp_path,monkeypatch,first):
    old,previous,segmented,s=registered(tmp_path,monkeypatch)
    parent=Path(previous['root'])/'campaign_spec.json';before=parent.read_bytes()
    a,b=(lambda:long_recovery.claim(s)),(lambda:time_recovery.claim(segmented))
    if first=='segmented':a,b=b,a
    a();a()
    with pytest.raises(FileExistsError,match='already claimed'):b()
    assert parent.read_bytes()==before
    with pytest.raises(ValueError,match='superseded'):campaign.remaining(previous,True)


def test_v5_rejects_resource_recipe_and_evidence_drift(tmp_path,monkeypatch):
    _,previous,segmented,s=registered(tmp_path,monkeypatch)
    for task,minutes in [('train_CONCAT',2880),('train_CONCAT',4321),('train_UE100',4320)]:
        bad=deepcopy(s);bad['minutes'][task]=minutes
        with pytest.raises(ValueError):campaign.check(with_content_hash(bad),live=False)
    for key,value in [('tasks',c.tasks(4)),('cpus',12),('recipe',dict(s['recipe'],batch_size=128))]:
        with pytest.raises(ValueError):campaign.check(with_content_hash(dict(s,**{key:value})),live=False)
    changed=deepcopy(s);changed['long_recovery']['policy']['persistent_particle_cache']=True
    changed['long_recovery']=with_content_hash(changed['long_recovery'])
    with pytest.raises(ValueError):campaign.check(with_content_hash(changed),live=False)
    bad=deepcopy(segmented);bad['minutes']['train_CONCAT']=4320
    with pytest.raises(ValueError):campaign.check(with_content_hash(bad),live=False)
    with monkeypatch.context() as m:
        m.setattr(campaign,'terminal',lambda job:(_ for _ in ()).throw(ValueError('live job')))
        with pytest.raises(ValueError,match='live job'):
            long_recovery.authenticate(Path(previous['root'])/'campaign_spec.json',payloads=False)
    write_immutable_json(Path(previous['root'])/'train_CONCAT/epochs/01.json',{'committed':True})
    with pytest.raises(ValueError,match='committed progress'):
        long_recovery.authenticate(Path(previous['root'])/'campaign_spec.json',payloads=False)


def test_v5_registration_changes_only_concat_budget_and_copies_scalar_metadata(tmp_path,monkeypatch):
    old,previous,_,s=registered(tmp_path,monkeypatch)
    dataset=tmp_path/'dataset';dataset.mkdir();manifest=dict(foundation_sha256=c.PARENT_HASH)
    monkeypatch.setattr(c,'TRANSFER_HASH',canonical_sha256(manifest))
    write_immutable_json(dataset/'transfer_manifest.json',manifest)
    write_immutable_json(dataset/'transfer_complete.json',dict(manifest_sha256=c.TRANSFER_HASH,all_files_verified=True))
    old=dict(old,dataset=str(dataset),environment_dir=str(Path(sys.prefix).resolve()))
    pop=Path(old['population_root']);write_immutable_json(pop/'population.json',old['population'])
    (pop/'scalar').write_bytes(b'population metadata only')
    monkeypatch.setattr(long_recovery,'authenticate',lambda p:(old,s['execution_recovery'],s['gradient_recovery'],s['long_recovery']))
    for name in ('locations','source','runtime','storage'):
        monkeypatch.setattr(campaign,name,lambda *a,**kw:{})
    monkeypatch.setattr(campaign,'backend',lambda:old['backend'])
    monkeypatch.setattr(campaign.lineage,'verify',lambda *a:{})
    monkeypatch.setattr(population,'load',lambda *a:None)
    monkeypatch.setattr(campaign,'check',lambda *a,**kw:None)
    p=Path(previous['root'])/'campaign_spec.json';before=p.read_bytes()
    result=campaign.create(dataset,tmp_path/'registered',tmp_path/'source','f'*40,sys.prefix,
        available_gib=18,recover=p,single_concat=True)
    assert result['schema_version']==5 and result['long_recovery']==s['long_recovery']
    assert 'time_recovery' not in result and result['tasks']==c.tasks(5)
    assert result['minutes']==s['minutes'] and result['recipe']==s['recipe']
    assert (Path(result['root'])/'population/scalar').read_bytes()==b'population metadata only'
    assert p.read_bytes()==before and not (Path(result['root'])/'train_OFFLINE').exists()
    with pytest.raises(ValueError,match='SPEC/v3'):
        campaign.create(dataset,tmp_path/'invalid',tmp_path/'source','f'*40,sys.prefix,
            available_gib=18,single_concat=True)


@pytest.mark.parametrize('interrupt',[False,True])
def test_single_allocation_worker_publishes_all_twenty_and_resumes_without_segment(tmp_path,monkeypatch,interrupt):
    s,jobs=segmented_worker(tmp_path,monkeypatch)
    jobs[:]=[j for j in jobs if j['task']!='train_CONCAT_part1']
    s=with_content_hash(dict(s,schema_version=5,contract=c.PREFIX+'_SPEC/v5',tasks=c.tasks(5),
        minutes={t:(4320 if t=='train_CONCAT' else 2880) for t in c.tasks(5)}))
    save=artifacts.save_epoch;epochs=[];before={}
    def record(root,folder,stage,epoch,*args):
        epochs.append(epoch);return save(root,folder,stage,epoch,*args)
    if interrupt:
        def stop(root,folder,stage,epoch,*args):
            if epoch==13:raise RuntimeError('injected interruption')
            return save(root,folder,stage,epoch,*args)
        monkeypatch.setattr(artifacts,'save_epoch',stop)
        with pytest.raises(RuntimeError,match='injected'):worker.run(s,'train_CONCAT',device='cpu')
        jobs.append(dict(task='train_CONCAT',job='904'))
        before={p:p.read_bytes() for p in (tmp_path/'train_CONCAT/epochs').glob('*.json')}
    monkeypatch.setattr(artifacts,'save_epoch',record)
    result=worker.run(s,'train_CONCAT',device='cpu')
    assert epochs==list(range(13 if interrupt else 1,21))
    assert result==artifacts.completed(s,'CONCAT')['receipt']
    assert before=={p:p.read_bytes() for p in before}
    assert not (tmp_path/'train_CONCAT/segments').exists()
    proof=load_json(result['files']['startup']['path'])
    assert proof['epoch_window']==[1,20] and proof['allocation_task']=='train_CONCAT'
    assert proof['timing']['remaining_epochs']==(8 if interrupt else 20)
