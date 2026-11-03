"""Relocation, fixed science, no student dependencies and exact submission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import numpy as np
import pytest
from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, canonical_sha256)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_contrastive import contracts as old, targets
from hlt_classification.luka_u100_contrastive_serial.contracts import artifact as parent_artifact
from hlt_classification.luka_u100_contrastive_hydra import (
    contracts as c, transfer, campaign, inputs, submission, worker)
from test_luka_u100_contrastive import imported, threads
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_u100_embedding import bank_fixture


def spec(root):
    reused = ['u100_w5_s0','mix_l0p3_s0','random_l0p3_s0']
    return dict(campaign.registration(),root=str(root),project_dir='/project',content_hash='e'*64,
        reused=reused,remaining=[a for a in c.ARMS if a not in reused],imported=imported())


def test_science_unchanged_and_every_student_is_independent(tmp_path):
    s = spec(tmp_path); setup = submission.render(s,'setup'); envelope = submission.render(s,'envelope')
    assert s['recipe'] == old.recipe() and s['stopping_policy'] == 'original_full30'
    assert s['max_concurrent_fits'] is None and len(envelope['commands']) == 18
    assert [r['task_id'] for r in envelope['commands'][:-1]] == s['remaining']
    for row in envelope['commands'][:-1]:
        assert row['dependencies'] == []
        assert not any(x.startswith(('--dependency','--array')) for x in row['command'])
        assert '--gres=gpu:nvidia_l40:1' in row['command']
    assert envelope['commands'][-1]['dependencies'] == s['remaining']
    assert len(setup['commands']) == 1
    measured = submission.render(s,'science',acceptance=old.artifact('ACCEPTANCE',minutes={a:120 for a in c.ARMS}))
    assert submission.within_envelope(envelope,measured)
    measured['commands'][0]['command'].append('--dependency=afterany:999')
    assert not submission.within_envelope(envelope,measured)


def test_exact_dispatch_queues_all_at_once_and_rejects_duplicate_or_failed_setup(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls = []
    def run(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(1000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(dag.subprocess,'run',run)
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(submission,'check_site',lambda *a:None)
    monkeypatch.setattr(submission.resources,'accepted',lambda s:old.artifact('ACCEPTANCE',minutes={a:120 for a in c.ARMS}))
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accounting',lambda j:['COMPLETED','0:0'])
    s = spec(tmp_path); review = submission.review(s)
    submission.dispatch(s,review['content_hash'],c.AUTHORIZATION)
    assert len(calls) == len(s['remaining'])+2
    assert all(not any(a.startswith('--dependency=') for a in cmd) for cmd in calls[:-1])
    assert len(submission.accepted_jobs(s,'science')) == len(s['remaining'])+1
    with pytest.raises(FileExistsError): submission.dispatch(s,review['content_hash'],c.AUTHORIZATION)
    failed = tmp_path/'failed'; failed.mkdir(); s2 = spec(failed); r2 = submission.review(s2)
    monkeypatch.setattr(submission,'accounting',lambda j:['FAILED','1:0'])
    with pytest.raises(RuntimeError,match='no students'): submission.dispatch(s2,r2['content_hash'],c.AUTHORIZATION)
    assert len(calls) == len(s['remaining'])+3 and not (failed/'submission_science').exists()


@pytest.fixture
def exported(tmp_path, monkeypatch):
    train, bank, _ = bank_fixture(tmp_path)
    origin = tmp_path/'original'; origin.mkdir(); setup = origin/'setup'; setup.mkdir()
    teacher = origin/'teacher.pt'; teacher.write_bytes(b'fixture teacher')
    source = dict(contract='hlt_classification_source_snapshot_v1',schema_version=1,
        git_commit=c.PARENT_COMMIT,git_tree='a'*40,tracked_files_sha256='b'*64,
        tracked_file_count=1,worktree_clean=True)
    source['source_snapshot_sha256'] = canonical_sha256({k:source[k] for k in ('git_commit','git_tree','tracked_files_sha256')})
    source = with_content_hash(source)
    imp = dict(imported(),content_hash='c'*64)
    anchor = dict(evidence=dict(prepared=train.foundation_sha256,models=dict(OFFLINE=dict(checkpoint=file_ref(teacher)))))
    owner = with_content_hash(dict(old.artifact('SPEC'),contract='LUKA_U100_CONTRASTIVE_SPEC/v1',schema_version=1,
        imported=imp,anchor=anchor,counts=dict(train=len(train),validation=22),recipe=old.recipe()))
    owner_path = origin/'owner.json'; write_immutable_json(owner_path,owner)
    origin_spec = old.artifact('SPEC',root=str(origin),project_dir=str(origin/'code'),source_commit=c.PARENT_COMMIT,
        source_snapshot=source,arms=old.ARMS,counts=owner['counts'],recipe=old.recipe(),imported=imp,anchor=anchor,
        setup_reuse=dict(origin_spec=file_ref(owner_path),parents=dict(origin=owner['content_hash'])),
        recovery_anchors=dict(M0HLT=.9,OFFLINE=.95))
    path = origin/'campaign_spec.json'; write_immutable_json(path,origin_spec)
    key = targets.publish(setup/'OFFLINE',np.random.default_rng(19).normal(size=(len(train),128)).astype(np.float32),
        train,file_ref(teacher),spec_hash=owner['content_hash'],imported_hash=imp['content_hash'])
    u100 = origin/'u100.json'; write_immutable_json(u100,bank.manifest)
    write_immutable_json(setup/'bundle.json',old.artifact('BUNDLE',offline_manifest=file_ref(setup/'OFFLINE/manifest.json')))
    folder = origin/'u100_w5_s0'; fit = folder/'fit'; fit.mkdir(parents=True)
    for name in ('training_report.json','receipt.json'):
        write_immutable_json(fit/name,old.artifact('FIXTURE'))
    (fit/'validation_predictions.npz').write_bytes(b'fixture predictions')
    write_immutable_json(folder/'receipt.json',old.artifact('TASK'))
    refs = [file_ref(folder/'receipt.json')]+[file_ref(p) for p in sorted(fit.iterdir())]
    jobs = {a:str(20000+i) for i,a in enumerate(old.ARMS)}; jobs['report']='21000'
    proof = parent_artifact('PARENT',parents=dict(spec=origin_spec['content_hash']),spec=file_ref(path),
        jobs=jobs,setup_jobs=dict(setup='19000'),
        completed=dict(u100_w5_s0=dict(job_id=jobs['u100_w5_s0'],files=refs)),
        setup_files=[file_ref(setup/'OFFLINE/manifest.json'),file_ref(setup/'OFFLINE/keys.npz'),file_ref(u100)])
    states = {j:['PENDING','0:0'] for j in jobs.values()}
    states.update({jobs['u100_w5_s0']:['COMPLETED','0:0'],'19000':['COMPLETED','0:0']})
    monkeypatch.setattr(transfer,'COUNTS',owner['counts'])
    monkeypatch.setattr(transfer,'validate_source_snapshot',lambda *a,**k:None)
    monkeypatch.setattr(transfer,'inspect_parent',lambda *a:proof)
    monkeypatch.setattr(transfer,'accounting',lambda *a:states)
    out = transfer.export(path,tmp_path/'export')
    d = transfer.Delta(out['directory'],out['manifest_hash'])
    return SimpleNamespace(d=d,out=out,train=train,bank=bank,key=key,owner=owner,old=origin_spec,
        original=origin,path=path,states=states,proof=proof,u100=u100)


def test_delta_minimal_payload_and_relocated_keybank_without_original_paths(exported,tmp_path,monkeypatch):
    f = exported; moved = tmp_path/'brown'; shutil.copytree(f.d.root,moved)
    d = transfer.Delta(moved,f.out['manifest_hash']); d.verify_all()
    assert set(d.rows) == {'keys/manifest.json','keys/keys.npz'} | {
        'completed/u100_w5_s0/'+name for name in transfer.COMPLETED_FILES}
    assert all(not name.endswith('.pt') and not name.startswith('prepared/') for name in d.rows)
    renamed = tmp_path/'original-unavailable'; f.original.rename(renamed)
    s = dict(delta_root=str(moved),delta_hash=f.out['manifest_hash'])
    bank = inputs.key_bank(s,f.train)
    np.testing.assert_array_equal(bank.identities,f.train.identities)
    assert bank.manifest['parents']['spec'] == f.owner['content_hash']
    assert bank.manifest == f.key
    with pytest.raises(PermissionError): d.path('teacher.pt')
    with pytest.raises(ValueError): d.path('../keys/keys.npz')
    with (moved/'keys/keys.npz').open('ab') as h: h.write(b'changed')
    with pytest.raises(ValueError): inputs.key_bank(s,f.train)


def test_transfer_refuses_missing_completion_receipt_or_failed_accounting(exported):
    f = exported; value = deepcopy(f.d.manifest)
    job = value['original_replay']['jobs']['mix_l1_s0']
    value['accounting'][job] = ['COMPLETED','0:0']
    with pytest.raises(ValueError,match='no authenticated receipt'): transfer.validate_manifest(with_content_hash(value))
    value = deepcopy(f.d.manifest)
    value['accounting'][value['original_replay']['jobs']['u100_w5_s0']] = ['FAILED','1:0']
    with pytest.raises(ValueError,match='not a successful'): transfer.validate_manifest(with_content_hash(value))
    with pytest.raises(ValueError,match='verified export hash'): transfer.Delta(f.d.root,'0'*64)


def test_metadata_bind_reuses_exact_existing_brown_inputs(exported,monkeypatch):
    f = exported
    bundle = SimpleNamespace(json=lambda name: f.old if name=='origin/matched_spec.json' else f.bank.manifest,
        rows={'targets/u100/manifest.json':file_ref(f.u100)})
    monkeypatch.setattr(inputs,'Bundle',lambda _:bundle)
    s = dict(bundle_root='/brown',imported=f.old['imported'],anchor=f.old['anchor'])
    inputs.bind(s,f.d)
    bad = deepcopy(s); bad['imported']['content_hash']='x'*64
    with pytest.raises(ValueError,match='different origin'): inputs.bind(bad,f.d)


def test_setup_imports_targets_and_runs_original_resource_kernel(exported,tmp_path,monkeypatch):
    f = exported; root = tmp_path/'run'; root.mkdir()
    s = dict(spec(root),delta_root=str(f.d.root),delta_hash=f.out['manifest_hash'],bundle_root='/brown')
    monkeypatch.setattr(inputs,'load_view',lambda *a:f.train)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:f.bank)
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda _:f.u100))
    monkeypatch.setattr(worker.portability,'setup_record',lambda *a:{'fixture':True})
    seen=[]
    monkeypatch.setattr(worker.resources,'measure',lambda *a,**k:seen.append((a,k)))
    worker.setup(s,device='cpu')
    assert len(seen)==1 and seen[0][0][3] is f.bank
    assert seen[0][0][4].manifest==f.key
    record=load_json(root/'setup/inputs.json')
    assert record['target_exported'] is False and record['teachers_retrained'] is False
    assert not (root/'setup/acceptance.json').exists()


def test_scripts_use_existing_venv_and_no_scheduler_cancellation():
    for path in ('scripts/queue_luka_u100_contrastive_hydra.sh','sbatch/run_luka_u100_contrastive_hydra.sh'):
        text=Path(path).read_text()
        assert 'PYTHONNOUSERSITE=1' in text and '${ENV_DIR}/lib' in text
        assert 'scancel' not in text and 'scontrol hold' not in text
    assert worker.training is __import__('hlt_classification.luka_u100_contrastive.training',fromlist=['training'])


def test_create_and_check_bind_existing_bundle_delta_and_original_full30(exported,tmp_path,monkeypatch):
    f=exported; bundle_root=tmp_path/'bundle'; bundle_root.mkdir()
    project=tmp_path/'project'; project.mkdir()
    diagnostic=tmp_path/'precision/spec.json'; diagnostic.parent.mkdir()
    pair_origin=dict(reference=dict(environment={'fixture':True}))
    docs={'origin/campaign_spec.json':pair_origin,'origin/matched_spec.json':f.old,
          'targets/u100/manifest.json':f.bank.manifest}
    b=SimpleNamespace(root=bundle_root,json=docs.__getitem__,rows={'targets/u100/manifest.json':file_ref(f.u100)})
    monkeypatch.setattr(inputs,'Bundle',lambda _:b)
    monkeypatch.setattr(campaign,'COUNTS',f.old['counts'])
    monkeypatch.setattr(campaign,'current',lambda *a:f.old['source_snapshot'])
    monkeypatch.setattr(campaign.site,'environment',lambda *a:None)
    evidence=old.artifact('PORTABILITY_FIXTURE')
    monkeypatch.setattr(campaign.portability,'authenticate',lambda *a:evidence)
    monkeypatch.setattr(campaign.portability,'validate_evidence',lambda *a:None)
    monkeypatch.setattr(campaign,'storage',lambda *a:None)
    args=(tmp_path/'fresh',bundle_root,f.d.root,f.out['manifest_hash'],diagnostic,project,c.BASE,8)
    s=campaign.create(*args)
    assert campaign.check(s)==s
    assert s['reused']==['u100_w5_s0'] and len(s['remaining'])==19
    assert s['recipe']==old.recipe() and s['max_concurrent_fits'] is None
    assert Path(s['environment_dir'])==Path(sys.prefix).resolve()
    with pytest.raises(ValueError,match='fresh separate'): campaign.create(*args)
    bad=with_content_hash(dict(s,remaining=s['remaining'][1:]))
    with pytest.raises(ValueError,match='Immutable'): campaign.check(bad)
    bad=with_content_hash(dict(s,student_dependencies='lanes',max_concurrent_fits=2))
    with pytest.raises(ValueError,match='Immutable'): campaign.check(bad)


def test_completed_setup_authenticates_receipt_but_zero_worker_does_not_open_keys(exported,tmp_path,monkeypatch):
    f=exported; root=tmp_path/'run'; root.mkdir()
    s=dict(spec(root),delta_root=str(f.d.root),delta_hash=f.out['manifest_hash'],bundle_root='/brown')
    monkeypatch.setattr(inputs,'load_view',lambda *a:f.train)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:f.bank)
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=lambda _:f.u100))
    monkeypatch.setattr(worker.portability,'setup_record',lambda *a:{'fixture':True})
    monkeypatch.setattr(worker.resources,'measure',lambda *a,**k:None)
    monkeypatch.setattr(worker.portability,'validate_setup',lambda *a:None)
    monkeypatch.setattr(worker.resources,'accepted',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{'setup':'131234'})
    worker.setup(s,device='cpu'); folder=root/'setup'
    for name in ('capacity.json','baseline_parity.json','resource_diagnostic.json','acceptance.json'):
        write_immutable_json(folder/name,{'fixture':True})
    for family in ('none','random','mix','kinmix','hard','cross'):
        (folder/('mini_'+family)).mkdir()
        write_immutable_json(folder/('mini_'+family)/'receipt.json',{'fixture':True})
    task=c.artifact('TASK',parents=dict(spec=s['content_hash']),task='setup',job_id='131234',
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',task)
    assert worker.completed_setup(s)==task
    (f.d.root/'keys/keys.npz').write_bytes(b'corrupt after setup')
    assert worker.completed_setup(s)==task  # no OFFLINE payload access for lambda zero
    with pytest.raises(ValueError): inputs.key_bank(s,f.train)
    (folder/'capacity.json').write_text('{}')
    with pytest.raises(ValueError): worker.completed_setup(s)


def test_full30_worker_native_fit_and_mixed_site_results(exported,tmp_path,monkeypatch):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    tiny(monkeypatch); f=exported; root=tmp_path/'science'; root.mkdir()
    arm='u100_w5_s1'; val=cache('validation','D000')
    s=dict(spec(root),counts=f.old['counts'],original50k_rows=len(val),reused=['u100_w5_s0'],
        remaining=[a for a in c.ARMS if a!='u100_w5_s0'],anchor=f.old['anchor'])
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(worker,'require_backend',lambda:None)
    monkeypatch.setattr(worker.site,'allocation',lambda *a:'131999')
    monkeypatch.setattr(worker,'completed_setup',lambda *a:None)
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a:{arm:'131999'})
    monkeypatch.setattr(worker,'installed_environment',lambda:{})
    monkeypatch.setattr(worker,'gpu_identity',lambda:{})
    monkeypatch.setattr(worker.resources,'accepted',lambda *a:dict(environment={},gpu={}))
    monkeypatch.setattr(inputs,'load_view',lambda _,role,*a:f.train if role=='train' else val)
    monkeypatch.setattr(inputs,'embedding_bank',lambda *a:f.bank)
    monkeypatch.setattr(inputs,'key_bank',lambda *a:pytest.fail('Zero control opened OFFLINE keys'))
    fit=worker.training.fit
    def cpu_fit(*args,**kwargs):
        kwargs['device']='cpu'
        return fit(*args,**kwargs)
    monkeypatch.setattr(worker.training,'fit',cpu_fit)
    task=worker.run(s,arm)
    report,predictions=worker.training.validate_fit(root/arm/'fit',s,arm)
    assert report['epochs']==30 and len(report['validation_history'])==30
    assert report['kernel']=='original_U100' and report['selected_state']=='native_HLT_only'
    assert worker.receipt(s,arm)==task
    with pytest.raises(FileExistsError): worker.run(s,arm)

    # Real original kernel and saved native predictions for the relocated seed0 control.
    oldfit=tmp_path/'old-fit'
    fit(oldfit,f.old,'u100_w5_s0',worker.new_model(f.old['imported'],'u100_w5_s0'),
        f.train,val,f.bank,None,device='cpu')
    dpaths={'keys/manifest.json':f.d.root/'keys/manifest.json',
        'completed/u100_w5_s0/report.json':oldfit/'training_report.json',
        'completed/u100_w5_s0/predictions.npz':oldfit/'validation_predictions.npz'}
    d=SimpleNamespace(path=dpaths.__getitem__,manifest=f.d.manifest)
    monkeypatch.setattr(inputs,'delta',lambda _:d)
    bpaths={}
    for name,boost in (('M0HLT',.01),('OFFLINE',.5)):
        p=np.random.default_rng(90).uniform(.1,1.,size=(len(val),11)).astype(np.float32)
        p/=p.sum(1,keepdims=True)
        p*=1-boost; p[np.arange(len(val)),val.labels]+=boost
        a=dict(identities=val.identities,labels=val.labels,probabilities=p)
        destination=tmp_path/(name+'.npz')
        atomic_publish_bytes(destination,deterministic_npz_bytes(a))
        bpaths[f'reference/{name}/predictions.npz']=destination
    subset=tmp_path/'subset.npz'
    atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(len(val),dtype=np.int64))))
    bpaths['reference/historical/subset.npz']=subset
    monkeypatch.setattr(inputs,'Bundle',lambda _:SimpleNamespace(path=bpaths.__getitem__,json=lambda _:f.bank.manifest))
    s['bundle_root']='/relocated'; s['portability_evidence']=dict(refs=dict(predictions=file_ref(oldfit/'validation_predictions.npz')))
    rows=worker.results(s)
    by_name={r['model']:r for r in rows['rows']}
    assert by_name['u100_w5_s0']['site']=='SPORC' and by_name[arm]['site']=='Hydra'
    assert by_name[arm]['matched_control_site']=='Hydra' and by_name[arm]['validation_delta_matched']==0.
    assert by_name['u100_w5_s0']['matched_control_site']=='SPORC'
    assert by_name['u100_w5_s0']['validation_recovery']['macro_ovr_auc'] is not None
    assert len(rows['pending'])==18 and rows['independent_test'] is False
    wrong=deepcopy(report); wrong['parents']['u100']='0'*64
    monkeypatch.setattr(worker.training,'validate_fit',lambda *a:(with_content_hash(wrong),predictions))
    with pytest.raises(ValueError,match='target lineage'): worker.results(s)
