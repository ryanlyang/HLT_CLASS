"""Exact-ninths physical views, seven-head losses and immutable Oscar continuation."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import threads, tiny
from test_luka_fullstat import fullstat, durable
from test_luka_fullstat_splithead import fixture as old_fixture
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, with_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.cms_proxy_ladder.campaign import paired_seed
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullstat_splithead import worker as old_worker, contracts as old_contracts
from hlt_classification.luka_fullstat_fine_tail import (
    contracts as c, source, campaign, banks, kd_banks, model, training, worker, submission, views)


def fixture(path, export=False):
    old, caches = old_fixture(path, export=export)
    imported = deepcopy(old['imported_banks']); external = deepcopy(old['external_refs'])
    for role in ('train','validation'):
        for co in ('D022','D011'):
            # Scientific-view construction is separately tested against real ROOT below.
            base=caches[role]['D000']
            caches[role][co]=RamCache(base.blocks,role=role,coordinate_name=co,foundation_sha256=base.foundation_sha256)
    if export:
        old_worker.export_target(old,'M0HLT','kd',caches['train']['D000'],device='cpu')
        imported['M0HLT']['kd']=file_ref(Path(old['root'])/'logit_targets/M0HLT/manifest.json')
    else:
        imported['M0HLT']['kd']={}
    for t in ('DE066','DE033'):
        if export:
            d=Path(old['root'])/('train_'+t); d.mkdir()
            save_state(d/'selected.pt',DelphesParticleTransformer().state_dict())
            write_immutable_json(d/'training_report.json',old_contracts.artifact('TRAINING_REPORT',
                node=old['nodes'][t],scientific_fit=True,technical=False,selected_state='native_backbone_only'))
            for kind in ('embedding','kd'):
                old_worker.export_target(old,t,kind,caches['train'][old_contracts.STAGES[t]],device='cpu')
            external[t]=old_worker.banks.refs_for(old,t)
            imported[t]=dict(embedding=file_ref(Path(old['root'])/'targets'/t/'manifest.json'),
                             kd=file_ref(Path(old['root'])/'logit_targets'/t/'manifest.json'))
        else:
            external[t]=external['OFFLINE']; imported[t]=dict(embedding={},kd={})
    receipt=path/'completed_parent.json'; write_immutable_json(receipt,{'synthetic':True})
    r=old['reuse']
    reuse=c.artifact('REUSE',parent_spec=file_ref(Path(old['root'])/'campaign_spec.json'),
        source_commit=source.DONOR,counts=dict(c.COUNTS),accepted=r,
        anchor=r['anchor'],references=r['references'],external_refs=external,imported_banks=imported,
        base_node=r['base_node'],execution_site=r['execution_site'],environment=r['environment'],
        gpu=r['gpu'],summaries=r['summaries'],completed_parent_receipt=file_ref(receipt),
        jobs=dict(finalize_DE033='102',train_DE000='103'),protected_roots=[old['root']],
        runtime_prefix=source.RUNTIME,parent_contract=source.PARENT_CONTRACTS[1])
    root=path/'fine_tail'; root.mkdir()
    s=c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(root),project_dir='/project',source_commit='f'*40,source_snapshot=dict(content_hash='c'*64),
        reuse=reuse,available_result_bytes=8*2**30,**campaign.policy(reuse))
    s=with_content_hash(dict(s,counts=old['counts']))
    write_immutable_json(root/'campaign_spec.json',s)
    return s,caches


def mock_execution(s,cs,monkeypatch):
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:os.environ['SLURM_JOB_ID'])
    meta=(dict(content_hash=s['anchor']['evidence']['prepared'],summaries=s['reuse']['summaries'],chunks=[]),None,None)
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:cs[r][co])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'train_'+stage:str(201+i) for i,stage in enumerate(c.STAGES)})
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,
        gpu_peak_bytes=max(1,int(torch.cuda.max_memory_allocated()))))
    def prepare(s,stage,meta):
        co=c.STAGES[stage]
        built=c.artifact('VIEW_BUILD',parents=dict(spec=s['content_hash'],prepared=meta[0]['content_hash'],view=c.view_recipe(co)['content_hash']),
            coordinate=co,chunks=[],construction='raw_saved_assignments_once_in_process_RAM',seconds=.01,
            workers=s['preparation_workers'],summaries={r:meta[0]['summaries'][r]['D000'] for r in ('train','validation')},
            no_test_particle_requests=True,endpoint_replay=True)
        return cs['train'][co],cs['validation'][co],built
    monkeypatch.setattr(views,'prepare',prepare)


def test_science_registration_and_exact_ninths(tmp_path):
    s,_=fixture(tmp_path)
    assert list(c.STAGES)==['DE022','DE011','DE000']
    assert c.FRACTIONS==dict(D022=Fraction(2,9),D011=Fraction(1,9),D000=Fraction(0))
    for i,stage in enumerate(c.STAGES):
        r=c.recipe(stage); ts=list(c.EXTERNAL)+list(c.STAGES)[:i]
        assert c.teachers(stage)==ts and len(ts)==5+i
        assert r['weights']=={t:40/len(ts) for t in ts}
        assert r['logit_weights']=={t:1/len(ts) for t in ts}
        assert r['ce_coefficient']==.5 and r['temperature']==2
        assert r['logit_total']==1 and r['auxiliary_total']==40
        assert r['configuration']=='k1_e40_plain' and r['main_head_supervision']=='CE_only'
        assert not r['relational'] and not r['warm_start'] and not r['attention']
        assert r['epochs']==r['base']['maximum_passes']==20 and r['base']['warmup_passes']==2
        assert r['batch_size']==128 and s['nodes'][stage]['deployable']==(stage=='DE000')
    assert s['new_scientific_fits']==3 and s['preflight_jobs']==s['setup_jobs']==0
    assert c.seed('D000','initialization')==paired_seed('D000','initialization')
    assert c.seed('D022','sampler')!=c.seed('D011','sampler')
    assert not s['final_test_accessed'] and s['final_test_sealed']
    with pytest.raises(ValueError): c.seed('D016','sampler')
    with pytest.raises(ValueError): c.teachers('DE033')
    compile(source.REPLAY,'replay','exec')


@pytest.mark.parametrize('stage',c.STAGES)
def test_objective_gradients_and_detached_teachers(stage):
    torch.manual_seed(4); ts=c.teachers(stage)
    x=torch.randn(5,11,requires_grad=True); y=torch.arange(5)
    aux={t:torch.randn(5,11,requires_grad=True) for t in ts}
    pred={t:torch.randn(5,128,requires_grad=True) for t in ts}
    q={t:torch.softmax(torch.randn(5,11),-1).requires_grad_(True) for t in ts}
    z={t:torch.randn(5,128,requires_grad=True) for t in ts}
    loss,ce,kd,emb=training.objective(x,y,aux,pred,q,z,stage=stage)
    torch.testing.assert_close(loss,.5*ce+sum(kd.values())/len(ts)+40*sum(emb.values())/len(ts))
    expected=torch.autograd.grad(.5*ce,x,retain_graph=True)[0]
    loss.backward(); torch.testing.assert_close(x.grad,expected,rtol=0,atol=0)
    assert all(v.grad is None for d in (q,z) for v in d.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for d in (aux,pred) for v in d.values())
    q[ts[0]]=torch.full((5,11),float('nan'))
    with pytest.raises(ValueError): training.objective(x,y,aux,pred,q,z,stage=stage)


def physical_pair():
    p4=np.array([[2.,1.,.2,3.],[1.,2.,.5,3.],[3.,1.,.1,4.]])
    pv=np.array([[1,1,1,1],[0,0,0,0],[1,0,1,0]],bool)
    ov=np.array([[1,0,1,0],[1,1,1,1],[1,1,1,1]],bool)
    proxy=views.original._particles(p4,[1,0,1],[0,1,4],np.where(pv,np.arange(12).reshape(3,4)/10,0),pv,('a','b','c'))
    off=views.original._particles(p4[:2]*1.7,[1,1],[3,4],np.where(ov,np.arange(12).reshape(3,4)/5,0)[:2],ov[:2],('d','e'))
    return proxy,off,np.array([0,1,-1],np.int32)


@pytest.mark.parametrize('name,alpha',[('D033',Fraction(1,3)),('D066',Fraction(2,3)),('D020',Fraction(1,5))])
def test_fraction_kernel_exact_original_parity_and_unmatched_slots(name,alpha):
    proxy,off,mapping=physical_pair()
    for n in range(30):
        kwargs=dict(identity=f'{n:064x}',proxy=proxy,offline=off,mapping=mapping)
        expected=views.original.build_view(coordinate=name,**kwargs)
        actual=views.fraction_view(alpha=alpha,**kwargs)
        for k in ('p4','charge','category','tracking','valid'):
            np.testing.assert_array_equal(getattr(actual,k),getattr(expected,k))
            np.testing.assert_array_equal(getattr(actual,k)[2],getattr(proxy,k)[2])
        assert actual.keys==expected.keys


def test_ninth_interpolation_nested_switches_and_native_endpoint():
    proxy,off,mapping=physical_pair()
    for n in range(100):
        kw=dict(identity=f'{n:064x}',proxy=proxy,offline=off,mapping=mapping)
        assert views.build_view(coordinate='D000',**kw) is proxy
        for co,a in c.FRACTIONS.items():
            result=views.build_view(coordinate=co,**kw)
            np.testing.assert_array_equal(result.p4[0],float(1-a)*proxy.p4[0]+float(a)*off.p4[0])
        flags=[views.original._choose_offline(kw['identity'],0,'identity',a)
               for a in (Fraction(0),Fraction(1,9),Fraction(2,9),Fraction(1,3))]
        assert flags==sorted(flags)
    with pytest.raises(ValueError): views.build_view(coordinate='D016',**kw)
    with pytest.raises(ValueError): views.fraction_view(alpha=.22,**kw)
    with pytest.raises(ValueError): views.build_view(coordinate='D022',**dict(kw,mapping=np.array([0,0,-1])))
    with pytest.raises(ValueError): views.build_view(coordinate='D022',**dict(kw,mapping=np.array([0.,1.,-1.])))


def view_spec(fullstat,durable,workers=1):
    container,foundation_root,f=fullstat; root,meta=durable
    return dict(content_hash='f'*64,anchor=dict(evidence=dict(prepared=meta[0]['content_hash'],
        raw_container=str(container),foundation_root=str(foundation_root),cache_root=str(root))),
        counts=f['counts'],memory_mb=180000,preparation_workers=workers)


@pytest.mark.parametrize('stage',['DE022','DE011'])
def test_real_root_views_sealed_reads_and_original_bytes(fullstat,durable,monkeypatch,stage):
    import uproot
    s=view_spec(fullstat,durable); meta=durable[1]
    original=uproot.behaviors.TTree.TTree.arrays
    def guarded(tree,expressions,*args,**kw):
        if any('part_' in x for x in expressions):
            index=next(i for i,r in enumerate(meta[1]['files']) if Path(tree.file.file_path).as_posix().endswith(r['path']))
            roles=meta[2][index]['roles'][kw['entry_start']:kw['entry_stop']]
            assert np.isin(roles,(0,1,255)).all()
        return original(tree,expressions,*args,**kw)
    monkeypatch.setattr(uproot.behaviors.TTree.TTree,'arrays',guarded)
    monkeypatch.setattr(views.original,'match_particles',lambda *a:pytest.fail('No rematching'))
    tr,val,built=views.prepare(s,stage,meta)
    assert len(tr)==264 and len(val)==66 and tr.coordinate_name==c.STAGES[stage]
    views.validate_build(s,stage,json.loads(json.dumps(built,sort_keys=True)),meta[0])
    for role,value in [('train',tr),('validation',val)]:
        old=views.cache.load(durable[0],meta,role=role,coordinate='D000',max_ram_bytes=2**30)
        np.testing.assert_array_equal(value.identities,old.identities)
        np.testing.assert_array_equal(value.labels,old.labels)
        assert value.nbytes==old.nbytes
    bad=deepcopy(built); bad['coordinate']='D000'
    with pytest.raises(ValueError): views.validate_build(s,stage,with_content_hash(bad),meta[0])
    with pytest.raises(ValueError): views.prepare(s,'DE000',meta)


def test_spawn_views_equal_serial(fullstat,durable):
    s=view_spec(fullstat,durable)
    a,b,r=views.prepare(s,'DE022',durable[1])
    x,y,p=views.prepare(dict(s,preparation_workers=2),'DE022',durable[1])
    assert r['chunks']==p['chunks']
    assert [views.digest(v) for v in a.blocks+b.blocks]==[views.digest(v) for v in x.blocks+y.blocks]


@pytest.mark.parametrize('field',['entries','offsets','mapping','identities','labels'])
def test_assignment_payload_cannot_change_role_order_or_type(durable,monkeypatch,field):
    root,meta=durable; manifest,foundation,members=meta; chunk=manifest['chunks'][0]
    i=chunk['file_index']
    expected=views.assignments(chunk,foundation['files'][i],members[i],root)
    changed={k:np.array(v,copy=True) for k,v in expected.items()}
    if field=='mapping': changed[field]=changed[field].astype(np.int64)
    else: changed[field].flat[0]+=1
    monkeypatch.setattr(views,'load_npz_arrays',lambda *a:changed)
    with pytest.raises(ValueError,match='membership differs'):
        views.assignments(chunk,foundation['files'][i],members[i],root)


def test_source_or_endpoint_mismatch_stops_before_training(fullstat,durable,monkeypatch):
    s=view_spec(fullstat,durable)
    original=views.digest
    # Stored D000 must replay exactly; new view generation cannot silently
    # substitute a changed decoder even if row identities still match.
    calls=[]
    def different(block):
        calls.append(block)
        return '0'*64 if len(calls)==1 else original(block)
    monkeypatch.setattr(views,'digest',different)
    with pytest.raises(ValueError,match='endpoint replay differs'):
        views.prepare(s,'DE022',durable[1])
    monkeypatch.setattr(views,'digest',original)
    def missing(*a): raise ValueError('Raw source bytes differ')
    monkeypatch.setattr(views.data,'authenticate_source',missing)
    with pytest.raises(ValueError,match='Raw source bytes differ'):
        views.prepare(s,'DE022',durable[1])


@pytest.mark.parametrize('stage',c.STAGES)
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Installed-Weaver physical128 GPU acceptance')
def test_native_parity_startup_all_seven_heads(tmp_path,monkeypatch,stage):
    s,cs=fixture(tmp_path)
    co=c.STAGES[stage]; tr,val=cs['train'][co],cs['validation'][co]; ts=c.teachers(stage)
    rng=np.random.default_rng(3)
    manifest=c.artifact('BUNDLE',prepared=tr.foundation_sha256,stage=stage,role='train',
        rows=len(tr),teacher_order=ts,temperature=2.)
    bank=SimpleNamespace(labels=tr.labels,identities=tr.identities,manifest=manifest,
        probabilities={t:np.full((len(tr),11),1/11,np.float32) for t in ts},
        targets={t:rng.normal(size=(len(tr),128)).astype(np.float32) for t in ts})
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    before=model.new_model(s['nodes'][stage]).native_state()
    record=worker.startup(s,stage,tr,val,bank,device='cuda',preparation_seconds=300.)
    worker.validate_startup(s,stage,record,manifest['content_hash'])
    assert record['native_fp32_bf16_parity'] and record['main_head_gradient_CE_only']
    assert record['relational'] is False and record['timing']['preparation_seconds']==300.
    assert len(record['head_gradient_norms'])==2*len(ts)
    after=model.new_model(s['nodes'][stage]).native_state()
    assert all(torch.equal(v,after[k]) for k,v in before.items())


def test_five_imports_keep_contracts_and_reject_wrong_join(tmp_path,monkeypatch):
    tiny(monkeypatch); s,cs=fixture(tmp_path,True); tr=cs['train']['D022']
    bank=banks.Bundle(s,'DE022',tr)
    assert set(bank.targets)==set(bank.probabilities)==set(c.EXTERNAL)
    assert load_json(s['imported_banks']['DE033']['embedding']['path'])['contract']=='LUKA_FULLSTAT_SPLITHEAD_BANK/v1'
    for role in ('validation','final_test','unused'):
        wrong=deepcopy(tr); wrong.role=role
        with pytest.raises(PermissionError): banks.Bundle(s,'DE022',wrong)
    wrong=deepcopy(tr); wrong.identities=wrong.identities[::-1]
    with pytest.raises(ValueError): banks.Bundle(s,'DE022',wrong)
    bad=deepcopy(s); bad['imported_banks']['UE100']=s['imported_banks']['DE033']
    with pytest.raises(ValueError): banks.Bundle(bad,'DE022',tr)
    path=Path(s['imported_banks']['DE066']['kd']['path']).parent/'probabilities.npz'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): banks.Bundle(s,'DE022',tr)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Full native architecture GPU miniature')
def test_three_full20_workers_exports_roundtrip_results(tmp_path,monkeypatch):
    s,cs=fixture(tmp_path,True); mock_execution(s,cs,monkeypatch)
    before={t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}
    for i,stage in enumerate(c.STAGES):
        monkeypatch.setenv('SLURM_JOB_ID',str(201+i))
        record=worker.run(s,'train_'+stage); assert record==worker.completed(s,stage)
        directory=Path(s['root'])/('train_'+stage)
        report=load_json(directory/'training_report.json')
        assert report['passes']==20 and len(report['validation_history'])==20
        bundle=load_json(Path(s['root'])/'bundles'/(stage+'.json'))
        worker.report_valid(json.loads(json.dumps(report,sort_keys=True)),s,stage,bundle['content_hash'])
        for name in ('total_loss','embedding_total','learning_rate'):
            bad=deepcopy(report); bad['validation_history'][0][name]+=.1
            with pytest.raises(ValueError): worker.report_valid(with_content_hash(bad),s,stage,bundle['content_hash'])
        state=torch.load(directory/'selected.pt',weights_only=True,map_location='cpu')
        assert not any(k.startswith(('predictors.','logit_heads.')) for k in state)
        native=DelphesParticleTransformer().cuda(); native.load_state_dict(state,strict=True)
        if stage!='DE000':
            p=load_npz_arrays(Path(s['root'])/'logit_targets'/stage/'probabilities.npz')
            np.testing.assert_array_equal(p['probabilities'],predict(native,cs['train'][c.STAGES[stage]],device='cuda',temperature=2.,batch_size=128))
        summary=worker.results(s); assert summary['pending']==list(c.STAGES)[i+1:]
        with pytest.raises(FileExistsError): worker.run(s,'train_'+stage)
    assert summary['rows'][-1]['input']=='native_HLT'
    assert not (Path(s['root'])/'targets/DE000').exists()
    assert before=={t:{k:file_ref(v['path']) for k,v in kinds.items()} for t,kinds in s['imported_banks'].items()}


def test_three_job_dry_live_binding_and_duplicate_guard(tmp_path,monkeypatch):
    s,_=fixture(tmp_path); monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(submission,'check_site',lambda _:None)
    p=submission.submit(s)
    assert p['jobs']==p['scientific_fits']==3 and p['preflight_jobs']==p['setup_jobs']==0
    with pytest.raises(PermissionError): submission.submit(s,execute=True,plan_hash=p['content_hash'])
    calls=[]
    def fake(args,**kw):
        calls.append(args); return SimpleNamespace(returncode=0,stdout=str(200+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(s)==dict(train_DE022='201',train_DE011='202',train_DE000='203')
    assert '--dependency=afterok:201:202' in calls[-1]
    assert all('--kill-on-invalid-dep=yes' in row for row in calls)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True,plan_hash=p['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==3


def test_timing_accounts_for_preparation_and_fails_closed(tmp_path):
    s,_=fixture(tmp_path)
    a=worker.timing_record(s,'DE022',.1,.01)
    b=worker.timing_record(s,'DE022',.1,.01,3600.)
    assert b['conservative_minutes']==a['conservative_minutes']+60
    for invalid in (float('nan'),-1,1e9):
        with pytest.raises(ValueError): worker.timing_record(s,'DE022',.1,.01,invalid)
    with pytest.raises(MemoryError): worker.limits(s,dict(peak_rss_bytes=180000*1024**2,gpu_peak_bytes=1))


def test_missing_ancestor_never_trains(tmp_path,monkeypatch):
    s,cs=fixture(tmp_path); mock_execution(s,cs,monkeypatch)
    monkeypatch.setenv('SLURM_JOB_ID','202')
    monkeypatch.setattr(worker,'fit',lambda *a,**kw:pytest.fail('No fit'))
    with pytest.raises(ValueError,match='incomplete'): worker.run(s,'train_DE011')
    assert not (Path(s['root'])/'train_DE011_attempt').exists()


@pytest.mark.parametrize('stage',c.STAGES)
def test_rehashed_wrong_weights_or_views_rejected(tmp_path,monkeypatch,stage):
    s,_=fixture(tmp_path); s=with_content_hash(dict(s,counts=dict(c.COUNTS)))
    monkeypatch.setattr(source,'current',lambda *a:s['source_snapshot'])
    monkeypatch.setattr(source,'authenticate',lambda *a:s['reuse'])
    monkeypatch.setattr(campaign,'load_json',lambda *a:s)
    monkeypatch.setattr(campaign,'protected',lambda *a:None)
    assert campaign.check(s)==s['reuse']
    s['recipes'][stage]=with_content_hash(dict(s['recipes'][stage],auxiliary_total=5.))
    s.update(with_content_hash(s))
    with pytest.raises(ValueError,match='registration changed'): campaign.check(s)
