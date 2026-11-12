"""Real frozen-bank copies, complete worker publication and exact continuation."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_fullstat_six_teacher import registered as donor_registered, probabilities
from test_luka_fullstat_six_teacher_targets import frozen,add_ladder
from test_luka_fullstat_six_teacher_s0 import registered
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as Offline
from hlt_classification.luka_fullstat_six_teacher import kd_banks as old_kd, recovery as old_recovery
from hlt_classification.luka_fullstat_six_teacher_s0 import (worker,targets,banks,model,training,
    startup,campaign,submission,recovery,kd_banks,contracts as c)


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def setup_banks(tmp_path, rows=330):
    (tmp_path/'banks').mkdir()
    parent,tr,offline,original_subclass=frozen(tmp_path/'banks',rows=rows)
    add_ladder(parent,tr)
    path=Path(parent['root'])/'campaign_spec.json';write_immutable_json(path,parent)
    baseline=c.artifact('BASELINE',spec=file_ref(path),exports=old_recovery.export_records(parent),reference={})
    new=with_content_hash(dict(parent,root=str(tmp_path/'luka_fullstat_six_teacher_s0_bank_fixture'),baseline=baseline))
    Path(new['root']).mkdir()
    return new,parent,tr,offline,original_subclass


def test_exact_six_bank_reuse_without_teacher_forward_or_reclustering(tmp_path,monkeypatch):
    spec,parent,tr,offline,original=setup_banks(tmp_path,rows=6006)
    monkeypatch.setattr(old_kd,'export',lambda *a:pytest.fail('No teacher forwards'))
    monkeypatch.setattr(old_kd.views,'load',lambda *a:pytest.fail('No privileged view loading'))
    from hlt_classification.luka_fullstat_subclass import targets as original_targets
    monkeypatch.setattr(original_targets,'fit',lambda *a:pytest.fail('No reclustering'))
    kd_banks.prepare(spec,'cpu',0)
    loaded,sampler=banks.load(spec,tr,'cpu')
    assert set(loaded)==set(c.TEACHERS) and 'U100' not in loaded
    for t in c.TEACHERS:
        np.testing.assert_array_equal(loaded[t].identities,tr.identities)
        np.testing.assert_array_equal(loaded[t].probabilities,probabilities(len(tr)))
    np.testing.assert_array_equal(sampler.keys.cpu().numpy(),offline.targets/np.maximum(np.linalg.norm(offline.targets,axis=1,keepdims=True),1e-12))
    np.testing.assert_array_equal(sampler.subclass_bank.targets,original_targets.Bank(original,tr).targets)
    for t in kd_banks.NAMES:
        for n in kd_banks.FILES:
            assert (kd_banks.folder(spec,t)/n).read_bytes()==(old_kd.folder(parent,t)/n).read_bytes()
    with pytest.raises(FileExistsError):kd_banks.prepare(spec,'cpu',0)
    (kd_banks.folder(spec,'FUSION')/'probabilities.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):kd_banks.manifest(spec,'FUSION')


@pytest.mark.parametrize('bad',['baseline_bytes','teacher_payload','join','role','prepared','source_copy'])
def test_bank_integrity_and_role_fail_closed(tmp_path,bad):
    s,parent,tr,offline,original=setup_banks(tmp_path)
    if bad=='baseline_bytes':
        Path(s['baseline']['spec']['path']).write_bytes(b'{}')
        with pytest.raises(ValueError):kd_banks.prepare(s,'cpu',0)
        return
    if bad=='teacher_payload':
        (old_kd.folder(parent,'CONCAT')/'probabilities.npz').write_bytes(b'corrupt')
        with pytest.raises(ValueError):kd_banks.prepare(s,'cpu',0)
        return
    kd_banks.prepare(s,'cpu',0)
    if bad=='join':tr.identities=tr.identities[::-1].copy()
    elif bad=='role':tr.role='validation'
    elif bad=='prepared':tr.foundation_sha256='f'*64
    else:
        m=load_json(kd_banks.folder(s,'CONCAT')/'manifest.json');m['parents']['source']='0'*40
        (kd_banks.folder(s,'CONCAT')/'manifest.json').write_text(__import__('json').dumps(with_content_hash(m)))
    with pytest.raises((ValueError,PermissionError)):kd_banks.arrays(s,'CONCAT',tr)


def test_completed_control_export_recovery_chain_is_preserved(tmp_path,monkeypatch):
    s,exporter,tr,_,_=setup_banks(tmp_path)
    original=Path(exporter['root'])/'campaign_spec.json'
    control=with_content_hash(dict(exporter,root=str(tmp_path/'recovered_control'),
        source_commit='e'*40,resume=dict(spec=file_ref(original),exports=old_recovery.export_records(exporter))))
    Path(control['root']).mkdir()
    monkeypatch.setattr(old_kd,'export',lambda *a:pytest.fail('Never regenerate frozen logits'))
    old_kd.prepare(control,'cpu',0)
    control_path=Path(control['root'])/'campaign_spec.json'
    write_immutable_json(control_path,control)
    baseline=c.artifact('BASELINE',spec=file_ref(control_path),
        exports=old_recovery.export_records(control),reference={})
    s=with_content_hash(dict(s,baseline=baseline))
    kd_banks.prepare(s,'cpu',0)
    for name in kd_banks.NAMES:
        assert kd_banks.manifest(s,name)['parents']['source']==exporter['source_commit']
        assert exporter['source_commit']!=control['source_commit']
        _,a=kd_banks.arrays(s,name,tr)
        np.testing.assert_array_equal(a['identities'],tr.identities)
        for filename in kd_banks.FILES:
            assert (kd_banks.folder(s,name)/filename).read_bytes()==(old_kd.folder(exporter,name)/filename).read_bytes()


def test_full_worker_late_failure_recovery_publication_and_results(tmp_path,monkeypatch):
    tiny(monkeypatch)
    (tmp_path/'banks').mkdir();base,tr,offline,original_subclass=frozen(tmp_path/'banks')
    val=cache('validation','D000',rows=22)
    (tmp_path/'original').mkdir();parent=donor_registered(tmp_path/'original')
    parent=with_content_hash(dict(parent,root=base['root'],counts=dict(c.COUNTS,train=len(tr),validation=len(val)),
        reuse=with_content_hash(dict(parent['reuse'],banks=base['reuse']['banks'],subclasses=base['reuse']['subclasses'],parent_spec=base['reuse']['parent_spec']))))
    parent['anchor']['evidence']['prepared']=tr.foundation_sha256;add_ladder(parent,tr);parent=with_content_hash(parent)
    oldroot=Path(parent['root']);write_immutable_json(oldroot/'campaign_spec.json',parent)
    baseline=c.artifact('BASELINE',spec=file_ref(oldroot/'campaign_spec.json'),exports=old_recovery.export_records(parent),reference={})
    (tmp_path/'new').mkdir();s=registered(tmp_path/'new')
    spec=with_content_hash(dict(s,baseline=baseline,reuse=parent['reuse'],counts=parent['counts'],anchor=parent['anchor']))
    root=Path(spec['root']);root.mkdir();write_immutable_json(root/'campaign_spec.json',spec)
    monkeypatch.setattr(old_kd,'export',lambda *a:pytest.fail('Must copy finished targets'))
    monkeypatch.setattr(campaign,'check',lambda *a:None);monkeypatch.setattr(campaign,'storage',lambda *a:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'9000001')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**k:{c.TASK:'9000001'})
    monkeypatch.setattr(submission,'accounting',lambda *a,**k:['9000001','FAILED'])
    monkeypatch.setattr(worker.views,'load',lambda *a:(tr,val,None))
    teacher_banks={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities,labels=tr.labels,probabilities=probabilities(len(tr),i)) for i,n in enumerate(c.TEACHERS)}
    keys=torch.nn.functional.normalize(torch.randn(1025,128),dim=-1)
    def load(s,*a):
        rec=s['reuse']['banks']['OFFLINE'];off=Offline(Path(rec['manifest']['path']).parent,tr,refs=rec['refs'])
        bank=targets.prepare(s,tr,off)
        sampler=SimpleNamespace(identities=tr.identities,labels=tr.labels,subclass_bank=bank,
            candidate_keys=lambda ix:keys.unsqueeze(0).expand(len(ix),-1,-1))
        return teacher_banks,sampler
    monkeypatch.setattr(banks,'load',load)
    class Batches:
        def __init__(self,s,order,epoch):self.order=order
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):
            for start in range(0,len(self.order),128):
                ix=self.order[start:start+128]
                yield ix,np.column_stack((ix,np.tile(np.arange(1024),(len(ix),1))))
    monkeypatch.setattr(training,'Lookahead',Batches)
    def probe(s,*a):
        proof=c.artifact('STARTUP',parents=dict(spec=s['content_hash']))
        write_immutable_json(Path(s['root'])/c.TASK/'startup_measurements.json',proof);return proof
    monkeypatch.setattr(startup,'run',probe);monkeypatch.setattr(startup,'check',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda *a:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=0))
    monkeypatch.setattr(worker,'check_memory',lambda *a:None)
    monkeypatch.setattr(worker,'CPUWorkers',lambda *a:nullcontext(SimpleNamespace(memory=lambda:0)))
    monkeypatch.setattr(worker,'reference_arrays',lambda *a:(None,dict(identities=val.identities,labels=val.labels)))
    publish=worker.publish
    monkeypatch.setattr(worker,'publish',lambda *a:(_ for _ in ()).throw(RuntimeError('late failure')))
    with pytest.raises(RuntimeError,match='late failure'):worker.run(spec,device='cpu')
    records=worker.epoch_records(spec);assert len(records)==20
    assert all(r['row']['losses']['weighted_subclass']==0 for r in records)
    resume=recovery.inspect(root/'campaign_spec.json',spec['project_dir'],spec['source_commit'])
    with pytest.raises(ValueError,match='same pinned source'):
        recovery.inspect(root/'campaign_spec.json',spec['project_dir'],'0'*40)
    resumed=with_content_hash(dict(spec,root=str(tmp_path/'luka_fullstat_six_teacher_s0_recovered'),resume=resume))
    new=Path(resumed['root']);new.mkdir();write_immutable_json(new/'campaign_spec.json',resumed)
    monkeypatch.setattr(worker,'publish',publish)
    monkeypatch.setattr(model,'objective',lambda *a,**k:pytest.fail('Finished fit repeated training'))
    rec=worker.run(resumed,device='cpu');report,pred=worker.validate_fit(resumed)
    assert report['parents']['recipe']==c.recipe()['content_hash']
    assert all(r['row']['subclass_coefficient']==0 for r in worker.epoch_records(resumed))
    for teacher in kd_banks.NAMES:
        for name in kd_banks.FILES:
            assert (root/'logit_targets'/teacher/name).read_bytes()==(new/'logit_targets'/teacher/name).read_bytes()
    def reference(s,name):
        r=deepcopy(report)
        if name=='M0HLT':r['validation']['macro_ovr_auc']=.4;r['validation']['accuracy']=.1
        elif name=='OFFLINE':r['validation']['macro_ovr_auc']=.7;r['validation']['accuracy']=.5
        return r,pred
    monkeypatch.setattr(worker,'reference_arrays',reference)
    result=worker.results(resumed)
    assert [r['model'] for r in result['rows']][-2:]==['SIX_TEACHER','SIX_TEACHER_S0']
    assert all('rejection_recovery' in r for r in result['rows'])
    (new/'subclasses/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):worker.completed(resumed)
