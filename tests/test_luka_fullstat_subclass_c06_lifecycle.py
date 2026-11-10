"""Target bank, complete worker, results, and recovery retain the same lineage."""
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_fullstat_subclass_c06 import registered
from test_luka_fullstat_subclass_c06_targets import frozen
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as Offline
from hlt_classification.luka_fullstat_subclass_c06 import (worker,targets,banks,model,training,
    startup,campaign,submission,recovery,contracts as c)


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_complete_target_bound_worker_and_publication_recovery(tmp_path,monkeypatch):
    tiny(monkeypatch)
    # Use real authenticated teacher-bank artifacts with enough rows per prototype.
    (tmp_path/'banks').mkdir();base,tr,offline,parent_bank=frozen(tmp_path/'banks')
    val=cache('validation','D000',rows=22)
    (tmp_path/'campaign').mkdir();original=registered(tmp_path/'campaign')
    reuse=with_content_hash(dict(original['reuse'],banks=base['reuse']['banks'],subclasses=base['reuse']['subclasses'],parent_spec=base['reuse']['parent_spec']))
    spec=with_content_hash(dict(original,reuse=reuse,parents=dict(original['parents'],reuse=reuse['content_hash']),
        counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    spec['anchor']['evidence']['prepared']=tr.foundation_sha256;spec=with_content_hash(spec)
    root=Path(spec['root']);root.mkdir();write_immutable_json(root/'campaign_spec.json',spec)
    monkeypatch.setattr(campaign,'check',lambda *a:None);monkeypatch.setattr(campaign,'storage',lambda *a:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'9000001')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**k:{c.TASK:'9000001'})
    monkeypatch.setattr(submission,'accounting',lambda *a,**k:['9000001','failed'])
    monkeypatch.setattr(worker.views,'load',lambda *a:(tr,val,None))
    teacher_banks={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities,labels=tr.labels) for i,n in enumerate(c.TEACHERS)}
    keys=torch.nn.functional.normalize(torch.randn(1025,128),dim=-1)
    def load(s,*a):
        rec=s['reuse']['banks']['OFFLINE']
        off=Offline(Path(rec['manifest']['path']).parent,tr,refs=rec['refs'])
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
    bankhash=targets.check_manifest(spec)['content_hash']
    assert all(r['parents']['subclasses']==bankhash for r in records)
    resume=recovery.inspect(root/'campaign_spec.json',spec['project_dir'],spec['source_commit'])
    resumed=with_content_hash(dict(spec,root=str(tmp_path/'luka_fullstat_subclass_c06_recovered'),resume=resume))
    new=Path(resumed['root']);new.mkdir();write_immutable_json(new/'campaign_spec.json',resumed)
    monkeypatch.setattr(worker,'publish',publish)
    monkeypatch.setattr(model,'objective',lambda *a,**k:pytest.fail('Completed fit must not train'))
    monkeypatch.setattr(targets.previous,'fit',lambda *a,**k:pytest.fail('Recovery must not recluster'))
    rec=worker.run(resumed,device='cpu')
    assert rec['outputs'][0]['path']=='subclasses/manifest.json'
    report,pred=worker.validate_fit(resumed)
    assert all(r['parents']['subclasses']==bankhash for r in worker.epoch_records(resumed))
    for name in targets.FILES:assert (root/'subclasses'/name).read_bytes()==(new/'subclasses'/name).read_bytes()
    def reference(s,name):
        r=deepcopy(report)
        # Exercise recovery denominator and reference ordering without inference.
        if name=='M0HLT':r['validation']['macro_ovr_auc']=.4;r['validation']['accuracy']=.1
        elif name=='OFFLINE':r['validation']['macro_ovr_auc']=.7;r['validation']['accuracy']=.5
        return r,pred
    monkeypatch.setattr(worker,'reference_arrays',reference)
    result=worker.results(resumed)
    assert [r['model'] for r in result['rows']]==['M0HLT','OFFLINE','U100_W5','MULTI3','MULTI3_MIX','SUBCLASS_C03','SUBCLASS_C06']
    assert all('rejection_recovery' in r for r in result['rows'])
    (new/'subclasses/targets.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):worker.completed(resumed)
