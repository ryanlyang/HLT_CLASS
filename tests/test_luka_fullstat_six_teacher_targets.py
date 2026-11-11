"""six-teacher must reuse exactly the completed C03 TRAIN targets without reclustering."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from hlt_classification.data.cache_contracts import write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as Offline
from hlt_classification.luka_fullstat_subclass import targets as previous
from hlt_classification.luka_fullstat_six_teacher import targets,banks
from test_luka_fullstat_subclass_targets import target_spec
import torch
import time
from test_luka_embedding_aux import tiny
from test_luka_attention import cache
from test_luka_fullstat_six_teacher import probabilities,node
from hlt_classification.luka_fullstat_six_teacher import kd_banks,contracts as c
from hlt_classification.data.cache_contracts import atomic_publish_bytes,deterministic_npz_bytes,load_json


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def add_ladder(spec,tr):
    from hlt_classification.luka_fullstat_hybrid_embedding import banks as hybrid, kd_banks as hybrid_kd
    from hlt_classification.luka_fullstat_splithead import banks as split, kd_banks as split_kd
    from hlt_classification.luka_embedding_aux.targets import standardize
    spec['source_commit']='f'*40;spec['minutes']={c.TASK:2880};spec['memory_mb']=192000
    ladder=dict(banks={},refs={})
    for t in ('OFFLINE',*c.LADDER):
        d=Path(spec['root'])/'donor'/t;d.mkdir(parents=True)
        if t=='OFFLINE':refs=spec['reuse']['banks'][t]['refs']
        else:
            (d/'checkpoint').write_bytes(t.encode());(d/'report').write_bytes(t.encode())
            refs=dict(checkpoint=file_ref(d/'checkpoint'),report=file_ref(d/'report'))
        ladder['refs'][t]=refs;ladder['banks'][t]={}
        coord=dict(OFFLINE='OFFLINE',UE100='U100',DE066='D066',DE033='D033')[t]
        eb=hybrid if t=='UE100' else split;kb=hybrid_kd if t in ('OFFLINE','UE100') else split_kd
        if t!='OFFLINE':
            x,mean,scale=standardize(np.random.default_rng(len(t)).normal(size=(len(tr),128)).astype(np.float32))
            atomic_publish_bytes(d/'targets.npz',deterministic_npz_bytes(dict(identities=tr.identities,labels=tr.labels,targets=x,mean=mean,scale=scale)))
            m=eb.artifact('BANK',parents=eb.parents(tr.foundation_sha256,refs,t),teacher=t,coordinate=coord,role='train',rows=len(tr),width=128,
                checkpoint=refs['checkpoint'],report=refs['report'],payload=file_ref(d/'targets.npz',root=d))
            write_immutable_json(d/'embedding.json',m);ladder['banks'][t]['embedding']=file_ref(d/'embedding.json')
        atomic_publish_bytes(d/'probabilities.npz',deterministic_npz_bytes(dict(identities=tr.identities,labels=tr.labels,probabilities=probabilities(len(tr)))))
        m=kb.artifact('KD_BANK',parents=kb.parents(tr.foundation_sha256,refs,t),teacher=t,coordinate=coord,role='train',rows=len(tr),temperature=2.,class_count=11,
            checkpoint=refs['checkpoint'],report=refs['report'],payload=file_ref(d/'probabilities.npz',root=d))
        write_immutable_json(d/'kd.json',m);ladder['banks'][t]['kd']=file_ref(d/'kd.json')
    spec['reuse']['ladder']=ladder
    for t in kd_banks.NAMES:
        d=kd_banks.folder(spec,t);d.mkdir(parents=True)
        atomic_publish_bytes(d/'probabilities.npz',deterministic_npz_bytes(dict(identities=tr.identities,labels=tr.labels,probabilities=probabilities(len(tr)))))
        m=c.artifact('KD_BANK',parents=kd_banks.parents(spec,t),teacher=t,coordinate=t,role='train',rows=len(tr),temperature=2.,class_count=11,export_batch=128,
            payload=file_ref(d/'probabilities.npz',root=d))
        write_immutable_json(d/'manifest.json',m)
    return spec


def test_six_actual_bank_readers_with_original_offline_keys_and_subclasses(tmp_path,monkeypatch):
    s,tr,offline,old=frozen(tmp_path,6006);add_ladder(s,tr)
    monkeypatch.setattr(previous,'fit',lambda *a:pytest.fail('No refit'))
    b,sampler=banks.load(s,tr,'cpu')
    assert set(b)==set(c.TEACHERS) and 'U100' not in b
    for t in c.TEACHERS:
        np.testing.assert_array_equal(b[t].identities,tr.identities)
        np.testing.assert_array_equal(b[t].probabilities,probabilities(len(tr)))
    np.testing.assert_array_equal(sampler.keys.cpu().numpy(),offline.targets/np.maximum(np.linalg.norm(offline.targets,axis=1,keepdims=True),1e-12))
    np.testing.assert_array_equal(sampler.subclass_bank.targets,previous.Bank(old,tr).targets)


@pytest.mark.parametrize('name',['CONCAT','FUSION'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_logit_export_and_strict_join(tmp_path,monkeypatch,name,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tiny(monkeypatch)
    from hlt_classification.luka_u100_multiview.views import PairedCache
    from hlt_classification.luka_u100_multiview import teachers
    tr=cache(coordinate='D000',rows=129)
    privileged=deepcopy(tr);privileged.coordinate_name='CONCAT'
    if name=='FUSION':privileged=PairedCache(tr,cache(coordinate='OFFLINE',rows=129))
    net=kd_banks.new_teacher(node(),'teacher_'+name)
    cp=tmp_path/'selected.pt';torch.save(net.state_dict(),cp)
    rp=tmp_path/'report.json';rp.write_text('{}')
    owner=tmp_path/'owner.json'
    write_immutable_json(owner,dict(content_hash='owner',nodes={'teacher_'+name:node()}))
    refs=dict(checkpoint=file_ref(cp),report=file_ref(rp))
    s=dict(root=str(tmp_path/'result'),source_commit='f'*40,counts=dict(train=len(tr)),memory_mb=192000,minutes={c.TASK:2880},
        contract='LUKA_FULLSTAT_SIX_TEACHER_SPEC/v1',
        anchor=dict(evidence=dict(prepared=tr.foundation_sha256)),reuse=dict(u100=dict(gpu=dict(total_memory_bytes=32*2**30)),banks={name:dict(refs=refs,owner=file_ref(owner))}))
    monkeypatch.setattr(kd_banks,'memory_snapshot',lambda *a:dict(peak_rss_bytes=2**30,
        gpu_peak_bytes=1 if device=='cpu' else int(torch.cuda.max_memory_allocated())))
    monkeypatch.setattr(kd_banks,'NAMES',(name,))
    calls=[]
    def view_loader(spec,task):
        assert task=='teacher_'+name;calls.append(task)
        return privileged,object(),None
    monkeypatch.setattr(kd_banks.views,'load',view_loader)
    kd_banks.prepare(s,device,time.monotonic())
    assert calls==['teacher_'+name]
    m,a=kd_banks.arrays(s,name,tr)
    net.to(device).eval().requires_grad_(False);expected=[]
    with torch.inference_mode():
        for start in range(0,len(tr),128):
            logits=teachers.forward(net,privileged.batch(np.arange(start,min(start+128,len(tr)))),device,bf16=device=='cuda',name=name)
            expected.append((logits.float()/2).softmax(1).cpu().numpy())
    np.testing.assert_array_equal(a['probabilities'],np.concatenate(expected))
    bad=deepcopy(tr);bad.identities=bad.identities[::-1].copy()
    with pytest.raises(ValueError):kd_banks.arrays(s,name,bad)
    bad=deepcopy(tr);bad.role='validation'
    with pytest.raises(PermissionError):kd_banks.arrays(s,name,bad)
    with pytest.raises(FileExistsError):kd_banks.export(s,name,privileged,net,device,time.monotonic())
    (kd_banks.folder(s,name)/'probabilities.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):kd_banks.manifest(s,name)


@pytest.mark.parametrize('bad',['none','source','dirty','replay_failure'])
def test_ladder_authentication_runs_the_frozen_original_source(tmp_path,monkeypatch,bad):
    import json
    import sys
    from types import SimpleNamespace
    from hlt_classification.luka_fullstat_six_teacher import source
    from hlt_classification.luka_fullstat_finetune.source import DONOR
    from hlt_classification.luka_fullstat_fine_tail.contracts import artifact as old_artifact
    root=tmp_path/'parent';root.mkdir();project=tmp_path/'original';project.mkdir()
    s=old_artifact('SPEC',root=str(root),project_dir=str(project),counts=c.COUNTS,
        source_commit='0'*40 if bad=='source' else DONOR)
    path=root/'campaign_spec.json';write_immutable_json(path,s)
    e=dict(anchor={},execution_site={},environment={},gpu={},summaries={},protected_roots=[],reference={},
        imported_banks={t:dict(embedding={},kd={}) for t in ('OFFLINE','UE100','DE066','DE033','M0HLT','DE022','DE011')},
        teacher_refs={t:{} for t in ('OFFLINE','UE100','DE066','DE033','M0HLT','DE022','DE011')})
    monkeypatch.setattr(source,'git',lambda p,*a:DONOR if a[0]=='rev-parse' else 'dirty' if bad=='dirty' else '')
    calls=[]
    def replay(args,**kw):
        calls.append((args,kw))
        assert args[0]==sys.executable and args[-1]==str(path.resolve())
        assert kw['cwd']==project and kw['env']['PYTHONPATH']==str(project/'src')
        assert 'worker.completed' in args[-2]
        return SimpleNamespace(returncode=1 if bad=='replay_failure' else 0,stderr='fixture failure',
            stdout='FINETUNE_REUSE='+json.dumps(e))
    monkeypatch.setattr(source.subprocess,'run',replay)
    if bad=='none':
        r=source.authenticate_ladder(path)
        assert set(r['banks'])=={'OFFLINE','UE100','DE066','DE033'} and r['spec']==file_ref(path)
        assert len(calls)==1
    else:
        with pytest.raises(ValueError):source.authenticate_ladder(path)
        assert len(calls)==(1 if bad=='replay_failure' else 0)


def frozen(tmp_path,rows=330):
    tmp_path.mkdir(exist_ok=True)
    old,tr=target_spec(tmp_path,rows=rows)
    rec=old['reuse']['banks']['OFFLINE']
    offline=Offline(Path(rec['manifest']['path']).parent,tr,refs=rec['refs'])
    previous.prepare(old,tr,offline)
    path=Path(old['root'])/'campaign_spec.json';write_immutable_json(path,old)
    folder=Path(old['root'])/'subclasses'
    proof=dict(manifest_hash=previous.check_manifest(old)['content_hash'],files=[file_ref(folder/n) for n in targets.FILES])
    spec=with_content_hash(dict(old,root=str(tmp_path/'luka_fullstat_six_teacher_fresh'),
        reuse=dict(old['reuse'],parent_spec=file_ref(path),subclasses=proof),resume=None))
    Path(spec['root']).mkdir()
    return spec,tr,offline,old


def test_fresh_and_recovery_keep_exact_bank_bytes_no_fit(tmp_path,monkeypatch):
    spec,tr,offline,old=frozen(tmp_path)
    monkeypatch.setattr(previous,'fit',lambda *a:pytest.fail('No subclass reclustering'))
    bank=targets.prepare(spec,tr,offline)
    for name in targets.FILES:
        assert (Path(spec['root'])/'subclasses'/name).read_bytes()==(Path(old['root'])/'subclasses'/name).read_bytes()
    assert bank.manifest['content_hash']==spec['reuse']['subclasses']['manifest_hash']
    assert not bank.targets.flags.writeable
    np.testing.assert_array_equal(bank.identities,tr.identities)
    with pytest.raises(FileExistsError):targets.prepare(spec,tr,offline)
    path=Path(spec['root'])/'campaign_spec.json';write_immutable_json(path,spec)
    resumed=with_content_hash(dict(spec,root=str(tmp_path/'luka_fullstat_six_teacher_retry'),resume=dict(spec=file_ref(path))))
    Path(resumed['root']).mkdir()
    again=targets.prepare(resumed,tr,offline)
    np.testing.assert_array_equal(again.targets,bank.targets)


@pytest.mark.parametrize('bad',['validation','coordinate','row_order','prepared','parent_bytes','target_bytes','manifest_hash','files'])
def test_wrong_role_population_or_target_identity_stops(tmp_path,bad):
    spec,tr,offline,old=frozen(tmp_path)
    tr=deepcopy(tr)
    if bad=='validation':tr.role='validation'
    elif bad=='coordinate':tr.coordinate_name='U100'
    elif bad=='row_order':tr.identities=tr.identities[::-1].copy()
    elif bad=='prepared':tr.foundation_sha256='f'*64
    elif bad=='parent_bytes':Path(spec['reuse']['parent_spec']['path']).write_bytes(b'{}')
    elif bad=='target_bytes':(Path(old['root'])/'subclasses/targets.npz').write_bytes(b'bad')
    elif bad=='manifest_hash':spec['reuse']['subclasses']['manifest_hash']='f'*64
    elif bad=='files':spec['reuse']['subclasses']['files']=[]
    with pytest.raises((ValueError,PermissionError)):targets.prepare(spec,tr,offline)
    assert not (Path(spec['root'])/'subclasses').exists()


