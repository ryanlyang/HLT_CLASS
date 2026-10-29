"""Physical ROOT/bank, installed-Weaver parity and complete production-fit tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import threads
from test_luka_neighborhood import population, particles
from test_luka_fullstat import fullstat, durable
from test_luka_fullstat_fine_tail import fixture as old_fixture
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_neighborhood.targets import descriptors
from hlt_classification.luka_fullstat import cache
from hlt_classification.luka_fullstat_neighborhood import (
    contracts as c, targets, banks, model, training, worker, campaign, source, submission)

STAGE = 'SMOOTH_W10'


def bare(path, export=False):
    old, caches = old_fixture(path, export=export)
    root = path/'luka_fullstat_neighborhood_results'; root.mkdir()
    reuse = old['reuse']
    reuse = c.artifact('REUSE', **{k:v for k,v in reuse.items() if k not in ('contract', 'schema_version', 'content_hash', 'parents', *c.FLAGS)})
    reuse = with_content_hash(dict(reuse,
        external_refs={t:old['external_refs'][t] for t in c.EXTERNAL},
        imported_banks={t:old['imported_banks'][t] for t in c.EXTERNAL},
        references=dict(reuse['references'], OLD_SPLITHEAD_DE000=reuse['references']['M0HLT'])))
    s = dict(root=str(root), target_root=str(path/'luka_fullstat_neighborhood_targets'),
        project_dir='/project', source_commit='f'*40, source_snapshot=dict(content_hash='c'*64),
        reuse=reuse, available_result_bytes=8*2**30, available_target_bytes=16*2**30, **campaign.policy(reuse))
    tr = caches['train']['D000']
    nt = sum(len(b.features) for b in tr.blocks)
    s.update(counts=old['counts'], resources=dict(rows=len(tr), tokens=nt, target_required_bytes=banks.size(len(tr), nt), conservative_host_bytes=8*2**30))
    s = c.artifact('SPEC', parents=dict(source='c'*64, reuse=reuse['content_hash']), **s)
    write_immutable_json(root/'campaign_spec.json', s)
    return s, caches


def block(cache, paired):
    values = [descriptors(h, o, smooth=True) for h,o in paired]
    return dict(identities=cache.identities, labels=cache.labels,
        offsets=np.r_[0,np.cumsum([len(x) for x in values])].astype(np.int64), values=np.concatenate(values))


def fake_chunks(tr):
    # Synthetic descriptors for the production-loop fixture, NOT remote evidence.
    offsets = np.r_[0,np.cumsum(np.concatenate([np.diff(b.offsets) for b in tr.blocks]))].astype(np.int64)
    return [dict(identities=tr.identities, labels=tr.labels, offsets=offsets,
        values=np.random.default_rng(4).uniform(0, 2, (int(offsets[-1]), 36)).astype(np.float32))]


def test_fixed_registration_no_control_or_preflight(tmp_path):
    s, _ = bare(tmp_path)
    r = c.recipe(STAGE)
    assert c.teachers(STAGE) == ['OFFLINE', 'UE100', 'DE066', 'DE033']
    assert r['weights'] == {t:10. for t in c.EXTERNAL} and r['logit_weights'] == {t:.25 for t in c.EXTERNAL}
    assert (r['ce_coefficient'],r['neighborhood_weight'],r['temperature']) == (.5,10.,2.)
    assert r['epochs'] == r['base']['maximum_passes'] == 20 and r['batch_size'] == 128
    assert not r['relational'] and not r['warm_start']
    assert s['preflight_jobs'] == 0 and s['setup_jobs'] == s['new_scientific_fits'] == 1
    p = submission.render(s)
    assert p['jobs'] == 2 and p['scientific_fits'] == 1 and p['preflight_jobs'] == 0
    assert '--partition=batch' in p['commands'][0]['command']
    assert not any(a.startswith('--gres') for a in p['commands'][0]['command'])
    assert '--dependency=afterok:${JOB_targets}' in p['commands'][1]['command']
    assert all('--no-requeue' in x['command'] for x in p['commands'])
    assert s['final_test_sealed'] and not s['final_test_accessed']
    for name in ('SMOOTH_W0', 'DE022', 'DE011', 'M0HLT'):
        with pytest.raises(ValueError): c.recipe(name)
    with pytest.raises(PermissionError): c.validate(with_content_hash(dict(c.artifact('X'), final_test_accessed=True)), 'X')


def test_target_bank_single_family_exact_normalization_and_join(tmp_path):
    tr, pairs = population()
    b = block(tr, pairs); root = tmp_path/'bank'
    s = dict(target_root=str(root), content_hash='b'*64)
    record = banks.build(root, [b], prepared=tr.foundation_sha256, source_hash=s['content_hash'],
        rows=len(tr), tokens=len(b['values']), available_bytes=8*2**30)
    bank = banks.NeighborhoodBank(s, tr)
    assert [r['path'] for r in record['outputs']] == ['index.npz', 'smooth.npy']
    ix = np.array([3,0,3], np.int64)
    actual = bank.batch(ix, 5)
    for j,i in enumerate(ix):
        expected = ((descriptors(*pairs[i], smooth=True).astype(np.float64)-bank.index['mean'])/bank.index['scale']).astype(np.float32)
        np.testing.assert_array_equal(actual[j,:3], expected)
    assert not actual[:,3:].any()
    with pytest.raises(ValueError): bank.batch(ix, 2)
    with pytest.raises(ValueError): bank.batch(np.array([-1],np.int64), 3)
    changed = deepcopy(tr); changed.labels[0] = (changed.labels[0]+1)%11
    with pytest.raises(ValueError): banks.NeighborhoodBank(s, changed)
    tr.role = 'validation'
    with pytest.raises(PermissionError): banks.NeighborhoodBank(s, tr)
    with open(root/'smooth.npy','ab') as f: f.write(b'changed')
    tr.role = 'train'
    with pytest.raises(ValueError): banks.NeighborhoodBank(s, tr)


def test_partial_targets_preserved_and_quotas_fail_closed(tmp_path):
    tr, pairs = population(); b = block(tr,pairs)
    args = dict(prepared='a'*64,source_hash='b'*64,rows=len(tr),tokens=len(b['values']),available_bytes=8*2**30)
    with pytest.raises(ValueError): banks.build(tmp_path/'partial', [], **args)
    assert (tmp_path/'partial/smooth.partial.npy').exists() and not (tmp_path/'partial/manifest.json').exists()
    with pytest.raises(FileExistsError): banks.build(tmp_path/'partial', [b], **args)
    args['available_bytes'] = 1
    with pytest.raises(OSError): banks.build(tmp_path/'small', [b], **args)
    assert not (tmp_path/'small').exists()


def test_real_fullstat_root_targets_and_sealed_roles(fullstat, durable, tmp_path, monkeypatch):
    import uproot
    container, _, _ = fullstat
    output, meta = durable
    s = dict(anchor=dict(evidence=dict(prepared=meta[0]['content_hash'], raw_container=str(container), cache_root=str(output))),
             preparation_workers=1, counts=dict(train=264))
    original = uproot.behaviors.TBranch.HasBranches.arrays
    by_path = {str(container/'raw'/r['path']):m for r,m in zip(meta[1]['files'],meta[2])}
    calls = []
    def guard(tree, expressions, *args, **kwargs):
        if any(str(e).startswith(('hlt_part_', 'part_')) for e in expressions):
            member = by_path[str(Path(tree.file.file_path))]
            roles = member['roles'][kwargs['entry_start']:kwargs['entry_stop']]
            assert not np.isin(roles, (2,3)).any()
            calls.append(1)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', guard)
    bb = list(targets.chunks(s,meta))
    assert calls and sum(len(b['labels']) for b in bb) == 264
    tr = cache.load(output,meta,role='train',coordinate='D000',max_ram_bytes=2**30)
    root = tmp_path/'bank'
    banks.build(root, bb, prepared=tr.foundation_sha256, source_hash='b'*64, rows=264,tokens=528,available_bytes=8*2**30)
    bank = banks.NeighborhoodBank(dict(target_root=str(root),content_hash='b'*64),tr)
    assert bank.batch(np.arange(264),2).shape == (264,2,36)
    s['preparation_workers'] = 2
    parallel = list(targets.chunks(s,meta))
    for a,b in zip(bb,parallel,strict=True):
        for k in a: np.testing.assert_array_equal(a[k],b[k])
    chunks = [c for c in meta[0]['chunks'] if c['role']=='validation']
    with pytest.raises(PermissionError): targets._work(chunks[0])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_installed_native_parity_rng_gradients_and_removal(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tr,_ = population(); raw = tr.batch(np.arange(4))
    torch.manual_seed(20); native = DelphesParticleTransformer().to(device).eval()
    after = torch.get_rng_state().clone()
    torch.manual_seed(20); student = model.Student(STAGE).to(device).eval()
    assert torch.equal(after,torch.get_rng_state())
    student.capture=True
    with parity_backend(device), torch.no_grad():
        for bf16 in ((False,True) if device=='cuda' else (False,)):
            torch.testing.assert_close(_forward(native,raw,device,bf16=bf16), _forward(student,raw,device,bf16=bf16),rtol=0,atol=0)
    student.train(); student.capture=True
    logits=_forward(student,raw,device,bf16=False)
    y=torch.from_numpy(raw['labels']).to(device)
    q={t:torch.randn(4,11,device=device).softmax(-1).requires_grad_(True) for t in c.EXTERNAL}
    z={t:torch.randn(4,128,device=device,requires_grad=True) for t in c.EXTERNAL}
    base,ce,kd,emb=training.objective(logits,y,student.auxiliary_logits,student.predictions,q,z,stage=STAGE)
    nb,_=model.neighborhood_loss(student.neighborhood_predictions,torch.ones_like(student.neighborhood_predictions),torch.from_numpy(raw['mask']).to(device))
    expected=torch.autograd.grad(.5*ce,tuple(student.mod.fc.parameters()),retain_graph=True)
    (base+10*nb).backward()
    for p,r in zip(student.mod.fc.parameters(),expected): torch.testing.assert_close(p.grad,r,rtol=0,atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
    assert all(v.grad is None for d in (q,z) for v in d.values())
    state=student.native_state(); assert not any(k.startswith(('predictors.','logit_heads.','neighborhood_head.')) for k in state)
    native.load_state_dict(state,strict=True); native.eval(); student.eval(); student.capture=False
    with parity_backend(device),torch.no_grad():
        torch.testing.assert_close(_forward(native,raw,device,bf16=False),_forward(student,raw,device,bf16=False),rtol=0,atol=0)


def test_neighborhood_only_loss_reaches_particles_not_classifier():
    tr,_=population(); raw=tr.batch(np.arange(3))
    m=model.Student(STAGE); m.capture=True
    _forward(m,raw,'cpu',bf16=False)
    loss,_=model.neighborhood_loss(m.neighborhood_predictions,torch.ones_like(m.neighborhood_predictions),torch.from_numpy(raw['mask']))
    loss.backward()
    assert all(p.grad is None for p in m.mod.fc.parameters())
    assert all(p.grad is None for p in m.mod.cls_blocks.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.mod.blocks.parameters())


def test_zero_weight_does_not_change_twenty_updates():
    tr,_=population(); raw=tr.batch(np.arange(11)); device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(9); left=model.Student(STAGE).to(device)
    torch.manual_seed(9); right=model.Student(STAGE).to(device)
    left.capture=right.capture=True
    cfg=c.recipe(STAGE)['base']
    optimizers=[training.optimizer(m,cfg) for m in (left,right)]
    rng=np.random.default_rng(40)
    q={t:torch.from_numpy(np.full((11,11),1/11,np.float32)).to(device) for t in c.EXTERNAL}
    z={t:torch.from_numpy(rng.normal(size=(11,128)).astype(np.float32)).to(device) for t in c.EXTERNAL}
    with parity_backend(device):
        for step in range(20):
            for i,(m,opt) in enumerate(zip((left,right),optimizers)):
                torch.manual_seed(100+step); m.train();opt.zero_grad(set_to_none=True)
                logits=_forward(m,raw,device,bf16=device=='cuda')
                base,*_=training.objective(logits,torch.from_numpy(raw['labels']).to(device),m.auxiliary_logits,m.predictions,q,z,stage=STAGE)
                if i:
                    extra,_=model.neighborhood_loss(m.neighborhood_predictions,torch.ones_like(m.neighborhood_predictions),torch.from_numpy(raw['mask']).to(device))
                    base=base+0*extra
                base.backward();opt.step()
    for k,v in left.native_state().items(): torch.testing.assert_close(v,right.native_state()[k],rtol=0,atol=0)


def test_source_scope_budgets_and_cli(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in source.REQUIRED)
    assert source.compatible_changes(source.ADDITIONS,source.ADDITIONS)
    assert not source.compatible_changes({'src/hlt_classification/luka_neighborhood/targets.py'},set())
    compile(source.REPLAY,'replay','exec')
    assert "('OFFLINE','UE100','DE066','DE033')" in source.REPLAY
    assert "worker.completed(s,'DE000')" in source.REPLAY
    subprocess.run([sys.executable,'scripts/luka_fullstat_neighborhood.py','--help'],check=True,capture_output=True)
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    for name in source.ENTRYPOINTS:
        if name.endswith('.sh'): subprocess.run([bash,'-n',str(root/name)],check=True,capture_output=True)
    s,_=bare(tmp_path)
    m=dict(content_hash=s['anchor']['evidence']['prepared'],summaries=s['reuse']['summaries'],
        chunks=[dict(role='train',views=dict(D000=dict(rows=s['counts']['train'],particles=100)))])
    budget=campaign.resource_budget(s,m)
    assert budget['target_required_bytes']==banks.size(s['counts']['train'],100)
    m['chunks'][0]['views']['D000']['particles']=10**10
    with pytest.raises(MemoryError): campaign.resource_budget(s,m)
    bad=deepcopy(s);bad['available_result_bytes']=1
    with pytest.raises(OSError): campaign.storage(bad)
    bad=deepcopy(s);bad['available_target_bytes']=1
    with pytest.raises(OSError): campaign.storage(bad)
    with pytest.raises(PermissionError): campaign.separate_outputs(s['root'],s['root'],s['reuse'],s['project_dir'])


def test_completed_snapshot_survives_publication_check_failure(tmp_path,monkeypatch):
    s,_=bare(tmp_path)
    fake=dict(validation={'synthetic':True})
    def train(*args,**kwargs):
        state={'synthetic_tensor':torch.ones(1)}
        kwargs['snapshot'](fake,state)
        return fake,state
    monkeypatch.setattr(training,'train',train)
    def reject(*a): raise ValueError('synthetic report-check failure')
    monkeypatch.setattr(worker,'report_valid',reject)
    directory=Path(s['root'])/'train_SMOOTH_W10'
    with pytest.raises(ValueError,match='report-check'):
        worker.fit(directory,s,STAGE,None,None,SimpleNamespace(manifest={'content_hash':'a'*64}),device='cpu')
    assert (directory/'completed_native.pt').exists()
    assert load_json(directory/'completed_training.json')==fake
    assert not (directory/'receipt.json').exists()


def test_complete_twenty_epoch_worker_and_readback(tmp_path,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('CUDA miniature requires GPU')
    s,cs=bare(tmp_path,export=True); tr,val=cs['train']['D000'],cs['validation']['D000']
    root=Path(s['root'])
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a,**kw:os.environ['SLURM_JOB_ID'])
    monkeypatch.setattr(worker,'metadata',lambda _: (dict(content_hash=tr.foundation_sha256,summaries=s['reuse']['summaries']),None,None))
    monkeypatch.setattr(worker,'data',lambda s,r,co,**kw:cs[r][co])
    monkeypatch.setattr(targets,'chunks',lambda *a:iter(fake_chunks(tr)))
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:{'targets':'201','train_SMOOTH_W10':'202'})
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'validation_arrays',lambda _:dict(identities=val.identities,labels=val.labels))
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,gpu_peak_bytes=max(1,torch.cuda.max_memory_allocated()),rss_bytes=2**30))
    monkeypatch.setenv('SLURM_JOB_ID','201'); worker.run(s,'targets')
    monkeypatch.setenv('SLURM_JOB_ID','202'); worker.run(s,'train_SMOOTH_W10')
    receipt=worker.completed(s,STAGE)
    assert receipt['stage']==STAGE
    report=load_json(root/'train_SMOOTH_W10/training_report.json')
    assert report['passes']==20 and len(report['validation_history'])==20
    assert report==load_json(root/'train_SMOOTH_W10/completed_training.json')
    training.validate_report(json.loads(json.dumps(report,sort_keys=True)),s,STAGE,receipt['bank_hash'])
    altered=deepcopy(report);altered['validation_history'][0]['neighborhood_weight']=0
    with pytest.raises(ValueError): training.validate_report(with_content_hash(altered),s,STAGE,receipt['bank_hash'])
    result=worker.results(s); assert result['pending']==[] and len(result['rows'])==4
    with pytest.raises(FileExistsError): worker.run(s,'train_SMOOTH_W10')
