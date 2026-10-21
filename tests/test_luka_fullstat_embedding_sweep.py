"""FullStat20 standalone CE sweep: losses, lineage, real CUDA miniature and guards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_attention import cache, parent_nodes
from test_luka_embedding_aux import tiny, threads
from test_luka_fullstat import fullstat, durable
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, write_immutable_json, with_content_hash,
    atomic_publish_bytes, deterministic_npz_bytes, load_npz_arrays)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_fullstat_embedding_sweep import (
    contracts as c, banks, campaign, source, submission, worker, preflight, training)


def scheduler_gate(root):
    snapshot = dict(content_hash='c'*64)
    anchor = c.artifact('ANCHOR', spec=dict(path='/original/campaign_spec.json'))
    return c.artifact('GATE', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir='/project', source_commit='b'*40, source_snapshot=snapshot,
        anchor=anchor, available_result_bytes=12*2**30, **campaign.registration(dict(nodes=parent_nodes())))


def fixture(root, *, export=True):
    root.mkdir()
    tr, val, off = cache(coordinate='D000'), cache('validation', coordinate='D000'), cache(coordinate='OFFLINE')
    teacher = DelphesParticleTransformer()
    save_state(root/'teacher.pt', teacher.state_dict())
    a = dict(identities=val.identities, labels=val.labels,
             probabilities=np.full((len(val), 11), 1/11, np.float32))
    atomic_publish_bytes(root/'parent_predictions.npz', deterministic_npz_bytes(a))
    report = with_content_hash(dict(validation=evaluate_probabilities(a['labels'], a['probabilities']), selected_pass=14))
    write_immutable_json(root/'parent_report.json', report)
    refs = dict(checkpoint=file_ref(root/'teacher.pt'), report=file_ref(root/'parent_report.json'),
                predictions=file_ref(root/'parent_predictions.npz'))
    anchor = c.artifact('ANCHOR', spec=dict(path='/parent/campaign_spec.json'), evidence=dict(
        prepared=tr.foundation_sha256, cache_root='/cache', foundation_root='/foundation',
        raw_container='/raw', models={n: refs for n in c.REFERENCES},
        environment=dict(fixture=True), cudnn_v8_disabled='unset'))
    gate = scheduler_gate(root)
    gate = with_content_hash(dict(gate, anchor=anchor,
        parents=dict(source=gate['source_snapshot']['content_hash'], anchor=anchor['content_hash']),
        counts=dict(train=len(tr), validation=len(val))))
    write_immutable_json(root/'gate_spec.json', gate)
    if export:
        banks.export(root/'targets', teacher, off, refs=refs, device='cpu')
    return gate, tr, val, off


def test_science_is_seven_matched_CE_fits_without_KD_or_ladder():
    reg = campaign.registration(dict(nodes=parent_nodes()))
    assert list(c.WEIGHTS.values()) == [0., 2., 10., 50., 100., 500., 1000.]
    seeds = set()
    for arm, node in reg['nodes'].items():
        r = c.recipe(c.WEIGHTS[arm])
        assert r['epochs'] == 20 and r['batch_size'] == 128
        assert r['ce_coefficient'] == 1 and r['kd_coefficient'] == 0
        assert r['base_usage'] == 'optimizer_and_schedule_only_not_KD_loss'
        assert node['teacher'] is None and node['context_coordinate'] is None
        assert node['coordinate'] == 'D000' and node['deployable']
        seeds.add((node['initialization_seed'], node['sampler_seed']))
        assert r['final_test_sealed'] and not r['final_test_accessed']
        assert not r['warm_start'] and not r['attention']
        assert r['embedding_teacher'] == 'OFFLINE'
    assert len(seeds) == 1 and len(c.tasks()) == 8
    assert reg['counts'] == dict(train=3000000, validation=500000, final_test=1500000, unused=105325)
    assert c.tasks()[-1]['dependencies'] == list(c.WEIGHTS)
    assert 'COARSE_D033' not in source.PARENT_CHECK
    assert reg['automatic_followup'] is False  # Separate explicit FOLLOWUP authorization.


@pytest.mark.parametrize('weight', c.WEIGHTS.values())
def test_loss_exact_and_target_frozen(weight):
    logits = torch.randn(3, 11, requires_grad=True)
    pred = torch.randn(3, 128, requires_grad=True)
    target = torch.randn(3, 128, requires_grad=True)
    labels = torch.tensor([1, 3, 5])
    loss, ce, aux = training.objective(logits, labels, pred, target, coefficient=weight)
    torch.testing.assert_close(loss, ce+weight*aux, rtol=0, atol=0)
    loss.backward()
    assert target.grad is None and logits.grad is not None
    assert (pred.grad is None) == (weight == 0)
    if weight:
        expected = weight*(pred.detach()-target.detach()).clamp(-1, 1)/pred.numel()
        torch.testing.assert_close(pred.grad, expected)


def test_registered_weights_and_nonfinite_inputs_fail_closed():
    x, y, p, t = torch.zeros(2,11), torch.tensor([0,1]), torch.zeros(2,128), torch.ones(2,128)
    for value in (-1., .3, 1., True, float('inf')):
        with pytest.raises(ValueError): training.objective(x,y,p,t,coefficient=value)
    with pytest.raises(ValueError): training.objective(x,y,p,t*float('nan'),coefficient=1000.)


def test_identical_initial_weights_including_predictor(monkeypatch):
    tiny(monkeypatch)
    ns = c.nodes(dict(nodes=parent_nodes()))
    states = [new_model(node).state_dict() for node in ns.values()]
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        assert all(torch.equal(v, states[0][k]) for k,v in state.items())


def test_bank_train_only_normalization_and_exact_join(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g, tr, val, off = fixture(tmp_path/'g')
    bank = banks.load(g, tr)
    assert not hasattr(bank, 'probabilities')
    assert bank.targets.shape == (len(tr),128) and not bank.targets.flags.writeable
    np.testing.assert_allclose(bank.targets.mean(0),0,atol=1e-6)
    banks.check_join(bank,tr)
    changed = deepcopy(tr); changed.identities = tr.identities[::-1].copy()
    with pytest.raises(ValueError, match='join'): banks.load(g,changed)
    for cache_ in (val,off):
        with pytest.raises(PermissionError): banks.load(g,cache_)
    for role, coordinate in [('validation','OFFLINE'), ('final_test','D000'), ('train','D033')]:
        with pytest.raises(PermissionError): worker.data(g,role,coordinate)
    with pytest.raises(PermissionError):
        banks.export(tmp_path/'bad',DelphesParticleTransformer(),val,
                     refs=g['anchor']['evidence']['models']['OFFLINE'],device='cpu')


@pytest.mark.parametrize('key,value', [('role','validation'), ('teacher','COARSE_D033'),
    ('coordinate','U100'), ('width',127), ('rows',21), ('final_test_accessed',True)])
def test_resigned_wrong_bank_rejected(tmp_path, monkeypatch,key,value):
    tiny(monkeypatch)
    g,tr,*_ = fixture(tmp_path/'g')
    path = Path(g['root'])/'targets/manifest.json'
    path.write_text(json.dumps(with_content_hash(dict(load_json(path),**{key:value}))))
    with pytest.raises((ValueError,PermissionError)): banks.load(g,tr)


def test_target_corruption_wrong_checkpoint_and_ce_parity(tmp_path,monkeypatch):
    tiny(monkeypatch)
    g,tr,val,_ = fixture(tmp_path/'g')
    bank = banks.load(g,tr)
    assert preflight.zero_parity(tr,val,bank,g['nodes']['w0'],'cpu')
    payload = Path(g['root'])/'targets/targets.npz'
    payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValueError): banks.load(g,tr)


def test_full20_reports_match_each_arm_and_strip_predictor(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g,tr,val,_ = fixture(tmp_path/'g')
    bank = banks.load(g,tr)
    for arm in c.WEIGHTS:
        scoped = c.arm_gate(g,arm)
        worker.fit(tmp_path/arm,scoped,tr,val,bank,device='cpu')
        r = worker.validate_fit(tmp_path/arm,scoped,bank.manifest['content_hash'])
        assert r['passes'] == 20 and r['selected_state'] == 'native_backbone_only'
        assert all('logit_kd' not in row for row in r['validation_history'])
        state = torch.load(tmp_path/arm/'selected.pt',weights_only=True)
        assert not any(k.startswith('predictor.') for k in state)
        wrong = with_content_hash(dict(r,technical=True))
        with pytest.raises(ValueError): worker.report_valid(wrong,scoped,bank.manifest['content_hash'])


def test_poor_validation_never_stops_training_and_selected_weights_restored(tmp_path, monkeypatch):
    tiny(monkeypatch)
    g,tr,val,_ = fixture(tmp_path/'g')
    bank = banks.load(g,tr)
    original_evaluate = training.evaluate_probabilities
    original_native_state = new_model(g['nodes']['w1000']).__class__.native_state
    selected_states = []
    calls = []
    def poor_metrics(labels, probabilities):
        metrics = original_evaluate(labels, probabilities)
        calls.append(True)
        metrics['macro_ovr_auc'] = .1 if len(calls) == 3 else .01
        return metrics
    def remember_native(model):
        state = original_native_state(model)
        selected_states.append({k:v.clone() for k,v in state.items()})
        return state
    monkeypatch.setattr(training,'evaluate_probabilities',poor_metrics)
    monkeypatch.setattr(new_model(g['nodes']['w1000']).__class__,'native_state',remember_native)
    report,state = training.train(tr,val,bank=bank,node=g['nodes']['w1000'],device='cpu')
    assert len(calls) == report['passes'] == 20
    assert report['selected_pass'] == 3 and report['validation']['macro_ovr_auc'] == .1
    assert all(torch.equal(v,selected_states[-1][k]) for k,v in state.items())
    worker.report_valid(report,c.arm_gate(g,'w1000'),bank.manifest['content_hash'])


def test_real_root_prepared_cache_to_offline_targets(durable,tmp_path,monkeypatch):
    tiny(monkeypatch)
    output,meta = durable
    gate = dict(memory_mb=180000,counts=meta[1]['counts'],
        anchor=dict(evidence=dict(prepared=meta[0]['content_hash'],cache_root=str(output))))
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    tr=worker.data(gate,'train','D000');off=worker.data(gate,'train','OFFLINE')
    val=worker.data(gate,'validation','D000')
    assert len(tr)==len(off)==264 and len(val)==66
    np.testing.assert_array_equal(tr.identities,off.identities)
    np.testing.assert_array_equal(tr.labels,off.labels)
    teacher=DelphesParticleTransformer()
    save_state(tmp_path/'teacher.pt',teacher.state_dict())
    write_immutable_json(tmp_path/'report.json',with_content_hash(dict(selected_pass=14)))
    refs=dict(checkpoint=file_ref(tmp_path/'teacher.pt'),report=file_ref(tmp_path/'report.json'))
    banks.export(tmp_path/'targets',teacher,off,refs=refs,device='cpu')
    bank=banks.Bank(tmp_path/'targets',tr,refs=refs)
    assert bank.targets.shape==(264,128)
    np.testing.assert_allclose(bank.targets.mean(0),0.,atol=1e-6)


def test_resource_gate_uses_time_memory_storage_not_scores(tmp_path):
    g = scheduler_gate(tmp_path)
    m = dict(gpu=dict(total_memory_bytes=48*2**30), peak_rss_bytes=2**30,
             gpu_peak_bytes=2**30,bank_bytes=2**30,fit_bytes=10**7,
             timings=dict(authentication=10.,load_HLT_targets=60.,train_epoch=1000.,
                          validation_epoch=50.,checkpoint_reporting=5.))
    assert 720 <= preflight.admission(g,m) <= 2880
    for key,value in [('peak_rss_bytes',180000*1024**2),('gpu_peak_bytes',48*2**30),
                       ('fit_bytes',2**30),('bank_bytes',20*2**30)]:
        with pytest.raises(ValueError): preflight.admission(g,dict(m,**{key:value}))
    for key,value in [('train_epoch',9000.),('train_epoch',float('nan')),('validation_epoch',0.)]:
        bad=deepcopy(m);bad['timings'][key]=value
        with pytest.raises(ValueError): preflight.admission(g,bad)


def test_source_guard_and_entrypoint_syntax(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    def git(cmd,**kw):
        return SimpleNamespace(stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else
                               '\n'.join(source.ADDITIONS|source.ENTRYPOINTS))
    monkeypatch.setattr(source.subprocess,'run',git)
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    for path in source.REQUIRED:
        assert (project/path).is_file(),path
    monkeypatch.setattr(source.subprocess,'run',lambda cmd,**kw:SimpleNamespace(
        stdout='\n'.join(source.REQUIRED) if 'ls-files' in cmd else 'src/hlt_classification/luka_fullstat/recipe.py'))
    with pytest.raises(ValueError):source.current(project,'b'*40)


def test_queue_worker_cli_and_runbook_syntax():
    project=Path(__file__).resolve().parents[1]
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():
        bash=shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    for file in ('scripts/queue_luka_fullstat_embedding_sweep.sh',
                 'sbatch/run_luka_fullstat_embedding_sweep.sh'):
        subprocess.run([str(bash),'-n',file],cwd=project,check=True,capture_output=True,timeout=30)
    runbook=(project/'docs/LUKA_FULLSTAT_EMBEDDING_SWEEP_RUNBOOK.md').read_text(encoding='utf-8')
    blocks=re.findall(r'^```bash\n(.*?)^```',runbook,re.M|re.S)
    assert len(blocks)==3
    for block in blocks:
        subprocess.run([str(bash),'-n'],input=block,cwd=project,check=True,text=True,
                       capture_output=True,timeout=30)
    result=subprocess.run([sys.executable,'-B','scripts/luka_fullstat_embedding_sweep.py','--help'],
        cwd=project,check=True,text=True,capture_output=True,timeout=60)
    assert 'followup' in result.stdout and 'create-gate' in result.stdout
    for path in source.REQUIRED:
        file=project/path
        text=file.read_text(encoding='utf-8')
        assert all(line==line.rstrip() for line in text.splitlines()),path
        if file.suffix=='.md':
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (file.parent/link).is_file(),(path,link)


def test_actual_installed_cuda_preflight_then_all_seven_full20_workers(tmp_path,monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Real Oscar preflight still required')
    g,tr,val,off=fixture(tmp_path/'g',export=False)
    monkeypatch.setattr(campaign,'check_gate',lambda _:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'77')
    monkeypatch.setattr(worker,'data',lambda g,r,co,**kw:off if co=='OFFLINE' else tr if r=='train' else val)
    meta=(dict(content_hash=tr.foundation_sha256,
        summaries=dict(train=dict(D000=dict(maximum=4),OFFLINE=dict(maximum=4)),
                       validation=dict(D000=dict(maximum=4)))),None,None)
    monkeypatch.setattr(worker,'metadata',lambda _:meta)
    monkeypatch.setattr(preflight,'installed_environment',lambda:g['anchor']['evidence']['environment'])
    monkeypatch.setattr(preflight,'gpu_identity',lambda:dict(name='FIXTURE NVIDIA L40S',
                         total_memory_bytes=48*2**30,compute_capability=[8,9]))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.delenv('TORCH_CUDNN_V8_API_DISABLED',raising=False)
    def memory(_):
        return dict(rss_bytes=2**30,peak_rss_bytes=2**30,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()))
    monkeypatch.setattr(preflight,'memory_snapshot',memory)
    monkeypatch.setattr(worker,'memory_snapshot',memory)
    report=preflight.run(g)
    assert report['technical_checks_passed'] and report['zero_aux_native_ce_parity']
    assert set(report['measurements'])==set(c.WEIGHTS)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(preflight='77'))
    monkeypatch.setattr(campaign,'preflight_accounting',lambda _:None)
    spec=campaign.science(g)
    # Partial results cannot silently use the historical M0HLT as matched control.
    worker.run(spec,'w2')
    partial=worker.results(spec,g)
    assert all(r['recovery'] is None for r in partial['rows']) and 'w0' in partial['pending']
    for arm in c.WEIGHTS:
        receipt=worker.run(spec,arm)
        assert worker.run(spec,arm)==receipt
    result=worker.run(spec,'aggregate')
    assert worker.run(spec,'aggregate')==result
    summary=worker.results(spec,g)
    assert summary['pending']==[] and len(summary['rows'])==9
    assert not summary['final_test_accessed'] and summary['final_test_sealed']
    w0=next(r for r in summary['rows'] if r['node_id']=='w0')
    assert w0['delta_auc_vs_matched_CE']==0
    for change in (dict(technical_checks_passed=False),dict(full_population_training_measured=False),
                   dict(train_minutes={arm:2879 for arm in c.WEIGHTS})):
        with pytest.raises(ValueError):preflight.validate_report(g,with_content_hash(dict(report,**change)))
    bad=deepcopy(report);bad['outputs']=[r for r in bad['outputs'] if r['path']!='preflight/runtime.json']
    with pytest.raises(ValueError):preflight.validate_report(g,with_content_hash(bad))
    pred=Path(spec['root'])/'w10/validation_predictions.npz'
    pred.write_bytes(pred.read_bytes()+b'corrupt')
    with pytest.raises(ValueError):worker.results(spec,g)
