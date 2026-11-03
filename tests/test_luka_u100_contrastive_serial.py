"""Stop semantics, unchanged updates, durable recovery and exact single-lane transition."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import numpy as np
import pytest
import torch
from test_luka_u100_contrastive import imported, sampler_fixture, spec as old_spec
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_u100_embedding import bank_fixture
from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_u100_contrastive import training as original, model, resources
from hlt_classification.luka_u100_contrastive_serial import contracts as c, training, source, campaign, submission, worker


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def history(values):
    return [dict(pass_number=i+1, validation=dict(macro_ovr_auc=x)) for i, x in enumerate(values)]


def spec(root):
    value = old_spec(root); value['recipe'] = c.recipe()
    return value


def test_policy_order_all_rows_no_weight_changes():
    assert len(c.ORDER) == len(set(c.ORDER)) == 20 and set(c.ORDER) == set(c.ARMS)
    assert c.ORDER[:5] == ('mix_l0p1_s0','hard_l0p1_s0','cross_l0p3_s0','mix_l0p3_s1','u100_w5_s1')
    assert c.recipe()['original'] == original.recipe()
    assert c.policy()['min_epochs'] == 15 and c.policy()['max_epochs'] == 30


@pytest.mark.parametrize('best,stop', [(12,17),(13,18),(14,19)])
def test_pasted_curve_stopping_boundaries(best, stop):
    values = [.8+i*.001 for i in range(best)] + [.7]*20
    assert c.stopping(history(values))['stop_epoch'] == stop
    assert c.stopping(history(values[:stop-1])) is None


def test_negative_recovery_never_stops_warmup_and_finite_poor_results_succeed():
    assert c.stopping(history([0.]*14)) is None
    assert c.stopping(history([0.]*15))['reason'] == 'patience'
    assert c.stopping(history([.5+i*.001 for i in range(30)]))['reason'] == 'max_epochs'
    for x in (float('nan'), float('inf'), -1, 1.1):
        with pytest.raises(ValueError): c.stopping(history([x]))


def test_min_delta_accumulates_against_last_meaningful_gain():
    values = [.5]+[.5+4e-6]*9+[.5+8e-6]*4
    decision = c.stopping(history(values+[.5+9e-6]))
    assert decision['stop_epoch'] == 15 and decision['last_meaningful_epoch'] == 1
    assert c.stopping(history(values+[.5+12e-6])) is None
    assert c.stopping(history(values+[.5+12e-6]+[.5]*5))['stop_epoch'] == 20


@pytest.mark.parametrize('arm', ['u100_w5_s0','mix_l0p3_s0'])
@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_installed_weaver_exact_original_updates_and_rng(tmp_path, monkeypatch, arm, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation','D000')
    sampler = None if arm.startswith('u100') else resources.SubsetSampler(sampler_fixture(device), np.arange(len(tr)))
    expected, selected, state = original.train(model.new_model(imported(), arm), tr, val, bank, sampler,
        imported=imported(), arm=arm, device=device, technical=True)
    rng = torch.get_rng_state().clone()
    actual, pick, native = training.train(model.new_model(imported(), arm), tr, val, bank, sampler,
        imported=imported(), arm=arm, device=device, technical=True)
    assert actual[0]['validation'] == expected[0]['validation'] and pick == selected
    for name in state:
        assert torch.equal(native[name], state[name]), name
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('arm', ['u100_w5_s1','mix_l0p3_s0'])
def test_successful_early_stop_native_reload_and_durable_bests(tmp_path, monkeypatch, arm):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation','D000')
    sampler = None if arm.startswith('u100') else resources.SubsetSampler(sampler_fixture(), np.arange(len(tr)))
    evaluate = training.evaluate_probabilities
    def poor(labels, probabilities):
        value = evaluate(labels, probabilities); value['macro_ovr_auc'] = 0.; return value
    monkeypatch.setattr(training, 'evaluate_probabilities', poor)
    folder = tmp_path/'fit'; s = spec(tmp_path)
    report = training.fit(folder, s, arm, model.new_model(imported(), arm), tr, val, bank, sampler,
                          device='cpu', key_hash='f'*64 if sampler else None)
    training.validate_fit(folder, s, arm)
    assert report['epochs'] == 15 and report['stopping']['reason'] == 'patience'
    assert report['selected_pass'] <= 15 and report['maximum_epochs'] == 30
    best = load_json(folder/'best'/f'epoch_{report["selected_pass"]:02d}.json')
    assert best['parents'] == dict(spec=s['content_hash'])
    assert (folder/best['checkpoint']['path']).is_file()
    assert not any(n.startswith(('matching.','predictor.')) for n in torch.load(folder/'selected.pt', weights_only=True))
    changed = deepcopy(report); changed['stopping']['reason'] = 'max_epochs'
    with pytest.raises(ValueError): training.validate_report(with_content_hash(changed), s, arm)
    (folder/'selected.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): training.validate_fit(folder, s, arm)


def test_partial_failure_retains_best_without_success_receipt(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation','D000'); arm='mix_l0p3_s0'
    sampler = resources.SubsetSampler(sampler_fixture(), np.arange(len(tr)))
    def fail(*args, **kwargs): raise RuntimeError('injected after best checkpoint publication')
    monkeypatch.setattr(training, 'diagnostic', fail)
    with pytest.raises(RuntimeError, match='injected'):
        training.fit(tmp_path/'fit', spec(tmp_path), arm, model.new_model(imported(), arm), tr, val,
                     bank, sampler, device='cpu', key_hash='f'*64)
    assert (tmp_path/'fit/best/epoch_01.pt').is_file()
    assert (tmp_path/'fit/best/epoch_01.json').is_file()
    assert not (tmp_path/'fit/receipt.json').exists()


def test_actual_best_smaller_than_patience_delta_is_kept(tmp_path, monkeypatch):
    tiny(monkeypatch); tr, bank, _ = bank_fixture(tmp_path); val = cache('validation','D000'); arm='u100_w5_s0'
    calls = []
    evaluate = training.evaluate_probabilities
    def metric(labels, probabilities):
        value = evaluate(labels, probabilities); calls.append(1)
        value['macro_ovr_auc'] = .5 + min(len(calls)-1, 14)*0.0000005
        return value
    monkeypatch.setattr(training, 'evaluate_probabilities', metric)
    folder=tmp_path/'fit'; folder.mkdir()
    rows, selected, _ = training.train(model.new_model(imported(),arm),tr,val,bank,None,
        imported=imported(),arm=arm,device='cpu',directory=folder,spec_hash='a'*64)
    assert len(rows) == selected == 15 and c.stopping(rows)['last_meaningful_epoch'] == 1
    assert (folder/'best/epoch_15.pt').exists()


def fixtures(root):
    old = old_spec(root/'original'); old.update(source_commit=source.PARENT_COMMIT,
        anchor=dict(evidence=dict(prepared_root='/prepared',raw_container='/raw')), original50k_rows=50000,
        recovery_anchors=dict(M0HLT=.9,OFFLINE=.95))
    jobs={a:str(21993055+i) for i,a in enumerate(c.ARMS)}; jobs['report']='21993075'
    done=['u100_w5_s0','mix_l0p3_s0','random_l0p3_s0','mix_l1_s0']
    proof=c.artifact('PARENT',spec=dict(path=str(root/'original/campaign_spec.json')),
        jobs=jobs,setup_jobs=dict(setup='21993008'),completed={a:dict(job_id=jobs[a]) for a in done},
        acceptance=dict(minutes={a:600 for a in c.ARMS}))
    request=c.artifact('REQUEST',root=str(root),project_dir='/project',source_commit='b'*40,source_snapshot={},
        parent=proof,storage_budget_bytes=8*2**30)
    states={j:dict(state='COMPLETED' if a in done or a=='setup' else 'RUNNING' if a=='kinmix_l0p3_s0' else 'PENDING',
                   exit_code='0:0') for a,j in (jobs|proof['setup_jobs']).items()}
    return old,proof,request,states


def test_pending_only_cancel_command_and_completed_identity_checks(tmp_path):
    old,proof,request,states=fixtures(tmp_path)
    command=submission.cancellation_command(proof['jobs'])
    assert command[:4] == ['scancel','--quiet','--me','--state=PENDING']
    assert set(command[4:]) == set(proof['jobs'].values())
    for invalid in ({},{'x':'all'},{'x':'1;rm'},{'x':'12','y':'12'}):
        with pytest.raises(ValueError): submission.cancellation_command(invalid)
    submission.validate_completed(proof,states)
    bad=deepcopy(states); bad[proof['jobs']['u100_w5_s0']]['exit_code']='1:0'
    with pytest.raises(ValueError): submission.validate_completed(proof,bad)
    bad=deepcopy(states); bad[proof['jobs']['hard_l1_s0']]['state']='COMPLETED'
    with pytest.raises(ValueError,match='receipt'): submission.validate_completed(proof,bad)


def test_automatic_transition_preserves_running_reuses_completion_and_serializes(tmp_path, monkeypatch):
    old,proof,request,states=fixtures(tmp_path)
    monkeypatch.setattr(campaign,'check_request',lambda r: old)
    monkeypatch.setattr(source,'validate_parent',lambda p: old)
    monkeypatch.setattr(submission,'check_site',lambda p: None)
    final=deepcopy(proof); final['completed']['kinmix_l0p3_s0']=dict(job_id=proof['jobs']['kinmix_l0p3_s0'])
    final=with_content_hash(final)
    monkeypatch.setattr(source,'inspect_parent',lambda p: final)
    current=deepcopy(states); calls=[]
    def accounting(jobs): return deepcopy(current)
    monkeypatch.setattr(submission,'accounting',accounting)
    def sleep(seconds):
        assert seconds == 30
        current[proof['jobs']['kinmix_l0p3_s0']]['state']='COMPLETED'
    monkeypatch.setattr(submission.time,'sleep',sleep)
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='scancel':
            for row in current.values():
                if row['state']=='PENDING': row['state']='CANCELLED'
            return SimpleNamespace(returncode=0)
        assert command[0]=='sbatch'; return SimpleNamespace(stdout=str(300000+len(calls)))
    monkeypatch.setattr(subprocess,'run',run)
    review=submission.review(request)
    submission.dispatch(request,plan_hash=review['content_hash'],authorization=c.AUTHORIZATION)
    s=load_json(tmp_path/'campaign_spec.json')
    assert len(s['remaining'])==15 and 'kinmix_l0p3_s0' in s['reused']
    assert s['remaining']==list(c.ORDER[:15]) and not (tmp_path/'setup').exists()
    assert len(calls)==17 and calls[0][0]=='scancel'
    graph=submission.plan(s)['commands']
    assert graph[0]['dependencies']==[]
    for prev,row in zip(graph,graph[1:]):
        assert row['dependencies']==[prev['task_id']]
        assert any(v.startswith('--dependency=afterany:') for v in row['command'])
        assert '--partition=debug' in row['command']
    assert len(submission.accepted_jobs(s))==16
    with pytest.raises(FileExistsError):
        submission.dispatch(request,plan_hash=review['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==17


def test_partial_submission_fails_closed(tmp_path,monkeypatch):
    old,proof,request,states=fixtures(tmp_path)
    s=dict(old,root=str(tmp_path),remaining=list(c.ORDER[:2]),parent=proof)
    submission.submit(s,execute=False)
    def fail(*args,**kwargs): raise RuntimeError('lost sbatch acknowledgement')
    monkeypatch.setattr(subprocess,'run',fail)
    with pytest.raises(RuntimeError): submission.submit(s,execute=True)
    with pytest.raises(FileExistsError): submission.submit(s,execute=True)


def test_parent_payload_and_measured_resource_binding(tmp_path, monkeypatch):
    old,proof,request,states=fixtures(tmp_path)
    old['source_snapshot']={}
    root=Path(old['root']); root.mkdir()
    write_immutable_json(root/'campaign_spec.json',old)
    write_immutable_json(root/'setup/acceptance.json',proof['acceptance'])
    lineage=[]
    for stage,jobs in (('setup',proof['setup_jobs']),('science',proof['jobs'])):
        for name,value in (('command_plan.json',{}),('submission_ledger.json',dict(jobs=jobs))):
            path=root/('submission_'+stage)/name; write_immutable_json(path,value); lineage.append(file_ref(path))
    payload=tmp_path/'external_keys.npz'; payload.write_bytes(b'original-published-keys')
    proof=c.artifact('PARENT',parents=dict(spec=old['content_hash']),spec=file_ref(root/'campaign_spec.json'),
        jobs=proof['jobs'],setup_jobs=proof['setup_jobs'],acceptance=proof['acceptance'],completed={},
        setup_files=[file_ref(root/'setup/acceptance.json'),file_ref(payload)],lineage_files=lineage)
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw: None)
    assert source.validate_parent(proof)==old
    bad=deepcopy(proof); bad['acceptance']['minutes']['mix_l1_s0']=1
    with pytest.raises(ValueError,match='resource'): source.validate_parent(with_content_hash(bad))
    payload.write_bytes(b'changed')
    with pytest.raises(ValueError): source.validate_parent(proof)


def test_combined_results_distinguish_original30_from_stopped_fits(tmp_path, monkeypatch):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    tr=cache('validation','D000'); labels=tr.labels
    arrays=dict(identities=tr.identities,labels=labels,
                probabilities=(np.eye(11,dtype=np.float32)[labels]*.9+.1/11))
    pred=tmp_path/'reference.npz'; atomic_publish_bytes(pred,deterministic_npz_bytes(arrays))
    subset=tmp_path/'subset.npz'; atomic_publish_bytes(subset,deterministic_npz_bytes(dict(indices=np.arange(11,dtype=np.int64))))
    manifest=tmp_path/'manifest.json'; write_immutable_json(manifest,dict(content_hash='u'*64))
    oldroot=tmp_path/'original'
    write_immutable_json(oldroot/'setup/bundle.json',dict(u100_manifest=file_ref(manifest),offline_manifest=file_ref(manifest)))
    oldreport=oldroot/'report.json'; write_immutable_json(oldreport,dict(selected_pass=13,epochs=30))
    arm='mix_l0p1_s0'; root=tmp_path/'new'; folder=root/arm
    folder.mkdir(parents=True); (folder/'receipt.json').write_text('{}')
    s=dict(root=str(root),content_hash='s'*64,recipe=c.recipe(),remaining=[arm,'hard_l0p1_s0'],
        counts=dict(validation=len(tr)),original50k_rows=11,
        imported=dict(files=dict(predictions=file_ref(pred),subset=file_ref(subset))),
        anchor=dict(evidence=dict(prepared='p'*64,models={n:dict(predictions=file_ref(pred)) for n in ('M0HLT','OFFLINE')})),
        parent=dict(completed=dict(u100_w5_s0=dict(report=file_ref(oldreport),predictions=file_ref(pred)))))
    monkeypatch.setattr(campaign,'check',lambda *a,**kw: dict(root=str(oldroot)))
    monkeypatch.setattr(worker,'receipt',lambda *a: dict(outputs=[dict(path='fit/'+p) for p in
        ('receipt.json','training_report.json','selected.pt','validation_predictions.npz')]))
    report=dict(parents=dict(spec=s['content_hash'],recipe=s['recipe']['content_hash'],prepared='p'*64,u100='u'*64,offline='u'*64),
                selected_pass=14,epochs=19,stopping=dict(reason='patience'))
    monkeypatch.setattr(training,'validate_fit',lambda *a: (report,arrays))
    result=worker.results(s); rows={r['model']:r for r in result['rows']}
    assert rows['u100_w5_s0']['regime']=='original30' and rows[arm]['regime']=='patience5'
    assert rows[arm]['epochs_run']==19 and rows[arm]['stop_reason']=='patience'
    assert rows[arm]['validation_delta_matched']==0 and result['pending']==['hard_l0p1_s0']


def test_accounting_rejects_wrong_owner_or_name(tmp_path, monkeypatch):
    user=submission.getpass.getuser()
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: SimpleNamespace(stdout=f'123|lukaoc_mix_l1_s0|{user}|COMPLETED|0:0\n'))
    assert submission.accounting(dict(mix_l1_s0='123'))['123']['state']=='COMPLETED'
    with pytest.raises(ValueError): submission.accounting(dict(hard_l1_s0='123'))
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: SimpleNamespace(stdout='123|lukaoc_mix_l1_s0|some_other_user|RUNNING|0:0\n'))
    with pytest.raises(ValueError): submission.accounting(dict(mix_l1_s0='123'))


def test_no_export_or_original_runtime_change_and_worker_environment():
    text=Path('src/hlt_classification/luka_u100_contrastive_serial/worker.py').read_text()
    assert 'targets.export(' not in text and 'resources.measure(' not in text
    for path in ('scripts/queue_luka_u100_contrastive_serial.sh','sbatch/run_luka_u100_contrastive_serial.sh'):
        value=Path(path).read_text()
        assert 'PYTHONNOUSERSITE=1' in value and '${CONDA_PREFIX}/lib' in value and 'atlas_kd_sporc' in value
    assert 'nohup' in Path('scripts/queue_luka_u100_contrastive_serial.sh').read_text()
