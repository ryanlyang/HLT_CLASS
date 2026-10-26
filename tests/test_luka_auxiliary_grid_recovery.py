"""Exact post-publication K5 recovery; never accept arbitrary failed science."""
from copy import deepcopy
from contextlib import redirect_stdout
import getpass
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (canonical_json_bytes, load_json, load_npz_arrays,
    write_immutable_json, with_content_hash, deterministic_npz_bytes, sha256_file)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_auxiliary_grid import reference, reference_replay as replay, contracts as c, worker
from hlt_classification.luka_auxiliary_logits_strong import training as old_training, worker as old_worker
from hlt_classification.luka_auxiliary_logits_strong import submission as old_submission, contracts as old_contracts
from test_luka_auxiliary_logits_strong import fixture as strong_fixture
from test_luka_auxiliary_grid import bare, fixture as grid_fixture
from test_luka_embedding_aux import tiny


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def nonsymmetric_losses(report):
    """Synthetic epoch means exercise rounding lost in one-batch tiny fixtures."""
    result=deepcopy(report); rng=random.Random(27)
    for row in result['validation_history']:
        row.update(ce_loss=.4,weighted_ce_loss=.2)
        row['logit_losses']={t:rng.random() for t in c.TEACHERS}
        row['embedding_losses']={t:rng.random() for t in c.TEACHERS}
        row['weighted_logit_losses']={t:1.25*row['logit_losses'][t] for t in c.TEACHERS}
        row['weighted_embedding_losses']={t:1.25*row['embedding_losses'][t] for t in c.TEACHERS}
        row['total_loss']=row['weighted_ce_loss']+sum(row['weighted_logit_losses'].values())+sum(row['weighted_embedding_losses'].values())
    return with_content_hash(result)


def failed_fixture(tmp_path, monkeypatch, device='cpu'):
    tiny(monkeypatch)
    prior,tr,val=strong_fixture(tmp_path,monkeypatch)
    parent=tmp_path/'parent.json'; write_immutable_json(parent,dict(fixture=True))
    prior['source_commit']=replay.FAILED_COMMIT
    prior['reuse'].update(spec=file_ref(parent),anchor=prior['anchor'])
    prior=old_contracts.artifact('SPEC',**{k:v for k,v in prior.items() if k!='content_hash'})
    root=Path(prior['root']); path=root/'campaign_spec.json'; path.write_text(json.dumps(prior))
    jobs=dict(targets='21962580',k5='21962581',k10='21962582',report='21962583')
    monkeypatch.setattr(old_submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(old_worker,'execution',lambda _,task:jobs[task])
    old_worker.run(prior,'targets',device=device)
    original_train=old_training.train
    def train(*a,**kw):
        model,report,state=original_train(*a,**kw)
        return model,nonsymmetric_losses(report),state
    monkeypatch.setattr(old_training,'train',train)
    with pytest.raises(ValueError,match=replay.ERROR):
        old_worker.run(prior,'k5',device=device)
    folder=Path(prior['project_dir'])/'src/hlt_classification/luka_auxiliary_logits_strong'
    log='Traceback (most recent call last):\n'+''.join(
        f'  File "{folder/name}", line 100, in {function}\n' for name,function in
        [('worker.py','validate_task'),('worker.py','validate_fit'),('training.py','validate_report')])
    (root/'k5-21962581.out').write_text(log+'ValueError: '+replay.ERROR+'\n')
    states=dict(targets=('COMPLETED','0:0'),k5=('FAILED','1:0'))
    real_run=subprocess.run
    def run(args,**kwargs):
        if args[0]=='sacct':
            job=args[args.index('-j')+1]; task=next(k for k,v in jobs.items() if v==job)
            state,code=states[task]
            return SimpleNamespace(stdout=f'{job}|lukaas_{task}|{getpass.getuser()}|{prior["execution_site"]["account"]}|{state}|{code}\n')
        return real_run(args,**kwargs)
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setattr(reference,'validate_source_snapshot',lambda *a,**kw:None)
    def execute(path,project,program,marker):
        monkeypatch.setattr(sys,'argv',['replay',str(path)])
        output=io.StringIO()
        with redirect_stdout(output): exec(program,{})
        return file_ref(path),json.loads(output.getvalue().split(marker)[1])
    monkeypatch.setattr(reference,'replay',execute)
    reuse=c.artifact('REUSE',**deepcopy(prior['reuse']))
    return prior,path,reuse,tr,val,states


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_failed_worker_replay_copy_readback_and_old_bytes_unchanged(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    prior,path,reuse,tr,val,states=failed_fixture(tmp_path,monkeypatch,device)
    root=Path(prior['root']); original={p:sha256_file(p) for p in root.rglob('*') if p.is_file()}
    report=load_json(root/'k5/fit/training_report.json')
    bundle=load_json(root/'targets/bundle.json')
    original_validator=old_training.validate_report
    with pytest.raises(ValueError,match=replay.ERROR):
        original_validator(report,prior,'k5',bundle['content_hash'])
    restored=replay.ordered_report(report,c.TEACHERS)
    assert canonical_json_bytes(restored)==canonical_json_bytes(report)
    original_validator(restored,prior,'k5',bundle['content_hash'])
    imported=reference.import_reference(path,reuse)
    assert old_training.validate_report is original_validator
    assert imported['replay']['mode']=='recovered_postfit_json_order'
    assert imported['replay']['accounting']['k5']['state']=='FAILED'
    assert imported['replay']['repair']['reordered_epochs']
    spec=bare(tmp_path/'grid'); Path(spec['root']).mkdir()
    spec.update(reuse=reuse,reference=imported,anchor=prior['anchor'],counts=prior['counts'])
    spec=with_content_hash(spec)
    def no_training(*a,**kw): raise AssertionError('Recovery must not retrain')
    monkeypatch.setattr(old_training,'train',no_training)
    acquired=reference.acquire(spec); reference.check_acquisition(acquired,spec)
    assert acquired['mode']=='verified_private_copy'
    for t in c.TEACHERS:
        source=root/'targets/banks'/t/'targets.npz'
        copied=Path(spec['root'])/'targets/banks'/t/'targets.npz'
        assert source.read_bytes()==copied.read_bytes() and not source.samefile(copied)
    checked=reference.readback(spec,val,'300',device); reference.check_readback(checked,spec,'300')
    assert checked['training_repeated'] is False and checked['predictions_byte_identical'] is True
    assert original=={p:sha256_file(p) for p in root.rglob('*') if p.is_file()}
    for field,value in [('job_id','301'),('training_repeated',True),('predictions_byte_identical',False)]:
        bad=with_content_hash(dict(checked,**{field:value}))
        with pytest.raises(ValueError): reference.check_readback(bad,spec,'300')


@pytest.mark.parametrize('damage',['missing_receipt','checkpoint_bytes','prediction_bytes','costs_bytes',
    'wrong_log','other_error','running','oom','target_failed','no_order_bug','bad_schedule','bad_sum'])
def test_failed_reference_rejects_every_unexplained_failure(tmp_path,monkeypatch,damage):
    prior,path,reuse,tr,val,states=failed_fixture(tmp_path,monkeypatch)
    root=Path(prior['root']); original_validator=old_training.validate_report
    if damage=='missing_receipt': (root/'k5/receipt.json').unlink()
    elif damage in ('checkpoint_bytes','prediction_bytes','costs_bytes'):
        name={'checkpoint_bytes':'k5/fit/selected.pt','prediction_bytes':'k5/fit/validation_predictions.npz',
              'costs_bytes':'k5/costs.json'}[damage]
        target=root/name; target.write_bytes(target.read_bytes()+b'corrupt')
    elif damage=='wrong_log': (root/'k5-21962581.out').write_text('different failure')
    elif damage=='other_error':
        log=root/'k5-21962581.out'; log.write_text(log.read_text().replace(replay.ERROR,'CUDA out of memory'))
    elif damage=='running': states['k5']=('RUNNING','0:0')
    elif damage=='oom': states['k5']=('OUT_OF_MEMORY','0:125')
    elif damage=='target_failed': states['targets']=('FAILED','1:0')
    else:
        report=load_json(root/'k5/fit/training_report.json')
        if damage=='no_order_bug':
            for row in report['validation_history']:
                for key in ('logit_losses','embedding_losses','weighted_logit_losses','weighted_embedding_losses'):
                    row[key]={t:0. for t in c.TEACHERS}
                row['total_loss']=row['weighted_ce_loss']
        elif damage=='bad_schedule': report['validation_history'][0]['update']+=1
        else: report['validation_history'][0]['total_loss']+=.001
        (root/'k5/fit/training_report.json').write_text(json.dumps(with_content_hash(report),sort_keys=True))
    with pytest.raises((ValueError,FileNotFoundError,AssertionError)):
        reference.import_reference(path,reuse)
    assert old_training.validate_report is original_validator


@pytest.mark.parametrize('damage',['owner','name','account','duplicate','missing'])
def test_exact_original_accounting_identity(monkeypatch,damage):
    row=['21962581','lukaas_k5',getpass.getuser(),'reu-aisocial','FAILED','1:0']
    if damage in ('owner','name','account'): row[dict(owner=2,name=1,account=3)[damage]]='other'
    text='|'.join(row)+'\n'
    if damage=='duplicate': text+=text
    if damage=='missing': text=''
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
    with pytest.raises(ValueError): replay.accounting(dict(execution_site=dict(account='reu-aisocial')),'k5','21962581')


@pytest.mark.parametrize('damage',['commit','job','retirement'])
def test_recovery_cannot_be_applied_to_other_executions(monkeypatch,damage):
    from hlt_classification.data import cache_contracts
    from hlt_classification.luka_auxiliary_logits_strong import campaign as old_campaign
    spec=dict(root='/original',source_commit=replay.FAILED_COMMIT)
    jobs=dict(k5=replay.FAILED_JOB)
    if damage=='commit': spec['source_commit']='a'*40
    if damage=='job': jobs['k5']='21962582'
    monkeypatch.setattr(cache_contracts,'load_json',lambda _:spec)
    monkeypatch.setattr(old_campaign,'check',lambda _:None)
    monkeypatch.setattr(old_worker,'retired_evidence',lambda _:{} if damage=='retirement' else None)
    monkeypatch.setattr(old_submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(replay,'accounting',lambda *a:dict(state='FAILED',exit_code='1:0'))
    with pytest.raises(ValueError,match='registered failed K5'):
        replay.main('/original/campaign_spec.json')


@pytest.mark.parametrize('damage',['checkpoint','identities','metrics'])
def test_readback_rejects_wrong_native_fit_even_with_fresh_hashes(tmp_path,monkeypatch,damage):
    tiny(monkeypatch); spec,tr,val=grid_fixture(tmp_path,monkeypatch)
    ref=spec['reference']
    if damage=='checkpoint':
        state=torch.load(ref['checkpoint']['path'],weights_only=True)
        for name in state:
            if name.startswith('mod.fc.') and name.endswith('bias'): state[name][0]+=1.
        torch.save(state,ref['checkpoint']['path']); ref['checkpoint']=file_ref(ref['checkpoint']['path'])
    elif damage=='identities':
        arrays=load_npz_arrays(ref['predictions']['path']); arrays['identities']=arrays['identities'][::-1].copy()
        Path(ref['predictions']['path']).write_bytes(deterministic_npz_bytes(arrays))
        ref['predictions']=file_ref(ref['predictions']['path'])
    else: ref['validation']['macro_ovr_auc']+=.01
    with pytest.raises(ValueError): reference.readback(spec,val,'101','cpu')


def test_failed_readback_blocks_target_receipt_and_science(tmp_path,monkeypatch):
    tiny(monkeypatch); spec,tr,val=grid_fixture(tmp_path,monkeypatch)
    def fail(*a,**kw): raise ValueError('injected native readback failure')
    monkeypatch.setattr(reference,'readback',fail)
    with pytest.raises(ValueError,match='readback'): worker.run(spec,'targets',device='cpu')
    assert not (Path(spec['root'])/'targets/receipt.json').exists()
    with pytest.raises(FileNotFoundError): worker.run(spec,'k5_e10_plain',device='cpu')
    assert not (Path(spec['root'])/'k5_e10_plain').exists()


def test_recovery_versions_do_not_change_scientific_recipes():
    for kind in ('SPEC','REFERENCE','TASK','PLAN','ACQUISITION'):
        assert c.artifact(kind)['schema_version']==2
    assert c.recipe('k5_e10_rel')['schema_version']==1
    assert c.recipe('k5_e10_rel')['ce_coefficient']==.5
    assert len(c.tasks())==22 and c.AUTHORIZATION.endswith('_V2')
