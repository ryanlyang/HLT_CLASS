"""Durable epoch commits and native selected-result authentication."""
import os
from pathlib import Path
import tempfile
import torch
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.jetclass2_delphes.runner import selection_key
from . import contracts as c,training


def save_state(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.link(temp,path)
    finally:os.unlink(temp)


def save_epoch(folder,epoch,row,state,parents):
    target=Path(folder)/'epochs'/f'{epoch:02d}';target.mkdir(parents=True,exist_ok=False)
    save_state(target/'state.pt',state)
    meta.write_immutable_json(target/'report.json',c.artifact('EPOCH',parents=parents,epoch=epoch,row=row))
    receipt=c.artifact('EPOCH_COMMIT',parents=parents,epoch=epoch,
        state=meta.file_ref(target/'state.pt'),report=meta.file_ref(target/'report.json'))
    meta.write_immutable_json(target/'receipt.json',receipt)
    return receipt


def epoch_refs(folder,parents,stage,validation_rows,sampler=None,start_epoch=1):
    refs=[];gap=False
    for epoch in range(start_epoch,c.EPOCHS+1):
        path=Path(folder)/'epochs'/f'{epoch:02d}'/'receipt.json'
        if not path.exists():gap=True;continue
        if gap:raise ValueError('Noncontiguous committed epoch history')
        receipt=meta.load_json(path);c.validate(receipt,'EPOCH_COMMIT',parents)
        if receipt['epoch']!=epoch:raise ValueError('Epoch binding differs')
        meta.validate_file_ref(receipt['state'])
        report=meta.load_json(meta.validate_file_ref(receipt['report']));c.validate(report,'EPOCH',parents)
        if report['epoch']!=epoch:raise ValueError('Epoch report cursor differs')
        training.check_row(report['row'],epoch,stage,validation_rows,sampler)
        refs.append(meta.file_ref(path))
    return refs


def restore(refs,parents,stage,validation_rows,sampler):
    history=[];receipts=[]
    for epoch,ref in enumerate(refs,1):
        expected=parents[epoch-1] if isinstance(parents,list) else parents
        receipt=meta.load_json(meta.validate_file_ref(ref));c.validate(receipt,'EPOCH_COMMIT',expected)
        row=meta.load_json(meta.validate_file_ref(receipt['report']));c.validate(row,'EPOCH',expected)
        if receipt['epoch']!=epoch or row['epoch']!=epoch:raise ValueError('Continuation cursor differs')
        training.check_row(row['row'],epoch,stage,validation_rows,sampler)
        history.append(row['row']);receipts.append(receipt)
    if not history:return None
    selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))['pass_number']
    state=torch.load(meta.validate_file_ref(receipts[-1]['state']),map_location='cpu',weights_only=True)
    best=state if selected==len(history) else torch.load(meta.validate_file_ref(receipts[selected-1]['state']),map_location='cpu',weights_only=True)
    return dict(state=state,best=best,history=history)


def completed(spec,stage,*,account=True):
    from . import submission,campaign
    if stage not in c.STAGES:raise ValueError('Unregistered completed stage')
    if stage in spec['completed']:
        ref=spec['completed'][stage]
        origin=meta.load_json(meta.validate_file_ref(ref['spec']));campaign.check(origin,live_source=False)
        result=completed(origin,stage,account=account)
        if meta.file_ref(result['folder']/'task_receipt.json')!=ref['receipt']:raise ValueError('Completed reuse binding differs')
        return result
    folder=Path(spec['root'])/('train_'+stage);path=folder/'task_receipt.json'
    receipt=meta.load_json(path);c.validate(receipt,'TASK_RECEIPT')
    if receipt['stage']!=stage or receipt['parents']['spec']!=spec['content_hash'] or receipt['population']!=spec['population']['content_hash']:
        raise ValueError('Task population/spec differs')
    jobs=submission.accepted_jobs(spec,partial=True)
    if receipt['job_id']!=jobs['train_'+stage]:raise ValueError('Task scheduler binding differs')
    if account:submission.accounting(spec,'train_'+stage)
    expected={'checkpoint','report','predictions','metrics','startup','view','binding'}|({'bank'} if stage!='DE000' else set())
    if set(receipt['files'])!=expected:raise ValueError('Result inventory differs')
    for ref in receipt['files'].values():meta.validate_file_ref(ref)
    binding=meta.load_json(receipt['files']['binding']['path'])
    c.validate(binding,'FIT_BINDING',dict(spec=spec['content_hash']))
    parents=dict(spec=spec['content_hash'],population=spec['population']['content_hash'])
    for teacher in c.TEACHERS[stage]:
        origin=spec
        while teacher in origin['completed']:
            reuse=origin['completed'][teacher]
            origin=meta.load_json(meta.validate_file_ref(reuse['spec']))
            meta.validate_file_ref(reuse['receipt'])
        previous=meta.load_json(Path(origin['root'])/('train_'+teacher)/'task_receipt.json')
        c.validate(previous,'TASK_RECEIPT')
        if previous['stage']!=teacher or previous['parents']['spec']!=origin['content_hash']:
            raise ValueError('Teacher receipt binding differs')
        parents['teacher_'+teacher]=previous['content_hash']
    if binding['stage']!=stage or binding['fit_parents']!=parents or receipt['fit_parents']!=parents:
        raise ValueError('Fit teacher lineage differs')
    startup=meta.load_json(receipt['files']['startup']['path'])
    c.validate(startup,'STARTUP',dict(spec=spec['content_hash']))
    required=('installed_weaver_parity','finite_connected_gradients','classifier_ce_only',
              'native_deployment_parity','full_support','probe_discarded')
    if (startup['stage']!=stage or any(startup.get(k) is not True for k in required)
            or startup.get('matching_serial_parity')!=(stage=='DE000')):
        raise ValueError('Completed task lacks integrated acceptance')
    report=meta.load_json(receipt['files']['report']['path'])
    training.validate_report(report,stage,receipt['fit_parents'],c.COUNTS['validation'])
    if receipt['parents']['report']!=report['content_hash'] or receipt['selected_pass']!=report['selected_pass']:
        raise ValueError('Report parent/selection differs')
    if len(receipt['inherited_epochs'])+len(receipt['new_epochs'])!=c.EPOCHS:
        raise ValueError('Completed task lacks twenty committed epochs')
    inherited=spec['continuations'].get(stage,{})
    if receipt['inherited_epochs']!=inherited.get('epochs',[]):
        raise ValueError('Completed continuation differs')
    refs=receipt['inherited_epochs']+receipt['new_epochs']
    epoch_parents=inherited.get('epoch_parents',[])+[parents]*len(receipt['new_epochs'])
    for epoch,(ref,ep) in enumerate(zip(refs,epoch_parents,strict=True),1):
        commit=meta.load_json(meta.validate_file_ref(ref));c.validate(commit,'EPOCH_COMMIT',ep)
        saved=meta.load_json(meta.validate_file_ref(commit['report']));c.validate(saved,'EPOCH',ep)
        if (commit['epoch']!=epoch or saved['epoch']!=epoch
                or saved['row']!=report['validation_history'][epoch-1]):
            raise ValueError('Committed epoch/report history differs')
    return dict(spec=spec,folder=folder,receipt=receipt,report=report,
                refs={k:receipt['files'][k] for k in ('checkpoint','report')})
