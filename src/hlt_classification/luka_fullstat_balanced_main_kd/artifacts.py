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


def completed(spec,stage='DE000',*,account=True):
    from . import campaign,submission,source,worker
    campaign.check(spec,live_source=False)
    if stage!='DE000':raise PermissionError('Only fresh DE000 results')
    folder=Path(spec['root'])/'train_DE000'
    receipt=meta.load_json(folder/'task_receipt.json');c.validate(receipt,'TASK_RECEIPT')
    if (receipt['stage']!=stage or receipt['parents']['spec']!=spec['content_hash']
            or receipt['population']!=spec['population']['content_hash']
            or receipt['job_id']!=submission.accepted_jobs(spec,partial=True)['train_DE000']):
        raise ValueError('Task identity differs')
    if account:submission.accounting(spec,'train_DE000')
    expected={'checkpoint','report','predictions','metrics','startup','view','binding'}
    if set(receipt['files'])!=expected:raise ValueError('Result inventory differs')
    for ref in receipt['files'].values():meta.validate_file_ref(ref)
    parent=source.check(spec['reuse'],spec['project_dir'],source_check=False)
    parents=worker.fit_parents(spec,stage,worker.dependencies_for(spec,parent))
    binding=meta.load_json(receipt['files']['binding']['path']);c.validate(binding,'FIT_BINDING',dict(spec=spec['content_hash']))
    if binding['stage']!=stage or binding['fit_parents']!=parents or receipt['fit_parents']!=parents:
        raise ValueError('Frozen teacher lineage differs')
    startup=meta.load_json(receipt['files']['startup']['path']);c.validate(startup,'STARTUP',dict(spec=spec['content_hash']))
    required=('installed_weaver_parity','finite_connected_gradients','classifier_ce_kd',
              'native_deployment_parity','full_support','matching_serial_parity','probe_discarded')
    if startup['stage']!=stage or any(startup.get(k) is not True for k in required) or startup.get('classifier_ce_only') is not False:
        raise ValueError('Missing main-head KD integrated acceptance')
    report=meta.load_json(receipt['files']['report']['path']);training.validate_report(report,stage,parents,c.COUNTS['validation'])
    if receipt['parents']['report']!=report['content_hash'] or receipt['selected_pass']!=report['selected_pass']:
        raise ValueError('Selected report binding differs')
    metrics=meta.load_json(receipt['files']['metrics']['path']);c.validate(metrics,'METRICS',dict(report=report['content_hash']))
    if metrics['populations']['natural']['summary']!=report['validation'] or metrics['checkpoint_reselection'] is not False:
        raise ValueError('Selected metrics binding differs')
    inherited=spec['continuations'].get(stage,{})
    if receipt['inherited_epochs']!=inherited.get('epochs',[]):raise ValueError('Inherited history differs')
    refs=receipt['inherited_epochs']+receipt['new_epochs']
    eps=inherited.get('epoch_parents',[])+[parents]*len(receipt['new_epochs'])
    if len(refs)!=c.EPOCHS:raise ValueError('Expected twenty committed epochs')
    for epoch,(ref,ep) in enumerate(zip(refs,eps,strict=True),1):
        commit=meta.load_json(meta.validate_file_ref(ref));c.validate(commit,'EPOCH_COMMIT',ep)
        meta.validate_file_ref(commit['state'])
        saved=meta.load_json(meta.validate_file_ref(commit['report']));c.validate(saved,'EPOCH',ep)
        if commit['epoch']!=epoch or saved['epoch']!=epoch or saved['row']!=report['validation_history'][epoch-1]:
            raise ValueError('Committed history differs from report')
    return dict(spec=spec,folder=folder,receipt=receipt,report=report)

