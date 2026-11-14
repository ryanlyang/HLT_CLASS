"""Completion-last epochs/results, explicit attempt recovery, no overwritten payloads."""
from pathlib import Path
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_fullstat_balanced_ladder.artifacts import save_state
from . import contracts as c, training


def fit_parents(spec,stage,dependencies):
    return dict(spec=spec['content_hash'],population=spec['population']['content_hash'],
                **{'teacher_'+t:r['receipt']['content_hash'] for t,r in dependencies.items()})


def save_epoch(root,folder,stage,epoch,row,state,parents,job):
    target=Path(folder)/'epochs'/f'{epoch:02d}'
    target.mkdir(parents=True,exist_ok=False)
    save_state(target/'state.pt',state)
    receipt=c.artifact('EPOCH_COMMIT',parents=parents,stage=stage,epoch=epoch,job_id=job,
                       row=row,state=file_ref(target/'state.pt'))
    # This marker is written last, after the complete state. Uncommitted attempts
    # remain preserved under their job folder and never look like reusable epochs.
    write_immutable_json(Path(root)/('train_'+stage)/'epochs'/f'{epoch:02d}.json',receipt)


def history(spec,stage,parents,*,sampler=None,states=True):
    paths=sorted((Path(spec['root'])/('train_'+stage)/'epochs').glob('*.json'))
    rows=[];receipts=[]
    if len(paths)>c.EPOCHS:raise ValueError('Excess epoch commits')
    from . import campaign
    accepted=campaign.accepted_jobs(spec)
    valid_jobs={r['job'] for r in accepted if r['task']=='train_'+stage}
    for epoch,path in enumerate(paths,1):
        r=load_json(path);c.validate(r,'EPOCH_COMMIT',parents)
        if path.name!=f'{epoch:02d}.json' or r['stage']!=stage or r['epoch']!=epoch or r['job_id'] not in valid_jobs:
            raise ValueError('Epoch cursor/stage/job differs')
        training.check_row(r['row'],epoch,stage,c.COUNTS['validation'],sampler)
        if states:validate_file_ref(r['state'])
        rows.append(r['row']);receipts.append(r)
    return rows,receipts


def restore(spec,stage,parents,sampler):
    rows,receipts=history(spec,stage,parents,sampler=sampler)
    if not rows:return None
    selected=max(rows,key=lambda r:selection_key(r['validation'],r['update']))['pass_number']
    state=torch.load(validate_file_ref(receipts[-1]['state']),map_location='cpu',weights_only=True)
    best=state if selected==len(rows) else torch.load(validate_file_ref(receipts[selected-1]['state']),map_location='cpu',weights_only=True)
    return dict(history=rows,state=state,best=best)


def completed(spec,stage,*,recurse=True,memo=None):
    memo={} if memo is None else memo
    if stage in memo:return memo[stage]
    folder=Path(spec['root'])/('train_'+stage)
    receipt=load_json(folder/'result.json');c.validate(receipt,'RESULT')
    if receipt['stage']!=stage or receipt['parents'].get('spec')!=spec['content_hash']:
        raise ValueError('Completed stage/spec differs')
    from . import campaign
    if not any(r['job']==receipt['job_id'] and r['task']=='train_'+stage for r in campaign.accepted_jobs(spec)):
        raise ValueError('Result scheduler binding differs')
    dependencies={t:completed(spec,t,memo=memo) for t in c.TEACHERS[stage]} if recurse else {}
    if recurse and receipt['fit_parents']!=fit_parents(spec,stage,dependencies):raise ValueError('Teacher lineage differs')
    required={'checkpoint','report','predictions','outside_predictions','metrics','startup','view'}|({'bank'} if stage!='DE000' else set())
    if set(receipt['files'])!=required:raise ValueError('Published result inventory differs')
    for ref in receipt['files'].values():validate_file_ref(ref)
    report=load_json(receipt['files']['report']['path'])
    training.validate_report(report,stage,receipt['fit_parents'],c.COUNTS['validation'])
    if receipt['parents']['report']!=report['content_hash'] or receipt['selected_epoch']!=report['selected_pass']:
        raise ValueError('Selected report binding differs')
    rows,commits=history(spec,stage,receipt['fit_parents'])
    if rows!=report['validation_history']:raise ValueError('Published history differs from committed epochs')
    startup=load_json(receipt['files']['startup']['path']);c.validate(startup,'STARTUP',dict(spec=spec['content_hash']))
    required_flags=('installed_weaver_parity','finite_connected_gradients','classifier_ce_only',
                    'native_deployment_parity','full_support','probe_discarded','integrated_two_epoch_miniature')
    if startup['stage']!=stage or any(startup.get(k) is not True for k in required_flags):
        raise ValueError('Missing integrated acceptance')
    if startup.get('matching_serial_parity')!=(stage=='DE000'):raise ValueError('Matching acceptance differs')
    result=dict(spec=spec,folder=Path(receipt['files']['report']['path']).parent,receipt=receipt,
                report=report,refs={k:receipt['files'][k] for k in ('checkpoint','report')})
    memo[stage]=result
    return result
