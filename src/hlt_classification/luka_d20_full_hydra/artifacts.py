"""Completion-last epochs and authenticated retirement of this fit's obsolete state."""
from pathlib import Path
import os
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_fullstat_balanced_ladder.artifacts import save_state
from hlt_classification.luka_reconstruction_stability_hydra.identity import tensor_hash
from . import contracts as c,model


def prepare_starts(spec):
    folder=Path(spec['root'])/'starts';folder.mkdir(exist_ok=True)
    rows={}
    for seed in c.SEEDS:
        net=model.fresh(f'D20_s{seed}',spec['maximum']);state=net.state_dict()
        path=folder/f'seed_{seed}.pt'
        if path.exists():
            existing=torch.load(path,map_location='cpu',weights_only=True)
            if tensor_hash(existing)!=tensor_hash(state):raise ValueError('Partial canonical start differs')
        else:save_state(path,state)
        rows[str(seed)]=dict(state=file_ref(path),components=components(state),untrained=True)
    result=c.artifact('STARTS',parents=dict(spec=spec['content_hash']),seeds=rows)
    path=folder/'manifest.json'
    if path.exists():
        if load_json(path)!=result:raise ValueError('Existing canonical starts differ')
    else:write_immutable_json(path,result)
    return result


def components(state):
    return {name:tensor_hash({k:v for k,v in state.items() if k.startswith(prefix)})
            for name,prefix in [('native','mod.'),('predictor','predictor.'),('decoder','aux.')]}


def start(spec,task):
    record=load_json(Path(spec['root'])/'starts/manifest.json');c.validate(record,'STARTS',dict(spec=spec['content_hash']))
    if set(record['seeds'])!={str(s) for s in c.SEEDS}:raise ValueError('Canonical seed coverage differs')
    row=record['seeds'][str(c.cell(task)['seed'])]
    if row['untrained'] is not True:raise ValueError('Science start must be untrained')
    state=torch.load(validate_file_ref(row['state']),map_location='cpu',weights_only=True)
    if components(state)!=row['components']:raise ValueError('Untrained tensors differ')
    return state,row['components']


def fit_root(spec,task):
    c.cell(task)
    return Path(spec['root'])/'fits'/task


def history(spec,task,parents):
    from .training import check_row
    from . import campaign
    jobs={r['job'] for r in campaign.accepted_jobs(spec,unambiguous=False) if r['task'] in c.GPU_TASKS and c.segment(r['task'])[0]==task}
    rows=[];commits=[]
    for epoch,path in enumerate(sorted((fit_root(spec,task)/'epochs').glob('*.json')),1):
        r=load_json(path);c.validate(r,'EPOCH',parents)
        if r['task']!=task or r['epoch']!=epoch or path.name!=f'{epoch:02d}.json' or epoch>c.EPOCHS:
            raise ValueError('Epoch commit sequence differs')
        if r['job_id'] not in jobs:raise ValueError('Epoch scheduler binding differs')
        check_row(r['row'],task,epoch,spec)
        rows.append(r['row']);commits.append(r)
    return rows,commits


def retire(spec,task,commit,reason):
    """Only a registered obsolete restart payload inside this exact fit."""
    root=fit_root(spec,task).resolve();path=Path(commit['state']['path']).absolute()
    if (path!=path.resolve() or not path.is_relative_to(root/'runs') or path.name!=f'state_{commit["epoch"]:02d}.pt'
            or any(p.is_symlink() for p in (path,*path.parents))):raise PermissionError('Retirement escapes own restart payload')
    folder=root/'retired';intent=folder/f'{commit["epoch"]:02d}.intent.json';done=folder/f'{commit["epoch"]:02d}.json'
    value=c.artifact('STATE_RETIREMENT',parents=dict(spec=spec['content_hash'],epoch=commit['content_hash']),
                     state=commit['state'],reason=reason)
    if intent.exists():
        if load_json(intent)!=value:raise ValueError('Retirement intent differs')
    else:
        validate_file_ref(commit['state']);write_immutable_json(intent,value)
    if path.exists():validate_file_ref(commit['state']);path.unlink()
    if done.exists():
        if load_json(done)!=value:raise ValueError('Retirement completion differs')
    else:write_immutable_json(done,value)


def compact(spec,task,parents,*,published=False):
    rows,commits=history(spec,task,parents)
    for commit in commits if published else commits[:-1]:
        retire(spec,task,commit,'published_native_outputs' if published and commit==commits[-1] else 'newer_epoch_committed')


def commit_epoch(spec,task,folder,row,state,parents,job):
    path=Path(folder)/f'state_{row["epoch"]:02d}.pt';save_state(path,state)
    r=c.artifact('EPOCH',parents=parents,task=task,epoch=row['epoch'],job_id=job,row=row,state=file_ref(path))
    write_immutable_json(fit_root(spec,task)/'epochs'/f'{row["epoch"]:02d}.json',r)
    compact(spec,task,parents)
    return r


def restore(spec,task,parents):
    rows,commits=history(spec,task,parents)
    if not rows:return None
    compact(spec,task,parents)
    state=torch.load(validate_file_ref(commits[-1]['state']),map_location='cpu',weights_only=True)
    if state['epoch']!=len(rows) or state['history']!=rows:raise ValueError('Restart/committed history differs')
    return state


def completed(spec,task,*,payloads=True):
    root=fit_root(spec,task);r=load_json(root/'result.json');c.validate(r,'RESULT')
    if r['task']!=task or r['parents']['spec']!=spec['content_hash']:raise ValueError('Result/spec differs')
    from . import campaign
    if not any(j['task']==task+'_p04' and j['job']==r['job_id'] for j in campaign.accepted_jobs(spec,unambiguous=False)):
        raise ValueError('Result scheduler binding differs')
    expected={'report','initialization'}|{n+'_'+k for n in c.readouts(task) for k in ('checkpoint','predictions','metrics')}
    if c.cell(task)['arm'] in ('OFFLINE','U100'):expected.add('bank')
    if set(r['files'])!=expected:raise ValueError('Published fit inventory differs')
    for ref in r['files'].values():
        if payloads:validate_file_ref(ref)
    report=load_json(validate_file_ref(r['files']['report']))
    from .training import validate_report
    validate_report(report,spec,task,r['fit_parents'])
    if r['parents']['report']!=report['content_hash']:raise ValueError('Result/report parent differs')
    initialization=load_json(validate_file_ref(r['files']['initialization']))
    c.validate(initialization,'INITIALIZATION',r['fit_parents'])
    if (initialization['content_hash']!=report['initialization'] or initialization['task']!=task
            or initialization['seeds']!=c.seeds(task) or initialization['world']!=c.RESOURCES['gpus']):
        raise ValueError('Published initialization binding differs')
    rows,_=history(spec,task,r['fit_parents'])
    if rows!=report['history']:raise ValueError('Published history differs')
    return dict(receipt=r,report=report,root=root)
