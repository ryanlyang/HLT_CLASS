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
    valid_jobs={r['job']:r['task'] for r in accepted if c.task_stage(r['task'])==stage}
    for epoch,path in enumerate(paths,1):
        r=load_json(path);c.validate(r,'EPOCH_COMMIT',parents)
        if path.name!=f'{epoch:02d}.json' or r['stage']!=stage or r['epoch']!=epoch or r['job_id'] not in valid_jobs:
            raise ValueError('Epoch cursor/stage/job differs')
        first,last=c.epoch_window(spec,valid_jobs[r['job_id']])
        if not first<=epoch<=last:raise ValueError('Epoch outside registered allocation window')
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
    from . import recovery
    original=recovery.origin(spec,stage)
    if original is not spec:return completed(original,stage,recurse=recurse,memo=memo)
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
    startup_evidence(spec,stage,receipt['files']['startup'],task='train_'+stage)
    if spec['schema_version']==4 and stage=='CONCAT':segment_completed(spec)
    result=dict(spec=spec,folder=Path(receipt['files']['report']['path']).parent,receipt=receipt,
                report=report,refs={k:receipt['files'][k] for k in ('checkpoint','report')})
    memo[stage]=result
    return result


def startup_evidence(spec,stage,ref,*,task):
    path=validate_file_ref(ref)
    startup=load_json(path);c.validate(startup,'STARTUP',dict(spec=spec['content_hash']))
    required_flags=('installed_weaver_parity','finite_connected_gradients','classifier_objective_verified',
                    'native_deployment_parity','full_support','probe_discarded','integrated_two_epoch_miniature')
    if startup['stage']!=stage or any(startup.get(k) is not True for k in required_flags):
        raise ValueError('Missing integrated acceptance')
    if startup.get('main_head')!=c.main_policy(stage):raise ValueError('Main-head acceptance differs')
    if startup.get('matching_serial_parity')!=(stage=='DE000'):raise ValueError('Matching acceptance differs')
    if spec['schema_version'] in (2,3,4,5):
        early_path=path.parent/'early_memory.json'
        early=load_json(validate_file_ref(startup['early_memory']))
        c.validate(early,'EARLY_MEMORY',dict(spec=spec['content_hash']))
        if Path(startup['early_memory']['path'])!=early_path or early['stage']!=stage or early['passed'] is not True:
            raise ValueError('Early full-support memory acceptance differs')
        if stage=='CONCAT':
            from .pair_storage import validate_storage_stats
            stats=startup['pair_storage'];validate_storage_stats(stats,calls=stats['calls'])
            if spec['schema_version'] in (3,4,5):
                from .pair_convolution import POLICY,validate_stats,validate_parities
                for witness in (early,startup):
                    if witness.get('pointwise_policy')!=POLICY:raise ValueError('Pointwise gradient policy differs')
                    validate_stats(witness['pointwise_statistics'],require_fallback=True)
                validate_parities(early['parity'])
    if spec['schema_version'] in (4,5):
        first,last=c.epoch_window(spec,task);before=startup.get('committed_epochs_before')
        if (startup.get('allocation_task')!=task or startup.get('epoch_window')!=[first,last]
                or type(before)!=int or not first-1<=before<=last
                or startup['timing']['remaining_epochs']!=last-before
                or startup['timing']['conservative_minutes']>spec['minutes'][task]):
            raise ValueError('Allocation startup scope/timing differs')
        from .startup import estimate
        timing=startup['timing']
        expected=estimate(stage,timing['train_step'],timing['val_step'],timing['export_step'],timing['elapsed'],last-before)
        if any(timing.get(k)!=v for k,v in expected.items()):raise ValueError('Allocation timing margins differ')
    return startup


def segment_completed(spec):
    """Authenticate the ten-epoch handoff; it is never a completed teacher."""
    from . import campaign
    if spec['schema_version']!=4:raise ValueError('Segments require SPEC/v4')
    root=Path(spec['root'])/'train_CONCAT';path=root/'segments/part1.json'
    parents=fit_parents(spec,'CONCAT',{});r=load_json(path);c.validate(r,'SEGMENT',parents)
    if (r['stage']!='CONCAT' or r['task']!='train_CONCAT_part1' or r['epoch_window']!=[1,10]
            or r['teacher_published'] is not False
            or not any(j['job']==r['job_id'] and j['task']==r['task'] for j in campaign.accepted_jobs(spec))):
        raise ValueError('Segment job/scope differs')
    rows,receipts=history(spec,'CONCAT',parents)
    expected=[file_ref(root/'epochs'/f'{i:02d}.json') for i in range(1,11)]
    if len(rows)<10 or r['epochs']!=expected:raise ValueError('Incomplete or changed segment epochs')
    folder=root/'runs'/r['job_id']
    if Path(r['startup']['path'])!=folder/'startup.json' or Path(r['view']['path'])!=folder/'view_build.json':
        raise ValueError('Segment evidence paths differ')
    startup_evidence(spec,'CONCAT',r['startup'],task=r['task'])
    view=load_json(validate_file_ref(r['view']));c.validate(view,'VIEW_BUILD',dict(spec=spec['content_hash']))
    if view['stage']!='CONCAT':raise ValueError('Segment input view differs')
    return r


def save_segment(spec,parents,job,folder):
    rows,_=history(spec,'CONCAT',parents)
    if len(rows)!=10:raise ValueError('Segment must commit exactly ten epochs')
    root=Path(spec['root'])/'train_CONCAT'
    r=c.artifact('SEGMENT',parents=parents,stage='CONCAT',task='train_CONCAT_part1',job_id=job,
        epoch_window=[1,10],teacher_published=False,
        epochs=[file_ref(root/'epochs'/f'{i:02d}.json') for i in range(1,11)],
        startup=file_ref(folder/'startup.json'),view=file_ref(folder/'view_build.json'))
    # Validate the startup before publishing a success marker.
    startup_evidence(spec,'CONCAT',r['startup'],task=r['task'])
    write_immutable_json(root/'segments/part1.json',r)
    return r
