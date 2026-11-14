"""Explicit fresh-root recovery, never automatic requeue or deletion of claims."""
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as meta
from . import contracts as c,campaign,source,submission,artifacts


def history(old,stage):
    folder=Path(old['root'])/('train_'+stage)
    inherited=old['continuations'].get(stage,{})
    refs=list(inherited.get('epochs',[]));parents=list(inherited.get('epoch_parents',[]))
    path=folder/'fit_binding.json'
    if path.exists():
        binding=meta.load_json(path);c.validate(binding,'FIT_BINDING',dict(spec=old['content_hash']))
        if binding['stage']!=stage:raise ValueError('Recovery fit identity differs')
        more=artifacts.epoch_refs(folder,binding['fit_parents'],stage,c.COUNTS['validation'],start_epoch=len(refs)+1)
        refs+=more;parents += [binding['fit_parents']]*len(more)
    elif (folder/'epochs').exists():raise ValueError('Epochs lack registered fit binding')
    return dict(epochs=refs,epoch_parents=parents,completed_epochs=len(refs)) if refs else None


def scan(old,*,unsuccessful=(),account=True):
    jobs=submission.accepted_jobs(old,partial=True)
    completed=dict(old['completed']);continuations={}
    for stage in c.STAGES:
        if stage in completed:continue
        task='train_'+stage;folder=Path(old['root'])/task
        state=submission.accounting(old,task,success=False) if account and task in jobs else None
        good=(state[4]=='COMPLETED' and state[5]=='0:0') if state is not None else stage not in unsuccessful
        receipt=folder/'task_receipt.json'
        if receipt.is_file() and good:
            artifacts.completed(old,stage,account=account)
            completed[stage]=dict(spec=meta.file_ref(Path(old['root'])/'campaign_spec.json'),receipt=meta.file_ref(receipt))
            continue
        value=history(old,stage)
        if value:continuations[stage]=value
    return completed,continuations


def validate_registration(spec):
    r=spec['recovery'];old=meta.load_json(meta.validate_file_ref(r['parent_spec']))
    if old['root']==spec['root']:raise ValueError('Recovery self-reference')
    campaign.check(old,live_source=False)
    for key in ('recipe','nodes','inputs','population','population_root','counts','eligible_train_rows'):
        if spec[key]!=old[key]:raise ValueError('Recovery changed scientific registration: '+key)
    old_files=old['source']['files'];new_files=spec['source']['files']
    runtime_names={k for k in old_files if k.startswith(('src/','scripts/','sbatch/','configs/'))}
    if runtime_names!={k for k in new_files if k.startswith(('src/','scripts/','sbatch/','configs/'))}:
        raise ValueError('Recovery runtime inventory differs')
    if any(old_files[k]!=new_files[k] for k in runtime_names):raise ValueError('Recovery requires the same runtime bytes')
    completed,continuations=scan(old,unsuccessful=r['unsuccessful_stages'],account=False)
    if completed!=spec['completed'] or continuations!=spec['continuations']:
        raise ValueError('Recovery snapshot changed')


def create(root,parent,project,commit,quota_gib):
    if type(quota_gib)!=int or quota_gib<campaign.MIN_QUOTA_GIB:raise ValueError('36 GiB additional available quota required')
    parent=Path(parent).resolve(strict=True);old=meta.load_json(parent);campaign.check(old,live_source=False)
    if parent!=Path(old['root'])/'campaign_spec.json':raise ValueError('Noncanonical recovery parent')
    if (Path(old['root'])/'balanced_recovery_claim_v1').exists():raise FileExistsError('Parent recovery already claimed')
    src=source.current(project,commit);source.replay(old['inputs'])
    completed,continuations=scan(old)
    if len(completed)==6:raise ValueError('Campaign already complete; no recovery required')
    states={task:submission.accounting(old,task,success=False) for task in submission.accepted_jobs(old,partial=True)}
    failed=[task.removeprefix('train_') for task,row in states.items() if row[4]!='COMPLETED' or row[5]!='0:0']
    protected=[old['root'],project,old['population_root'],*(old['inputs']['evidence'][k] for k in ('raw_container','cache_root','foundation_root'))]
    root=campaign.root_check(root,protected)
    fields={k:v for k,v in old.items() if k not in ('content_hash','contract','schema_version','parents',*c.FLAGS)}
    fields.update(root=str(root),project_dir=str(Path(project).resolve()),source_commit=commit,source=src,
        available_result_bytes=quota_gib*2**30,completed=completed,continuations=continuations,
        recovery=dict(parent_spec=meta.file_ref(parent),unsuccessful_stages=failed,accounting=states),
        tasks={task:[d for d in deps if d.removeprefix('train_') not in completed]
               for task,deps in c.tasks().items() if task.removeprefix('train_') not in completed})
    spec=c.artifact('SPEC',**fields);campaign.check(spec)
    root.mkdir(parents=True,exist_ok=False);meta.write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def claim(spec,plan):
    validate_registration(spec)
    old=meta.load_json(meta.validate_file_ref(spec['recovery']['parent_spec']))
    states={task:submission.accounting(old,task,success=False) for task in submission.accepted_jobs(old,partial=True)}
    if states!=spec['recovery']['accounting']:raise ValueError('Old terminal scheduler accounting changed')
    folder=Path(old['root'])/'balanced_recovery_claim_v1';folder.mkdir(exist_ok=False)
    meta.write_immutable_json(folder/'intent.json',c.artifact('RECOVERY_CLAIM',parents=dict(spec=spec['content_hash'],plan=plan['content_hash'])))
