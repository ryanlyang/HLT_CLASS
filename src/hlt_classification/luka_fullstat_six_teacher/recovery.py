"""Same-source continuation, or reviewed pre-training export-only recovery."""
from pathlib import Path
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.runner import selection_key
from .contracts import artifact,validate,recipe,TASK

CLAIM='six_teacher_recovery_claim_v1'


def inspect(path,project,commit):
    from . import campaign,worker,submission
    old=load_json(path)
    if old['source_commit']!=commit or Path(old['project_dir']).resolve()!=Path(project).resolve():
        from . import export_recovery
        return export_recovery.inspect(path,project,commit)
    campaign.check(old);state=submission.accounting(old,success=False)
    if (Path(old['root'])/TASK/'task_receipt.json').exists():raise ValueError('Already published; inspect before recovering')
    if (Path(old['root'])/CLAIM).exists():raise FileExistsError('Original already claimed by a recovery')
    records=worker.epoch_records(old,complete=False) if (Path(old['root'])/TASK/'epochs/01/receipt.json').is_file() else []
    exports=export_records(old)
    if not records and not exports:raise ValueError('No completed exports or epochs to continue; preserve diagnostic evidence')
    return artifact('RESUME',parents=dict(spec=old['content_hash'],recipe=recipe()['content_hash']),
        spec=file_ref(Path(path)),accounting=state,exports=exports,epochs=[dict(receipt=file_ref(Path(old['root'])/TASK/'epochs'/('%02d'%e)/'receipt.json'),
        state=r['state']) for e,r in enumerate(records,1)])


def check(spec):
    resume=spec.get('resume')
    if resume is None:return
    if resume['contract']=='LUKA_FULLSTAT_SIX_TEACHER_EXPORT_RECOVERY/v1':
        from . import export_recovery
        return export_recovery.check(spec)
    from .worker import epoch_records
    old=load_json(validate_file_ref(resume['spec']))
    validate(resume,'RESUME',dict(spec=old['content_hash'],recipe=recipe()['content_hash']))
    if (old['source_commit']!=spec['source_commit'] or old['recipe']!=spec['recipe']
            or old['reuse']!=spec['reuse'] or old['nodes']!=spec['nodes'] or old['root']==spec['root']):
        raise ValueError('Continuation changes science, source or inputs')
    rows=epoch_records(old,complete=False) if (Path(old['root'])/TASK/'epochs/01/receipt.json').is_file() else []
    expected=[dict(receipt=file_ref(Path(old['root'])/TASK/'epochs'/('%02d'%e)/'receipt.json'),state=r['state'])
              for e,r in enumerate(rows,1)]
    if (resume['epochs']!=expected or resume['exports']!=export_records(old)
            or (not expected and not resume['exports'])):raise ValueError('Saved continuation history changed')


def export_records(spec):
    from . import kd_banks
    found={}
    for name in kd_banks.NAMES:
        f=kd_banks.folder(spec,name)
        if (f/'manifest.json').is_file():
            kd_banks.manifest(spec,name)
            found[name]=[file_ref(f/n) for n in kd_banks.FILES]
    return found


def claim(spec,plan):
    if spec.get('resume') is None:return
    from . import submission
    old=load_json(validate_file_ref(spec['resume']['spec']))
    if submission.accounting(old,success=False)!=spec['resume']['accounting']:
        raise ValueError('Original terminal accounting changed')
    folder=Path(old['root'])/CLAIM;folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json',artifact('CLAIM',parents=dict(spec=spec['content_hash'],plan=plan['content_hash'])))


def import_epochs(spec):
    check(spec)
    if spec.get('resume') is None or not spec['resume']['epochs']:return None
    from .worker import parents
    history=[];states=[]
    for epoch,item in enumerate(spec['resume']['epochs'],1):
        old=load_json(validate_file_ref(item['receipt']));path=validate_file_ref(item['state'])
        folder=Path(spec['root'])/TASK/'epochs'/('%02d'%epoch);folder.mkdir(parents=True,exist_ok=False)
        atomic_publish_bytes(folder/'state.pt',Path(path).read_bytes())
        write_immutable_json(folder/'receipt.json',artifact('EPOCH',parents=dict(parents(spec),recipe=recipe()['content_hash']),
            row=old['row'],state=file_ref(folder/'state.pt')))
        history.append(old['row']);states.append(folder/'state.pt')
    selected=max(range(len(history)),key=lambda i:selection_key(history[i]['validation'],history[i]['update']))
    return dict(history=history,state=torch.load(states[-1],map_location='cpu',weights_only=True),
                best=torch.load(states[selected],map_location='cpu',weights_only=True))
