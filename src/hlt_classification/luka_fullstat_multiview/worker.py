"""Three production fits, authenticated dependency banks and native deployment."""
from pathlib import Path
import gc
import numpy as np
import time
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_fullstat_readout.worker import reference_arrays
from hlt_classification.luka_fullstat_u100_embedding import banks as original_banks
from hlt_classification.luka_fullstat_fine_tail.worker import rejection
from .resources import check_memory
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from . import campaign,submission,model,training,targets,views,startup
from .execution import execution
from .contracts import artifact,validate,recipe,TASKS,TEACHERS

FIT_FILES=('selected.pt','selected_full.pt','selected_training_state.pt','completed_training.json',
           'training_report.json','validation_predictions.npz','startup.json')


def output_names(task):
    names=[task+'/'+f for f in FIT_FILES]+[task+'/epochs/%02d/%s'%(e,f)
        for e in range(1,21) for f in ('state.pt','receipt.json')]
    if model.teacher_name(task): names += [task+'/bank/manifest.json',task+'/bank/targets.npz']
    return names


def teacher_refs(spec,name):
    folder=Path(spec['root'])/('teacher_'+name)
    return {'checkpoint':file_ref(folder/'selected.pt'),'report':file_ref(folder/'training_report.json')}


def fit_parents(spec,task):
    p=dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'])
    if task=='train_multi3':
        p['U100']=spec['reuse']['u100']['bank_hash']
        for name in ('CONCAT','FUSION'):
            m=load_json(Path(spec['root'])/('teacher_'+name)/'bank/manifest.json')
            validate(m,'BANK',targets.parents(spec,name,teacher_refs(spec,name)));p[name]=m['content_hash']
    return p


def load_banks(spec,task,tr):
    if model.teacher_name(task): return {}
    for name in ('CONCAT','FUSION'): completed(spec,'teacher_'+name)
    path=Path(validate_file_ref(spec['bank_manifest'])).parent
    banks={'U100':original_banks.Bank(path,tr,refs=spec['teacher_refs'])}
    if banks['U100'].manifest['content_hash']!=spec['reuse']['u100']['bank_hash']:
        raise ValueError('Original U100 bank changed')
    for name in ('CONCAT','FUSION'):
        banks[name]=targets.Bank(Path(spec['root'])/('teacher_'+name)/'bank',tr,name,teacher_refs(spec,name),spec)
    return banks


def validate_fit(spec,task):
    folder=Path(spec['root'])/task
    startup.check(spec,task,load_json(folder/'startup.json'))
    report=load_json(folder/'training_report.json');parents=fit_parents(spec,task)
    training.validate_report(report,spec,task,parents)
    if report!=load_json(folder/'completed_training.json'): raise ValueError('Completed report differs')
    for epoch in range(1,21):
        directory=folder/'epochs'/('%02d'%epoch);rec=load_json(directory/'receipt.json')
        validate(rec,'EPOCH',dict(parents,recipe=recipe()['content_hash']))
        if rec['row']!=report['validation_history'][epoch-1] or rec['state']!=file_ref(directory/'state.pt'):
            raise ValueError('Epoch receipt/state differs')
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    if file_ref(chosen)['sha256']!=file_ref(folder/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected training state differs')
    full=torch.load(chosen,map_location='cpu',weights_only=True)
    saved=torch.load(folder/'selected_full.pt',map_location='cpu',weights_only=True)
    if (full['epoch']!=report['selected_pass'] or saved.keys()!=full['student'].keys()
            or any(not torch.equal(v,full['student'][k]) for k,v in saved.items())):
        raise ValueError('Selected full model differs')
    net=model.new_model(spec['nodes'][task],task);net.load_state_dict(saved,strict=True)
    deployed=model.deploy(net,task);state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    if state.keys()!=deployed.state_dict().keys() or any(not torch.equal(v,deployed.state_dict()[k]) for k,v in state.items()):
        raise ValueError('Deployed state differs from selected full state')
    deployed.load_state_dict(state,strict=True)
    a=load_npz_arrays(folder/'validation_predictions.npz');_,ref=reference_arrays(spec,'M0HLT')
    if (not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels'])
            or evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']):
        raise ValueError('Published validation population/metrics differ')
    return report,a


def completed(spec,task,*,accounting=True):
    if task not in TASKS: raise PermissionError('Unknown task')
    root=Path(spec['root']);rec=load_json(root/task/'task_receipt.json');validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id']!=task or rec['source_commit']!=spec['source_commit']
            or rec['job_id']!=submission.accepted_jobs(spec,partial=True).get(task)
            or [r['path'] for r in rec['outputs']]!=output_names(task)):
        raise ValueError('Task identity/output inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=root)
    report,_=validate_fit(spec,task)
    if report['content_hash']!=rec['report_hash']: raise ValueError('Task report differs')
    name=model.teacher_name(task)
    if name:
        m=load_json(root/task/'bank/manifest.json');validate(m,'BANK',targets.parents(spec,name,teacher_refs(spec,name)))
        if m['teacher']!=name or m['role']!='train' or m['rows']!=spec['counts']['train']:
            raise ValueError('Teacher target role/population differs')
        validate_file_ref(m['payload'],root=root/task/'bank')
    if accounting: submission.successful(spec,task,rec['job_id'])
    return rec


def run(spec,task,*,device='cuda'):
    begun=time.monotonic();campaign.check(spec)
    job=execution(spec,task,spec['reuse']['u100'])
    if submission.accepted_jobs(spec,partial=True).get(task)!=job: raise PermissionError('Wrong exact submitted job')
    folder=Path(spec['root'])/task;folder.mkdir(exist_ok=False)
    log=PhaseLog(folder,lambda:memory_snapshot(torch))
    with log.phase('load_authenticated_fullstat_views'):
        tr,val,meta=views.load(spec,task);banks=load_banks(spec,task,tr)
    with log.phase('integrated_new_graph_checks'):
        proof=startup.run(spec,task,tr,val,banks,meta,device,begun)
        startup.check(spec,task,proof);write_immutable_json(folder/'startup.json',proof)
    gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    parents=fit_parents(spec,task)
    def snapshot(epoch,row,state):
        target=folder/'epochs'/('%02d'%epoch);target.mkdir(parents=True,exist_ok=False)
        save_state(target/'state.pt',state)
        write_immutable_json(target/'receipt.json',artifact('EPOCH',parents=dict(parents,recipe=recipe()['content_hash']),
            row=row,state=file_ref(target/'state.pt')))
        campaign.storage(spec)
    with log.phase('twenty_scientific_epochs'):
        net,report,full=training.train(tr,val,banks=banks,node=spec['nodes'][task],task=task,
            parents=parents,device=device,save_epoch=snapshot)
    save_state(folder/'selected_full.pt',training.cpu_state(net.state_dict()))
    selected=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',selected.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,task,parents)
    del banks,full
    deployed=model.deploy(net,task).to(device);save_state(folder/'selected.pt',training.cpu_state(deployed.state_dict()))
    # Use a fresh deployment graph and reload the published weights before inference.
    name=model.teacher_name(task)
    restored=(model.new_model(spec['nodes'][task],task) if name else DelphesParticleTransformer()).to(device)
    restored.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True),strict=True)
    del net,deployed;gc.collect()
    with log.phase('selected_deployment_readback'):
        p=training.predict(restored,val,device,name=name)
        if evaluate_probabilities(val.labels,p)!=report['validation']: raise ValueError('Selected metric replay differs')
        atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=p)))
        write_immutable_json(folder/'training_report.json',report)
        validate_fit(spec,task)
    if name:
        with log.phase('export_selected_TRAIN_embeddings'):
            targets.export(folder/'bank',restored,tr,name,teacher_refs(spec,name),spec,device)
            targets.Bank(folder/'bank',tr,name,teacher_refs(spec,name),spec)
    check_memory(spec,memory_snapshot(torch));campaign.check(spec)
    if execution(spec,task,spec['reuse']['u100'])!=job: raise PermissionError('Allocation changed')
    rec=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task,job_id=job,source_commit=spec['source_commit'],
        report_hash=report['content_hash'],outputs=[file_ref(Path(spec['root'])/p,root=spec['root']) for p in output_names(task)])
    write_immutable_json(folder/'task_receipt.json',rec)
    return rec


def results(spec):
    campaign.check(spec);rows=[];pending=[]
    items=[(n,*reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','U100_W5')]
    for task in TASKS:
        if not (Path(spec['root'])/task/'task_receipt.json').is_file():
            pending.append(dict(task=task,status='no_completed_receipt'));continue
        completed(spec,task);items.append((task,*validate_fit(spec,task)))
    ref=items[0][2]
    for name,report,a in items:
        if not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Comparison row identity/order differs')
        rows.append(dict(model=name,privileged=name=='OFFLINE' or name.startswith('teacher_'),
            selected_epoch=report['selected_pass'],validation=report['validation'],
            rejection_30=rejection(a['labels'],a['probabilities'],.3),rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi=rows[:2]
    for row in rows:
        row['recovery']=recovery(row['validation'],lo['validation'],hi['validation'])
        row['rejection_recovery']={k:{c:None if v is None or lo[k][c] is None or hi[k][c] is None or hi[k][c]==lo[k][c]
            else 100*(v-lo[k][c])/(hi[k][c]-lo[k][c]) for c,v in row[k].items()} for k in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
                    historical_U100_reference=True,single_seed=True,adaptive_validation=True)
