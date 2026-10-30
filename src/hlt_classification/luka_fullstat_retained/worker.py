"""Integrated real-graph admission, fresh fits and retained-graph publication."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_fullstat_attention.worker import metadata
from hlt_classification.luka_fullstat_u100_embedding.worker import data,longest
from hlt_classification.luka_fullstat_u100_embedding import banks
from hlt_classification.luka_fullstat_readout import worker as old_worker,model as old_model
from hlt_classification.luka_fullstat_fine_tail.worker import rejection
from . import campaign,submission,training,model
from .execution import execution
from .contracts import artifact,validate,recipe,TASKS

FIT_FILES=('selected.pt','selected_training_state.pt','completed_training.json',
           'training_report.json','validation_predictions.npz','startup.json','inference_cost.json')


def output_names(task):
    return [task+'/'+n for n in FIT_FILES]+[task+'/epochs/%02d/%s'%(e,n)
        for e in range(1,21) for n in ('state.pt','receipt.json')]


def load_inputs(spec,task):
    if task not in TASKS: raise PermissionError('Unregistered task')
    meta=metadata(spec)
    if meta[0]['summaries']!=spec['reuse']['u100']['summaries']: raise ValueError('Prepared support changed')
    tr,val=(data(spec,r,'D000',authenticated=meta) for r in ('train','validation'))
    bank=None
    if task=='train_retained':
        directory=Path(validate_file_ref(spec['bank_manifest'])).parent
        bank=banks.Bank(directory,tr,refs=spec['teacher_refs'])
        if bank.manifest['content_hash']!=spec['reuse']['u100']['bank_hash']: raise ValueError('Original TRAIN bank changed')
    return tr,val,bank,meta


def check_memory(spec,value):
    if (value['gpu_peak_bytes']<=0 or value['peak_rss_bytes']<=0
            or value['gpu_peak_bytes']>=.9*spec['reuse']['u100']['gpu']['total_memory_bytes']
            or value['peak_rss_bytes']+8*2**30>=.85*spec['memory_mb']*1024**2):
        raise MemoryError('Retained graph exceeds inherited Oscar memory envelope')


def timing(spec,task,train_step,val_step,preparation):
    if any(not math.isfinite(v) or v<=0 for v in (train_step,val_step)) or not math.isfinite(preparation) or preparation<0:
        raise ValueError('Invalid timing witness')
    minutes=math.ceil(preparation/60+120+1.5*(20*math.ceil(spec['counts']['train']/128)*train_step
        +22*math.ceil(spec['counts']['validation']/128)*val_step)/60)
    if minutes>spec['minutes'][task]: raise ValueError('Measured conservative timing exceeds allocation')
    return dict(train_seconds_per_update=train_step,validation_seconds_per_batch=val_step,
        preparation_seconds=preparation,conservative_minutes=minutes,full_epoch_measured=False)


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize()


def inference_cost(spec,task,student,val,device):
    # Preserve/check the scientific fit peak before resetting CUDA counters
    # for a separate deployment-memory measurement.
    before=memory_snapshot(torch); check_memory(spec,before)
    cuda=torch.device(device).type=='cuda'
    student.eval(); student.capture=False
    native=DelphesParticleTransformer().to(device).eval()
    native.load_state_dict({k:v for k,v in student.state_dict().items() if k.startswith('mod.')},strict=True)
    raw=val.batch(np.arange(min(128,len(val)))); rows={}
    with torch.inference_mode():
        for name,net in (('retained',student),('native_reference',native)):
            for _ in range(3): _forward(net,raw,device,bf16=torch.device(device).type=='cuda')
            student.predictions={}; sync(device)
            if cuda:
                baseline=int(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            sync(device); begun=time.monotonic()
            for _ in range(10): _forward(net,raw,device,bf16=torch.device(device).type=='cuda')
            sync(device)
            rows[name]=dict(parameters=sum(p.numel() for p in net.parameters()),
                state_bytes=sum(v.numel()*v.element_size() for v in net.state_dict().values()),
                seconds_per_batch=(time.monotonic()-begun)/10)
            rows[name]['cuda_peak_increment_bytes']=int(torch.cuda.max_memory_allocated())-baseline if cuda else None
    student.predictions={}
    return artifact('INFERENCE_COST',parents=dict(spec=spec['content_hash']),task_id=task,
        batch=len(raw['labels']),warmups=3,repeats=10,includes_transfer=True,
        population='first_validation_batch',production_HLT_latency_certified=False,measurements=rows,
        device=torch.device(device).type,memory_before_measurement=before,
        memory_scope='peak increment above resident models; state_bytes reported separately')


def startup(spec,task,tr,val,bank,meta,device,started):
    witness=None
    if task=='train_retained':
        readout=old_model.make_readout(bank,Path(spec['bank_manifest']['path']).parent,spec['teacher_refs'],device)
        witness=old_worker.teacher_parity(spec,bank,readout,meta,device)
        del readout
    elif bank is not None: raise PermissionError('CE control cannot access teacher targets')
    with parity_backend(device): installed_parity(tr,device=device)
    student=model.new_model(spec['nodes'][task],task).to(device).eval()
    ordinary=DelphesParticleTransformer().to(device).eval()
    ordinary.load_state_dict({k:v for k,v in student.state_dict().items() if k.startswith('mod.')},strict=True)
    raw=tr.batch(np.arange(min(4,len(tr))))
    errors={}
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            a=_forward(student,raw,device,bf16=bf16); b=_forward(ordinary,raw,device,bf16=bf16)
            tolerance=1e-2 if bf16 else 2e-5
            # The retained classifier intentionally emits FP32; the native
            # head emits BF16 under autocast. Compare values in common FP32.
            torch.testing.assert_close(a.float(),b.float(),rtol=tolerance,atol=tolerance)
            errors['bf16' if bf16 else 'fp32']=float((a.float()-b.float()).abs().max())
    del ordinary
    student.train(); opt=training.optimizer(student,recipe()['base'])
    index,maximum=longest(tr); ids=np.full(128,index,np.int64)
    frozen=training.cpu_state(student.mod.fc.state_dict())
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss,parts=model.batch_loss(student,tr.batch(ids),bank,ids,task,device)
        expected=torch.autograd.grad(parts['ce'],tuple(student.combined.parameters()),retain_graph=True)
        loss.backward(); training.finite_gradients(student)
        for p,g in zip(student.combined.parameters(),expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        opt.step(); student.predictions={}
    if any(not torch.equal(v.cpu(),frozen[k]) for k,v in student.mod.fc.state_dict().items()):
        raise ValueError('Unused native classifier changed')
    if not torch.any(student.combined.weight[:,128:]!=0): raise ValueError('Prediction classifier branch stayed disconnected')
    # Strict deployment readback using the same actual expanded graph, no teacher.
    deployed=model.load_deployed(spec['nodes'][task],task,training.cpu_state(student.state_dict()),device)
    student.eval(); student.capture=False
    with parity_backend(device),torch.inference_mode():
        torch.testing.assert_close(_forward(student,raw,device,bf16=False),_forward(deployed,raw,device,bf16=False),rtol=0,atol=0)
    del deployed
    vi,vmaximum=longest(val)
    with torch.inference_mode():
        out=_forward(student,val.batch(np.full(128,vi,np.int64)),device,bf16=torch.device(device).type=='cuda')
        if not torch.isfinite(out).all(): raise ValueError('Invalid maximum-support validation forward')
    rng=np.random.default_rng(73012); student.train(); student.capture=True
    for step in range(68):
        if step==4: sync(device); begun=time.monotonic()
        ids=rng.integers(0,len(tr),128); opt.zero_grad(set_to_none=True)
        loss,_=model.batch_loss(student,tr.batch(ids),bank,ids,task,device)
        loss.backward(); training.finite_gradients(student); opt.step(); student.predictions={}
    sync(device); train_step=(time.monotonic()-begun)/64
    student.eval(); student.capture=False; sync(device); begun=time.monotonic()
    with torch.inference_mode():
        for _ in range(16):
            out=_forward(student,val.batch(rng.integers(0,len(val),128)),device,bf16=torch.device(device).type=='cuda')
            if not torch.isfinite(out).all(): raise ValueError('Invalid validation timing witness')
    sync(device); val_step=(time.monotonic()-begun)/16
    memory=memory_snapshot(torch); check_memory(spec,memory)
    estimate=timing(spec,task,train_step,val_step,time.monotonic()-started)
    print('RETAINED integrated conservative_minutes='+str(estimate['conservative_minutes']),flush=True)
    return artifact('STARTUP',parents=dict(spec=spec['content_hash']),task_id=task,teacher_witness=witness,
        initial_logits_parity=True,initial_logits_maximum_errors=errors,installed_weaver_parity=True,
        retained_deployment_parity=True,combined_classifier_CE_only=True,finite_connected_gradients=True,
        unused_native_classifier_frozen=True,teacher_targets_used=bank is not None,
        batch=128,stress_steps=3,train_maximum=maximum,validation_maximum=vmaximum,
        probe_discarded=True,memory=memory,timing=estimate)


def validate_startup(spec,task,value):
    validate(value,'STARTUP',dict(spec=spec['content_hash']))
    summaries=spec['reuse']['u100']['summaries']
    if (value['task_id']!=task or any(value[k] is not True for k in ('initial_logits_parity','installed_weaver_parity',
            'retained_deployment_parity','combined_classifier_CE_only','finite_connected_gradients',
            'unused_native_classifier_frozen','probe_discarded'))
            or value['teacher_targets_used']!=(task=='train_retained')
            or value['batch']!=128 or value['stress_steps']!=3
            or value['train_maximum']!=summaries['train']['D000']['maximum']
            or value['validation_maximum']!=summaries['validation']['D000']['maximum']):
        raise ValueError('Retained startup evidence differs')
    if task=='train_retained':
        old_worker.validate_teacher_witness(spec,value['teacher_witness'])
        if value['teacher_witness']['teacher_train_maximum']!=summaries['train']['U100']['maximum']:
            raise ValueError('Teacher maximum differs')
    elif value['teacher_witness'] is not None: raise ValueError('Control used teacher replay')
    errors=value['initial_logits_maximum_errors']
    if set(errors)!={'fp32','bf16'} or any(not math.isfinite(v) or v<0 for v in errors.values()):
        raise ValueError('Invalid initialization parity errors')
    check_memory(spec,value['memory']); t=value['timing']
    if t!=timing(spec,task,t['train_seconds_per_update'],t['validation_seconds_per_batch'],t['preparation_seconds']):
        raise ValueError('Timing record differs')


def fit_parents(spec,task):
    parents=dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'])
    if task=='train_retained': parents['bank']=spec['reuse']['u100']['bank_hash']
    return parents


def validate_fit(spec,task):
    folder=Path(spec['root'])/task
    validate_startup(spec,task,load_json(folder/'startup.json'))
    report=load_json(folder/'training_report.json'); parents=fit_parents(spec,task)
    training.validate_report(report,spec,task,parents)
    if load_json(folder/'completed_training.json')!=report: raise ValueError('Completed training changed')
    for epoch in range(1,21):
        target=folder/'epochs'/('%02d'%epoch); rec=load_json(target/'receipt.json')
        validate(rec,'EPOCH',dict(parents,recipe=recipe()['content_hash']))
        if rec['row']!=report['validation_history'][epoch-1] or rec['state']!=file_ref(target/'state.pt'):
            raise ValueError('Epoch snapshot binding differs')
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    if file_ref(chosen)['sha256']!=file_ref(folder/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected full state differs from epoch')
    full=torch.load(chosen,map_location='cpu',weights_only=True)
    state=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    model.load_deployed(spec['nodes'][task],task,state,'cpu')
    if (state.keys()!=full['student'].keys() or any(not torch.equal(v,full['student'][k]) for k,v in state.items())
            or full['epoch']!=report['selected_pass']): raise ValueError('Selected retained/full state differs')
    arrays=load_npz_arrays(folder/'validation_predictions.npz')
    _,reference=old_worker.reference_arrays(spec,'M0HLT')
    if (not np.array_equal(arrays['identities'],reference['identities']) or not np.array_equal(arrays['labels'],reference['labels'])
            or evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']):
        raise ValueError('Selected prediction population/metrics differ')
    cost=load_json(folder/'inference_cost.json'); validate(cost,'INFERENCE_COST',dict(spec=spec['content_hash']))
    if (cost['task_id']!=task or cost['batch']!=min(128,spec['counts']['validation'])
            or cost['warmups']!=3 or cost['repeats']!=10 or cost['includes_transfer'] is not True
            or cost['production_HLT_latency_certified'] is not False
            or set(cost['measurements'])!={'retained','native_reference'}): raise ValueError('Inference cost identity differs')
    check_memory(spec,cost['memory_before_measurement'])
    if (cost['device'] not in ('cuda','cpu') or cost['population']!='first_validation_batch'
            or cost['memory_scope']!='peak increment above resident models; state_bytes reported separately'):
        raise ValueError('Inference measurement scope differs')
    for row in cost['measurements'].values():
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('parameters','state_bytes','seconds_per_batch')):
            raise ValueError('Invalid inference cost')
        peak=row['cuda_peak_increment_bytes']
        if ((cost['device']=='cpu' and peak is not None) or (cost['device']=='cuda'
                and (type(peak) is not int or peak<0))): raise ValueError('Invalid inference memory')
    return report,arrays


def completed(spec,task,*,accounting=True):
    root=Path(spec['root']); rec=load_json(root/task/'task_receipt.json')
    validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id']!=task or rec['job_id']!=submission.accepted_jobs(spec)[task]
            or rec['source_commit']!=spec['source_commit'] or [r['path'] for r in rec['outputs']]!=output_names(task)):
        raise ValueError('Completed task binding/inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=root)
    report,_=validate_fit(spec,task)
    if rec['report_hash']!=report['content_hash']: raise ValueError('Task/report binding differs')
    if accounting: submission.successful(spec,task,rec['job_id'])
    return rec


def results(spec):
    campaign.check(spec); rows=[]; pending=[]
    items=[(n,*old_worker.reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','U100_W5','READOUT_CE')]
    for task in TASKS:
        if not (Path(spec['root'])/task/'task_receipt.json').is_file():
            pending.append(dict(task=task,status='no_completed_receipt')); continue
        completed(spec,task); items.append((spec['nodes'][task]['node_id'],*validate_fit(spec,task)))
    ref=items[0][2]
    for name,report,a in items:
        if not np.array_equal(a['identities'],ref['identities']) or not np.array_equal(a['labels'],ref['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=report['selected_pass'],validation=report['validation'],
            rejection_30=rejection(a['labels'],a['probabilities'],.3),rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi=rows[:2]
    for row in rows:
        row['recovery']=recovery(row['validation'],lo['validation'],hi['validation'])
        row['rejection_recovery']={key:{c:None if v is None or lo[key][c] is None or hi[key][c] is None
            or hi[key][c]==lo[key][c] else 100*(v-lo[key][c])/(hi[key][c]-lo[key][c])
            for c,v in row[key].items()} for key in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
        historical_U100_reference=True,single_seed=True,adaptive_validation=True)


def run(spec,task,*,device='cuda'):
    started=time.monotonic()
    if task not in TASKS: raise PermissionError('Only registered retained/control tasks')
    campaign.check(spec); job=execution(spec,task,spec['reuse']['u100'])
    # The first job can start while the second sbatch is still returning.
    if submission.accepted_jobs(spec,partial=True).get(task)!=job: raise PermissionError('Wrong exact submitted job')
    folder=Path(spec['root'])/task; folder.mkdir(exist_ok=False)
    log=PhaseLog(folder/'phases',lambda:memory_snapshot(torch))
    with log.phase('load_exact_HLT_views_and_declared_targets'):
        tr,val,bank,meta=load_inputs(spec,task)
    with log.phase('integrated_retained_graph_checks'):
        checks=startup(spec,task,tr,val,bank,meta,device,started)
        validate_startup(spec,task,checks); write_immutable_json(folder/'startup.json',checks)
    gc.collect()
    if torch.device(device).type=='cuda': torch.cuda.empty_cache()
    parents=fit_parents(spec,task)
    def snapshot(epoch,row,state):
        target=folder/'epochs'/('%02d'%epoch); target.mkdir(parents=True,exist_ok=False)
        save_state(target/'state.pt',state)
        write_immutable_json(target/'receipt.json',artifact('EPOCH',parents=dict(parents,recipe=recipe()['content_hash']),
            row=row,state=file_ref(target/'state.pt')))
        campaign.storage(spec)
    with log.phase('twenty_scientific_epochs'):
        report,state,full=training.train(tr,val,bank=bank,node=spec['nodes'][task],task=task,
            parents=parents,device=device,save_epoch=snapshot)
    save_state(folder/'selected.pt',state)
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',chosen.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,task,parents)
    # Deployment requires only HLT tensors and retained weights. Release training targets.
    del bank
    deployed=model.load_deployed(spec['nodes'][task],task,state,device)
    probabilities=predict(deployed,val,device=device,batch_size=128)
    if evaluate_probabilities(val.labels,probabilities)!=report['validation']: raise ValueError('Retained deployment readback differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    write_immutable_json(folder/'training_report.json',report)
    write_immutable_json(folder/'inference_cost.json',inference_cost(spec,task,deployed,val,device))
    validate_fit(spec,task); check_memory(spec,memory_snapshot(torch)); campaign.check(spec)
    if execution(spec,task,spec['reuse']['u100'])!=job: raise PermissionError('Allocation changed')
    rec=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task,job_id=job,
        source_commit=spec['source_commit'],report_hash=report['content_hash'],
        outputs=[file_ref(Path(spec['root'])/p,root=spec['root']) for p in output_names(task)])
    write_immutable_json(folder/'task_receipt.json',rec)
    return rec
