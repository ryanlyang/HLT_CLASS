"""One useful GPU job: authenticate, calibrate, refine and publish read-only comparisons."""
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
from hlt_classification.luka_fullstat_fine_tail import banks,training as old_training
from hlt_classification.luka_fullstat_fine_tail.worker import data as old_data, rejection, longest
from . import campaign,submission,training
from .execution import execution
from .contracts import artifact,validate,recipe,EPOCHS

FIT_FILES=('selected.pt','selected_training_state.pt','completed_training.json',
           'training_report.json','validation_predictions.npz','initializer.pt','initializer.json','startup.json')


def output_names():
    return ['finetune/'+n for n in FIT_FILES]+['finetune/epochs/%02d/%s'%(e,n)
        for e in range(EPOCHS+1) for n in ('state.pt','receipt.json')]


def load_inputs(spec):
    meta=metadata(spec)
    if meta[0]['summaries']!=spec['reuse']['summaries']: raise ValueError('Prepared support changed')
    tr,val=(old_data(spec,role,'D000',authenticated=meta) for role in ('train','validation'))
    parent=load_json(validate_file_ref(spec['reuse']['parent_spec']))
    bank=banks.Bundle(parent,'DE000',tr)
    if bank.manifest!=load_json(validate_file_ref(spec['reuse']['bundle'])):
        raise ValueError('Original seven-teacher bundle changed')
    return tr,val,bank


def reference_arrays(spec,name):
    refs=spec['references'][name]
    report=load_json(validate_file_ref(refs['report']))
    arrays=load_npz_arrays(validate_file_ref(refs['predictions']))
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
        raise ValueError('Saved reference metrics differ')
    return report,arrays


def baseline_readback(spec,val,native,device):
    report,arrays=reference_arrays(spec,'FINE_TAIL_DE000')
    if (report['validation']!=spec['reuse']['parent_validation']
            or not np.array_equal(arrays['identities'],val.identities)
            or not np.array_equal(arrays['labels'],val.labels)):
        raise ValueError('Initializer population or report mismatch')
    model=DelphesParticleTransformer().to(device); model.load_state_dict(native,strict=True)
    actual=predict(model,val,device=device,batch_size=128)
    # Original selected prediction readback uses the same native BF16 path.
    np.testing.assert_allclose(actual,arrays['probabilities'],rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']:
        raise ValueError('Initializer validation metrics do not reproduce')
    return report['validation']


def check_memory(spec,value):
    if (value['gpu_peak_bytes']<=0 or value['peak_rss_bytes']<=0
            or value['gpu_peak_bytes']>=.9*spec['reuse']['gpu']['total_memory_bytes']
            or value['peak_rss_bytes']+8*2**30>=.85*spec['memory_mb']*1024**2):
        raise MemoryError('Refinement exceeds inherited Oscar memory envelope')


def timing(spec,train_step,val_step,preparation):
    if any(not math.isfinite(v) or v<=0 for v in (train_step,val_step)) or not math.isfinite(preparation) or preparation<0:
        raise ValueError('Invalid timing witness')
    # Overbounds three cached-head passes and one frozen feature extraction by
    # four full training passes; also allows initializer/final readbacks.
    minutes=math.ceil(preparation/60+120+1.5*(14*math.ceil(spec['counts']['train']/128)*train_step
        +14*math.ceil(spec['counts']['validation']/128)*val_step)/60)
    if minutes>spec['minutes']['finetune']:
        raise ValueError('Conservative integrated timing exceeds allocated refinement time')
    return dict(train_seconds_per_update=train_step,validation_seconds_per_batch=val_step,
        preparation_seconds=preparation,conservative_minutes=minutes,full_epoch_measured=False)


def startup(spec,tr,val,bank,native,device,preparation_seconds):
    with parity_backend(device): installed_parity(tr,device=device)
    model=training.restore(spec['node'],native,device).eval(); model.capture=True
    ordinary=DelphesParticleTransformer().to(device).eval(); ordinary.load_state_dict(native,strict=True)
    raw=tr.batch(np.arange(min(4,len(tr))))
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            torch.testing.assert_close(_forward(model,raw,device,bf16=bf16),
                _forward(ordinary,raw,device,bf16=bf16),rtol=0,atol=0)
    del ordinary
    model.train(); opt=old_training.optimizer(model,recipe()['objective']['base'])
    idx,max_train=longest(tr); ix=np.full(128,idx,np.int64)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss,ce,*_=old_training.batch_loss(model,tr.batch(ix),bank,ix,'DE000',device)
        expected=torch.autograd.grad(.5*ce,tuple(model.mod.fc.parameters()),retain_graph=True)
        loss.backward(); old_training.finite_gradients(model)
        for p,g in zip(model.mod.fc.parameters(),expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        if any(p.grad is None for p in model.parameters() if p.requires_grad):
            raise ValueError('Disconnected native or auxiliary parameter')
        opt.step(); model.predictions=model.auxiliary_logits=None
        del loss,ce,expected
    vi,max_val=longest(val); model.eval(); model.capture=False
    with torch.inference_mode():
        out=_forward(model,val.batch(np.full(128,vi,np.int64)),device,bf16=str(device).startswith('cuda'))
        if not torch.isfinite(out).all(): raise ValueError('Nonfinite maximum validation support')
    def sync():
        if str(device).startswith('cuda'): torch.cuda.synchronize()
    rng=np.random.default_rng(73012); model.train(); model.capture=True
    for step in range(36):
        if step==4: sync(); begun=time.monotonic()
        ix=rng.integers(0,len(tr),128); opt.zero_grad(set_to_none=True)
        loss,*_=old_training.batch_loss(model,tr.batch(ix),bank,ix,'DE000',device)
        loss.backward(); old_training.finite_gradients(model); opt.step()
        model.predictions=model.auxiliary_logits=None; del loss
    sync(); train_step=(time.monotonic()-begun)/32
    model.eval(); model.capture=False; sync(); begun=time.monotonic()
    with torch.inference_mode():
        for _ in range(16):
            out=_forward(model,val.batch(rng.integers(0,len(val),128)),device,bf16=str(device).startswith('cuda'))
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite validation timing witness')
    sync(); val_step=(time.monotonic()-begun)/16
    memory=memory_snapshot(torch); check_memory(spec,memory)
    value=timing(spec,train_step,val_step,preparation_seconds)
    print('FINETUNE integrated conservative_minutes='+str(value['conservative_minutes']),flush=True)
    return artifact('STARTUP',parents=dict(spec=spec['content_hash'],bank=bank.manifest['content_hash']),
        native_parity=True,installed_weaver_parity=True,main_head_CE_only=True,finite_connected_gradients=True,
        batch=128,stress_steps=3,train_maximum=max_train,validation_maximum=max_val,
        probe_discarded=True,memory=memory,timing=value)


def validate_startup(spec,value,bank_hash):
    validate(value,'STARTUP',dict(spec=spec['content_hash'],bank=bank_hash))
    if (any(value[k] is not True for k in ('native_parity','installed_weaver_parity','main_head_CE_only',
            'finite_connected_gradients','probe_discarded')) or value['batch']!=128 or value['stress_steps']!=3
            or value['train_maximum']!=spec['reuse']['summaries']['train']['D000']['maximum']
            or value['validation_maximum']!=spec['reuse']['summaries']['validation']['D000']['maximum']):
        raise ValueError('Integrated support/parity evidence differs')
    check_memory(spec,value['memory']); t=value['timing']
    if t!=timing(spec,t['train_seconds_per_update'],t['validation_seconds_per_batch'],t['preparation_seconds']):
        raise ValueError('Timing record differs')


def fit_parents(spec,initializer,bank):
    return dict(spec=spec['content_hash'],initializer=initializer['content_hash'],bank=bank,
                prepared=spec['anchor']['evidence']['prepared'])


def validate_fit(spec):
    root=Path(spec['root'])/'finetune'; init=load_json(root/'initializer.json')
    bank=load_json(validate_file_ref(spec['reuse']['bundle']))['content_hash']
    validate(init,'INITIALIZER',dict(spec=spec['content_hash'],bank=bank,
        checkpoint=spec['initial_checkpoint']['sha256']))
    training.validate_calibration(init['calibration'],spec['counts']['train'])
    if init['state']!=file_ref(root/'initializer.pt'): raise ValueError('Initializer bytes differ')
    initial=torch.load(validate_file_ref(init['state']),map_location='cpu',weights_only=True)
    original=torch.load(validate_file_ref(spec['initial_checkpoint']),map_location='cpu',weights_only=True)
    m=training.restore(spec['node'],original,'cpu'); m.load_state_dict(initial,strict=True)
    if any(not torch.equal(v,m.native_state()[k]) for k,v in original.items()):
        raise ValueError('Calibrated initializer changed native state')
    validate_startup(spec,load_json(root/'startup.json'),bank)
    report=load_json(root/'training_report.json'); parents=fit_parents(spec,init,bank)
    training.validate_report(report,spec,parents)
    if load_json(root/'completed_training.json')!=report: raise ValueError('Completed report changed')
    for epoch in range(EPOCHS+1):
        folder=root/'epochs'/('%02d'%epoch); rec=load_json(folder/'receipt.json')
        validate(rec,'EPOCH',dict(parents,recipe=recipe()['content_hash']))
        expected=report['validation_history'][epoch-1] if epoch else dict(pass_number=0,
            absolute_epoch=20,update=0,validation=report['baseline'])
        if rec['row']!=expected or rec['state']!=file_ref(folder/'state.pt'):
            raise ValueError('Completed epoch snapshot binding differs')
    chosen=root/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    if file_ref(chosen)['sha256']!=file_ref(root/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected full-state bytes differ from selected epoch')
    full=torch.load(chosen,map_location='cpu',weights_only=True)
    native=torch.load(root/'selected.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(full['student'],strict=True)
    if (native.keys()!=m.native_state().keys() or any(not torch.equal(v,m.native_state()[k]) for k,v in native.items())
            or full['epoch']!=report['selected_pass']): raise ValueError('Selected native/full state differs')
    arrays=load_npz_arrays(root/'validation_predictions.npz')
    _,reference=reference_arrays(spec,'FINE_TAIL_DE000')
    if (not np.array_equal(arrays['identities'],reference['identities'])
            or not np.array_equal(arrays['labels'],reference['labels'])
            or evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']):
        raise ValueError('Saved selected probabilities or row ordering differ')
    return report,arrays


def completed(spec,*,accounting=True):
    root=Path(spec['root']); path=root/'finetune/task_receipt.json'
    if not path.exists(): return None
    rec=load_json(path); validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id']!='finetune' or rec['job_id']!=submission.accepted_jobs(spec)['finetune']
            or rec['source_commit']!=spec['source_commit'] or [r['path'] for r in rec['outputs']]!=output_names()):
        raise ValueError('Completed task identity or output inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=root)
    report,_=validate_fit(spec)
    if rec['report_hash']!=report['content_hash']: raise ValueError('Task/report binding differs')
    if accounting: submission.successful(spec,'finetune',rec['job_id'])
    return rec


def results(spec):
    campaign.check(spec)
    if completed(spec) is None: raise ValueError('Refinement has not published a completed task')
    report,arrays=validate_fit(spec); rows=[]
    items=[(n,*reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','FINE_TAIL_DE000')]
    items.append(('FINETUNED_DE000',report,arrays))
    for name,r,a in items:
        if (not np.array_equal(a['identities'],arrays['identities']) or not np.array_equal(a['labels'],arrays['labels'])):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=r['selected_pass'],validation=r['validation'],
            rejection_30=rejection(a['labels'],a['probabilities'],.3),rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi=rows[0],rows[1]
    for row in rows:
        row['recovery']=recovery(row['validation'],lo['validation'],hi['validation'])
        row['delta_parent_auc']=row['validation']['macro_ovr_auc']-rows[2]['validation']['macro_ovr_auc']
        row['rejection_recovery']={key:{c:None if v is None or lo[key][c] is None or hi[key][c] is None
            or hi[key][c]==lo[key][c] else 100*(v-lo[key][c])/(hi[key][c]-lo[key][c])
            for c,v in row[key].items()} for key in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],report=report['content_hash']),rows=rows,
        baseline_retained=report['baseline_retained'],best_new_validation=report['best_new_validation'],
        best_new_added_epoch=report['best_new_pass'],single_seed=True,adaptive_validation=True,
        new_epoch_numbers_are_additional=True)


def run(spec,task,*,device='cuda'):
    started=time.monotonic()
    if task!='finetune': raise PermissionError('Only one refinement task; no test/preflight operation')
    campaign.check(spec); job=execution(spec,spec['reuse'])
    if submission.accepted_jobs(spec).get(task)!=job: raise PermissionError('Wrong exact submitted job')
    folder=Path(spec['root'])/'finetune'; folder.mkdir(exist_ok=False)
    log=PhaseLog(folder/'phases',lambda:memory_snapshot(torch))
    with log.phase('load_original_HLT_and_seven_readonly_banks'):
        tr,val,bank=load_inputs(spec)
        native=torch.load(validate_file_ref(spec['initial_checkpoint']),map_location='cpu',weights_only=True)
        baseline=baseline_readback(spec,val,native,device)
    with log.phase('integrated_technical_checks'):
        checks=startup(spec,tr,val,bank,native,device,time.monotonic()-started)
        validate_startup(spec,checks,bank.manifest['content_hash']); write_immutable_json(folder/'startup.json',checks)
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    with log.phase('frozen_native_head_calibration'):
        model=training.restore(spec['node'],native,device)
        initial,evidence=training.calibrate(model,tr,bank,spec['node'],device)
        training.validate_calibration(evidence,len(tr)); save_state(folder/'initializer.pt',initial)
        init=artifact('INITIALIZER',parents=dict(spec=spec['content_hash'],bank=bank.manifest['content_hash'],
            checkpoint=spec['initial_checkpoint']['sha256']),calibration=evidence,state=file_ref(folder/'initializer.pt'))
        write_immutable_json(folder/'initializer.json',init)
    del model; gc.collect()
    parents=fit_parents(spec,init,bank.manifest['content_hash'])
    def snapshot(epoch,row,state):
        target=folder/'epochs'/('%02d'%epoch); target.mkdir(parents=True,exist_ok=False)
        save_state(target/'state.pt',state)
        write_immutable_json(target/'receipt.json',artifact('EPOCH',parents=dict(parents,recipe=recipe()['content_hash']),
            row=row,state=file_ref(target/'state.pt')))
    with log.phase('ten_additional_scientific_epochs'):
        report,state,full=training.train(tr,val,bank,spec['node'],initial,baseline,parents,device,save_epoch=snapshot)
    # Persist completed work before expensive finalization/readback can fail.
    save_state(folder/'selected.pt',state)
    selected_file=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',selected_file.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,parents)
    restored=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    ordinary=DelphesParticleTransformer().to(device); ordinary.load_state_dict(restored,strict=True)
    probabilities=predict(ordinary,val,device=device,batch_size=128)
    if evaluate_probabilities(val.labels,probabilities)!=report['validation']:
        raise ValueError('Selected native inference readback differs')
    atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=probabilities)))
    write_immutable_json(folder/'training_report.json',report)
    validate_fit(spec); check_memory(spec,memory_snapshot(torch)); campaign.check(spec)
    if execution(spec,spec['reuse'])!=job: raise PermissionError('Allocation changed')
    rec=artifact('TASK',parents=dict(spec=spec['content_hash']),task_id=task,job_id=job,
        source_commit=spec['source_commit'],report_hash=report['content_hash'],
        outputs=[file_ref(Path(spec['root'])/p,root=spec['root']) for p in output_names()])
    write_immutable_json(folder/'task_receipt.json',rec)
    return rec
