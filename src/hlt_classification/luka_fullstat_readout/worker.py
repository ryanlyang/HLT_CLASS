"""Authenticate, check the new readout path, fit once, and publish native results."""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward, predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullstat_attention.worker import metadata
from hlt_classification.luka_fullstat_u100_embedding import banks
from hlt_classification.luka_fullstat_u100_embedding.worker import data, longest
from hlt_classification.luka_fullstat_fine_tail.worker import rejection
from . import campaign, submission, training, model
from .execution import execution
from .contracts import artifact, validate, recipe, teacher_replay

TASK = 'train_readout'
FIT_FILES = ('selected.pt','selected_training_state.pt','completed_training.json',
             'training_report.json','validation_predictions.npz','readout.pt','readout.json','startup.json')


def output_names():
    return [TASK+'/'+n for n in FIT_FILES]+[TASK+'/epochs/%02d/%s'%(e,n)
        for e in range(1,21) for n in ('state.pt','receipt.json')]


def load_inputs(spec):
    meta = metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Prepared support changed')
    tr, val = (data(spec,r,'D000',authenticated=meta) for r in ('train','validation'))
    directory = Path(validate_file_ref(spec['bank_manifest'])).parent
    bank = banks.Bank(directory, tr, refs=spec['teacher_refs'])
    if bank.manifest['content_hash'] != spec['reuse']['bank_hash']:
        raise ValueError('Original TRAIN bank changed')
    return tr, val, bank, meta


def readout_parents(spec):
    return dict(spec=spec['content_hash'], bank=spec['reuse']['bank_hash'],
        checkpoint=spec['teacher_refs']['checkpoint']['sha256'], report=spec['teacher_refs']['report']['sha256'])


def save_readout(spec, bank, device):
    folder = Path(spec['root'])/TASK
    readout = model.make_readout(bank, Path(spec['bank_manifest']['path']).parent, spec['teacher_refs'], device)
    save_state(folder/'readout.pt', training.cpu_state(readout.state_dict()))
    rec = artifact('READOUT', parents=readout_parents(spec),
        role='train', teacher='COARSE_U100', payload=file_ref(folder/'readout.pt'),
        precision='FP32', trainable_parameters=0, native_classifier='mod.fc.0')
    write_immutable_json(folder/'readout.json', rec)
    return readout, rec


def validate_readout(spec):
    folder = Path(spec['root'])/TASK
    rec = load_json(folder/'readout.json'); validate(rec,'READOUT',readout_parents(spec))
    if (rec['role']!='train' or rec['teacher']!='COARSE_U100' or rec['precision']!='FP32'
            or rec['trainable_parameters']!=0 or rec['native_classifier']!='mod.fc.0'
            or rec['payload']!=file_ref(folder/'readout.pt')):
        raise ValueError('Frozen readout identity differs')
    manifest = load_json(validate_file_ref(spec['bank_manifest']))
    if manifest['content_hash'] != spec['reuse']['bank_hash']: raise ValueError('Original bank hash differs')
    expected = model.make_readout(SimpleNamespace(manifest=manifest),
        Path(spec['bank_manifest']['path']).parent, spec['teacher_refs'], 'cpu').state_dict()
    state = torch.load(validate_file_ref(rec['payload']), map_location='cpu', weights_only=True)
    if state.keys()!=expected.keys() or any(not torch.equal(v,expected[k]) for k,v in state.items()):
        raise ValueError('Readout classifier/normalization differs from original teacher/bank')
    return rec


def teacher_batches(rows, maximum_row):
    if (type(rows) is not int or rows<=0 or type(maximum_row) is not int
            or not 0<=maximum_row<rows):
        raise ValueError('Invalid teacher TRAIN replay population/witness')
    policy=teacher_replay(); batch=policy['batch_size']; count=math.ceil(rows/batch)
    selected=np.unique(np.r_[np.linspace(0,count-1,min(policy['spread_batches'],count),dtype=np.int64),
                               maximum_row//batch])
    return [dict(start=int(i)*batch,stop=min((int(i)+1)*batch,rows)) for i in selected]


def replay_backend(device):
    # Record, do not mutate: the historical banks.export uses ordinary FP32 eval.
    # Its manifest did not record backend flags; these describe this replay only.
    return dict(device=torch.device(device).type, autocast=False,
        cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def teacher_parity(spec, bank, readout, meta, device):
    policy=teacher_replay()
    if spec['teacher_replay']!=policy: raise ValueError('Teacher replay policy differs')
    u = data(spec,'train','U100',authenticated=meta)
    if (u.role!='train' or u.coordinate_name!='U100' or len(u)!=spec['counts']['train']
            or not np.array_equal(u.identities,bank.identities) or not np.array_equal(u.labels,bank.labels)):
        raise ValueError('Teacher TRAIN row join differs')
    teacher = DelphesParticleTransformer().to(device).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(validate_file_ref(spec['teacher_refs']['checkpoint']),
                                      map_location='cpu',weights_only=True),strict=True)
    index, maximum = longest(u)
    batches=teacher_batches(len(u),int(index)); backend=replay_backend(device)
    errors=dict(embedding_max_abs=0.,logit_max_abs=0.)
    print('READOUT teacher replay backend='+json.dumps(backend,sort_keys=True),flush=True)
    # Reproduce the exporter's contiguous batch geometry (including dynamic
    # padding). Strict parity_backend belongs only around fresh-vs-fresh probes:
    # disabling TF32 here compares the historical bank to a different computation.
    with torch.inference_mode(),torch.autocast(torch.device(device).type,enabled=False):
        for span in batches:
            ids=np.arange(span['start'],span['stop']); raw=u.batch(ids)
            tensors={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
            logits,h=model.head_input(teacher,**tensors)
            z=torch.from_numpy(bank.targets[ids]).to(device)
            reconstructed=readout.mean+readout.scale*z; replayed=readout(z)
            values=(reconstructed,h,logits,replayed)
            if any(not torch.isfinite(v).all() for v in values):
                raise ValueError('Nonfinite teacher/readout TRAIN replay')
            observed=dict(embedding_max_abs=float((reconstructed-h.float()).abs().max()),
                          logit_max_abs=float((replayed-logits.float()).abs().max()))
            print('READOUT teacher replay '+json.dumps(dict(span,padded_particles=raw['mask'].shape[-1],
                  **observed),sort_keys=True),flush=True)
            torch.testing.assert_close(reconstructed,h.float(),rtol=policy['rtol'],atol=policy['atol'])
            torch.testing.assert_close(replayed,logits.float(),rtol=policy['rtol'],atol=policy['atol'])
            errors={k:max(errors[k],v) for k,v in observed.items()}
    if backend!=replay_backend(device): raise ValueError('Teacher replay changed backend settings')
    return dict(train_rows=[i for b in batches for i in range(b['start'],b['stop'])],
        teacher_batches=batches,teacher_maximum_row=int(index),teacher_replay_hash=policy['content_hash'],
        teacher_replay_backend=backend,teacher_replay_errors=errors,
        teacher_train_maximum=maximum,teacher_readout_parity=True)


def check_memory(spec,value):
    if (value['gpu_peak_bytes']<=0 or value['peak_rss_bytes']<=0
            or value['gpu_peak_bytes']>=.9*spec['reuse']['gpu']['total_memory_bytes']
            or value['peak_rss_bytes']+8*2**30>=.85*spec['memory_mb']*1024**2):
        raise MemoryError('Readout exceeds inherited Oscar memory envelope')


def timing(spec,train_step,val_step,preparation):
    if any(not math.isfinite(v) or v<=0 for v in (train_step,val_step)) or not math.isfinite(preparation) or preparation<0:
        raise ValueError('Invalid timing witness')
    minutes=math.ceil(preparation/60+120+1.5*(20*math.ceil(spec['counts']['train']/128)*train_step
        +22*math.ceil(spec['counts']['validation']/128)*val_step)/60)
    if minutes>spec['minutes'][TASK]: raise ValueError('Measured conservative timing exceeds allocation')
    return dict(train_seconds_per_update=train_step,validation_seconds_per_batch=val_step,
        preparation_seconds=preparation,conservative_minutes=minutes,full_epoch_measured=False)


def startup(spec,tr,val,bank,readout,readout_record,meta,device,started):
    witness=teacher_parity(spec,bank,readout,meta,device)
    with parity_backend(device): installed_parity(tr,device=device)
    student=model.new_model(spec['node']).to(device).eval(); student.capture=True
    ordinary=DelphesParticleTransformer().to(device).eval()
    ordinary.load_state_dict(student.native_state(),strict=True)
    raw=tr.batch(np.arange(min(4,len(tr))))
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            torch.testing.assert_close(_forward(student,raw,device,bf16=bf16),
                _forward(ordinary,raw,device,bf16=bf16),rtol=0,atol=0)
    del ordinary
    student.train(); opt=training.optimizer(student,recipe()['base'])
    index,maximum=longest(tr); ix=np.full(128,index,np.int64)
    frozen=training.cpu_state(readout.state_dict())
    if list(readout.parameters()): raise ValueError('Readout has trainable parameters')
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss,ce,aux,rc,_=model.batch_loss(student,tr.batch(ix),bank,ix,readout,device,3.)
        main=torch.autograd.grad(ce,tuple(student.mod.fc.parameters()),retain_graph=True)
        rg=torch.autograd.grad(rc,student.prediction,retain_graph=True)[0]
        if not torch.isfinite(rg).all() or not torch.any(rg!=0): raise ValueError('Readout input gradient disconnected')
        loss.backward(); training.finite_gradients(student)
        for p,g in zip(student.mod.fc.parameters(),main): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        opt.step(); student.prediction=None
    if any(not torch.equal(v.cpu(),frozen[k]) for k,v in readout.state_dict().items()):
        raise ValueError('Frozen readout mutated')
    vi,vmaximum=longest(val); student.eval(); student.capture=False
    with torch.inference_mode():
        out=_forward(student,val.batch(np.full(128,vi,np.int64)),device,bf16=str(device).startswith('cuda'))
        if not torch.isfinite(out).all(): raise ValueError('Invalid largest validation forward')
    def sync():
        if str(device).startswith('cuda'): torch.cuda.synchronize()
    rng=np.random.default_rng(73012); student.train(); student.capture=True
    for step in range(68):
        if step==4: sync(); begun=time.monotonic()
        ids=rng.integers(0,len(tr),128); opt.zero_grad(set_to_none=True)
        loss,*_=model.batch_loss(student,tr.batch(ids),bank,ids,readout,device,3.)
        loss.backward(); training.finite_gradients(student); opt.step(); student.prediction=None
    sync(); train_step=(time.monotonic()-begun)/64
    student.eval(); student.capture=False; sync(); begun=time.monotonic()
    with torch.inference_mode():
        for _ in range(16):
            out=_forward(student,val.batch(rng.integers(0,len(val),128)),device,bf16=str(device).startswith('cuda'))
            if not torch.isfinite(out).all(): raise ValueError('Invalid validation timing witness')
    sync(); val_step=(time.monotonic()-begun)/16
    memory=memory_snapshot(torch); check_memory(spec,memory)
    value=timing(spec,train_step,val_step,time.monotonic()-started)
    print('READOUT integrated conservative_minutes='+str(value['conservative_minutes']),flush=True)
    return artifact('STARTUP',parents=dict(spec=spec['content_hash'],readout=readout_record['content_hash']),
        **witness,native_parity=True,installed_weaver_parity=True,main_head_CE_only=True,
        readout_frozen=True,readout_input_gradient=True,finite_connected_gradients=True,
        batch=128,stress_steps=3,train_maximum=maximum,validation_maximum=vmaximum,
        probe_discarded=True,memory=memory,timing=value)


def validate_startup(spec,value,readout_record):
    validate(value,'STARTUP',dict(spec=spec['content_hash'],readout=readout_record['content_hash']))
    validate_teacher_witness(spec,value)
    if (any(value[k] is not True for k in ('native_parity','installed_weaver_parity','main_head_CE_only',
            'readout_frozen','readout_input_gradient','finite_connected_gradients','probe_discarded','teacher_readout_parity'))
            or value['batch']!=128 or value['stress_steps']!=3
            or value['train_maximum']!=spec['reuse']['summaries']['train']['D000']['maximum']
            or value['validation_maximum']!=spec['reuse']['summaries']['validation']['D000']['maximum']
            or value['teacher_train_maximum']!=spec['reuse']['summaries']['train']['U100']['maximum']):
        raise ValueError('Integrated support/parity evidence differs')
    check_memory(spec,value['memory']); t=value['timing']
    if t!=timing(spec,t['train_seconds_per_update'],t['validation_seconds_per_batch'],t['preparation_seconds']):
        raise ValueError('Timing record differs')


def validate_teacher_witness(spec,value):
    policy=teacher_replay()
    batches=teacher_batches(spec['counts']['train'],value['teacher_maximum_row'])
    if (spec['teacher_replay']!=policy or value['teacher_replay_hash']!=policy['content_hash']
            or value['teacher_batches']!=batches
            or value['train_rows']!=[i for b in batches for i in range(b['start'],b['stop'])]
            or value['teacher_readout_parity'] is not True):
        raise ValueError('Teacher replay witness differs')
    backend=value['teacher_replay_backend']; expected=replay_backend('cpu')
    if (backend.keys()!=expected.keys() or backend['device'] not in ('cpu','cuda')
            or backend['autocast'] is not False
            or backend['float32_matmul_precision'] not in ('highest','high','medium')
            or any(type(backend[k]) is not bool for k,v in expected.items() if type(v) is bool)):
        raise ValueError('Invalid teacher replay backend record')
    errors=value['teacher_replay_errors']
    if (set(errors)!={'embedding_max_abs','logit_max_abs'}
            or any(not math.isfinite(v) or v<0 for v in errors.values())):
        raise ValueError('Invalid teacher replay error record')


def fit_parents(spec,readout_record):
    return dict(spec=spec['content_hash'],readout=readout_record['content_hash'],
        bank=spec['reuse']['bank_hash'],prepared=spec['anchor']['evidence']['prepared'])


def reference_arrays(spec,name):
    refs=spec['references'][name]
    report=load_json(validate_file_ref(refs['report']))
    arrays=load_npz_arrays(validate_file_ref(refs['predictions']))
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
        raise ValueError('Historical reference metrics differ')
    return report,arrays


def validate_fit(spec):
    folder=Path(spec['root'])/TASK; readout_record=validate_readout(spec)
    validate_startup(spec,load_json(folder/'startup.json'),readout_record)
    parents=fit_parents(spec,readout_record); report=load_json(folder/'training_report.json')
    training.validate_report(report,spec,parents)
    if load_json(folder/'completed_training.json')!=report: raise ValueError('Completed training changed')
    for epoch in range(1,21):
        target=folder/'epochs'/('%02d'%epoch); rec=load_json(target/'receipt.json')
        validate(rec,'EPOCH',dict(parents,recipe=recipe()['content_hash']))
        if rec['row']!=report['validation_history'][epoch-1] or rec['state']!=file_ref(target/'state.pt'):
            raise ValueError('Epoch snapshot binding differs')
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    if file_ref(chosen)['sha256']!=file_ref(folder/'selected_training_state.pt')['sha256']:
        raise ValueError('Selected full state differs from selected epoch')
    full=torch.load(chosen,map_location='cpu',weights_only=True)
    native=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    student=model.new_model(spec['node']); student.load_state_dict(full['student'],strict=True)
    expected=student.native_state()
    if native.keys()!=expected.keys() or any(not torch.equal(v,expected[k]) for k,v in native.items()) or full['epoch']!=report['selected_pass']:
        raise ValueError('Selected native/full state differs')
    arrays=load_npz_arrays(folder/'validation_predictions.npz'); _,reference=reference_arrays(spec,'U100_W5')
    if (not np.array_equal(arrays['identities'],reference['identities']) or not np.array_equal(arrays['labels'],reference['labels'])
            or evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']):
        raise ValueError('Selected prediction metrics/population differ')
    return report,arrays


def completed(spec,*,accounting=True):
    root=Path(spec['root']); rec=load_json(root/TASK/'task_receipt.json')
    validate(rec,'TASK',dict(spec=spec['content_hash']))
    if (rec['task_id']!=TASK or rec['job_id']!=submission.accepted_jobs(spec)[TASK]
            or rec['source_commit']!=spec['source_commit'] or [r['path'] for r in rec['outputs']]!=output_names()):
        raise ValueError('Completed task identity/inventory differs')
    for ref in rec['outputs']: validate_file_ref(ref,root=root)
    report,_=validate_fit(spec)
    if rec['report_hash']!=report['content_hash']: raise ValueError('Task/report binding differs')
    if accounting: submission.successful(spec,TASK,rec['job_id'])
    return rec


def results(spec):
    campaign.check(spec); completed(spec)
    report,arrays=validate_fit(spec); rows=[]
    items=[(n,*reference_arrays(spec,n)) for n in ('M0HLT','OFFLINE','U100_W5')]
    items.append(('U100_READOUT_CE',report,arrays))
    for name,r,a in items:
        if not np.array_equal(a['identities'],arrays['identities']) or not np.array_equal(a['labels'],arrays['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name,selected_epoch=r['selected_pass'],validation=r['validation'],
            rejection_30=rejection(a['labels'],a['probabilities'],.3),rejection_50=rejection(a['labels'],a['probabilities'],.5)))
    lo,hi=rows[0],rows[1]
    for row in rows:
        row['recovery']=recovery(row['validation'],lo['validation'],hi['validation'])
        row['rejection_recovery']={key:{c:None if v is None or lo[key][c] is None or hi[key][c] is None
            or hi[key][c]==lo[key][c] else 100*(v-lo[key][c])/(hi[key][c]-lo[key][c])
            for c,v in row[key].items()} for key in ('rejection_30','rejection_50')}
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],report=report['content_hash']),rows=rows,
        historical_control=True,single_seed=True,adaptive_validation=True)


def run(spec,task,*,device='cuda'):
    started=time.monotonic()
    if task!=TASK: raise PermissionError('Only registered HLT fit; no test/preflight operation')
    campaign.check(spec); job=execution(spec,spec['reuse'])
    if submission.accepted_jobs(spec).get(task)!=job: raise PermissionError('Wrong exact submitted job')
    folder=Path(spec['root'])/TASK; folder.mkdir(exist_ok=False)
    log=PhaseLog(folder/'phases',lambda:memory_snapshot(torch))
    with log.phase('load_original_HLT_and_U100_TRAIN_bank'):
        tr,val,bank,meta=load_inputs(spec)
        readout,record=save_readout(spec,bank,device)
    with log.phase('integrated_readout_checks'):
        checks=startup(spec,tr,val,bank,readout,record,meta,device,started)
        validate_startup(spec,checks,record); write_immutable_json(folder/'startup.json',checks)
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    parents=fit_parents(spec,record)
    def snapshot(epoch,row,state):
        target=folder/'epochs'/('%02d'%epoch); target.mkdir(parents=True,exist_ok=False)
        save_state(target/'state.pt',state)
        write_immutable_json(target/'receipt.json',artifact('EPOCH',parents=dict(parents,recipe=recipe()['content_hash']),
            row=row,state=file_ref(target/'state.pt')))
        campaign.storage(spec)
    with log.phase('twenty_scientific_epochs'):
        report,native,full=training.train(tr,val,bank=bank,node=spec['node'],readout=readout,
            parents=parents,device=device,save_epoch=snapshot)
    # Save expensive completed work before readback or publication can fail.
    save_state(folder/'selected.pt',native)
    chosen=folder/'epochs'/('%02d'%report['selected_pass'])/'state.pt'
    atomic_publish_bytes(folder/'selected_training_state.pt',chosen.read_bytes())
    write_immutable_json(folder/'completed_training.json',report)
    training.validate_report(report,spec,parents)
    ordinary=DelphesParticleTransformer().to(device)
    ordinary.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    probabilities=predict(ordinary,val,device=device,batch_size=128)
    if evaluate_probabilities(val.labels,probabilities)!=report['validation']: raise ValueError('Native readback differs')
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
