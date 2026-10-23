"""Real data technical gate, two independent fits and saved-only partial reporting."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import math
import shutil
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.runner import _forward,predict
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities,recovery
from hlt_classification.luka_attention.worker import data,execution
from hlt_classification.luka_attention.banks import counts
from hlt_classification.luka_u100_embedding.worker import teacher_readback,validate_predictions
from hlt_classification.luka_fullsim import regular_cache,alljets_data
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from .contracts import artifact,validate,recipe,ARMS
from . import banks,training,model,campaign,submission,source


def fit(directory,spec,arm,train,val,bank,*,device,technical=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    student=model.new_model(spec['nodes'][arm])
    report,state=training.train(student,train,val,bank,spec['nodes'][arm],device=device,technical=technical)
    training.validate_report(report,spec,arm,bank.manifest['content_hash'],technical=technical)
    expected=predict(student,val,device=device,batch_size=128)
    save_state(directory/'selected.pt',state)
    saved=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if saved.keys()!=state.keys() or any(not torch.equal(saved[k],state[k]) for k in state):
        raise ValueError('Selected native state readback differs')
    del student
    native=DelphesParticleTransformer().to(device).eval(); native.load_state_dict(saved,strict=True)
    actual=predict(native,val,device=device,batch_size=128)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    if evaluate_probabilities(val.labels,actual)!=report['validation']:
        raise ValueError('Native selected validation metrics differ')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(dict(
        identities=val.identities,labels=val.labels,probabilities=actual)))
    write_immutable_json(directory/'training_report.json',report)
    receipt=artifact('FIT',parents=dict(spec=spec['content_hash'],bank=bank.manifest['content_hash']),
        arm=arm,technical=technical,checkpoint_probability_roundtrip=True,
        outputs=[file_ref(directory/n,root=directory) for n in ('selected.pt','training_report.json','validation_predictions.npz')])
    write_immutable_json(directory/'receipt.json',receipt)
    del native; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return report,receipt


def validate_fit(directory,spec,arm,bank_hash,*,technical=False):
    directory=Path(directory); receipt=load_json(directory/'receipt.json')
    validate(receipt,'FIT',dict(spec=spec['content_hash'],bank=bank_hash))
    if (receipt['arm']!=arm or receipt['technical'] is not technical or not receipt['checkpoint_probability_roundtrip']
            or [r['path'] for r in receipt['outputs']]!=['selected.pt','training_report.json','validation_predictions.npz']):
        raise ValueError('Fit publication differs')
    for ref in receipt['outputs']: validate_file_ref(ref,root=directory)
    report=load_json(directory/'training_report.json')
    training.validate_report(report,spec,arm,bank_hash,technical=technical)
    arrays=load_npz_arrays(directory/'validation_predictions.npz'); validate_predictions(arrays,spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
        raise ValueError('Saved metrics differ')
    return report,arrays


def snapshot_references(root,spec,val,meta):
    root=Path(root); root.mkdir(exist_ok=False)
    refs={n:spec['anchor']['evidence']['models'][n]['predictions'] for n in ('M0HLT','OFFLINE')}
    refs.update(COARSE_U100=spec['teacher']['files']['predictions'],
        U100_EMBED_W5=spec['reference']['files']['predictions'],
        OFFLINE_EMBED_W5=spec['reference']['reference']['files']['predictions'])
    for name,ref in refs.items():
        validate_file_ref(ref); a=load_npz_arrays(ref['path']); validate_predictions(a,len(val))
        if not np.array_equal(a['identities'],val.identities) or not np.array_equal(a['labels'],val.labels):
            raise ValueError('Reference population differs')
        atomic_publish_bytes(root/(name+'.npz'),deterministic_npz_bytes(a)); validate_file_ref(ref)
    ix=alljets_data.comparison_indices(meta[2],meta[3],val.identities,val.labels)
    atomic_publish_bytes(root/'original50k.npz',deterministic_npz_bytes(dict(indices=ix)))


def native_parity(spec,train,device):
    checks={}
    # Native production graph equality and gradient parity, with head RNG isolated.
    raw=train.batch(np.arange(min(4,len(train))))
    for bf16 in (False,True):
        torch.manual_seed(spec['nodes']['aligned']['initialization_seed'])
        native=DelphesParticleTransformer().to(device).eval()
        student=model.new_model(spec['nodes']['aligned']).to(device).eval(); student.capture=True
        for k,v in student.native_state().items():
            torch.testing.assert_close(v,native.state_dict()[k].cpu(),rtol=0,atol=0)
        y1=_forward(native,raw,device,bf16=bf16); y2=_forward(student,raw,device,bf16=bf16)
        torch.testing.assert_close(y1,y2,rtol=0,atol=0)
        y1.square().mean().backward(); y2.square().mean().backward()
        for k,p in native.named_parameters():
            other=dict(student.named_parameters())[k].grad
            if p.grad is None:
                if other is not None: raise ValueError('Gradient topology differs')
            else: torch.testing.assert_close(p.grad,other,rtol=0,atol=0)
        checks['native_parity_bf16_'+str(bf16)]=True
        del native,student; gc.collect()
    return checks


def parity_and_stress(spec,train,val,bank,device):
    with parity_backend(device):
        checks=native_parity(spec,train,device)
    longest=int(np.argmax(counts(train))); ix=np.full(128,longest,dtype=np.int64)
    raw=train.batch(ix); target=torch.from_numpy(bank.batch(ix,raw['mask'].shape[-1])).to(device)
    mask=torch.from_numpy(raw['mask']).to(device); labels=torch.from_numpy(raw['labels']).to(device)
    for arm in ARMS:
        student=model.new_model(spec['nodes'][arm]).to(device).train(); student.capture=True
        opt=torch.optim.AdamW(student.parameters(),lr=1e-4)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss,_,_=model.objective(_forward(student,raw,device,bf16=True),labels,student.prediction,target,mask,arm)
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None):
                raise ValueError('Maximum-support gradient nonfinite')
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in student.predictor.parameters()):
                raise ValueError('Prediction head receives no gradients')
            opt.step()
        student.eval(); student.capture=False; student.prediction=None
        maximum=val.batch(np.full(128,int(np.argmax(counts(val))),dtype=np.int64))
        with torch.inference_mode():
            if not torch.isfinite(_forward(student,maximum,device,bf16=True)).all():
                raise ValueError('Maximum validation support nonfinite')
        checks[arm+'_max_train_tokens']=int(counts(train).max())
        checks[arm+'_max_validation_tokens']=int(counts(val).max())
        checks[arm+'_batch']=128
        del student,opt; gc.collect(); torch.cuda.empty_cache()
    return checks


def projected_minutes(setup_seconds,history):
    epoch=history[0]
    return math.ceil((1.75*(setup_seconds+30*(epoch['train_seconds']+epoch['validation_seconds'])
                           +2*epoch['validation_seconds'])+600)/60)


def scratch_directory(spec):
    """Explicit registered disk scratch; never infer a fallback from environment."""
    base=Path(spec['scratch_base'])
    if not base.is_absolute() or not base.is_dir() or base.absolute()!=base.resolve(strict=True):
        raise PermissionError('Registered canonical node-local scratch is unavailable')
    if (base.is_relative_to(Path(spec['root']).parent)
            or base.is_relative_to(Path(spec['anchor']['evidence']['raw_container']))):
        raise PermissionError('Scratch must not be inside shared checkpoints/dataset')
    # Adapt the established SPORC attention-weight scratch mount check. tmpfs is
    # deliberately excluded: these large target banks must not consume RAM quota.
    mounts=[]
    for line in mountinfo().splitlines():
        left,right=line.split(' - ',1)
        mount=Path(left.split()[4].replace('\\040',' '))
        if base.is_relative_to(mount): mounts.append((len(mount.parts),right.split()[0]))
    if not mounts or max(mounts)[1] not in ('ext4','xfs','btrfs','overlay'):
        raise PermissionError('Token-bank scratch must be node-local disk, not Ceph/NFS/tmpfs')
    return base


def mountinfo():
    return Path('/proc/self/mountinfo').read_text()


def retained_bank(root,spec):
    """Authenticate retained descriptors, not the explicitly retired scratch payload."""
    root=Path(root); m=load_json(root/'bank_manifest.json')
    validate(m,'BANK',dict(recipe=recipe()['content_hash'],teacher=spec['teacher']['content_hash'],
                         prepared=spec['anchor']['evidence']['prepared']))
    if ((m['role'],m['coordinate'],m['width'],m['dtype'],m['rows'])!=('train','U100',128,'float32',spec['counts']['train'])
            or m['checkpoint']!=spec['teacher']['files']['checkpoint']
            or [r['path'] for r in m['outputs']]!=['tokens.npy','index.npz']):
        raise ValueError('Retained bank semantics differ')
    actual=file_ref(root/'bank_index.npz')
    if any(actual[k]!=m['outputs'][1][k] for k in ('bytes','sha256')):
        raise ValueError('Retained bank index hash differs')
    a=load_npz_arrays(root/'bank_index.npz'); n=spec['counts']['train']
    if (a['offsets'].dtype!=np.int64 or a['offsets'].shape!=(n+1,) or a['offsets'][0]!=0
            or a['offsets'][-1]!=m['tokens'] or np.any(np.diff(a['offsets'])<1) or np.any(np.diff(a['offsets'])>512)
            or a['identities'].dtype!=np.uint8 or a['identities'].shape!=(n,32)
            or len(np.unique(a['identities'],axis=0))!=n or a['labels'].dtype!=np.int64 or a['labels'].shape!=(n,)
            or any(a[k].dtype!=np.float64 or a[k].shape!=(128,) or not np.isfinite(a[k]).all() for k in ('mean','scale'))
            or np.any(a['scale']<1e-3)):
        raise ValueError('Retained bank index/moments differ')
    return m


def validate_preflight(spec,*,accounting=True):
    root=Path(spec['root'])/'preflight'; r=load_json(root/'receipt.json')
    validate(r,'PREFLIGHT',dict(spec=spec['content_hash']))
    if (r['technical_checks_passed'] is not True or r['environment']!=spec['anchor']['evidence']['environment']
            or set(r['projections'])!=set(ARMS) or not r['scratch_payloads_retired']
            or r['setup_seconds']<=0 or not math.isfinite(r['setup_seconds'])
            or r['checks'].get('native_parity_bf16_False') is not True
            or r['checks'].get('native_parity_bf16_True') is not True
            or any(r['checks'].get(a+'_batch')!=128 for a in ARMS)
            or not 0<r['memory']['peak_rss_bytes']<.85*spec['memory_mb']*1024**2
            or not 0<r['memory']['gpu_peak_bytes']<.90*r['gpu']['total_memory_bytes']):
        raise ValueError('Preflight technical/resource evidence differs')
    expected={'teacher_readback.json','bank_manifest.json','bank_index.npz'}
    expected|={'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100','U100_EMBED_W5','original50k')}
    expected|={a+'/'+n for a in ARMS for n in ('selected.pt','training_report.json','validation_predictions.npz','receipt.json')}
    if len(r['outputs'])!=len(expected) or {v['path'] for v in r['outputs']}!=expected:
        raise ValueError('Preflight payload coverage differs')
    for ref in r['outputs']: validate_file_ref(ref,root=root)
    bank=retained_bank(root,spec)
    from hlt_classification.luka_u100_embedding.contracts import validate as old_validate
    proof=load_json(root/'teacher_readback.json')
    old_validate(proof,'TEACHER_READBACK',dict(teacher=spec['teacher']['content_hash']))
    if (proof['rows']!=spec['counts']['validation'] or proof['role']!='validation' or proof['coordinate']!='U100'
            or proof['native_probability_parity'] is not True or proof['normalization_used'] is not False
            or (proof['rtol'],proof['atol'])!=(1e-5,1e-6)
            or not 0<=proof['maximum_absolute_error']<=1.1e-5):
        raise ValueError('Teacher readback evidence differs')
    for arm in ARMS:
        report,_=validate_fit(root/arm,spec,arm,bank['content_hash'],technical=True)
        if (r['projections'][arm]!=projected_minutes(r['setup_seconds'],report['validation_history'])
                or type(r['projections'][arm]) is not int or not 1<=r['projections'][arm]<=spec['train_minutes']):
            raise ValueError('Measured fixed30 projection exceeds the reviewed debug allocation')
    if accounting: submission.successful_preflight(spec,r['job_id'])
    return r


def run(spec,task):
    if task not in ('preflight',*ARMS): raise ValueError('Unknown task')
    started=time.perf_counter(); campaign.check(spec); campaign.storage(spec)
    admitted=None if task=='preflight' else validate_preflight(spec)
    job=execution(spec,admitted)
    if installed_environment()!=spec['anchor']['evidence']['environment']:
        raise ValueError('Original model environment changed')
    if submission.accepted_jobs(spec).get(task)!=job: raise ValueError('Worker is not the accepted exact job')
    root=Path(spec['root'])/task; root.mkdir(exist_ok=False)
    write_immutable_json(root/'claim.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    local=scratch_directory(spec)
    log=PhaseLog(root,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticate_inputs_and_teacher'):
        meta=regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
        train=data(spec,'train','D000',meta); val=data(spec,'validation','D000',meta)
        rich=data(spec,'train','U100',meta); banks.check_pair(rich,train)
        required=int(counts(train).sum())*128*4+512*2**20
        if shutil.disk_usage(local).free<required: raise OSError(f'Need {required/2**30:.2f}GiB job-local scratch')
        teacher=DelphesParticleTransformer().to('cuda').eval()
        ref=spec['teacher']['files']['checkpoint']; validate_file_ref(ref)
        teacher.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
        richval=data(spec,'validation','U100',meta)
        proof=teacher_readback(teacher,richval,spec['teacher'],'cuda')
        write_immutable_json(root/'teacher_readback.json',proof); del richval
        snapshot_references(root/'references',spec,val,meta)
    # Only the unique child created by TemporaryDirectory is retired. Shared
    # Shared scratch base, dataset, source and historical paths are never deleted.
    with TemporaryDirectory(prefix='lukapp-'+job+'-',dir=local) as temporary:
        bank=None
        try:
            target_root=Path(temporary)/'targets'
            with log.phase('export_TRAIN_block8_targets'):
                record=banks.export(target_root,teacher,rich,train,spec['teacher'],device='cuda')
                if task!='preflight' and record!=retained_bank(Path(spec['root'])/'preflight',spec):
                    raise ValueError('Regenerated targets differ from preflight; no student fit allowed')
                write_immutable_json(root/'bank_manifest.json',record)
                atomic_publish_bytes(root/'bank_index.npz',(target_root/'index.npz').read_bytes())
                del teacher,rich; gc.collect(); torch.cuda.empty_cache()
                bank=banks.Bank(target_root,train,spec['teacher'])
            setup=time.perf_counter()-started
            if task=='preflight':
                with log.phase('native_parity_and_maximum_support'):
                    checks=parity_and_stress(spec,train,val,bank,'cuda')
                reports={}
                for arm in ARMS:
                    with log.phase('technical_full_epoch_'+arm):
                        reports[arm],_=fit(root/arm,spec,arm,train,val,bank,device='cuda',technical=True)
                projections={a:projected_minutes(setup,reports[a]['validation_history']) for a in ARMS}
                measurement=artifact('MEASUREMENT',parents=dict(spec=spec['content_hash']),job_id=job,
                    projections=projections,setup_seconds=setup,memory=memory_snapshot(torch),
                    gpu=gpu_identity(),environment=installed_environment(),checks=checks)
                write_immutable_json(root/'measurement.json',measurement)
            else:
                with log.phase('science_fixed30_'+task):
                    fit(root/'fit',spec,task,train,val,bank,device='cuda')
        finally:
            if bank is not None: bank.close()
    print('Owned job-local token bank retired; selected models, predictions and evidence retained.',flush=True)
    if source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']:
        raise ValueError('Source changed during worker')
    campaign.storage(spec)
    common=['teacher_readback.json','bank_manifest.json','bank_index.npz']
    common+=['references/'+n+'.npz' for n in ('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100','U100_EMBED_W5','original50k')]
    if task=='preflight':
        files=common+[a+'/'+n for a in ARMS for n in ('selected.pt','training_report.json','validation_predictions.npz','receipt.json')]
        r=artifact('PREFLIGHT',parents=dict(spec=spec['content_hash']),job_id=job,technical_checks_passed=True,
            projections=projections,setup_seconds=setup,memory=measurement['memory'],gpu=measurement['gpu'],
            environment=measurement['environment'],checks=checks,scratch_payloads_retired=True,
            outputs=[file_ref(root/f,root=root) for f in files])
        # Publish only after numerical/resource validation; failed measurements remain diagnostic.
        if (max(projections.values())>spec['train_minutes'] or measurement['memory']['peak_rss_bytes']>=.85*spec['memory_mb']*1024**2
                or measurement['memory']['gpu_peak_bytes']>=.90*measurement['gpu']['total_memory_bytes']):
            raise ValueError('New token envelope failed; preserve measurement.json, science blocked')
        write_immutable_json(root/'receipt.json',r); validate_preflight(spec,accounting=False)
        return r
    files=common+['fit/'+n for n in ('selected.pt','training_report.json','validation_predictions.npz','receipt.json')]
    receipt=artifact('SCIENCE',parents=dict(spec=spec['content_hash'],preflight=admitted['content_hash']),
        arm=task,job_id=job,scratch_payloads_retired=True,outputs=[file_ref(root/f,root=root) for f in files])
    write_immutable_json(root/'receipt.json',receipt)
    return receipt


def results(spec):
    campaign.validate_spec(spec)
    root=Path(spec['root']); pre=validate_preflight(spec,accounting=False)
    refroot=root/'preflight/references'
    names=('M0HLT','OFFLINE','OFFLINE_EMBED_W5','COARSE_U100','U100_EMBED_W5')
    arrays={n:load_npz_arrays(refroot/(n+'.npz')) for n in names}; pending=[]
    for arm in ARMS:
        path=root/arm/'receipt.json'
        if not path.is_file(): pending.append(arm); continue
        receipt=load_json(path); validate(receipt,'SCIENCE',dict(spec=spec['content_hash'],preflight=pre['content_hash']))
        expected={'teacher_readback.json','bank_manifest.json','bank_index.npz'}
        expected|={'references/'+n+'.npz' for n in (*names,'original50k')}
        expected|={'fit/'+n for n in ('selected.pt','training_report.json','validation_predictions.npz','receipt.json')}
        if (receipt['arm']!=arm or receipt['scratch_payloads_retired'] is not True
                or len(receipt['outputs'])!=len(expected) or {r['path'] for r in receipt['outputs']}!=expected):
            raise ValueError('Science publication coverage differs')
        for ref in receipt['outputs']: validate_file_ref(ref,root=root/arm)
        bank=retained_bank(root/arm,spec)
        if bank!=retained_bank(root/'preflight',spec):
            raise ValueError('Arms do not share exact preflight targets')
        _,arrays['TOKEN_'+arm.upper()]=validate_fit(root/arm/'fit',spec,arm,bank['content_hash'])
    ix=load_npz_arrays(refroot/'original50k.npz')['indices']; baseline=arrays['M0HLT']
    if ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix) or np.any(ix<0) or np.any(ix>=len(baseline['labels'])):
        raise ValueError('Report-only original50k selection differs')
    rows=[]
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if not np.array_equal(a['identities'],baseline['identities']) or not np.array_equal(a['labels'],baseline['labels']):
            raise ValueError('Comparison population/order differs')
        rows.append(dict(model=name,validation=evaluate_probabilities(a['labels'],a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])))
    reference=next(r for r in rows if r['model']=='U100_EMBED_W5')
    for row in rows:
        for role in ('validation','original50k'):
            row[role+'_recovery']=recovery(row[role],rows[0][role],rows[1][role])
            row[role+'_delta_U100_w5']=row[role]['macro_ovr_auc']-reference[role]['macro_ovr_auc']
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending)


def print_results(value):
    print('U100 block8 particle prediction: CE+5*auxiliary; NO KD. No independent test; original50k report-only.')
    for role in ('validation','original50k'):
        print('\n'+role); print(f'{"model":24} {"accuracy":>10} {"AUC":>10} {"recovery%":>10} {"delta U100-w5":>14}')
        for row in value['rows']:
            rec=row[role+'_recovery']['macro_ovr_auc']; formatted='n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:24} {row[role]["accuracy"]:10.6f} {row[role]["macro_ovr_auc"]:10.6f} {formatted:>10} {row[role+"_delta_U100_w5"]:+14.8f}')
    print('Pending:',', '.join(value['pending']) or 'none')
