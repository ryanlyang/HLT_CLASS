"""Balanced draw-clock training, independent EMA selection and compact resumes."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_sixclass_balanced_oscar.reporting import evaluate_probabilities,summary
from hlt_classification.luka_sixclass_balanced_oscar.sampling import BalancedSampler
from hlt_classification.luka_fullstat_readout.training import cpu_state
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from . import contracts as c,kernel,model,readouts,artifacts,banks


def predict(net,cache,ctx):
    net.eval()
    if hasattr(net,'capture'):net.capture=False;net.clear()
    ix_all=[];probs=[]
    with torch.inference_mode():
        for start in range(0,len(cache),c.BATCH):
            ix=np.arange(start,min(start+c.BATCH,len(cache)),dtype=np.int64)
            local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            raw=cache.batch(ix);p=kernel.native(net,{k:v[local] for k,v in raw.items()},ctx.device).softmax(1)
            ix_all.append(ix[local]);probs.append(p.cpu().numpy())
    local=(np.concatenate(ix_all) if ix_all else np.empty(0,np.int64),np.concatenate(probs) if probs else np.empty((0,6),np.float32))
    pairs=ctx.all(local);indices=np.concatenate([i for i,_ in pairs]);p=np.concatenate([v for _,v in pairs])
    if p.shape!=(len(cache),6) or not np.array_equal(np.sort(indices),np.arange(len(cache))):raise ValueError('Prediction row coverage differs')
    return p[np.argsort(indices)]


def evaluate(net,cache,ctx):
    p=predict(net,cache,ctx)
    return ctx.main(lambda:evaluate_probabilities(cache.labels,p)),p


def fit_parents(spec,task,bank,normalization):
    starts=load_json(Path(spec['root'])/'starts/manifest.json')
    result=dict(spec=spec['content_hash'],population=spec['population']['content_hash'],starts=starts['content_hash'])
    if bank is not None:result.update(teacher=bank.teacher_receipt,bank=bank.manifest['content_hash'])
    if normalization is not None:result['reconstruction_normalization']=normalization['content_hash']
    return result


def check_inputs(spec,task,train,val,pt,bank,targets):
    x=c.cell(task)
    if (train.role!='train' or val.role!='validation' or train.coordinate_name!=x['coordinate']
            or val.coordinate_name!=x['coordinate'] or train.foundation_sha256!=spec['population']['content_hash']
            or val.foundation_sha256!=train.foundation_sha256 or len(train)!=c.COUNTS['train'] or len(val)!=c.COUNTS['validation']):
        raise PermissionError('Wrong training population/view/roles')
    if bool(x['teacher'])!=(bank is not None) or x['reconstruction']!=(targets is not None):raise PermissionError('Wrong target capabilities')
    if bank is not None and (not np.array_equal(bank.identities,train.identities) or not np.array_equal(bank.labels,train.labels)):
        raise ValueError('Teacher/student identity join differs')
    if not np.isfinite(pt).all() or len(pt)!=len(train):raise ValueError('Invalid TRAIN pT')
    a=set(map(bytes,train.identities));b=set(map(bytes,val.identities))
    if len(a)!=len(train) or len(b)!=len(val) or a&b:raise ValueError('Repeated/overlapping split identities')


def fit(spec,task,caches,targets,bank,pt,ctx,folder,job,*,validation_pt):
    tr,val=caches['train'],caches['validation'];check_inputs(spec,task,tr,val,pt,bank,targets)
    if ctx.world!=c.RESOURCES['gpus']:raise PermissionError('Science requires the registered GPU rank count')
    parents=fit_parents(spec,task,bank,None if targets is None else targets.normalization)
    folder=Path(folder);ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
    state,expected=artifacts.start(spec,task)
    net,opt=kernel.build(task,spec['maximum'],ctx,state)
    actual=artifacts.components(net.state_dict());parts=('native','predictor','decoder') if c.cell(task)['reconstruction'] else ('native','predictor')
    if any(actual[k]!=expected[k] for k in parts):raise ValueError('Canonical starting tensors differ')
    initialization=c.artifact('INITIALIZATION',parents=parents,task=task,components=actual,seeds=c.seeds(task),world=ctx.world)
    ctx.main(lambda:write_immutable_json(folder/'initialization.json',initialization))
    sampler=BalancedSampler(tr.labels,pt,seed=c.seeds(task)['sampler'])
    membership=readouts.membership(tr,pt,spec['population']['content_hash'])
    averages=readouts.Averages();history=[];selected={};best={}
    resume=ctx.main(lambda:artifacts.restore(spec,task,parents))
    if resume is not None:
        if len(resume['rng'])!=ctx.world or resume['initial_components']!=expected:raise ValueError('Resume rank/start differs')
        net.load_state_dict(resume['model'],strict=True);opt.load_state_dict(resume['optimizer'])
        averages.restore(resume['averages'],ctx.device)
        history=resume['history'];selected=resume['selected'];best=resume['best']
        torch.set_rng_state(resume['rng'][ctx.rank]['cpu'])
        if ctx.device.type=='cuda':torch.cuda.set_rng_state(resume['rng'][ctx.rank]['cuda'],ctx.device)
    del state,resume
    for epoch in range(len(history)+1,c.EPOCHS+1):
        from . import storage
        ctx.main(lambda:storage.check(spec))
        order=sampler.draw(epoch,draws=c.COUNTS['train']);sums=None;ctx.barrier();ctx.sync();begun=time.monotonic()
        for start in range(0,len(order),c.BATCH):
            ix=order[start:start+c.BATCH];position=epoch-1+(start+len(ix))/len(order)
            target=targets.batch(ix) if targets is not None else None
            emb=np.asarray(bank.targets[ix]) if bank is not None and c.cell(task)['embedding_weight'] else None
            q=np.asarray(bank.probabilities[ix]) if bank is not None and c.cell(task)['main_kd_weight'] else None
            losses=kernel.update(net,opt,tr.batch(ix),emb,q,target,task,ctx,position)
            averages.update(net,position,len(ix))
            if sums is None:sums=dict.fromkeys(losses,0.)
            for name,value in losses.items():sums[name]+=value*len(ix)
            if ctx.rank==0 and ((start//c.BATCH)%128==0 or start+len(ix)==len(order)):
                elapsed=time.monotonic()-begun
                print(f'BALANCED_RECO {task} epoch={epoch}/{c.EPOCHS} batch={start//c.BATCH+1}/{math.ceil(len(order)/c.BATCH)} elapsed={elapsed:.1f}',flush=True)
        ctx.sync();train_seconds=max(ctx.all(time.monotonic()-begun));begun=time.monotonic()
        measured=readouts.evaluate_all(net,averages,tr,val,membership,ctx,evaluate,kernel.native,model.NativeParticleTransformer)
        metrics={name:value[0] for name,value in measured.items()}
        row=dict(epoch=epoch,update=epoch*math.ceil(c.COUNTS['train']/c.BATCH),position=float(epoch),
            learning_rate=learning_rate(epoch),sampling=sampler.evidence(order),readouts=metrics,
            losses={k:v/len(order) for k,v in sums.items()},train_seconds=train_seconds,
            validation_seconds=max(ctx.all(time.monotonic()-begun)),bn_membership=membership['content_hash'])
        for name,(metric,prob,state) in measured.items():
            if name not in selected or selection_key(metric,row['update'])>selection_key(selected[name]['validation'],selected[name]['update']):
                selected[name]=dict(epoch=epoch,update=row['update'],validation=metric);best[name]=state
        del measured
        row=ctx.main(lambda:row);history.append(row)
        rng=ctx.all(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else torch.empty(0,dtype=torch.uint8)))
        def checkpoint():
            saved=cpu_state(dict(model=net.state_dict(),optimizer=opt.state_dict(),averages=averages.values,
                epoch=epoch,history=history,selected=selected,best=best,rng=rng,initial_components=expected))
            artifacts.commit_epoch(spec,task,folder,row,saved,parents,job)
        ctx.main(checkpoint)
        if ctx.rank==0:
            for name,value in metrics.items():print(f'BALANCED_RECO {task} {name} epoch={epoch} AUC={value["macro_ovr_auc"]:.8f}',flush=True)
    if set(selected)!=set(c.READOUTS):raise ValueError('Incomplete selected readouts')
    files={};metrics={}
    for name in c.READOUTS:
        native=readouts_native(best[name],ctx);p=predict(native,val,ctx)
        measured=evaluate_probabilities(val.labels,p)
        if measured!=selected[name]['validation']:raise ValueError('Selected native reload differs')
        def publish_readout():
            out=folder/name;out.mkdir(exist_ok=False)
            artifacts.save_state(out/'selected.pt',best[name])
            atomic_publish_bytes(out/'predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=p)))
            if np.asarray(validation_pt).shape!=(len(val),) or not np.all((validation_pt>=600)&(validation_pt<2600)):
                raise ValueError('Validation pT differs')
            per_pt={}
            for low in range(600,2600,100):
                mask=(validation_pt>=low)&(validation_pt<low+100)
                per_pt[f'{low}_{low+100}']=summary(val.labels[mask],p[mask])
            record=c.artifact('METRICS',parents=parents,task=task,readout=name,selected=selected[name],
                measurements=summary(val.labels,p),per_pt=per_pt)
            write_immutable_json(out/'metrics.json',record)
            return {f'{name}_{k}':file_ref(out/f) for k,f in [('checkpoint','selected.pt'),('predictions','predictions.npz'),('metrics','metrics.json')]}
        files.update(ctx.main(publish_readout));del native
    report=c.artifact('TRAINING_REPORT',parents=parents,task=task,cell=c.cell(task),recipe=c.recipe(),
        counts=c.COUNTS,epochs=c.EPOCHS,history=history,selected=selected,primary='EMA05',
        initialization=initialization['content_hash'],world=ctx.world,precision='FP32_TF32_off',native_reload=True)
    ctx.main(lambda:write_immutable_json(folder/'training_report.json',report))
    files.update(report=file_ref(folder/'training_report.json'),initialization=file_ref(folder/'initialization.json'))
    # Export from selected files before the RESULT receipt; never accept half an outgoing teacher.
    if c.cell(task)['arm'] in ('OFFLINE','U100'):
        banks.export(spec,task,tr,ctx,folder=folder)
        files['bank']=file_ref(folder/'bank/manifest.json')
    def publish_result():
        from . import campaign
        campaign.check(spec)
        receipt=c.artifact('RESULT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
            fit_parents=parents,task=task,job_id=job,files=files)
        write_immutable_json(artifacts.fit_root(spec,task)/'result.json',receipt)
        artifacts.completed(spec,task)
        artifacts.compact(spec,task,parents,published=True)
        return receipt
    receipt=ctx.main(publish_result);del net,opt,averages,best;d.release_memory(ctx);return receipt


def readouts_native(state,ctx):
    from hlt_classification.luka_reconstruction_stability_hydra.readouts import native_copy
    return native_copy(state,ctx,model.NativeParticleTransformer)


def check_row(row,task,epoch,spec):
    expected=set(c.READOUTS) if epoch>=3 else {'RAW','RAW_BN'}
    if (row['epoch']!=epoch or row['update']!=epoch*math.ceil(c.COUNTS['train']/c.BATCH)
            or row['position']!=float(epoch) or row['learning_rate']!=learning_rate(epoch)
            or set(row['readouts'])!=expected or any(v['rows']!=c.COUNTS['validation'] for v in row['readouts'].values())
            or row['sampling']['draws']!=c.COUNTS['train'] or row['sampling']['unique_jets']>c.COUNTS['train']):
        raise ValueError('Epoch objective/population/schedule differs')
    exposure=row['sampling'];counts=np.asarray(exposure['class_bin_counts'])
    if (counts.shape!=(6,20) or counts.dtype.kind not in 'iu' or np.any(counts<0) or counts.sum()!=c.COUNTS['train']
            or not 0<exposure['unique_jets']<=c.COUNTS['train'] or exposure['repeated_draws']!=c.COUNTS['train']-exposure['unique_jets']):
        raise ValueError('Invalid balanced-draw exposure')
    losses=row['losses'];x=c.cell(task)
    if set(losses)!={'ce','main_kd','embedding','reconstruction','total','encoder_reconstruction','decoder_reconstruction'}:
        raise ValueError('Reported loss inventory differs')
    if (not np.isfinite(list(losses.values())).all() or min(losses.values())<-1e-6
            or not np.isclose(losses['total'],x['ce_weight']*losses['ce']+.75*losses['main_kd']+5*losses['embedding']+losses['encoder_reconstruction'],rtol=2e-6,atol=1e-7)
            or (not x['embedding_weight'] and losses['embedding']!=0) or (not x['main_kd_weight'] and losses['main_kd']!=0)
            or (not x['reconstruction'] and losses['reconstruction']!=0)):
        raise ValueError('Invalid reported loss composition')


def validate_report(report,spec,task,parents):
    c.validate(report,'TRAINING_REPORT',parents)
    if (report['task']!=task or report['cell']!=c.cell(task) or report['recipe']!=c.recipe()
            or report['counts']!=c.COUNTS or report['epochs']!=c.EPOCHS or len(report['history'])!=c.EPOCHS
            or report['primary']!='EMA05' or set(report['selected'])!=set(c.READOUTS) or report['native_reload'] is not True):
        raise ValueError('Fit semantics differ')
    if report['world']!=c.RESOURCES['gpus'] or report['precision']!='FP32_TF32_off':raise ValueError('Execution/precision differs')
    for epoch,row in enumerate(report['history'],1):check_row(row,task,epoch,spec)
    for name in c.READOUTS:
        best=max((r for r in report['history'] if name in r['readouts']),key=lambda r:selection_key(r['readouts'][name],r['update']))
        if report['selected'][name]!=dict(epoch=best['epoch'],update=best['update'],validation=best['readouts'][name]):
            raise ValueError('Readout checkpoint selection differs')
