"""Production training, complete validation and selected native-only export."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_next_supervision_hydra.contracts import seed
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact, validate, cell, weights, scientific_cfg, seeds
from . import kernel as k

LOSS_KEYS = ('ce','embedding','reconstruction','weighted_embedding','encoder_weighted_reconstruction','decoder_weighted_reconstruction')
OUTPUTS = ('selected.pt','selected_full.pt','validation_predictions.npz','training_report.json','deployment.json','initialization.json')


def parents(spec):
    return dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],original=spec['parents']['original'])


def predict(model,cache,ctx):
    model.eval()
    if hasattr(model,'capture'):model.capture=False;k.clear(model)
    indices=[];predictions=[]
    with torch.no_grad():
        for start in range(0,len(cache),128):
            ix=np.arange(start,min(start+128,len(cache)),dtype=np.int64)
            local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            if not len(local):continue
            raw=cache.batch(ix)
            logits=k.native(model,{key:value[local] for key,value in raw.items()},ctx.device)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite native validation')
            indices.append(ix[local]);predictions.append(logits.softmax(1).cpu().numpy())
    pair=(np.concatenate(indices) if indices else np.empty(0,np.int64),
          np.concatenate(predictions) if predictions else np.empty((0,11),np.float32))
    return d.merge_predictions(ctx.all(pair),len(cache))


def evaluate(model,val,ctx):
    probabilities=predict(model,val,ctx)
    metrics=ctx.main(lambda:evaluate_probabilities(val.labels,probabilities))
    return metrics,probabilities


def check_inputs(spec,train,val,bank,technical):
    if (train.role!='train' or val.role!='validation' or train.coordinate_name!='D000' or val.coordinate_name!='D000'
            or train.foundation_sha256!=val.foundation_sha256 or train.foundation_sha256!=spec['foundation_hash']
            or bank.manifest['content_hash']!=spec['embedding_hash']
            or bank.targets.shape!=(len(train),128) or not np.isfinite(bank.targets).all()):
        raise PermissionError('Wrong authenticated TRAIN/validation roles or embedding targets')
    if not technical and dict(train=len(train),validation=len(val))!=spec['counts']:raise ValueError('Wrong scientific population')
    a,b=set(map(bytes,train.identities)),set(map(bytes,val.identities))
    if len(a)!=len(train) or len(b)!=len(val) or a&b:raise ValueError('Duplicate/overlapping jet identities')


def batch(train,bank,targets,ix,epoch,task):
    target=None if cell(task)['control'] else targets.batch(ix,epoch,scientific_cfg(task))[0]
    return train.batch(ix),bank.targets[ix],target


def fit(folder,spec,task,train,val,bank,targets,ctx,*,technical=False):
    check_inputs(spec,train,val,bank,technical)
    if not technical and ctx.world!=cell(task)['world_size']:raise PermissionError('Wrong registered rank count')
    if cell(task)['worker']!='curriculum':raise PermissionError('Original arm must use the original worker')
    folder=Path(folder);ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
    model,opt=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
    from .identity import initial_record
    initial=ctx.main(lambda:initial_record(spec,task,model,ctx.world))
    ctx.main(lambda:write_immutable_json(folder/'initialization.json',initial))
    epochs=2 if technical else 30;steps=math.ceil(len(train)/128);history=[];selected=None;best_state=None
    sampler=seeds(spec,task)['sampler']
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(np.random.SeedSequence([sampler,epoch])).permutation(len(train))
        sums=dict.fromkeys(LOSS_KEYS,0.);diagnostic=None;ctx.barrier();ctx.sync();started=time.perf_counter()
        for start in range(0,len(train),128):
            ix=order[start:start+128];update=(epoch-1)*steps+start//128+1
            position=(7.,20.)[epoch-1] if technical else update/steps
            losses,details=k.update(model,opt,batch(train,bank,targets,ix,epoch,task),task,
                ctx.rank,ctx.world,ctx.device,position,probe=start==0)
            if start==0:diagnostic=ctx.all(details['diagnostics'])
            for key,value in losses.items():sums[key]+=value*len(ix)
            if ctx.rank==0 and (start//128%100==0 or start+128>=len(train)):
                print(f'R24_DIAGNOSTIC {task} epoch={epoch}/{epochs} batch={start//128+1}/{steps}',flush=True)
        ctx.sync();seconds=max(ctx.all(time.perf_counter()-started))
        val_seconds,(metrics,_)=d.timed(lambda:evaluate(model,val,ctx),ctx)
        row=artifact('EPOCH',parents=parents(spec),task=task,epoch=epoch,update=update,position=position,
            technical=technical,validation=metrics,losses={key:value/len(train) for key,value in sums.items()},
            encoder_weight=weights(task,position)[0],decoder_weight=weights(task,position)[1],
            learning_rate=learning_rate(position),gradient_diagnostics=diagnostic,
            train_seconds=seconds,validation_seconds=val_seconds)
        # Timing values were globally reduced; publish rank0's exact record.
        row=ctx.main(lambda:row);history.append(row)
        if selected is None or selection_key(metrics,update)>selection_key(selected['validation'],selected['update']):
            selected=row;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        ctx.main(lambda:write_immutable_json(folder/f'epoch-{epoch:02}.json',row))
        if ctx.rank==0:print(f'R24_DIAGNOSTIC {task} epoch={epoch}/{epochs} AUC={metrics["macro_ovr_auc"]:.8f} '
            f'recovery={recovery_percent(metrics["macro_ovr_auc"],spec["recovery_anchors"]):.1f}% best_epoch={selected["epoch"]}',flush=True)
    started=time.perf_counter();model.load_state_dict(best_state,strict=True);model.eval();model.capture=False
    ctx.main(lambda:save_state(folder/'selected_full.pt',best_state))
    ctx.main(lambda:save_state(folder/'selected.pt',model.deployment_state()))
    ctx.barrier()
    with torch.random.fork_rng(devices=[]):native=DelphesParticleTransformer().to(ctx.device).eval()
    native._diagnostic_precision=cell(task)['precision']
    native.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True),strict=True)
    actual=predict(native,val,ctx);expected=predict(model,val,ctx)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    def verify():
        if evaluate_probabilities(val.labels,actual)!=selected['validation']:raise ValueError('Selected native replay differs')
        atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=actual)))
    ctx.main(verify);memory=d.memory(ctx);ctx.sync();export_seconds=max(ctx.all(time.perf_counter()-started))
    report=artifact('TRAINING_REPORT',parents=parents(spec),task=task,cell=cell(task),technical=technical,
        counts=dict(train=len(train),validation=len(val)),epochs=epochs,selected_epoch=selected['epoch'],
        validation=selected['validation'],history=history,native_reload=True,oracle=False,
        fit_memory=memory,precision=cell(task)['precision'],world_size=ctx.world,global_batch=128,export_seconds=export_seconds,
        initialization=initial['content_hash'],
        target_hash=bank.manifest['content_hash'])
    report=ctx.main(lambda:report)
    def publish():
        write_immutable_json(folder/'deployment.json',artifact('DEPLOYMENT',parents=parents(spec),task=task,
            checkpoint=file_ref(folder/'selected.pt',root=folder),native_only=True,additional_parameters=0,
            parameters=sum(p.numel() for p in native.parameters()),state_keys=list(native.state_dict())))
        write_immutable_json(folder/'training_report.json',report)
        write_immutable_json(folder/'receipt.json',artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
            task=task,outputs=[file_ref(folder/n,root=folder) for n in OUTPUTS]))
        validate_fit(folder,spec,task,technical=technical)
    ctx.main(publish)
    del native,model,opt,best_state;d.release_memory(ctx);return report


def validate_epoch(spec,task,row,epoch,counts,world,technical):
    validate(row,'EPOCH',parents(spec));steps=math.ceil(counts['train']/128)
    position=(7.,20.)[epoch-1] if technical else float(epoch)
    if (row['task']!=task or row['epoch']!=epoch or row['technical']!=technical or row['update']!=epoch*steps
            or row['position']!=position or row['learning_rate']!=learning_rate(position)
            or (row['encoder_weight'],row['decoder_weight'])!=weights(task,position)
            or row['validation']['rows']!=counts['validation'] or set(row['losses'])!=set(LOSS_KEYS)
            or len(row['gradient_diagnostics'])!=world):raise ValueError('Epoch scope/schedule differs')
    losses=row['losses']
    if (not np.isfinite(list(losses.values())).all() or min(losses.values())<0
            or not np.isclose(losses['weighted_embedding'],5*losses['embedding'],rtol=1e-6,atol=1e-8)
            or not all(np.isfinite(row[k]) and row[k]>0 for k in ('train_seconds','validation_seconds'))):
        raise ValueError('Invalid epoch losses/timing')
    p=(7.,20.)[epoch-1] if technical else epoch-1+1/steps
    for diag in row['gradient_diagnostics']:
        if (diag['encoder_weight'],diag['decoder_weight'])!=weights(task,p):raise ValueError('Probe schedule differs')
        for key,value in diag.items():
            if isinstance(value,(int,float)) and not np.isfinite(value):raise ValueError('Nonfinite gradient diagnostic')
        if weights(task,p)[0]==0 and not cell(task)['control'] and (diag['weighted_auxiliary_jet_gradient_norm']!=0 or diag['weighted_auxiliary_token_gradient_norm']!=0):
            raise ValueError('Reconstruction leaked into encoder during delay')


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder);report=load_json(folder/'training_report.json');receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT',parents(spec));validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    if receipt['task']!=task or [r['path'] for r in receipt['outputs']]!=list(OUTPUTS):raise ValueError('Fit output inventory differs')
    for ref in receipt['outputs']:validate_file_ref(ref,root=folder)
    epochs=2 if technical else 30
    if (report['task']!=task or report['cell']!=cell(task) or report['technical']!=technical or report['epochs']!=epochs
            or len(report['history'])!=epochs or not report['native_reload'] or report['oracle']
            or report['precision']!=cell(task)['precision'] or report['global_batch']!=128 or report['world_size'] not in (1,4)
            or report['target_hash']!=spec['embedding_hash']
            or (not technical and (report['world_size']!=cell(task)['world_size'] or report['counts']!=spec['counts']))):
        raise ValueError('Fit meaning differs')
    from .identity import validate_initial
    initial=load_json(folder/'initialization.json');validate_initial(spec,task,initial,report['world_size'])
    if report['initialization']!=initial['content_hash']:raise ValueError('Initialization parent differs')
    for e,row in enumerate(report['history'],1):
        if row!=load_json(folder/f'epoch-{e:02}.json'):raise ValueError('Epoch file differs')
        validate_epoch(spec,task,row,e,report['counts'],report['world_size'],technical)
    selected=max(report['history'],key=lambda row:selection_key(row['validation'],row['update']))
    if report['selected_epoch']!=selected['epoch'] or report['validation']!=selected['validation']:raise ValueError('Selection differs')
    deployment=load_json(folder/'deployment.json');validate(deployment,'DEPLOYMENT',parents(spec))
    if (deployment['task']!=task or deployment['native_only'] is not True or deployment['additional_parameters']!=0
            or deployment['checkpoint']!=file_ref(folder/'selected.pt',root=folder)
            or not deployment['state_keys'] or any(not key.startswith('mod.') for key in deployment['state_keys'])):
        raise ValueError('Deployment includes a training head')
    arrays=load_npz_arrays(folder/'validation_predictions.npz');validate_predictions(arrays,report['counts']['validation'])
    if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:raise ValueError('Predictions differ')
    return report,arrays
