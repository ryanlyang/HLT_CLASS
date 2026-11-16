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
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact, validate, cell, weights, scientific_cfg, seeds
from . import kernel as k, readouts
from .contracts import READOUTS

LOSS_KEYS = ('ce','embedding','reconstruction','weighted_embedding','encoder_weighted_reconstruction','decoder_weighted_reconstruction')
OUTPUTS = ('training_report.json','initialization.json','final_training_state.pt') + tuple(
    f'{name}/{file}' for name in READOUTS for file in ('selected.pt','validation_predictions.npz','deployment.json'))


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


def check_inputs(spec,train,val,bank,technical,task):
    if (train.role!='train' or val.role!='validation' or train.coordinate_name!='D000' or val.coordinate_name!='D000'
            or train.foundation_sha256!=val.foundation_sha256 or train.foundation_sha256!=spec['foundation_hash']):
        raise PermissionError('Wrong authenticated TRAIN/validation roles or embedding targets')
    if cell(task)['embedding_weight']:
        if (bank is None or bank.manifest['content_hash']!=spec['embedding_hash'] or
            bank.targets.shape!=(len(train),128) or not np.isfinite(bank.targets).all()):
            raise PermissionError('Wrong U100 TRAIN targets')
    elif bank is not None:
        raise PermissionError('CE-only training must not load U100 targets')
    if not technical and dict(train=len(train),validation=len(val))!=spec['counts']:raise ValueError('Wrong scientific population')
    a,b=set(map(bytes,train.identities)),set(map(bytes,val.identities))
    if len(a)!=len(train) or len(b)!=len(val) or a&b:raise ValueError('Duplicate/overlapping jet identities')


def batch(train,bank,targets,ix,epoch,task):
    target=None if cell(task)['control'] else targets.batch(ix,epoch,scientific_cfg(task))[0]
    return train.batch(ix),None if cell(task)['embedding_weight']==0 else bank.targets[ix],target


def fit(folder,spec,task,train,val,bank,targets,ctx,*,technical=False,calibration_train=None):
    check_inputs(spec,train,val,bank,technical,task)
    if not technical and ctx.world!=cell(task)['world_size']:raise PermissionError('Wrong registered rank count')
    folder=Path(folder);ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
    calibration_train=train if calibration_train is None else calibration_train
    membership=readouts.load_membership(spec,calibration_train,technical=technical)
    model,optimizer=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
    from .identity import initial_record,starts
    initial=ctx.main(lambda:initial_record(spec,task,model,ctx.world))
    ctx.main(lambda:write_immutable_json(folder/'initialization.json',initial))
    from .identity import check_start
    check_start(spec,task,initial)
    averages=readouts.Averages();epochs=2 if technical else 30
    effective=128*cell(task)['accumulation'];steps=math.ceil(len(train)/effective)
    history=[];selected={};selected_states={};selected_probabilities={};sampler=seeds(spec,task)['sampler']
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(np.random.SeedSequence([sampler,epoch])).permutation(len(train))
        sums=dict.fromkeys(LOSS_KEYS,0.);diag_sums={};diag_max={};diag_examples={}
        counters=dict(optimizer_updates=0,decoder_extra_updates=0,capped_updates=0,projected_updates=0)
        probe=None;ctx.barrier();ctx.sync();began=time.perf_counter()
        for start in range(0,len(train),effective):
            ix=order[start:start+effective];update=(epoch-1)*steps+start//effective+1
            position=(4.5,20.)[epoch-1] if technical else epoch-1+(start+len(ix))/len(train)
            batches=[batch(train,bank,targets,ix[i:i+128],epoch,task) for i in range(0,len(ix),128)]
            losses,details=k.update_group(model,optimizer,batches,task,ctx.rank,ctx.world,ctx.device,position,probe=start==0)
            averages.update(model,position,len(ix))
            if start==0:probe=ctx.all(details['diagnostics'])
            counters['optimizer_updates']+=1
            counters['decoder_extra_updates']+=details['diagnostics']['extra_decoder_steps']
            counters['capped_updates']+=int(details['diagnostics'].get('capped',False))
            counters['projected_updates']+=int(details['diagnostics'].get('projected',False))
            for key,value in losses.items():sums[key]+=value*len(ix)
            for key,value in {**details['diagnostics'],**details['reconstruction_losses']}.items():
                if value is None or not isinstance(value,(int,float,bool)):continue
                if not np.isfinite(value):raise ValueError('Nonfinite training diagnostic: '+key)
                diag_sums[key]=diag_sums.get(key,0.)+float(value)*len(ix)
                diag_examples[key]=diag_examples.get(key,0)+len(ix)
                diag_max[key]=max(diag_max.get(key,-math.inf),float(value))
            if ctx.rank==0 and (start//effective%100==0 or start+effective>=len(train)):
                print(f'EMA_CONTROL {task} epoch={epoch}/{epochs} update={start//effective+1}/{steps}',flush=True)
        ctx.sync();seconds=max(ctx.all(time.perf_counter()-began))
        val_seconds,measured=d.timed(lambda:readouts.evaluate_all(model,averages,calibration_train,val,membership,ctx,evaluate,k.native,DelphesParticleTransformer),ctx)
        metrics={name:value[0] for name,value in measured.items()}
        row=artifact('EPOCH',parents=parents(spec),task=task,epoch=epoch,update=update,position=position,
            technical=technical,validation=metrics['RAW'],readouts=metrics,
            losses={key:value/len(train) for key,value in sums.items()},
            encoder_weight=weights(task,position)[0],decoder_weight=weights(task,position)[1],
            learning_rate=learning_rate(position),gradient_diagnostics=probe,
            diagnostic_means=ctx.all({key:value/diag_examples[key] for key,value in diag_sums.items()}),
            diagnostic_maxima=ctx.all(diag_max),diagnostic_examples=ctx.all(diag_examples),update_counters=counters,
            loss_semantics='ce_plus_registered_u100_plus_scheduled_hard_reconstruction',bn_membership=membership['content_hash'],
            optimizer_updates_this_epoch=steps,effective_batch=effective,
            train_seconds=seconds,validation_seconds=val_seconds)
        row=ctx.main(lambda:row);history.append(row)
        for name,(metric,probabilities,state) in measured.items():
            if name not in selected or selection_key(metric,update)>selection_key(selected[name]['validation'],selected[name]['update']):
                selected[name]=dict(epoch=epoch,update=update,validation=metric)
                selected_states[name]=state;selected_probabilities[name]=probabilities.copy()
        del measured
        ctx.main(lambda:write_immutable_json(folder/f'epoch-{epoch:02}.json',row))
        if ctx.rank==0:
            for name,metric in metrics.items():print(f'EMA_CONTROL {task} {name} epoch={epoch} AUC={metric["macro_ovr_auc"]:.8f} recovery={recovery_percent(metric["macro_ovr_auc"],spec["recovery_anchors"]):.1f}%',flush=True)
    began=time.perf_counter()
    if set(selected)!=set(READOUTS):raise ValueError('Incomplete readout coverage')
    ctx.main(lambda:save_state(folder/'final_training_state.pt',dict(
        **k.qualification_state(model,optimizer),averages={n:{key:v.cpu() for key,v in state.items()} for n,state in averages.values.items()},
        epochs=epochs,optimizer_updates=epochs*steps)))
    for name in READOUTS:
        destination=folder/name;ctx.main(lambda:destination.mkdir())
        state=selected_states[name]
        ctx.main(lambda:save_state(destination/'selected.pt',state))
        saved_state=torch.load(destination/'selected.pt',map_location='cpu',weights_only=True)
        native=readouts.native_copy(saved_state,ctx,DelphesParticleTransformer)
        actual=predict(native,val,ctx);expected=selected_probabilities[name]
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        def verify():
            if evaluate_probabilities(val.labels,actual)!=selected[name]['validation']:raise ValueError('Selected native readout replay differs')
            atomic_publish_bytes(destination/'validation_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
            write_immutable_json(destination/'deployment.json',artifact('DEPLOYMENT',parents=parents(spec),task=task,readout=name,
                checkpoint=file_ref(destination/'selected.pt',root=folder),native_only=True,additional_parameters=0,
                parameters=sum(p.numel() for p in native.parameters()),state_keys=list(native.state_dict())))
        ctx.main(verify);del native
    memory=d.memory(ctx);ctx.sync();export_seconds=max(ctx.all(time.perf_counter()-began))
    report=artifact('TRAINING_REPORT',parents=parents(spec),task=task,cell=cell(task),technical=technical,
        counts=dict(train=len(train),validation=len(val)),epochs=epochs,selected_epoch=selected['EMA05']['epoch'],
        validation=selected['EMA05']['validation'],selected_readouts=selected,primary_readout='EMA05',history=history,native_reload=True,oracle=False,
        fit_memory=memory,precision='fp32',world_size=ctx.world,global_batch=128,effective_batch=effective,export_seconds=export_seconds,
        initialization=initial['content_hash'],initial_states=starts(spec)['content_hash'],
        target_hash=None if bank is None else bank.manifest['content_hash'],bn_membership=membership['content_hash'])
    report=ctx.main(lambda:report)
    def publish():
        write_immutable_json(folder/'training_report.json',report)
        write_immutable_json(folder/'receipt.json',artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,
            outputs=[file_ref(folder/n,root=folder) for n in OUTPUTS]))
        validate_fit(folder,spec,task,technical=technical)
    ctx.main(publish)
    del model,optimizer,averages,selected_states,selected_probabilities;d.release_memory(ctx);return report


def validate_epoch(spec,task,row,epoch,counts,world,technical):
    validate(row,'EPOCH',parents(spec));effective=128*cell(task)['accumulation'];steps=math.ceil(counts['train']/effective)
    position=(4.5,20.)[epoch-1] if technical else float(epoch)
    available=set(READOUTS) if technical or epoch>=3 else {'RAW','RAW_BN'}
    counters=row['update_counters'];expected_extra=cell(task)['decoder_extra_steps']*steps if 3.<position<=9. else 0
    if (row['task']!=task or row['epoch']!=epoch or row['technical']!=technical or row['update']!=epoch*steps
        or row['position']!=position or row['learning_rate']!=learning_rate(position)
        or (row['encoder_weight'],row['decoder_weight'])!=weights(task,position)
        or set(row['readouts'])!=available or row['readouts']['RAW']!=row['validation']
        or any(v['rows']!=counts['validation'] or not np.isfinite(v['macro_ovr_auc']) for v in row['readouts'].values())
        or row['effective_batch']!=effective or row['optimizer_updates_this_epoch']!=steps
        or set(counters)!={'optimizer_updates','decoder_extra_updates','capped_updates','projected_updates'}
        or counters['optimizer_updates']!=steps or counters['decoder_extra_updates']!=expected_extra
        or any(type(v)!=int or v<0 for v in counters.values())
        or any(counters[n]>steps for n in ('capped_updates','projected_updates'))
        or (cell(task)['gradient_rule']!='G30' and counters['capped_updates']!=0)
        or (cell(task)['gradient_rule']!='PROJ' and counters['projected_updates']!=0)
        or row['loss_semantics']!='ce_plus_registered_u100_plus_scheduled_hard_reconstruction'
        or set(row['losses'])!=set(LOSS_KEYS) or len(row['gradient_diagnostics'])!=world
        or len(row['diagnostic_means'])!=world or len(row['diagnostic_maxima'])!=world
        or len(row['diagnostic_examples'])!=world):raise ValueError('Epoch meaning differs')
    losses=row['losses']
    if cell(task)['recipe']=='CE' and any(losses[n]!=0 for n in LOSS_KEYS if n!='ce'):
        raise ValueError('CE-only report includes a privileged loss')
    if (not np.isfinite(list(losses.values())).all() or min(losses.values())<0
        or not np.isclose(losses['weighted_embedding'],cell(task)['embedding_weight']*losses['embedding'],rtol=1e-6,atol=1e-8)
        or not all(np.isfinite(row[key]) and row[key]>0 for key in ('train_seconds','validation_seconds'))):raise ValueError('Invalid epoch values')
    for diag in (*row['gradient_diagnostics'],*row['diagnostic_means'],*row['diagnostic_maxima']):
        if any(isinstance(v,(int,float)) and not np.isfinite(v) for v in diag.values()):raise ValueError('Nonfinite diagnostic')


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder);report=load_json(folder/'training_report.json');receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT',parents(spec));validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    if receipt['task']!=task or [r['path'] for r in receipt['outputs']]!=list(OUTPUTS):raise ValueError('Fit output inventory differs')
    for ref in receipt['outputs']:validate_file_ref(ref,root=folder)
    epochs=2 if technical else 30
    if (report['task']!=task or report['cell']!=cell(task) or report['technical']!=technical or report['epochs']!=epochs
        or len(report['history'])!=epochs or report['native_reload'] is not True or report['oracle'] is not False
        or report['primary_readout']!='EMA05' or set(report['selected_readouts'])!=set(READOUTS)
        or report['precision']!='fp32' or report['global_batch']!=128 or report['world_size']!=1
        or report['effective_batch']!=128*cell(task)['accumulation'] or report['target_hash']!=(spec['embedding_hash'] if cell(task)['embedding_weight'] else None)
        or (not technical and (report['world_size']!=1 or report['counts']!=spec['counts']))):raise ValueError('Fit meaning differs')
    from .identity import validate_initial
    initial=load_json(folder/'initialization.json');validate_initial(spec,task,initial,report['world_size'])
    from .identity import check_start,starts
    check_start(spec,task,initial)
    if report['initial_states']!=starts(spec)['content_hash']:raise ValueError('Canonical initialization parent differs')
    if report['initialization']!=initial['content_hash']:raise ValueError('Initialization parent differs')
    for e,row in enumerate(report['history'],1):
        if row!=load_json(folder/f'epoch-{e:02}.json') or row['bn_membership']!=report['bn_membership']:raise ValueError('Epoch file differs')
        validate_epoch(spec,task,row,e,report['counts'],report['world_size'],technical)
    reference=None
    for name in READOUTS:
        best=max((r for r in report['history'] if name in r['readouts']),key=lambda r:selection_key(r['readouts'][name],r['update']))
        selected=report['selected_readouts'][name]
        if selected!=dict(epoch=best['epoch'],update=best['update'],validation=best['readouts'][name]):raise ValueError('Readout selection differs')
        deployment=load_json(folder/name/'deployment.json');validate(deployment,'DEPLOYMENT',parents(spec))
        if (deployment['task']!=task or deployment['readout']!=name or deployment['native_only'] is not True or deployment['additional_parameters']!=0
            or deployment['checkpoint']!=file_ref(folder/name/'selected.pt',root=folder)
            or not deployment['state_keys'] or any(not n.startswith('mod.') for n in deployment['state_keys'])):raise ValueError('Deployment includes a training head')
        arrays=load_npz_arrays(folder/name/'validation_predictions.npz');validate_predictions(arrays,report['counts']['validation'])
        if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=selected['validation']:raise ValueError('Predictions differ')
        if reference is None:reference=arrays
        elif any(not np.array_equal(reference[n],arrays[n]) for n in ('identities','labels')):raise ValueError('Readout identities differ')
    if report['selected_epoch']!=report['selected_readouts']['EMA05']['epoch'] or report['validation']!=report['selected_readouts']['EMA05']['validation']:raise ValueError('Primary selection differs')
    return report,reference
