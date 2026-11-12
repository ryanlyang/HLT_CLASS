"""Full thirty-epoch fits, frozen selection rules, target diagnostics and native reload."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,load_npz_arrays,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact,validate,recipe
from .registry import cell,CONDITIONS,SCALES,PRODUCERS
from .storage import validate_population
from . import distributed as d,model,banks,losses,execution


def selection(task,row):
    if cell(task)['role']=='assistant':return (-row['u100_validation_error'],row['validation']['macro_ovr_auc'],-row['update'])
    if task in ('TR','TP'):
        m=list(row['components'].values())
        return (float(np.mean([x['macro_ovr_auc'] for x in m])),-float(np.mean([x['cross_entropy'] for x in m])),-row['update'])
    return selection_key(row['validation'],row['update'])


def component_names(task):
    if task=='TR':return list(CONDITIONS)
    if task=='TP':return [source+'_'+name for source in ('HLT','OFFLINE') for name in ('whole',*SCALES)]
    return [cell(task)['input']]


def evaluate(student,task,ds,ctx):
    c=cell(task);view=ds.view('validation','D000');components={};arrays={};error=None
    def add(name,**kwargs):
        p,_=d.predict(student,ds,'validation',ctx,**kwargs)
        arrays[name]=p;components[name]=ctx.main(lambda:evaluate_probabilities(view.labels,p))
    if task=='TR':
        for condition in CONDITIONS:add(condition,condition=condition)
        primary='B'
    elif task=='TP':
        for source in ('HLT','OFFLINE'):
            add(source+'_whole',source=source)
            for scale in SCALES:add(source+'_'+scale,source=source,scale=scale)
        primary='OFFLINE_whole'
    else:
        primary=c['input'];add(primary,coordinate=primary)
        if c['role']=='assistant':
            z,_=d.predict(student,ds,'validation',ctx,coordinate=primary,predicted=True)
            u=ds.u100('validation')
            error=float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(z),torch.from_numpy(np.array(u['targets'])),beta=1.,reduction='mean'))
    return dict(validation=components[primary],components=components,u100_validation_error=error),arrays


def target_diagnostics(student,task,ds,target,ctx):
    """Fixed validation panel, selected/frozen targets, no optimization capability."""
    view=ds.view('validation','D000');ix=banks.panel(view);c=cell(task);student.eval();results={}
    u=ds.u100('validation');p,_=d.predict(student,ds,'validation',ctx,coordinate=c['input'] if c['input'] not in ('conditions','sources') else 'D000',
                                         indices=ix,predicted=c['role']!='teacher',embedding=c['role']=='teacher')
    if c['role']!='teacher':
        results['original_u100']=error_summary(p,u['targets'][ix],np.asarray(ds.u100('train')['targets']).mean(0))
    else:
        results['native_embedding']=dict(variance=float(np.var(p.astype(np.float64),axis=0).mean()))
        if task=='TR':
            energies={}
            for condition in CONDITIONS[1:]:
                h,_=d.predict(student,ds,'validation',ctx,condition=condition,indices=ix,embedding=True)
                energies[condition]=float(np.sqrt(np.mean((h.astype(np.float64)-p)**2)))
            results['raw_condition_response_rms']=energies
        else:
            occupancy={}
            for source in ('HLT','OFFLINE'):
                for scale in SCALES:
                    counts=[]
                    for start in range(0,len(ix),128):
                        _,v=ds.crops('validation',ix[start:start+128],source,scale);counts.append(v)
                    occupancy[source+'_'+scale]=np.concatenate(counts).mean(0).tolist()
            results['regional_valid_fraction']=occupancy
    if target.parent is not None:
        parent_task=c['parent'];checkpoint=target.parent.manifest['checkpoint']
        parent=banks.Bank(target.parent.store.root.parent,ds.spec,parent_task,view,checkpoint,
                          technical=target.parent.manifest['technical'])
        # The producer and consumer use the same fixed panel; no target-driven sampling.
        ix=parent.indices;pred=[];positions=[]
        with torch.inference_mode():
            for start in range(0,len(ix),128):
                pos=np.arange(start,min(start+128,len(ix)));local=np.array_split(pos,ctx.world)[ctx.rank]
                if not len(local):continue
                raw=ds.view('validation',c['input']).batch(ix[local]);out=student(**d.arrays(raw,ctx))
                pred.append(out['u100' if c['role']=='assistant' else 'extra'].cpu().numpy());positions.append(local)
        shape=(128,) if c['family']=='ladder' else (len(c.get('queries',[])) or (1 if c['scales']==['whole'] else sum(2**(SCALES.index(s)+1) for s in c['scales'])),128)
        all_parts=ctx.all((np.concatenate(positions) if positions else np.empty(0,np.int64),np.concatenate(pred) if pred else np.empty((0,*shape),np.float32)))
        positions=np.concatenate([a for a,_ in all_parts]);prediction=np.concatenate([a for _,a in all_parts])[np.argsort(positions)]
        if not np.array_equal(np.sort(positions),np.arange(len(ix))):raise ValueError('Diagnostic panel coverage differs')
        fake=SimpleNamespace(parent=parent,c=c)
        expected=banks.Targets.extra(fake,np.arange(len(ix)))
        # TRAIN mean is computed in bounded slices; no validation fitting.
        count=None;mean=None
        for start in range(0,len(target.parent.targets),4096):
            values=target.extra(np.arange(start,min(start+4096,len(target.parent.targets))))
            a=values['extra'];mask=values.get('valid',np.ones(a.shape[:-1],bool))
            summed=(a.astype(np.float64)*mask[...,None]).sum(0);n=mask.sum(0)[...,None]
            mean=summed if mean is None else mean+summed;count=n if count is None else count+n
        result=error_summary(prediction,expected['extra'],mean/np.maximum(count,1),expected.get('valid'))
        if c['family']=='region':
            result['equal_scale_smooth_l1']=float(losses.regional(torch.from_numpy(prediction),torch.from_numpy(expected['extra']),
                torch.from_numpy(expected['valid']),expected['groups']))
        results['new_target']=result
    return results


def error_summary(prediction,target,mean,valid=None):
    p=torch.from_numpy(np.array(prediction));t=torch.from_numpy(np.array(target));m=torch.from_numpy(np.array(np.broadcast_to(mean,target.shape),copy=True)).to(t.dtype)
    error=losses.smooth(p,t);baseline=losses.smooth(m,t)
    v=torch.ones_like(error,dtype=torch.bool) if valid is None else torch.from_numpy(np.asarray(valid))
    return dict(smooth_l1=float((error*v).sum()/v.sum().clamp_min(1)),mean_predictor_smooth_l1=float((baseline*v).sum()/v.sum().clamp_min(1)),
        valid_fraction=float(v.float().mean()),prediction_variance=float(np.var(prediction.astype(np.float64),axis=0).mean()),
        target_variance=float(np.var(np.asarray(target,np.float64),axis=0).mean()))


def validate_diagnostics(row):
    times=[row[k] for k in ('train_seconds','validation_seconds','diagnostic_seconds','jets_per_second')]
    if not np.isfinite(times).all() or min(times[:2])<=0 or times[2]<0 or times[3]<=0:
        raise ValueError('Invalid epoch throughput/timing')
    coverage=row['target_coverage']
    if set(coverage)!={'valid_query_fraction','valid_jet_fraction'} or any(not np.isfinite(v) or not 0<=v<=1 for v in coverage.values()):
        raise ValueError('Invalid epoch target coverage')
    memory=row['memory']
    if memory['host_peak_bytes']<=0 or not memory['ranks'] or any(r['peak_rss_bytes']<=0 or r['gpu_peak_bytes']<0 for r in memory['ranks']):
        raise ValueError('Missing epoch resource observation')
    expected=row['epoch'] in recipe()['diagnostic_epochs']
    if bool(row['gradient_diagnostic'] is not None)!=expected:raise ValueError('Required gradient observation missing')
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return isinstance(value,(int,float)) and np.isfinite(value)
    if expected:
        gradient=row['gradient_diagnostic']
        norms={k+'_norm' for k in ('ce','weighted_embedding','weighted_extra')}
        cosines={a+'__'+b+'_cosine' for a,b in (('ce','weighted_embedding'),('ce','weighted_extra'),('weighted_embedding','weighted_extra'))}
        if set(gradient)!=norms|cosines or not finite(gradient) or any(gradient[k]<0 for k in norms):raise ValueError('Gradient diagnostic differs')
        if not row['technical'] and (not row['target_diagnostic'] or not finite(row['target_diagnostic'])):
            raise ValueError('Required evaluation target observation missing')


def fit(folder,spec,task,ds,target,ctx,*,technical=False,epochs=None,train_indices=None,val_diagnostics=True):
    c=cell(task);folder=Path(folder);ctx.main(lambda:folder.mkdir(parents=True,exist_ok=False))
    if not technical and (ctx.world!=execution.world_size(spec,task) or epochs is not None or train_indices is not None):
        raise PermissionError('Science requires the registered rank layout and full30 population')
    train=ds.view('train','D000');val=ds.view('validation','D000')
    if set(map(bytes,train.identities)) & set(map(bytes,val.identities)):raise ValueError('Training/validation overlap')
    if not technical and dict(train=len(train),validation=len(val))!=spec['counts']:raise ValueError('Population differs')
    epoch_count=(2 if epochs is None else epochs) if technical else 30
    indices=np.arange(len(train),dtype=np.int64) if train_indices is None else np.asarray(train_indices,np.int64)
    steps=math.ceil(len(indices)/128);parents=dict(spec=spec['content_hash'],recipe=recipe()['content_hash'],**target.parents)
    student,wrapped,opt=d.build(spec,task,ctx);history=[];best=None;selected=None;state=None
    for epoch in range(1,epoch_count+1):
        order=indices[np.random.default_rng(np.random.SeedSequence([spec['nodes'][task]['sampler_seed'],epoch])).permutation(len(indices))]
        sums={};coverage_sums={};diagnostic=None
        def epoch_updates():
            nonlocal diagnostic
            for batch,start in enumerate(range(0,len(order),128)):
                ix=order[start:start+128];update=(epoch-1)*steps+batch+1
                coverage={}
                values,diag=d.step(student,wrapped,opt,task,ds,target,ix,epoch,batch,update/steps,ctx,
                    diagnose=batch==0 and epoch in recipe()['diagnostic_epochs'],coverage=coverage)
                if diag is not None:diagnostic=diag
                for k,v in values.items():sums[k]=sums.get(k,0.)+v*len(ix)
                for k,v in coverage.items():coverage_sums[k]=coverage_sums.get(k,0.)+v*len(ix)
                if ctx.rank==0 and (batch%100==0 or batch+1==steps):
                    print(f'EMBEDDING_BRIDGE {task} epoch={epoch}/{epoch_count} batch={batch+1}/{steps}',flush=True)
        seconds,_=d.timed(epoch_updates,ctx)
        validation_seconds,(metrics,predictions)=d.timed(lambda:evaluate(student,task,ds,ctx),ctx)
        diag_seconds=0.;td=None
        if val_diagnostics and epoch in recipe()['diagnostic_epochs']:
            diag_seconds,td=d.timed(lambda:target_diagnostics(student,task,ds,target,ctx),ctx)
        row=artifact('EPOCH',parents=parents,task=task,epoch=epoch,update=epoch*steps,learning_rate=learning_rate(epoch),
            weighted_losses={k:v/len(order) for k,v in sums.items()},gradient_diagnostic=diagnostic,target_diagnostic=td,
            memory=d.memory(ctx),jets_per_second=len(order)/seconds,target_coverage={k:v/len(order) for k,v in coverage_sums.items()},
            technical=technical,train_seconds=seconds,validation_seconds=validation_seconds,diagnostic_seconds=diag_seconds,**metrics)
        history.append(row);key=selection(task,row)
        if best is None or key>best:
            best=key;selected=row;state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        ctx.main(lambda:write_immutable_json(folder/f'epoch-{epoch:02}.json',row))
        if ctx.rank==0:
            auc=row['validation']['macro_ovr_auc']; rec=recovery_percent(auc,spec['recovery_anchors'])
            print(f'EMBEDDING_BRIDGE {task} epoch={epoch}/{epoch_count} AUC={auc:.8f} recovery={rec:.1f}% best_epoch={selected["epoch"]}',flush=True)
    student.load_state_dict(state,strict=True);student.eval()
    ctx.main(lambda:save_state(folder/'selected_full.pt',state));ctx.main(lambda:save_state(folder/'selected.pt',student.native_state()))
    metrics,probabilities=evaluate(student,task,ds,ctx)
    if metrics['validation']!=selected['validation'] or metrics['components']!=selected['components']:
        raise ValueError('Selected checkpoint replay differs')
    native=model.Native().to(ctx.device).eval();native.load_state_dict(student.native_state(),strict=True)
    coord=c['input'] if c['input'] not in ('conditions','sources') else 'D000' if task=='TR' else 'OFFLINE'
    actual,_=d.predict(native,ds,'validation',ctx,coordinate=coord)
    primary=coord if task not in ('TR','TP') else 'B' if task=='TR' else 'OFFLINE_whole'
    np.testing.assert_allclose(actual,probabilities[primary],rtol=1e-5,atol=1e-6)
    names=list(probabilities)
    def publish_predictions():
        atomic_publish_bytes(folder/'validation_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,
            labels=val.labels,probabilities=probabilities[primary],components=np.stack([probabilities[k] for k in names],1))))
    ctx.main(publish_predictions);fit_memory=d.memory(ctx)
    report=artifact('TRAINING_REPORT',parents=parents,task=task,technical=technical,epochs=epoch_count,history=history,
        selected_epoch=selected['epoch'],validation=selected['validation'],components=selected['components'],component_names=names,
        counts=dict(train=len(indices),validation=len(val)),native_reload=True,model=c,world_size=ctx.world,precision='fp32',
        node=spec['nodes'][task],fit_memory=fit_memory,oracle=c['role']!='hlt')
    def publish():
        write_immutable_json(folder/'training_report.json',report)
        write_immutable_json(folder/'receipt.json',artifact('FIT',parents=dict(spec=spec['content_hash'],report=report['content_hash']),task=task,
            outputs=[file_ref(folder/n,root=folder) for n in ('selected_full.pt','selected.pt','validation_predictions.npz','training_report.json')]))
    ctx.main(publish)
    del native,wrapped,opt,state;d.release_memory(ctx)
    return student,report


def validate_fit(folder,spec,task,*,technical=False):
    folder=Path(folder);report=load_json(folder/'training_report.json');receipt=load_json(folder/'receipt.json')
    validate(report,'TRAINING_REPORT');validate(receipt,'FIT',dict(spec=spec['content_hash'],report=report['content_hash']))
    if (receipt['task']!=task or [r['path'] for r in receipt['outputs']]!=['selected_full.pt','selected.pt','validation_predictions.npz','training_report.json']):
        raise ValueError('Fit output inventory differs')
    for ref in receipt['outputs']:validate_file_ref(ref,root=folder)
    if (report['task']!=task or report['technical']!=technical or report['model']!=cell(task)
            or report['parents'].get('spec')!=spec['content_hash'] or report['parents'].get('recipe')!=recipe()['content_hash']
            or report['node']!=spec['nodes'][task] or report['native_reload'] is not True or report['precision']!='fp32'
            or report['oracle']!=(cell(task)['role']!='hlt') or len(report['history'])!=report['epochs']
            or report['component_names']!=component_names(task) or set(report['components'])!=set(component_names(task))
            or (not technical and (report['epochs']!=30 or report['world_size']!=execution.world_size(spec,task) or report['counts']!=spec['counts']))):
        raise ValueError('Fit scientific registration differs')
    for e,row in enumerate(report['history'],1):
        validate(row,'EPOCH',report['parents'])
        if (row!=load_json(folder/f'epoch-{e:02}.json') or row['epoch']!=e or row['task']!=task
                or row['technical']!=technical or row['learning_rate']!=learning_rate(e)
                or row['update']!=e*math.ceil(report['counts']['train']/128)
                or not np.isfinite(list(row['weighted_losses'].values())).all()):raise ValueError('Epoch evidence differs')
        validate_diagnostics(row)
    selected=max(report['history'],key=lambda row:selection(task,row))
    if report['selected_epoch']!=selected['epoch'] or report['validation']!=selected['validation'] or report['components']!=selected['components']:
        raise ValueError('Checkpoint selection differs')
    a=load_npz_arrays(folder/'validation_predictions.npz')
    validate_population(a['identities'],a['labels'])
    if (len(a['labels'])!=report['counts']['validation'] or a['components'].shape!=(report['counts']['validation'],len(report['component_names']),11)
            or a['probabilities'].shape!=(report['counts']['validation'],11)):raise ValueError('Evaluation matrix shape differs')
    if evaluate_probabilities(a['labels'],a['probabilities'])!=report['validation']:raise ValueError('Saved native metrics differ')
    for i,name in enumerate(report['component_names']):
        if evaluate_probabilities(a['labels'],a['components'][:,i])!=report['components'][name]:raise ValueError('Saved teacher condition metrics differ')
    return report,a
