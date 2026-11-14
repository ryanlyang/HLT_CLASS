"""Original FullStat20 CE updates, plus authenticated epoch-boundary recovery."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, deterministic_npz_bytes,
    load_json, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.inputs import new_model
from hlt_classification.jetclass2_delphes.cache import RamBlock,RamCache
from hlt_classification.jetclass2_delphes.runner import _forward,predict,selection_key,train_kernel
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_readout.training import optimizer,cpu_state
from hlt_classification.luka_fullstat_balanced_ladder.artifacts import save_state
from . import contracts as c


def committed(root, spec_hash, steps):
    rows=[];states=[]
    for receipt in sorted(Path(root).glob('epochs/*/receipt.json')):
        value=load_json(receipt);c.validate(value,'EPOCH_COMMIT')
        if value['spec']!=spec_hash:raise ValueError('Epoch belongs to another spec')
        row=value['row'];epoch=row['pass_number']
        if epoch!=len(rows)+1 or row['update']!=epoch*steps or row['learning_rate']!=learning_rate(epoch):
            raise ValueError('Noncontiguous or duplicate epoch commits')
        state=validate_file_ref(value['state'],root=Path(root))
        rows.append(row);states.append(state)
    return rows,states


def run(tr,val,root,job,*,spec_hash,device='cuda',model_factory=None,epochs=20,batch=128):
    # Small overrides are test-only; the production wrapper pins all three.
    if tr.role!='train' or val.role!='validation' or tr.foundation_sha256!=val.foundation_sha256:
        raise ValueError('Training/validation bindings differ')
    if tr.coordinate_name!='OFFLINE' or val.coordinate_name!='OFFLINE':raise ValueError('OFFLINE view required')
    root=Path(root);node=c.node();cfg=c.recipe()['base'];steps=math.ceil(len(tr)/batch)
    net=(new_model(node) if model_factory is None else model_factory()).to(device)
    opt=optimizer(net,cfg);history,states=committed(root,spec_hash,steps)
    if len(history)>epochs:raise ValueError('Too many saved epochs')
    if history:
        state=torch.load(states[-1],map_location='cpu',weights_only=True)
        if state['epoch']!=len(history) or state['update']!=len(history)*steps:raise ValueError('Resume cursor differs')
        net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif state['cuda_rng']:raise ValueError('Cannot resume CUDA fit on CPU')
    for epoch in range(len(history)+1,epochs+1):
        net.train();order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],epoch])).permutation(len(tr))
        begun=time.monotonic();loss_sum=0.
        for done,start in enumerate(range(0,len(tr),batch),1):
            update=(epoch-1)*steps+done;lr=learning_rate(update/steps)
            for group in opt.param_groups:group['lr']=lr
            indices=order[start:start+batch];raw=tr.batch(indices)
            opt.zero_grad(set_to_none=True)
            logits=_forward(net,raw,device,bf16=str(device).startswith('cuda'))
            loss=torch.nn.functional.cross_entropy(logits.float(),torch.from_numpy(raw['labels']).to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CE')
            loss.backward()
            finite=[torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all():raise ValueError('Nonfinite required gradients')
            opt.step();loss_sum+=float(loss.detach())*len(indices)
            if done%128==0:
                elapsed=time.monotonic()-begun
                print(f'OFFLINE progress epoch={epoch}/20 batch={done}/{steps} elapsed_seconds={elapsed:.1f} '
                      f'epoch_remaining_seconds={elapsed*(steps-done)/done:.1f}',flush=True)
        train_seconds=time.monotonic()-begun;begun=time.monotonic()
        metrics=evaluate_probabilities(val.labels,predict(net,val,device=device,batch_size=batch))
        row=dict(pass_number=epoch,update=update,learning_rate=lr,validation=metrics,
                 train_ce=loss_sum/len(tr),train_seconds=train_seconds,validation_seconds=time.monotonic()-begun)
        state=dict(model=cpu_state(net.state_dict()),optimizer=cpu_state(opt.state_dict()),epoch=epoch,update=update,
                   cpu_rng=torch.get_rng_state().clone(),cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()]
                   if str(device).startswith('cuda') else [])
        folder=root/'epochs'/f'{epoch:02d}_{job}';folder.mkdir(parents=True,exist_ok=False)
        save_state(folder/'state.pt',state)
        receipt=c.artifact('EPOCH_COMMIT',spec=spec_hash,row=row,state=file_ref(folder/'state.pt',root=root))
        write_immutable_json(folder/'receipt.json',receipt)
        history.append(row);states.append(folder/'state.pt')
        best=max(history,key=lambda r:selection_key(r['validation'],r['update']))
        print(f'OFFLINE epoch={epoch}/20 auc={metrics["macro_ovr_auc"]:.8f} accuracy={metrics["accuracy"]:.6f} '
              f'best_epoch={best["pass_number"]} best_auc={best["validation"]["macro_ovr_auc"]:.8f}',flush=True)
    selected=max(history,key=lambda r:selection_key(r['validation'],r['update']))
    state=torch.load(states[selected['pass_number']-1],map_location='cpu',weights_only=True)
    net.load_state_dict(state['model'],strict=True);net.eval()
    return net,history,selected


def subset(cache,indices):
    features=[];vectors=[];offsets=[0]
    for i in indices:
        raw=cache.batch(np.asarray([i],np.int64));n=int(raw['mask'].sum())
        features.append(raw['features'][0,:,:n].T);vectors.append(raw['vectors'][0,:,:n].T);offsets.append(offsets[-1]+n)
    block=RamBlock(0,np.asarray(offsets,np.int64),np.concatenate(features),np.concatenate(vectors),
                   cache.identities[indices].copy(),cache.labels[indices].copy())
    return RamCache([block],role=cache.role,foundation_sha256=cache.foundation_sha256,coordinate_name='OFFLINE')


def qualify(tr,val,folder,*,spec,remaining_epochs,started):
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
    from hlt_classification.luka_fullstat.preflight import stress
    folder=Path(folder);device='cuda';torch.cuda.reset_peak_memory_stats()
    with parity_backend(device):parity=installed_parity(tr,device=device)
    samples=[]
    for data,count in ((tr,32),(val,16)):
        indices=np.concatenate([np.flatnonzero(data.labels==k)[:count] for k in range(11)])
        if len(indices)!=11*count:raise ValueError('Miniature lacks class coverage')
        samples.append(subset(data,indices))
    net=new_model(c.node());mini,state=train_kernel(net,*samples,node=c.node(),device=device,
        training_recipe=c.recipe()['base'],acceptance_passes=2)
    save_state(folder/'miniature.pt',state)
    net.load_state_dict(torch.load(folder/'miniature.pt',map_location='cpu',weights_only=True))
    if evaluate_probabilities(samples[1].labels,predict(net,samples[1],device=device,batch_size=128))!=mini['validation']:
        raise ValueError('Miniature selected checkpoint replay differs')
    maxima={r:stress(net,data,batch=128,training=r=='train') for r,data in [('train',tr),('validation',val)]}
    # Representative full batches, not the tiny balanced miniature, drive timing.
    opt=optimizer(net,c.recipe()['base']);net.train();rng=np.random.default_rng(7713);times=[]
    for i in range(36):
        torch.cuda.synchronize();begin=time.monotonic();ix=rng.choice(len(tr),128,replace=False);raw=tr.batch(ix)
        opt.zero_grad(set_to_none=True);logits=_forward(net,raw,device,bf16=True)
        loss=torch.nn.functional.cross_entropy(logits.float(),torch.from_numpy(raw['labels']).to(device))
        if not torch.isfinite(loss):raise ValueError('Nonfinite timing loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            raise ValueError('Nonfinite timing gradient')
        opt.step();torch.cuda.synchronize()
        if i>=4:times.append(time.monotonic()-begin)
    net.eval();vt=[]
    with torch.inference_mode():
        for _ in range(16):
            torch.cuda.synchronize();begin=time.monotonic();raw=val.batch(rng.choice(len(val),128,replace=False))
            out=_forward(net,raw,device,bf16=True)
            if not torch.isfinite(out).all():raise ValueError('Nonfinite timing prediction')
            torch.cuda.synchronize();vt.append(time.monotonic()-begin)
    import resource
    gpu_peak=torch.cuda.max_memory_allocated();gpu_total=torch.cuda.get_device_properties(0).total_memory
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    elapsed=time.monotonic()-started
    seconds=elapsed+3600+2*(remaining_epochs*math.ceil(len(tr)/128)*float(np.mean(times))+
                          (remaining_epochs+1)*math.ceil(len(val)/128)*float(np.mean(vt)))
    proof=c.artifact('QUALIFICATION',spec=spec['content_hash'],parity=parity,miniature=mini,maxima=maxima,
        train_step_seconds=float(np.mean(times)),validation_step_seconds=float(np.mean(vt)),
        projected_total_seconds=seconds,timing_margin=2.,fixed_margin_seconds=3600,
        gpu_peak_bytes=gpu_peak,gpu_total_bytes=gpu_total,host_peak_bytes=rss,
        remaining_epochs=remaining_epochs,scientific_initialization_reused=False)
    write_immutable_json(folder/'qualification.json',proof)
    if gpu_peak>.9*gpu_total or rss>.85*spec['memory_mb']*1024**2:raise MemoryError('Measured qualification exceeds headroom')
    if seconds>spec['minutes']*60:raise RuntimeError('Measured runtime exceeds registered walltime; inspect qualification.json')
    del net,opt,samples;torch.cuda.empty_cache()
    return proof


def rejection(labels,probabilities,efficiency):
    from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES
    result={};qcd=labels==0
    for i,name in enumerate(CLASS_NAMES[1:],1):
        den=probabilities[:,i].astype(np.float64)+probabilities[:,0]
        score=np.divide(probabilities[:,i],den,out=np.zeros(len(labels),np.float64),where=den>0)
        values=score[labels==i];rank=max(1,math.ceil(efficiency*len(values)))
        threshold=float(np.partition(values,len(values)-rank)[len(values)-rank])
        passing=int((score[qcd]>=threshold).sum());n=int(qcd.sum())
        result[name]=dict(rejection=None if passing==0 else n/passing,background_passing=passing,
            background_count=n,threshold=threshold,achieved_signal_efficiency=float((values>=threshold).mean()))
    return result
