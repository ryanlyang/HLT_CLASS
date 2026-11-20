"""Real four-card parity, discarded miniatures, complete support and measured release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_sixclass_balanced_oscar.sampling import BalancedSampler
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from hlt_classification.luka_reconstruction_stability_hydra.kernel import snapshot,compare
from hlt_classification.luka_reconstruction_stability_hydra.readouts import recalibrate,native_copy
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as assignment
from hlt_classification.luka_fullstat_readout.training import optimizer
from hlt_classification.luka_fullstat.recipe import learning_rate
from . import contracts as c,kernel,model,artifacts,readouts,training,diagnostics as log

TOLERANCES=dict(loss_abs=1e-3,gradient_relative_l2=5e-4,parameters_relative_l2=5e-4,
                adamw_moments_relative_l2=5e-4,buffer_abs=1e-4)


def host_snapshot(pid=None):
    """PSS counts the shared mmap pages once across the four worker processes."""
    pid=os.getpid() if pid is None else pid
    fields={}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0] in ('Pss:','Rss:'):fields[parts[0][:-1]]=int(parts[1])*1024
    if set(fields)!={'Pss','Rss'}:raise RuntimeError('Linux shared-memory accounting unavailable')
    return fields


def memory(ctx):
    measured=d.memory(ctx)
    local=host_snapshot()
    local['transient_peak']=max(0,d.host_usage()-local['Rss'])
    rows=ctx.all(local)
    def launchers():
        data_pid=int(os.environ['LUKABREC_INPUT_LAUNCHER_PID'])
        return {str(pid):host_snapshot(pid) for pid in {os.getppid(),data_pid}}
    launcher=ctx.main(launchers)
    measured.update(host_accounting='sum_PSS_plus_private_transient_peak_and_launcher',host_processes=rows,
        launcher=launcher,host_peak_bytes=sum(r['Pss']+r['transient_peak'] for r in rows)+sum(r['Pss'] for r in launcher.values()))
    return measured


class Subset:
    def __init__(self,cache,indices):
        self.source=cache;self.indices=np.asarray(indices,np.int64);self.labels=cache.labels[self.indices]
        self.identities=cache.identities[self.indices];self.role=cache.role;self.coordinate_name=cache.coordinate_name
    def __len__(self):return len(self.indices)
    def batch(self,ix):return self.source.batch(self.indices[ix])


def target_values(task,indices,bank=None):
    x=c.cell(task);n=len(indices);rng=np.random.default_rng(2026111931)
    # Technical-only substitutes are never published as a scientific target bank.
    emb=(np.asarray(bank.targets[indices]) if bank is not None else rng.normal(size=(n,128)).astype(np.float32)) if x['embedding_weight'] else None
    q=(np.asarray(bank.probabilities[indices]) if bank is not None else np.full((n,6),1/6,np.float32)) if x['main_kd_weight'] else None
    return emb,q


def independent_step(net,opt,raw,emb,q,targets,task,device,position):
    """Full graph, no decoder chunks/VJP injection/reduction helpers."""
    net.train();net.capture=bool(c.cell(task)['embedding_weight']);opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=learning_rate(position)
    logits=kernel.native(net,raw,device);x=c.cell(task)
    base=x['ce_weight']*F.cross_entropy(logits,torch.as_tensor(raw['labels'],device=device).long())
    if emb is not None:base=base+5*F.smooth_l1_loss(net.prediction,torch.as_tensor(emb,device=device))
    if q is not None:base=base+3*F.kl_div(F.log_softmax(logits/2,dim=-1),torch.as_tensor(q,device=device),reduction='batchmean')
    ew,dw=c.weights(task,position);aux=base.new_zeros(())
    if targets is not None and dw>0:
        # Decoder gets dw, encoder gets ew, including D20's detached preparation.
        ratio=ew/dw;h=net.jet.detach()+ratio*(net.jet-net.jet.detach())
        t=net.particle_tokens.detach()+ratio*(net.particle_tokens-net.particle_tokens.detach())
        scores=net.aux(h,t,net.particle_mask);rows,_=assignment.rows(scores,targets,False)
        aux=rows.mean()
    loss=base+dw*aux
    if targets is not None and dw==0:loss=loss+sum((p.sum()*0 for p in net.aux.parameters()),base.new_zeros(()))
    loss.backward();opt.step()
    return snapshot(net,opt,float((base+ew*aux).detach()))


def parity(spec,task,tr,targets,bank,ctx,position=8.):
    # A full16 reference fits on one card while the production path shards4x4.
    ix=np.arange(min(16,len(tr)));raw=tr.batch(ix);emb,q=target_values(task,ix,bank)
    truth=targets.batch(ix) if targets is not None else None
    state,_=artifacts.start(spec,task)
    local=SimpleNamespace(rank=0,world=1,device=ctx.device)
    reference,ro=kernel.build(task,spec['maximum'],local,state,parity=True)
    expected=independent_step(reference,ro,raw,emb,q,truth,task,ctx.device,position)
    del reference,ro;d.release_memory(ctx)
    actual,ao=kernel.build(task,spec['maximum'],ctx,state,parity=True)
    losses=kernel.update(actual,ao,raw,emb,q,truth,task,ctx,position)
    measured=snapshot(actual,ao,losses['total']);result=compare(expected,measured,TOLERANCES)
    del actual,ao;d.release_memory(ctx)
    ranks=ctx.all(result)
    if any(r['passed'] is not True for r in ranks):raise ValueError('Independent full-graph/four-rank parity failed: '+str(ranks))
    return dict(position=position,ranks=ranks,tolerances=TOLERANCES)


def representative(cache):
    counts=np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    selected=[int(np.flatnonzero(cache.labels==label)[0]) for label in range(6)]
    selected+=[int(counts.argmax()),*range(min(128,len(cache)))]
    return np.asarray(list(dict.fromkeys(selected)),np.int64),int(counts.argmax())


def miniature(spec,task,caches,targets,bank,pt,ctx):
    tr,val=caches['train'],caches['validation'];vi,_=representative(val);small=Subset(val,vi)
    sampler=BalancedSampler(tr.labels,pt,seed=c.seeds(task)['sampler'])
    order=sampler.draw(1,draws=2*c.BATCH);state,_=artifacts.start(spec,task)
    net,opt=kernel.build(task,spec['maximum'],ctx,state);averages=readouts.Averages()
    membership=readouts.membership(tr,pt,spec['population']['content_hash']);epochs=[]
    for epoch,position in enumerate((2.,8.),1):
        # One ordinary batch and the exact final64 scientific batch size.
        for ix in (order[:c.BATCH],order[c.BATCH:c.BATCH+64]):
            emb,q=target_values(task,ix,bank)
            kernel.update(net,opt,tr.batch(ix),emb,q,targets.batch(ix) if targets else None,task,ctx,position)
            averages.update(net,position,len(ix))
        measured=readouts.evaluate_all(net,averages,tr,small,membership,ctx,training.evaluate,kernel.native,model.NativeParticleTransformer)
        if epoch==2 and set(measured)!=set(c.READOUTS):raise ValueError('Miniature missing EMA/native readouts')
        epochs.append({k:v[0] for k,v in measured.items()})
        for name,(metric,p,selected) in measured.items():
            native=training.readouts_native(selected,ctx)
            np.testing.assert_allclose(training.predict(native,small,ctx),p,rtol=1e-5,atol=1e-6)
            del native
    del net,opt,averages;d.release_memory(ctx)
    return dict(discarded=True,epochs=epochs,partial_global_batch=64,native_reload=True)


def measure(spec,task,caches,targets,bank,pt,ctx,*,load_seconds):
    tr,val=caches['train'],caches['validation'];state,_=artifacts.start(spec,task)
    net,opt=kernel.build(task,spec['maximum'],ctx,state)
    sampler=BalancedSampler(tr.labels,pt,seed=c.seeds(task)['sampler']);order=sampler.draw(1,draws=20*c.BATCH)
    _,largest=representative(tr);stress=[np.full(c.BATCH,largest,np.int64)]
    if targets is not None:stress.append(np.full(c.BATCH,int(np.diff(targets.offsets).argmax()),np.int64))
    for ix in stress:
        emb,q=target_values(task,ix,bank);kernel.update(net,opt,tr.batch(ix),emb,q,targets.batch(ix) if targets else None,task,ctx,10.)
    ctx.sync();begun=time.monotonic()
    for start in range(0,len(order),c.BATCH):
        if start==4*c.BATCH:ctx.sync();begun=time.monotonic()
        ix=order[start:start+c.BATCH];emb,q=target_values(task,ix,bank)
        kernel.update(net,opt,tr.batch(ix),emb,q,targets.batch(ix) if targets else None,task,ctx,10.)
    ctx.sync();step=max(ctx.all((time.monotonic()-begun)/16))
    net.capture=False;net.eval();ctx.sync();begun=time.monotonic()
    with torch.inference_mode():
        for i in range(16):
            ix=np.arange(i*c.BATCH,(i+1)*c.BATCH)%len(val);raw=val.batch(ix)
            local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
            p=kernel.native(net,{k:v[local] for k,v in raw.items()},ctx.device).softmax(1).cpu().numpy()
    ctx.sync();vstep=max(ctx.all((time.monotonic()-begun)/16))
    membership=readouts.membership(tr,pt,spec['population']['content_hash'])
    native=native_copy(net.deployment_state(),ctx,model.NativeParticleTransformer)
    bseconds,_=d.timed(lambda:recalibrate(native,tr,membership,ctx,kernel.native),ctx)
    del native
    export_step=0.
    if c.cell(task)['arm'] in ('OFFLINE','U100'):
        ctx.sync();begun=time.monotonic()
        with torch.inference_mode():
            for i in range(8):
                ix=order[i*c.BATCH:(i+1)*c.BATCH];raw=tr.batch(ix);local=np.array_split(np.arange(len(ix)),ctx.world)[ctx.rank]
                raw={k:v[local] for k,v in raw.items()}
                tensors=[torch.as_tensor(raw[k],device=ctx.device) for k in ('features','vectors','mask')]
                _,h=model.head_input(net,*tensors);h.float().cpu().numpy()
        ctx.sync();export_step=max(ctx.all((time.monotonic()-begun)/8))
    measured_memory=memory(ctx)
    for i,r in enumerate(measured_memory['ranks']):
        if r['gpu_peak_bytes']>torch.cuda.get_device_properties(ctx.device).total_memory*.88:
            raise MemoryError('Full-support GPU peak exceeds88% memory envelope')
    if measured_memory['host_peak_bytes']>c.RESOURCES['memory_mb']*1024**2*.85:raise MemoryError('Measured aggregate host peak exceeds85% allocation')
    # Includes actual all-gather, row-order reconstruction and six-class metrics;
    # a forward-only extrapolation can hide slow CPU validation on Pascal nodes.
    validation_seconds,_=d.timed(lambda:training.evaluate(net,val,ctx),ctx)
    train_steps=math.ceil(c.COUNTS['train']/c.BATCH);val_steps=math.ceil(c.COUNTS['validation']/c.BATCH)
    seconds=1.5*(load_seconds+c.EPOCHS*(train_steps*step+4*validation_seconds+3*bseconds)+4*validation_seconds+train_steps*export_step)+3600
    if seconds>c.RESOURCES['minutes']*60:raise RuntimeError(f'Measured {seconds/3600:.1f}h exceeds48h; explicit resource revision required')
    del net,opt;d.release_memory(ctx)
    return dict(train_step_seconds=step,validation_step_seconds=vstep,bn_seconds=bseconds,export_step_seconds=export_step,
        validation_pass_seconds=validation_seconds,input_build_seconds=load_seconds,
        conservative_seconds=seconds,memory=measured_memory,full_support=True)


def qualify(spec,views,pt,ctx,folder):
    if ctx.world!=4:raise PermissionError('Authoritative qualification requires four actual GPUs')
    rows={};folder=Path(folder)
    from hlt_classification.luka_sixclass_balanced_oscar.model import installed_parity
    with log.stage('installed-Weaver parity'):
        installed=installed_parity(views['D000'][0]['train'],device=ctx.device)
    for arm in c.ARMS:
        task=f'{arm}_s{c.SEEDS[0]}';coordinate=c.cell(task)['coordinate']
        caches,targets,node=views[coordinate]
        target=targets if c.cell(task)['reconstruction'] else None
        if ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats(ctx.device)
        with log.stage(f'qualification arm={arm} full-weight parity'):
            proof=parity(spec,task,caches['train'],target,None,ctx)
        # Check both sides of the delayed/asymmetric decoder transition.
        if c.cell(task)['reconstruction']:
            with log.stage(f'qualification arm={arm} delayed-weight parity'):
                proof['delay']=parity(spec,task,caches['train'],target,None,ctx,position=2.)
        mini={}
        for seed in c.SEEDS:
            with log.stage(f'qualification arm={arm} seed={seed} miniature'):
                mini[str(seed)]=miniature(spec,f'{arm}_s{seed}',caches,target,None,pt,ctx)
        with log.stage(f'qualification arm={arm} resource measurement and full validation'):
            measured=measure(spec,task,caches,target,None,pt,ctx,load_seconds=node['build_seconds'])
        row=c.artifact('QUALIFIED_ARM',parents=dict(spec=spec['content_hash']),task=task,
            parity=proof,miniature=mini,measured=measured,teacher_targets='technical_synthetic_only',passed=True)
        ctx.main(lambda:write_immutable_json(folder/(arm+'.json'),row));rows[arm]=row['content_hash']
        log.mark('qualification arm proof published: '+arm)
    result=c.artifact('QUALIFICATION',parents=dict(spec=spec['content_hash']),arms=rows,
        world=4,gpu_type=c.RESOURCES['gpu_type'],complete_support=True,miniatures_discarded=True,passed=True,
        installed_weaver_parity=installed,
        target_normalization_sha256=views['D000'][1].normalization['content_hash'],
        files={a:file_ref(folder/(a+'.json')) for a in rows})
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'qualification.json',result));return result


def qualification(spec):
    q=load_json(Path(spec['root'])/'qualification.json');c.validate(q,'QUALIFICATION',dict(spec=spec['content_hash']))
    if (set(q['arms'])!=set(c.ARMS) or q['world']!=4 or q['gpu_type']!=c.RESOURCES['gpu_type']
            or q['passed'] is not True or q['miniatures_discarded'] is not True or q['complete_support'] is not True):
        raise ValueError('Missing real four-card qualification')
    if set(q['files'])!=set(c.ARMS) or q['installed_weaver_parity'].get('passed') is not True:
        raise ValueError('Missing installed-Weaver or arm coverage')
    for arm,ref in q['files'].items():
        row=load_json(validate_file_ref(ref));c.validate(row,'QUALIFIED_ARM',dict(spec=spec['content_hash']))
        if (row['content_hash']!=q['arms'][arm] or row['passed'] is not True
                or set(row['miniature'])!={str(seed) for seed in c.SEEDS}):raise ValueError('Qualification arm differs')
        if row['measured']['conservative_seconds']>2880*60:raise ValueError('Qualification exceeds walltime')
        proofs=[row['parity'],*([row['parity']['delay']] if 'delay' in row['parity'] else [])]
        if (row['task']!=f'{arm}_s{c.SEEDS[0]}' or row['teacher_targets']!='technical_synthetic_only'
                or row['measured']['full_support'] is not True
                or any(len(p['ranks'])!=4 or any(r['passed'] is not True for r in p['ranks']) for p in proofs)
                or any(m['discarded'] is not True or m['native_reload'] is not True or len(m['epochs'])!=2
                       for m in row['miniature'].values())):raise ValueError('Incomplete four-card measured proof')
    return q
