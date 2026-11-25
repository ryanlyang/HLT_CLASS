"""Check longest real authorized rows before building the full resident inputs."""
import gc
import time
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import sha256_file, write_immutable_json
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot
from . import contracts as c, population, views, model, training, recovery, pair_storage


def witnesses(spec,stage,loaded):
    """Only three particle rows: longest eligible TRAIN, in-val and outside-val."""
    p,f,members,roles,_,_=loaded
    coordinate=c.COORDINATES[stage]
    key='concat' if coordinate=='CONCAT' else 'offline' if coordinate=='OFFLINE' else 'hlt'
    chosen={r:None for r in ('train','validation','outside_validation')}
    for i,(a,rs) in enumerate(zip(members,roles,strict=True)):
        sizes=(a['jet_nparticles']+a['hlt_jet_nparticles'] if key=='concat' else
               a['jet_nparticles'] if key=='offline' else a['hlt_jet_nparticles'])
        for role,mask in population.masks(a,rs).items():
            ix=np.flatnonzero(mask)
            if not len(ix):continue
            j=int(ix[np.argmax(sizes[ix])]);n=int(sizes[j])
            if chosen[role] is None or n>chosen[role][0]:chosen[role]=(n,i,j)
    caches={};records={};hashed={}
    for code,(role,witness) in enumerate(chosen.items()):
        if witness is None:raise ValueError('Missing early memory witness: '+role)
        n,i,j=witness
        if n!=p['maximum_support'][role][key]:raise ValueError('Longest witness differs from population')
        record=f['files'][i];path=Path(spec['dataset'])/record['path']
        if i not in hashed:
            if sha256_file(path)!=record['sha256']:raise ValueError('Witness raw source differs')
            hashed[i]=path
        a=members[i];member={k:a[k][j:j+1] for k in ('jet_nparticles','hlt_jet_nparticles','labels')}
        blocks,_=views.work((spec['dataset'],record,i,j,member,np.array([code],np.uint8),coordinate,p['content_hash']))
        if set(blocks)!={role}:raise ValueError('Witness read crossed its authorized role')
        cache=RamCache([blocks[role]],role='train' if role=='train' else 'validation',
                       coordinate_name=coordinate,foundation_sha256=p['content_hash'])
        if len(cache)!=1 or int(views.lengths(cache)[0])!=n:raise ValueError('Witness particle support differs')
        caches[role]=cache
        records[role]=dict(file_index=i,entry=j,particles=n,identity=cache.identities[0].tobytes().hex())
    for i,path in hashed.items():
        if sha256_file(path)!=f['files'][i]['sha256']:raise ValueError('Witness source changed during read')
    return caches,records


def run(spec,stage,loaded,device,folder):
    begun=time.monotonic();device=torch.device(device)
    caches,records=witnesses(spec,stage,loaded)
    storage=recovery.enabled(spec,stage)
    pointwise=spec['schema_version']==3 and stage=='CONCAT'
    print('BALANCED early memory: longest real rows loaded; before full particle construction',flush=True)
    parity=[]
    if storage:
        # A small batch with full particle support exercises the installed CUDA kernels.
        raw=caches['train'].batch(np.zeros(2,np.int64))
        for bf16 in (False,True):parity.append(pair_storage.parity(raw,device,bf16=bf16))
        if pointwise:
            for bf16 in (False,True):parity.append(pair_storage.parity(raw,device,bf16=bf16,pointwise_dgrad=True))
        del raw;gc.collect();torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    net=model.new_model(stage,**recovery.model_options(spec,stage)).to(device).train()
    opt=training.optimizer(net,c.recipe())
    raw=caches['train'].batch(np.zeros(c.BATCH,np.int64))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits=model.forward(net,raw,device,bf16=True)
        loss=torch.nn.functional.cross_entropy(logits.float(),torch.as_tensor(raw['labels'],device=device,dtype=torch.long))
        loss.backward()
        if not torch.isfinite(loss):raise ValueError('Nonfinite early memory loss')
        bad=[name for name,p in net.mod.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        if bad:raise ValueError('Nonfinite/disconnected native gradient: '+', '.join(bad))
        opt.step()
        del logits,loss
    stats=dict(net.pair_storage.stats) if storage else None
    if storage:pair_storage.validate_storage_stats(stats,calls=3)
    gradient={}
    if pointwise:
        from .pair_convolution import POLICY,validate_stats
        g=dict(net.pair_convolution.stats);validate_stats(g,require_fallback=True)
        gradient=dict(pointwise_policy=POLICY,pointwise_statistics=g)
    opt.zero_grad(set_to_none=True);net.eval()
    with torch.inference_mode():
        for role in ('validation','outside_validation'):
            output=model.forward(net,caches[role].batch(np.zeros(c.BATCH,np.int64)),device,bf16=True)
            if not torch.isfinite(output).all():raise ValueError('Invalid early validation forward')
            del output
        if stage!='DE000':
            output=model.forward(net,caches['train'].batch(np.zeros(c.EXPORT_BATCH,np.int64)),device,bf16=False)
            if not torch.isfinite(output).all():raise ValueError('Invalid early FP32 export forward')
            del output
    torch.cuda.synchronize(device);memory=memory_snapshot(torch)
    measured=c.artifact('EARLY_MEMORY',parents=dict(spec=spec['content_hash']),stage=stage,
        witnesses=records,parity=parity,pair_storage=stats,batch_size=c.BATCH,export_batch=c.EXPORT_BATCH,
        native_steps=3,seconds=time.monotonic()-begun,memory=memory,passed=True,
        scope='native_graph_only_full_objective_acceptance_still_required',**gradient)
    if memory['gpu_peak_bytes']>torch.cuda.get_device_properties(device).total_memory*.92:
        raise MemoryError('Early full-support CUDA peak lacks headroom; full input load not started')
    if memory['peak_rss_bytes']+12*2**30>spec['memory_mb']*1024**2*.95:
        raise MemoryError('Early host peak lacks headroom')
    write_immutable_json(Path(folder)/'early_memory.json',measured)
    del net,opt,caches,raw;gc.collect();torch.cuda.empty_cache()
    print('BALANCED early memory passed; now building full resident inputs',flush=True)
    return measured
