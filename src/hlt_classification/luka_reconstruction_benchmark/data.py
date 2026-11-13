"""Publish bounded, identical batches from authenticated original TRAIN/validation."""
from pathlib import Path
import time
import numpy as np
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_next_supervision_hydra import inputs, reconstruction_targets
from hlt_classification.luka_next_supervision_hydra.contracts import setting, seed
from hlt_classification.luka_u100_multiview.views import lengths
from . import campaign as c


NATIVE = ('features', 'vectors', 'mask')


def pack(raw, embedding=None, targets=None):
    a=dict(raw)
    if targets is not None:
        if embedding is None or len(targets)!=len(raw['labels']): raise ValueError('Incomplete target join')
        a.update(embedding=embedding, offsets=np.r_[0,np.cumsum([len(t['values']) for t in targets])].astype(np.int64))
        for key in ('values','valid'): a['target_'+key]=np.concatenate([t[key] for t in targets])
        for key in ('mean','scale','frame','aggregate'): a['target_'+key]=np.stack([t[key] for t in targets])
    return a


def unpack(a):
    raw={k:a[k] for k in (*NATIVE,'identities','labels') if k in a}
    if 'embedding' not in a:
        if set(a)!=set(NATIVE): raise ValueError('Validation must contain native inputs only')
        return raw,None,None
    expected=set(NATIVE)|{'labels','identities','embedding','offsets'}|{'target_'+k for k in ('values','valid','mean','scale','frame','aggregate')}
    if set(a)!=expected: raise ValueError('Unexpected benchmark payload fields')
    n=len(a['labels']); offsets=a['offsets']
    if (a['labels'].shape!=(n,) or a['labels'].dtype!=np.int64 or (a['labels']<0).any() or (a['labels']>10).any()
            or a['identities'].shape!=(n,32) or a['identities'].dtype!=np.uint8
            or len(np.unique(a['identities'],axis=0))!=n
            or a['embedding'].shape!=(n,128) or not np.isfinite(a['embedding']).all()):
        raise ValueError('Invalid labels, identity join or U100 targets')
    if offsets.shape!=(n+1,) or offsets[0]!=0 or (np.diff(offsets)<1).any() or offsets[-1]!=len(a['target_values']):
        raise ValueError('Invalid ragged batch')
    targets=[]
    for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        t={k:a['target_'+k][start:end] for k in ('values','valid')}
        t.update({k:a['target_'+k][i] for k in ('mean','scale','frame','aggregate')}); targets.append(t)
    return raw,a['embedding'],targets


def prepare(spec):
    # The parent has authenticated full completion evidence before creating
    # its claim. Recheck both source identities and pinned references here;
    # avoid repeating the full science-fit scan in every preparation child.
    began=time.perf_counter(); old=c.check(spec)
    folder=Path(spec['root'])/'batches'; folder.mkdir(exist_ok=False)
    train=inputs.hlt(old,'train'); val=inputs.hlt(old,'validation'); bank=inputs.embedding_bank(old,train)
    reconstruction_targets.authenticate(old); target=reconstruction_targets.Targets(old,train,'OFFLINE')
    cfg=setting('R24'); count=len(train); size=spec['global_batch']
    order=np.random.default_rng(np.random.SeedSequence([seed(old['imported'],cfg,'sampler_seed'),1])).permutation(count)
    nb=(count+size-1)//size; full=count//size
    indices={f'train_{i:02}':order[j*size:(j+1)*size] for i,j in enumerate(
        np.linspace(0,full-1,spec['warmup_batches']+spec['measured_batches'],dtype=int))}
    indices.update(longest_hlt=np.argsort(-lengths(train),kind='stable')[:size],
        largest_offline=np.argsort(-target.counts,kind='stable')[:size],partial11=order[:11],tail=order[(nb-1)*size:])
    for i,case in enumerate(spec['parity_cases']): indices[f'parity_{i}']=order[:case['rows']]
    outputs={}; times=[]; val_times=[]
    try:
        for name,ix in indices.items():
            start=time.perf_counter(); raw=train.batch(ix); emb=bank.targets[ix]; targets,_=target.batch(ix,1,cfg)
            elapsed=time.perf_counter()-start
            if name.startswith('train_'): times.append(elapsed)
            path=folder/(name+'.npz'); atomic_publish_bytes(path,deterministic_npz_bytes(pack(raw,emb,targets)))
            outputs[name]=dict(file=file_ref(path,root=folder),indices=ix.tolist(),role='train',rows=len(ix),
                construction_seconds=elapsed,hlt_maximum=int(raw['mask'].shape[-1]),offline_maximum=max(len(t['values']) for t in targets))
        for i,j in enumerate(np.linspace(0,len(val)//size-1,spec['validation_batches'],dtype=int)):
            ix=np.arange(j*size,(j+1)*size); start=time.perf_counter(); raw=val.batch(ix)
            elapsed=time.perf_counter()-start; val_times.append(elapsed); name=f'val_{i:02}'; path=folder/(name+'.npz')
            atomic_publish_bytes(path,deterministic_npz_bytes({k:raw[k] for k in NATIVE}))
            outputs[name]=dict(file=file_ref(path,root=folder),indices=ix.tolist(),role='validation',rows=len(ix),construction_seconds=elapsed)
    finally: target.close()
    value=c.artifact('BATCHES',parents=dict(spec=spec['content_hash']),outputs=outputs,
        mean_input_seconds=float(np.mean(times)),mean_validation_input_seconds=float(np.mean(val_times)),
        preparation_seconds=time.perf_counter()-began,maximum=spec['maximum'],native_validation_only=True)
    c.storage(spec); write_immutable_json(folder/'manifest.json',value)
    print(f'R24_BENCH prepared {len(outputs)} bounded batches, K={spec["maximum"]}',flush=True)
    return value


def read(spec):
    folder=Path(spec['root'])/'batches'; m=load_json(folder/'manifest.json'); c.validate(m,'BATCHES',dict(spec=spec['content_hash']))
    expected={f'train_{i:02}' for i in range(spec['warmup_batches']+spec['measured_batches'])}
    expected|={f'val_{i:02}' for i in range(spec['validation_batches'])}
    expected|={f'parity_{i}' for i in range(len(spec['parity_cases']))}|{'longest_hlt','largest_offline','partial11','tail'}
    if set(m['outputs'])!=expected or m['maximum']!=spec['maximum'] or not m['native_validation_only']:
        raise ValueError('Benchmark batch inventory differs')
    batches={}
    for name,row in m['outputs'].items():
        if row['file']['path']!=name+'.npz' or row['role']!=('validation' if name.startswith('val_') else 'train'):
            raise ValueError('Batch role/path differs')
        a=load_npz_arrays(validate_file_ref(row['file'],root=folder)); raw,emb,targets=unpack(a)
        n=row['rows']; size=raw['features'].shape[-1]
        if (raw['features'].shape!=(n,17,size) or raw['vectors'].shape!=(n,4,size) or raw['mask'].shape!=(n,1,size)
                or raw['mask'].dtype!=np.bool_ or not raw['mask'].any(-1).all()
                or any(not np.isfinite(raw[k]).all() for k in NATIVE)):
            raise ValueError('Invalid native benchmark batch')
        batches[name]=(raw,emb,targets)
    return m,batches
