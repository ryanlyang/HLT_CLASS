"""TRAIN-only out-of-fold membership, predictions and bounded difficulty weights."""
import hashlib
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import FOLDS, artifact, validate, recipe


def identity_hash(ids):
    return hashlib.sha256(np.ascontiguousarray(ids,dtype=np.uint8).tobytes()).hexdigest()


def assignment(identities,labels):
    n=len(labels)
    if identities.dtype != np.uint8 or identities.shape != (n,32) or labels.dtype != np.int64 or len(np.unique(identities,axis=0)) != n:
        raise ValueError('Invalid unique TRAIN identities/labels')
    if set(labels) != set(range(11)): raise ValueError('All eleven TRAIN classes required')
    fold=np.empty(n,np.int64); salt=str(recipe()['fold_seed']).encode()
    for k in range(11):
        ix=np.flatnonzero(labels==k)
        if len(ix)<3: raise ValueError('Each class needs at least three TRAIN jets')
        order=sorted(ix,key=lambda i:hashlib.sha256(salt+bytes(identities[i])).digest())
        fold[order]=np.arange(len(order))%3
    return fold


def build(spec,train):
    if train.role!='train' or train.coordinate_name!='OFFLINE': raise PermissionError('Folds require OFFLINE TRAIN')
    a=dict(identities=train.identities,labels=train.labels,fold=assignment(train.identities,train.labels))
    folder=Path(spec['root'])/'setup'
    atomic_publish_bytes(folder/'folds.npz',deterministic_npz_bytes(a))
    value=artifact('FOLDS',parents=dict(spec=spec['content_hash'],prepared=train.foundation_sha256),
        role='train',rows=len(train),counts=[int((a['fold']==i).sum()) for i in range(3)],
        payload=file_ref(folder/'folds.npz',root=folder))
    write_immutable_json(folder/'folds.json',value)
    return value


def load(spec,train=None):
    folder=Path(spec['root'])/'setup'; value=load_json(folder/'folds.json')
    validate(value,'FOLDS',dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared']))
    if value['role']!='train' or value['payload']['path']!='folds.npz' or value['rows']!=spec['counts']['train']:
        raise ValueError('Fold role/population differs')
    validate_file_ref(value['payload'],root=folder); a=load_npz_arrays(folder/'folds.npz')
    if set(a)!={'identities','labels','fold'} or len(a['labels'])!=value['rows'] or not np.array_equal(a['fold'],assignment(a['identities'],a['labels'])):
        raise ValueError('Fold assignment differs')
    if value['counts']!=[int((a['fold']==i).sum()) for i in range(3)]: raise ValueError('Fold counts differ')
    if train is not None and (train.role!='train' or any(not np.array_equal(a[k],getattr(train,k)) for k in ('identities','labels'))):
        raise PermissionError('Fold TRAIN join differs')
    return value,a


def split(spec,train,task):
    if task not in FOLDS: raise ValueError('Unknown fold')
    _,a=load(spec,train); held=np.flatnonzero(a['fold']==int(task[1])); kept=np.flatnonzero(a['fold']!=int(task[1]))
    return Subset(train,kept),Subset(train,held)


def export(spec,task,native,train,device):
    if task not in FOLDS: raise ValueError('Unknown fold')
    manifest,a=load(spec,train); kept,held=split(spec,train,task)
    folder=Path(spec['root'])/task
    from .training import validate_fit
    report,_=validate_fit(folder/'fit',spec,task)
    if report['trained_identity_hash']!=identity_hash(kept.identities): raise ValueError('OOF checkpoint training set differs')
    checkpoint=file_ref(folder/'fit/selected.pt')
    probabilities=predict(native,held,device=device,batch_size=128)
    arrays=dict(identities=held.identities,labels=held.labels,probabilities=probabilities)
    validate_predictions(arrays,len(held))
    atomic_publish_bytes(folder/'oof.npz',deterministic_npz_bytes(arrays))
    value=artifact('OOF',parents=dict(spec=spec['content_hash'],folds=manifest['content_hash'],checkpoint=checkpoint['sha256']),
        task=task,role='train',selection='last_epoch30',checkpoint=checkpoint,
        trained_identity_hash=identity_hash(kept.identities),held_identity_hash=identity_hash(held.identities),
        trained_rows=len(kept),held_rows=len(held),payload=file_ref(folder/'oof.npz',root=folder))
    write_immutable_json(folder/'oof.json',value)
    return value


def read_oof(spec,task):
    fm,a=load(spec); folder=Path(spec['root'])/task; m=load_json(folder/'oof.json')
    cp=file_ref(folder/'fit/selected.pt')
    validate(m,'OOF',dict(spec=spec['content_hash'],folds=fm['content_hash'],checkpoint=cp['sha256']))
    selected=a['fold']==int(task[1]); kept=a['identities'][~selected]; held=a['identities'][selected]
    if (m['task']!=task or m['role']!='train' or m['selection']!='last_epoch30' or m['checkpoint']!=cp
        or m['trained_identity_hash']!=identity_hash(kept) or m['held_identity_hash']!=identity_hash(held)
        or m['trained_rows']!=len(kept) or m['held_rows']!=len(held) or m['payload']['path']!='oof.npz'):
        raise ValueError('OOF training exclusion/checkpoint lineage differs')
    validate_file_ref(m['payload'],root=folder); p=load_npz_arrays(folder/'oof.npz'); validate_predictions(p,len(held))
    if not np.array_equal(p['identities'],held) or not np.array_equal(p['labels'],a['labels'][selected]):
        raise ValueError('OOF held-out join differs')
    return m,p


def difficulty(labels,probabilities):
    if probabilities.shape!=(len(labels),11) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1.,atol=2e-6,rtol=0):
        raise ValueError('Invalid OOF probabilities')
    raw=1+2*(1-probabilities[np.arange(len(labels)),labels].astype(np.float64))
    weights=raw.copy()
    for k in range(11):
        ix=labels==k
        if not ix.any(): raise ValueError('Missing TRAIN class')
        weights[ix]/=raw[ix].mean()
    return weights.astype(np.float32)


def assemble(spec,train=None):
    fm,a=load(spec,train); p=np.empty((len(a['labels']),11),np.float32); parents=dict(folds=fm['content_hash'])
    for task in FOLDS:
        from .worker import receipt
        from .training import validate_fit
        receipt(spec,task); validate_fit(Path(spec['root'])/task/'fit',spec,task)
        m,row=read_oof(spec,task); p[a['fold']==int(task[1])]=row['probabilities']; parents[task]=m['content_hash']
    weights=difficulty(a['labels'],p)
    m=artifact('DIFFICULTY',parents=dict(spec=spec['content_hash'],**parents),role='train',rows=len(a['labels']),
        identities=identity_hash(a['identities']),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        formula=recipe()['difficulty'],minimum=float(weights.min()),maximum=float(weights.max()))
    return weights,m
