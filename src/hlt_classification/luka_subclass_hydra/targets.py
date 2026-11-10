"""Frozen, class-conditional OFFLINE prototype targets fitted only on TRAIN."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from .contracts import artifact, validate, target_recipe, COUNTS_PER_CLASS
from hlt_classification.luka_fullstat_subclass.targets import fit


FILES = ('manifest.json', 'targets.npz', 'prototypes.npz')


def parents(spec):
    return dict(spec=spec['content_hash'], recipe=target_recipe()['content_hash'],
        offline=load_json(spec['teacher_refs']['bank']['path'])['content_hash'],
        checkpoint=spec['teacher_refs']['checkpoint']['sha256'],
        prepared=spec['anchor']['evidence']['prepared'])


def build(spec,train,offline):
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or len(train) != spec['counts']['train']
            or train.foundation_sha256 != spec['anchor']['evidence']['prepared']
            or offline.manifest['role'] != 'train' or offline.manifest['coordinate'] != 'OFFLINE'
            or not np.array_equal(offline.identities,train.identities)
            or not np.array_equal(offline.labels,train.labels)):
        raise PermissionError('Subclass fitting requires authenticated ordered TRAIN rows')
    from .inputs import validate_sources
    validate_sources(spec)
    refs = spec['teacher_refs']
    if offline.manifest['content_hash'] != load_json(refs['bank']['path'])['content_hash']:
        raise ValueError('Subclass OFFLINE source changed')
    with np.load(validate_file_ref(refs['targets']),allow_pickle=False) as a:
        mean,scale = a['mean'],a['scale']
    state = torch.load(validate_file_ref(refs['checkpoint']),map_location='cpu',weights_only=True)
    w,b = (state[n].detach().numpy() for n in ('mod.fc.0.weight','mod.fc.0.bias'))
    q,p,d = fit(offline.targets,train.labels,mean,scale,w,b)
    folder = Path(spec['root'])/'setup/subclasses'; folder.mkdir(exist_ok=False)
    atomic_publish_bytes(folder/'targets.npz',deterministic_npz_bytes(dict(
        identities=train.identities,labels=train.labels,targets=q)))
    atomic_publish_bytes(folder/'prototypes.npz',deterministic_npz_bytes(p))
    m = artifact('SUBCLASS_BANK',parents=parents(spec),role='train',coordinate='OFFLINE',rows=len(train),
        class_counts=list(COUNTS_PER_CLASS),recipe=target_recipe(),diagnostics=d,
        targets=file_ref(folder/'targets.npz',root=folder),prototypes=file_ref(folder/'prototypes.npz',root=folder))
    write_immutable_json(folder/'manifest.json',m)
    return Bank(spec,train)


def check_manifest(spec):
    folder = Path(spec['root'])/'setup/subclasses'; m = load_json(folder/'manifest.json')
    validate(m,'SUBCLASS_BANK',parents(spec))
    if (m['role'] != 'train' or m['coordinate'] != 'OFFLINE' or m['rows'] != spec['counts']['train']
            or m['class_counts'] != list(COUNTS_PER_CLASS) or m['recipe'] != target_recipe()
            or m['targets']['path'] != 'targets.npz' or m['prototypes']['path'] != 'prototypes.npz'):
        raise ValueError('Subclass bank semantics differ')
    for key in ('targets','prototypes'):
        validate_file_ref(m[key],root=folder)
    return m


class Bank:
    def __init__(self,spec,train):
        if (train.role != 'train' or train.coordinate_name != 'D000'
                or train.foundation_sha256 != spec['anchor']['evidence']['prepared']):
            raise PermissionError('Subclass labels are TRAIN-only supervision')
        self.manifest = m = check_manifest(spec)
        folder = Path(spec['root'])/'setup/subclasses'
        a = load_npz_arrays(folder/'targets.npz'); p = load_npz_arrays(folder/'prototypes.npz')
        if (set(a) != {'identities','labels','targets'} or a['targets'].dtype != np.float32
                or a['targets'].shape != (len(train),10) or a['labels'].dtype != np.int64
                or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(train),32)
                or len(np.unique(a['identities'],axis=0)) != len(train)
                or not np.isin(a['labels'],np.arange(11)).all()
                or not np.array_equal(a['identities'],train.identities)
                or not np.array_equal(a['labels'],train.labels)
                or not np.isfinite(a['targets']).all() or (a['targets']<0).any()
                or not np.allclose(a['targets'].sum(1),1.,rtol=0,atol=2e-6)
                or np.any(a['targets'][a['labels']!=0,5:] != 0)):
            raise ValueError('Subclass target identity/shape/probability mismatch')
        shapes=dict(centers=(11,10,138),margin_mean=(11,10),margin_scale=(11,10),
            temperatures=(11,),embedding_mean=(128,),embedding_scale=(128,))
        if (set(p)!=set(shapes) or any(p[k].dtype!=np.float64 or p[k].shape!=v
                or not np.isfinite(p[k]).all() for k,v in shapes.items())
                or (p['margin_scale']<.001).any() or (p['embedding_scale']<.001).any()
                or (p['temperatures']<1e-8).any() or (p['temperatures']>1e8).any()
                or np.any(p['centers'][1:,5:]!=0)):
            raise ValueError('Subclass prototype/statistics mismatch')
        if len(m['diagnostics'])!=11 or any(d['label']!=i or d['rows']!=int((train.labels==i).sum())
                or d['subclasses']!=COUNTS_PER_CLASS[i] for i,d in enumerate(m['diagnostics'])):
            raise ValueError('Subclass diagnostics population mismatch')
        self.targets,self.labels,self.identities = a['targets'],a['labels'],a['identities']
        for value in a.values(): value.flags.writeable=False
