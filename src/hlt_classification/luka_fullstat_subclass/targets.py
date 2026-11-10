"""Frozen, class-conditional OFFLINE prototype targets fitted only on TRAIN."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays,
    atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json)
from .contracts import artifact, validate, subclass_recipe, COUNTS_PER_CLASS, CLUSTER_SEED


FILES = ('manifest.json', 'targets.npz', 'prototypes.npz')


def parents(spec):
    b = spec['reuse']['banks']['OFFLINE']
    return dict(recipe=subclass_recipe()['content_hash'], offline=b['manifest_hash'],
                checkpoint=b['refs']['checkpoint']['sha256'], prepared=spec['anchor']['evidence']['prepared'])


def distances(features, centers):
    """Bounded direct FP64 squared distances avoid cancellation at prototypes."""
    result = np.empty((len(features), len(centers)), np.float64)
    for start in range(0, len(features), 4096):
        x = np.asarray(features[start:start+4096], dtype=np.float64)
        result[start:start+len(x)] = ((x[:, None, :]-centers[None, :, :])**2).sum(2)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite subclass distances')
    return result


def probabilities(d, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    z = -(d-d.min(1, keepdims=True))/temperature
    e = np.exp(z)
    return e/e.sum(1, keepdims=True)


def calibrate(d):
    """Ties can make .8 unattainable; retain finite targets and report that fact."""
    low, high = subclass_recipe()['temperature_bounds']
    goal = subclass_recipe()['temperature_goal']
    measure = lambda t: float(np.median(probabilities(d, t).max(1)))
    if measure(low) <= goal:
        return low, measure(low)
    if measure(high) >= goal:
        return high, measure(high)
    for _ in range(subclass_recipe()['temperature_bisections']):
        mid = math.sqrt(low*high)
        if measure(mid) > goal:
            low = mid
        else:
            high = mid
    t = math.sqrt(low*high)
    return t, measure(t)


def fit(embedding, labels, mean, scale, weight, bias):
    """Pure array kernel. Callers must authenticate TRAIN role and ordered joins."""
    n = len(labels)
    if (embedding.dtype != np.float32 or embedding.shape != (n,128)
            or labels.dtype != np.int64 or labels.shape != (n,)
            or mean.shape != (128,) or scale.shape != (128,)
            or weight.shape != (11,128) or bias.shape != (11,)
            or np.any(scale < .001) or not np.isin(labels, np.arange(11)).all()
            or any(not np.isfinite(a).all() for a in (embedding,mean,scale,weight,bias))):
        raise ValueError('Invalid frozen OFFLINE arrays')
    out = np.zeros((n,10), np.float32)
    centers = np.zeros((11,10,138), np.float64)
    means = np.zeros((11,10), np.float64); scales = np.ones((11,10), np.float64)
    temperatures = np.ones(11, np.float64); diagnostics = []
    config = subclass_recipe()
    # Bound BLAS/OpenMP independently of caller environment; no sampling RNG changes.
    with threadpool_limits(limits=1):
        for label,k in enumerate(COUNTS_PER_CLASS):
            ix = np.flatnonzero(labels == label)
            if len(ix) < k:
                raise ValueError('Insufficient TRAIN rows to define registered prototype count')
            tick = time.monotonic()
            z = embedding[ix]
            other = np.delete(np.arange(11), label)
            margins = np.empty((len(ix),10), np.float64)
            for start in range(0,len(ix),32768):
                raw = z[start:start+32768].astype(np.float64)*scale+mean
                logits = raw @ weight.T.astype(np.float64)+bias
                margins[start:start+len(raw)] = logits[:,other]-logits[:,label,None]
            means[label] = margins.mean(0, dtype=np.float64)
            scales[label] = np.maximum(margins.std(0, dtype=np.float64), .001)
            x = np.concatenate((z*np.sqrt(.75/128),
                ((margins-means[label])/scales[label]*np.sqrt(.25/10)).astype(np.float32)),axis=1)
            del margins,z
            km = MiniBatchKMeans(n_clusters=k, random_state=CLUSTER_SEED+label,
                batch_size=config['batch_size'], n_init=config['n_init'], max_iter=config['max_iter'],
                max_no_improvement=config['max_no_improvement'], reassignment_ratio=0.,
                tol=0., compute_labels=False).fit(x)
            centers[label,:k] = km.cluster_centers_
            d = distances(x,centers[label,:k]); del x
            temperature,achieved = calibrate(d); temperatures[label] = temperature
            q = probabilities(d,temperature)
            out[ix,:k] = q.astype(np.float32)
            diagnostic = dict(label=label,rows=len(ix),subclasses=k,temperature=float(temperature),
                median_max_membership=achieved,softness_target_reached=abs(achieved-.8)<1e-5,
                hard_occupancy=np.bincount(q.argmax(1),minlength=k).tolist(),
                soft_occupancy=q.sum(0).tolist(),entropy=float(-(q*np.log(np.maximum(q,1e-300))).sum(1).mean()),
                mean_nearest_squared_distance=float(d.min(1).mean()),
                distinct_centers=int(len(np.unique(centers[label,:k],axis=0))),
                kmeans_steps=int(km.n_steps_))
            diagnostics.append(diagnostic)
            print(f'SUBCLASS targets class={label} rows={len(ix)} k={k} '
                  f'median_max={achieved:.6f} seconds={time.monotonic()-tick:.1f}',flush=True)
    return out,dict(centers=centers,margin_mean=means,margin_scale=scales,
        temperatures=temperatures,embedding_mean=mean,embedding_scale=scale),diagnostics


def build(spec,train,offline):
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or not np.array_equal(offline.identities,train.identities)
            or not np.array_equal(offline.labels,train.labels)):
        raise PermissionError('Subclass fitting requires authenticated ordered TRAIN rows')
    refs = spec['reuse']['banks']['OFFLINE']
    if offline.manifest['content_hash'] != refs['manifest_hash']:
        raise ValueError('Subclass OFFLINE source changed')
    with np.load(validate_file_ref(refs['payload']),allow_pickle=False) as a:
        mean,scale = a['mean'],a['scale']
    state = torch.load(validate_file_ref(refs['refs']['checkpoint']),map_location='cpu',weights_only=True)
    w,b = (state[n].detach().numpy() for n in ('mod.fc.0.weight','mod.fc.0.bias'))
    q,p,d = fit(offline.targets,train.labels,mean,scale,w,b)
    folder = Path(spec['root'])/'subclasses'; folder.mkdir(exist_ok=False)
    atomic_publish_bytes(folder/'targets.npz',deterministic_npz_bytes(dict(
        identities=train.identities,labels=train.labels,targets=q)))
    atomic_publish_bytes(folder/'prototypes.npz',deterministic_npz_bytes(p))
    m = artifact('SUBCLASS_BANK',parents=parents(spec),role='train',coordinate='OFFLINE',rows=len(train),
        class_counts=list(COUNTS_PER_CLASS),recipe=subclass_recipe(),diagnostics=d,
        targets=file_ref(folder/'targets.npz',root=folder),prototypes=file_ref(folder/'prototypes.npz',root=folder))
    write_immutable_json(folder/'manifest.json',m)
    return Bank(spec,train)


def check_manifest(spec):
    folder = Path(spec['root'])/'subclasses'; m = load_json(folder/'manifest.json')
    validate(m,'SUBCLASS_BANK',parents(spec))
    if (m['role'] != 'train' or m['coordinate'] != 'OFFLINE' or m['rows'] != spec['counts']['train']
            or m['class_counts'] != list(COUNTS_PER_CLASS) or m['recipe'] != subclass_recipe()
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
        folder = Path(spec['root'])/'subclasses'
        a = load_npz_arrays(folder/'targets.npz'); p = load_npz_arrays(folder/'prototypes.npz')
        if (set(a) != {'identities','labels','targets'} or a['targets'].dtype != np.float32
                or a['targets'].shape != (len(train),10) or a['labels'].dtype != np.int64
                or a['identities'].dtype != np.uint8 or a['identities'].shape != (len(train),32)
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


def prepare(spec,train,offline):
    """Fresh fits build once; recovery copies verified frozen targets exactly."""
    if spec.get('resume') is None:
        return build(spec,train,offline)
    old = load_json(validate_file_ref(spec['resume']['spec']))
    bank = Bank(old,train)
    dest = Path(spec['root'])/'subclasses'; dest.mkdir(exist_ok=False)
    for name in FILES:
        atomic_publish_bytes(dest/name,(Path(old['root'])/'subclasses'/name).read_bytes())
    return bank
