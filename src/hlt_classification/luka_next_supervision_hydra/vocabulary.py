"""Frozen TRAIN dictionaries and compact count/scalar-pT frequency targets."""
import hashlib
import numpy as np
from .questions import reservoir_rows

BLOCKS=(np.arange(0,5),np.arange(5,12),np.arange(12,22))


def block_scales(features):
    squares=np.zeros(3,np.float64)
    for start in range(0,len(features),65536):
        x=features[start:start+65536].astype(np.float64)
        for j,indices in enumerate(BLOCKS): squares[j]+=(x[:,indices]**2).sum()
    return np.sqrt(3*np.maximum(squares/len(features),1e-12))


def transform(features,scales):
    x=np.asarray(features,np.float64).copy()
    if np.asarray(scales).shape!=(3,) or not np.isfinite(scales).all() or (np.asarray(scales)<=0).any():
        raise ValueError('Invalid dictionary feature block scales')
    for indices,scale in zip(BLOCKS,scales): x[:,indices]/=scale
    return x.astype(np.float32)


def fit(particles,*,size=128,design_jets=32768):
    if particles.role!='train': raise PermissionError('Dictionary fitting is TRAIN-only')
    rows=reservoir_rows(particles.identities,min(design_jets,len(particles.identities))); selected=[]
    for row in rows:
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(bytes(particles.identities[row])+b'V_design').digest(),'little'))
        own=np.arange(particles.offsets[row],particles.offsets[row+1]); selected.extend(rng.choice(own,min(8,len(own)),replace=False))
    ix=np.asarray(selected,np.int64); scales=block_scales(particles.features)
    sample=transform(particles.features[ix],scales)
    from sklearn.cluster import MiniBatchKMeans
    estimator=MiniBatchKMeans(n_clusters=size,init='k-means++',n_init=3,random_state=2026110513,
        max_iter=100,batch_size=4096,max_no_improvement=20,reassignment_ratio=0.,compute_labels=False)
    estimator.fit(sample)
    centers=np.asarray(estimator.cluster_centers_,np.float32)
    if centers.shape!=(size,22) or not np.isfinite(centers).all(): raise ValueError('Nonfinite frozen dictionary')
    return dict(centers=centers,block_scales=scales,mean=particles.mean,scale=particles.scale,
        reservoir_parents=rows,reservoir_particles=ix),dict(reservoir_particles=len(ix),reservoir_jets=len(rows),
        duplicate_centers=size-len(np.unique(centers,axis=0)),iterations=int(estimator.n_iter_),
        labels_used=False,source=particles.source,feature_blocks=[i.tolist() for i in BLOCKS])


def assign(features,centers,scales,*,chunk=1024):
    centers=np.asarray(centers,np.float64)
    if centers.ndim!=2 or centers.shape[1]!=22 or not np.isfinite(centers).all(): raise ValueError('Invalid frozen centers')
    result=np.empty(len(features),np.int64); distances=np.empty(len(features),np.float64)
    for start in range(0,len(features),chunk):
        x=transform(features[start:start+chunk],scales).astype(np.float64)
        distance=((x[:,None]-centers[None])**2).sum(-1)
        chosen=np.argmin(distance,axis=-1); result[start:start+len(x)]=chosen
        distances[start:start+len(x)]=distance[np.arange(len(x)),chosen]
    return result,distances


def audit(particles,dictionary,codes,distances):
    size=len(dictionary['centers'])
    categories=particles.local[:,6:12].argmax(-1)
    validity=(particles.local[:,18:22]*np.array([1,2,4,8])).sum(-1).astype(np.int64)
    error=np.zeros(3,np.float64)
    for start in range(0,len(codes),65536):
        stop=min(start+65536,len(codes)); residual=transform(particles.features[start:stop],dictionary['block_scales']).astype(np.float64)-dictionary['centers'][codes[start:stop]]
        for j,block in enumerate(BLOCKS): error[j]+=(residual[:,block]**2).sum()
    counts=np.bincount(categories,minlength=6); sums=np.bincount(categories,weights=distances,minlength=6)
    return dict(occupancy=np.bincount(codes,minlength=size).tolist(),mean_squared_distance=float(distances.mean()),
        maximum_squared_distance=float(distances.max()),mean_block_squared_distance=(error/len(codes)).tolist(),
        category_counts=counts.tolist(),category_mean_squared_distance=np.divide(sums,counts,out=np.zeros(6),where=counts>0).tolist(),
        category_occupancy=np.bincount(codes*6+categories,minlength=size*6).reshape(size,6).tolist(),
        validity_occupancy=np.bincount(codes*16+validity,minlength=size*16).reshape(size,16).tolist())


def frequencies(particles,assignments,pt,size=128):
    if particles.role!='train': raise PermissionError('Frequency targets are TRAIN-only')
    codes=np.asarray(assignments); pt=np.asarray(pt,np.float64)
    if (codes.shape!=(len(particles.local),) or codes.dtype.kind not in 'iu' or codes.min()<0 or codes.max()>=size
            or pt.shape!=codes.shape or not np.isfinite(pt).all() or (pt<0).any()): raise ValueError('Invalid dictionary assignments/weights')
    result=np.empty((len(particles.labels),2,size),np.float32)
    for row,(a,b) in enumerate(zip(particles.offsets[:-1],particles.offsets[1:])):
        count=np.bincount(codes[a:b],minlength=size); momentum=np.bincount(codes[a:b],weights=pt[a:b],minlength=size)
        if not count.sum()>0 or not momentum.sum()>0: raise ValueError('Invalid source frequency denominator')
        result[row,0]=count/count.sum(); result[row,1]=momentum/momentum.sum()
    if not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.sum(-1),1.,atol=1e-6):
        raise ValueError('Frequency normalization differs')
    return result


class FrequencyCandidates:
    def __init__(self,arrays,dictionary_hash,*,source,count_only=False):
        self.arrays=arrays; self.count_only=count_only; self.source=source
        n=len(arrays['identities']); targets=arrays['targets']
        if (set(arrays)!={'identities','labels','targets','prior'} or targets.shape!=(n,2,128)
                or targets.dtype!=np.float32 or not np.isfinite(targets).all() or (targets<0).any()
                or arrays['prior'].shape!=(2,128) or arrays['prior'].dtype!=np.float32
                or not np.array_equal(arrays['prior'],targets.mean(0,dtype=np.float64).astype(np.float32))):
            raise ValueError('Invalid frozen frequency targets')
        if not np.allclose(targets.sum(-1),1.,atol=1e-6): raise ValueError('Frozen frequencies do not sum to one')
        self.capacity=dict(rows=n,class_counts=np.bincount(arrays['labels'],minlength=11).tolist(),
            source=source,dictionary=dictionary_hash,targets=1 if count_only else 2,codes=128)

    def batch(self,anchors,epoch,cfg,mine=None):
        target=self.arrays['targets'][anchors,:1 if self.count_only else 2]
        return dict(targets=target,prior=np.broadcast_to(self.arrays['prior'][:len(target[0])],target.shape).copy()),[]
