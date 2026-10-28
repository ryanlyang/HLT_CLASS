"""Label-blind, matching-free physical neighborhood descriptors."""
import numpy as np
from .contracts import FAMILIES


def geometry(p):
    v=np.asarray(p.p4,dtype=np.float64); c=np.asarray(p.category)
    if (v.ndim!=2 or v.shape[1]!=4 or not 1<=len(v)<=512 or c.shape!=(len(v),)
            or c.dtype.kind not in 'iu' or np.any(c<0) or np.any(c>5) or not np.isfinite(v).all()):
        raise ValueError('Invalid real particle geometry/category/support')
    pt=np.hypot(v[:,0],v[:,1])
    if np.any(pt<=0): raise ValueError('Nonpositive real particle pt')
    return pt,np.arcsinh(v[:,2]/pt),np.arctan2(v[:,1],v[:,0])


def descriptors(query, context, *, smooth, chunk=32):
    if type(smooth)!=bool or type(chunk)!=int or not 1<=chunk<=512: raise ValueError('Invalid kernel/chunk')
    qpt,qeta,qphi=geometry(query); pt,eta,phi=geometry(context)
    # Ordering only summation operands: never reorder query/model inputs.
    order=np.lexsort((context.category,*[context.p4[:,i] for i in (3,2,1,0)]))
    pt,eta,phi,c=pt[order],eta[order],phi[order],np.asarray(context.category)[order]
    denominator=np.sort(qpt).sum(dtype=np.float64)
    if not np.isfinite(denominator) or denominator<=0: raise ValueError('Invalid HLT pt sum')
    output=np.empty((len(query),36),np.float32)
    for start in range(0,len(query),chunk):
        stop=min(start+chunk,len(query))
        dphi=(phi[None,:]-qphi[start:stop,None]+np.pi)%(2*np.pi)-np.pi
        d2=(eta[None,:]-qeta[start:stop,None])**2+dphi**2
        if smooth:
            kernels=[np.exp(-.5*d2/s**2) for s in (.02,.05,.10)]
        else:
            kernels=[((d2>=lo**2)&(d2<hi**2)).astype(np.float64) for lo,hi in ((0.,.05),(.05,.10),(.10,.20))]
        counts=np.empty((stop-start,3,6),np.float64); momentum=np.empty_like(counts)
        for s,k in enumerate(kernels):
            for category in range(6):
                selected=k[:,c==category]
                counts[:,s,category]=selected.sum(-1,dtype=np.float64)
                momentum[:,s,category]=(selected*pt[c==category][None,:]).sum(-1,dtype=np.float64)/denominator
        output[start:stop]=np.concatenate((np.log1p(counts).reshape(stop-start,18),
                                          np.log1p(100*momentum).reshape(stop-start,18)),axis=1)
    if not np.isfinite(output).all(): raise ValueError('Nonfinite neighborhood descriptor')
    return output


def all_descriptors(hlt, offline):
    return dict(ann=descriptors(hlt,offline,smooth=False),smooth=descriptors(hlt,offline,smooth=True),
                hlt_ann=descriptors(hlt,hlt,smooth=False))


class Moments:
    """Stable mixture-of-jets moments: every jet has mass one."""
    def __init__(self):
        self.n=0; self.mean=np.zeros(36,np.float64); self.m2=np.zeros(36,np.float64)

    def add(self, values):
        x=np.asarray(values,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=36 or not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid moments input')
        mean=x.mean(0); variance=((x-mean)**2).mean(0); self.n+=1; delta=mean-self.mean
        self.mean+=delta/self.n
        self.m2+=variance+delta**2*(self.n-1)/self.n

    def result(self):
        if not self.n: raise ValueError('Empty TRAIN moments')
        variance=self.m2/self.n
        if not np.isfinite(variance).all() or np.any(variance<0): raise ValueError('Invalid TRAIN variance')
        return self.mean.copy(),np.maximum(np.sqrt(variance),.1)
