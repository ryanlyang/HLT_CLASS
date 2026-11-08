"""Identity-keyed physical corruptions and visible-only native preprocessing."""
import hashlib
import numpy as np
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.literature_context.transform import build_inputs
from .contracts import TEACHERS
from .structure import canonical, checked, coordinates, mass, wrap, summary, MAP_EDGES, PAIR_EDGES, BALANCE_EDGES


def rng_for(identity, task, epoch, draw):
    if task not in TEACHERS or epoch<0 or draw<0:
        raise ValueError('Unregistered corruption key')
    key=bytes(identity)+f'/{task}/{epoch}/{draw}/physical_v1'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16],'little'))


def corrupt(values, identity, task, epoch, draw=0):
    a=canonical(values); n=len(a); rng=rng_for(identity,task,epoch,draw)
    pt,eta,phi,_=coordinates(a[:,:4]); remove=np.zeros(n,bool)
    if n<2: return a.copy(),a,remove
    q={'T01':(.3,.5),'T02':(.15,.3),'T03':(.4,), 'T04':(.3,.5),
       'T05':(.3,.5),'T06':(.2,),'T07':(.2,),'T08':(.2,)}[task]
    fraction=q[int(rng.integers(len(q)))]; m=max(1,min(n-1,int(np.floor(fraction*n+.5))))
    if task in ('T02','T04'):
        centre=int(np.argmax(pt)) if task=='T02' else int(rng.integers(n))
        order=np.argsort(np.hypot(eta-eta[centre],wrap(phi-phi[centre])),kind='stable')
        # Centre is always included even for coincident particles.
        order=np.r_[centre,order[order!=centre]]
        remove[order[:m]]=True
    elif task=='T03':
        pool=np.argsort(pt,kind='stable')[:max(1,int(np.ceil(.7*n)))]
        remove[rng.choice(pool,min(m,len(pool)),replace=False)]=True
    else: remove[rng.choice(n,m,replace=False)]=True
    # Removing all positive-pT particles cannot define a visible frame. Keep
    # the highest-pT member and exchange a zero-pT member when one is visible.
    # This deterministic exception is recorded by the physical_v1 recipe.
    if pt[~remove].sum()<=0:
        top=int(np.argmax(pt)); replacement=np.flatnonzero(~remove)[0]
        remove[top]=False; remove[replacement]=True
    visible=a[~remove].copy()
    if task=='T08':
        p,e,f,_=coordinates(visible[:,:4]); noise=rng.normal(0,.1,len(visible))
        visible[:,:4]=p4_from_coordinates(p*np.exp(noise),e+rng.normal(0,.01,len(p)),
            wrap(f+rng.normal(0,.01,len(p))),mass(visible[:,:4]))
    checked(visible)
    return visible,a,remove


def native(values):
    a=checked(values)
    p=Particles(a[:,:4],a[:,4],a[:,5].astype(int),a[:,6:10],a[:,10:].astype(bool),tuple(map(str,range(len(a)))))
    return build_inputs(p,capacity=max(512,len(a)))


def batch(values):
    built=[native(a) for a in values]; width=max(16,max(len(a) for a in values)); n=len(values)
    result=dict(features=np.zeros((n,17,width),np.float32),vectors=np.zeros((n,4,width),np.float32),
                mask=np.zeros((n,1,width),bool))
    for i,(a,b) in enumerate(zip(values,built)):
        result['features'][i,:,:len(a)]=b.features.T
        result['vectors'][i,:,:len(a)]=b.vectors.T
        result['mask'][i,0,:len(a)]=True
    return result


def frame(visible):
    visible=checked(visible); p=visible[:,:4].sum(0)
    _,eta,phi,_=coordinates(p)
    return np.array([eta,phi,coordinates(visible[:,:4])[0].sum()],np.float64)


def set_target(target,visible):
    ref=frame(visible); a=np.asarray(target,np.float64)
    pt,eta,phi,_=coordinates(a[:,:4])
    k=np.column_stack((np.log(np.maximum(pt/ref[2],1e-8)),np.log(np.maximum(a[:,3]/ref[2],1e-8)),
        eta-ref[0],np.sin(phi-ref[1]),np.cos(phi-ref[1]),np.log1p(mass(a[:,:4])/ref[2])))
    track=np.column_stack((np.arcsinh(a[:,6:8]),np.log(np.maximum(a[:,8:10],1e-8))))
    flags=a[:,10:].astype(bool); track=np.where(flags,track,0.)
    return dict(kinematic=k,track=track,validity=flags,category=a[:,5].astype(np.int64),
        charge=(a[:,4]+1).astype(np.int64),pt=pt,frame=ref)


def patch_map(clean,removed,visible):
    a=clean[removed]; ref=frame(visible); pt,eta,phi,_=coordinates(a[:,:4]); cat=a[:,5].astype(int)
    e=np.zeros((6,10,10)); count=e.copy()
    for category in range(6):
        choose=cat==category
        count[category]=np.histogram2d(eta[choose]-ref[0],wrap(phi[choose]-ref[1]),bins=(MAP_EDGES,MAP_EDGES))[0]
        e[category]=np.histogram2d(eta[choose]-ref[0],wrap(phi[choose]-ref[1]),bins=(MAP_EDGES,MAP_EDGES),weights=pt[choose]/ref[2])[0]
    return summary(np.r_[np.log1p(e.ravel()),np.log1p(count.ravel())],blocks=[0]*600+[1]*600)


def missing_pairs(clean,removed):
    a=checked(clean); pt,_,phi,y=coordinates(a[:,:4]); z=pt/pt.sum()
    i,j=np.triu_indices(len(a),1); use=removed[i]|removed[j]; i,j=i[use],j[use]
    angle=np.hypot(y[i]-y[j],wrap(phi[i]-phi[j])); balance=np.minimum(pt[i],pt[j])/np.maximum(pt[i]+pt[j],1e-12)
    charge=(a[:,4]!=0).astype(int); channel=2-charge[i]-charge[j]
    h=np.zeros((3,13,7))
    for c in range(3):
        ok=channel==c; h[c]=np.histogram2d(angle[ok],balance[ok],bins=(PAIR_EDGES,BALANCE_EDGES),weights=(z[i]*z[j])[ok])[0]
    total=h.sum(); values=h.ravel()/total if total>0 else h.ravel()
    return summary(np.r_[values,total],[total>0]*h.size+[True],blocks=[0]*h.size+[1])
