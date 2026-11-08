"""Complete-particle, permutation-invariant measured jet summaries.

No particle ancestry, class labels, source indices or HLT/OFFLINE matching.
Each family returns values, applicability, binary-channel mask and block IDs.
"""
from dataclasses import dataclass
import heapq
import numpy as np

RADII = np.array([.005,.01,.02,.04,.08,.12,.20,.30,.45,.65,1.,2.,np.inf])
MAP_EDGES = np.array([-np.inf,-.8,-.4,-.2,-.1,0.,.1,.2,.4,.8,np.inf])
PAIR_EDGES = np.r_[0., RADII]
BALANCE_EDGES = np.array([0.,.025,.05,.10,.20,.30,.40,.50])


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def mass(p):
    p = np.asarray(p, np.float64)
    m2 = p[...,3]**2 - (p[...,:3]**2).sum(-1)
    tolerance = 1e-6*np.maximum.reduce([p[...,3]**2, (p[...,:3]**2).sum(-1), np.ones_like(m2)])
    if np.any(m2 < -tolerance):
        raise ValueError('Spacelike measured four-vector exceeds registered tolerance')
    return np.sqrt(np.maximum(m2, 0.))


def coordinates(p):
    p = np.asarray(p, np.float64)
    pt = np.hypot(p[...,0], p[...,1])
    phi = np.arctan2(p[...,1], p[...,0])
    # A fixed floor defines the finite zero-pT/beam-collinear convention.
    eta = np.arcsinh(p[...,2]/np.maximum(pt, 1e-12))
    y = .5*np.log(np.maximum(p[...,3]+p[...,2],1e-12)/np.maximum(p[...,3]-p[...,2],1e-12))
    return pt, eta, phi, y


def checked(values):
    a = np.asarray(values, np.float64)
    if (a.ndim != 2 or a.shape[1] != 14 or not len(a) or not np.isfinite(a).all()
            or (a[:,3] < 0).any() or not np.isin(a[:,4], [-1,0,1]).all()
            or not np.isin(a[:,5], range(6)).all() or not np.isin(a[:,10:], [0,1]).all()
            or (a[:,8:10] < 0).any()):
        raise ValueError('Invalid physical measurements')
    mass(a[:,:4])
    if coordinates(a[:,:4])[0].sum() <= 0:
        raise ValueError('Nonpositive scalar pT sum')
    return a


def canonical(values):
    a = checked(values)
    return a[np.lexsort(tuple(a[:,i] for i in range(13,-1,-1)))]


@dataclass
class Summary:
    values: np.ndarray
    valid: np.ndarray
    binary: np.ndarray
    blocks: np.ndarray


def summary(values, valid=None, binary=None, blocks=None):
    v = np.asarray(values, np.float64)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('Nonfinite structure target')
    return Summary(v, np.ones(len(v),bool) if valid is None else np.asarray(valid,bool),
        np.zeros(len(v),bool) if binary is None else np.asarray(binary,bool),
        np.zeros(len(v),np.int64) if blocks is None else np.asarray(blocks,np.int64))


def correlations(a):
    a = canonical(a); pt, _, phi, y = coordinates(a[:,:4]); z = pt/pt.sum()
    distance = np.hypot(y[:,None]-y, wrap(phi[:,None]-phi))
    pairs = np.triu_indices(len(a),1)
    c2, c3 = [], []
    for beta in (.5,1.,2.):
        aij = np.sqrt(z[:,None]*z)*distance**beta
        np.fill_diagonal(aij, 0.)
        c2.append(float((z[:,None]*z*distance**beta)[pairs].sum()))
        c3.append(float(np.sum((aij@aij)*aij)/6.))
    p2, p3 = [], []
    base = np.sqrt(z[:,None]*z)*distance
    np.fill_diagonal(base,0.)
    for r in RADII:
        gate = distance <= r
        v = base*gate
        p2.append(float((z[:,None]*z*distance*gate)[pairs].sum()))
        p3.append(float(np.sum((v@v)*v)/6.))
    common = [np.log1p(len(a)), float(z@z)]
    result = {}
    for name, totals, profile in (('C2',c2,p2),('C3',c3,p3)):
        den = totals[1]
        result[name] = summary([*np.log1p(np.asarray(totals)/1e-4),
            *(np.asarray(profile)/den if den > 0 else np.zeros(len(profile))), *common],
            [True]*3+[den>0]*len(profile)+[True]*2,
            blocks=[0]*3+[1]*len(profile)+[2]*2)
    return result


@dataclass
class Branch:
    p4: np.ndarray
    key: tuple
    left: object = None
    right: object = None


def cluster(a):
    """Exclusive C/A with E-scheme, no beam removal; deterministic physical ties."""
    a = canonical(a)
    active = {i: Branch(v[:4].copy(), (tuple(v),)) for i,v in enumerate(a) if np.hypot(*v[:2]) > 0}
    keys = list(active)
    _,_,phi,rap = coordinates(np.stack([active[i].p4 for i in keys]))
    positions={i:(phi[k],rap[k]) for k,i in enumerate(keys)}
    distances=(rap[:,None]-rap)**2+wrap(phi[:,None]-phi)**2
    heap=[]
    for k,i in enumerate(keys):
        for l in range(k+1,len(keys)):
            j=keys[l]; x,y=active[i],active[j]
            heap.append((float(distances[k,l]),min(x.key,y.key),max(x.key,y.key),i,j))
    heapq.heapify(heap)
    serial = len(a)
    while len(active)>1:
        _,_,_,i,j = heapq.heappop(heap)
        if i not in active or j not in active: continue
        x,y = active.pop(i),active.pop(j)
        del positions[i],positions[j]
        b = Branch(x.p4+y.p4, tuple(sorted(x.key+y.key)), x,y)
        _,_,ph,rap=coordinates(b.p4); keys=list(active)
        if keys:
            previous=np.array([positions[k] for k in keys])
            distances=(previous[:,1]-rap)**2+wrap(previous[:,0]-ph)**2
            for k,distance in zip(keys,distances):
                key=active[k].key
                heapq.heappush(heap,(float(distance),min(key,b.key),max(key,b.key),k,serial))
        active[serial] = b
        positions[serial]=(ph,rap)
        serial += 1
    return next(iter(active.values()))


def branches(a):
    a = checked(a); total = coordinates(a[:,:4])[0].sum(); node = cluster(a)
    rootmass = float(mass(node.p4)); splits=[]; softdrop=None; zero=0
    while node.left is not None:
        x,y = node.left,node.right
        pt,_,phi,rap = coordinates(np.stack([x.p4,y.p4]))
        if pt[0] < pt[1] or (pt[0] == pt[1] and x.key > y.key):
            x,y = y,x; pt=pt[::-1]
        z = float(pt[1]/max(pt.sum(),1e-12)); theta=float(np.hypot(rap[0]-rap[1],wrap(phi[0]-phi[1])))
        kt = pt[1]*theta/total; pm=float(mass(node.p4))
        md=float(max(mass(x.p4),mass(y.p4))/pm) if pm else 0.
        splits.append((z,theta,kt,md,pm>0,float(pt[1]/total)))
        if z>.1 and softdrop is None: softdrop=(z,theta,pm/total)
        node=x
    hard=[s for s in splits if s[0]>.05]; values=[]; valid=[]; binary=[]; blocks=[]
    def add(v,ok=True,bit=False,block=0):
        values.append(v); valid.append(ok); binary.append(bit); blocks.append(block)
    for index in range(2):
        present=index<len(hard); add(float(present),bit=True)
        s=hard[index] if present else (0.,0.,0.,0.,False,0.)
        for v,ok in zip((s[0],np.log1p(s[1]/1e-3),np.log1p(s[2]/1e-4),s[3]),(True,True,True,s[4])):
            add(v,present and ok)
    add(np.log1p(len(hard))); add(np.log1p(rootmass/total/1e-4))
    add(float(softdrop is not None),bit=True,block=1)
    sd=softdrop if softdrop else (0.,0.,float(mass(node.p4))/total)
    for i,v in enumerate((sd[0],np.log1p(sd[1]/1e-3),np.log1p(sd[2]/1e-4))):
        add(v,softdrop is not None or i==2,block=1)
    xe=np.r_[-np.inf,np.linspace(-2,8,9),np.inf]; ye=np.r_[-np.inf,np.linspace(-12,0,9),np.inf]
    count=np.zeros((10,10)); weighted=count.copy()
    for z,t,k,_,_,w in splits:
        if k<=0 or t<=0: zero+=1; continue
        i=np.searchsorted(xe,np.log(1/t),side='right')-1; j=np.searchsorted(ye,np.log(k),side='right')-1
        count[i,j]+=1; weighted[i,j]+=w
    for block,field in ((2,count),(3,weighted)):
        for v in np.log1p(field).ravel(): add(v,block=block)
    add(np.log1p(zero),block=2)
    return summary(values,valid,binary,blocks)


def radial_identity_tracking(a):
    a=canonical(a); pt,_,phi,y=coordinates(a[:,:4]); z=pt/pt.sum()
    _,_,ap,ay=coordinates(a[:,:4].sum(0)); radius=np.hypot(y-ay,wrap(phi-ap))
    values=[]; valid=[]; bits=[]; blocks=[]
    def extend(v,block,ok=None,binary=None):
        values.extend(v); blocks.extend([block]*len(v)); valid.extend([True]*len(v) if ok is None else ok)
        bits.extend([False]*len(v) if binary is None else binary)
    extend(np.histogram(radius,[0,.02,.05,.10,.20,.35,.50,.80,np.inf],weights=z)[0],0)
    cat=a[:,5].astype(int)
    extend(np.bincount(cat,minlength=6)/len(a),1); extend(np.bincount(cat,weights=z,minlength=6),1)
    extend([z[a[:,4]!=0].sum(),np.log1p(len(a)),1/(z@z)],1)
    leading=np.sort(z)[::-1][:4]; extend(np.pad(leading,(0,4-len(leading))),1,
        ok=[i<len(leading) for i in range(4)])
    extend([float(i<len(leading)) for i in range(4)],1,binary=[True]*4)
    for j in range(2):
        ok=a[:,10+j].astype(bool)&a[:,12+j].astype(bool)&(a[:,8+j]>0)
        sig=np.divide(a[:,6+j],a[:,8+j],out=np.zeros(len(a)),where=ok)
        for cut in (1,2,4,8):
            hit=ok&(np.abs(sig)>cut)
            extend([hit.sum()/max(ok.sum(),1),z[hit].sum()/max(z[ok].sum(),1e-12)],2,[ok.any()]*2)
        extend([(sig[ok]>0).mean() if ok.any() else 0.,(sig[ok]<0).mean() if ok.any() else 0.],2,[ok.any()]*2)
        extend([1-ok.mean(),z[~ok].sum()],2)
    return summary(values,valid,bits,blocks)


def all_families(a):
    return dict(**correlations(a), B=branches(a), R=radial_identity_tracking(a))


def channel_mask(name, families, width):
    """Shared multiplicity/dispersion have exactly one owner."""
    keep=np.ones(width,bool)
    if name=='C3' and 'C2' in families: keep[-2:]=False
    # R index21 = log(1+N), index22 = effective multiplicity (not dispersion).
    if name=='R' and any(x in families for x in ('C2','C3')): keep[21]=False
    return keep
