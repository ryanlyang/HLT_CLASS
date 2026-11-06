"""Measured, validity-aware questions in the paired HLT jet's angular frame."""
import hashlib
import json
import itertools
from dataclasses import dataclass
import numpy as np
from .contracts import artifact
from .structured_contracts import QUESTION_COLUMNS, SEMANTIC_WIDTH, recipe


@dataclass
class Geometry:
    identities: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    fraction: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    charge: np.ndarray
    category: np.ndarray
    displacement: np.ndarray
    valid: np.ndarray
    source: str
    role: str='train'

    def check(self):
        n=len(self.labels); m=int(self.offsets[-1])
        if (self.role!='train' or self.source not in ('HLT','OFFLINE')
                or self.identities.shape!=(n,32) or self.identities.dtype!=np.uint8
                or self.labels.dtype!=np.int64 or self.labels.shape!=(n,) or not np.isin(self.labels,range(11)).all()
                or self.offsets.shape!=(n+1,)
                or self.offsets.dtype!=np.int64 or self.offsets[0]!=0 or (np.diff(self.offsets)<1).any()
                or any(getattr(self,k).shape!=(m,) for k in ('fraction','eta','phi','charge','category'))
                or self.displacement.shape!=(m,4) or self.valid.shape!=(m,4) or self.valid.dtype!=bool
                or any(getattr(self,k).dtype!=np.float64 for k in ('fraction','eta','phi','displacement'))
                or (self.displacement<0).any() or np.any(self.displacement[~self.valid]!=0)
                or not np.isin(self.category,range(6)).all() or not np.isin(self.charge,[-1,0,1]).all()
                or (self.fraction<0).any() or (self.fraction>1+1e-12).any()
                or not all(np.isfinite(getattr(self,k)).all() for k in ('fraction','eta','phi','displacement'))):
            raise ValueError('Invalid TRAIN question geometry')
        totals=np.add.reduceat(self.fraction,self.offsets[:-1])
        if not np.allclose(totals,1.,rtol=0,atol=1e-12): raise ValueError('Question momentum fractions do not normalize')
        return self

    def arrays(self):
        return {k:getattr(self,k) for k in ('identities','labels','offsets','fraction','eta','phi','charge','category','displacement','valid')}


def measured(values,axis):
    """One full source jet; no particle correspondence is assumed."""
    x=np.asarray(values,np.float64); pt=np.hypot(x[:,0],x[:,1])
    if not len(x) or pt.sum()<=0: raise ValueError('Invalid physical question support')
    eta=np.arcsinh(x[:,2]/np.maximum(pt,1e-12))-axis[0]
    phi=(np.arctan2(x[:,1],x[:,0])-axis[1]+np.pi)%(2*np.pi)-np.pi
    d=np.zeros((len(x),4),np.float64); valid=np.zeros((len(x),4),bool)
    valid[:,:2]=x[:,10:12].astype(bool); d[:,:2]=np.where(valid[:,:2],abs(x[:,6:8]),0.)
    valid[:,2:]=valid[:,:2]&x[:,12:14].astype(bool)&(x[:,8:10]>0)
    np.divide(abs(x[:,6:8]),x[:,8:10],out=d[:,2:],where=valid[:,2:])
    return dict(fraction=pt/pt.sum(),eta=eta,phi=phi,charge=x[:,4].astype(np.int8),
        category=x[:,5].astype(np.int8),displacement=d,valid=valid)


def answers(geometry,anchor,questions,*,question_chunk=64,pair_chunk=512):
    q=np.asarray(questions,np.float64)
    if q.ndim!=2 or q.shape[1]!=len(QUESTION_COLUMNS) or not np.isfinite(q).all(): raise ValueError('Malformed semantic questions')
    a,b=geometry.offsets[anchor:anchor+2]; z=geometry.fraction[a:b]; eta=geometry.eta[a:b]; phi=geometry.phi[a:b]
    charge=geometry.charge[a:b]; cat=geometry.category[a:b]; disp=geometry.displacement[a:b]; valid=geometry.valid[a:b]
    radius=np.hypot(eta,phi); result=np.empty(len(q),np.float32)
    # A per-jet triangular index is bounded in chunks; no dataset pair table.
    for start in range(0,len(q),question_chunk):
        v=q[start:start+question_chunk]; family=v[:,0].astype(int); field=v[:,3].astype(int)
        selected=((v[:,1,None]==6)|(cat[None]==v[:,1,None]))&((v[:,2,None]==3)|(charge[None]==v[:,2,None]-1))
        selected&=(radius[None]>=v[:,4,None])&(radius[None]<v[:,5,None])
        selected&=(z[None]>=v[:,6,None])&(z[None]<v[:,7,None])
        tracked=family!=0; indices=np.minimum(field,3)
        selected[tracked]&=(charge[None]!=0)&valid[:,indices[tracked]].T&(disp[:,indices[tracked]].T>v[tracked,8,None])
        values=(selected*z[None]).sum(-1)
        counts=selected.sum(-1); values[family==1]=counts[family==1]>=v[family==1,9]
        for j in np.flatnonzero(family==3):
            members=np.flatnonzero(selected[j]); found=False
            for first in range(0,len(members),pair_chunk):
                left=members[first:first+pair_chunk]
                for second in range(first,len(members),pair_chunk):
                    right=members[second:second+pair_chunk]
                    dphi=(phi[left,None]-phi[right][None]+np.pi)%(2*np.pi)-np.pi
                    dr=np.hypot(eta[left,None]-eta[right][None],dphi)
                    ok=(left[:,None]<right[None])&(dr>=v[j,10])&(dr<v[j,11])
                    relation=int(v[j,12])
                    if relation: ok&=charge[left,None]*charge[right][None]==(1 if relation==1 else -1)
                    if ok.any(): found=True; break
                if found: break
            values[j]=found
        result[start:start+len(v)]=np.clip(values,0.,1.)
    return result


def semantic(questions,scales):
    q=np.asarray(questions,np.float64); fam=q[:,0].astype(int); field=q[:,3].astype(int)
    threshold=np.arcsinh(q[:,8]/np.asarray(scales['displacement'])[np.minimum(field,3)])
    numbers=np.column_stack((q[:,4:6]/scales['radius'],q[:,6:8],threshold,
        np.log1p(q[:,9])/np.log(35.),q[:,10:12]/scales['separation']))
    result=np.column_stack((np.eye(4)[fam],np.eye(7)[q[:,1].astype(int)],np.eye(4)[q[:,2].astype(int)],
        np.eye(5)[field],np.eye(3)[q[:,12].astype(int)],numbers)).astype(np.float32)
    if result.shape!=(len(q),SEMANTIC_WIDTH) or not np.isfinite(result).all(): raise ValueError('Invalid semantic query encoding')
    return result


def reservoir_rows(identities,limit):
    ranks=np.array([hashlib.sha256(b'GROUP_B_DESIGN/v1'+bytes(x)).digest() for x in identities],dtype='S32')
    return np.argsort(ranks,kind='stable')[:limit]


def propose(geometry,*,per_family=2048,design_jets=32768):
    """Global TRAIN-derived scales/ranges; no per-anchor answer enters a query."""
    geometry.check(); design=reservoir_rows(geometry.identities,min(design_jets,len(geometry.labels)))
    chosen=[]
    for i in design:
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(bytes(geometry.identities[i])+b'Q_design').digest(),'little'))
        own=np.arange(geometry.offsets[i],geometry.offsets[i+1]); chosen.extend(rng.choice(own,min(8,len(own)),replace=False))
    ix=np.array(chosen,np.int64); rad=np.hypot(geometry.eta[ix],geometry.phi[ix]); frac=geometry.fraction[ix]
    radius=max(float(np.hypot(geometry.eta,geometry.phi).max()),1e-6)*1.000001
    radii=np.unique(np.r_[0.,np.quantile(rad,[.1,.25,.5,.75,.9,.97]),radius])
    fractions=np.unique(np.r_[0.,np.quantile(frac,[.1,.25,.5,.75,.9,.99]),1.000001])
    ranges=lambda edges: np.array([(a,b) for i,a in enumerate(edges) for b in edges[i+1:] if b>a])
    rrange,zrange=ranges(radii),ranges(fractions); thresholds=[]; scales=[]
    for field in range(4):
        values=geometry.displacement[ix,field][geometry.valid[ix,field]&(geometry.charge[ix]!=0)]
        thresholds.append(np.unique(np.r_[0.,np.quantile(values,[.25,.5,.75,.9,.97,.99]) if len(values) else [0.]]))
        scales.append(max(float(np.quantile(values,.9)) if len(values) else 0.,1e-6))
    separations=np.unique(np.r_[0.,np.quantile(rad,[.1,.25,.5,.75,.9,.97])*2,2*radius])
    srange=ranges(separations); rows=[]
    # Reject only globally impossible category/charge combinations, not rare
    # combinations or a query that is negative for a particular anchor.
    pairs={}
    for field in range(5):
        valid=np.ones(len(geometry.charge),bool) if field==4 else (geometry.charge!=0)&geometry.valid[:,field]
        values=np.bincount(geometry.category[valid].astype(np.int64)*3+geometry.charge[valid]+1,minlength=18).reshape(6,3)
        support={(int(a),int(b)) for a,b in zip(*np.nonzero(values))}
        support|={(a,3) for a in np.flatnonzero(values.sum(-1))}
        support|={(6,b) for b in np.flatnonzero(values.sum(0))}; support.add((6,3))
        pairs[field]=sorted(support)
    for family in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([2026110511,family])); seen=set(); made=[]
        fields=[4] if family==0 else list(range(4))
        capacity=sum(len(pairs[f])*len(rrange)*len(zrange)*(1 if family==0 else len(thresholds[f]))*
            (8 if family==1 else 1)*(len(srange)*3 if family==3 else 1) for f in fields)
        target=min(per_family,capacity)
        if capacity<=per_family:
            for field in fields:
                for pair,radial,momentum in itertools.product(pairs[field],rrange,zrange):
                    for threshold,count,separation,relation in itertools.product([0.] if family==0 else thresholds[field],
                            [1,2,3,5,8,13,21,34] if family==1 else [0],srange if family==3 else [(0.,0.)],
                            range(3) if family==3 else [0]):
                        made.append((family,*pair,field,*radial,*momentum,threshold,count,*separation,relation))
        while len(made)<target:
            field=4 if family==0 else int(rng.integers(4))
            category,charge=pairs[field][rng.integers(len(pairs[field]))]
            rlo,rhi=rrange[rng.integers(len(rrange))]; zlo,zhi=zrange[rng.integers(len(zrange))]
            threshold=0. if family==0 else float(rng.choice(thresholds[field]))
            count=int(rng.choice([1,2,3,5,8,13,21,34])) if family==1 else 0
            slo,shi=srange[rng.integers(len(srange))] if family==3 else (0.,0.)
            relation=int(rng.integers(3)) if family==3 else 0
            value=(family,category,charge,field,rlo,rhi,zlo,zhi,threshold,count,slo,shi,relation)
            if value not in seen: seen.add(value); made.append(value)
        rows.extend(made)
    scale=dict(radius=radius,separation=2*radius,displacement=scales)
    return np.asarray(rows,np.float64),scale,design


def freeze(proposals,scales,statistics,*,goal=1024):
    q=np.asarray(proposals); chosen=[]; applicable=[]; counts=[]
    for family in range(4):
        rows=np.flatnonzero(q[:,0]==family); supported=rows[statistics['maximum'][rows]>statistics['minimum'][rows]]
        keep=supported[:goal] if len(supported) else rows[:goal]
        chosen.extend(keep); applicable.extend([bool(len(supported))]*len(keep))
        counts.append(dict(family=family,proposed=len(rows),nonconstant=len(supported),retained=len(keep),applicable=bool(len(supported))))
    chosen=np.array(chosen,np.int64)
    return dict(questions=q[chosen],encoded=semantic(q[chosen],scales),applicable=np.array(applicable,bool),
        prior=(statistics['sum'][chosen]/statistics['rows']).astype(np.float32),
        **{k:v[chosen] for k,v in statistics.items() if k!='rows'}),counts


def validate_catalog(arrays,record):
    names={'questions','encoded','applicable','prior','sum','square_sum','minimum','maximum'}
    if set(arrays)!=names: raise ValueError('Unexpected catalog arrays')
    questions=arrays['questions']; n=len(questions)
    if (questions.shape!=(n,13) or questions.dtype!=np.float64 or not n
            or arrays['encoded'].shape!=(n,31) or arrays['encoded'].dtype!=np.float32
            or arrays['applicable'].shape!=(n,) or arrays['applicable'].dtype!=bool
            or arrays['prior'].shape!=(n,) or arrays['prior'].dtype!=np.float32
            or any(arrays[k].shape!=(n,) or arrays[k].dtype!=np.float64 for k in ('sum','square_sum','minimum','maximum'))
            or not all(np.isfinite(a).all() for a in arrays.values())
            or len(np.unique(questions,axis=0))!=n or set(questions[:,0])!={0,1,2,3}
            or not np.isin(questions[:,1],range(7)).all() or not np.isin(questions[:,2],range(4)).all()
            or not np.isin(questions[:,3],range(5)).all() or not np.isin(questions[:,12],range(3)).all()
            or (questions[:,4]>=questions[:,5]).any() or (questions[:,6]>=questions[:,7]).any()
            or (arrays['minimum']<0).any() or (arrays['maximum']>1).any()
            or (arrays['minimum']>arrays['maximum']).any() or (arrays['prior']<0).any() or (arrays['prior']>1).any()
            or record['role']!='train' or record['source']!='OFFLINE' or record['quality_gate'] is not False):
        raise ValueError('Invalid frozen question catalog')
    if (not np.array_equal(arrays['encoded'],semantic(questions,record['scales']))
            or not np.array_equal(arrays['prior'],(arrays['sum']/record['rows']).astype(np.float32))):
        raise ValueError('Catalog encoding/prior differs from frozen semantics')
    for row in record['coverage']:
        selected=questions[:,0]==row['family']
        if selected.sum()!=row['retained'] or not np.all(arrays['applicable'][selected]==row['applicable']):
            raise ValueError('Catalog support coverage differs')
    return arrays


def audit(geometry,arrays,design):
    questions=arrays['questions']; encoded=arrays['encoded']; pairs=[]
    for family in range(4):
        indices=np.flatnonzero(questions[:,0]==family)
        for i in indices[np.linspace(0,len(indices)-1,min(32,len(indices)),dtype=int)]:
            distance=((encoded[indices]-encoded[i])**2).sum(-1); distance[indices==i]=np.inf
            if len(indices)>1: pairs.append((int(i),int(indices[distance.argmin()])))
    result=[]
    if pairs:
        pair_array=np.asarray(pairs); unique,inverse=np.unique(pair_array,return_inverse=True)
        inverse=inverse.reshape(-1)
        values=np.stack([answers(geometry,int(i),questions[unique]) for i in design[:64]])
        for j,(a,b) in enumerate(pair_array):
            x,y=values[:,inverse[2*j]].astype(np.float64),values[:,inverse[2*j+1]].astype(np.float64)
            a0,b0=x-x.mean(),y-y.mean(); denominator=np.sqrt(np.dot(a0,a0)*np.dot(b0,b0))
            corr=float(np.dot(a0,b0)/denominator) if denominator>0 else None
            result.append(dict(first=int(a),second=int(b),correlation=corr,equal_fraction=float(np.mean(x==y))))
    families=[]
    for family in range(4):
        selected=questions[:,0]==family; qs=questions[selected]
        mean=arrays['sum'][selected]/len(geometry.labels)
        variance=np.maximum(arrays['square_sum'][selected]/len(geometry.labels)-mean**2,0.)
        families.append(dict(family=family,category_counts=np.bincount(qs[:,1].astype(int),minlength=7).tolist(),
            field_counts=np.bincount(qs[:,3].astype(int),minlength=5).tolist(),
            annuli=len(np.unique(qs[:,4:6],axis=0)),momentum_windows=len(np.unique(qs[:,6:8],axis=0)),
            minimum_mean=float(mean.min()),maximum_mean=float(mean.max()),mean_variance=float(variance.mean())))
    return dict(families=families,nearby_query_pairs=result,redundancy_population='up_to64_frozen_TRAIN_design_rows_not_independent')


class QuestionCandidates:
    def __init__(self,geometry,catalog,catalog_hash,*,replicate=0,families=(0,1,2,3)):
        geometry.check(); self.p=geometry; self.catalog=catalog; self.replicate=replicate; self.families=list(families)
        self.catalog_hash=catalog_hash
        valid_counts=np.add.reduceat(((geometry.charge!=0)&geometry.valid.any(-1)).astype(np.int64),geometry.offsets[:-1])
        self.capacity=dict(rows=len(geometry.labels),class_counts=np.bincount(geometry.labels,minlength=11).tolist(),
            source=geometry.source,catalog=catalog_hash,question_families=self.families,queries=64,
            maximum_source_particles=int(np.diff(geometry.offsets).max()),maximum_valid_charged=int(valid_counts.max()))

    def batch(self,anchors,epoch,cfg,mine=None):
        if type(epoch) is not int or epoch<1: raise ValueError('Epoch is one-based')
        batches=[]; info=[]
        for anchor in anchors:
            indices=[]
            for family in self.families:
                available=np.flatnonzero(self.catalog['questions'][:,0]==family)
                key=[2026110512,self.replicate,epoch,bytes(self.p.identities[anchor]).hex(),family]
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest(),'little'))
                pool=rng.choice(available,32,replace=len(available)<32)
                indices.extend(pool[:64//len(self.families)])
            ix=np.asarray(indices,np.int64); qs=self.catalog['questions'][ix]
            batches.append(dict(questions=self.catalog['encoded'][ix],targets=answers(self.p,int(anchor),qs),
                families=qs[:,0].astype(np.int64),applicable=self.catalog['applicable'][ix],prior=self.catalog['prior'][ix]))
            info.append(dict(queries=len(ix),source=self.p.source,catalog=self.catalog_hash,families=self.families))
        return {k:np.stack([r[k] for r in batches]) for k in batches[0]},info

    def statistics(self,anchors,proposals):
        values=np.stack([answers(self.p,int(i),proposals) for i in anchors]).astype(np.float64)
        return dict(rows=len(anchors),sum=values.sum(0),square_sum=(values*values).sum(0),minimum=values.min(0),maximum=values.max(0))
