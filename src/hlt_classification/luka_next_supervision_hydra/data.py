"""TRAIN-only physical measurements, explicit units, and parent-local descriptors."""
from pathlib import Path
import shutil
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, atomic_publish_bytes,
    deterministic_npz_bytes, write_immutable_json)
from hlt_classification.luka_fullsim import alljets_data
from hlt_classification.luka_fullsim.particles import physical
from .contracts import artifact, validate, FEATURES, COUNTS, recipe


def physical_columns(p):
    return np.column_stack((p.p4, p.charge, p.category, p.tracking, p.valid)).astype(np.float64)


def descriptors(values):
    """22 measurements; no source identity, absolute axis, rank or parent size."""
    a=np.asarray(values,np.float64)
    if (a.ndim!=2 or a.shape[1]!=14 or not len(a) or not np.isfinite(a).all()
            or not np.isin(a[:,4],[-1,0,1]).all() or not np.isin(a[:,5],range(6)).all()
            or not np.isin(a[:,10:], [0,1]).all() or (a[:,8:10]<0).any()):
        raise ValueError('Invalid physical particle collection')
    pt=np.hypot(a[:,0],a[:,1]); energy=a[:,3]; total=a[:,:4].sum(0)
    if (energy<0).any() or pt.sum()<=0 or energy.sum()<=0: raise ValueError('Invalid measured momentum')
    eta=np.arcsinh(a[:,2]/np.maximum(pt,1e-12)); axis=np.arcsinh(total[2]/max(np.hypot(*total[:2]),1e-12))
    phi=np.arctan2(a[:,1],a[:,0])-np.arctan2(total[1],total[0])
    tracking=np.where(a[:,10:].astype(bool),a[:,6:10],0.)
    sig=np.zeros((len(a),2))
    for j in range(2):
        ok=a[:,10+j].astype(bool)&a[:,12+j].astype(bool)&(a[:,8+j]>0)
        sig[ok,j]=a[ok,6+j]/a[ok,8+j]
    signed=lambda x: np.sign(x)*np.log1p(np.abs(x))
    result=np.column_stack((np.log(np.maximum(pt/pt.sum(),1e-12)),
        np.log(np.maximum(energy/energy.sum(),1e-12)),eta-axis,np.sin(phi),np.cos(phi),a[:,4],
        np.eye(6)[a[:,5].astype(int)],signed(tracking),np.arcsinh(sig),a[:,10:]))
    if result.shape!=(len(a),len(FEATURES)) or not np.isfinite(result).all(): raise ValueError('Nonfinite local descriptor')
    return result.astype(np.float32)


CONTINUOUS=np.array([0,1,2,3,4,12,13,14,15,16,17])


class Particles:
    def __init__(self, arrays, *, role='train', source, parent_hash):
        if role!='train' or source not in ('HLT','OFFLINE'): raise PermissionError('Raw sources are TRAIN-only')
        if set(arrays)!={'identities','labels','offsets','values'}: raise ValueError('Unexpected raw fields')
        self.__dict__.update(arrays); self.role=role; self.source=source; self.parent_hash=parent_hash
        n=len(self.labels)
        if (self.identities.dtype!=np.uint8 or self.identities.shape!=(n,32)
                or len(np.unique(self.identities,axis=0))!=n or self.labels.dtype!=np.int64
                or self.labels.shape!=(n,) or not np.isin(self.labels,range(11)).all()
                or self.offsets.dtype!=np.int64 or self.offsets.shape!=(n+1,) or self.offsets[0]!=0
                or np.any(np.diff(self.offsets)<1) or self.offsets[-1]!=len(self.values)
                or self.values.dtype!=np.float64 or self.values.shape!=(self.offsets[-1],14)):
            raise ValueError('Raw TRAIN population/offsets differ')
        self.local=np.empty((len(self.values),len(FEATURES)),np.float32)
        for a,b in zip(self.offsets[:-1],self.offsets[1:]):
            self.local[a:b]=descriptors(self.values[a:b])
        self.mean=np.zeros(len(FEATURES),np.float64); self.scale=np.ones(len(FEATURES),np.float64)
        # Chunked FP64 moments avoid a full duplicate FP64 feature matrix.
        sums=np.zeros(len(CONTINUOUS)); squares=sums.copy()
        for start in range(0,len(self.local),65536):
            v=self.local[start:start+65536,CONTINUOUS].astype(np.float64)
            sums+=v.sum(0); squares+=(v*v).sum(0)
        self.mean[CONTINUOUS]=sums/len(self.local)
        self.scale[CONTINUOUS]=np.maximum(np.sqrt(np.maximum(squares/len(self.local)-self.mean[CONTINUOUS]**2,0)),.001)
        self.features=np.empty_like(self.local)
        for start in range(0,len(self.local),65536):
            stop=min(start+65536,len(self.local))
            self.features[start:stop]=((self.local[start:stop]-self.mean)/self.scale).astype(np.float32)
        self.parents=np.repeat(np.arange(n,dtype=np.int64),np.diff(self.offsets))
        self.normalization=artifact('NORMALIZATION',parents=dict(raw=parent_hash),source=source,role='train',
            rows=n,tokens=len(self.local),mean=self.mean.tolist(),scale=self.scale.tolist(),features=list(FEATURES))

    def join(self, hlt):
        if hlt.role!='train' or any(not np.array_equal(getattr(hlt,k),getattr(self,k)) for k in ('identities','labels')):
            raise ValueError('Ordered TRAIN raw/HLT join differs')


def export(prepared_root, original_spec, output):
    """On SPORC: read selected TRAIN rows only; publish small physical shards."""
    from hlt_classification.luka_u100_pair_transfer import bundle as transfer
    old=transfer.checked(load_json(original_spec)); proof=transfer.authenticate(Path(original_spec),old)
    p,f,inventory,splits=alljets_data.load_prepared(prepared_root)
    if (p['content_hash']!=old['anchor']['evidence']['prepared'] or p['counts']!=COUNTS
            or proof['prepared']!=p['content_hash']): raise ValueError('Original prepared population differs')
    output=Path(output).absolute()
    sources=[Path(prepared_root).resolve(),Path(f['input_container']).resolve(),Path(old['root']).resolve()]
    if (output.exists() or not output.parent.is_dir() or any(q.is_symlink() for q in (output,*output.parents))
            or any(output.is_relative_to(q) or q.is_relative_to(output) for q in sources)):
        raise ValueError('Use a new, separate minimal raw export')
    teacher_ref=old['anchor']['evidence']['models']['M0HLT']['checkpoint']
    particles=sum(p['summaries']['train'][coord]['particles'] for coord in ('D000','OFFLINE'))
    upper_bound=int(1.05*(112*particles+96*COUNTS['train']+8192*len(splits['memberships']['train'])+teacher_ref['bytes']))
    if shutil.disk_usage(output.parent).free<upper_bound+2**30:
        raise OSError(f'Raw export needs conservative {upper_bound} bytes plus one GiB headroom')
    output.mkdir(); rows=[]; chunk=[]; last=None; n=0
    def publish(jets,index):
        for source in ('HLT','OFFLINE'):
            values=[physical_columns(physical(getattr(j,source.lower()),p['conventions'],source.lower())) for j in jets]
            arrays=dict(identities=np.array([np.frombuffer(bytes.fromhex(j.identity),np.uint8) for j in jets]),
                labels=np.array([j.label for j in jets],np.int64),offsets=np.r_[0,np.cumsum([len(v) for v in values])].astype(np.int64),
                values=np.concatenate(values))
            name=f'{source}/{index:04d}.npz'; path=output/name; path.parent.mkdir(exist_ok=True)
            atomic_publish_bytes(path,deterministic_npz_bytes(arrays)); rows.append(file_ref(path,root=output))
    for jet in alljets_data.ParticleReader(f['input_container'],inventory,splits,role='train',include_offline=True):
        if last is not None and jet.file_index!=last:
            publish(chunk,last); n+=len(chunk); chunk=[]
            if n%20000<1000: print(f'GROUP_A raw TRAIN export: {n}/{COUNTS["train"]}',flush=True)
        last=jet.file_index; chunk.append(jet)
    if chunk: publish(chunk,last); n+=len(chunk)
    if n!=COUNTS['train']: raise ValueError('Incomplete raw export')
    ref=old['anchor']['evidence']['models']['M0HLT']['checkpoint']; validate_file_ref(ref)
    with open(ref['path'],'rb') as src,(output/'M0HLT.pt').open('xb') as dst: shutil.copyfileobj(src,dst)
    rows.append(file_ref(output/'M0HLT.pt',root=output))
    value=artifact('RAW_TRANSFER',parents=dict(origin=old['content_hash'],prepared=p['content_hash'],
        source=old['source_snapshot']['content_hash'],conventions=p['conventions']['content_hash']),
        origin_proof=proof,counts=COUNTS,role='train',sources=['HLT','OFFLINE'],conventions=p['conventions'],
        original_m0=ref,files=rows,payload_bytes=sum(r['bytes'] for r in rows),event_groups_available=False,
        payload_upper_bound_bytes=upper_bound,
        exclusion_note='Unique source rows; cross-file event independence unverified',columns='p4_charge_category_tracking_mm_validity')
    write_immutable_json(output/'manifest.json',value)
    return dict(directory=str(output),manifest_hash=value['content_hash'],payload_bytes=value['payload_bytes'])


class RawBundle:
    def __init__(self,root,expected_hash):
        self.root=Path(root).resolve(strict=True); self.manifest=load_json(self.root/'manifest.json')
        validate(self.manifest,'RAW_TRANSFER'); m=self.manifest
        if (m['content_hash']!=expected_hash or m['counts']!=COUNTS or m['role']!='train'
                or m['sources']!=['HLT','OFFLINE'] or m['event_groups_available'] is not False
                or len({r['path'] for r in m['files']})!=len(m['files'])
                or m['payload_bytes']!=sum(r['bytes'] for r in m['files'])): raise ValueError('Raw transfer registration differs')
        self.rows={r['path']:r for r in m['files']}
        for name in self.rows:
            if name!='M0HLT.pt' and not (name.startswith(('HLT/','OFFLINE/')) and name.endswith('.npz')):
                raise ValueError('Unexpected raw transfer member')

    def path(self,name):
        if name not in self.rows: raise PermissionError('Not a registered raw input')
        p=self.root/name
        if any(q.is_symlink() for q in (p,*p.parents)) or not p.resolve().is_relative_to(self.root): raise ValueError('Redirected raw input')
        validate_file_ref(self.rows[name],root=self.root); return p

    def load(self,source,hlt):
        chunks=[load_npz_arrays(self.path(n)) for n in sorted(self.rows) if n.startswith(source+'/')]
        if not chunks: raise ValueError('Missing raw source shards')
        offsets=[np.array([0],np.int64)]; end=0
        for a in chunks:
            if (set(a)!={'identities','labels','offsets','values'} or a['offsets'].dtype!=np.int64
                    or a['offsets'].shape!=(len(a['labels'])+1,) or a['offsets'][0]!=0
                    or (np.diff(a['offsets'])<1).any() or a['offsets'][-1]!=len(a['values'])):
                raise ValueError('Invalid raw shard inventory/offsets')
            offsets.append(a['offsets'][1:]+end); end+=int(a['offsets'][-1])
        arrays={k:np.concatenate([a[k] for a in chunks]) for k in ('identities','labels','values')}
        arrays['offsets']=np.concatenate(offsets)
        # Raw shards and their concatenation otherwise coexist throughout all
        # descriptor construction and inflate the process-lifetime RSS peak.
        del chunks,a,offsets
        result=Particles(arrays,source=source,parent_hash=self.manifest['content_hash']); result.join(hlt)
        return result


def pair_features(local, mask):
    """Symmetric ten-channel measurements for each unordered real pair."""
    x=np.asarray(local); mask=np.asarray(mask,bool); values=[]; valid_pairs=[]
    for i in range(x.shape[-2]):
        for j in range(i+1,x.shape[-2]):
            a,b=x[...,i,:],x[...,j,:]; valid=mask[...,i]&mask[...,j]
            dphi=np.arctan2(a[...,3]*b[...,4]-a[...,4]*b[...,3],a[...,4]*b[...,4]+a[...,3]*b[...,3])
            deta=a[...,2]-b[...,2]; pt1,pt2=np.exp(a[...,0]),np.exp(b[...,0])
            vd=a[...,18:20]*b[...,18:20]
            v=np.stack((np.hypot(deta,dphi),np.abs(deta),np.cos(dphi),pt1+pt2,pt1*pt2,
                np.abs(a[...,12]-b[...,12])*vd[...,0],np.abs(a[...,13]-b[...,13])*vd[...,1],
                np.sign(a[...,12]*b[...,12])*vd[...,0],np.sign(a[...,13]*b[...,13])*vd[...,1],vd.mean(-1)),-1)
            values.append(np.where(valid[...,None],v,0.)); valid_pairs.append(valid)
    if not values:
        return np.zeros((*mask.shape[:-1],0,10),np.float32),np.zeros((*mask.shape[:-1],0),bool)
    return np.stack(values,-2).astype(np.float32),np.stack(valid_pairs,-1)


def relations(local,mask):
    values,valid=pair_features(local,mask)
    return values.sum(-2)/np.maximum(valid.sum(-1,keepdims=True),1)
