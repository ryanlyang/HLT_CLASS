"""Bounded persistent preparation, role-separated views and TRAIN target access."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hlt_classification.cms_proxy_ladder import views as kernel
from hlt_classification.luka_fullsim.stage2 import ViewMeter
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, embedding_bank, metadata
from hlt_classification.luka_selective_teacher_hydra.contracts import GROUPS
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate, COUNTS, recipe
from .registry import COORDINATES, CONDITIONS, cell, SLICES
from .storage import Writer, Store, join, validate_population
from . import regions, transfer, native


def transferred(spec):
    b=Bundle(spec['bundle_root']); origin=b.json('origin/campaign_spec.json')
    t=transfer.Transfer(spec['transfer_root'],spec['transfer_hash'],origin)
    prepared=b.json('prepared/prepared.json')
    if any(t.manifest[k]!=prepared[k] for k in ('summaries','assignments','conventions')):
        raise ValueError('Transfer preparation proof differs from original bundle')
    return t


def data_parents(spec):
    return dict(spec=spec['content_hash'],prepared=spec['anchor']['evidence']['prepared'],
                transfer=spec['transfer_hash'],native=spec['native_hash'],recipe=recipe()['content_hash'])


def prepare(spec):
    """Single disposable process, one physical source shard resident at a time."""
    t=transferred(spec); sources=native.Sources(spec); counts=spec['counts']; root=Path(spec['root'])/'setup/data'
    root.mkdir(exist_ok=False); manifests={}
    for role,n in counts.items():
        summaries=t.manifest['summaries'][role]; htotal=summaries['D000']['particles']; ototal=summaries['OFFLINE']['particles']
        raw=Writer(root/role/'physical',dict(identities=(np.uint8,(n,32)),labels=(np.int64,(n,)),
            HLT_offsets=(np.int64,(n+1,)),OFFLINE_offsets=(np.int64,(n+1,)),
            HLT_values=(np.float64,(htotal,14)),OFFLINE_values=(np.float64,(ototal,14))))
        writers={}
        for coord in COORDINATES:
            total=ototal if coord=='OFFLINE' else htotal
            writers[coord]=Writer(root/role/coord,dict(features=(np.float32,(total,17)),vectors=(np.float32,(total,4))))
        meters={coord:ViewMeter() for coord in COORDINATES}; row=0; ends=dict(HLT=0,OFFLINE=0)
        for s,a in t.shards(role):
            batch_rows=len(a['labels']); raw.arrays['identities'][row:row+batch_rows]=a['identities'];raw.arrays['labels'][row:row+batch_rows]=a['labels']
            for source in ends:
                v=a[source+'_values']; raw.arrays[source+'_values'][ends[source]:ends[source]+len(v)]=v
                raw.arrays[source+'_offsets'][row:row+batch_rows+1]=a[source+'_offsets']+ends[source]
            # Preserve the authenticated input bytes. Re-encoding the physical
            # endpoints on another platform can change even one float32 ULP.
            for coord in native.HISTORICAL:
                block=sources.block(role,coord,s['index'],a)
                start=ends['OFFLINE' if coord=='OFFLINE' else 'HLT']
                for key in ('features','vectors'):
                    writers[coord].arrays[key][start:start+len(block.features)]=getattr(block,key)
                for j,label in enumerate(block.labels):
                    lo,hi=block.offsets[j:j+2]
                    meters[coord].add(SimpleNamespace(identity=bytes(block.identities[j]).hex(),label=int(label)),
                        SimpleNamespace(features=block.features[lo:hi],vectors=block.vectors[lo:hi]))
            for j in range(batch_rows):
                hs,he=a['HLT_offsets'][j:j+2];os_,oe=a['OFFLINE_offsets'][j:j+2]
                h=regions.physical(a['HLT_values'][hs:he]);o=regions.physical(a['OFFLINE_values'][os_:oe])
                identity=bytes(a['identities'][j]).hex();jet=SimpleNamespace(identity=identity,label=int(a['labels'][j]))
                for coord in ('D025','D050','D075'):
                    v=kernel.build_view(identity=identity,proxy=h,offline=o,coordinate=coord,mapping=a['mapping'][hs:he])
                    inp=build_inputs(v,capacity=512); meters[coord].add(jet,inp)
                    start=(ends['OFFLINE']+os_) if coord=='OFFLINE' else ends['HLT']+hs
                    writers[coord].arrays['features'][start:start+len(v)]=inp.features
                    writers[coord].arrays['vectors'][start:start+len(v)]=inp.vectors
            row+=batch_rows
            for source in ends: ends[source]+=len(a[source+'_values'])
            if row%20000<1000:print(f'EMBEDDING_BRIDGE prepare {role}: {row}/{n}',flush=True)
        if row!=n or ends!=dict(HLT=htotal,OFFLINE=ototal): raise ValueError('Preparation incomplete')
        proofs={coord:meter.report() for coord,meter in meters.items()}
        for coord in native.HISTORICAL:
            if proofs[coord]!=summaries[coord]: raise ValueError('Original native/third-view replay differs: '+role+'/'+coord)
        manifests[role]={}
        for coord,w in writers.items():
            manifests[role][coord]=w.finish('VIEW',parents=data_parents(spec),role=role,coordinate=coord,rows=n,proof=proofs[coord])['content_hash']
        manifests[role]['physical']=raw.finish('PHYSICAL',parents=data_parents(spec),role=role,rows=n,
            columns='p4_charge_category_tracking_mm_validity',usage='optimization' if role=='train' else 'evaluation_only')['content_hash']
    value=artifact('DATA',parents=data_parents(spec),views=manifests,counts=counts)
    write_immutable_json(root/'manifest.json',value)
    # Original TRAIN target meaning and bytes are retained; only container encoding changes.
    ds=Dataset(spec);u=embedding_bank(spec,ds.view('train','D000'))
    payload=load_json(Bundle(spec['bundle_root']).path('targets/u100/manifest.json'))
    if payload['content_hash']!=spec['embedding_hash']: raise ValueError('Original U100 bank changed')
    from hlt_classification.data.cache_contracts import load_npz_arrays
    original=load_npz_arrays(Bundle(spec['bundle_root']).path('targets/u100/targets.npz'))
    w=Writer(root/'u100_train',{k:(v.dtype,v.shape) for k,v in original.items()})
    for k,v in original.items():w.arrays[k][:]=v
    w.finish('U100_BANK',parents=dict(**data_parents(spec),original=spec['embedding_hash']),role='train',
             coordinate='U100',rows=counts['train'],meaning='original_standardized_U100',source_manifest=payload)
    return value


class View:
    def __init__(self, store, physical, *, role, coordinate, prepared):
        self.store,self.physical=store,physical; self.role=role; self.coordinate_name=coordinate;self.foundation_sha256=prepared
        self.identities=physical['identities'];self.labels=physical['labels'];self.offsets=physical['OFFLINE_offsets' if coordinate=='OFFLINE' else 'HLT_offsets']
        validate_population(self.identities,self.labels)
        self.features=store['features'];self.vectors=store['vectors']
        if (self.features.shape!=(self.offsets[-1],17) or self.vectors.shape!=(self.offsets[-1],4)
                or self.features.dtype!=np.float32 or self.vectors.dtype!=np.float32):raise ValueError('View shape differs')
    def __len__(self):return len(self.labels)
    def batch(self,indices):
        ix=indices_checked(indices,len(self));parts=[]
        for i in ix:
            a,b=self.offsets[i:i+2];parts.append(SimpleNamespace(features=self.features[a:b],vectors=self.vectors[a:b]))
        return regions.padded(parts,self.labels[ix],self.identities[ix])


def indices_checked(indices,count):
    a=np.asarray(indices)
    if a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a) or (a<0).any() or (a>=count).any():
        raise ValueError('Invalid row indices')
    return a


class Dataset:
    def __init__(self,spec):
        self.spec=spec;self.root=Path(spec['root'])/'setup/data';m=load_json(self.root/'manifest.json')
        validate(m,'DATA',data_parents(spec))
        if m['counts']!=spec['counts'] or set(m['views'])!=set(COUNTS):raise ValueError('Data roles differ')
        self.manifest=m;self._physical={};self._views={};self._u={}

    def physical(self,role):
        if role not in COUNTS:raise PermissionError('No final-test reader')
        if role not in self._physical:
            self._physical[role]=Store(self.root/role/'physical','PHYSICAL',parents=data_parents(self.spec),digest=self.manifest['views'][role]['physical'])
            m=self._physical[role].manifest
            if m['role']!=role or m['rows']!=self.spec['counts'][role]:raise ValueError('Physical role differs')
        return self._physical[role]

    def view(self,role,coord):
        if role not in COUNTS or coord not in COORDINATES:raise PermissionError('Undeclared view')
        key=(role,coord)
        if key not in self._views:
            s=Store(self.root/role/coord,'VIEW',parents=data_parents(self.spec),digest=self.manifest['views'][role][coord])
            if s.manifest['role']!=role or s.manifest['coordinate']!=coord:raise ValueError('View meaning differs')
            self._views[key]=View(s,self.physical(role),role=role,coordinate=coord,prepared=self.spec['anchor']['evidence']['prepared'])
        return self._views[key]

    def condition(self,role,indices,condition):
        if condition not in CONDITIONS:raise ValueError('Unknown information condition')
        h=self.view(role,'D000').batch(indices)
        if condition=='B':return h
        u=self.view(role,'U100').batch(indices)
        if not np.array_equal(h['mask'],u['mask']):raise ValueError('U100/D000 slots differ')
        fields='KIT' if condition=='KIT' else condition
        for group in fields:h['features'][:,GROUPS[group],:]=u['features'][:,GROUPS[group],:]
        if 'K' in fields:h['vectors']=u['vectors']
        return h

    def crops(self,role,indices,source,scale):
        from . import regional_inputs
        p=self.physical(role);ix=indices_checked(indices,len(p['labels']))
        def rows(name):
            return [p[name+'_values'][p[name+'_offsets'][i]:p[name+'_offsets'][i+1]] for i in ix]
        if source not in ('HLT','OFFLINE'):raise ValueError('Unknown crop source')
        return regional_inputs.crops(rows('HLT'),rows(source),p['labels'][ix],scale)

    def u100(self,role):
        if role not in COUNTS:raise PermissionError('Undeclared target role')
        if role in self._u:return self._u[role]
        parents=dict(**data_parents(self.spec),original=self.spec['embedding_hash'])
        store=Store(self.root/('u100_'+role),'U100_BANK',parents=parents)
        m=store.manifest;n=self.spec['counts'][role]
        if (m['role']!=role or m['coordinate']!='U100' or m['rows']!=n or m['meaning']!='original_standardized_U100'
                or set(store.arrays)!={'identities','labels','targets','mean','scale'} or store['targets'].shape!=(n,128)
                or store['targets'].dtype!=np.float32):raise PermissionError('Original target meaning differs')
        for k in ('mean','scale'):
            a=store[k]
            if a.shape!=(128,) or a.dtype!=np.float64 or not np.isfinite(a).all() or (k=='scale' and (a<.001).any()):
                raise ValueError('Original target normalization differs')
        for start in range(0,n,4096):
            if not np.isfinite(store['targets'][start:start+4096]).all():raise ValueError('Nonfinite original targets')
        join(self.view(role,'D000'),store)
        self._u[role]=store
        return store


class SubsetDataset:
    """Technical-only row selection; production geometry and frontend stay intact."""
    def __init__(self,base,indices):
        self.base,self.spec,self.indices=base,base.spec,indices
        self._views={};self._u={}
    def view(self,role,coord):
        key=(role,coord)
        if key not in self._views:
            v=self.base.view(role,coord);ix=self.indices[role]
            class Slice:
                role=v.role
                coordinate_name=v.coordinate_name
                foundation_sha256=v.foundation_sha256
                identities=v.identities[ix]
                labels=v.labels[ix]
                def __len__(self):return len(ix)
                def batch(self,indices):return v.batch(ix[indices_checked(indices,len(ix))])
            self._views[key]=Slice()
        return self._views[key]
    def condition(self,role,indices,condition):return self.base.condition(role,self.indices[role][indices],condition)
    def crops(self,role,indices,source,scale):return self.base.crops(role,self.indices[role][indices],source,scale)
    def u100(self,role):
        if role not in self._u:
            original=self.base.u100(role);ix=self.indices[role]
            class Slice(dict):pass
            s=Slice({k:np.asarray(original[k][ix]) for k in ('identities','labels','targets')})
            for k in ('mean','scale'):s[k]=original[k]
            s.manifest=original.manifest;self._u[role]=s
        return self._u[role]
