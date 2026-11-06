"""One-time authenticated Q geometries/catalog and V codebooks/frequency banks."""
from pathlib import Path
import gc
import os
import time
from types import SimpleNamespace
import numpy as np
from hlt_classification.data.cache_contracts import (load_json,load_npz_arrays,write_immutable_json,
    atomic_publish_bytes,deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from .contracts import artifact,validate,setting
from .structured_contracts import recipe
from . import inputs,questions,vocabulary
from .parallel import ParallelCandidates


def publish(folder,name,arrays):
    path=Path(folder)/name; atomic_publish_bytes(path,deterministic_npz_bytes(arrays)); return file_ref(path,root=folder)


def geometry(spec,source,train,axes):
    if train.role!='train': raise PermissionError('Question targets are TRAIN-only')
    blocks={k:[] for k in ('fraction','eta','phi','charge','category','displacement','valid')}; sizes=[]; row=0
    for shard in inputs.raw(spec).shards(source,train):
        chunk={k:[] for k in blocks}
        for a,b in zip(shard['offsets'][:-1],shard['offsets'][1:]):
            value=questions.measured(shard['values'][a:b],axes[row]); row+=1; sizes.append(b-a)
            for k in blocks: chunk[k].append(value[k])
        # Per-file arrays are coalesced to keep Python-object count bounded.
        for k in blocks: blocks[k].append(np.concatenate(chunk[k]))
    return questions.Geometry(train.identities,train.labels,np.r_[0,np.cumsum(sizes)],
        **{k:np.concatenate(v) for k,v in blocks.items()},source=source).check()


def hlt_axes(spec,train):
    axes=np.empty((len(train.labels),2),np.float64); row=0
    for shard in inputs.raw(spec).shards('HLT',train):
        totals=np.add.reduceat(shard['values'][:,:4],shard['offsets'][:-1],axis=0)
        n=len(totals); axes[row:row+n,0]=np.arcsinh(totals[:,2]/np.maximum(np.hypot(totals[:,0],totals[:,1]),1e-12))
        axes[row:row+n,1]=np.arctan2(totals[:,1],totals[:,0]); row+=n
    return axes


class StatisticsSource:
    def __init__(self,geometry,proposals):
        self.p=geometry; self.proposals=proposals; self.capacity={}
    def statistics(self,indices):
        values=np.stack([questions.answers(self.p,int(i),self.proposals) for i in indices]).astype(np.float64)
        return dict(rows=len(indices),sum=values.sum(0),square_sum=(values*values).sum(0),minimum=values.min(0),maximum=values.max(0))


def statistics(geometry,proposals,workers=4):
    source=StatisticsSource(geometry,proposals); pool=None
    if os.name=='posix':
        pool=ParallelCandidates(source,workers); batches=pool.statistics(np.arange(len(geometry.labels)))
    else: batches=(source.statistics(np.arange(i,min(i+32,len(geometry.labels)))) for i in range(0,len(geometry.labels),32))
    result=dict(rows=0,sum=np.zeros(len(proposals)),square_sum=np.zeros(len(proposals)),
        minimum=np.full(len(proposals),np.inf),maximum=np.full(len(proposals),-np.inf))
    peak=None
    try:
        for value in batches:
            result['rows']+=value['rows']; result['sum']+=value['sum']; result['square_sum']+=value['square_sum']
            result['minimum']=np.minimum(result['minimum'],value['minimum']); result['maximum']=np.maximum(result['maximum'],value['maximum'])
            if result['rows']%4096<32 or result['rows']==len(geometry.labels):
                print(f'GROUP_B question support: {result["rows"]}/{len(geometry.labels)} TRAIN jets',flush=True)
                if pool is not None:
                    import resource
                    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+pool.child_pss_bytes()
                    peak=max(peak or 0,value)
    finally:
        if pool is not None: pool.close()
    return result,peak


def prepare(spec,train):
    root=Path(spec['root'])/'setup'; began=time.perf_counter(); outputs=[]
    axes=hlt_axes(spec,train)
    for source in ('OFFLINE','HLT'):
        print(f'GROUP_B preparing {source} question geometry',flush=True)
        g=geometry(spec,source,train,axes)
        payload=publish(root,'question_geometry_'+source+'.npz',g.arrays()); outputs.append(payload)
        record=artifact('QUESTION_GEOMETRY',parents=dict(spec=spec['content_hash'],raw=spec['raw_hash']),
            source=source,role='train',rows=len(train.labels),tokens=len(g.fraction),frame='paired_HLT_sum_p4_axis',payload=payload)
        write_immutable_json(root/('question_geometry_'+source+'.json'),record)
        outputs.append(file_ref(root/('question_geometry_'+source+'.json'),root=root))
        if source=='OFFLINE':
            proposals,scales,design=questions.propose(g)
            stats,preparation_peak=statistics(g,proposals,spec['candidate_workers']); arrays,coverage=questions.freeze(proposals,scales,stats)
            payload=publish(root,'question_catalog.npz',arrays); outputs.append(payload)
            catalog=artifact('QUESTION_CATALOG',parents=dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']),
                role='train',source='OFFLINE',rows=len(train.labels),coverage=coverage,scales=scales,
                diagnostic=questions.audit(g,arrays,design),
                statistics_population='all_TRAIN',design_rows=design.tolist(),payload=payload,quality_gate=False)
            write_immutable_json(root/'question_catalog.json',catalog); outputs.append(file_ref(root/'question_catalog.json',root=root))
            print('GROUP_B frozen question coverage: '+str(coverage),flush=True)
        del g; gc.collect()
    del axes
    dictionaries={}
    for source in ('OFFLINE','HLT'):
        print(f'GROUP_B fitting frozen {source} K128 dictionary',flush=True)
        p=inputs.raw(spec).load(source,train); pt=np.hypot(p.values[:,0],p.values[:,1]); p.values=None
        dictionary,diagnostic=vocabulary.fit(p); codes,distance=vocabulary.assign(p.features,dictionary['centers'],dictionary['block_scales'])
        frequencies=vocabulary.frequencies(p,codes,pt)
        diagnostic.update(vocabulary.audit(p,dictionary,codes,distance))
        payload=publish(root,'dictionary_'+source+'.npz',dictionary); outputs.append(payload)
        record=artifact('DICTIONARY',parents=dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']),
            role='train',source=source,size=128,rows=len(train.labels),normalization=p.normalization,
            diagnostic=diagnostic,payload=payload,quality_gate=False)
        write_immutable_json(root/('dictionary_'+source+'.json'),record); outputs.append(file_ref(root/('dictionary_'+source+'.json'),root=root))
        dictionaries[source]=record['content_hash']
        arrays=dict(identities=train.identities,labels=train.labels,targets=frequencies,prior=frequencies.mean(0,dtype=np.float64).astype(np.float32))
        payload=publish(root,'frequencies_'+source+'.npz',arrays); outputs.append(payload)
        record=artifact('FREQUENCY_BANK',parents=dict(spec=spec['content_hash'],dictionary=record['content_hash'],raw=spec['raw_hash']),
            role='train',source=source,rows=len(train.labels),codes=128,targets=['count','scalar_pt'],payload=payload)
        write_immutable_json(root/('frequencies_'+source+'.json'),record); outputs.append(file_ref(root/('frequencies_'+source+'.json'),root=root))
        del p,pt,codes,distance,frequencies,arrays; gc.collect()
    value=artifact('STRUCTURED_TARGETS',parents=dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']),
        outputs=outputs,catalog=catalog['content_hash'],dictionaries=dictionaries,preparation_seconds=time.perf_counter()-began,
        statistics_peak_host_bytes=preparation_peak,role='train')
    write_immutable_json(root/'structured_targets.json',value)
    return value


def authenticate(spec):
    root=Path(spec['root'])/'setup'; record=load_json(root/'structured_targets.json')
    validate(record,'STRUCTURED_TARGETS',dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']))
    names=[r['path'] for r in record['outputs']]
    expected={f'{kind}_{source}.{ext}' for kind in ('question_geometry','dictionary','frequencies') for source in ('HLT','OFFLINE') for ext in ('json','npz')}|{'question_catalog.json','question_catalog.npz'}
    if set(names)!=expected or len(names)!=len(set(names)) or record['role']!='train': raise ValueError('Structured target inventory differs')
    for ref in record['outputs']: validate_file_ref(ref,root=root)
    catalog=load_json(root/'question_catalog.json'); validate(catalog,'QUESTION_CATALOG',dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']))
    if catalog['content_hash']!=record['catalog'] or catalog['rows']!=spec['counts']['train'] or catalog['statistics_population']!='all_TRAIN':
        raise ValueError('Question catalog population differs')
    questions.validate_catalog(load_npz_arrays(root/'question_catalog.npz'),catalog)
    by_name={r['path']:r for r in record['outputs']}
    if catalog['payload']!=by_name['question_catalog.npz']: raise ValueError('Question payload reference differs')
    for source in ('HLT','OFFLINE'):
        d=load_json(root/('dictionary_'+source+'.json')); validate(d,'DICTIONARY',dict(spec=spec['content_hash'],raw=spec['raw_hash'],recipe=recipe()['content_hash']))
        f=load_json(root/('frequencies_'+source+'.json')); validate(f,'FREQUENCY_BANK',dict(spec=spec['content_hash'],dictionary=d['content_hash'],raw=spec['raw_hash']))
        g=load_json(root/('question_geometry_'+source+'.json')); validate(g,'QUESTION_GEOMETRY',dict(spec=spec['content_hash'],raw=spec['raw_hash']))
        if (d['content_hash']!=record['dictionaries'][source] or any(x['source']!=source or x['role']!='train' or x['rows']!=spec['counts']['train'] for x in (d,f,g))):
            raise ValueError('Structured target source/row binding differs')
        for kind,row in (('dictionary',d),('frequencies',f),('question_geometry',g)):
            if row['payload']!=by_name[kind+'_'+source+'.npz']: raise ValueError('Structured payload reference differs')
        arrays=load_npz_arrays(root/('dictionary_'+source+'.npz'))
        if (set(arrays)!={'centers','block_scales','mean','scale','reservoir_parents','reservoir_particles'}
                or arrays['centers'].shape!=(128,22) or arrays['centers'].dtype!=np.float32
                or arrays['block_scales'].shape!=(3,) or (arrays['block_scales']<=0).any()
                or arrays['mean'].shape!=(22,) or arrays['scale'].shape!=(22,) or (arrays['scale']<.001).any()
                or not all(np.isfinite(a).all() for a in arrays.values())
                or not np.array_equal(arrays['mean'],d['normalization']['mean'])
                or not np.array_equal(arrays['scale'],d['normalization']['scale'])
                or d['normalization']['source']!=source or d['normalization']['parents']!={'raw':spec['raw_hash']}
                or d['size']!=128 or d['quality_gate'] is not False or f['codes']!=128 or f['targets']!=['count','scalar_pt']):
            raise ValueError('Frozen dictionary normalization/semantics differs')
    return record


def candidates(spec,task,train):
    if train.role!='train': raise PermissionError('Structured targets are TRAIN-only')
    record=authenticate(spec); root=Path(spec['root'])/'setup'; cfg=setting(task); source=cfg['source']
    if cfg['family']=='Q':
        arrays=load_npz_arrays(root/('question_geometry_'+source+'.npz')); g=questions.Geometry(**arrays,source=source).check()
        if any(not np.array_equal(getattr(g,k),getattr(train,k)) for k in ('identities','labels')): raise ValueError('Question target ordered join differs')
        result=questions.QuestionCandidates(g,load_npz_arrays(root/'question_catalog.npz'),record['catalog'],
            replicate=cfg['seed_index'],families=cfg['question_families'])
        if os.name=='posix': return ParallelCandidates(result,spec['candidate_workers'])
        return result
    arrays=load_npz_arrays(root/('frequencies_'+source+'.npz'))
    if any(not np.array_equal(arrays[k],getattr(train,k)) for k in ('identities','labels')): raise ValueError('Frequency target ordered join differs')
    return vocabulary.FrequencyCandidates(arrays,record['dictionaries'][source],source=source,count_only=cfg['count_only'])


def miniature_indices(source,native_indices):
    """Keep native extremes and add full source/valid-track support tails."""
    target=getattr(source,'source',source)
    if not isinstance(target,questions.QuestionCandidates): return native_indices
    g=target.p; counts=np.diff(g.offsets)
    valid=np.add.reduceat(((g.charge!=0)&g.valid.any(-1)).astype(np.int64),g.offsets[:-1])
    ranked=np.r_[np.argsort(-counts,kind='stable')[:128],np.argsort(-valid,kind='stable')[:128],native_indices]
    selected=[]; seen=set()
    for row in ranked:
        if int(row) not in seen: selected.append(int(row)); seen.add(int(row))
    while len(selected)%128 in (0,1):
        extra=next((i for i in range(len(g.labels)) if i not in seen),None)
        if extra is None: raise ValueError('Need a non-singleton partial target-support batch')
        selected.append(extra); seen.add(extra)
    return np.asarray(selected,np.int64)
