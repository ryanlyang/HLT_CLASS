"""Role-separated native HLT, physical OFFLINE and selected TRAIN targets."""
from pathlib import Path
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, embedding_bank
from hlt_classification.luka_embedding_aux.targets import Bank as OldBank
from .contracts import artifact, validate, weights, dependencies, STRUCTURES
from .transfer import sources
from .target_banks import PhysicalBank, StructureBank, prepare_raw, prepare_structure, prepare_pretext_statistics


def views(spec,task=None):
    return tuple(load_view(spec,role,'D000') for role in ('train','validation'))


def physical(spec,cache,source):
    result=PhysicalBank(Path(spec['root'])/'setup'/f'physical_{cache.role}_{source}',cache,source)
    p,_=preparation(spec)
    if p['physicals'][cache.role+'_'+source]!=result.manifest['content_hash']:
        raise ValueError('Prepared physical inventory changed')
    return result


def structure(spec,cache,source):
    raw=physical(spec,cache,source)
    result=StructureBank(Path(spec['root'])/'setup'/f'structure_{cache.role}_{source}',raw); p,_=preparation(spec)
    if (p['structures'][cache.role+'_'+source]!=result.manifest['content_hash']
            or any(n['parents']!={'physical':p['physicals']['train_'+source]} for n in result.normalizers.values())):
        raise ValueError('Structure bank or TRAIN normalizer lineage differs')
    return result


def prepare(spec,train,val):
    import shutil
    raw,delta=sources(spec); folder=Path(spec['root'])/'setup'; physicals={}; structures={}
    def reserve(size):
        used=sum(p.stat().st_size for p in Path(spec['root']).rglob('*') if p.is_file())
        if (used+size+2*2**30>spec['storage_budget_bytes']
                or shutil.disk_usage(folder).free<size+2*2**30):
            raise OSError('Prepared arrays exceed registered storage/headroom before allocation')
    for source,cache in (('OFFLINE',train),('OFFLINE',val),('HLT',train)):
        transfer=raw if cache.role=='train' else delta
        p=prepare_raw(folder/f'physical_{cache.role}_{source}',lambda:transfer.shards(source,cache),cache,source,transfer.manifest['content_hash'],reserve=reserve)
        physicals[cache.role,source]=p
        norm=None if cache.role=='train' else structures['train',source].normalizers
        structures[cache.role,source]=prepare_structure(folder/f'structure_{cache.role}_{source}',p,normalizers=norm,reserve=reserve)
    stats=prepare_pretext_statistics(folder,physicals['train','OFFLINE'],structures['train','OFFLINE'])
    value=artifact('PREPARATION',parents=dict(spec=spec['content_hash'],statistics=stats['content_hash']),
        maximum=max(physicals[r,'OFFLINE'].manifest['maximum'] for r in ('train','validation')),
        physicals={r+'_'+s:p.manifest['content_hash'] for (r,s),p in physicals.items()},
        structures={r+'_'+s:p.manifest['content_hash'] for (r,s),p in structures.items()})
    write_immutable_json(folder/'preparation.json',value)
    return physicals,structures,stats,value


def preparation(spec):
    root=Path(spec['root'])/'setup'; p=load_json(root/'preparation.json'); s=load_json(root/'statistics.json')
    validate(s,'STATISTICS'); validate(p,'PREPARATION',dict(spec=spec['content_hash'],statistics=s['content_hash']))
    return p,s


def teacher_parents(spec):
    p,s=preparation(spec)
    return dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],
        train=p['physicals']['train_OFFLINE'],validation=p['physicals']['validation_OFFLINE'],
        train_structure=p['structures']['train_OFFLINE'],validation_structure=p['structures']['validation_OFFLINE'],
        statistics=s['content_hash'])


def banks(spec,task,train,*,technical=False):
    from .embedding_banks import Bank
    result=dict(u100=embedding_bank(spec,train))
    if task=='C_CE':
        for k in ('offline_checkpoint','offline_bank','offline_targets','teacher_receipt'): validate_file_ref(spec['reused'][k])
        result['ce_teacher']=OldBank(Path(spec['reused']['offline_bank']['path']).parent,train,teacher_ref=spec['reused']['offline_checkpoint'])
    for teacher in dependencies(task):
        if not technical:
            from .worker import validate_teacher
            validate_teacher(spec,teacher)
        folder=Path(spec['root'])/('setup/technical_'+teacher if technical else teacher)
        result[teacher]=Bank(folder/'bank',train,spec,teacher,file_ref(folder/'fit/selected.pt'),technical=technical)
    if task in STRUCTURES:
        result['structure']=structure(spec,train,'HLT' if task=='S08' else 'OFFLINE')
    return result


def fit_parents(spec,task):
    result=dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],
                u100=Bundle(spec['bundle_root']).json('targets/u100/manifest.json')['content_hash'])
    if task=='C_CE':
        validate_file_ref(spec['reused']['offline_bank'])
        result['ce_teacher']=load_json(spec['reused']['offline_bank']['path'])['content_hash']
    for teacher in dependencies(task):
        m=load_json(Path(spec['root'])/teacher/'bank/manifest.json'); validate(m,'EMBEDDING_BANK')
        if m['technical'] or m['task']!=teacher or m['parents']['spec']!=spec['content_hash']:
            raise ValueError('Scientific teacher bank lineage differs')
        result[teacher]=m['content_hash']
    if task in STRUCTURES:
        source='HLT' if task=='S08' else 'OFFLINE'
        p,_=preparation(spec)
        m=load_json(Path(spec['root'])/'setup'/f'structure_train_{source}/manifest.json')
        validate(m,'STRUCTURE_BANK',dict(physical=p['physicals']['train_'+source]))
        if m['role']!='train' or m['source']!=source or m['content_hash']!=p['structures']['train_'+source]:
            raise ValueError('Structure fit target lineage differs')
        result['structure']=m['content_hash']
    return result
