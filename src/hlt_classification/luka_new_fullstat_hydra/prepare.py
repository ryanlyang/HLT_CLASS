"""Bounded OFFLINE-only preparation; future HLT views share frozen membership."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import multiprocessing
import shutil
import numpy as np
import uproot
from hlt_classification.data.cache_contracts import (atomic_publish_bytes, deterministic_npz_bytes,
    load_json, load_npz_arrays, sha256_file, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache, _limit_worker_threads
from hlt_classification.luka_fullsim.particles import FIELDS, _matrix, _ranges, physical
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.luka_fullstat.cache import conventions
from hlt_classification.literature_context.transform import build_inputs
from . import contracts as c, population


def resident_bytes(f):
    rows=sum(f['counts'][r] for r in ('train','validation'))
    particles=sum(x['role_particles'][r]['offline'] for x in f['files'] for r in ('train','validation'))
    return particles*84+rows*88+len(f['files'])*32


def selected_rows(path, record, a):
    # Ineligible rows can lie inside a requested ROOT span and are ignored.
    # Sealed final-test and unused rows can never lie inside one.
    branches=['jet_nparticles']+['part_'+k for k in FIELDS]
    with uproot.open(path,object_cache=None,array_cache=None) as handle:
        tree=handle[record['tree_key']]
        if len(a['roles'])!=tree.num_entries:raise ValueError('Source entry count changed')
        allowed=np.flatnonzero(np.isin(a['roles'],[0,1,255]))
        if not len(allowed):return
        for lo,hi in _ranges(allowed.tolist(),c.BLOCK):
            if not np.isin(a['roles'][lo:hi],[0,1]).any():continue
            values=tree.arrays(branches,entry_start=lo,entry_stop=hi,library='ak',how=dict)
            for offset,entry in enumerate(range(lo,hi)):
                role=int(a['roles'][entry])
                if role==255:continue
                if role not in (0,1):raise PermissionError('Sealed row in particle span')
                n=int(values['jet_nparticles'][offset])
                if n!=int(a['jet_nparticles'][entry]):raise ValueError('Particle/scalar cardinality differs')
                yield c.ROLES[role],entry,_matrix(values,'part_',offset,n)


def prepare_file(args):
    root,foundation_root,output,i,r,parent=args
    root,foundation_root,output=map(Path,(root,foundation_root,output))
    receipt=output/'files'/f'{i:05d}.json'
    if receipt.exists():
        saved=load_json(receipt);c.validate(saved,'PREPARED_FILE')
        if saved['parent']!=parent or saved['file_index']!=i:raise ValueError('Prepared-file reuse differs')
        for ref in saved['blocks']:validate_file_ref(ref,root=output)
        return saved
    path=c.safe(root,r['path'])
    if sha256_file(path)!=r['sha256']:raise ValueError('Raw source drift')
    a=load_npz_arrays(validate_file_ref(r['membership'],root=foundation_root));config=conventions(parent)
    pending={'train':[],'validation':[]};serial=0;refs=[]
    def flush(role):
        nonlocal serial
        rows=pending[role]
        if not rows:return
        b=_block(i,rows);target=output/'blocks'/f'{i:05d}_{serial:05d}_{role}.npz';serial+=1
        payload={k:v for k,v in vars(b).items() if k!='file_index'}
        # Exact existing partial products are reusable; mismatched ones fail closed.
        data=deterministic_npz_bytes(payload)
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Interrupted prepared block differs')
        else:atomic_publish_bytes(target,data)
        refs.append(dict(file_ref(target,root=output),role=role,file_index=i,rows=len(b.labels),
                         particles=len(b.features),maximum=int(np.diff(b.offsets).max())))
        pending[role]=[]
    for role,entry,raw in selected_rows(path,r,a):
        light=SimpleNamespace(identity=population.identity(r,entry),label=int(a['labels'][entry]))
        pending[role].append((light,build_inputs(physical(raw,config,'offline'),capacity=512)))
        if len(pending[role])==c.BLOCK:flush(role)
    for role in pending:flush(role)
    if sha256_file(path)!=r['sha256']:raise ValueError('Raw source changed during preparation')
    result=c.artifact('PREPARED_FILE',parent=parent,file_index=i,blocks=refs)
    write_immutable_json(receipt,result)
    return result


def build(root, output, *, workers=4, available_gib, production=True):
    root=Path(root);output=Path(output);f,_=population.load(root/'foundation',production=production)
    need=int(resident_bytes(f)*1.15)+4*2**30
    if not 1<=workers<=8 or available_gib*2**30<need:raise ValueError('Insufficient declared preparation quota or invalid workers')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<need and not (output/'preparation_intent.json').exists():raise OSError('Insufficient cache free space')
    intent=c.artifact('PREPARATION_INTENT',parent=f['content_hash'],coordinate='OFFLINE')
    ip=output/'preparation_intent.json'
    if ip.exists():
        if load_json(ip)!=intent:raise ValueError('Cache directory belongs to another population')
    elif any(output.iterdir()):raise FileExistsError('Nonempty foreign cache directory')
    else:write_immutable_json(ip,intent)
    args=[(str(root),str(root/'foundation'),str(output),i,r,f['content_hash']) for i,r in enumerate(f['files'])]
    products=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_limit_worker_threads) as pool:
        tasks=iter(args);pending=[]
        for _ in range(workers):
            arg=next(tasks,None)
            if arg is not None:pending.append(pool.submit(prepare_file,arg))
        while pending:
            products.append(pending.pop(0).result())
            if len(products)%100==0 or len(products)<=6 or len(products)==len(args):
                print(f'OFFLINE prepared files {len(products)}/{len(args)}',flush=True)
            arg=next(tasks,None)
            if arg is not None:pending.append(pool.submit(prepare_file,arg))
    blocks=[b for p in products for b in p['blocks']]
    if any(sum(b['rows'] for b in blocks if b['role']==role)!=f['counts'][role] for role in ('train','validation')):
        raise ValueError('Prepared row coverage differs')
    result=c.artifact('CACHE',parent=f['content_hash'],conventions=conventions(f['content_hash']),
        coordinate='OFFLINE',counts={r:f['counts'][r] for r in ('train','validation')},blocks=blocks)
    target=output/'cache.json'
    if target.exists():
        if load_json(target)!=result:raise ValueError('Existing cache manifest differs')
    else:write_immutable_json(target,result)
    return result


def load(root, f, members, *, role, max_ram_bytes):
    if role not in ('train','validation'):raise PermissionError('Sealed cache role')
    root=Path(root);cache=load_json(root/'cache.json');c.validate(cache,'CACHE')
    if cache['parent']!=f['content_hash'] or cache['coordinate']!='OFFLINE' or cache['conventions']!=conventions(f['content_hash']):
        raise ValueError('Prepared population/input convention differs')
    blocks=[];resident=0;cursor={};entries_by_file={}
    for ref in cache['blocks']:
        if ref['role']!=role:continue
        i=ref['file_index'];a=load_npz_arrays(validate_file_ref(ref,root=root));n=ref['rows']
        if set(a)!={'offsets','features','vectors','identities','labels'}:raise ValueError('Prepared fields differ')
        if i not in entries_by_file:entries_by_file[i]=np.flatnonzero(members[i]['roles']==c.ROLES.index(role))
        entries=entries_by_file[i]
        start=cursor.get(i,0);chosen=entries[start:start+n];cursor[i]=start+n
        ids=np.asarray([np.frombuffer(bytes.fromhex(population.identity(f['files'][i],e)),np.uint8) for e in chosen])
        if (len(chosen)!=n or a['offsets'].dtype!=np.int64 or a['offsets'].shape!=(n+1,)
                or a['offsets'][0]!=0 or a['offsets'][-1]!=ref['particles']
                or not np.array_equal(np.diff(a['offsets']),members[i]['jet_nparticles'][chosen])
                or np.any(np.diff(a['offsets'])>512) or np.any(np.diff(a['offsets'])<=0)
                or a['features'].dtype!=np.float32 or a['features'].shape!=(ref['particles'],17)
                or a['vectors'].dtype!=np.float32 or a['vectors'].shape!=(ref['particles'],4)
                or not np.isfinite(a['features']).all() or not np.isfinite(a['vectors']).all()
                or a['identities'].dtype!=np.uint8 or not np.array_equal(a['identities'],ids)
                or a['labels'].dtype!=np.int64 or not np.array_equal(a['labels'],members[i]['labels'][chosen])):
            raise ValueError('Prepared shapes/ordered row join differ')
        b=RamBlock(i,**a);resident+=b.nbytes+n*40
        if resident>max_ram_bytes:raise MemoryError('Cache RAM budget exceeded')
        blocks.append(b)
    if any(cursor.get(i,0)!=r['role_counts'][role] for i,r in enumerate(f['files'])):raise ValueError('Missing/duplicate file coverage')
    result=RamCache(blocks,role=role,coordinate_name='OFFLINE',foundation_sha256=f['content_hash'])
    if len(result)!=f['counts'][role]:raise ValueError('Prepared role budget differs')
    return result
