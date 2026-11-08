"""Minimal physical validation delta, with explicit role/parent capabilities."""
from pathlib import Path
import shutil
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.luka_fullsim import alljets_data
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_next_supervision_hydra.data import physical_columns, RawBundle
from .contracts import COUNTS, artifact, validate


def export(prepared_root, origin_spec, output, source_snapshot):
    from hlt_classification.luka_u100_pair_transfer import bundle as transfer
    old=transfer.checked(load_json(origin_spec)); proof=transfer.authenticate(Path(origin_spec),old)
    p,f,inventory,splits=alljets_data.load_prepared(prepared_root)
    if p['content_hash']!=proof['prepared'] or p['counts']!=COUNTS:
        raise ValueError('Physical validation population differs')
    output=Path(output).absolute()
    protected=[Path(prepared_root).resolve(),Path(f['input_container']).resolve(),Path(old['root']).resolve()]
    if (output.exists() or not output.parent.is_dir() or any(q.is_symlink() for q in (output,*output.parents))
            or any(output.is_relative_to(q) or q.is_relative_to(output) for q in protected)):
        raise ValueError('Use a fresh separate export directory')
    tokens=sum(p['summaries']['validation'][c]['particles'] for c in ('D000','OFFLINE'))
    if shutil.disk_usage(output.parent).free<tokens*120+COUNTS['validation']*128+2**30:
        raise OSError('Insufficient validation export space')
    output.mkdir(); files=[]; chunk=[]; last=None; count=0
    def publish(jets,index):
        for source in ('HLT','OFFLINE'):
            rows=[physical_columns(physical(getattr(j,source.lower()),p['conventions'],source.lower())) for j in jets]
            arrays=dict(identities=np.array([np.frombuffer(bytes.fromhex(j.identity),np.uint8) for j in jets]),
                labels=np.array([j.label for j in jets],np.int64),
                offsets=np.r_[0,np.cumsum([len(a) for a in rows])].astype(np.int64),values=np.concatenate(rows))
            dest=output/source/f'{index:04d}.npz'; dest.parent.mkdir(exist_ok=True)
            atomic_publish_bytes(dest,deterministic_npz_bytes(arrays)); files.append(file_ref(dest,root=output))
    for jet in alljets_data.ParticleReader(f['input_container'],inventory,splits,role='validation',include_offline=True):
        if last is not None and jet.file_index!=last:
            publish(chunk,last); count+=len(chunk); chunk=[]
            print(f'PRETEXT validation export: {count}/{COUNTS["validation"]}',flush=True)
        chunk.append(jet); last=jet.file_index
    if chunk: publish(chunk,last); count+=len(chunk)
    if count!=COUNTS['validation']: raise ValueError('Incomplete export')
    value=artifact('TRANSFER',parents=dict(prepared=p['content_hash'],source=source_snapshot['content_hash'],
        conventions=p['conventions']['content_hash']),source_snapshot=source_snapshot,origin_proof=proof,
        conventions=p['conventions'],role='validation',sources=['HLT','OFFLINE'],counts=COUNTS,
        columns='p4_charge_category_tracking_mm_validity',files=files,payload_bytes=sum(r['bytes'] for r in files))
    write_immutable_json(output/'manifest.json',value)
    return dict(directory=str(output),manifest_hash=value['content_hash'],payload_bytes=value['payload_bytes'])


class ValidationDelta:
    def __init__(self,root,digest,raw):
        self.root=Path(root).resolve(strict=True); self.manifest=load_json(self.root/'manifest.json')
        m=self.manifest; validate(m,'TRANSFER')
        if (m['content_hash']!=digest or m['role']!='validation' or m['counts']!=COUNTS
                or m['sources']!=['HLT','OFFLINE'] or m['origin_proof']!=raw.manifest['origin_proof']
                or m['conventions']!=raw.manifest['conventions'] or m['columns']!=raw.manifest['columns']
                or m['payload_bytes']!=sum(r['bytes'] for r in m['files'])
                or len({r['path'] for r in m['files']})!=len(m['files'])):
            raise ValueError('Validation physical delta lineage differs')
        for r in m['files']:
            p=Path(r['path'])
            if len(p.parts)!=2 or p.parts[0] not in ('HLT','OFFLINE') or p.suffix!='.npz':
                raise ValueError('Unexpected validation member')
    def shards(self,source,cache):
        if cache.role!='validation' or source not in ('HLT','OFFLINE'):
            raise PermissionError('Validation-only physical capability')
        start=0
        for ref in self.manifest['files']:
            if not ref['path'].startswith(source+'/'): continue
            p=self.root/ref['path']
            if any(q.is_symlink() for q in (p,*p.parents)): raise ValueError('Redirected input')
            validate_file_ref(ref,root=self.root); a=load_npz_arrays(p); n=len(a['labels'])
            check_shard(a,cache,start); start+=n; yield a
        if start!=len(cache): raise ValueError('Validation physical coverage differs')


def check_shard(a,cache,start):
    n=len(a['labels'])
    if (set(a)!={'identities','labels','offsets','values'} or a['values'].dtype!=np.float64
            or a['values'].shape!=(a['offsets'][-1],14) or a['offsets'].dtype!=np.int64
            or a['offsets'].shape!=(n+1,) or a['offsets'][0]!=0 or (np.diff(a['offsets'])<1).any()
            or any(not np.array_equal(a[k],getattr(cache,k)[start:start+n]) for k in ('identities','labels'))
            or not np.isfinite(a['values']).all()):
        raise ValueError('Physical shard shape/ordered identity join differs')


def sources(spec):
    raw=RawBundle(spec['raw_root'],spec['raw_hash'])
    return raw,ValidationDelta(spec['validation_root'],spec['validation_hash'],raw)
