"""One-time source export: exact original assignments, physical rows and U100 weights."""
from pathlib import Path
import itertools
import os
import shutil
import hashlib
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.luka_fullsim import alljets_data, stage2
from hlt_classification.luka_fullsim.particles import physical
from hlt_classification.luka_next_supervision_hydra.data import physical_columns
from hlt_classification.luka_u100_pair_transfer import bundle as original_transfer
from hlt_classification.luka_cross_message_hydra.inputs import teacher_files, origin_proof
from .contracts import artifact, validate, COUNTS, recipe
from .storage import regular, validate_population
from hlt_classification.provenance import validate_source_snapshot_payload


def fresh(root, protected):
    root=Path(root).absolute()
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root
            or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate output directory; preserve incomplete outputs')
    return root


def export(prepared_root, origin_spec, output, *, source_snapshot):
    old=original_transfer.checked(load_json(origin_spec))
    proof=original_transfer.authenticate(Path(origin_spec),old)
    if proof!=origin_proof(old): raise ValueError('Original source replay differs')
    p,f,inventory,splits=alljets_data.load_prepared(prepared_root)
    if p['content_hash']!=proof['prepared'] or p['counts']!=COUNTS:
        raise ValueError('Prepared population differs')
    expected_teacher=teacher_files(old)[0]; checkpoint=old['imported']['teacher']['files']['checkpoint']
    validate_file_ref(checkpoint)
    root=fresh(output,[prepared_root,origin_spec,f['input_container'],checkpoint['path']])
    particles=sum(p['summaries'][role][coord]['particles'] for role in COUNTS for coord in ('D000','OFFLINE'))
    bound=int(particles*120+sum(COUNTS.values())*256+checkpoint['bytes']+2**30)
    if shutil.disk_usage(root.parent).free < bound+2**30: raise OSError('Insufficient export space including temporary headroom')
    root.mkdir(); shards=[]
    for role in COUNTS:
        refs={v['file_index']:v for v in p['assignments'][role]}; count=0
        reader=alljets_data.ParticleReader(f['input_container'],inventory,splits,role=role,include_offline=True)
        for index,group in itertools.groupby(reader,key=lambda j:j.file_index):
            jets=list(group); assigned=stage2.load_assignment(Path(prepared_root),refs[index])
            ids=np.array([np.frombuffer(bytes.fromhex(j.identity),np.uint8) for j in jets]); labels=np.array([j.label for j in jets],np.int64)
            if any(not np.array_equal(assigned[k],a) for k,a in (('identities',ids),('labels',labels))):
                raise ValueError('Original assignment identity order differs')
            values=dict(identities=ids,labels=labels,mapping=assigned['mapping'])
            for source in ('HLT','OFFLINE'):
                parts=[physical_columns(physical(getattr(j,source.lower()),p['conventions'],source.lower())) for j in jets]
                values[source+'_offsets']=np.r_[0,np.cumsum([len(a) for a in parts])].astype(np.int64)
                values[source+'_values']=np.concatenate(parts)
            if not np.array_equal(values['HLT_offsets'],assigned['offsets']): raise ValueError('Mapping support differs')
            name=f'{role}/{index:04d}.npz'; (root/role).mkdir(exist_ok=True)
            atomic_publish_bytes(root/name,deterministic_npz_bytes(values))
            shards.append(dict(file=file_ref(root/name,root=root),role=role,index=index,rows=len(jets),assignment=refs[index]))
            count+=len(jets)
            if len(shards)%20==0: print(f'EMBEDDING_BRIDGE export {role}: {count}/{COUNTS[role]}',flush=True)
        if count!=COUNTS[role]: raise ValueError('Export population incomplete')
    with open(checkpoint['path'],'rb') as src,(root/'U100.pt').open('xb') as dst:
        shutil.copyfileobj(src,dst,2**20);dst.flush();os.fsync(dst.fileno())
    if file_ref(root/'U100.pt',root=root)!=expected_teacher: raise ValueError('Copied teacher bytes differ')
    value=artifact('TRANSFER',parents=dict(origin=old['content_hash'],prepared=p['content_hash'],
        source=source_snapshot['content_hash'],recipe=recipe()['content_hash']),
        source_snapshot=source_snapshot,original_replay=proof,counts=COUNTS,conventions=p['conventions'],
        summaries=p['summaries'],assignments=p['assignments'],shards=shards,teacher=expected_teacher,
        teacher_record=old['imported']['teacher'],roles=['train','validation'],validation_usage='evaluation_only',
        payload_bytes=sum(s['file']['bytes'] for s in shards)+expected_teacher['bytes'],physical_columns='p4_charge_category_tracking_mm_validity')
    write_immutable_json(root/'manifest.json',value)
    return dict(directory=str(root),manifest_hash=value['content_hash'],payload_bytes=value['payload_bytes'])


class Transfer:
    def __init__(self,root,digest,origin):
        self.root=Path(root).resolve(strict=True); self.manifest=load_json(regular(self.root,'manifest.json'));m=self.manifest
        validate(m,'TRANSFER')
        source=validate_source_snapshot_payload(m['source_snapshot'])
        if (m['content_hash']!=digest or m['parents']['origin']!=origin['content_hash']
                or m['parents']['prepared']!=origin['anchor']['evidence']['prepared']
                or m['parents']['recipe']!=recipe()['content_hash'] or m['counts']!=COUNTS
                or m['original_replay']!=origin_proof(origin) or m['teacher']!=teacher_files(origin)[0]
                or m['teacher_record']!=origin['imported']['teacher']
                or m['roles']!=['train','validation'] or m['validation_usage']!='evaluation_only'
                or m['parents']['source']!=source or m['physical_columns']!='p4_charge_category_tracking_mm_validity'
                or any(s['role'] not in COUNTS or s['file']['path']!=f"{s['role']}/{s['index']:04d}.npz" for s in m['shards'])
                or m['payload_bytes']!=sum(s['file']['bytes'] for s in m['shards'])+m['teacher']['bytes']):
            raise ValueError('Physical transfer meaning or lineage differs')
        for role in COUNTS:
            rows=[s for s in m['shards'] if s['role']==role]
            if ([s['index'] for s in rows]!=[a['file_index'] for a in m['assignments'][role]]
                    or sum(s['rows'] for s in rows)!=COUNTS[role]
                    or [s['assignment'] for s in rows]!=m['assignments'][role]): raise ValueError('Transfer assignment coverage differs')
        names=[s['file']['path'] for s in m['shards']]
        if len(names)!=len(set(names)): raise ValueError('Duplicate transfer members')

    def path(self,ref):
        p=regular(self.root,ref['path']);validate_file_ref(ref,root=self.root);return p

    def shards(self,role):
        if role not in COUNTS: raise PermissionError('Final test unavailable')
        for s in self.manifest['shards']:
            if s['role']!=role:continue
            a=load_npz_arrays(self.path(s['file']));validate_population(a['identities'],a['labels'])
            if (len(a['labels'])!=s['rows'] or set(a)!={'identities','labels','mapping','HLT_offsets','HLT_values','OFFLINE_offsets','OFFLINE_values'}):
                raise ValueError('Transfer shard shape/inventory differs')
            from .regions import physical
            for source in ('HLT','OFFLINE'):
                off=a[source+'_offsets']; v=a[source+'_values']; physical(v)
                if off.dtype!=np.int64 or off.shape!=(s['rows']+1,) or off[0]!=0 or off[-1]!=len(v) or (np.diff(off)<1).any():
                    raise ValueError('Invalid physical offsets')
            if a['mapping'].dtype!=np.int32 or len(a['mapping'])!=len(a['HLT_values']): raise ValueError('Assignment cardinality differs')
            original=deterministic_npz_bytes(dict(offsets=a['HLT_offsets'],mapping=a['mapping'],identities=a['identities'],labels=a['labels']))
            if len(original)!=s['assignment']['bytes'] or hashlib.sha256(original).hexdigest()!=s['assignment']['sha256']:
                raise ValueError('Original assignment bytes were changed')
            yield s,a

    def verify_all(self):
        for role in COUNTS:
            for _ in self.shards(role): pass
        self.path(self.manifest['teacher'])
