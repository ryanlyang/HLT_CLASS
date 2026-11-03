"""Small oracle-only delta and authenticated, slot-preserving input joins."""
from pathlib import Path
import os
import shutil
import numpy as np
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, validate_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_u100_pair_transfer import bundle as transfer
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, load_view, metadata, embedding_bank
from .contracts import artifact, validate, COUNTS, MANIFEST_HASH


def expected_files(old, cache):
    anchor = old['anchor']['evidence']
    if cache['content_hash'] != anchor['cache']: raise ValueError('Oracle cache lineage differs')
    rows = [dict(anchor['models']['M0HLT']['checkpoint'], path='M0HLT.pt')]
    for r in cache['blocks']['validation']['U100']:
        if r['path'] != f'inputs/validation/U100/{r["file_index"]:04d}.npz': raise ValueError('Unexpected oracle shard')
        rows.append(dict(path='prepared/'+r['path'], bytes=r['bytes'], sha256=r['sha256']))
    return sorted(rows, key=lambda r:r['path'])


def export(original_spec, output):
    path = transfer.regular(original_spec); old = load_json(path)
    if old['contract'] != 'LUKA_U100_PAIR_CORRECTION_SPEC/v1' or old['counts'] != COUNTS:
        raise ValueError('Use the original pair campaign')
    proof = transfer.authenticate(path, old)
    anchor = old['anchor']['evidence']; prepared = Path(anchor['prepared_root'])
    cache = load_json(prepared/'cache_manifest.json')
    validate_content_hash(cache, expected_contract='LUKA_FULLSIM_ALLJETS_CACHE/v1')
    rows = expected_files(old, cache)
    sources = {'M0HLT.pt':Path(anchor['models']['M0HLT']['checkpoint']['path'])}
    sources.update({r['path']:prepared/r['path'].removeprefix('prepared/') for r in rows if r['path'] != 'M0HLT.pt'})
    for row in rows:
        actual = file_ref(transfer.regular(sources[row['path']]))
        if any(actual[k] != row[k] for k in ('bytes','sha256')): raise ValueError('Original oracle artifact changed')
    total = sum(r['bytes'] for r in rows)
    if total > 2**30: raise ValueError('Oracle delta unexpectedly exceeds one GiB')
    output = Path(output).absolute()
    if (output.exists() or any(p.is_symlink() for p in (output,*output.parents))
            or any(output.is_relative_to(p.parent) or p.is_relative_to(output) for p in sources.values())):
        raise ValueError('Use a fresh separate delta directory')
    if shutil.disk_usage(output.parent).free < total+2**30: raise OSError('Need delta bytes plus one GiB headroom')
    output.mkdir()
    for row in rows:
        dest = output/row['path']; dest.parent.mkdir(parents=True, exist_ok=True)
        with sources[row['path']].open('rb') as source, dest.open('xb') as handle:
            shutil.copyfileobj(source, handle, 2**20); handle.flush(); os.fsync(handle.fileno())
        validate_file_ref(row, root=output)
    result = artifact('TRANSFER', parents=dict(origin=old['content_hash'], transfer=MANIFEST_HASH),
        original_replay=proof, files=rows, payload_bytes=total, roles=['oracle_initialization','oracle_validation_U100'])
    write_immutable_json(output/'manifest.json', result)
    return dict(directory=str(output), manifest_hash=result['content_hash'], payload_bytes=total, files=len(rows))


class Delta:
    def __init__(self, root, expected_hash, bundle):
        self.root = Path(root).resolve(strict=True)
        value = load_json(transfer.regular(self.root/'manifest.json')); old = bundle.json('origin/campaign_spec.json')
        validate(value, 'TRANSFER', dict(origin=old['content_hash'], transfer=MANIFEST_HASH))
        proof = value['original_replay']; anchor = old['anchor']['evidence']
        expected_proof = dict(campaign=old['content_hash'], reference=old['reference']['content_hash'],
            prepared=anchor['prepared'], cache=anchor['cache'], source_commit=old['source_commit'],
            control_job='21993056', counts=COUNTS, environment=old['reference']['environment'], original_source_replay=True)
        if (value['content_hash'] != expected_hash or proof != expected_proof
                or value['files'] != expected_files(old,bundle.json('prepared/cache_manifest.json'))
                or value['payload_bytes'] != sum(r['bytes'] for r in value['files'])
                or not 0 < value['payload_bytes'] <= 2**30
                or value['roles'] != ['oracle_initialization','oracle_validation_U100']):
            raise ValueError('Wrong oracle-only input delta')
        self.manifest = value; self.rows = {r['path']:r for r in value['files']}

    def path(self, name):
        transfer.relative(name)
        if name not in self.rows: raise PermissionError('Not an oracle delta member')
        result = transfer.regular(self.root/name); validate_file_ref(self.rows[name], root=self.root)
        return result

    def verify_all(self):
        for name in self.rows: self.path(name)


def delta(spec):
    return Delta(spec['delta_root'],spec['delta_hash'],Bundle(spec['bundle_root']))


def oracle_validation(spec):
    from hlt_classification.luka_fullsim import regular_cache
    from hlt_classification.jetclass2_delphes.cache import RamCache
    d = delta(spec); m,p,inventory,splits = metadata(spec)
    refs = m['blocks']['validation']['U100']; members = splits['memberships']['validation']
    if len(refs) != len(members): raise ValueError('Oracle validation coverage differs')
    budget = int(spec['memory_mb']*1024**2*.22)
    if 4*p['summaries']['validation']['U100']['resident_bytes']+64*2**20 > budget:
        raise MemoryError('Oracle validation exceeds RAM budget')
    meter = regular_cache.s.ViewMeter(); blocks = []
    for ref, member in zip(refs,members):
        d.path('prepared/'+ref['path'])
        blocks.append(regular_cache.read_block(d.root/'prepared',ref,member,inventory,meter))
    if meter.report() != p['summaries']['validation']['U100']: raise ValueError('Oracle fingerprint differs')
    result = RamCache(blocks,role='validation',coordinate_name='U100',foundation_sha256=p['content_hash'])
    if len(result) != spec['counts']['validation'] or result.nbytes > budget: raise ValueError('Oracle population differs')
    return result


def aligned(hlt,u100):
    if (hlt.coordinate_name != 'D000' or u100.coordinate_name != 'U100' or hlt.role != u100.role
            or hlt.foundation_sha256 != u100.foundation_sha256 or len(hlt.blocks) != len(u100.blocks)
            or not np.array_equal(hlt.identities,u100.identities) or not np.array_equal(hlt.labels,u100.labels)):
        raise ValueError('HLT/U100 population differs')
    for h,u in zip(hlt.blocks,u100.blocks):
        if h.file_index != u.file_index or not np.array_equal(h.offsets,u.offsets): raise ValueError('HLT/U100 slots differ')


class OracleCache:
    """Explicit oracle-only evaluation wrapper; never passed to a student."""
    def __init__(self,hlt,u100): aligned(hlt,u100); self.hlt,self.u100 = hlt,u100
    def __len__(self): return len(self.hlt)
    def __getattr__(self,key): return getattr(self.hlt,key)
    def batch(self,indices):
        raw = self.hlt.batch(indices); other = self.u100.batch(indices)
        if not np.array_equal(raw['mask'],other['mask']): raise ValueError('Oracle padding differs')
        return dict(raw,u_features=other['features'],u_vectors=other['vectors'])
