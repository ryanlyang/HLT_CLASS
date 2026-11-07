"""Minimal native OFFLINE delta; historical paths remain evidence, never destinations."""
from pathlib import Path
import hashlib
import shutil
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.luka_u100_pair_transfer import bundle as old_transfer
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle, metadata
from hlt_classification.luka_u100_multiview.offline_cache import load as offline_load
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.jetclass2_delphes.cache import RamCache
from .contracts import COUNTS, artifact, validate


def export(origin_spec, output, source_snapshot):
    original = old_transfer.checked(load_json(origin_spec))
    proof = old_transfer.authenticate(Path(origin_spec).resolve(), original)
    anchor = original['anchor']['evidence']; meta = regular_cache.metadata(anchor['prepared_root'])
    if (proof['counts'] != COUNTS or proof['prepared'] != meta[1]['content_hash']
            or proof['cache'] != meta[0]['content_hash']):
        raise ValueError('Original native preparation differs')
    output = Path(output).absolute()
    protected = [Path(anchor[k]).resolve() for k in ('prepared_root', 'raw_container')]+[Path(original['root']).resolve()]
    if (output.exists() or not output.parent.is_dir() or any(p.is_symlink() for p in (output, *output.parents))
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected)):
        raise ValueError('Use a new separate native OFFLINE export directory')
    refs = [r for role in COUNTS for r in meta[0]['blocks'][role]['OFFLINE']]
    total = sum(r['bytes'] for r in refs)
    if shutil.disk_usage(output.parent).free < total+2**30:
        raise OSError('Insufficient export disk space')
    output.mkdir(); files = []
    for role in ('train', 'validation'):
        cache = offline_load(anchor, role, meta, 16*2**30)
        for block, ref in zip(cache.blocks, meta[0]['blocks'][role]['OFFLINE']):
            payload = deterministic_npz_bytes({k: v for k, v in vars(block).items() if k != 'file_index'})
            if len(payload) != ref['bytes'] or hashlib.sha256(payload).hexdigest() != ref['sha256']:
                raise ValueError('Export differs from historical native bytes')
            destination = output/old_transfer.relative(ref['path']); destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_publish_bytes(destination, payload); files.append(file_ref(destination, root=output))
        print(f'TEACHER_SWEEP exported OFFLINE {role}: {len(cache)} jets', flush=True)
        del cache
    value = artifact('TRANSFER', parents=dict(prepared=proof['prepared'], cache=proof['cache'], source=source_snapshot['content_hash']),
        origin=proof, source_snapshot=source_snapshot, files=files, payload_bytes=total,
        roles=['train', 'validation'], coordinate='OFFLINE', validation_use='teacher_selection_only')
    write_immutable_json(output/'manifest.json', value)
    return dict(directory=str(output), manifest_hash=value['content_hash'], payload_bytes=total)


class Delta:
    def __init__(self, root, expected_hash, bundle_root):
        self.root = Path(root).resolve(strict=True)
        old_transfer.regular(self.root/'manifest.json')
        self.manifest = load_json(self.root/'manifest.json'); m = self.manifest
        validate(m, 'TRANSFER')
        self.bundle = Bundle(bundle_root)
        origin = self.bundle.manifest['origin']
        cache = self.bundle.json('prepared/cache_manifest.json')
        expected = [dict(path=r['path'], sha256=r['sha256'], bytes=r['bytes'])
            for role in COUNTS for r in cache['blocks'][role]['OFFLINE']]
        if (m['content_hash'] != expected_hash or m['origin'] != origin or m['files'] != expected
                or m['parents'] != dict(prepared=origin['prepared'], cache=origin['cache'], source=m['source_snapshot']['content_hash'])
                or m['roles'] != ['train', 'validation'] or m['coordinate'] != 'OFFLINE'
                or m['validation_use'] != 'teacher_selection_only' or m['payload_bytes'] != sum(r['bytes'] for r in expected)):
            raise ValueError('Native OFFLINE delta differs from pinned original cache inventory')
        self.rows = {r['path']: r for r in m['files']}
        if len(self.rows) != len(m['files']):
            raise ValueError('Duplicate transfer member')

    def path(self, name):
        name = old_transfer.relative(name)
        if name not in self.rows:
            raise PermissionError('Unregistered OFFLINE transfer member')
        path = old_transfer.regular(self.root/name); validate_file_ref(self.rows[name], root=self.root)
        return path

    def verify_all(self):
        for i, name in enumerate(self.rows, 1):
            self.path(name)
            if i % 200 == 0: print(f'TEACHER_SWEEP verified {i}/{len(self.rows)} native OFFLINE files', flush=True)
        return self.manifest

    def load(self, spec, role):
        if role not in ('train', 'validation'):
            raise PermissionError('Final test is sealed')
        m, p, inventory, splits = metadata(spec)
        refs = m['blocks'][role]['OFFLINE']; members = splits['memberships'][role]
        if len(refs) != len(members): raise ValueError('OFFLINE membership coverage differs')
        budget = int(spec['memory_mb']*1024**2*.3)
        if 4*p['summaries'][role]['OFFLINE']['resident_bytes']+64*2**20 > budget:
            raise MemoryError('Native OFFLINE replay exceeds RAM budget')
        blocks = []; meter = regular_cache.s.ViewMeter()
        for ref, member in zip(refs, members):
            self.path(ref['path']); blocks.append(regular_cache.read_block(self.root, ref, member, inventory, meter))
        if meter.report() != p['summaries'][role]['OFFLINE']:
            raise ValueError('Native OFFLINE fingerprint differs')
        result = RamCache(blocks, role=role, coordinate_name='OFFLINE', foundation_sha256=p['content_hash'])
        if len(result) != spec['counts'][role] or result.nbytes > budget:
            raise ValueError('Native OFFLINE population/budget differs')
        print(f'TEACHER_SWEEP loaded OFFLINE {role}: {len(result)} authenticated jets', flush=True)
        return result
