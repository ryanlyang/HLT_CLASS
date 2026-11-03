"""Allowlisted relocation without altering or following historical paths."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from hlt_classification.luka_u100_pair_transfer import bundle as transfer


def unpack(archive, output, expected_sha256):
    """Verify, then copy regular members into a fresh directory, never extractall."""
    archive = transfer.regular(Path(archive).resolve(strict=True))
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Extraction requires a new directory without redirected parents')
    plan = transfer.verify(archive, expected_sha256)
    if shutil.disk_usage(output.parent).free < plan['payload_bytes']+transfer.HEADROOM:
        raise OSError('Extraction needs actual payload bytes plus one GiB headroom')
    output.mkdir()
    expected = {r['path']: r for r in plan['files']}
    seen = set()
    with tarfile.open(archive, 'r|') as tar:
        first = tar.next()
        if (first is None or first.name != 'manifest.json' or not first.isfile()
                or first.issparse() or first.size > transfer.MAX_JSON
                or json.load(tar.extractfile(first)) != plan):
            raise ValueError('Archive changed after verification')
        while True:
            info = tar.next()
            if info is None:
                break
            name = transfer.relative(info.name)
            if (name not in expected or name in seen or not info.isfile() or info.issparse()
                    or info.size != expected[name]['bytes']):
                raise ValueError('Archive member differs during extraction')
            path = output/name
            path.parent.mkdir(parents=True, exist_ok=True)
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('Redirected extraction destination')
            digest = hashlib.sha256(); count = 0
            with path.open('xb') as target, tar.extractfile(info) as source:
                for block in iter(lambda: source.read(2**20), b''):
                    target.write(block); digest.update(block); count += len(block)
                target.flush(); os.fsync(target.fileno())
            if count != expected[name]['bytes'] or digest.hexdigest() != expected[name]['sha256']:
                raise ValueError('Extracted file differs; preserve incomplete output')
            seen.add(name)
    if seen != set(expected) or transfer.sha256(archive) != expected_sha256:
        raise ValueError('Archive changed or extraction incomplete')
    (output/'manifest.json').write_bytes(transfer.canonical(plan)+b'\n')
    receipt = transfer.seal(dict(contract='LUKA_U100_PAIR_HYDRA_EXTRACTION/v1', schema_version=1,
        archive_sha256=expected_sha256, manifest_hash=plan['content_hash'], files=len(seen)))
    partial = output/'extraction_receipt.partial'
    with partial.open('xb') as handle:
        handle.write(transfer.canonical(receipt)+b'\n'); handle.flush(); os.fsync(handle.fileno())
    os.link(partial, output/'extraction_receipt.json'); partial.unlink()
    return receipt


class Bundle:
    def __init__(self, root):
        from .contracts import ARCHIVE_SHA256, MANIFEST_HASH
        self.root = Path(root).resolve(strict=True)
        self.manifest = transfer.validate_manifest(transfer.read_json(self.root/'manifest.json'))
        receipt = transfer.checked(transfer.read_json(self.root/'extraction_receipt.json'))
        expected = transfer.seal(dict(contract='LUKA_U100_PAIR_HYDRA_EXTRACTION/v1', schema_version=1,
            archive_sha256=ARCHIVE_SHA256, manifest_hash=MANIFEST_HASH, files=len(self.manifest['files'])))
        if self.manifest['content_hash'] != MANIFEST_HASH or receipt != expected:
            raise ValueError('Only the verified campaign transfer is registered')
        self.rows = {r['path']: r for r in self.manifest['files']}

    def path(self, name):
        name = transfer.relative(name)
        if name not in self.rows:
            raise PermissionError('Not a registered transferred input: '+name)
        row = self.rows[name]; path = transfer.regular(self.root/name)
        if path.stat().st_size != row['bytes'] or transfer.sha256(path) != row['sha256']:
            raise ValueError('Transferred input changed: '+name)
        return path

    def json(self, name):
        return transfer.checked(transfer.read_json(self.path(name)))

    def verify_all(self):
        for name in self.rows:
            self.path(name)


def metadata(spec):
    from hlt_classification.luka_fullsim import alljets_data
    bundle = Bundle(spec['bundle_root'])
    m = bundle.json('prepared/cache_manifest.json'); p = bundle.json('prepared/prepared.json')
    bundle.path('foundation/inventory.json'); bundle.path('foundation/split_manifest.json')
    bundle.path('foundation/foundation.json')
    f, inventory, splits = alljets_data.load_foundation(bundle.root/'foundation')
    origin = bundle.manifest['origin']
    if (m['content_hash'] != origin['cache'] or p['content_hash'] != origin['prepared']
            or p['parents']['foundation'] != f['content_hash'] or p['counts'] != spec['counts']):
        raise ValueError('Transferred prepared/foundation lineage differs')
    # Original source replay and pinned manifest bind the untouched full metadata.
    # Validation assignments and other richer views are intentionally not opened.
    return m, p, inventory, splits


def load_view(spec, role, coordinate, meta=None):
    from hlt_classification.luka_fullsim import regular_cache
    from hlt_classification.jetclass2_delphes.cache import RamCache
    if (role, coordinate) not in transfer.VIEWS:
        raise PermissionError('Only transferred TRAIN D000/U100 and validation D000 are admitted')
    bundle = Bundle(spec['bundle_root']); m, p, inventory, splits = meta or metadata(spec)
    refs = m['blocks'][role][coordinate]; members = splits['memberships'][role]
    if len(refs) != len(members):
        raise ValueError('Input membership coverage differs')
    budget = int(spec['memory_mb']*1024**2*.22)
    if 4*p['summaries'][role][coordinate]['resident_bytes']+64*2**20 > budget:
        raise MemoryError('Prepared replay exceeds RAM budget')
    blocks = []; meter = regular_cache.s.ViewMeter()
    for ref, member in zip(refs, members):
        bundle.path('prepared/'+ref['path'])
        blocks.append(regular_cache.read_block(bundle.root/'prepared', ref, member, inventory, meter))
    if meter.report() != p['summaries'][role][coordinate]:
        raise ValueError('Transferred input fingerprint differs')
    cache = RamCache(blocks, role=role, coordinate_name=coordinate, foundation_sha256=p['content_hash'])
    if len(cache) != spec['counts'][role] or cache.nbytes > budget:
        raise ValueError('Transferred population/RAM differs')
    print(f'HYDRA loaded {role}/{coordinate}: {len(cache)} authenticated jets', flush=True)
    return cache


def embedding_bank(spec, train):
    from hlt_classification.luka_u100_embedding.targets import Bank
    bundle = Bundle(spec['bundle_root'])
    bundle.path('targets/u100/manifest.json'); bundle.path('targets/u100/targets.npz')
    return Bank(bundle.root/'targets/u100', train, teacher_record=spec['imported']['teacher'])


def export(spec, train, *, device):
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes, write_immutable_json
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_fullsim import regular_cache
    from hlt_classification.luka_u100_pair_correction import targets as old
    from hlt_classification.luka_u100_pair_correction.contracts import artifact, recipe
    bundle = Bundle(spec['bundle_root']); meta = metadata(spec)
    u100 = load_view(spec, 'train', 'U100', meta)
    refs = meta[1]['assignments']['train']
    paths = [bundle.path('prepared/'+r['path']) for r in refs]
    assigned = [regular_cache.s.load_assignment(bundle.root/'prepared', r) for r in refs]
    arrays = old.pack(train, u100, assigned)
    del u100, assigned
    bank = old.ParticleBank(arrays, None); stats = old.statistics(train, bank, device)
    payload = deterministic_npz_bytes(arrays)
    if len(payload)+2*2**30 > spec['storage_budget_bytes']:
        raise OSError('Ragged targets exceed registered budget')
    root = Path(spec['root'])/'setup/particles'; root.mkdir(exist_ok=False)
    atomic_publish_bytes(root/'targets.npz', payload)
    manifest = artifact('PARTICLE_BANK', parents=dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
        prepared=train.foundation_sha256), role='train', coordinate='U100', rows=len(train), statistics=stats,
        assignment_refs=[file_ref(path) for path in paths], payload=file_ref(root/'targets.npz', root=root),
        targets='physical_p4_on_authenticated_matched_HLT_slots', pair_payload_stored=False)
    write_immutable_json(root/'manifest.json', manifest)
    return old.ParticleBank(arrays, manifest)


def particle_bank(spec, train):
    from hlt_classification.luka_u100_pair_correction.targets import particle_bank as old
    return old(spec, train)
