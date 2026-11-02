"""Compact matched U100 p4 targets; no persistent quadratic pair bank."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, atomic_publish_bytes,
    deterministic_npz_bytes, write_immutable_json)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_cache import _values, _block
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_u100_embedding.targets import Bank as EmbeddingBank
from .contracts import artifact, validate, recipe
from . import geometry as g


def metadata(spec):
    anchor = spec['anchor']['evidence']
    value = regular_cache.metadata(anchor['prepared_root'])
    if value[0]['content_hash'] != anchor['cache'] or value[1]['content_hash'] != anchor['prepared']:
        raise ValueError('Original prepared input lineage differs')
    return value


def load_view(spec, role, coordinate, meta=None):
    if role not in ('train', 'validation') or coordinate not in ('D000', 'U100'):
        raise PermissionError('Only registered HLT/U100 ordinary views allowed')
    if coordinate == 'U100' and role != 'train':
        raise PermissionError('Physical correction targets are TRAIN-only')
    manifest, prepared, inventory, splits = meta or metadata(spec)
    anchor = spec['anchor']['evidence']; root = Path(anchor['prepared_root'])
    budget = int(spec['memory_mb']*1024**2*.22)
    if 4*prepared['summaries'][role][coordinate]['resident_bytes']+64*2**20 > budget:
        raise MemoryError('Authenticated input replay exceeds RAM budget')
    blocks = []; meter = regular_cache.s.ViewMeter()
    assignments = {r['file_index']: r for r in prepared['assignments'][role]}
    for ref, member in zip(manifest['blocks'][role][coordinate], splits['memberships'][role]):
        path = root/ref['path']
        if path.is_symlink():
            raise ValueError('Redirected historical inputs forbidden')
        if path.exists():
            block = regular_cache.read_block(root, ref, member, inventory, meter)
        else:
            assigned = regular_cache.s.load_assignment(root, assignments[member['file_index']])
            rows = []
            for i, jet in enumerate(regular_cache.selected_file_rows(anchor['raw_container'], inventory, member)):
                value = _values(jet, assigned, i, prepared, (coordinate,))[coordinate]
                meter.add(jet, value)
                # Drop raw particle arrays before accumulating file-block inputs.
                from hlt_classification.luka_fullsim.particles import RawJet
                rows.append((RawJet(jet.identity, jet.label, jet.file_index, jet.entry, None, None), value))
            block = _block(member['file_index'], rows)
            raw = deterministic_npz_bytes({k: v for k, v in vars(block).items() if k != 'file_index'})
            if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('Missing input reconstruction differs from historical bytes')
            del raw, rows
        blocks.append(block)
        print(f'CORRECTION cache={role}/{coordinate} files={len(blocks)}', flush=True)
    if meter.report() != prepared['summaries'][role][coordinate]:
        raise ValueError('Original input fingerprint differs')
    result = RamCache(blocks, role=role, coordinate_name=coordinate, foundation_sha256=prepared['content_hash'])
    if len(result) != spec['counts'][role] or result.nbytes > budget:
        raise ValueError('Input population/RAM bound differs')
    return result


def embedding_bank(spec, train):
    ref = spec['reference']['u100_manifest']; validate_file_ref(ref)
    return EmbeddingBank(Path(ref['path']).parent, train, teacher_record=spec['imported']['teacher'])


def pack(train, u100, assigned):
    if (train.role != 'train' or u100.role != 'train' or train.coordinate_name != 'D000'
            or u100.coordinate_name != 'U100' or train.foundation_sha256 != u100.foundation_sha256
            or not np.array_equal(train.identities, u100.identities) or not np.array_equal(train.labels, u100.labels)
            or len(assigned) != len(train.blocks) or len(u100.blocks) != len(train.blocks)):
        raise ValueError('TRAIN U100/HLT/assignment lineage or ordered population differs')
    offsets, vectors, matched = [0], [], []
    for h, u, a in zip(train.blocks, u100.blocks, assigned):
        if (h.file_index != u.file_index or not np.array_equal(h.offsets, u.offsets)
                or not np.array_equal(h.offsets, a['offsets'])
                or not np.array_equal(h.identities, a['identities']) or not np.array_equal(h.labels, a['labels'])
                or np.any(a['mapping'] < -1)):
            raise ValueError('Particle slot/mapping identity join differs')
        for lo, hi in zip(a['offsets'][:-1], a['offsets'][1:]):
            m = a['mapping'][lo:hi]; m = m[m >= 0]
            if len(np.unique(m)) != len(m):
                raise ValueError('Assignment is not one-to-one')
        valid = a['mapping'] >= 0
        if not np.array_equal(h.vectors[~valid], u.vectors[~valid]):
            raise ValueError('Unmatched U100 slots must retain HLT p4')
        vectors.append(u.vectors); matched.append(valid)
        offsets.extend((h.offsets[1:]+offsets[-1]).tolist())
    return dict(offsets=np.asarray(offsets, np.int64), vectors=np.concatenate(vectors),
                matched=np.concatenate(matched), identities=train.identities, labels=train.labels)


class ParticleBank:
    def __init__(self, arrays, manifest):
        self.manifest = manifest
        for key, value in arrays.items():
            setattr(self, key, value)

    def batch(self, indices, length):
        vectors = np.zeros((len(indices), length, 4), np.float32)
        matched = np.zeros((len(indices), length), np.bool_)
        for row, index in enumerate(indices):
            lo, hi = self.offsets[index:index+2]
            if hi-lo > length:
                raise ValueError('Target support exceeds input padding')
            vectors[row, :hi-lo] = self.vectors[lo:hi]
            matched[row, :hi-lo] = self.matched[lo:hi]
        return vectors, matched

    @property
    def counts(self):
        return np.array([self.matched[l:h].sum() for l, h in zip(self.offsets[:-1], self.offsets[1:])])


def statistics(train, bank, device):
    if (train.role != 'train' or train.coordinate_name != 'D000'
            or not np.array_equal(train.identities, bank.identities) or not np.array_equal(train.labels, bank.labels)):
        raise PermissionError('Only aligned TRAIN HLT rows may fit correction statistics')
    sums = {k: torch.zeros(4, dtype=torch.float64, device=device) for k in ('particle', 'pair', 'native', 'native2')}
    counts = dict(particle=0, pair=0, native=0); audited = 0; matched_particles = matched_pairs = 0
    def accumulate(name, values, batch, size):
        number = torch.bincount(batch, minlength=size)
        total = values.new_zeros((size, 4)).index_add(0, batch, values)
        use = number > 0
        sums[name] += (total[use]/number[use, None]).sum(0)
        return int(use.sum())
    with torch.no_grad():
        for start in range(0, len(train), 128):
            ix = np.arange(start, min(start+128, len(train))); raw = train.batch(ix)
            target, match = bank.batch(ix, raw['mask'].shape[-1])
            p = torch.from_numpy(raw['vectors'].transpose(0, 2, 1)).to(device, torch.float64)
            u = torch.from_numpy(target).to(device, torch.float64)
            mask = torch.from_numpy(raw['mask'][:, 0]).to(device)
            matched = torch.from_numpy(match).to(device)
            audited += g.physical_audit(p[mask])+g.physical_audit(u[mask])
            bb, pp = torch.where(matched); matched_particles += len(bb)
            residual = g.difference(g.coordinates(u[bb, pp]), g.coordinates(p[bb, pp]))
            counts['particle'] += accumulate('particle', residual.square(), bb, len(ix))
            b, i, j = g.pair_indices(mask)
            native = g.pair_features(p[b, i], p[b, j])
            counts['native'] += accumulate('native', native, b, len(ix))
            accumulate('native2', native.square(), b, len(ix))
            ok = matched[b, i] & matched[b, j]; b, i, j = b[ok], i[ok], j[ok]
            matched_pairs += len(b)
            delta = g.pair_features(u[b, i], u[b, j])-native[ok]
            counts['pair'] += accumulate('pair', delta.square(), b, len(ix))
            if start % (128*100) == 0:
                print(f'CORRECTION TRAIN statistics {start+len(ix)}/{len(train)}', flush=True)
    if not all(counts.values()):
        raise ValueError('No eligible TRAIN population for required frozen statistics')
    mean = sums['native']/counts['native']
    pscale = torch.maximum((sums['particle']/counts['particle']).sqrt(), mean.new_tensor([1e-3, 1e-4, 1e-4, 1e-4]))
    result = dict(particle_scale=pscale.tolist(), pair_scale=(sums['pair']/counts['pair']).sqrt().clamp_min(1e-3).tolist(),
        pair_mean=mean.tolist(), pair_std=(sums['native2']/counts['native']-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3).tolist(),
        eligible_jets=counts, matched_particles=matched_particles, matched_pairs=matched_pairs,
        residual_second_moments={k:(sums[k]/counts[k]).tolist() for k in ('particle','pair')},
        native_second_moments=(sums['native2']/counts['native']).tolist(),
        scale_floors=dict(particle=[1e-3,1e-4,1e-4,1e-4], pair=1e-3, native_std=1e-3),
        roundoff_negative_m2=audited, role='train', weighting='equal_eligible_jets', residual_center='zero',
        native_population='all_real_HLT_unique_pairs', new_loss_population='all_jets_including_zero_eligible')
    return result


def export(spec, train, *, device):
    meta = metadata(spec); prepared = meta[1]; root = Path(spec['root'])/'setup/particles'
    root.mkdir(exist_ok=False)
    u100 = load_view(spec, 'train', 'U100', meta)
    refs = prepared['assignments']['train']; base = Path(spec['anchor']['evidence']['prepared_root'])
    assigned = [regular_cache.s.load_assignment(base, r) for r in refs]
    arrays = pack(train, u100, assigned)
    del u100, assigned
    bank = ParticleBank(arrays, None)
    stats = statistics(train, bank, device)
    payload = deterministic_npz_bytes(arrays)
    # Actual target size plus an explicit reserve for selected models and reports.
    if len(payload)+2*2**30 > spec['storage_budget_bytes']:
        raise OSError('Actual ragged targets exceed reviewed storage allowance; no automatic deletion')
    atomic_publish_bytes(root/'targets.npz', payload)
    manifest = artifact('PARTICLE_BANK', parents=dict(spec=spec['content_hash'], recipe=recipe()['content_hash'],
        prepared=train.foundation_sha256), role='train', coordinate='U100', rows=len(train), statistics=stats,
        assignment_refs=[file_ref(base/r['path']) for r in refs], payload=file_ref(root/'targets.npz', root=root),
        targets='physical_p4_on_authenticated_matched_HLT_slots', pair_payload_stored=False)
    write_immutable_json(root/'manifest.json', manifest)
    return ParticleBank(arrays, manifest)


def particle_bank(spec, train):
    root = Path(spec['root'])/'setup/particles'; m = load_json(root/'manifest.json')
    validate(m, 'PARTICLE_BANK', dict(spec=spec['content_hash'], recipe=recipe()['content_hash'], prepared=train.foundation_sha256))
    if (train.role != 'train' or train.coordinate_name != 'D000' or m['role'] != 'train'
            or m['coordinate'] != 'U100' or m['rows'] != len(train) or m['payload']['path'] != 'targets.npz'
            or m['pair_payload_stored'] is not False):
        raise PermissionError('Wrong correction bank semantics or population')
    validate_file_ref(m['payload'], root=root)
    for ref in m['assignment_refs']:
        validate_file_ref(ref)
    a = load_npz_arrays(root/'targets.npz')
    if (set(a) != {'offsets', 'vectors', 'matched', 'identities', 'labels'}
            or a['offsets'].dtype != np.int64 or a['offsets'].shape != (len(train)+1,)
            or a['offsets'][0] != 0 or a['offsets'][-1] != len(a['vectors'])
            or np.any(np.diff(a['offsets']) <= 0) or np.any(np.diff(a['offsets']) > 512)
            or a['vectors'].shape != (a['offsets'][-1], 4) or a['vectors'].dtype != np.float32
            or a['matched'].shape != (len(a['vectors']),) or a['matched'].dtype != np.bool_
            or not np.isfinite(a['vectors']).all() or not np.array_equal(a['identities'], train.identities)
            or not np.array_equal(a['labels'], train.labels)):
        raise ValueError('Invalid aligned ragged particle bank')
    from hlt_classification.luka_u100_multiview.views import lengths
    if not np.array_equal(np.diff(a['offsets']), lengths(train)):
        raise ValueError('Particle bank/HLT support differs')
    return ParticleBank(a, m)
