"""Replay missing native-U100 blocks in RAM against their original bytes.

Only absent payloads may be reconstructed. Existing corrupt payloads, changed
raw inputs or different feature fingerprints remain fatal. Nothing is written
to the historical preparation, raw dataset or campaign directory.
"""
import hashlib
from pathlib import Path

from hlt_classification.data.cache_contracts import deterministic_npz_bytes
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.particles import RawJet
from hlt_classification.luka_fullsim.preflight_cache import _block, _values


POLICY = 'missing_U100_only_raw_to_RAM_exact_historical_bytes_and_fingerprint_v1'


def load(anchor, meta, max_ram_bytes):
    role = 'validation'
    if role not in ('train', 'validation'):
        raise PermissionError('No final-test/unused U100 replay')
    if type(max_ram_bytes) is not int or max_ram_bytes <= 0:
        raise ValueError('Invalid U100 replay RAM budget')
    manifest, prepared, inventory, splits = meta
    summary = prepared['summaries'][role]['U100']
    # Includes per-file row arrays, the ragged block and deterministic NPZ buffers.
    if 4*summary['resident_bytes'] + 64*2**20 > max_ram_bytes:
        raise MemoryError('U100 replay exceeds explicit RAM budget')
    refs = manifest['blocks'][role]['U100']
    members = splits['memberships'][role]
    if len(refs) != len(members):
        raise ValueError('U100 replay block coverage differs')
    root = Path(anchor['prepared_root'])
    meter, blocks = regular_cache.s.ViewMeter(), []
    rebuilt = 0
    assignments = {r['file_index']: r for r in prepared['assignments'][role]}
    for ref, member in zip(refs, members):
        index = member['file_index']
        if ref['file_index'] != index or ref['path'] != f'inputs/{role}/U100/{index:04d}.npz':
            raise ValueError('U100 replay block path/order differs')
        path = root/ref['path']
        # A broken symlink is not an authorized missing historical payload.
        if path.is_symlink():
            raise ValueError('Redirected U100 payload is not supported')
        try:
            path.stat()
        except FileNotFoundError:
            rows = []
            assigned = regular_cache.s.load_assignment(root, assignments[index])
            for row, jet in enumerate(regular_cache.selected_file_rows(anchor['raw_container'], inventory, member)):
                value = _values(jet, assigned, row, prepared, ('U100',))['U100']
                meter.add(jet, value)
                light = RawJet(jet.identity, jet.label, index, jet.entry, None, None)
                rows.append((light, value))
            if len(rows) != ref['rows'] or not rows:
                raise ValueError('U100 replay row coverage differs')
            block = _block(index, rows)
            del rows
            payload = deterministic_npz_bytes({k: v for k, v in vars(block).items() if k != 'file_index'})
            if len(payload) != ref['bytes'] or hashlib.sha256(payload).hexdigest() != ref['sha256']:
                raise ValueError(f'U100 replay bytes differ from historical block: {role}/{index:04d}')
            del payload
            rebuilt += 1
            print(f'SELECTIVE_TEACHER U100_REPLAY role={role} file={index:04d} '
                  f'historical_bytes_verified=True persisted=False sha256={ref["sha256"]}', flush=True)
        else:
            # Do not catch corruption or permission failures and silently rebuild.
            block = regular_cache.read_block(root, ref, member, inventory, meter)
        blocks.append(block)
        if sum(b.nbytes for b in blocks) > max_ram_bytes:
            raise MemoryError('U100 resident cache exceeds budget')
    if meter.report() != summary:
        raise ValueError('U100 replay fingerprint differs from historical prepared inputs')
    result = RamCache(blocks, role=role, coordinate_name='U100',
                      foundation_sha256=prepared['content_hash'])
    if len(result) != prepared['counts'][role] or result.nbytes > max_ram_bytes:
        raise ValueError('U100 replay population/budget differs')
    print(f'SELECTIVE_TEACHER U100_REPLAY role={role} rebuilt_files={rebuilt}/{len(blocks)} '
          f'input_sha256={summary["input_sha256"]} persisted=False', flush=True)
    return result
