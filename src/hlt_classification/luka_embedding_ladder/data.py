"""Exact prepared views, with registered RAM reconstruction of deleted U caches."""
from pathlib import Path
import numpy as np
from hlt_classification.luka_fullsim import regular_cache, stage2
from hlt_classification.luka_fullsim.preflight_cache import _block, _values
from hlt_classification.luka_fullsim.particles import RawJet
from hlt_classification.jetclass2_delphes.cache import RamCache
from hlt_classification.luka_embedding_aux.targets import Bank
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from .contracts import VIEWS


def load_views(gate, role, *, authenticated=None):
    if role not in ('train', 'validation'):
        raise PermissionError('No final-test access')
    root = Path(gate['anchor']['evidence']['prepared_root'])
    meta = regular_cache.metadata(root) if authenticated is None else authenticated
    manifest, prepared, inventory, splits = meta
    if (manifest['content_hash'] != gate['anchor']['evidence']['cache']
            or prepared['content_hash'] != gate['anchor']['evidence']['prepared']):
        raise ValueError('Prepared view lineage differs')
    budget = int(gate['memory_mb']*1024**2*.30)
    if 2*sum(prepared['summaries'][role][c]['resident_bytes'] for c in VIEWS)+64*2**20 > budget:
        raise MemoryError('All-view load exceeds registered role memory bound')
    result, missing = {}, []
    for co in VIEWS:
        paths = [root/r['path'] for r in manifest['blocks'][role][co]]
        present = [p.exists() or p.is_symlink() for p in paths]
        if all(present):
            result[co] = regular_cache.load(root, role=role, coordinate=co,
                                            max_ram_bytes=budget, authenticated=meta)
        elif co in ('U000', 'U050') and not any(present):
            missing.append(co)
        else:
            raise ValueError('Partial/missing nonregistered cache; preserve evidence: '+role+'/'+co)
    if missing:
        refs = {r['file_index']: r for r in prepared['assignments'][role]}
        meters, blocks = {c:stage2.ViewMeter() for c in missing}, {c:[] for c in missing}
        for member in splits['memberships'][role]:
            index = member['file_index']
            assigned = stage2.load_assignment(root, refs[index])
            rows = {c:[] for c in missing}
            count = 0
            for jet in regular_cache.selected_file_rows(gate['anchor']['evidence']['raw_container'], inventory, member):
                values = _values(jet, assigned, count, prepared, missing)
                for c, value in values.items():
                    meters[c].add(jet, value)
                    rows[c].append((RawJet(jet.identity, jet.label, index, jet.entry, None, None), value))
                count += 1
            if count != refs[index]['rows']:
                raise ValueError('Rebuilt view coverage differs')
            for c in missing:
                blocks[c].append(_block(index, rows[c]))
            print(f'EMBED ladder rebuild role={role} files={len(blocks[missing[0]])} views={missing}', flush=True)
        for c in missing:
            if meters[c].report() != prepared['summaries'][role][c]:
                raise ValueError('Rebuilt input fingerprint differs: '+c)
            result[c] = RamCache(blocks[c], role=role, coordinate_name=c, foundation_sha256=prepared['content_hash'])
    anchor = result['D000']
    if sum(v.nbytes for v in result.values()) > budget:
        raise MemoryError('Resident views exceed memory bound')
    for co, view in result.items():
        if (len(view) != gate['counts'][role] or not np.array_equal(view.identities, anchor.identities)
                or not np.array_equal(view.labels, anchor.labels)):
            raise ValueError('Cross-view population differs: '+co)
    return result


def embedding_bank(gate, native_train):
    imported = gate['embedding_import']
    for key in ('bank', 'payload'):
        validate_file_ref(imported[key])
    bank = Bank(Path(imported['bank']['path']).parent, native_train,
                teacher_ref=gate['anchor']['evidence']['models']['OFFLINE']['checkpoint'])
    return bank


def check_join(bank, train):
    if (train.role != 'train' or train.coordinate_name not in VIEWS
            or bank.manifest['parents']['prepared'] != train.foundation_sha256
            or not np.array_equal(bank.identities, train.identities)
            or not np.array_equal(bank.labels, train.labels)):
        raise ValueError('Fixed OFFLINE target join differs')
