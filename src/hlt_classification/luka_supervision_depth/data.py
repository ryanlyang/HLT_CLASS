"""Authenticated ordinary roles, including RAM-only deleted OFFLINE rebuild."""
from pathlib import Path
from hlt_classification.luka_trial_hybrid.worker import data as inherited_data
from hlt_classification.luka_fullsim import regular_cache,stage2
from hlt_classification.luka_fullsim.particles import physical,RawJet
from hlt_classification.luka_fullsim.preflight_cache import _block
from hlt_classification.jetclass2_delphes.cache import RamCache


def load(spec,role,coordinate,meta):
    if role not in ('train','validation'): raise PermissionError('No test role')
    if coordinate!='OFFLINE': return inherited_data(spec,role,coordinate,meta)
    root=Path(spec['anchor']['evidence']['prepared_root']); manifest,prepared,inventory,splits=meta
    if manifest['content_hash']!=spec['anchor']['evidence']['cache'] or prepared['content_hash']!=spec['anchor']['evidence']['prepared']:
        raise ValueError('Prepared metadata differs')
    present=[(root/r['path']).exists() or (root/r['path']).is_symlink() for r in manifest['blocks'][role]['OFFLINE']]
    budget=int(spec['memory_mb']*1024**2*.25)
    if all(present):
        result=regular_cache.load(root,role=role,coordinate='OFFLINE',max_ram_bytes=budget,authenticated=meta)
    else:
        if any(present): raise ValueError('Partially missing OFFLINE cache; preserve evidence')
        if 2*prepared['summaries'][role]['OFFLINE']['resident_bytes']+64*2**20>budget:
            raise MemoryError('Offline reconstruction exceeds bound')
        blocks=[]; meter=stage2.ViewMeter()
        for member in splits['memberships'][role]:
            rows=[]
            for jet in regular_cache.selected_file_rows(spec['anchor']['evidence']['raw_container'],inventory,member):
                values=stage2.build_inputs(physical(jet.offline,prepared['conventions'],'offline'),capacity=prepared['inputs']['capacity'])
                meter.add(jet,values); rows.append((RawJet(jet.identity,jet.label,jet.file_index,jet.entry,None,None),values))
            blocks.append(_block(member['file_index'],rows))
            print(f'DEPTH rebuild OFFLINE {role} files={len(blocks)} rows={meter.rows}',flush=True)
        if meter.report()!=prepared['summaries'][role]['OFFLINE']: raise ValueError('Rebuilt OFFLINE fingerprint differs')
        result=RamCache(blocks,role=role,coordinate_name='OFFLINE',foundation_sha256=prepared['content_hash'])
    if len(result)!=spec['counts'][role] or result.nbytes>budget: raise ValueError('OFFLINE population/memory differs')
    return result
