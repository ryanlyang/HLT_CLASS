"""A bounded new campaign and an explicit remaining reservation for the live tail."""
from pathlib import Path
import math
import shutil
import torch
from hlt_classification.data.cache_contracts import load_json
from . import contracts as c,model


def usage(root):
    root=Path(root)
    if not root.exists():return 0
    total=0
    for path in root.rglob('*'):
        if path.is_symlink():raise PermissionError('Campaign payload symlink')
        if path.is_file():total+=path.stat().st_size
    return total


def other_remaining(spec):
    from hlt_classification.luka_sixclass_tail_hydra.campaign import committed_bytes
    from hlt_classification.luka_sixclass_tail_hydra.contracts import validate
    p=Path(spec['protected_campaign']['root'])/'campaign_spec.json';other=load_json(p);validate(other,'SPEC')
    if other['content_hash']!=spec['protected_campaign']['spec_sha256'] or other['root']!=str(p.parent):
        raise ValueError('Protected ladder reservation differs')
    return max(0,c.OTHER_RESERVATION-committed_bytes(p.parent))


def check(spec,*,declared_gib=None):
    root=Path(spec['root']);used=usage(root)
    if used>c.RESULT_BYTES:raise OSError('Own campaign exceeded its result allowance; preserve outputs')
    needed=max(0,c.RESULT_BYTES-used)+c.RESERVE_BYTES+other_remaining(spec)
    anchor=root
    while not anchor.exists():anchor=anchor.parent
    free=shutil.disk_usage(anchor).free
    if declared_gib is not None:
        if type(declared_gib)!=int or declared_gib<0:raise ValueError('Declare current whole free GiB')
        free=min(free,declared_gib*2**30)
    if free<needed:raise OSError(f'Need {math.ceil(needed/2**30)} GiB currently available including protected ladder reservation')
    return dict(used_bytes=used,needed_free_bytes=needed,filesystem_free_bytes=free,other_remaining_bytes=other_remaining(spec))


def estimate(maximum):
    """Conservative complete tensor inventory, moments, EMA, selected and staging copies."""
    native=restart=0;teacher=0;maximum_restart=0
    for task in c.FITS:
        net=model.fresh(task,maximum);state=net.state_dict()
        n=sum(v.numel()*v.element_size() for k,v in state.items() if k.startswith('mod.'))
        all_state=sum(v.numel()*v.element_size() for v in state.values())
        params=sum(p.numel()*p.element_size() for p in net.parameters())
        # Current + staged new complete model/Adam/RNG/EMA/best readout state.
        size=all_state+2*params+6*n+8*2**20
        maximum_restart=max(maximum_restart,size);native+=4*n
        teacher+=c.COUNTS['train']*((128*4+40) if c.cell(task)['arm']=='U100' else (6*4+40) if c.cell(task)['arm']=='OFFLINE' else 0)
        del net
    predictions=len(c.FITS)*len(c.READOUTS)*c.COUNTS['validation']*(32+8+6*4)
    # Completed fits keep only native outputs, not full restart/decoder payloads.
    result=int(1.15*(native+predictions+teacher+2*maximum_restart)+2*2**30)
    if result>c.RESULT_BYTES:raise OSError(f'Measured tensor/row output inventory requires {math.ceil(result/2**30)} GiB; revise storage explicitly')
    return dict(native_selected_bytes=native,prediction_bytes=predictions,teacher_bank_bytes=teacher,
        maximum_restart_bytes=maximum_restart,conservative_output_bytes=result,allowance_bytes=c.RESULT_BYTES)
