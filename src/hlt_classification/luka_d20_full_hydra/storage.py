"""Initial HOME admission only; later writes may fail naturally at the filesystem."""
from pathlib import Path
import math
import shutil
import torch
from hlt_classification.data.cache_contracts import with_content_hash
from . import contracts as c,model


def policy():
    return with_content_hash(dict(contract=c.PREFIX+'_STORAGE_POLICY/v2',schema_version=2,
        parents={},**c.FLAGS,admission='registration_and_explicit_resume_only',
        automatic_home_rechecks=False,runtime_output_limit=False,
        disk_exhaustion='propagate_filesystem_write_error',initial_output_allowance_bytes=c.RESULT_BYTES,
        initial_reserve_bytes=c.RESERVE_BYTES,protected_initial_allowance_bytes=0,
        other_campaign_dependency=None))


def usage(root):
    root=Path(root)
    if not root.exists():return 0
    total=0
    for path in root.rglob('*'):
        if path.is_symlink():raise PermissionError('Campaign payload symlink')
        if path.is_file():total+=path.stat().st_size
    return total


def check(spec,*,declared_gib=None):
    root=Path(spec['root']);used=usage(root)
    needed=max(0,c.RESULT_BYTES-used)+c.RESERVE_BYTES
    anchor=root
    while not anchor.exists():anchor=anchor.parent
    free=shutil.disk_usage(anchor).free
    if declared_gib is not None:
        if type(declared_gib)!=int or declared_gib<0:raise ValueError('Declare current whole free GiB')
        free=min(free,declared_gib*2**30)
    if free<needed:raise OSError(f'Need {math.ceil(needed/2**30)} GiB currently available for this campaign')
    return dict(used_bytes=used,needed_free_bytes=needed,filesystem_free_bytes=free,other_remaining_bytes=0)


def estimate(maximum):
    """Conservative complete tensor inventory, moments, EMA, selected and staging copies."""
    native=restart=0;teacher=0;maximum_restart=0;restart_total=0
    for task in c.FITS:
        net=model.fresh(task,maximum);state=net.state_dict()
        n=sum(v.numel()*v.element_size() for k,v in state.items() if k.startswith('mod.'))
        all_state=sum(v.numel()*v.element_size() for v in state.values())
        params=sum(p.numel()*p.element_size() for p in net.parameters())
        # Current + staged new complete model/Adam/RNG/EMA/best readout state.
        size=all_state+2*params+6*n+8*2**20
        maximum_restart=max(maximum_restart,size);restart_total+=2*size;native+=len(c.readouts(task))*n
        teacher+=c.COUNTS['train']*((128*4+40) if c.cell(task)['arm']=='U100' else (6*4+40) if c.cell(task)['arm']=='OFFLINE' else 0)
        del net
    predictions=sum(len(c.readouts(t)) for t in c.FITS)*c.COUNTS['validation']*(32+8+6*4)
    # Completed fits keep only native outputs, not full restart/decoder payloads.
    result=int(1.15*(native+predictions+teacher+restart_total)+2*2**30)
    if result>c.RESULT_BYTES:raise OSError(f'Measured tensor/row output inventory requires {math.ceil(result/2**30)} GiB; revise storage explicitly')
    return dict(native_selected_bytes=native,prediction_bytes=predictions,teacher_bank_bytes=teacher,
        maximum_restart_bytes=maximum_restart,concurrent_restart_allowance=restart_total,conservative_output_bytes=result,allowance_bytes=c.RESULT_BYTES)
