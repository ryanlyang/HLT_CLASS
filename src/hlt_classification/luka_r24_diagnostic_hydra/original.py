"""Run the immutable 58d0c0ed training functions, with a separate binding receipt."""
from pathlib import Path
import time
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_next_supervision_hydra import training as donor
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as batched
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from .contracts import artifact,validate,cell,original_task


def configure():
    # Original worker uses PyTorch defaults, single CPU compute thread and the
    # donor assignment pool. Restore these after new-worker setup measurements.
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    if batched.solver_pool.cache_info().currsize:batched.solver_pool().shutdown(wait=True)
    batched.ASSIGNMENT_WORKERS=6;batched.solver_pool.cache_clear()


def fit(folder,spec,task,train,val,bank,targets,ctx,*,technical=False):
    if cell(task)['worker']!='original' or ctx.world!=1:raise PermissionError('Original worker requires its single-A6000 arm')
    configure();old=load_json(spec['original_spec']['path']);started=time.perf_counter()
    report=donor.fit(folder,old,original_task(task),train,val,bank,
        candidates=None if cell(task)['control'] else targets,device=str(ctx.device),technical=technical)
    binding=artifact('ORIGINAL_FIT',parents=dict(spec=spec['content_hash'],original=old['content_hash'],report=report['content_hash']),
        task=task,original_task=original_task(task),technical=technical,precision='bf16_native',world_size=1,
        receipt=file_ref(Path(folder)/'receipt.json',root=folder),memory=d.memory(ctx),elapsed_seconds=time.perf_counter()-started)
    write_immutable_json(Path(folder)/'diagnostic_binding.json',binding)
    validate_fit(folder,spec,task,technical=technical);d.release_memory(ctx)
    return report,binding


def validate_fit(folder,spec,task,*,technical=False):
    old=load_json(spec['original_spec']['path'])
    report,arrays=donor.validate_fit(folder,old,original_task(task),technical=technical)
    binding=load_json(Path(folder)/'diagnostic_binding.json')
    validate(binding,'ORIGINAL_FIT',dict(spec=spec['content_hash'],original=old['content_hash'],report=report['content_hash']))
    if (cell(task)['worker']!='original' or binding['task']!=task or binding['original_task']!=original_task(task)
        or binding['technical']!=technical or binding['precision']!='bf16_native' or binding['world_size']!=1
        or binding['receipt']!=file_ref(Path(folder)/'receipt.json',root=folder)):
        raise ValueError('Original worker binding differs')
    validate_file_ref(binding['receipt'],root=folder)
    return report,arrays
