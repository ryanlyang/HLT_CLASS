"""Bounded disposable in-worker checks, not a separate queued preflight."""
import numpy as np
import torch
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_attention.banks import counts
from hlt_classification.luka_fullsim.preflight_v2 import memory_snapshot,check_headroom
from .contracts import artifact,validate
from .model import new_model
from .training import batch_loss,optimizer,finite_gradients,clear,registered


def run(spec,tr,val,bank,node,device):
    model=new_model(node).to(device).eval(); model.capture=True
    torch.manual_seed(node['initialization_seed']); native=DelphesParticleTransformer().to(device).eval()
    raw=tr.batch(np.arange(min(4,len(tr)),dtype=np.int64))
    with parity_backend(device),torch.no_grad():
        for bf16 in (False,True):
            torch.testing.assert_close(_forward(model,raw,device,bf16=bf16),_forward(native,raw,device,bf16=bf16),rtol=0,atol=0)
    del native; clear(model)
    tr_counts=counts(tr); val_counts=counts(val)
    ix=np.full(128,int(np.argmax(tr_counts)),dtype=np.int64)
    model.train(); model.capture=True; opt=optimizer(model,registered(node['arm'])['base'])
    opt.zero_grad(set_to_none=True)
    loss,terms,_=batch_loss(model,tr,bank,ix,device)
    head=tuple(model.mod.fc.parameters())
    expected=torch.autograd.grad(registered(node['arm'])['ce_coefficient']*terms['ce'],head,retain_graph=True)
    loss.backward(); finite_gradients(model)
    for p,g in zip(head,expected): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    opt.step(); clear(model); model.eval(); model.capture=False
    with torch.inference_mode():
        out=_forward(model,val.batch(np.full(128,int(np.argmax(val_counts)),dtype=np.int64)),device,bf16=str(device).startswith('cuda'))
        if not torch.isfinite(out).all(): raise ValueError('Invalid maximum-validation output')
    memory=memory_snapshot(torch)
    if str(device).startswith('cuda'):
        check_headroom(memory,memory_mb=spec['memory_mb'],future_cache_bytes=0,gpu_bytes=torch.cuda.get_device_properties(torch.device(device)).total_memory)
    del model,opt,loss,terms,expected,out
    return artifact('STARTUP',parents=dict(spec=spec['content_hash'],bundle=bank.manifest['content_hash']),task=node['arm'],
        passed=True,native_parity=True,main_head_CE_only=True,discarded=True,batch_size=128,updates=1,
        maximum_train=int(tr_counts.max()),maximum_validation=int(val_counts.max()),memory=memory)


def check(value,spec,task,bundle_hash):
    validate(value,'STARTUP',dict(spec=spec['content_hash'],bundle=bundle_hash))
    if (value['task']!=task or any(value[k] is not True for k in ('passed','native_parity','main_head_CE_only','discarded'))
            or (value['batch_size'],value['updates'])!=(128,1)
            or any(type(value[k])!=int or not 1<=value[k]<=512 for k in ('maximum_train','maximum_validation'))):
        raise ValueError('Startup evidence differs')
