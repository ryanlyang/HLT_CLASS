"""Real installed-model acceptance, complete D-graph rehearsal and resource gate."""
from copy import deepcopy
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment,distillation_loss
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from . import campaign,banks,worker
from .contracts import artifact,validate
from .model import AttentionParT,new_model,attention_surface,attention_kl


def native_surface_check(module, device):
    """Verify full readout against installed attention output with dropout off."""
    a=deepcopy(module).to(device).float().eval()
    d,h=a.embed_dim,a.num_heads
    x=torch.randn(2,5,d,device=device)
    padding=torch.tensor([[False,False,False,True,True],[False]*5],device=device)
    bias=torch.randn(2,h,5,5,device=device)
    mha=isinstance(a,nn.MultiheadAttention)
    query=x.transpose(0,1) if mha and not a.batch_first else x
    kwargs=dict(key_padding_mask=padding,attn_mask=bias.reshape(2*h,5,5) if mha else bias)
    with torch.no_grad():
        actual=a(query,query,query,**kwargs)[0]
        surface=attention_surface(a,(query,query,query),kwargs,headwise=True)
        weight=a.in_proj_weight if mha else a.in_proj.weight
        bias_weight=a.in_proj_bias if mha else a.in_proj.bias
        v=F.linear(x,weight[2*d:],None if bias_weight is None else bias_weight[2*d:])
        v=v.reshape(2,5,h,d//h).transpose(1,2)
        reconstructed=(surface@v).transpose(1,2).reshape(2,5,d)
        if not mha and (a.headwise_attn_output_gate or a.elementwise_attn_output_gate):
            raise ValueError('Registered model should not contain attention output gates')
        reconstructed=a.out_proj(reconstructed)
        if mha and not a.batch_first: reconstructed=reconstructed.transpose(0,1)
        torch.testing.assert_close(actual,reconstructed,rtol=2e-5,atol=2e-6)
    return True


def parity(raw, device, *, bf16=False):
    native=DelphesParticleTransformer().to(device)
    captured=AttentionParT().to(device)
    captured.load_state_dict(native.state_dict(),strict=True)
    captured.capture_attention=True
    snapshots=[]
    atol,rtol=(5e-4,.01) if bf16 else (2e-6,2e-5)
    with parity_backend(device):
        for model in (native,captured):
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
            model.train()
            torch.manual_seed(478)
            tensors={k:torch.from_numpy(raw[k]).to(device).clone() for k in ('features','vectors','mask')}
            tensors['features'].requires_grad_(); tensors['vectors'].requires_grad_()
            with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):
                logits=model(**tensors)
            logits.float().square().mean().backward()
            gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
            if (not torch.isfinite(logits).all() or not torch.isfinite(tensors['features'].grad).all()
                    or not torch.stack([torch.isfinite(g).all() for g in gradients.values()]).all()):
                raise ValueError('Nonfinite required parity quantities')
            optimizer.step()
            state=deepcopy(model.state_dict())
            snapshots.append((logits.detach(),tensors['features'].grad,tensors['vectors'].grad,
                              gradients,state,deepcopy(optimizer.state_dict())))
        def compare(a,b):
            if isinstance(a,dict):
                if a.keys()!=b.keys(): raise ValueError('Parity state keys differ')
                for k in a: compare(a[k],b[k])
            elif isinstance(a,list):
                if len(a)!=len(b): raise ValueError('Parity lengths differ')
                for x,y in zip(a,b): compare(x,y)
            elif isinstance(a,torch.Tensor):
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol,equal_nan=True)
            elif a!=b: raise ValueError('Parity state differs')
        for a,b in zip(snapshots[0],snapshots[1]): compare(a,b)
        native_surface_check(native.mod.blocks[7].attn,device)
    del native,captured
    return dict(logits=True,input_gradients=True,parameter_gradients=True,buffers=True,
                optimizer_updates=True,native_attention_surface=True,bf16=bf16,rtol=rtol,atol=atol)


def stress(cache,node,device,*,backward):
    n=banks.counts(cache)
    witness=int(np.argmax(n))
    raw=cache.batch(np.repeat(witness,128))
    if int(raw['mask'].sum(-1).max())!=int(n.max()): raise ValueError('Wrong longest witness')
    model=new_model(node).to(device)
    model.train(backward); model.capture_attention=backward
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits=_forward(model,raw,device,bf16=str(device).startswith('cuda'))
            if backward:
                mask=torch.from_numpy(raw['mask']).to(device)
                a=model.take_attention()
                q=torch.ones_like(a)
                loss=distillation_loss(logits,torch.from_numpy(raw['labels']).to(device),
                                       teacher_probabilities=torch.full_like(logits.float(),1/11))
                loss=loss+.1*attention_kl(a,q,mask)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite stress gradient')
                optimizer.step()
            elif not torch.isfinite(logits).all(): raise ValueError('Nonfinite validation stress')
    del model,optimizer
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(role=cache.role,coordinate=cache.coordinate_name,maximum=int(n.max()),
                batch=128,steps=3,backward=backward)


def budgets(measurements, authentication_seconds=0.):
    if not measurements or not math.isfinite(authentication_seconds) or authentication_seconds<0:
        raise ValueError('Missing/invalid timing evidence')
    if any(type(m[k]) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for m in measurements
           for k in ('load_seconds','epoch_seconds','readback_seconds','export_seconds')):
        raise ValueError('Invalid timing evidence')
    train_minutes=max(math.ceil(1.75*(authentication_seconds+m['load_seconds']+30*m['epoch_seconds']+m['readback_seconds'])/60)
                      for m in measurements)
    export_minutes=max(math.ceil(2*(authentication_seconds+m['load_seconds']+m['export_seconds'])/60) for m in measurements)
    if train_minutes>1440 or export_minutes>1440: raise ValueError('Measured request exceeds debug 24h; no science')
    return max(30,train_minutes),max(15,export_minutes)


def run(gate):
    authentication_start=time.monotonic()
    campaign.check_gate(gate)
    authentication_seconds=time.monotonic()-authentication_start
    job=worker.execution(gate)
    if installed_environment()!=gate['anchor']['evidence']['environment']:
        raise PermissionError('Installed environment differs from the imported fixed30 anchor')
    root=Path(gate['root'])/'preflight'; root.mkdir(exist_ok=False)
    log=PhaseLog(root,lambda:memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    measurements=[]; stress_rows=[]; validation_alignment=None
    with log.phase('anchor_cache_and_storage'):
        started=time.monotonic(); tr=worker.data(gate,'train','U100'); anchor_load=time.monotonic()-started
        storage=campaign.storage(gate,banks.counts(tr)); common_alignment=banks.alignment(tr)
    with log.phase('installed_native_parity_fp32_and_bf16'):
        raw=tr.batch(np.arange(4))
        parities=[parity(raw,'cuda',bf16=b) for b in (False,True)]
    with log.phase('real_anchor_bank'):
        path,parent_hashes=worker.checkpoint(gate,None,'ANCHOR')
        anchor_node=dict(gate['nodes'][0],coordinate='U100',node_id='ANCHOR')
        model=new_model(anchor_node)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        started=time.monotonic()
        banks.export(root/'export_ANCHOR/bank',model,tr,parents=parent_hashes,
                     teacher='ANCHOR',attention=True,device='cuda')
        banks.Bank(root/'export_ANCHOR/bank',tr,parents=parent_hashes,teacher='ANCHOR',require_attention=True)
        anchor_export=time.monotonic()-started
        del model,tr
    # This full-population technical rehearsal executes both complete graphs.
    # One-pass probes are never imported into science, which exports U100 anew.
    tiny_spec=dict(root=str(root))
    for node in gate['nodes']:
        gc.collect(); torch.cuda.empty_cache()
        with log.phase('probe_'+node['node_id']):
            started=time.monotonic()
            tr=worker.data(gate,'train',node['coordinate']); val=worker.data(gate,'validation',node['coordinate'])
            if banks.alignment(tr)!=common_alignment: raise ValueError('Persistent-slot transport differs')
            aligned=banks.alignment(val)
            if validation_alignment is None: validation_alignment=aligned
            if aligned!=validation_alignment: raise ValueError('Validation persistent-slot transport differs')
            path,parents=worker.checkpoint(gate,tiny_spec,node['teacher'])
            bank=banks.Bank(root/('export_'+node['teacher'])/'bank',tr,parents=parents,
                            teacher=node['teacher'],require_attention=node['branch']=='ATTN')
            load_seconds=time.monotonic()-started
            if node['branch']=='ATTN':
                stress_rows.extend([stress(tr,node,'cuda',backward=True),stress(val,node,'cuda',backward=False)])
            started=time.monotonic()
            report,model=worker.fit(root/('train_'+node['node_id']),gate,node,tr,val,bank,device='cuda',technical=True)
            full_fit=time.monotonic()-started
            epoch_seconds=report['validation_history'][0]['train_seconds']+report['validation_history'][0]['validation_seconds']
            started=time.monotonic()
            if node['coordinate']!='D000':
                _,parents=worker.checkpoint(gate,tiny_spec,node['node_id'])
                banks.export(root/('export_'+node['node_id'])/'bank',model,tr,parents=parents,
                    teacher=node['node_id'],attention=node['branch']=='ATTN',device='cuda')
                banks.Bank(root/('export_'+node['node_id'])/'bank',tr,parents=parents,
                    teacher=node['node_id'],require_attention=node['branch']=='ATTN')
            measurements.append(dict(node_id=node['node_id'],load_seconds=max(load_seconds,anchor_load),
                epoch_seconds=epoch_seconds,readback_seconds=max(0.,full_fit-epoch_seconds),
                export_seconds=max(anchor_export,time.monotonic()-started)))
            del tr,val,bank,model
    snapshot=memory_snapshot(torch); gpu=gpu_identity()
    if snapshot['peak_rss_bytes']>=.85*gate['memory_mb']*1024**2 or snapshot['gpu_peak_bytes']>=.9*gpu['total_memory_bytes']:
        raise MemoryError('Measured memory headroom failed; no science')
    authentication_start=time.monotonic()
    campaign.check_gate(gate)
    authentication_seconds+=time.monotonic()-authentication_start
    # Full acceptance replay also hashes miniature payloads. Time this I/O and
    # reserve two such reads per science stage, in addition to both source replays.
    hash_start=time.monotonic()
    outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()]
    authentication_seconds+=2*(time.monotonic()-hash_start)
    train_minutes,export_minutes=budgets(measurements,authentication_seconds)
    result=artifact('PREFLIGHT',parents=dict(gate=gate['content_hash']),job_id=job,
        environment=installed_environment(),gpu=gpu,parities=parities,stress=stress_rows,
        train_alignment=common_alignment,validation_alignment=validation_alignment,
        measurements=measurements,memory=snapshot,storage=storage,train_minutes=train_minutes,
        authentication_seconds=authentication_seconds,
        export_minutes=export_minutes,technical_checks_passed=True,scientific_fit=False,
        outputs=outputs)
    write_immutable_json(root/'receipt.json',result)
    return result


def validate_report(gate,report):
    validate(report,'PREFLIGHT',dict(gate=gate['content_hash']))
    if (report['technical_checks_passed'] is not True or report['scientific_fit'] is not False
            or len(report['parities'])!=2 or [p['bf16'] for p in report['parities']]!=[False,True]
            or [r['node_id'] for r in report['measurements']]!=[n['node_id'] for n in gate['nodes']]
            or (report['train_minutes'],report['export_minutes'])!=budgets(report['measurements'],report['authentication_seconds'])):
        raise ValueError('Incomplete technical evidence')
    for p in report['parities']:
        if any(p.get(k) is not True for k in ('logits','input_gradients','parameter_gradients','buffers','optimizer_updates','native_attention_surface')):
            raise ValueError('Missing parity check')
    expected={(r,co) for r in ('train','validation') for co in ('D066','D033','D000')}
    if len(report['stress'])!=6 or {(r['role'],r['coordinate']) for r in report['stress']}!=expected:
        raise ValueError('Missing role-specific stress')
    from hlt_classification.luka_fullsim.regular_cache import metadata
    p=metadata(gate['anchor']['evidence']['prepared_root'])[1]
    for row in report['stress']:
        maximum=p['summaries'][row['role']][row['coordinate']]['maximum']
        if (row['maximum']!=maximum or row['batch']!=128 or row['steps']!=3
                or row['backward']!=(row['role']=='train')):
            raise ValueError('Stress support/role differs')
    memory=report['memory']
    if not 0<memory['peak_rss_bytes']<.85*gate['memory_mb']*1024**2 or not 0<memory['gpu_peak_bytes']<.9*report['gpu']['total_memory_bytes']:
        raise ValueError('Memory admission failed')
    root=Path(gate['root'])/'preflight'
    paths=[ref['path'] for ref in report['outputs']]
    required={'export_ANCHOR/bank/manifest.json'} | {f'train_{n["node_id"]}/{f}' for n in gate['nodes']
               for f in ('training_report.json','selected.pt','validation_predictions.npz')}
    if len(paths)!=len(set(paths)) or not required<=set(paths): raise ValueError('Missing miniature payloads')
    for ref in report['outputs']: validate_file_ref(ref,root=root)
    if (report['environment']!=gate['anchor']['evidence']['environment']
            or report['train_alignment'].get('prepared')!=gate['anchor']['evidence']['prepared']
            or report['train_alignment'].get('role')!='train'
            or report['train_alignment'].get('rows')!=gate['counts']['train']
            or report['validation_alignment'].get('prepared')!=gate['anchor']['evidence']['prepared']
            or report['validation_alignment'].get('role')!='validation'
            or report['validation_alignment'].get('rows')!=gate['counts']['validation']):
        raise ValueError('Technical population/environment differs')
    for node in gate['nodes']:
        bank_name='export_'+node['teacher']+'/bank/manifest.json'
        if bank_name not in paths: raise ValueError('Missing intermediate technical bank')
        bank=load_json(root/bank_name)
        _,parents=worker.checkpoint(gate,dict(root=str(root)),node['teacher'])
        validate(bank,'BANK',parents)
        if (bank['alignment']!=report['train_alignment'] or bank['teacher']!=node['teacher']
                or (node['branch']=='ATTN' and bank['attention'] is not True)):
            raise ValueError('Technical bank arm/teacher/alignment differs')
        for ref in [bank['population']]+[s['payload'] for s in bank['shards']]:
            bound=dict(ref,path='export_'+node['teacher']+'/bank/'+ref['path'])
            if bound not in report['outputs']:
                raise ValueError('Unbound technical bank payload')
        training=load_json(root/('train_'+node['node_id'])/'training_report.json')
        worker.report_valid(training,node,bank['content_hash'],gate['anchor']['evidence']['prepared'],
                            gate['counts']['validation'],technical=True)
    storage=report['storage']
    expected_bytes=7*storage['attention_bank_bytes']+5*2**30+20*gate['counts']['train']*11*4
    if (storage['attention_bank_bytes']<=0 or storage['required_bytes']!=expected_bytes
            or storage['declared_quota_gib']!=gate['available_quota_gib']
            or expected_bytes>min(storage['filesystem_free_bytes'],gate['available_quota_gib']*2**30)):
        raise ValueError('Storage acceptance differs')
