"""Installed-model parity, full graph technical rehearsal and debug admission."""
from copy import deepcopy
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer,installed_environment,distillation_loss
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog,memory_snapshot
from hlt_classification.luka_attention.preflight import native_surface_check
from . import worker,banks,campaign
from .contracts import artifact,validate,recipe
from .model import GatedParT,new_model,log_gate,centered_gate_loss


def parity(raw,device,*,bf16=False):
    native=DelphesParticleTransformer().to(device)
    model=GatedParT('privileged').to(device)
    model.mod.load_state_dict(native.mod.state_dict(),strict=True)
    # Identity gate parity checks native weights/updates; gate learning is
    # separately required below rather than artificially forcing its gradient0.
    for p in model.gate.parameters(): p.requires_grad_(False)
    tensors={k:torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')}
    outputs=[]; grads=[]; states=[]; vector_grads=[]
    tol=(.01,5e-4) if bf16 else (2e-5,2e-6)
    with parity_backend(device):
        for m in (native,model):
            m.train(); torch.manual_seed(804)
            optimizer=torch.optim.AdamW(m.mod.parameters(),lr=3e-4)
            # Four-vectors are fixed in production. Witness their auxiliary
            # finite/NaN/Inf topology without declaring native padded singularities
            # to be trainable-parameter failures.
            x=dict(tensors,features=tensors['features'].clone().requires_grad_(),
                   vectors=tensors['vectors'].clone().requires_grad_())
            if m is model: x.update(context_features=tensors['features'],context_mask=tensors['mask'])
            with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):
                out=m(**x)
            out.float().square().mean().backward()
            if not torch.isfinite(out).all() or not torch.isfinite(x['features'].grad).all():
                raise ValueError('Nonfinite identity parity')
            g={n:p.grad.clone() for n,p in m.mod.named_parameters() if p.grad is not None}
            if any(not torch.isfinite(v).all() for v in g.values()): raise ValueError('Nonfinite parity gradient')
            optimizer.step(); outputs.append(out.detach()); grads.append((x['features'].grad,g))
            vector_grads.append(x['vectors'].grad)
            states.append(deepcopy(m.mod.state_dict()))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=tol[0],atol=tol[1])
        torch.testing.assert_close(grads[0],grads[1],rtol=tol[0],atol=tol[1])
        torch.testing.assert_close(states[0],states[1],rtol=tol[0],atol=tol[1])
        for topology in (torch.isfinite,torch.isnan,torch.isposinf,torch.isneginf):
            if not torch.equal(topology(vector_grads[0]),topology(vector_grads[1])):
                raise ValueError('Auxiliary vector gradient topology differs')
        finite=torch.isfinite(vector_grads[0])
        torch.testing.assert_close(vector_grads[0][finite],vector_grads[1][finite],rtol=tol[0],atol=tol[1])
        # The readout witness deep-copies its module. Use the identical native
        # module, not a bound hook that recursively references a live autograd graph.
        native_surface_check(native.mod.blocks[7].attn,device)
    for p in model.gate.parameters(): p.requires_grad_(True)
    model.zero_grad(set_to_none=True); model.eval()
    x=dict(tensors,context_features=tensors['features'],context_mask=tensors['mask'])
    model(**x).float().square().mean().backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.gate.parameters()):
        raise ValueError('Gate has no classification gradient')
    with torch.no_grad(): model.gate.net[-1].weight[4:].normal_(0,.15)
    # Check the actual input gate is sensitive, and replacement refuses a
    # privileged second argument even if it happens to contain HLT values.
    with torch.no_grad():
        model(**x); first=model.factors.clone()
        altered=tensors['features'].clone(); altered[:,0]+=3
        model(**dict(x,context_features=altered)); second=model.factors
    if torch.allclose(log_gate(first,tensors['mask']),log_gate(second,tensors['mask'])):
        raise ValueError('Privileged gate ignores context')
    endpoint=GatedParT('self').to(device)
    endpoint.load_state_dict(model.state_dict(),strict=True)
    try: endpoint(**x)
    except PermissionError: pass
    else: raise ValueError('HLT endpoint accepted privileged inputs')
    # Multiplicative gate = additive log correction, independent of QKV.
    scores=torch.randn(2,3,4,4,device=device); r=torch.randn(2,1,4,4,device=device).tanh()
    weighted=scores.softmax(-1)*r.exp(); weighted=weighted/weighted.sum(-1,keepdim=True)
    torch.testing.assert_close(weighted,(scores+r).softmax(-1),rtol=2e-5,atol=2e-6)
    return dict(bf16=bf16,identity_logits=True,identity_gradients=True,identity_updates=True,
        native_attention_surface=True,gate_gradient=True,context_sensitivity=True,
        endpoint_context_rejected=True,multiplicative_equivalence=True,
        auxiliary_vector_gradient_topology=True)


def stress(cache,context,node,device,*,backward):
    banks.paired(cache,context)
    lengths=banks.counts(cache); ix=np.repeat(int(np.argmax(lengths)),128)
    raw=cache.batch(ix); ctx=None if context is None else context.batch(ix)
    model=new_model(node).to(device)
    if model.mode!='none':
        with torch.no_grad(): model.gate.net[-1].weight[4:].normal_(0,.15)
    model.train(backward); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    start=time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            out=banks.forward(model,raw,ctx,device)
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite stress logits')
            if backward:
                loss=distillation_loss(out,torch.from_numpy(raw['labels']).to(device),
                    teacher_probabilities=torch.full_like(out.float(),1/11))
                if node['gate_weight']:
                    target=model.factors.detach()+.01
                    loss=loss+node['gate_weight']*centered_gate_loss(model.factors,target,torch.from_numpy(raw['mask']).to(device))
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite stress gradients')
                optimizer.step()
        model.factors=None
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(node_id=node['node_id'],role=cache.role,coordinate=cache.coordinate_name,
        maximum=int(lengths.max()),batch=128,steps=3,backward=backward,seconds=time.monotonic()-start)


def budgets(measurements,authentication_seconds=0.):
    if not math.isfinite(authentication_seconds) or authentication_seconds<0:
        raise ValueError('Invalid authentication timing')
    fits=[r for r in measurements if r['kind']=='train']; exports=[r for r in measurements if r['kind']=='export']
    if not fits or not exports: raise ValueError('Missing measured task kinds')
    for m in measurements:
        keys=['load_seconds']+(['epoch_seconds','other_seconds'] if m['kind']=='train' else ['export_seconds'])
        if any(type(m[k]) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for k in keys):
            raise ValueError('Invalid measured timing')
    # Frozen-only technical STAGED epoch is not a joint-training estimate.
    joint=max(m['epoch_seconds'] for m in fits if not m['frozen_backbone'])
    train=max(math.ceil(1.75*(authentication_seconds+m['load_seconds']+
        30*max(m['epoch_seconds'],joint if m['frozen_backbone'] else 0)+m['other_seconds'])/60) for m in fits)
    export=max(math.ceil(2*(authentication_seconds+m['load_seconds']+m['export_seconds'])/60) for m in exports)
    if train>1440 or export>1440: raise ValueError('Measured request exceeds debug24h; science blocked')
    return max(30,train),max(15,export)


def run(gate):
    begin=time.monotonic(); campaign.check_gate(gate); auth=time.monotonic()-begin
    job=worker.execution(gate)
    if installed_environment()!=gate['anchor']['evidence']['environment']:
        raise PermissionError('Installed environment differs from anchor')
    root=Path(gate['root'])/'preflight'; root.mkdir(exist_ok=False)
    log=PhaseLog(root,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticated_views_storage_and_parity'):
        tr=worker.data(gate,'train','D000'); val=worker.data(gate,'validation','D000')
        storage=campaign.storage(gate,banks.counts(tr))
        alignments=dict(train=banks.alignment(tr),validation=banks.alignment(val))
        raw=tr.batch(np.arange(min(4,len(tr))))
        parities=[parity(raw,'cuda',bf16=b) for b in (False,True)]
        del tr,val
    # This rehearsal uses real population caches and the same fit/export workers.
    # Probe checkpoints remain entirely below preflight, never science parents.
    probe=artifact('SPEC',parents=dict(gate=gate['content_hash']),root=str(root),technical=True)
    write_immutable_json(root/'probe_spec.json',probe)
    measurements=[]; stresses=[]
    for task in gate['tasks']:
        if task['kind']=='aggregate': continue
        gc.collect(); torch.cuda.empty_cache()
        if task['kind']=='train':
            node=next(n for n in gate['nodes'] if n['node_id']==task['node_id'])
            with log.phase('stress_'+node['node_id']):
                for role in ('train','validation'):
                    cache=worker.data(gate,role,node['coordinate'])
                    if banks.alignment(cache)!=alignments[role]: raise ValueError('Cross-view slot alignment differs')
                    ctx=None if node['context_coordinate'] is None else worker.data(gate,role,node['context_coordinate'])
                    stresses.append(stress(cache,ctx,node,'cuda',backward=role=='train'))
                    del cache,ctx
        with log.phase('probe_'+task['task_id']):
            row=worker.execute_task(probe,gate,task['task_id'],technical=True)
            measurements.append(row['measurement'])
    memory=memory_snapshot(torch); gpu=gpu_identity()
    if memory['peak_rss_bytes']>=.85*gate['memory_mb']*1024**2 or memory['gpu_peak_bytes']>=.9*gpu['total_memory_bytes']:
        raise MemoryError('Measured host/GPU headroom failed')
    begin=time.monotonic(); campaign.check_gate(gate); auth+=time.monotonic()-begin
    begin=time.monotonic(); outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()]
    auth+=2*(time.monotonic()-begin)
    train_minutes,export_minutes=budgets(measurements,auth)
    result=artifact('PREFLIGHT',parents=dict(gate=gate['content_hash']),job_id=job,
        environment=installed_environment(),gpu=gpu,parities=parities,stress=stresses,
        alignments=alignments,measurements=measurements,memory=memory,storage=storage,
        train_minutes=train_minutes,export_minutes=export_minutes,authentication_seconds=auth,
        technical_checks_passed=True,scientific_fit=False,outputs=outputs)
    write_immutable_json(root/'receipt.json',result)
    return result


def validate_report(gate,r):
    validate(r,'PREFLIGHT',dict(gate=gate['content_hash']))
    expected=[(t['node_id'],t['kind']) for t in gate['tasks'] if t['kind']!='aggregate']
    if (r['technical_checks_passed'] is not True or r['scientific_fit'] is not False
            or [(m['node_id'],m['kind']) for m in r['measurements']]!=expected
            or (r['train_minutes'],r['export_minutes'])!=budgets(r['measurements'],r['authentication_seconds'])
            or r['environment']!=gate['anchor']['evidence']['environment']):
        raise ValueError('Incomplete technical evidence/budget/environment')
    checks=['identity_logits','identity_gradients','identity_updates','native_attention_surface',
            'gate_gradient','context_sensitivity','endpoint_context_rejected','multiplicative_equivalence',
            'auxiliary_vector_gradient_topology']
    if len(r['parities'])!=2 or [p['bf16'] for p in r['parities']]!=[False,True] or any(p.get(k) is not True for p in r['parities'] for k in checks):
        raise ValueError('Missing installed model parity')
    from hlt_classification.luka_fullsim.regular_cache import metadata
    prepared=metadata(gate['anchor']['evidence']['prepared_root'])[1]
    expected={(n['node_id'],role) for n in gate['nodes'] for role in ('train','validation')}
    if len(r['stress'])!=len(expected) or {(s['node_id'],s['role']) for s in r['stress']}!=expected:
        raise ValueError('Missing stress configuration')
    for s in r['stress']:
        node=next(n for n in gate['nodes'] if n['node_id']==s['node_id'])
        if (s['coordinate']!=node['coordinate'] or s['maximum']!=prepared['summaries'][s['role']][node['coordinate']]['maximum']
                or s['batch']!=128 or s['steps']!=3 or s['backward']!=(s['role']=='train')):
            raise ValueError('Stress support mismatch')
    if (not 0<r['memory']['peak_rss_bytes']<.85*gate['memory_mb']*1024**2
            or not 0<r['memory']['gpu_peak_bytes']<.9*r['gpu']['total_memory_bytes']):
        raise ValueError('Memory admission failed')
    root=Path(gate['root'])/'preflight'; paths=[f['path'] for f in r['outputs']]
    required={'probe_spec.json'} | {t['task_id']+'/receipt.json' for t in gate['tasks'] if t['kind']!='aggregate'}
    if len(paths)!=len(set(paths)) or not required<=set(paths): raise ValueError('Missing bound technical tasks')
    for ref in r['outputs']: validate_file_ref(ref,root=root)
    probe=load_json(root/'probe_spec.json'); validate(probe,'SPEC',dict(gate=gate['content_hash']))
    if Path(probe['root'])!=root or probe.get('technical') is not True: raise ValueError('Wrong technical root')
    seen={}
    for task in gate['tasks']:
        if task['kind']=='aggregate': continue
        receipt=worker.completed(probe,gate,task['task_id'],seen,technical=True)
        if receipt is None: raise ValueError('Incomplete technical graph')
        for ref in receipt['outputs']:
            if ref not in r['outputs']: raise ValueError('Unbound technical task payload')
        if receipt['measurement'] not in r['measurements']: raise ValueError('Unbound timing evidence')
    for role,a in r['alignments'].items():
        if a.get('role')!=role or a.get('rows')!=gate['counts'][role] or a.get('prepared')!=gate['anchor']['evidence']['prepared']:
            raise ValueError('Technical population binding differs')
    if set(r['alignments'])!={'train','validation'}: raise ValueError('Missing population binding')
    s=r['storage']; required=3*s['factor_bank_bytes']+5*2**30+24*gate['counts']['train']*11*4
    if (s['factor_bank_bytes']<=0 or s['required_bytes']!=required or s['declared_quota_gib']!=gate['available_quota_gib']
            or required>min(s['filesystem_free_bytes'],gate['available_quota_gib']*2**30)):
        raise ValueError('Storage admission differs')
