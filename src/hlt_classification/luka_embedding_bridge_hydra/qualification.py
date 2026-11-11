"""Installed native parity, one-versus-two-rank updates and neutral diagnostics."""
from pathlib import Path
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.model import model_config
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot,compare
from hlt_classification.luka_subclass_ddp_hydra.resources import TOLERANCES,PROBE_TOLERANCES
from .contracts import artifact,validate
from .registry import teacher_condition
from . import model,distributed as d,inputs

CHECKS=('C0','LS_D050','LM_D033','LS_D000','R8','P7','TR_B','TR_KIT',
        'TP_HLT_coarse','TP_HLT_medium','TP_HLT_fine','TP_OFFLINE_coarse','TP_OFFLINE_medium','TP_OFFLINE_fine')


def zero_control(spec,ds,target,device):
    from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
    from hlt_classification.luka_u100_embedding.training import objective
    from hlt_classification.luka_embedding_curriculum_hydra.resources import equal_state
    ix=np.arange(16);ctx=d.Context(0,1,device)
    torch.manual_seed(spec['nodes']['C0']['initialization_seed']);original=AuxiliaryParT().to(device)
    candidate,wrapped,co=d.build(spec,'C0',ctx,parity=True)
    model.disable_dropout(original);original.capture=True;original.train();oo=d.optimizer(original)
    for group in oo.param_groups:group['lr']=d.learning_rate(3.)
    raw=ds.view('train','D000').batch(ix);logits=original(**d.arrays(raw,ctx))
    total,_,_=objective(logits,torch.as_tensor(raw['labels'],device=device),original.prediction,
                       torch.as_tensor(target.batch(ix)['u100'],device=device),coefficient=5.)
    total.backward();oo.step()
    d.step(candidate,wrapped,co,'C0',ds,target,ix,3,0,3.,ctx)
    state=lambda m,o:{n:o.state[p] for n,p in m.named_parameters() if p in o.state}
    if not equal_state(original.state_dict(),candidate.state_dict()) or not equal_state(state(original,oo),state(candidate,co)):
        raise ValueError('Original U100 control AdamW update differs')
    return dict(model_update_equal=True,adamw_equal=True)


def qualify(spec,ds,targets,ctx):
    checks={}
    for label in CHECKS:
        task=label.split('_')[0] if label.startswith(('TR_','TP_')) else label
        batch=0
        if task=='TR':batch=next(i for i in range(6) if teacher_condition(3,i)==label.split('_')[1])
        if task=='TP':batch=next(i for i in range(6) if teacher_condition(3,i,regional=True)==tuple(label.split('_')[1:]))
        ix=np.arange(15 if task in ('TP','R8','P7') else 16);target=targets(task)
        def reference():
            single=d.Context(0,1,ctx.device);m,w,o=d.build(spec,task,single,parity=True)
            if task=='C0':
                m.eval();torch.manual_seed(spec['nodes'][task]['initialization_seed'])
                native=load_weaver_particle_transformer_class()(**model_config()).to(ctx.device).eval()
                x=d.arrays(ds.view('train','D000').batch(ix),single)
                with torch.no_grad():torch.testing.assert_close(m(**x,auxiliary=False),native(x['features'],v=x['vectors'],mask=x['mask']),rtol=2e-5,atol=5e-6)
                del native
            values,_=d.step(m,w,o,task,ds,target,ix,3,batch,3.,single)
            return parity_snapshot(m,o,values['total'])
        expected=ctx.main(reference);d.release_memory(ctx)
        m,w,o=d.build(spec,task,ctx,parity=True)
        values,_=d.step(m,w,o,task,ds,target,ix,3,batch,3.,ctx)
        observed=ctx.main(lambda:parity_snapshot(m,o,values['total']))
        checks[label]=compare(expected,observed,TOLERANCES)
        del m,w,o,expected,observed;d.release_memory(ctx)
    baseline=ctx.main(lambda:zero_control(spec,ds,targets('C0'),ctx.device))
    neutrality={}
    for task in ('C0','R8','P7','TR','TP'):
        snapshots=[];rng=[]
        for diagnose in (False,True):
            m,w,o=d.build(spec,task,ctx);values,_=d.step(m,w,o,task,ds,targets(task),np.arange(16),3,0,3.,ctx,diagnose=diagnose)
            snapshots.append(ctx.main(lambda:parity_snapshot(m,o,values['total'])))
            rng.append((torch.get_rng_state(),torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else None))
            del m,w,o;d.release_memory(ctx)
        result=compare(*snapshots,PROBE_TOLERANCES)
        result['rng_equal']=all(ctx.all(all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(*rng))))
        result['passed']=result['passed'] and result['rng_equal'];neutrality[task]=result
    value=artifact('PARITY',parents=dict(spec=spec['content_hash']),checks=checks,tolerances=TOLERANCES,baseline=baseline,
        factory_fp32=True,dropout_disabled=True,probe_neutrality=neutrality,probe_tolerances=PROBE_TOLERANCES)
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup/parity.json',value))
    check_parity(value,spec)
    return value


def check_parity(p,spec):
    validate(p,'PARITY',dict(spec=spec['content_hash']))
    if (p['tolerances']!=TOLERANCES or p['probe_tolerances']!=PROBE_TOLERANCES or set(p['checks'])!=set(CHECKS)
            or p['factory_fp32']is not True or p['dropout_disabled']is not True
            or p['baseline']!=dict(model_update_equal=True,adamw_equal=True)
            or set(p['probe_neutrality'])!={'C0','R8','P7','TR','TP'}):raise ValueError('Incomplete parity coverage')
    for rows,tolerances in ((p['checks'],TOLERANCES),(p['probe_neutrality'],PROBE_TOLERANCES)):
        for r in rows.values():
            if r['passed']is not True or any(not np.isfinite(r[k]) or not 0<=r[k]<=v for k,v in tolerances.items()):raise ValueError('Numerical parity failed')
    if not all(r['rng_equal']is True for r in p['probe_neutrality'].values()):raise ValueError('Diagnostics changed RNG')


def replay(spec,ds,ctx):
    b=inputs.Bundle(spec['bundle_root']);saved=load_npz_arrays(b.path('reference/matched/predictions.npz'));val=ds.view('validation','D000')
    if any(not np.array_equal(saved[k],getattr(val,k)) for k in ('identities','labels')):raise ValueError('Replay identity mismatch')
    native=model.Native().to(ctx.device).eval();ref=file_ref(b.path('reference/matched/checkpoint.pt'))
    native.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    actual,_=d.predict(native,ds,'validation',ctx);del native;d.release_memory(ctx)
    def publish():
        root=Path(spec['root'])/'setup';atomic_publish_bytes(root/'replay_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
        a,e=[evaluate_probabilities(val.labels,p)['macro_ovr_auc'] for p in (actual,saved['probabilities'])]
        value=artifact('REPLAY',parents=dict(spec=spec['content_hash']),checkpoint=ref,predictions=file_ref(root/'replay_predictions.npz',root=root),
            actual_auc=a,historical_auc=e,auc_abs=abs(a-e),tolerance=5e-5,precision='fp32',rows=len(val))
        write_immutable_json(root/'replay.json',value)
        if abs(a-e)>5e-5:raise ValueError('Historical FP32 replay differs')
        return value
    return ctx.main(publish)


def check_replay(spec):
    root=Path(spec['root'])/'setup';r=load_json(root/'replay.json');validate(r,'REPLAY',dict(spec=spec['content_hash']))
    b=inputs.Bundle(spec['bundle_root']);expected=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if r['checkpoint']!=file_ref(b.path('reference/matched/checkpoint.pt')) or r['predictions']['path']!='replay_predictions.npz':raise ValueError('Replay ownership differs')
    actual=load_npz_arrays(validate_file_ref(r['predictions'],root=root))
    if any(not np.array_equal(actual[k],expected[k]) for k in ('identities','labels')):raise ValueError('Replay join differs')
    a,e=[evaluate_probabilities(v['labels'],v['probabilities'])['macro_ovr_auc'] for v in (actual,expected)]
    if (r['actual_auc']!=a or r['historical_auc']!=e or r['auc_abs']!=abs(a-e) or abs(a-e)>5e-5
            or r['tolerance']!=5e-5 or r['precision']!='fp32' or r['rows']!=spec['counts']['validation']):raise ValueError('Replay differs')
    check_parity(load_json(root/'parity.json'),spec)
    u=load_json(root/'u100_replay.json');t=inputs.transferred(spec)
    validate(u,'U100_REPLAY',dict(spec=spec['content_hash'],teacher=t.manifest['teacher']['sha256'],original=spec['embedding_hash']))
    if u['passed']is not True or u['rows']!=4096 or (u['atol'],u['rtol'])!=(1e-4,1e-4) or not np.isfinite(u['maximum_error']):
        raise ValueError('U100 reproduction witness differs')
    return r
