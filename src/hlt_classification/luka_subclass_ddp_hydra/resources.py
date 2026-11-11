"""Real distributed miniatures, FP32 replay, numerical parity and measured release."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays,write_immutable_json,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.model import installed_environment,model_config
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_selective_teacher_hydra.resources import subset_banks,mini_indices
from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot,compare
from .contracts import FITS,artifact,validate
from . import model,distributed as d,inputs,targets,training,execution

TOLERANCES=dict(loss_abs=1e-3,gradient_relative_l2=5e-4,buffer_abs=5e-5)
PROBE_TOLERANCES=dict(loss_abs=1e-6,gradient_relative_l2=5e-6,buffer_abs=1e-6,
    parameters_relative_l2=5e-6,adamw_moments_relative_l2=5e-6)


def miniature(folder,spec,task,train,val,banks,ctx):
    ix,vi=mini_indices(train),mini_indices(val)
    return training.fit(folder,spec,task,Subset(train,ix),Subset(val,vi),subset_banks(banks,ix),ctx,technical=True)


def zero_control(spec,raw,banks,ix,device):
    """Original U100 versus zero-subclass native/predictor and AdamW states."""
    from hlt_classification.luka_embedding_aux.model import AuxiliaryParT
    from hlt_classification.luka_u100_embedding.training import objective
    from hlt_classification.luka_embedding_curriculum_hydra.resources import equal_state
    torch.manual_seed(spec['nodes']['C0']['initialization_seed']);original=AuxiliaryParT().to(device)
    candidate,wrapped,co=d.build(spec,'C0',d.Context(0,1,device),parity=True)
    model.disable_dropout(original);original.capture=True;original.train();oo=d.optimizer(original)
    for group in oo.param_groups:group['lr']=d.learning_rate(3.)
    logits=original(**{k:torch.as_tensor(raw[k],device=device) for k in ('features','vectors','mask')})
    total,_,_=objective(logits,torch.as_tensor(raw['labels'],device=device),original.prediction,
        torch.as_tensor(banks['u100'].targets[ix],device=device),coefficient=5.)
    total.backward();oo.step()
    d.step(candidate,wrapped,co,'C0',raw,banks,ix,3.,d.Context(0,1,device))
    common=lambda m:{k:v for k,v in m.state_dict().items() if not k.startswith('subclass.')}
    state=lambda m,o:{n:o.state[p] for n,p in m.named_parameters() if p in o.state and not n.startswith('subclass.')}
    if not equal_state(common(original),common(candidate)) or not equal_state(state(original,oo),state(candidate,co)):
        raise ValueError('FP32 U100 control update differs')
    return dict(model_update_equal=True,adamw_equal=True)


def qualification(spec,train,banks,ctx):
    checks={}
    for task,n in (('C0',16),('S0p5',16),('S100',15)):
        ix=np.arange(n);raw=train.batch(ix)
        def reference():
            single=d.Context(0,1,ctx.device)
            m,w,o=d.build(spec,task,single,parity=True)
            m.eval();m.capture=False
            torch.manual_seed(spec['nodes'][task]['initialization_seed'])
            factory=load_weaver_particle_transformer_class()(**model_config()).to(ctx.device).eval()
            tensors={k:torch.as_tensor(raw[k],device=ctx.device) for k in ('features','vectors','mask')}
            with torch.no_grad():
                torch.testing.assert_close(m(**tensors),factory(tensors['features'],v=tensors['vectors'],mask=tensors['mask']),rtol=2e-5,atol=5e-6)
            values,_,_=d.step(m,w,o,task,raw,banks,ix,3.,single)
            snap=parity_snapshot(m,o,values['total'])
            del m,w,o,factory;d.release_memory(ctx)
            return snap
        # Small CPU snapshots are sent to both ranks for the same comparison.
        expected=ctx.main(reference)
        m,w,o=d.build(spec,task,ctx,parity=True)
        values,_,_=d.step(m,w,o,task,raw,banks,ix,3.,ctx)
        observed=ctx.main(lambda:parity_snapshot(m,o,values['total']))
        checks[task]=compare(expected,observed,TOLERANCES)
        del m,w,o;d.release_memory(ctx)
    baseline=ctx.main(lambda:zero_control(spec,train.batch(np.arange(16)),banks,np.arange(16),ctx.device))
    neutrality=probe_neutrality(spec,train,banks,ctx)
    value=artifact('PARITY',parents=dict(spec=spec['content_hash']),checks=checks,tolerances=TOLERANCES,
        baseline=baseline,factory_fp32=True,dropout_disabled=True,probe_neutrality=neutrality,probe_tolerances=PROBE_TOLERANCES)
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup/parity.json',value))
    if not all(c['passed'] for c in checks.values()) or not neutrality['passed']:
        raise ValueError('Distributed numerical parity failed; preserve diagnostics')
    return value


def probe_neutrality(spec,train,banks,ctx):
    snapshots=[];random_states=[];ix=np.arange(16)
    for diagnose in (False,True):
        m,w,o=d.build(spec,'S100',ctx)
        values,_,_=d.step(m,w,o,'S100',train.batch(ix),banks,ix,3.,ctx,diagnose=diagnose)
        snapshots.append(ctx.main(lambda:parity_snapshot(m,o,values['total'])))
        random_states.append((torch.get_rng_state(),torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else None))
        del m,w,o;d.release_memory(ctx)
    equal=all(a is None and b is None or a is not None and b is not None and torch.equal(a,b)
              for a,b in zip(*random_states))
    result=compare(*snapshots,PROBE_TOLERANCES)
    result['rng_equal']=all(ctx.all(equal));result['passed']=result['passed'] and result['rng_equal']
    return result


def replay(spec,val,ctx):
    bundle=inputs.Bundle(spec['bundle_root']);saved=load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k],getattr(val,k)) for k in ('identities','labels')):raise ValueError('Replay identity mismatch')
    native=model.Native().to(ctx.device).eval();ref=file_ref(bundle.path('reference/matched/checkpoint.pt'))
    native.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    actual=d.predict(native,val,ctx);del native;d.release_memory(ctx)
    def publish():
        folder=Path(spec['root'])/'setup'
        atomic_publish_bytes(folder/'replay_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
        a,e=[evaluate_probabilities(val.labels,p)['macro_ovr_auc'] for p in (actual,saved['probabilities'])]
        value=artifact('REPLAY',parents=dict(spec=spec['content_hash']),checkpoint=ref,
            predictions=file_ref(folder/'replay_predictions.npz',root=folder),actual_auc=a,historical_auc=e,
            auc_abs=abs(a-e),tolerance=5e-5,precision='fp32',rows=len(val))
        write_immutable_json(folder/'replay.json',value)
        if abs(a-e)>5e-5:raise ValueError('Historical FP32 replay differs; preserve evidence')
        return value
    return ctx.main(publish)


def check_replay(spec):
    folder=Path(spec['root'])/'setup';r=load_json(folder/'replay.json');validate(r,'REPLAY',dict(spec=spec['content_hash']))
    b=inputs.Bundle(spec['bundle_root']);expected=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if r['checkpoint']!=file_ref(b.path('reference/matched/checkpoint.pt')) or r['predictions']['path']!='replay_predictions.npz':raise ValueError('Replay ownership differs')
    actual=load_npz_arrays(validate_file_ref(r['predictions'],root=folder))
    if any(not np.array_equal(actual[k],expected[k]) for k in ('identities','labels')):raise ValueError('Replay rows differ')
    a,e=[evaluate_probabilities(v['labels'],v['probabilities'])['macro_ovr_auc'] for v in (actual,expected)]
    if (r['actual_auc']!=a or r['historical_auc']!=e or r['auc_abs']!=abs(a-e) or abs(a-e)>5e-5
            or r['tolerance']!=5e-5 or r['precision']!='fp32' or r['rows']!=spec['counts']['validation']):raise ValueError('Replay diagnostics differ')
    p=load_json(folder/'parity.json');validate(p,'PARITY',dict(spec=spec['content_hash']))
    if (p['tolerances']!=TOLERANCES or p['probe_tolerances']!=PROBE_TOLERANCES
            or set(p['checks'])!={'C0','S0p5','S100'} or not p['factory_fp32'] or not p['dropout_disabled']
            or p['baseline']!=dict(model_update_equal=True,adamw_equal=True)):raise ValueError('Parity coverage differs')
    for row in p['checks'].values():
        if row['passed'] is not True or any(not np.isfinite(row[k]) or not 0<=row[k]<=v for k,v in TOLERANCES.items()):
            raise ValueError('Parity failed')
    n=p['probe_neutrality']
    if (n['passed'] is not True or n['rng_equal'] is not True
            or any(not np.isfinite(n[k]) or not 0<=n[k]<=v for k,v in PROBE_TOLERANCES.items())):
        raise ValueError('Gradient probes altered training')
    return r


def measure(spec,train,val,banks,ctx,devices,load_seconds):
    if ctx.world!=2 or ctx.device.type!='cuda':raise PermissionError('Real two-GPU setup required')
    root=Path(spec['root'])/'setup';profiles={};qualification(spec,train,banks,ctx);replay(spec,val,ctx)
    prep=load_json(root/'preparation.json');validate(prep,'PREPARATION',dict(spec=spec['content_hash']))
    for task in FITS:
        mini=miniature(root/('mini_'+task),spec,task,train,val,banks,ctx);d.release_memory(ctx)
        torch.cuda.reset_peak_memory_stats(ctx.device)
        m,w,o=d.build(spec,task,ctx);order=np.random.default_rng(20261108).permutation(len(train))
        def step(ix,diagnose=False):return d.step(m,w,o,task,train.batch(ix),banks,ix,3.,ctx,diagnose=diagnose)
        step(np.argsort(lengths(train),kind='stable')[-128:]);step(order[:11])
        tail=len(train)%128
        if tail:step(order[-tail:])
        step(order[:128]);step(order[128:256])
        samples=[d.timed(lambda i=i:step(order[i*128:(i+1)*128]),ctx)[0] for i in range(2,14)]
        probe,_=d.timed(lambda:step(order[:128],True),ctx)
        val_seconds,_=d.timed(lambda:d.predict(m,val,ctx),ctx)
        memory=d.memory(ctx)
        memory['host_peak_bytes']=max(memory['host_peak_bytes'],mini['fit_memory']['host_peak_bytes'],prep['host_peak_bytes'])
        for i,row in enumerate(memory['ranks']):row['gpu_peak_bytes']=max(row['gpu_peak_bytes'],mini['fit_memory']['ranks'][i]['gpu_peak_bytes'])
        profiles[task]=dict(step_samples=samples,step_seconds=float(np.mean(samples)),validation_seconds=val_seconds,
            probe_seconds=probe,load_seconds=load_seconds+prep['seconds'],startup_seconds=sum(e['train_seconds']+e['validation_seconds'] for e in mini['history']),
            memory=memory,real_targets=True,miniature_complete=True,deployment_reload=True,largest_support_backward=True,
            partial_batch_backward=True,train_rows=len(train),validation_rows=len(val))
        if ctx.rank==0:print(f'SUBCLASS_DDP setup {task}: step={np.mean(samples):.3f}s validation={val_seconds:.1f}s',flush=True)
        del m,w,o;d.release_memory(ctx)
        value=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),profiles=profiles,devices=devices,
            environment=installed_environment(),status='complete' if len(profiles)==len(FITS) else 'partial')
        ctx.main(lambda:write_immutable_json(root/f'resource_{task}.json',value))
    ctx.main(lambda:write_immutable_json(root/'resource_diagnostic.json',value))
    accepted_value=accept(spec,value);ctx.main(lambda:write_immutable_json(root/'acceptance.json',accepted_value))
    return accepted_value


def check_memory(spec,memory,devices):
    execution.check_devices(devices)
    if (len(memory['ranks'])!=2 or not 0<memory['host_peak_bytes']<.85*spec['memory_mb']*1024**2
            or memory['host_peak_bytes']<sum(m['peak_rss_bytes'] for m in memory['ranks'])):
        raise MemoryError('Aggregate host envelope exceeded')
    for m,g in zip(memory['ranks'],devices):
        if (not np.isfinite([m['gpu_peak_bytes'],m['peak_rss_bytes']]).all()
                or not 0<m['gpu_peak_bytes']<.88*g['total_memory_bytes'] or not 0<m['peak_rss_bytes']):
            raise MemoryError('Rank memory envelope exceeded')


def accept(spec,diagnostic):
    validate(diagnostic,'RESOURCES',dict(spec=spec['content_hash']))
    if (diagnostic['status']!='complete' or set(diagnostic['profiles'])!=set(FITS)
            or diagnostic['environment']!=spec['reference']['environment']):raise ValueError('Incomplete measured setup')
    execution.check_devices(diagnostic['devices']);minutes={}
    for task,p in diagnostic['profiles'].items():
        if (not all(p[k] is True for k in ('real_targets','miniature_complete','deployment_reload','largest_support_backward','partial_batch_backward'))
                or p['train_rows']!=spec['counts']['train'] or p['validation_rows']!=spec['counts']['validation']
                or len(p['step_samples'])!=12 or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0
                or p['step_seconds']!=float(np.mean(p['step_samples']))):raise ValueError('Incomplete measurement coverage')
        check_memory(spec,p['memory'],diagnostic['devices'])
        times=[p[k] for k in ('validation_seconds','probe_seconds','load_seconds','startup_seconds')]
        if not np.isfinite(times).all() or min(times)<=0:raise ValueError('Invalid timings')
        seconds=30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds'])
        seconds+=5*p['probe_seconds']+p['load_seconds']+p['startup_seconds']+3*p['validation_seconds']
        minutes[task]=max(60,math.ceil((2*seconds+900)/60))+15
        if minutes[task]>spec['fit_max_minutes']:raise RuntimeError('Measured walltime exceeds registered cap')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=diagnostic['content_hash']),
        minutes=minutes,devices=diagnostic['devices'],environment=diagnostic['environment'],technical_weights_discarded=True)


def accepted(spec):
    root=Path(spec['root'])/'setup';value=load_json(root/'acceptance.json')
    if value!=accept(spec,load_json(root/'resource_diagnostic.json')):raise ValueError('Acceptance differs from measurements')
    return value
