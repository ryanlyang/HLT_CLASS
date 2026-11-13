"""Four-rank production qualification; no accuracy gate or fallback precision."""
from pathlib import Path
from types import SimpleNamespace
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_next_supervision_hydra.inputs import SubsetCandidates, Bundle
from hlt_classification.luka_next_supervision_hydra.contracts import seed
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from .contracts import FITS, artifact, validate, cell, scientific_cfg
from . import kernel as k, training

TOLERANCES = dict(loss_abs=1e-3,gradient_relative_l2=5e-4,buffer_abs=1e-4,
                  parameters_relative_l2=5e-4,adamw_moments_relative_l2=5e-4)
CASES = (('D01',7.,16),('D02',3.,15),('D02',4.5,16),('D02',7.,15),('D04',7.,15),
         ('D18',1.5,15),('D18',4.5,16),('D18',7.,15),('D20',5.,15),('D17',24.,15),('D16',7.,15))


def miniature(folder,spec,task,train,val,bank,targets,ctx):
    def indices(cache):
        first=[int(np.flatnonzero(cache.labels==c)[0]) for c in range(11)]
        return np.array(list(dict.fromkeys([*first,*range(min(139,len(cache)))]))[:min(139,len(cache))],np.int64)
    ti,vi=indices(train),indices(val)
    return training.fit(folder,spec,task,Subset(train,ti),Subset(val,vi),
        SimpleNamespace(targets=bank.targets[ti],manifest=bank.manifest),SubsetCandidates(targets,ti),ctx,technical=True)


def scalar(losses):return losses['ce']+losses['weighted_embedding']+losses['decoder_weighted_reconstruction']


def identity_and_warmup(spec,batch,device):
    """Actual native graph/init parity and two warm updates with a changing decoder."""
    control,co=k.build(spec,'D01',0,1,device,parity=True)
    warm,wo=k.build(spec,'D18',0,1,device,parity=True)
    raw=batch[0]
    torch.manual_seed(seed(spec['imported'],scientific_cfg('D01'),'initialization_seed'))
    factory=DelphesParticleTransformer().to(device).eval();control.eval();control.capture=False
    with torch.no_grad():torch.testing.assert_close(k.native(control,raw,device),k.native(factory,raw,device),rtol=2e-5,atol=5e-6)
    before={n:p.detach().clone() for n,p in warm.aux.named_parameters()}
    for position in (1.,2.):
        k.update(control,co,(raw,batch[1],None),'D01',0,1,device,position)
        _,detail=k.update(warm,wo,batch,'D18',0,1,device,position,probe=True)
        assert detail['diagnostics']['weighted_auxiliary_jet_gradient_norm']==0
        assert detail['diagnostics']['weighted_auxiliary_token_gradient_norm']==0
        native_control={n:v for n,v in control.state_dict().items() if not n.startswith('aux.')}
        native_warm={n:v for n,v in warm.state_dict().items() if not n.startswith('aux.')}
        for n,v in native_control.items():torch.testing.assert_close(v,native_warm[n],rtol=5e-6,atol=5e-7)
        cp=dict(control.named_parameters());wp=dict(warm.named_parameters())
        for n,p in cp.items():
            if n.startswith('aux.') or p not in co.state:continue
            for field in ('exp_avg','exp_avg_sq'):torch.testing.assert_close(co.state[p][field],wo.state[wp[n]][field],rtol=5e-6,atol=1e-8)
    if all(torch.equal(before[n],p) for n,p in warm.aux.named_parameters()):raise ValueError('Warm decoder did not train')
    return dict(factory=True,native_and_adamw_unchanged_by_decoder=True,decoder_changed=True,updates=2)


def qualification(spec,train,bank,targets,ctx):
    checks={}
    for task,position,n in CASES:
        data=training.batch(train,bank,targets,np.arange(n),1,task)
        def reference():
            m,o=k.build(spec,task,0,1,ctx.device,parity=True)
            losses,_=k.update(m,o,data,task,0,1,ctx.device,position)
            result=k.snapshot(m,o,scalar(losses));del m,o;d.release_memory(ctx);return result
        expected=ctx.main(reference)
        m,o=k.build(spec,task,ctx.rank,ctx.world,ctx.device,parity=True)
        losses,_=k.update(m,o,data,task,ctx.rank,ctx.world,ctx.device,position)
        result=ctx.main(lambda:k.compare(expected,k.snapshot(m,o,scalar(losses)),TOLERANCES))
        checks[f'{task}_{position:g}_{n}']=result
        del m,o,expected;d.release_memory(ctx)
    baseline=ctx.main(lambda:identity_and_warmup(spec,training.batch(train,bank,targets,np.arange(15),1,'D18'),ctx.device))
    d.release_memory(ctx)
    # Compare identical stochastic updates with/without probes, including RNG.
    snaps=[];rngs=[]
    for probe in (False,True):
        m,o=k.build(spec,'D18',ctx.rank,ctx.world,ctx.device)
        losses,_=k.update(m,o,training.batch(train,bank,targets,np.arange(16),1,'D18'),
            'D18',ctx.rank,ctx.world,ctx.device,4.5,probe=probe)
        snaps.append(ctx.main(lambda:k.snapshot(m,o,scalar(losses))))
        rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state(ctx.device)))
        del m,o;d.release_memory(ctx)
    neutrality=k.compare(*snaps,TOLERANCES);neutrality['rng_equal']=all(ctx.all(all(torch.equal(a,b) for a,b in zip(*rngs))))
    neutrality['passed']=neutrality['passed'] and neutrality['rng_equal']
    result=artifact('PARITY',parents=dict(spec=spec['content_hash']),checks=checks,cases=[list(v) for v in CASES],
        tolerances=TOLERANCES,baseline=baseline,probe_neutrality=neutrality,world_size=ctx.world,dropout_disabled_for_parity=True)
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup/parity.json',result))
    check_parity(spec,result);return result


def check_parity(spec,p):
    validate(p,'PARITY',dict(spec=spec['content_hash']))
    names={f'{t}_{pos:g}_{n}' for t,pos,n in CASES}
    if (p['world_size']!=4 or p['cases']!=[list(v) for v in CASES] or set(p['checks'])!=names
            or p['tolerances']!=TOLERANCES or p['dropout_disabled_for_parity'] is not True
            or p['baseline']!=dict(factory=True,native_and_adamw_unchanged_by_decoder=True,decoder_changed=True,updates=2)
            or p['probe_neutrality']['rng_equal'] is not True):raise ValueError('Qualification coverage differs')
    for row in [*p['checks'].values(),p['probe_neutrality']]:
        if row['passed'] is not True or any(not np.isfinite(row[k]) or not 0<=row[k]<=v for k,v in TOLERANCES.items()):
            raise ValueError('Four-rank numerical qualification failed')


def replay(spec,old,val,ctx):
    bundle=Bundle(old['bundle_root']);saved=load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k],getattr(val,k)) for k in ('identities','labels')):raise ValueError('Replay rows differ')
    native=DelphesParticleTransformer().to(ctx.device).eval();ref=file_ref(bundle.path('reference/matched/checkpoint.pt'))
    native.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    actual=training.predict(native,val,ctx);del native;d.release_memory(ctx)
    def publish():
        folder=Path(spec['root'])/'setup'
        atomic_publish_bytes(folder/'replay_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
        a,e=[evaluate_probabilities(val.labels,p)['macro_ovr_auc'] for p in (actual,saved['probabilities'])]
        result=artifact('REPLAY',parents=dict(spec=spec['content_hash']),checkpoint=ref,
            predictions=file_ref(folder/'replay_predictions.npz',root=folder),actual_auc=a,historical_auc=e,
            auc_abs=abs(a-e),tolerance=5e-5,rows=len(val),precision='fp32')
        write_immutable_json(folder/'replay.json',result)
        if abs(a-e)>5e-5:raise ValueError('Historical FP32 replay differs')
        return result
    return ctx.main(publish)


def check_replay(spec,old):
    folder=Path(spec['root'])/'setup';r=load_json(folder/'replay.json');validate(r,'REPLAY',dict(spec=spec['content_hash']))
    b=Bundle(old['bundle_root']);saved=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if r['predictions']['path']!='replay_predictions.npz':raise ValueError('Replay path differs')
    actual=load_npz_arrays(validate_file_ref(r['predictions'],root=folder))
    if any(not np.array_equal(actual[k],saved[k]) for k in ('identities','labels')):raise ValueError('Replay identity differs')
    a,e=[evaluate_probabilities(x['labels'],x['probabilities'])['macro_ovr_auc'] for x in (actual,saved)]
    if (r['actual_auc']!=a or r['historical_auc']!=e or r['auc_abs']!=abs(a-e) or abs(a-e)>5e-5
            or r['tolerance']!=5e-5 or r['precision']!='fp32' or r['rows']!=spec['counts']['validation']
            or r['checkpoint']!=file_ref(b.path('reference/matched/checkpoint.pt'))):raise ValueError('Replay evidence differs')


def check_devices(devices):
    if len(devices)!=4:raise ValueError('Four GPUs required')
    for row in devices:
        if (row['name'] not in ('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti')
                or row['compute_capability']!=[6,1] or row['total_memory_bytes']<10*2**30):raise ValueError('GTX1080Ti required')


def check_memory(spec,memory,devices):
    check_devices(devices)
    if (len(memory['ranks'])!=4 or not 0<memory['host_peak_bytes']<spec['resource_policy']['host_fraction']*spec['memory_mb']*1024**2
            or memory['host_peak_bytes']<sum(m['peak_rss_bytes'] for m in memory['ranks'])):raise MemoryError('Aggregate host envelope exceeded')
    for m,g in zip(memory['ranks'],devices):
        if not (0<m['gpu_peak_bytes']<spec['resource_policy']['gpu_fraction']*g['total_memory_bytes'] and 0<m['peak_rss_bytes']):
            raise MemoryError('Rank memory envelope exceeded')


def measure(spec,old,train,val,bank,targets,ctx,devices,load_seconds):
    if ctx.world!=4 or ctx.device.type!='cuda':raise PermissionError('Real four-GPU setup required')
    root=Path(spec['root'])/'setup';qualification(spec,train,bank,targets,ctx);replay(spec,old,val,ctx)
    profiles={};order=np.random.default_rng(20261113).permutation(len(train))
    stress=[np.argsort(lengths(train),kind='stable')[-128:],np.argsort(targets.counts,kind='stable')[-128:],order[:11],order[-(len(train)%128):]]
    for task in FITS:
        mini=miniature(root/('mini_'+task),spec,task,train,val,bank,targets,ctx);d.release_memory(ctx)
        torch.cuda.reset_peak_memory_stats(ctx.device)
        m,o=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
        # Full weight for every cell, including long delays and decoder warmup.
        position=cell(task)['delay']+cell(task)['ramp']+.5
        def step(ix,probe=False):return k.update(m,o,training.batch(train,bank,targets,ix,1,task),task,
            ctx.rank,ctx.world,ctx.device,position,probe=probe)
        for ix in stress:step(ix)
        step(order[:128]);step(order[128:256])
        samples=[d.timed(lambda i=i:step(order[i*128:(i+1)*128]),ctx)[0] for i in range(2,14)]
        probe,_=d.timed(lambda:step(order[:128],True),ctx)
        val_seconds,_=d.timed(lambda:training.evaluate(m,val,ctx),ctx)
        memory=d.memory(ctx);memory['host_peak_bytes']=max(memory['host_peak_bytes'],mini['fit_memory']['host_peak_bytes'])
        for i,row in enumerate(memory['ranks']):row['gpu_peak_bytes']=max(row['gpu_peak_bytes'],mini['fit_memory']['ranks'][i]['gpu_peak_bytes'])
        profiles[task]=dict(step_samples=samples,step_seconds=float(np.mean(samples)),validation_seconds=val_seconds,
            probe_seconds=probe,load_seconds=load_seconds,export_seconds=mini['export_seconds'],
            miniature_seconds=sum(e['train_seconds']+e['validation_seconds'] for e in mini['history']),memory=memory,
            miniature=mini['content_hash'],position=position,full_hlt_support=True,full_offline_support=True,
            partial11=True,tail_rows=len(train)%128,train_rows=len(train),validation_rows=len(val))
        if ctx.rank==0:print(f'R24_CURRICULUM setup {task}: step={np.mean(samples):.3f}s full_validation={val_seconds:.1f}s',flush=True)
        del m,o;d.release_memory(ctx)
        value=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),profiles=profiles,devices=devices,
            environment=installed_environment(),status='complete' if len(profiles)==20 else 'partial')
        ctx.main(lambda:write_immutable_json(root/f'resource_{task}.json',value))
    ctx.main(lambda:write_immutable_json(root/'resource_diagnostic.json',value))
    accepted_value=accept(spec,value);ctx.main(lambda:write_immutable_json(root/'acceptance.json',accepted_value));return accepted_value


def accept(spec,value):
    validate(value,'RESOURCES',dict(spec=spec['content_hash']))
    if value['status']!='complete' or set(value['profiles'])!=set(FITS) or value['environment']!=spec['environment']:
        raise ValueError('Incomplete measured setup')
    check_devices(value['devices']);minutes={}
    for task,p in value['profiles'].items():
        if (not all(p[k] is True for k in ('full_hlt_support','full_offline_support','partial11'))
                or p['tail_rows']!=spec['counts']['train']%128 or p['train_rows']!=spec['counts']['train']
                or p['validation_rows']!=spec['counts']['validation'] or len(p['step_samples'])!=12
                or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0
                or p['step_seconds']!=float(np.mean(p['step_samples']))
                or p['position']!=cell(task)['delay']+cell(task)['ramp']+.5):raise ValueError('Incomplete resource coverage')
        check_memory(spec,p['memory'],value['devices'])
        if any(not np.isfinite(p[k]) or p[k]<=0 for k in ('validation_seconds','probe_seconds','load_seconds','export_seconds','miniature_seconds')):
            raise ValueError('Invalid measured timings')
        seconds=30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds']+p['probe_seconds'])
        seconds+=p['load_seconds']+p['export_seconds']+2*p['validation_seconds']+p['miniature_seconds']
        minutes[task]=max(60,math.ceil((spec['resource_policy']['safety_multiplier']*seconds+spec['resource_policy']['fixed_overhead_seconds'])/60))
        if minutes[task]>spec['fit_max_minutes']:raise RuntimeError('Measured walltime exceeds registered 48h cap')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=value['content_hash']),minutes=minutes,
        devices=value['devices'],environment=value['environment'],technical_weights_discarded=True)


def accepted(spec):
    root=Path(spec['root'])/'setup';value=load_json(root/'acceptance.json')
    if value!=accept(spec,load_json(root/'resource_diagnostic.json')):raise ValueError('Acceptance differs from measurements')
    return value
