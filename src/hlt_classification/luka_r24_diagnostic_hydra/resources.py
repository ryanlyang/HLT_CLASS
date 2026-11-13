"""Independent hardware qualification with twelve-update transition diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,load_npz_arrays,atomic_publish_bytes,deterministic_npz_bytes
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment,DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_next_supervision_hydra.inputs import SubsetCandidates,Bundle
from hlt_classification.luka_u100_contrastive.resources import Subset
from hlt_classification.luka_u100_multiview.views import lengths
from hlt_classification.luka_subclass_ddp_hydra import distributed as d
from .contracts import artifact,validate,cell,tasks,original_task
from . import kernel as k,training,original,campaign

TOLERANCES=dict(loss_abs=1e-3,gradient_relative_l2=5e-4,buffer_abs=1e-4,
    parameters_relative_l2=5e-4,adamw_moments_relative_l2=5e-4)
POSITIONS=(1.,2.,2.9,3.,3.01,3.5,4.,4.5,5.,5.9,6.,7.)


def setup_root(spec,lane):return Path(spec['root'])/('setup_'+lane)


def miniature(folder,spec,task,train,val,bank,targets,ctx):
    def indices(cache):
        first=[int(np.flatnonzero(cache.labels==c)[0]) for c in range(11)]
        return np.array(list(dict.fromkeys([*first,*range(min(139,len(cache)))]))[:min(139,len(cache))],np.int64)
    ti,vi=indices(train),indices(val);a,b=Subset(train,ti),Subset(val,vi)
    emb=SimpleNamespace(targets=bank.targets[ti],manifest=bank.manifest,identities=a.identities,labels=a.labels)
    target=SubsetCandidates(targets,ti)
    if cell(task)['worker']=='original':
        r,binding=original.fit(folder,spec,task,a,b,emb,target,ctx,technical=True)
        return dict(content_hash=r['content_hash'],fit_memory=binding['memory'],seconds=binding['elapsed_seconds'])
    k.configure(ctx.world,campaign.hardware(cell(task)['lane'])['cpus'])
    r=training.fit(folder,spec,task,a,b,emb,target,ctx,technical=True)
    return dict(content_hash=r['content_hash'],fit_memory=r['fit_memory'],seconds=r['export_seconds']+
        sum(e['train_seconds']+e['validation_seconds'] for e in r['history']))


def scalar(losses):return losses['ce']+losses['weighted_embedding']+losses['decoder_weighted_reconstruction']


def qualification(spec,train,bank,targets,ctx,lane):
    """Reference snapshots go to bounded setup storage, never retained on every rank."""
    root=setup_root(spec,lane);checks={};refs=[]
    # Single-card A6000 validates the new BF16 path against the original native
    # forward separately; distributed parity is meaningful only on Pascal.
    if lane=='pascal':
        for task in ('P0_C','P0_R'):
            def reference():
                m,o=k.build(spec,task,0,1,ctx.device,parity=True)
                paths=[]
                for i,pos in enumerate(POSITIONS):
                    ix=np.arange(i*17,i*17+(15 if i%2 else 16))
                    losses,_=k.update(m,o,training.batch(train,bank,targets,ix,1,task),task,0,1,ctx.device,pos)
                    path=root/'parity_reference'/f'{task}_{i:02}.pt'
                    path.parent.mkdir(exist_ok=True)
                    save_state(path,k.snapshot(m,o,scalar(losses)));paths.append(file_ref(path,root=root))
                    print(f'R24_DIAGNOSTIC serial parity {task} update={i+1}/{len(POSITIONS)} position={pos:g}',flush=True)
                del m,o;d.release_memory(ctx);return paths
            paths=ctx.main(reference);refs+=paths
            m,o=k.build(spec,task,ctx.rank,ctx.world,ctx.device,parity=True)
            for i,pos in enumerate(POSITIONS):
                ix=np.arange(i*17,i*17+(15 if i%2 else 16))
                losses,_=k.update(m,o,training.batch(train,bank,targets,ix,1,task),task,ctx.rank,ctx.world,ctx.device,pos)
                def compare():
                    ref=torch.load(root/paths[i]['path'],map_location='cpu',weights_only=True)
                    return k.compare(ref,k.snapshot(m,o,scalar(losses)),TOLERANCES)
                checks[f'{task}_{i:02}']=ctx.main(compare)
                record=artifact('PARITY_STEP',parents=dict(spec=spec['content_hash'],reference=paths[i]['sha256']),
                    task=task,step=i+1,position=pos,comparison=checks[f'{task}_{i:02}'])
                def publish_step():
                    write_immutable_json(root/f'parity_{task}_{i:02}.json',record)
                    print(f'R24_DIAGNOSTIC four-rank parity {task} update={i+1}: {record["comparison"]}',flush=True)
                ctx.main(publish_step)
            del m,o;d.release_memory(ctx)
    else:
        from hlt_classification.luka_cross_message_hydra.training import forward
        m,o=k.build(spec,'NB_R',0,1,ctx.device,parity=True)
        m.eval();raw=train.batch(np.arange(16))
        with torch.no_grad():
            expected=forward(m,raw,str(ctx.device)).float()
            actual=k.native(m,raw,ctx.device)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        checks['native_BF16_donor_forward']=dict(passed=True,maximum_abs=float((actual-expected).abs().max()))
        del m,o;d.release_memory(ctx)
    value=artifact('QUALIFICATION',parents=dict(spec=spec['content_hash']),lane=lane,world_size=ctx.world,
        checks=checks,positions=list(POSITIONS) if lane=='pascal' else [],tolerances=TOLERANCES,
        dropout_disabled_for_parity=True,reference_snapshots=refs)
    ctx.main(lambda:write_immutable_json(root/'qualification.json',value))
    check_qualification(spec,lane,value);return value


def check_qualification(spec,lane,value):
    from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
    validate(value,'QUALIFICATION',dict(spec=spec['content_hash']))
    names={f'{t}_{i:02}' for t in ('P0_C','P0_R') for i in range(len(POSITIONS))} if lane=='pascal' else {'native_BF16_donor_forward'}
    if (value['lane']!=lane or value['world_size']!=campaign.hardware(lane)['world_size']
        or set(value['checks'])!=names or value['tolerances']!=TOLERANCES or not value['dropout_disabled_for_parity']
        or value['positions']!=(list(POSITIONS) if lane=='pascal' else [])):
        raise ValueError('Qualification coverage differs')
    for row in value['checks'].values():
        if row['passed'] is not True:raise ValueError('Numerical diagnostic failed; preserve evidence')
        if lane=='pascal' and any(not np.isfinite(row[k]) or not 0<=row[k]<=v for k,v in TOLERANCES.items()):
            raise ValueError('Numerical tolerance exceeded')
        if lane=='a6000' and row['maximum_abs']!=0:raise ValueError('Native precision route differs')
    expected=[f'parity_reference/{t}_{i:02}.pt' for t in ('P0_C','P0_R') for i in range(len(POSITIONS))] if lane=='pascal' else []
    if [r['path'] for r in value['reference_snapshots']]!=expected:raise ValueError('Reference coverage differs')
    for ref in value['reference_snapshots']:validate_file_ref(ref,root=setup_root(spec,lane))


def check_devices(devices,lane):
    if len(devices)!=campaign.hardware(lane)['world_size']:raise ValueError('Wrong GPU count')
    for g in devices:
        names=('GeForce GTX 1080 Ti','NVIDIA GeForce GTX 1080 Ti') if lane=='pascal' else ('NVIDIA RTX A6000',)
        if g['name'] not in names or g['compute_capability']!=([6,1] if lane=='pascal' else [8,6]) or g['total_memory_bytes']<(10 if lane=='pascal' else 40)*2**30:
            raise ValueError('Wrong typed GPU hardware')


def check_memory(spec,memory,devices,lane):
    check_devices(devices,lane);hw=campaign.hardware(lane)
    if (len(memory['ranks'])!=hw['world_size'] or not 0<memory['host_peak_bytes']<spec['resource_policy']['host_fraction']*hw['memory_mb']*1024**2
        or memory['host_peak_bytes']<sum(r['peak_rss_bytes'] for r in memory['ranks'])):
        raise MemoryError('Aggregate host envelope exceeded')
    for m,g in zip(memory['ranks'],devices):
        if not 0<m['gpu_peak_bytes']<spec['resource_policy']['gpu_fraction']*g['total_memory_bytes'] or m['peak_rss_bytes']<=0:
            raise MemoryError('Per-rank memory envelope exceeded')


def measure(spec,old,train,val,bank,targets,ctx,devices,load_seconds,lane):
    if ctx.world!=campaign.hardware(lane)['world_size'] or ctx.device.type!='cuda':raise PermissionError('Real registered hardware required')
    root=setup_root(spec,lane);qualification(spec,train,bank,targets,ctx,lane)
    replay(spec,old,val,ctx,lane)
    profiles={};order=np.random.default_rng(20261113).permutation(len(train))
    stress=[np.argsort(lengths(train),kind='stable')[-128:],np.argsort(targets.counts,kind='stable')[-128:],order[:11],order[-(len(train)%128):]]
    for task in tasks(lane):
        mini=miniature(root/('mini_'+task),spec,task,train,val,bank,targets,ctx)
        d.release_memory(ctx);torch.cuda.reset_peak_memory_stats(ctx.device)
        is_old=cell(task)['worker']=='original'
        if is_old:
            original.configure();m=original.donor.new_model(old['imported'],original_task(task)).to(ctx.device)
            if not cell(task)['control']:targets.initialize(m)
            o=original.donor.optimizer(m)
        else:
            k.configure(ctx.world,campaign.hardware(lane)['cpus']);m,o=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
        def step(ix,probe=False):
            if is_old:
                return original.donor.step(m,o,train.batch(ix),bank.targets[ix],str(ctx.device),
                    candidates=None if cell(task)['control'] else targets,indices=ix,epoch=1,position=7.,probe=probe)
            return k.update(m,o,training.batch(train,bank,targets,ix,1,task),task,ctx.rank,ctx.world,ctx.device,7.,probe=probe)
        for ix in stress:step(ix)
        step(order[:128]);step(order[128:256])
        samples=[d.timed(lambda i=i:step(order[i*128:(i+1)*128]),ctx)[0] for i in range(2,14)]
        probe,_=d.timed(lambda:step(order[:128],True),ctx)
        def evaluate():
            if is_old:
                from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
                return evaluate_probabilities(val.labels,original.donor.predict(m,val,str(ctx.device)))
            return training.evaluate(m,val,ctx)
        val_seconds,_=d.timed(evaluate,ctx)
        memory=d.memory(ctx);memory['host_peak_bytes']=max(memory['host_peak_bytes'],mini['fit_memory']['host_peak_bytes'])
        for i,r in enumerate(memory['ranks']):r['gpu_peak_bytes']=max(r['gpu_peak_bytes'],mini['fit_memory']['ranks'][i]['gpu_peak_bytes'])
        profiles[task]=dict(step_samples=samples,step_seconds=float(np.mean(samples)),validation_seconds=val_seconds,
            probe_seconds=probe,load_seconds=load_seconds,miniature_seconds=mini['seconds'],memory=memory,
            miniature=mini['content_hash'],full_hlt_support=True,full_offline_support=True,partial11=True,
            tail_rows=len(train)%128,train_rows=len(train),validation_rows=len(val),precision=cell(task)['precision'],worker=cell(task)['worker'])
        if ctx.rank==0:print(f'R24_DIAGNOSTIC {task}: step={np.mean(samples):.3f}s validation={val_seconds:.1f}s',flush=True)
        del m,o;d.release_memory(ctx)
        value=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),lane=lane,profiles=profiles,devices=devices,
            environment=installed_environment(),status='complete' if len(profiles)==len(tasks(lane)) else 'partial')
        ctx.main(lambda:write_immutable_json(root/f'resource_{task}.json',value))
    ctx.main(lambda:write_immutable_json(root/'resource_diagnostic.json',value))
    result=accept(spec,value,lane);ctx.main(lambda:write_immutable_json(root/'acceptance.json',result));return result


def accept(spec,value,lane):
    validate(value,'RESOURCES',dict(spec=spec['content_hash']))
    if value['status']!='complete' or value['lane']!=lane or set(value['profiles'])!=set(tasks(lane)) or value['environment']!=spec['environment']:
        raise ValueError('Incomplete hardware setup')
    check_devices(value['devices'],lane);minutes={}
    for task,p in value['profiles'].items():
        if (not all(p[k] is True for k in ('full_hlt_support','full_offline_support','partial11'))
            or p['tail_rows']!=spec['counts']['train']%128 or p['train_rows']!=spec['counts']['train']
            or p['validation_rows']!=spec['counts']['validation'] or len(p['step_samples'])!=12
            or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0
            or p['step_seconds']!=float(np.mean(p['step_samples']))
            or p['precision']!=cell(task)['precision'] or p['worker']!=cell(task)['worker']):raise ValueError('Incomplete resource coverage')
        check_memory(spec,p['memory'],value['devices'],lane)
        if any(not np.isfinite(p[k]) or p[k]<=0 for k in ('validation_seconds','probe_seconds','load_seconds','miniature_seconds')):
            raise ValueError('Invalid measured timings')
        seconds=30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds']+p['probe_seconds'])
        seconds+=p['load_seconds']+2*p['validation_seconds']+2*p['miniature_seconds']
        minutes[task]=max(60,math.ceil((spec['resource_policy']['safety_multiplier']*seconds+spec['resource_policy']['fixed_overhead_seconds'])/60))
        if minutes[task]>spec['fit_max_minutes']:raise RuntimeError('Measured walltime exceeds 48h cap')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=value['content_hash']),lane=lane,
        minutes=minutes,devices=value['devices'],environment=value['environment'],technical_weights_discarded=True)


def accepted(spec,lane):
    root=setup_root(spec,lane);value=load_json(root/'acceptance.json')
    if value!=accept(spec,load_json(root/'resource_diagnostic.json'),lane):raise ValueError('Acceptance differs from measurements')
    return value


def replay(spec,old,val,ctx,lane):
    bundle=Bundle(old['bundle_root']);saved=load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    if any(not np.array_equal(saved[k],getattr(val,k)) for k in ('identities','labels')):raise ValueError('Replay rows differ')
    native=DelphesParticleTransformer().to(ctx.device).eval();ref=file_ref(bundle.path('reference/matched/checkpoint.pt'))
    native.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True),strict=True)
    actual=training.predict(native,val,ctx);del native;d.release_memory(ctx)
    def publish():
        folder=setup_root(spec,lane)
        atomic_publish_bytes(folder/'replay_predictions.npz',deterministic_npz_bytes(dict(identities=val.identities,labels=val.labels,probabilities=actual)))
        a,e=[evaluate_probabilities(val.labels,p)['macro_ovr_auc'] for p in (actual,saved['probabilities'])]
        result=artifact('REPLAY',parents=dict(spec=spec['content_hash']),checkpoint=ref,
            predictions=file_ref(folder/'replay_predictions.npz',root=folder),actual_auc=a,historical_auc=e,
            auc_abs=abs(a-e),tolerance=5e-5,rows=len(val),precision='fp32')
        write_immutable_json(folder/'replay.json',result)
        if abs(a-e)>5e-5:raise ValueError('Historical FP32 replay differs')
        return result
    return ctx.main(publish)


def check_replay(spec,old,lane):
    folder=setup_root(spec,lane);r=load_json(folder/'replay.json');validate(r,'REPLAY',dict(spec=spec['content_hash']))
    b=Bundle(old['bundle_root']);saved=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    if r['predictions']['path']!='replay_predictions.npz':raise ValueError('Replay path differs')
    actual=load_npz_arrays(validate_file_ref(r['predictions'],root=folder))
    if any(not np.array_equal(actual[k],saved[k]) for k in ('identities','labels')):raise ValueError('Replay identity differs')
    a,e=[evaluate_probabilities(x['labels'],x['probabilities'])['macro_ovr_auc'] for x in (actual,saved)]
    if (r['actual_auc']!=a or r['historical_auc']!=e or r['auc_abs']!=abs(a-e) or abs(a-e)>5e-5
            or r['tolerance']!=5e-5 or r['precision']!='fp32' or r['rows']!=spec['counts']['validation']
            or r['checkpoint']!=file_ref(b.path('reference/matched/checkpoint.pt'))):raise ValueError('Replay evidence differs')


