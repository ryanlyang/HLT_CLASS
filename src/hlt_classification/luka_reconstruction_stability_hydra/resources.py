"""Fourteen-configuration common-start qualification, measured miniatures and full support."""
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
from .contracts import artifact,validate,cell,tasks,QUALIFICATION_PROTOCOL,QUALIFICATION_TASKS,qualification_tasks
from . import kernel as k,training,campaign,readouts,calibration
from .contracts import RECIPES,READOUTS

TOLERANCES=dict(loss_abs=1e-3,gradient_relative_l2=5e-4,buffer_abs=1e-4,
    parameters_relative_l2=5e-4,adamw_moments_relative_l2=5e-4)
POSITIONS=(3.,3.01,4.5,8.99,9.)


def setup_root(spec,lane):return Path(spec['root'])/('setup_'+lane)


def miniature(folder,spec,task,train,val,bank,targets,ctx):
    def indices(cache):
        first=[int(np.flatnonzero(cache.labels==c)[0]) for c in range(11)]
        return np.array(list(dict.fromkeys([*first,*range(min(139,len(cache)))]))[:min(139,len(cache))],np.int64)
    ti,vi=indices(train),indices(val);a,b=Subset(train,ti),Subset(val,vi)
    emb=SimpleNamespace(targets=bank.targets[ti],manifest=bank.manifest,identities=a.identities,labels=a.labels)
    target=SubsetCandidates(targets,ti)
    k.configure(ctx.world,campaign.hardware(cell(task)['lane'])['cpus'])
    r=training.fit(folder,spec,task,a,b,emb,target,ctx,technical=True,calibration_train=train)
    return dict(content_hash=r['content_hash'],fit_memory=r['fit_memory'],seconds=r['export_seconds']+
        sum(e['train_seconds']+e['validation_seconds'] for e in r['history']))


def scalar(losses):return losses['ce']+losses['weighted_embedding']+losses['decoder_weighted_reconstruction']


def probabilities(model,raw,device):
    """Same bounded HLT batch on every rank; eval-mode BN never synchronizes."""
    model.eval()
    with torch.no_grad():value=k.native(model,raw,device).softmax(-1).detach().cpu()
    k.clear(model)
    if not torch.isfinite(value).all():raise ValueError('Nonfinite qualification evaluation')
    return value


def replica_buffers(model,ctx):
    buffers={name:value.detach().cpu() for name,value in model.named_buffers()}
    gathered=ctx.all(buffers)
    comparisons=[k.buffer_differences(gathered[0],other) for other in gathered]
    return dict(maximum_abs=max((row['absolute'] for rows in comparisons for row in rows.values()),default=0.),
        integer_equal=all(row['integer_equal'] for rows in comparisons for row in rows.values()))


def compare_step(ref,model,opt,losses,raw,ctx):
    actual=k.snapshot(model,opt,scalar(losses))
    probability=probabilities(model,raw,ctx.device)
    row=k.compare(ref['after'],actual,TOLERANCES)
    row['eval_probability_abs']=float((probability-ref['probabilities']).abs().max())
    row['replica_buffers']=replica_buffers(model,ctx)
    return row


def qualification(spec,train,bank,targets,ctx,lane):
    """Five common-start transition updates per recipe; complete references in RAM.

    Every rank receives the full before state and independent serial after state.
    Retain hashes and named numerical comparisons, not 70 redundant tensor dumps.
    Any failed comparison is published before the gate is checked.
    """
    import hashlib
    import io
    root=setup_root(spec,lane);checks={};digests={}
    if lane!='pascal':raise ValueError('Only Pascal qualification is registered')
    for task in qualification_tasks(spec):
        serial_model,serial_opt=k.build(spec,task,0,1,ctx.device,parity=True) if ctx.rank==0 else (None,None)
        model,optimizer=k.build(spec,task,ctx.rank,ctx.world,ctx.device,parity=True)
        for i,pos in enumerate(POSITIONS):
            pieces=qualification_batches(train,bank,targets,task,i)
            def reference():
                before=k.qualification_state(serial_model,serial_opt)
                losses,_=k.update_group(serial_model,serial_opt,pieces,task,0,1,ctx.device,pos)
                value=dict(before=before,after=k.snapshot(serial_model,serial_opt,scalar(losses)),
                    probabilities=probabilities(serial_model,pieces[0][0],ctx.device))
                buffer=io.BytesIO();torch.save(value,buffer)
                value['sha256']=hashlib.sha256(buffer.getbuffer()).hexdigest();return value
            ref=ctx.main(reference)
            k.restore_qualification_state(model,optimizer,ref['before'])
            losses,_=k.update_group(model,optimizer,pieces,task,ctx.rank,ctx.world,ctx.device,pos)
            row=compare_step(ref,model,optimizer,losses,pieces[0][0],ctx)
            name=f'{task}_{i:02}';checks[name]=ctx.main(lambda:row);digests[name]=ref['sha256']
            record=artifact('PARITY_STEP',parents=dict(spec=spec['content_hash'],reference=ref['sha256']),
                task=task,step=i+1,position=pos,comparison=checks[name],same_start_state=True,
                reference_storage='transient_complete_CPU_state_broadcast_to_all_ranks')
            ctx.main(lambda:write_immutable_json(root/f'parity_{name}.json',record))
            if ctx.rank==0:print(f'STABILITY parity {name} position={pos:g} passed={row["passed"]}',flush=True)
            del ref
        del model,optimizer,serial_model,serial_opt;d.release_memory(ctx)
    value=artifact('QUALIFICATION',parents=dict(spec=spec['content_hash']),lane=lane,world_size=ctx.world,
        checks=checks,positions=list(POSITIONS),tolerances=TOLERANCES,dropout_disabled_for_parity=True,
        reference_digests=digests,protocol=QUALIFICATION_PROTOCOL)
    ctx.main(lambda:write_immutable_json(root/'qualification.json',value))
    check_qualification(spec,lane,value);return value


def check_qualification(spec,lane,value):
    import re
    validate(value,'QUALIFICATION',dict(spec=spec['content_hash']))
    names={f'{t}_{i:02}' for t in qualification_tasks(spec) for i in range(len(POSITIONS))}
    if (lane!='pascal' or value['lane']!=lane or value['world_size']!=4
        or set(value['checks'])!=names or set(value['reference_digests'])!=names or value['tolerances']!=TOLERANCES
        or value['dropout_disabled_for_parity'] is not True or value['positions']!=list(POSITIONS)
        or value['protocol']!=QUALIFICATION_PROTOCOL):raise ValueError('Qualification coverage differs')
    for row in value['checks'].values():check_comparison(row,gate=True)
    for task in qualification_tasks(spec):
        for i,pos in enumerate(POSITIONS):
            name=f'{task}_{i:02}';digest=value['reference_digests'][name]
            if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid numerical reference digest')
            record=artifact('PARITY_STEP',parents=dict(spec=spec['content_hash'],reference=digest),task=task,step=i+1,position=pos,
                comparison=value['checks'][name],same_start_state=True,reference_storage='transient_complete_CPU_state_broadcast_to_all_ranks')
            if load_json(setup_root(spec,lane)/f'parity_{name}.json')!=record:raise ValueError('Step diagnostic differs')


def check_comparison(row,*,gate):
    # Nonfinite state, counter mismatches and unsynchronized replicas are fatal.
    if any(not np.isfinite(row[key]) or row[key]<0 for key in (*TOLERANCES,'eval_probability_abs')):
        raise ValueError('Invalid numerical diagnostics')
    for b in row['buffers'].values():
        if b['integer_equal'] is not True or any(not np.isfinite(b[key]) or b[key]<0
                for key in ('absolute','relative_l2','reference_abs_max','actual_abs_max')):
            raise ValueError('Invalid named buffer diagnostics')
    if row['buffer_abs']!=max((b['absolute'] for b in row['buffers'].values()),default=0.):
        raise ValueError('Named buffer aggregate differs')
    replica=row['replica_buffers']
    if replica['integer_equal'] is not True or not np.isfinite(replica['maximum_abs']) or not 0<=replica['maximum_abs']<=TOLERANCES['buffer_abs']:
        raise ValueError('Distributed running buffers are not synchronized')
    passed=all(row[key]<=limit for key,limit in TOLERANCES.items())
    if row['passed'] is not passed:raise ValueError('Diagnostic tolerance outcome differs')
    if gate and not passed:raise ValueError('Same-start numerical diagnostic failed; preserve evidence')


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


def qualification_batches(train,bank,targets,task,index):
    # Distinct microbatches and a short final group, preserving their individual
    # normalization populations in both serial and distributed references.
    factor=cell(task)['accumulation'];out=[]
    for micro in range(factor):
        n=11 if micro==factor-1 else 16
        ix=(np.arange(n)+index*17+micro*19)%len(train)
        out.append(training.batch(train,bank,targets,ix,1,task))
    return out


def _groups(train,bank,targets,task,indices):
    return [training.batch(train,bank,targets,indices[start:start+128],1,task) for start in range(0,len(indices),128)]


def measure(spec,old,train,val,bank,targets,ctx,devices,load_seconds,lane):
    if ctx.world!=4 or ctx.device.type!='cuda':raise PermissionError('Four real registered GPUs required')
    root=setup_root(spec,lane);membership=readouts.save_membership(spec,train,ctx)
    temperature=calibration.prepare(spec,train,targets,ctx)
    qualification(spec,train,bank,targets,ctx,lane);replay(spec,old,val,ctx,lane)
    evolved=None
    if 'recovery_parent' in spec:
        from .recovery import evolved_probe,validate_probe
        evolved=evolved_probe(spec,train,bank,targets,ctx);validate_probe(spec,evolved)
        check_memory(spec,evolved['memory'],devices,lane)
    profiles={};minis={};order=np.random.default_rng(20261115).permutation(len(train))
    for task in tasks(lane,spec):
        mini=miniature(root/('mini_'+task),spec,task,train,val,bank,targets,ctx);minis[task]=mini
        if task not in qualification_tasks(spec):continue
        d.release_memory(ctx);torch.cuda.reset_peak_memory_stats(ctx.device)
        k.configure(ctx.world,campaign.hardware(lane)['cpus']);m,o=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
        effective=128*cell(task)['accumulation']
        def step(ix,position,probe=False):
            return k.update_group(m,o,_groups(train,bank,targets,task,ix),task,ctx.rank,ctx.world,ctx.device,position,probe=probe)
        stress=[np.argsort(lengths(train),kind='stable')[-128:],np.argsort(targets.counts,kind='stable')[-128:],order[:11],order[-(len(train)%effective):]]
        positions=(4.5,8.99,20.) if cell(task)['soft_assignment'] else (4.5,20.)
        for position in positions:
            for ix in stress:step(ix,position)
        phases={}
        for position in positions:
            step(order[:effective],position);step(order[effective:2*effective],position)
            samples=[d.timed(lambda i=i:step(order[i*effective:(i+1)*effective],position),ctx)[0] for i in range(2,8)]
            if evolved is not None and np.mean(evolved['phase_samples'][str(position)])>np.mean(samples):
                samples=evolved['phase_samples'][str(position)]
            phases[str(position)]=samples
        probe_seconds,_=d.timed(lambda:step(order[:effective],4.5,True),ctx)
        averages=readouts.Averages();averages.update(m,3.,effective)
        val_seconds,_=d.timed(lambda:readouts.evaluate_all(m,averages,train,val,membership,ctx,
            training.evaluate,k.native,DelphesParticleTransformer),ctx)
        memory=d.memory(ctx);memory['host_peak_bytes']=max(memory['host_peak_bytes'],mini['fit_memory']['host_peak_bytes'])
        for i,r in enumerate(memory['ranks']):r['gpu_peak_bytes']=max(r['gpu_peak_bytes'],mini['fit_memory']['ranks'][i]['gpu_peak_bytes'])
        profiles[cell(task)['recipe']]=dict(phase_samples=phases,step_seconds=max(float(np.mean(v)) for v in phases.values()),
            validation_seconds=val_seconds,probe_seconds=probe_seconds,load_seconds=load_seconds,memory=memory,
            full_hlt_support=True,full_offline_support=True,partial11=True,tail_rows=len(train)%effective,
            effective_batch=effective,train_rows=len(train),validation_rows=len(val),readouts=list(READOUTS))
        if ctx.rank==0:print(f'STABILITY {task}: phases={phases}, four readouts+BN={val_seconds:.1f}s',flush=True)
        del m,o,averages;d.release_memory(ctx)
        ctx.main(lambda:write_immutable_json(root/f'resource_{task}.json',artifact('RESOURCE_PARTIAL',parents=dict(spec=spec['content_hash']),profile=profiles[cell(task)['recipe']])))
    from .identity import validate_pairs
    ctx.main(lambda:validate_pairs(spec,{t:load_json(root/('mini_'+t)/'initialization.json') for t in tasks(lane,spec)},complete=True))
    used=ctx.main(lambda:sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    projected=ctx.main(lambda:sum(measured_fit_bytes(root/('mini_'+t),spec) for t in tasks(lane,spec)))
    value=artifact('RESOURCES',parents=dict(spec=spec['content_hash'],bn_membership=membership['content_hash'],soft_calibration=temperature['content_hash']),
        lane=lane,profiles=profiles,miniatures=minis,devices=devices,environment=installed_environment(),status='complete',
        storage=dict(setup_bytes=used,scientific_bytes=projected,required_bytes=math.ceil(1.25*(used+projected))+2**30))
    ctx.main(lambda:write_immutable_json(root/'resource_diagnostic.json',value))
    result=accept(spec,value,lane);ctx.main(lambda:write_immutable_json(root/'acceptance.json',result));return result


def measured_fit_bytes(folder,spec):
    # Real serialized states from production miniatures; validation arrays scale
    # to the full population, histories/Slurm diagnostics receive an explicit cap.
    states=sum(p.stat().st_size for p in folder.rglob('*.pt'))
    predictions=4*(spec['counts']['validation']*(32+8+11*4)+1024**2)
    startup=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    return states+predictions+startup+32*1024**2


def accept(spec,value,lane):
    validate(value,'RESOURCES')
    if (value['parents'].get('spec')!=spec['content_hash'] or set(value['parents'])!={'spec','bn_membership','soft_calibration'}
        or value['status']!='complete' or value['lane']!=lane or set(value['profiles'])!={cell(t)['recipe'] for t in tasks(lane,spec)}
        or set(value['miniatures'])!=set(tasks(lane,spec)) or value['environment']!=spec['environment']):raise ValueError('Incomplete hardware setup')
    check_devices(value['devices'],lane);minutes={}
    if 'recovery_parent' in spec:
        from .recovery import validate_probe
        evolved=load_json(setup_root(spec,lane)/'evolved_soft_probe.json');validate_probe(spec,evolved)
        check_memory(spec,evolved['memory'],value['devices'],lane)
        for profile in value['profiles'].values():
            if any(np.mean(profile['phase_samples'][p])<np.mean(samples) for p,samples in evolved['phase_samples'].items()):
                raise ValueError('Resource estimate omits evolved solver cost')
    storage=value['storage']
    if (storage['required_bytes']!=math.ceil(1.25*(storage['setup_bytes']+storage['scientific_bytes']))+2**30
        or min(storage.values())<=0 or storage['required_bytes']>spec['output_budget_bytes']):raise OSError('Measured 42-fit storage exceeds registered allowance')
    for task in tasks(lane,spec):
        p=value['profiles'][cell(task)['recipe']];mini=value['miniatures'][task];effective=128*cell(task)['accumulation']
        positions={'4.5','8.99','20.0'} if cell(task)['soft_assignment'] else {'4.5','20.0'}
        if (not all(p[k] is True for k in ('full_hlt_support','full_offline_support','partial11'))
            or p['tail_rows']!=spec['counts']['train']%effective or p['effective_batch']!=effective
            or p['train_rows']!=spec['counts']['train'] or p['validation_rows']!=spec['counts']['validation']
            or p['readouts']!=list(READOUTS) or set(p['phase_samples'])!=positions
            or any(len(v)!=6 or not np.isfinite(v).all() or min(v)<=0 for v in p['phase_samples'].values())
            or p['step_seconds']!=max(float(np.mean(v)) for v in p['phase_samples'].values())):raise ValueError('Resource coverage differs')
        check_memory(spec,p['memory'],value['devices'],lane);check_memory(spec,mini['fit_memory'],value['devices'],lane)
        if any(not np.isfinite(p[k]) or p[k]<=0 for k in ('validation_seconds','probe_seconds','load_seconds')) or mini['seconds']<=0:raise ValueError('Invalid timings')
        # Catch-up/soft matching operate only during positions3-9. Use measured
        # phase durations for those six passes and the hard joint path for24.
        # Every epoch still conservatively pays for a full-size final update.
        steps=math.ceil(p['train_rows']/effective)
        if cell(task)['soft_assignment'] or cell(task)['decoder_extra_steps']:
            active=max(float(np.mean(v)) for name,v in p['phase_samples'].items() if name!='20.0')
            training_seconds=steps*(6*active+24*float(np.mean(p['phase_samples']['20.0'])))
        else:training_seconds=30*steps*p['step_seconds']
        seconds=training_seconds+30*(p['validation_seconds']+p['probe_seconds'])
        seconds+=p['load_seconds']+2*p['validation_seconds']+2*mini['seconds']
        minutes[task]=max(60,math.ceil((spec['resource_policy']['safety_multiplier']*seconds+spec['resource_policy']['fixed_overhead_seconds'])/60))
        if minutes[task]>spec['fit_max_minutes']:raise RuntimeError('Measured walltime exceeds 48h cap; preserve diagnostics')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=value['content_hash']),lane=lane,
        minutes=minutes,devices=value['devices'],environment=value['environment'],technical_weights_discarded=True,storage=storage)


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
