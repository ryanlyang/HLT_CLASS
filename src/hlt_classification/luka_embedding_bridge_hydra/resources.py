"""Real production miniatures and measured, quality-independent resource admission."""
from pathlib import Path
import math
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from .contracts import artifact,validate
from .registry import FITS,PRODUCERS,cell,teacher_condition
from . import inputs,banks,training,qualification,distributed as d,execution,qualified,regional_inputs


def selected_indices(ds,*,wide=False):
    selected={}
    for role in ('train','validation'):
        view=ds.view(role,'D000');p=ds.physical(role);n=len(view)
        ix=[np.flatnonzero(view.labels==k)[:24] for k in range(11)]
        if any(not len(a) for a in ix):raise ValueError('Miniature needs every native class')
        for source in ('HLT','OFFLINE'):
            ix.append(np.argsort(np.diff(p[source+'_offsets']),kind='stable')[-min(128,n):])
        if wide and role=='train':ix.append(np.random.default_rng(20261110).permutation(n)[:min(1536,n)])
        selected[role]=np.unique(np.concatenate(ix)).astype(np.int64)
    return selected


class SubsetTargets:
    def __init__(self,target,indices):
        self.target,self.indices=target,indices;self.parents=target.parents;self.parent=target.parent
    def batch(self,ix):return self.target.batch(self.indices[ix])


def miniature(folder,spec,task,ds,target,ctx):
    selected=selected_indices(ds);small=inputs.SubsetDataset(ds,selected)
    m,r=training.fit(folder,spec,task,small,SubsetTargets(target,selected['train']),ctx,technical=True,val_diagnostics=False)
    del m;d.release_memory(ctx)
    return r


def technical_target(spec,task,ds,root):
    parent=cell(task)['parent'];b=None
    if parent:
        folder,owner=qualified.miniature(spec,parent)
        b=banks.Bank(folder/'bank',owner,parent,ds.view('train','D000'),file_ref(folder/'selected_full.pt'),technical=True)
    return banks.Targets(spec,task,ds,parent_bank=b)


def measure(spec,ds,ctx,devices,load_seconds):
    if ctx.world!=execution.world_size(spec,'setup') or ctx.device.type!='cuda':raise PermissionError('Registered CUDA production setup required')
    root=Path(spec['root'])/'setup'
    recovered=qualified.profiles(spec,ds,devices)
    if set(recovered)!=set(qualified.REUSED):raise ValueError('Mixed-hardware setup requires all 29 completed Pascal profiles')
    old_devices=qualified.profile_devices(spec)
    task_devices={t:devices if t=='TP' else old_devices for t in FITS}
    frontend=ctx.main(lambda:regional_inputs.qualify(spec,ds))
    # Fail early on the most activation-heavy teacher, before numerical replay,
    # other families' miniatures, exports, or full validation. Prepared data reuse
    # still authenticates every input first.
    tp_target=banks.Targets(spec,'TP',ds)
    checkpoint=qualification.tp_checkpoint(spec,ds,tp_target,ctx)
    stress=tp_stress(spec,ds,tp_target,ctx,devices)
    del tp_target
    banks.prepare_u100(spec,ds,ctx);qualification.replay(spec,ds,ctx)
    selected=selected_indices(ds,wide=True);small=inputs.SubsetDataset(ds,selected)
    ctx.main(lambda:write_immutable_json(root/'mini_population.json',artifact('MINI_POPULATION',parents=dict(spec=spec['content_hash']),
        indices={k:v.tolist() for k,v in selected.items()},technical=True)))
    preparation=load_json(root/'preparation.json');validate(preparation,'PREPARATION',dict(spec=spec['content_hash']))
    profiles=dict(recovered);targets=lambda task:technical_target(spec,task,small,root)
    # Measure the expensive teacher first. Its time bound must pass before any
    # unaffected fitting or remaining qualification work is attempted.
    for task in ('TP',*[t for t in FITS if t!='TP' and t not in recovered]):
        target=targets(task);folder=root/('mini_'+task)
        fit_seconds,(m,report)=d.timed(lambda:training.fit(folder,spec,task,small,target,ctx,technical=True),ctx)
        export_seconds=0.
        if task in PRODUCERS:
            export_seconds,_=d.timed(lambda:banks.export(folder/'bank',spec,task,m,small,ctx,file_ref(folder/'selected_full.pt'),technical=True),ctx)
        del m;d.release_memory(ctx);torch.cuda.reset_peak_memory_stats(ctx.device)
        m,w,o=d.build(spec,task,ctx);n=len(small.view('train','D000'))
        order=np.random.default_rng(20261110).permutation(n);samples=[];coverage=[]
        # Exercise every teacher condition and source/scale at the largest support.
        for batch in range(6):
            source=teacher_condition(3,batch,regional=True)[0] if task=='TP' else 'HLT'
            lengths=np.diff(ds.physical('train')[source+'_offsets'])[selected['train']]
            ix=np.argsort(lengths,kind='stable')[-128:]
            d.step(m,w,o,task,small,target,ix,3,batch,3.,ctx)
            coverage.append(teacher_condition(3,batch,regional=task=='TP') if task in ('TR','TP') else 'native')
        for count in (11,spec['counts']['train']%128):
            if count:d.step(m,w,o,task,small,target,order[:count],3,6,3.,ctx)
        # One six-condition warm cycle, then two measured cycles; representative rows
        # are real and matched to miniature banks rather than fabricated targets.
        for batch in range(18):
            start=(batch*128)%(n-128+1);ix=order[start:start+128]
            seconds,_=d.timed(lambda:d.step(m,w,o,task,small,target,ix,3,batch,3.,ctx),ctx)
            if batch>=6:samples.append(seconds)
        if task=='TP':
            # A training-only bound can reject an impossible allocation without
            # paying for another full validation sweep. It can never admit a fit.
            lower=max(60,math.ceil((2*30*math.ceil(spec['counts']['train']/128)*float(np.mean(samples))+1800)/60))
            timing=artifact('TP_TRAINING_TIMING',parents=dict(spec=spec['content_hash']),
                step_samples=samples,step_seconds=float(np.mean(samples)),training_only_minutes=lower,
                excludes_validation_export_and_other_overhead=True,admission=False)
            ctx.main(lambda:write_immutable_json(root/'tp_training_timing.json',timing))
            if ctx.rank==0:print(f'EMBEDDING_BRIDGE TP training step={np.mean(samples):.3f}s; training-only request={lower}min',flush=True)
            if lower>spec['fit_max_minutes']:
                raise RuntimeError('TP training alone exceeds the 48-hour allocation; preserve tp_training_timing.json; '
                                   'full validation and unaffected fits were not rerun')
        probe,_=d.timed(lambda:d.step(m,w,o,task,small,target,order[:128],3,0,3.,ctx,diagnose=True),ctx)
        validation_seconds,_=d.timed(lambda:training.evaluate(m,task,ds,ctx),ctx)
        panel_seconds,_=d.timed(lambda:training.target_diagnostics(m,task,small,target,ctx),ctx)
        memory=d.memory(ctx);memory['host_peak_bytes']=max(memory['host_peak_bytes'],preparation['host_peak_bytes'],report['fit_memory']['host_peak_bytes'])
        for i,row in enumerate(memory['ranks']):row['gpu_peak_bytes']=max(row['gpu_peak_bytes'],report['fit_memory']['ranks'][i]['gpu_peak_bytes'])
        # Extrapolate actual serialized exports, with full-fit safety applied later.
        export_rows=sum(len(small.view(role,'D000')) for role in ('train','validation'))
        export_estimate=export_seconds*(spec['counts']['train']+4096)/export_rows
        profiles[task]=dict(step_samples=samples,step_seconds=float(np.mean(samples)),validation_seconds=validation_seconds,
            probe_seconds=probe,panel_seconds=panel_seconds*4096/len(small.view('validation','D000')),
            load_seconds=load_seconds+preparation['seconds'],startup_seconds=fit_seconds,
            export_measured_seconds=export_seconds,export_measured_rows=export_rows,export_estimate_seconds=export_estimate,
            checkpoint_seconds=max(1.,fit_seconds-sum(e['train_seconds']+e['validation_seconds']+e['diagnostic_seconds'] for e in report['history'])),
            memory=memory,technical_targets=True,miniature_complete=True,native_reload=True,largest_support_backward=True,
            partial_batch_backward=True,condition_coverage=coverage,train_rows=spec['counts']['train'],validation_rows=spec['counts']['validation'])
        print(f'EMBEDDING_BRIDGE resource {task}: step={np.mean(samples):.3f}s validation={validation_seconds:.1f}s export_estimate={export_estimate:.1f}s',flush=True) if ctx.rank==0 else None
        del m,w,o,target;d.release_memory(ctx)
        value=artifact('RESOURCES',parents=dict(spec=spec['content_hash']),profiles=profiles,devices=devices,
            task_devices=task_devices,
            tp_checkpoint=checkpoint,tp_stress=stress,
            crop_frontend=frontend,qualification_reuse=spec.get('qualification_reuse'),
            environment=installed_environment(),status='complete' if len(profiles)==len(FITS) else 'partial')
        ctx.main(lambda:write_immutable_json(root/f'resource_{task}.json',value))
        requested=request_minutes(profiles[task])
        if requested>spec['fit_max_minutes']:
            raise RuntimeError(f'{task}: measured {requested} minutes exceeds registered 48-hour ceiling; '
                               f'preserve resource_{task}.json; unaffected fits were not rerun')
    qualification.qualify(spec,small,targets,ctx)
    ctx.main(lambda:write_immutable_json(root/'resource_diagnostic.json',value))
    accepted_value=accept(spec,value);ctx.main(lambda:write_immutable_json(root/'acceptance.json',accepted_value))
    return accepted_value


def request_minutes(p):
    seconds=30*(math.ceil(p['train_rows']/128)*p['step_seconds']+p['validation_seconds'])
    seconds+=5*(p['probe_seconds']+p['panel_seconds'])+sum(p[k] for k in ('load_seconds','startup_seconds','checkpoint_seconds','export_estimate_seconds'))+3*p['validation_seconds']
    return max(60,math.ceil((2*seconds+1800)/60))


def profile_minutes(spec,diagnostic,*,historical=False):
    if historical:
        from hlt_classification.data.cache_contracts import validate_content_hash
        validate_content_hash(diagnostic,expected_contract='LUKA_EMBEDDING_BRIDGE_HYDRA_RESOURCES/v2',expected_schema_version=2)
        if diagnostic['parents']!=dict(spec=spec['content_hash']):raise ValueError('Historical resource parent differs')
        execution.check_devices(diagnostic['devices'],'1080ti_2_fp32')
        task_devices={t:diagnostic['devices'] for t in FITS}
    else:
        validate(diagnostic,'RESOURCES',dict(spec=spec['content_hash']))
        regional_inputs.check(diagnostic['crop_frontend'],spec)
        if diagnostic['qualification_reuse']!=spec.get('qualification_reuse'):raise ValueError('Resource reuse lineage differs')
        task_devices=diagnostic['task_devices']
        if set(task_devices)!=set(FITS) or task_devices['TP']!=diagnostic['devices']:
            raise ValueError('Incomplete task hardware measurements')
        execution.check_devices(diagnostic['devices'],execution.TP_PROFILE)
        for task in FITS:execution.check_devices(task_devices[task],execution.profile(spec,task))
        if spec.get('qualification_reuse') is not None:
            old=qualified.check(spec['qualification_reuse'],spec)
            original=load_json(Path(old['root'])/'setup/resource_diagnostic.json')
            if any(task_devices[t]!=original['devices'] or diagnostic['profiles'][t]!=original['profiles'][t] for t in qualified.REUSED):
                raise ValueError('Recovered profiles changed')
    qualification.check_tp_checkpoint(diagnostic['tp_checkpoint'],spec,historical=historical)
    check_tp_stress(spec,diagnostic['tp_stress'],diagnostic['devices'])
    if (diagnostic['status']!='complete' or set(diagnostic['profiles'])!=set(FITS)
            or diagnostic['environment']!=spec['reference']['environment']):raise ValueError('Incomplete measured setup')
    execution.check_devices(diagnostic['devices']);minutes={}
    for task,p in diagnostic['profiles'].items():
        if (not all(p[k]is True for k in ('technical_targets','miniature_complete','native_reload','largest_support_backward','partial_batch_backward'))
                or p['train_rows']!=spec['counts']['train'] or p['validation_rows']!=spec['counts']['validation']
                or len(p['step_samples'])!=12 or not np.isfinite(p['step_samples']).all() or min(p['step_samples'])<=0
                or p['step_seconds']!=float(np.mean(p['step_samples']))):raise ValueError('Incomplete measurement coverage')
        expected=[teacher_condition(3,i,regional=task=='TP') if task in ('TR','TP') else 'native' for i in range(6)]
        expected=[list(x) if isinstance(x,tuple) else x for x in expected]
        actual=[list(x) if isinstance(x,tuple) else x for x in p['condition_coverage']]
        if actual!=expected:raise ValueError('Teacher condition coverage differs')
        check_memory(spec,p['memory'],task_devices[task])
        times=[p[k] for k in ('validation_seconds','probe_seconds','panel_seconds','load_seconds','startup_seconds','checkpoint_seconds')]
        if not np.isfinite(times).all() or min(times)<=0:raise ValueError('Invalid required timing')
        if task in PRODUCERS:
            e=p['export_measured_seconds']*(spec['counts']['train']+4096)/p['export_measured_rows']
            if not np.isfinite(e) or e<=0 or e!=p['export_estimate_seconds']:raise ValueError('Unmeasured producer export')
        elif p['export_measured_seconds']!=0 or p['export_estimate_seconds']!=0:raise ValueError('Unexpected target export')
        minutes[task]=request_minutes(p)
    return minutes


def accept(spec,diagnostic):
    minutes=profile_minutes(spec,diagnostic)
    for task in FITS:
        if minutes[task]>spec['fit_max_minutes']:raise RuntimeError(f'{task}: measured {minutes[task]} minutes exceeds registered 48-hour ceiling; preserve diagnostics')
    return artifact('ACCEPTANCE',parents=dict(spec=spec['content_hash'],diagnostic=diagnostic['content_hash']),minutes=minutes,
        devices=diagnostic['devices'],task_devices=diagnostic['task_devices'],environment=diagnostic['environment'],technical_weights_discarded=True)


def check_memory(spec,memory,devices):
    """Aggregate host and per-GPU limits for the actual registered rank count."""
    execution.check_devices(devices)
    if (len(memory['ranks'])!=len(devices) or not 0<memory['host_peak_bytes']<.85*spec['memory_mb']*1024**2
            or memory['host_peak_bytes']<sum(m['peak_rss_bytes'] for m in memory['ranks'])):
        raise MemoryError('Aggregate host envelope exceeded')
    for m,g in zip(memory['ranks'],devices):
        if (not np.isfinite([m['gpu_peak_bytes'],m['peak_rss_bytes']]).all()
                or not 0<m['gpu_peak_bytes']<.88*g['total_memory_bytes'] or not 0<m['peak_rss_bytes']):
            raise MemoryError('Rank memory envelope exceeded')


def tp_stress(spec,ds,target,ctx,devices):
    """Global128 real longest-support jets, then uneven rank splits, all scales."""
    d.release_memory(ctx);torch.cuda.reset_peak_memory_stats(ctx.device)
    m,w,o=d.build(spec,'TP',ctx);cases=[]
    counts=list(dict.fromkeys([128,11,spec['counts']['train']%128]))
    try:
        for count in counts:
            if not count:continue
            for batch in range(6):
                source,scale=teacher_condition(3,batch,regional=True)
                lengths=np.diff(ds.physical('train')[source+'_offsets'])
                ix=np.argsort(lengths,kind='stable')[-count:]
                if ctx.rank==0:print(f'EMBEDDING_BRIDGE TP early backward: {source}/{scale}; global batch={count}',flush=True)
                seconds,_=d.timed(lambda:d.step(m,w,o,'TP',ds,target,ix,3,batch,3.,ctx),ctx)
                cases.append(dict(condition=[source,scale],rows=count,seconds=seconds,
                    maximum_particles=int(lengths[ix].max())))
        memory=d.memory(ctx)
    finally:
        del m,w,o;d.release_memory(ctx)
    value=artifact('TP_STRESS',parents=dict(spec=spec['content_hash']),cases=cases,memory=memory,
        selection='longest_full_source_train_support',technical_weights_discarded=True)
    ctx.main(lambda:write_immutable_json(Path(spec['root'])/'setup/tp_stress.json',value))
    check_tp_stress(spec,value,devices)
    if ctx.rank==0:print('EMBEDDING_BRIDGE TP early memory passed: '+str(memory),flush=True)
    return value


def check_tp_stress(spec,value,devices):
    validate(value,'TP_STRESS',dict(spec=spec['content_hash']))
    expected=[(list(teacher_condition(3,batch,regional=True)),count)
        for count in dict.fromkeys([128,11,spec['counts']['train']%128]) if count for batch in range(6)]
    if ([(x['condition'],x['rows']) for x in value['cases']]!=expected
            or value['selection']!='longest_full_source_train_support'
            or value['technical_weights_discarded']is not True
            or any(not np.isfinite(x['seconds']) or x['seconds']<=0 or x['maximum_particles']<1 for x in value['cases'])):
        raise ValueError('Incomplete TP early memory evidence')
    check_memory(spec,value['memory'],devices)
    return value


def accepted(spec):
    root=Path(spec['root'])/'setup';value=load_json(root/'acceptance.json')
    if value!=accept(spec,load_json(root/'resource_diagnostic.json')):raise ValueError('Acceptance differs from evidence')
    return value


def check_cpu(spec,value):
    validate(value,'CPU_RESOURCES',dict(spec=spec['content_hash']))
    full_calls=sum(2+len(training.component_names(t)) for t in FITS)
    names=('release_seconds','reference_seconds','miniature_validation_seconds','full_metric_seconds','subset_metric_seconds')
    if (any(not np.isfinite(value[k]) or value[k]<=0 for k in names)
            or value['full_metric_calls']!=full_calls or value['subset_metric_calls']!=len(FITS)
            or value['method']!='real_metrics_and_miniature_receipts_extrapolated'
            or value['train_rows']!=spec['counts']['train'] or value['validation_rows']!=spec['counts']['validation']):
        raise ValueError('Incomplete CPU resource evidence')
    estimate=value['reference_seconds']+15*value['miniature_validation_seconds']+full_calls*value['full_metric_seconds']+len(FITS)*value['subset_metric_seconds']
    if value['report_estimate_seconds']!=estimate:raise ValueError('CPU report estimate differs')
    for task,seconds,peak in (('release',value['release_seconds'],value['release_peak_bytes']),
            ('report',estimate+value['release_seconds'],value['report_peak_bytes'])):
        site=spec['cpu_site' if task=='release' else 'report_site']
        if not np.isfinite(peak) or not 0<peak<site['memory_mb']*2**20*.85:
            raise MemoryError(task+': CPU process/launcher memory exceeds allocation; preserve diagnostics')
        if 2*seconds+600>spec[task+'_minutes']*60:
            raise RuntimeError(task+': measured CPU bound exceeds registration; preserve diagnostics')
    return value
