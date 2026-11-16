"""Separate three-fit recovery; immutable original campaign remains authoritative."""
import importlib
import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from .contracts import artifact,validate
from .transport_v2 import PROTOCOL

PARENT_COMMIT='3ccc35ae709c63d8833cb0388c618b53dcde5b9e'
TASKS=('SOFT_s7','SOFT_s8','SOFT_s9')


def runtime(parent):
    folder=Path(parent['project_dir'])/'src/hlt_classification/luka_reconstruction_stability_hydra'
    name='stability_soft_recovery_parent'
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve()!=folder/'__init__.py':raise ValueError('Ambiguous parent runtime')
    else:
        loader=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(loader);sys.modules[name]=module;loader.loader.exec_module(module)
    return lambda part:importlib.import_module(name+'.'+part)


def parent(spec):
    validate_file_ref(spec['recovery_parent']);p=load_json(spec['recovery_parent']['path'])
    if p['source_commit']!=PARENT_COMMIT:raise ValueError('Recovery requires the original 3ccc35ae stability campaign')
    return p


def registration(p):
    from .campaign import registration as base
    result=base();result['tasks']=list(TASKS)
    result['recipe']=artifact('RECOVERY_RECIPE',parents=dict(original_recipe=p['recipe']['content_hash']),
        original_recipe=p['recipe'],tasks=list(TASKS),soft_solver=PROTOCOL,
        restart='fresh_same_seed_30_epochs',temperature_scale='reuse_exact_parent_calibration')
    return result


def failed_evidence(p):
    r=runtime(p);jobs=r('submission').accepted_jobs(p,'science_pascal');out={}
    for task in TASKS:
        job=jobs[task]
        if r('submission').accounting(job)!=['FAILED','1:0']:raise ValueError('Recovery requires the exact failed SOFT jobs')
        folder=Path(p['root'])/task
        if (folder/'receipt.json').exists() or (folder/'fit/receipt.json').exists():raise ValueError('Completed fit cannot be replaced')
        log=Path(p['root'])/(task+'-'+job+'.out')
        if 'Soft assignment did not converge:' not in log.read_text():raise ValueError('Unexpected failure; inspect before recovery')
        initial=load_json(folder/'fit/initialization.json');r('identity').validate_initial(p,task,initial,4)
        out[task]=dict(job=job,log=file_ref(log),initialization=file_ref(folder/'fit/initialization.json'))
    return out


def create(root,parent_spec,project,commit,quota_gib):
    from . import campaign
    ref=file_ref(Path(parent_spec).resolve());p=parent(dict(recovery_parent=ref));r=runtime(p)
    print('Authenticating original setup, failed SOFT jobs and frozen calibration; this can take several minutes.',flush=True)
    r('campaign').check(p,full=True,check_storage=False);r('worker').completed_setup(p,'pascal')
    failures=failed_evidence(p);snap=campaign.source(project,commit)
    root=Path(root).absolute();project=Path(project).resolve()
    old=load_json(p['original_spec']['path'])
    protected=[Path(p['root']),Path(p['project_dir']),project]+[Path(old[k]) for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or
        any(x.is_symlink() for x in (root,*root.parents)) or any(root.is_relative_to(x.resolve()) or x.resolve().is_relative_to(root) for x in protected)):
        raise ValueError('Use a fresh, separate recovery root')
    if type(quota_gib) is not int or quota_gib<6:raise ValueError('Reserve six personally available GiB for the recovery')
    inherited={k:deepcopy(p[k]) for k in ('original_spec','benchmark_spec','benchmark_outputs','environment_dir','environment','counts','maximum',
        'imported','recovery_anchors','reference','embedding_hash','foundation_hash')}
    value=artifact('RECOVERY_SPEC',parents=dict(original=p['parents']['original'],recovery=p['content_hash'],source=snap['content_hash']),
        **registration(p),**inherited,root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snap,
        recovery_parent=ref,failures=failures,soft_solver=PROTOCOL,quota_gib=quota_gib,output_budget_bytes=5*2**30,
        reused_calibration=file_ref(Path(p['root'])/'setup_pascal/soft_calibration.json'),
        reused_membership=file_ref(Path(p['root'])/'setup_pascal/bn_membership.json'))
    campaign.storage(value);root.mkdir();write_immutable_json(root/'campaign_spec.json',value);return value


def check(spec,*,full=False,check_storage=True):
    from . import campaign
    validate(spec,'RECOVERY_SPEC');p=parent(spec);r=runtime(p)
    old=r('campaign').check(p,full=full,check_storage=False)
    snap=campaign.source(spec['project_dir'],spec['source_commit'])
    for ref in (spec['reused_calibration'],spec['reused_membership']):validate_file_ref(ref)
    for entry in spec['failures'].values():
        validate_file_ref(entry['log']);validate_file_ref(entry['initialization'])
    expected=dict(original=p['parents']['original'],recovery=p['content_hash'],source=snap['content_hash'])
    if (spec['parents']!=expected or spec['source_snapshot']!=snap or spec['soft_solver']!=PROTOCOL or
        any(spec.get(k)!=v for k,v in registration(p).items()) or spec['output_budget_bytes']!=5*2**30 or spec['quota_gib']<6 or
        any(spec[k]!=p[k] for k in ('original_spec','benchmark_spec','benchmark_outputs','environment_dir','environment','counts','maximum','imported',
                                  'recovery_anchors','reference','embedding_hash','foundation_hash')) or
        spec['reused_calibration']!=file_ref(Path(p['root'])/'setup_pascal/soft_calibration.json') or
        spec['reused_membership']!=file_ref(Path(p['root'])/'setup_pascal/bn_membership.json') or set(spec['failures'])!=set(TASKS) or
        load_json(Path(spec['root'])/'campaign_spec.json')!=spec):raise ValueError('Recovery identity differs')
    jobs=r('submission').accepted_jobs(p,'science_pascal')
    for task,entry in spec['failures'].items():
        folder=Path(p['root'])/task
        if entry!=dict(job=jobs[task],log=file_ref(Path(p['root'])/(task+'-'+jobs[task]+'.out')),initialization=file_ref(folder/'fit/initialization.json')):
            raise ValueError('Recovery failure lineage differs')
    if check_storage:campaign.storage(spec)
    return old


def calibration(spec):
    p=parent(spec);value=runtime(p)('calibration').load(p)
    # Original scale, all 512 identities, and original initialization are retained.
    from hlt_classification.data.cache_contracts import with_content_hash
    return with_content_hash(dict(value,parents=dict(spec=spec['content_hash']),content_hash=value['content_hash'],
        reused_from=value['content_hash'],solver_protocol=PROTOCOL))


def check_initial(spec,task,initial):
    ref=spec['failures'][task]['initialization'];validate_file_ref(ref);old=load_json(ref['path'])
    for key in ('components','seeds','dropout_rank_seeds','epoch1_order_hash','world_size'):
        if initial[key]!=old[key]:raise ValueError('Recovery starting state differs from failed fit: '+key)


def check_membership(spec,value):
    validate_file_ref(spec['reused_membership']);old=load_json(spec['reused_membership']['path'])
    if any(value[k]!=old[k] for k in ('rows','domain','role','indices','identities','global_batch','dropout')):
        raise ValueError('Recovery BN membership differs')


def controls(spec):
    p=parent(spec);r=runtime(p);reports={};initial={};populations={}
    for task in ('C128_s7','C128_s8','C128_s9'):
        folder=Path(p['root'])/task
        if not (folder/'receipt.json').exists():continue
        r('worker').receipt(p,task)
        reports[task],populations[task]=r('training').validate_fit(folder/'fit',p,task)
        initial[task]=load_json(folder/'fit/initialization.json')
    return reports,initial,populations


def evolved_probe(spec,train,bank,targets,ctx):
    """Exercise the real evolved-cost failure regime before releasing replacements.

    Four TRAIN passes for seed8, whose original marginal AND gap checks failed.
    No validation or checkpoint selection. These weights are always discarded.
    """
    import math
    import time
    import numpy as np
    from . import kernel as k,training
    from .contracts import seeds
    from .identity import initial_record
    from .transport_v2 import solve,ConvergenceError
    from hlt_classification.luka_subclass_ddp_hydra import distributed as d
    from hlt_classification.context_fusion.worker import save_state
    task='SOFT_s8';root=Path(spec['root'])/'setup_pascal'
    if len(train)!=spec['counts']['train']:raise ValueError('Full TRAIN required for evolved solver probe')
    if ctx.rank==0:print('SOFT_RECOVERY: four discarded seed8 TRAIN passes before automatic release.',flush=True)
    # Numerical stress is independent of labels and exercised on every GPU.
    cases=[]
    import torch
    for n,scale in ((30,10.),(298,10.),(20,100.)):
        generator=torch.Generator().manual_seed(21)
        cost=(torch.rand(298,n,generator=generator,dtype=torch.float64)*scale).to(ctx.device)
        mass=torch.ones(n,device=ctx.device,dtype=torch.float64);mass[-1]=299-n
        _,info=solve(cost,mass,.01);cases.append(info)
    model,opt=k.build(spec,task,ctx.rank,ctx.world,ctx.device)
    ctx.main(lambda:check_initial(spec,task,initial_record(spec,task,model,ctx.world)))
    sampler=seeds(spec,task)['sampler'];steps=math.ceil(len(train)/128);began=time.perf_counter()
    last={}
    for epoch in range(1,5):
        order=np.random.default_rng(np.random.SeedSequence([sampler,epoch])).permutation(len(train))
        for start in range(0,len(train),128):
            ix=order[start:start+128];position=epoch-1+(start+len(ix))/len(train)
            try:
                _,last=k.update(model,opt,training.batch(train,bank,targets,ix,epoch,task),task,ctx.rank,ctx.world,ctx.device,position)
            except ConvergenceError as exc:
                save_state(root/f'transport_failure_rank{ctx.rank}.pt',dict(exc.problem,task=task,
                    spec_hash=spec['content_hash'],epoch=epoch,position=position,indices=ix.tolist()))
                raise
            if ctx.rank==0 and (start//128%200==0 or start+128>=len(train)):
                print(f'SOFT_RECOVERY evolved probe epoch={epoch}/4 update={start//128+1}/{steps}',flush=True)
    phases={}
    for position in (4.5,8.99,20.):
        def step(i):return k.update(model,opt,training.batch(train,bank,targets,order[i*128:(i+1)*128],4,task),
            task,ctx.rank,ctx.world,ctx.device,position)
        step(0);step(1)
        phases[str(position)]=[d.timed(lambda i=i:step(i),ctx)[0] for i in range(2,8)]
    value=artifact('EVOLVED_SOFT_PROBE',parents=dict(spec=spec['content_hash']),task=task,train_passes=4,
        train_rows=len(train),updates=4*steps,discarded=True,validation_accessed=False,
        phase_samples=phases,memory=d.memory(ctx),seconds=max(ctx.all(time.perf_counter()-began)),
        numerical_cases=ctx.all(cases),solver=PROTOCOL)
    value=ctx.main(lambda:value);ctx.main(lambda:write_immutable_json(root/'evolved_soft_probe.json',value))
    del model,opt;d.release_memory(ctx);return value


def validate_probe(spec,value):
    import math
    import numpy as np
    validate(value,'EVOLVED_SOFT_PROBE',dict(spec=spec['content_hash']))
    if (value['task']!='SOFT_s8' or value['train_passes']!=4 or value['train_rows']!=spec['counts']['train'] or
        value['updates']!=4*math.ceil(spec['counts']['train']/128) or value['discarded'] is not True or
        value['validation_accessed'] is not False or value['solver']!=PROTOCOL or value['seconds']<=0 or
        set(value['phase_samples'])!={'4.5','8.99','20.0'} or len(value['numerical_cases'])!=4):raise ValueError('Incomplete evolved probe')
    for values in value['phase_samples'].values():
        if len(values)!=6 or not np.isfinite(values).all() or min(values)<=0:raise ValueError('Invalid evolved timings')
    for rank in value['numerical_cases']:
        if len(rank)!=3:raise ValueError('Missing solver stress case')
        for case in rank:
            if (not np.isfinite([case['marginal_residual'],case['normalized_gap']]).all() or
                not 0<=case['marginal_residual']<=1e-4 or not 0<=case['normalized_gap']<=1e-4):raise ValueError('Solver stress did not pass')
