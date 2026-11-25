"""Actual GPU communication and native/auxiliary parity before full decoding."""
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,require_sha256,with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.operations.hydra_reconstruction_gpu_diagnostic import SyntheticCache
from . import contracts as c,resources,transport,diagnostics as log

MINUTES=5
COLLECTIVE_SECONDS=45


def policy():
    return with_content_hash(dict(contract=c.PREFIX+'_EARLY_GPU_POLICY/v2',schema_version=2,parents={},**c.FLAGS,
        before='population_replay_and_all_particle_preparation',gpu_transport=transport.policy(),
        execution='separate_srun_step_with_hard_walltime',minutes=MINUTES,
        collective_seconds=COLLECTIVE_SECONDS,control_backend='gloo',tensor_backend='nccl',
        control_callback='broadcast_then_host_barrier',synthetic_only=True,
        qualification_arms=list(c.BASE_ARMS),science_arms='current_task_only',
        initializations='authenticated_canonical_untrained_starts',outputs='discarded_technical_updates'))


def folder(spec,job):return Path(spec['root'])/'early_gpu'/job


def target(index,maximum):
    if maximum<1:raise ValueError('Positive reconstruction support required')
    n=1+int(index)%min(4,maximum);x=np.zeros((n,22),np.float32)
    x[:,0]=.1;x[:,4]=1;x[np.arange(n),5+np.arange(n)%6]=1
    x[np.arange(n),11+np.arange(n)%3]=1;x[:,18:22]=1
    return dict(values=x,valid=np.ones((n,4),bool),mean=np.zeros(22),scale=np.ones(22),
        frame=np.array([0.,0.,20.]),aggregate=np.array([1.,0.,0.,2.,1.]))


def probe_tasks(task):
    return [f'{arm}_s{c.SEEDS[0]}' for arm in c.BASE_ARMS] if task=='qualify' else [task]


def run(spec,task,ctx,job):
    transport.check()
    if ctx.world!=4:raise PermissionError('Early acceptance requires four actual ranks')
    root=folder(spec,job);cache=SyntheticCache()
    log.mark('EARLY_GPU CPU control broadcast/barrier')
    ctx.main(lambda:None)
    log.mark('EARLY_GPU NCCL tensor SUM and broadcast')
    values=torch.full((8,),float(ctx.rank+1),device=ctx.device);dist.all_reduce(values);ctx.sync()
    if not bool(torch.all(values==10)):raise ValueError('Early GPU tensor SUM differs')
    values.fill_(42 if ctx.rank==0 else 0);dist.broadcast(values,src=0);ctx.sync()
    if not bool(torch.all(values==42)):raise ValueError('Early GPU tensor broadcast differs')
    from hlt_classification.luka_sixclass_balanced_oscar.model import installed_parity
    log.mark('EARLY_GPU installed-Weaver parity')
    installed=installed_parity(cache,device=ctx.device)
    if installed.get('passed') is not True:raise ValueError('Early installed-Weaver parity failed')
    weaver=ctx.all(installed['content_hash'])
    class Targets:
        def batch(self,ix):return [target(i,spec['maximum']) for i in ix]
    parents=dict(spec=spec['content_hash'],policy=policy()['content_hash']);proofs={}
    for candidate in probe_tasks(task):
        log.mark('EARLY_GPU native/full-weight parity '+candidate)
        targets=Targets() if c.cell(candidate)['reconstruction'] else None
        proof=resources.parity(spec,candidate,cache,targets,None,ctx)
        if targets is not None:
            log.mark('EARLY_GPU delayed-weight parity '+candidate)
            proof['delay']=resources.parity(spec,candidate,cache,targets,None,ctx,position=2.)
        row=c.artifact('EARLY_GPU_ARM',parents=parents,job=job,task=candidate,proof=proof,
            synthetic_only=True,discarded=True)
        path=root/(candidate+'.json');ctx.main(lambda:write_immutable_json(path,row));proofs[candidate]=file_ref(path,root=root)
    result=c.artifact('EARLY_GPU',parents=parents,job=job,task=task,world=4,
        transport_sha256=transport.policy()['content_hash'],
        control_backend='gloo',tensor_backend='nccl',synthetic_only=True,discarded=True,
        arms=proofs,installed_weaver=weaver,passed=True)
    ctx.main(lambda:write_immutable_json(root/'complete.json',result))
    log.mark('EARLY_GPU PASSED: full input preparation may begin')
    return result


def check(spec,task,job):
    root=folder(spec,job);parents=dict(spec=spec['content_hash'],policy=policy()['content_hash'])
    result=load_json(root/'complete.json');c.validate(result,'EARLY_GPU',parents)
    if (result['job']!=job or result['task']!=task or result['world']!=4 or result['passed'] is not True
            or result['synthetic_only'] is not True or result['discarded'] is not True
            or result['control_backend']!='gloo' or result['tensor_backend']!='nccl'
            or result.get('transport_sha256')!=transport.policy()['content_hash']
            or set(result['arms'])!=set(probe_tasks(task)) or len(result['installed_weaver'])!=4):
        raise ValueError('Early GPU receipt differs')
    for digest in result['installed_weaver']:require_sha256(digest,name='installed-Weaver proof')
    for name,ref in result['arms'].items():
        path=validate_file_ref(ref,root=root);row=load_json(path);c.validate(row,'EARLY_GPU_ARM',parents)
        checks=[row['proof']]+([row['proof']['delay']] if c.cell(name)['reconstruction'] else [])
        if (row['job']!=job or row['task']!=name or row['synthetic_only'] is not True
                or row['discarded'] is not True or any(len(p['ranks'])!=4 or
                p['position']!=position or p['tolerances']!=resources.TOLERANCES or
                any(r['passed'] is not True for r in p['ranks']) for p,position in zip(checks,(8.,2.)))):
            raise ValueError('Incomplete early GPU arm proof')
    return result
