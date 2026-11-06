"""Setup qualification and campaign-owned allocation workers."""
from pathlib import Path
import time
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from .contracts import ORDER,artifact,validate
from . import campaign,inputs,training,resources,submission,site


def receipt(spec,task):
    folder=Path(spec['root'])/task; r=load_json(folder/'receipt.json'); validate(r,'TASK',dict(spec=spec['content_hash']))
    stage='bootstrap' if task=='setup' else 'science'
    if (r['task']!=task or r['job_id']!=submission.accepted_jobs(spec,stage).get(task)
            or len({x['path'] for x in r['outputs']})!=len(r['outputs'])): raise ValueError('Exact job/output ownership differs')
    for ref in r['outputs']: validate_file_ref(ref,root=folder)
    return r


def completed_setup(spec):
    r=receipt(spec,'setup')
    required={'acceptance.json','baseline_parity.json','validation_embeddings.json','validation_embeddings.npz'}|{
        f'{prefix}_{task}/receipt.json' for task in ORDER for prefix in ('warm','mini')}
    if not required<={x['path'] for x in r['outputs']}: raise ValueError('Incomplete setup receipt')
    return resources.accepted(spec)


def setup(spec):
    folder=Path(spec['root'])/'setup'; folder.mkdir(exist_ok=False); began=time.perf_counter()
    inputs.Bundle(spec['bundle_root']).verify_all()
    inputs.donor.delta(spec['teacher_context']).verify_all(); inputs.donor.encoder_delta(spec['teacher_context']).verify_all()
    train=inputs.hlt(spec,'train'); val=inputs.hlt(spec,'validation'); bank=inputs.train_bank(spec,train)
    vbank=inputs.build(spec,val)
    write_immutable_json(folder/'baseline_parity.json',resources.baseline(spec,train,bank,'cuda'))
    resources.measure(spec,train,val,bank,vbank,time.perf_counter()-began)


def run(spec,task):
    if task not in (*ORDER,'setup','release'): raise ValueError('Unregistered task')
    campaign.check(spec)
    if task=='release': job=site.cpu_allocation(spec)
    else: require_backend(); job=site.allocation(spec)
    if installed_environment()!=spec['reference']['environment']: raise PermissionError('Installed software differs')
    stage='bootstrap' if task in ('setup','release') else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact job')
    if task=='release': return submission.finish(spec,job)
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    if task=='setup': setup(spec)
    else:
        a=completed_setup(spec)
        if a['gpu']!=gpu_identity(): raise PermissionError('Measured GPU differs')
        train=inputs.hlt(spec,'train'); val=inputs.hlt(spec,'validation')
        bank=inputs.train_bank(spec,train); vb=inputs.validation_bank(spec,val)
        training.fit(Path(spec['root'])/task/'fit',spec,task,train,val,bank,vb)
    campaign.check(spec); folder=Path(spec['root'])/task
    value=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value)
    return value
