"""Production setup and independent native student allocations; no dispatcher process."""
from pathlib import Path
import gc
import time
import torch
from hlt_classification.data.cache_contracts import load_json,write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_cross_message_hydra.execution import allocation
from .contracts import artifact,validate
from . import campaign,inputs,resources,submission,training


def receipt(spec,task):
    folder=Path(spec['root'])/task; record=load_json(folder/'receipt.json')
    validate(record,'TASK',dict(spec=spec['content_hash']))
    if (record['task']!=task or record['job_id']!=submission.accepted_jobs(spec,submission.setup_stage(spec) if task=='setup' else 'science').get(task)
            or len({r['path'] for r in record['outputs']})!=len(record['outputs'])): raise ValueError('Exact job/output ownership differs')
    for r in record['outputs']: validate_file_ref(r,root=folder)
    return record


def completed_setup(spec):
    r=receipt(spec,'setup'); names={x['path'] for x in r['outputs']}
    required={'acceptance.json','baseline_parity.json'}
    if spec.get('scope')=='direct_reconstruction_only': required|={'reconstruction_targets.json','reconstruction_HLT.json','reconstruction_OFFLINE.json'}
    elif spec.get('scope')=='group_C_only': required|={'probabilistic_targets.json','set_HLT.json','set_OFFLINE.json','density_centers.npz'}
    elif spec.get('scope')=='group_B_only': required|={'structured_targets.json','question_catalog.json'}
    else: required|={'hlt_keys.json','hlt_keys.npz','normalization_HLT.json','normalization_OFFLINE.json'}
    if not required<=names: raise ValueError('Incomplete setup inventory')
    return resources.accepted(spec)


def setup(spec,device='cuda'):
    folder=Path(spec['root'])/'setup'; folder.mkdir(exist_ok=False); began=time.perf_counter()
    inputs.Bundle(spec['bundle_root']).verify_all()
    train=inputs.hlt(spec,'train'); val=inputs.hlt(spec,'validation'); bank=inputs.embedding_bank(spec,train)
    shared_preparation=time.perf_counter()-began
    write_immutable_json(folder/'baseline_parity.json',resources.baseline(spec,train,device))
    if spec.get('scope')=='direct_reconstruction_only':
        from .reconstruction_targets import prepare
        prepare(spec,train)
    elif spec.get('scope')=='group_C_only':
        from .probabilistic_targets import prepare
        prepare(spec,train,bank)
    elif spec.get('scope')=='group_B_only':
        from .structured_targets import prepare
        prepare(spec,train)  # One-time target work is accounted separately from each fit.
    else:
        inputs.build_hlt_keys(spec,train,device)
        for source in ('HLT','OFFLINE'):
            p=inputs.raw(spec).load(source,train)
            write_immutable_json(folder/('normalization_'+source+'.json'),p.normalization)
            del p; gc.collect()
    resources.measure(spec,train,val,bank,device,shared_preparation_seconds=shared_preparation)


def run(spec,task):
    if task=='report' and spec.get('campaign_kind')=='R8_FOLLOWUP_V2':
        from .reconstruction_release import report
        return report(spec)
    if task=='release' and submission.automatic(spec):
        from .reconstruction_release import run
        return run(spec)
    if task not in ('setup',*spec['remaining']): raise ValueError('Unregistered, deferred or reused task')
    parent=campaign.check(spec); require_backend(); job=allocation(parent)
    if installed_environment()!=spec['reference']['environment'] or gpu_identity()!=spec['qualification']['acceptance']['gpu']:
        raise PermissionError('Qualified software/GPU differs')
    stage=submission.setup_stage(spec) if task=='setup' else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec,stage).get(task)==job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact job')
    claim=Path(spec['root'])/'claims'/task; claim.mkdir(parents=True,exist_ok=False)
    write_immutable_json(claim/'intent.json',artifact('RUN_CLAIM',parents=dict(spec=spec['content_hash']),task=task,job_id=job))
    if task=='setup': setup(spec)
    else:
        completed_setup(spec)
        root=Path(spec['root'])/task; root.mkdir(exist_ok=False)
        train=inputs.hlt(spec,'train'); val=inputs.hlt(spec,'validation'); bank=inputs.embedding_bank(spec,train)
        candidates=inputs.candidates(spec,task,train,'cuda')
        try: training.fit(root/'fit',spec,task,train,val,bank,candidates=candidates)
        finally:
            if hasattr(candidates,'close'): candidates.close()
    campaign.check(spec); root=Path(spec['root'])/task
    record=artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()])
    write_immutable_json(root/'receipt.json',record)
    return record
