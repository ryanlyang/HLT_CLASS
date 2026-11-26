"""Checkpointed CONCAT execution after the authenticated 48-hour estimate stop."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from . import contracts as c, gradient_recovery, startup

PREVIOUS_HASH='2fea86c9d8109a20d1738b096095d3287630babdf03407f3b4c265809f5cb8fc'
PREVIOUS_COMMIT='2bf345f96ffd45f46f0cd9e6934798af62349401'
PREVIOUS_ROOT='/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_wide_gradient_2bf345f9_r1'
FAILED_JOB='9620918'
MEASUREMENTS_HASH='1b8cb8d35b6d554b93964e23a6671ff0ebe64b3cabaec15c612756a53b5183dc'
FAILURE_MESSAGE='Measured remaining work exceeds walltime; retained diagnostic requires explicit resource revision'
POLICY=dict(stage='CONCAT',windows=[[1,10],[11,20]],schedule_epochs=20,
    continuation='model_optimizer_cpu_cuda_rng_and_global_update',
    selected_checkpoint='all_twenty_epochs',input_rebuild_each_allocation=True,
    final_export_only=True,timing_factor=1.3,fixed_margin_minutes=60)


def previous(path):
    from . import campaign
    s=load_json(path)
    if (s.get('schema_version')!=3 or s.get('content_hash')!=PREVIOUS_HASH
            or s.get('commit')!=PREVIOUS_COMMIT or s.get('root')!=PREVIOUS_ROOT
            or Path(path)!=Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Time recovery requires the exact failed 2bf345f9 execution')
    campaign.check(s,live=False)
    return s


def evidence(s):
    root=Path(s['root']);folder=root/'train_CONCAT/runs'/FAILED_JOB
    failure=load_json(folder/'failure.json');c.validate(failure,'FAILURE',dict(spec=s['content_hash']))
    if (failure['job_id']!=FAILED_JOB or failure['stage']!='CONCAT'
            or failure['error_type']!='RuntimeError' or failure['message']!=FAILURE_MESSAGE):
        raise ValueError('Time failure witness differs')
    phases={}
    for name in ('parity','miniature','stress','deployment','measurements'):
        path=folder/'startup_diagnostics'/f'{name}.json';value=load_json(path)
        c.validate(value,'STARTUP_PHASE',dict(spec=s['content_hash']))
        if value['stage']!='CONCAT' or value['phase']!=name:raise ValueError('Startup phase differs')
        phases[name]=file_ref(path)
    measured=load_json(phases['measurements']['path']);t=measured['timing']
    expected=startup.estimate('CONCAT',t['train_step'],t['val_step'],t['export_step'],t['elapsed'],20)
    if (measured['content_hash']!=MEASUREMENTS_HASH or any(t.get(k)!=v for k,v in expected.items())
            or expected['conservative_minutes']<=s['minutes']['train_CONCAT']):
        raise ValueError('Bound walltime measurements differ')
    segment_estimate=startup.estimate('CONCAT',t['train_step'],t['val_step'],t['export_step'],t['elapsed'],10)
    if segment_estimate['conservative_minutes']>s['minutes']['train_CONCAT']:
        raise ValueError('Ten-epoch allocation still exceeds measured walltime')
    early=load_json(folder/'early_memory.json');c.validate(early,'EARLY_MEMORY',dict(spec=s['content_hash']))
    if early['stage']!='CONCAT' or early['passed'] is not True:raise ValueError('Early acceptance differs')
    parent=gradient_recovery.validate(s)
    claim=Path(parent['root'])/'concat_gradient_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=parent['content_hash'],recovery=s['content_hash']),
                      root=s['root'],commit=s['commit'])
    if load_json(claim)!=wanted:raise ValueError('Previous gradient recovery claim differs')
    return c.artifact('TIME_RECOVERY',parents=dict(spec=s['content_hash']),
        previous_spec=file_ref(root/'campaign_spec.json'),failure=file_ref(folder/'failure.json'),
        measurements=phases,early_memory=file_ref(folder/'early_memory.json'),previous_claim=file_ref(claim),
        policy=POLICY,segment_estimate=segment_estimate,scientific_recipe_unchanged=True)


def authenticate(path,*,payloads=True):
    from . import campaign
    s=previous(path)
    if capture_source_snapshot(Path(s['project']),require_clean=True)!=s['source']:
        raise ValueError('Previous time-recovery source snapshot differs')
    old,record,gradient=gradient_recovery.authenticate(Path(s['gradient_recovery']['previous_spec']['path']),payloads=payloads)
    if record!=s['execution_recovery'] or gradient!=s['gradient_recovery']:
        raise ValueError('Historical execution evidence changed')
    jobs=campaign.accepted_jobs(s)
    if len(jobs)!=4 or {r['task'] for r in jobs}!={'train_'+t for t in c.STAGES[2:]}:
        raise ValueError('Expected four unambiguous previous submissions')
    states={r['job']:campaign.terminal(r['job']) for r in jobs}
    if (not any(r['job']==FAILED_JOB and r['task']=='train_CONCAT' for r in jobs)
            or states[FAILED_JOB][1].split()[0]!='FAILED' or states[FAILED_JOB][2]!='1:0'):
        raise ValueError('Time failure accounting differs')
    for stage in c.STAGES[2:]:
        folder=Path(s['root'])/('train_'+stage)
        if (folder/'result.json').exists() or list((folder/'epochs').glob('*.json')):
            raise ValueError('Previous execution has committed progress; cannot discard it')
    return old,record,gradient,evidence(s)


def validate(spec):
    r=spec['time_recovery'];c.validate(r,'TIME_RECOVERY',dict(spec=PREVIOUS_HASH))
    s=previous(validate_file_ref(r['previous_spec']))
    if r!=evidence(s) or r['policy']!=POLICY:raise ValueError('Time recovery evidence/policy differs')
    if spec['root']==s['root'] or spec['project']==s['project']:
        raise ValueError('Time recovery requires new source and output roots')
    if spec['execution_recovery']!=s['execution_recovery'] or spec['gradient_recovery']!=s['gradient_recovery']:
        raise ValueError('Time recovery must preserve baseline and numerical repair bindings')
    return s


def claim(spec):
    s=validate(spec);path=Path(s['root'])/'concat_time_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=s['content_hash'],recovery=spec['content_hash']),
                      root=spec['root'],commit=spec['commit'])
    if path.exists():
        if load_json(path)!=wanted:raise FileExistsError('Another time recovery already claimed this execution')
    else:write_immutable_json(path,wanted)
