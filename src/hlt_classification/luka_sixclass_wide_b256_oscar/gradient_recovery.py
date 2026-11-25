"""A new execution after the zero-epoch pair-convolution backward failure."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from . import contracts as c, recovery, pair_convolution

PREVIOUS_HASH = '8f9ed0893cbe6a0c7f99beae9bab76d8870a75b3460ffc6d1b2894b6550b9d6f'
PREVIOUS_COMMIT = '44aa1522d4720fd8045e55d16f977689bd7df0aa'
PREVIOUS_ROOT = '/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_wide_memory_44aa1522_r1'
FAILED_JOB = '9615391'
DIAGNOSTIC_JOB = '9618095'
DIAGNOSTIC_LOG = '/oscar/home/rlyang/hlt_classification/reports/concat-gradient-9618095.out'


def previous(path):
    from . import campaign
    s=load_json(path)
    if (s.get('schema_version')!=2 or s.get('content_hash')!=PREVIOUS_HASH
            or s.get('commit')!=PREVIOUS_COMMIT or s.get('root')!=PREVIOUS_ROOT
            or Path(path)!=Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Kernel repair requires the exact failed 44aa1522 execution')
    campaign.check(s,live=False)
    return s


def evidence(s):
    root=Path(s['root'])
    failure_path=root/'train_CONCAT/runs'/FAILED_JOB/'failure.json'
    failure=load_json(failure_path);c.validate(failure,'FAILURE',dict(spec=s['content_hash']))
    if (failure['job_id']!=FAILED_JOB or failure['stage']!='CONCAT'
            or failure['error_type']!='ValueError' or failure['message']!='Nonfinite/disconnected native gradient'):
        raise ValueError('Gradient failure witness differs')
    diagnostic=Path(DIAGNOSTIC_LOG);text=diagnostic.read_text()
    if any(token not in text for token in ('Job ID : '+DIAGNOSTIC_JOB,
            str(Path(s['project'])/'src/hlt_classification/luka_sixclass_wide_b256_oscar/early_memory.py'),
            "Function 'ConvolutionBackward0' returned nan values in its 0th output.")):
        raise ValueError('Convolution backward diagnostic witness differs')
    original=recovery.validate(s)
    claim_path=Path(original['root'])/'concat_memory_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=original['content_hash'],recovery=s['content_hash']),
                     root=s['root'],commit=s['commit'])
    if load_json(claim_path)!=wanted:raise ValueError('Previous recovery claim differs')
    return c.artifact('GRADIENT_RECOVERY',parents=dict(spec=s['content_hash']),
        previous_spec=file_ref(root/'campaign_spec.json'),failure=file_ref(failure_path),
        diagnostic=file_ref(diagnostic),previous_claim=file_ref(claim_path),
        pointwise_policy=pair_convolution.POLICY,scientific_recipe_unchanged=True)


def authenticate(path, *, payloads=True):
    from . import campaign
    s=previous(path)
    if capture_source_snapshot(Path(s['project']),require_clean=True)!=s['source']:
        raise ValueError('Previous recovery source snapshot differs')
    original,record=recovery.authenticate(Path(s['execution_recovery']['parent_spec']['path']),payloads=payloads)
    if record!=s['execution_recovery']:raise ValueError('Original baseline evidence differs')
    jobs=campaign.accepted_jobs(s)
    if len(jobs)!=4 or {r['task'] for r in jobs}!={'train_'+t for t in c.STAGES[2:]}:
        raise ValueError('Expected four unambiguous prior recovery submissions')
    states={r['job']:campaign.terminal(r['job']) for r in jobs}
    if not any(r['job']==FAILED_JOB and r['task']=='train_CONCAT' for r in jobs):
        raise ValueError('Gradient failure job binding differs')
    for job in (FAILED_JOB,DIAGNOSTIC_JOB):
        row=states[job] if job in states else campaign.terminal(job)
        if row[1].split()[0]!='FAILED' or row[2]!='1:0':raise ValueError('Gradient/diagnostic job accounting differs')
    for stage in c.STAGES[2:]:
        folder=Path(s['root'])/('train_'+stage)
        if (folder/'result.json').exists() or list((folder/'epochs').glob('*.json')):
            raise ValueError('Prior recovery has committed progress; cannot discard it')
    return original,record,evidence(s)


def validate(spec):
    r=spec['gradient_recovery'];c.validate(r,'GRADIENT_RECOVERY',dict(spec=PREVIOUS_HASH))
    s=previous(validate_file_ref(r['previous_spec']))
    if r!=evidence(s) or r['pointwise_policy']!=pair_convolution.POLICY:
        raise ValueError('Pointwise gradient recovery evidence/policy differs')
    if spec['root']==s['root'] or spec['project']==s['project']:
        raise ValueError('Kernel repair requires new source and output roots')
    if spec['execution_recovery']!=s['execution_recovery']:
        raise ValueError('Kernel repair must keep the exact completed baseline bindings')
    return s


def claim(spec):
    s=validate(spec)
    path=Path(s['root'])/'concat_gradient_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=s['content_hash'],recovery=spec['content_hash']),
                     root=spec['root'],commit=spec['commit'])
    if path.exists():
        if load_json(path)!=wanted:raise FileExistsError('Another kernel recovery already claimed this execution')
    else:write_immutable_json(path,wanted)
