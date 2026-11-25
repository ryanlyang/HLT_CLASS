"""Explicit v2 execution fork; retain the completed v1 baselines unchanged."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from . import contracts as c
from .pair_storage import POLICY

PARENT_COMMIT = 'd534fcfea4f11a4bc845127d03c31b779e391a8c'
PARENT_HASH = '38c65fcb5d6850d4a74f79470337b314d73cb5c1d31c1dd3f7d6c035fb448f33'
PARENT_ROOT = '/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_wide_b256_oscar_d534fcfe_r1'
FAILED_JOB = '9544072'
BASELINES = ('OFFLINE','M0HLT')
BASELINE_JOBS = dict(OFFLINE='9544070',M0HLT='9544071')


def parent(path):
    from . import campaign
    s=load_json(path)
    if (s.get('schema_version')!=1 or s.get('content_hash')!=PARENT_HASH
            or s.get('commit')!=PARENT_COMMIT or s.get('root')!=PARENT_ROOT
            or Path(path)!=Path(s['root'])/'campaign_spec.json'):
        raise ValueError('Recovery requires the exact failed d534fcfe campaign')
    campaign.check(s,live=False)
    return s


def authenticate(path, *, payloads=True):
    """Run at registration and submission; never infer terminal state from queue absence."""
    from . import campaign, artifacts
    s=parent(path);root=Path(s['root'])
    if capture_source_snapshot(Path(s['project']),require_clean=True)!=s['source']:
        raise ValueError('Historical source snapshot differs')
    jobs=campaign.accepted_jobs(s)
    if len(jobs)!=6 or {r['task'] for r in jobs}!=set(c.tasks()):
        raise ValueError('Expected the six original, unambiguous submission records')
    states={r['job']:campaign.terminal(r['job']) for r in jobs}
    if not any(r['job']==FAILED_JOB and r['task']=='train_CONCAT' for r in jobs):
        raise ValueError('CONCAT failure is not bound to the campaign')
    if states[FAILED_JOB][1].split()[0]!='FAILED' or states[FAILED_JOB][2]!='1:0':
        raise ValueError('Unexpected CONCAT accounting')
    for stage in c.STAGES[2:]:
        folder=root/('train_'+stage)
        if (folder/'result.json').exists() or list((folder/'epochs').glob('*.json')):
            raise ValueError('Unfinished stage has committed progress; this fresh-tail repair cannot discard it')
    failure_path=root/'train_CONCAT/runs'/FAILED_JOB/'failure.json'
    failure=load_json(failure_path);c.validate(failure,'FAILURE',dict(spec=s['content_hash']))
    if (failure['job_id']!=FAILED_JOB or failure['stage']!='CONCAT'
            or failure['error_type']!='OutOfMemoryError' or 'CUDA out of memory' not in failure['message']):
        raise ValueError('Recovery failure witness differs')
    imported={}
    for stage in BASELINES:
        path=root/('train_'+stage)/'result.json'
        receipt=load_json(path);c.validate(receipt,'RESULT')
        job=BASELINE_JOBS[stage]
        if (receipt['job_id']!=job or receipt['stage']!=stage or receipt['parents']['spec']!=s['content_hash']
                or not any(r['job']==job and r['task']=='train_'+stage for r in jobs)
                or states[job][1].split()[0]!='COMPLETED' or states[job][2]!='0:0'):
            raise ValueError('Completed baseline job binding differs')
        if payloads:
            result=artifacts.completed(s,stage)
            # Bank identities/labels/normalization are revalidated against resident TRAIN in the worker.
            bank_path=validate_file_ref(result['receipt']['files']['bank'])
            bank=load_json(bank_path);c.validate(bank,'BANK')
            for ref in bank['payloads'].values():validate_file_ref(ref,root=bank_path.parent)
        imported[stage]=file_ref(path)
    return s,c.artifact('EXECUTION_RECOVERY',parents=dict(spec=s['content_hash']),
        parent_spec=file_ref(root/'campaign_spec.json'),failure=file_ref(failure_path),
        imported=imported,pair_storage=POLICY,early_memory_check=True,
        fresh_stages=list(c.STAGES[2:]),scientific_recipe_unchanged=True)


def validate(spec):
    r=spec['execution_recovery'];c.validate(r,'EXECUTION_RECOVERY',dict(spec=PARENT_HASH))
    s=parent(validate_file_ref(r['parent_spec']))
    if (r.get('pair_storage')!=POLICY or r.get('early_memory_check') is not True
            or r.get('fresh_stages')!=list(c.STAGES[2:]) or set(r.get('imported',{}))!=set(BASELINES)
            or r.get('scientific_recipe_unchanged') is not True
            or spec['root']==s['root'] or spec['project']==s['project']):
        raise ValueError('Recovery execution policy differs')
    for key in ('recipe','population','counts','nodes','tasks','dataset','environment_dir','environment',
                'backend','foundation_producer','transfer_manifest_sha256','gpu','memory_mb','cpus',
                'preparation_workers','candidate_workers','minutes','execution_site'):
        if spec[key]!=s[key]:raise ValueError('Recovery changed '+key)
    for stage,ref in r['imported'].items():
        if Path(ref['path'])!=Path(s['root'])/('train_'+stage)/'result.json':
            raise ValueError('Imported baseline locator differs')
        receipt=load_json(validate_file_ref(ref));c.validate(receipt,'RESULT')
        if (receipt['parents']['spec']!=PARENT_HASH or receipt['stage']!=stage
                or receipt['job_id']!=BASELINE_JOBS[stage]):raise ValueError('Imported baseline binding differs')
    failure=load_json(validate_file_ref(r['failure']));c.validate(failure,'FAILURE',dict(spec=PARENT_HASH))
    if Path(r['failure']['path'])!=Path(s['root'])/'train_CONCAT/runs'/FAILED_JOB/'failure.json':
        raise ValueError('Failure locator differs')
    return s


def origin(spec,stage):
    if spec.get('schema_version') in (2,3) and stage in BASELINES:
        return validate(spec)
    return spec


def enabled(spec,stage):
    return spec.get('schema_version') in (2,3) and stage=='CONCAT'


def model_options(spec,stage):
    if not enabled(spec,stage):return {}
    return dict(pair_storage=True,**({'pointwise_dgrad':True} if spec['schema_version']==3 else {}))


def claim(spec):
    """Called under the parent's submission lock; a second recovery cannot also launch."""
    s=validate(spec)
    path=Path(s['root'])/'concat_memory_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=s['content_hash'],recovery=spec['content_hash']),
                      root=spec['root'],commit=spec['commit'])
    if path.exists():
        if load_json(path)!=wanted:raise FileExistsError('Another recovery already claimed this campaign')
    else:write_immutable_json(path,wanted)
