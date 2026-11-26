"""One 72-hour CONCAT allocation after the authenticated 48-hour estimate stop."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from . import contracts as c, time_recovery, startup

POLICY=dict(stage='CONCAT',epoch_window=[1,20],schedule_epochs=20,
    allocation_minutes=c.LONG_CONCAT_MINUTES,gpus=1,cpus=6,
    input_preparation='once_per_attempt_in_RAM',persistent_particle_cache=False,
    selected_checkpoint='all_twenty_epochs',timing_factor=1.3,fixed_margin_minutes=60)


def evidence(s):
    prior=time_recovery.evidence(s)
    t=load_json(prior['measurements']['measurements']['path'])['timing']
    estimate=startup.estimate('CONCAT',t['train_step'],t['val_step'],t['export_step'],t['elapsed'],20)
    if estimate['conservative_minutes']>c.LONG_CONCAT_MINUTES:
        raise ValueError('Full CONCAT estimate exceeds the single allocation')
    fields={k:prior[k] for k in ('previous_spec','failure','measurements','early_memory','previous_claim')}
    return c.artifact('LONG_RECOVERY',parents=dict(spec=s['content_hash']),**fields,
        policy=POLICY,full_estimate=estimate,scientific_recipe_unchanged=True)


def authenticate(path,*,payloads=True):
    old,record,gradient,_=time_recovery.authenticate(path,payloads=payloads)
    return old,record,gradient,evidence(time_recovery.previous(path))


def validate(spec):
    r=spec['long_recovery'];c.validate(r,'LONG_RECOVERY',dict(spec=time_recovery.PREVIOUS_HASH))
    s=time_recovery.previous(validate_file_ref(r['previous_spec']))
    if r!=evidence(s) or r['policy']!=POLICY:raise ValueError('Long recovery evidence/policy differs')
    if spec['root']==s['root'] or spec['project']==s['project']:
        raise ValueError('Long recovery requires new source and output roots')
    if spec['execution_recovery']!=s['execution_recovery'] or spec['gradient_recovery']!=s['gradient_recovery']:
        raise ValueError('Long recovery must preserve baseline and numerical repair bindings')
    return s


def claim(spec):
    # Share the v3-parent claim with v4: the two alternatives cannot both run.
    s=validate(spec);path=Path(s['root'])/'concat_time_recovery_claim_v1.json'
    wanted=c.artifact('RECOVERY_CLAIM',parents=dict(spec=s['content_hash'],recovery=spec['content_hash']),
                     root=spec['root'],commit=spec['commit'])
    if path.exists():
        if load_json(path)!=wanted:raise FileExistsError('Another time recovery already claimed this execution')
    else:write_immutable_json(path,wanted)
