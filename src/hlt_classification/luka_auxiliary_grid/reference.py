"""Retained k5 comparison and optional authenticated, independently owned bank copies."""
import hashlib
import re
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_auxiliary_logits_strong import contracts as previous
from .contracts import artifact, validate, TEACHERS


# The adapter imports only the original modules in the isolated subprocess.
# Its own bytes are included in the new source snapshot; old worktrees stay clean.
REPLAY = Path(__file__).with_name('reference_replay.py').read_text(encoding='utf-8')+'\nmain(sys.argv[1])\n'


def inspect(path, reuse):
    path=Path(path).resolve(strict=True); old=load_json(path); previous.validate(old,'SPEC')
    if (path != Path(old['root'])/'campaign_spec.json' or old['reuse']['spec']!=reuse['spec']
            or old['reuse']['teachers']!=reuse['teachers'] or old['anchor']!=reuse['anchor']):
        raise ValueError('Wrong strong reference parent/teachers/population')
    validate_source_snapshot(old['source_snapshot'],repository=old['project_dir'],require_clean=True)
    ref,evidence=replay(path,old['project_dir'],REPLAY,'AUXGRID_REFERENCE=')
    if (evidence['c50']!=reuse['spec'] or evidence['teachers']!=reuse['teachers']
            or evidence['anchor']!=reuse['anchor'] or evidence['root']!=old['root']
            or evidence['project_dir']!=old['project_dir'] or set(evidence['banks'])!=set(TEACHERS)
            or type(evidence['retired']) is not bool):
        raise ValueError('Original strong replay identity differs')
    return ref,evidence


def import_reference(path, reuse):
    print('Authenticating K5/E5 (completed or verified post-fit JSON-order recovery) in original source...',flush=True)
    ref,e=inspect(path,reuse)
    value=artifact('REFERENCE',parents=dict(reuse=reuse['content_hash'],spec=ref['sha256']),
        spec=ref,root=e['root'],project_dir=e['project_dir'],predictions=e['predictions'],report=e['report'],
        checkpoint=e['checkpoint'], validation=load_json(e['report']['path'])['validation'],
        replay=artifact('REFERENCE_REPLAY',parents=dict(spec=ref['sha256']),**e['recovery']))
    validate_reference(value,reuse)
    state=e['recovery']['accounting']['k5']
    print(f"K5 reference: {e['recovery']['mode']}; original job {state['job_id']} "
          f"remains {state['state']}/{state['exit_code']}. Native readback required in targets.",flush=True)
    return value


def validate_reference(value, reuse):
    validate(value,'REFERENCE',dict(reuse=reuse['content_hash'],spec=value['spec']['sha256']))
    for key in ('spec','predictions','report','checkpoint'):
        validate_file_ref(value[key])
    old=load_json(value['spec']['path']); previous.validate(old,'SPEC')
    root=Path(value['root'])
    if (root!=Path(old['root']) or value['project_dir']!=old['project_dir']
            or value['spec']['path']!=str(root/'campaign_spec.json')
            or old['reuse']['teachers']!=reuse['teachers'] or old['reuse']['spec']!=reuse['spec']
            or old['anchor']!=reuse['anchor']
            or value['predictions']!=file_ref(root/'k5/fit/validation_predictions.npz')
            or value['report']!=file_ref(root/'k5/fit/training_report.json')
            or value['checkpoint']!=file_ref(root/'k5/fit/selected.pt')
            or value['validation']!=load_json(value['report']['path'])['validation']):
        raise ValueError('Retained K5 reference differs')
    evidence=value['replay']; validate(evidence,'REFERENCE_REPLAY',dict(spec=value['spec']['sha256']))
    mode=evidence['mode']; states=evidence['accounting']
    if set(states)!={'targets','k5'}:
        raise ValueError('Reference accounting coverage differs')
    for task, state in states.items():
        receipt=load_json(root/task/'receipt.json')
        if (state['job_id']!=receipt['job_id'] or state['name']!='lukaas_'+task
                or state['account']!=old['execution_site']['account']
                or state['user']!=states['k5']['user']):
            raise ValueError('Reference accounting identity differs')
    if (states['targets']['state'],states['targets']['exit_code'])!=('COMPLETED','0:0'):
        raise ValueError('Reference target ancestor is not completed')
    if mode=='completed':
        if evidence['repair'] is not None or (states['k5']['state'],states['k5']['exit_code'])!=('COMPLETED','0:0'):
            raise ValueError('Completed reference evidence differs')
    elif mode=='recovered_postfit_json_order':
        from .reference_replay import FAILED_COMMIT, FAILED_JOB, reordered_epochs
        repair=evidence['repair']; report=load_json(value['report']['path'])
        if (old['source_commit']!=FAILED_COMMIT or states['k5']['job_id']!=FAILED_JOB
                or (states['k5']['state'],states['k5']['exit_code'])!=('FAILED','1:0')
                or not repair['reordered_epochs'] or repair['reordered_epochs']!=reordered_epochs(report,TEACHERS)
                or repair['rule']!='exact_original_teacher_order_no_value_or_tolerance_change'
                or repair['failure_log']!=file_ref(root/('k5-'+FAILED_JOB+'.out'))):
            raise ValueError('Registered failed-reference recovery differs')
        validate_file_ref(repair['failure_log'])
    else:
        raise ValueError('Unregistered reference recovery mode')
    expected={'k5/receipt.json','k5/costs.json','k5/fit/receipt.json','k5/fit/selected.pt',
        'k5/fit/training_report.json','k5/fit/validation_predictions.npz',
        'targets/receipt.json','targets/bundle.json','targets/startup.json'}
    if set(evidence['files'])!=expected:
        raise ValueError('Reference replay artifact coverage differs')
    for name, ref in evidence['files'].items():
        validate_file_ref(ref)
        if ref!=file_ref(root/name):
            raise ValueError('Reference replay artifact path differs')


def acquire(spec):
    """Return acquisition evidence; copy only after all source payloads were read/hashed."""
    from . import banks
    ref=spec['reference']; validate_file_ref(ref['spec'])
    _,e=inspect(ref['spec']['path'],spec['reuse'])
    evidence=artifact('REFERENCE_REPLAY',parents=dict(spec=ref['spec']['sha256']),**e['recovery'])
    if (e['predictions']!=ref['predictions'] or e['report']!=ref['report']
            or e['checkpoint']!=ref['checkpoint'] or evidence!=ref['replay']):
        raise ValueError('Reference changed before acquisition')
    payloads={}
    if not e['retired']:
        for t in TEACHERS:
            folder=Path(ref['root'])/'targets/banks'/t
            m=banks.donor.manifest(folder,banks.record(spec,t),spec['anchor']['evidence']['prepared'],spec['counts']['train'])
            if m!=e['banks'][t]:
                raise ValueError('Target manifest changed during copy')
            path=folder/'targets.npz'
            if path.is_symlink() or folder!=folder.resolve(strict=True):
                raise PermissionError('Noncanonical target source')
            raw=path.read_bytes()
            if len(raw)!=m['payload']['bytes'] or hashlib.sha256(raw).hexdigest()!=m['payload']['sha256']:
                raise ValueError('Target payload changed during copy')
            payloads[t]=raw
        # All bytes now owned in memory, independent of an old cleanup.
        for t in TEACHERS:
            folder=banks.directory(spec,t); folder.mkdir(parents=True,exist_ok=False)
            atomic_publish_bytes(folder/'targets.npz',payloads[t])
            write_immutable_json(folder/'manifest.json',e['banks'][t])
    return artifact('ACQUISITION',parents=dict(spec=spec['content_hash'],reference=ref['content_hash']),
        mode='export_after_authenticated_retirement' if e['retired'] else 'verified_private_copy',
        source_banks={t:e['banks'][t]['content_hash'] for t in TEACHERS},
        source_retired=e['retired'],source_spec=ref['spec'])


def check_acquisition(value, spec):
    validate(value,'ACQUISITION',dict(spec=spec['content_hash'],reference=spec['reference']['content_hash']))
    expected='export_after_authenticated_retirement' if value['source_retired'] else 'verified_private_copy'
    if (type(value['source_retired']) is not bool or value['mode']!=expected
            or value['source_spec']!=spec['reference']['spec'] or set(value['source_banks'])!=set(TEACHERS)
            or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in value['source_banks'].values())):
        raise ValueError('Target acquisition evidence differs')
    if value['mode']=='verified_private_copy':
        from . import banks
        if value['source_banks']!={t:m['content_hash'] for t,m in banks.manifests(spec,payloads=False).items()}:
            raise ValueError('Copied target identity differs')


def readback(spec, val, job, device):
    """Native saved-K5 inference only; no training, checkpoint selection or writes to old roots."""
    import numpy as np
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
    from hlt_classification.luka_trial_hybrid.worker import native, release
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    ref=spec['reference']
    for name in ('checkpoint','predictions'):
        validate_file_ref(ref[name])
    arrays=load_npz_arrays(ref['predictions']['path']); validate_predictions(arrays,len(val))
    if (val.role!='validation' or val.coordinate_name!='D000'
            or val.foundation_sha256!=spec['anchor']['evidence']['prepared']
            or len(val)!=spec['counts']['validation']
            or any(not np.array_equal(arrays[k],getattr(val,k)) for k in ('identities','labels'))):
        raise ValueError('Reference readback validation population differs')
    model=native(ref['checkpoint'],device)
    try:
        actual=predict(model,val,device=device,batch_size=128)
    finally:
        del model; release()
    saved=arrays['probabilities']
    if actual.shape!=saved.shape or actual.dtype!=saved.dtype or actual.tobytes()!=saved.tobytes():
        raise ValueError('Reference native checkpoint predictions differ from saved K5')
    metrics=evaluate_probabilities(val.labels,actual)
    if metrics!=ref['validation']:
        raise ValueError('Reference native checkpoint metrics differ')
    for name in ('checkpoint','predictions'):
        validate_file_ref(ref[name])
    result=artifact('REFERENCE_READBACK',parents=dict(spec=spec['content_hash'],reference=ref['content_hash']),
        job_id=job,original_accounting=ref['replay']['accounting'],reference_mode=ref['replay']['mode'],
        checkpoint=ref['checkpoint'],predictions=ref['predictions'],rows=len(val),validation=metrics,
        predictions_byte_identical=True,training_repeated=False,checkpoint_reselected=False)
    check_readback(result,spec,job)
    return result


def check_readback(value, spec, job):
    ref=spec['reference']
    validate(value,'REFERENCE_READBACK',dict(spec=spec['content_hash'],reference=ref['content_hash']))
    if (value['job_id']!=job or value['original_accounting']!=ref['replay']['accounting']
            or value['reference_mode']!=ref['replay']['mode'] or value['checkpoint']!=ref['checkpoint']
            or value['predictions']!=ref['predictions'] or value['rows']!=spec['counts']['validation']
            or value['validation']!=ref['validation'] or value['predictions_byte_identical'] is not True
            or value['training_repeated'] is not False or value['checkpoint_reselected'] is not False):
        raise ValueError('Saved reference readback evidence differs')
