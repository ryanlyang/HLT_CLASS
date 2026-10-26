"""Retained k5 comparison and optional authenticated, independently owned bank copies."""
import hashlib
import re
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_auxiliary_logits_strong import contracts as previous
from .contracts import artifact, validate, TEACHERS


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_auxiliary_logits_strong import campaign,worker,submission,banks
s=load_json(sys.argv[1]); campaign.check(s)
r=worker.retired_evidence(s); seen={}
worker.validate_task(s,'k5',seen,retired=r)
for task,row in seen.items(): submission.successful(s,task,row['job_id'])
root=Path(s['root'])
print('AUXGRID_REFERENCE='+json.dumps(dict(root=str(root),project_dir=s['project_dir'],
    c50=s['reuse']['spec'],teachers=s['reuse']['teachers'],anchor=s['anchor'],
    predictions=file_ref(root/'k5/fit/validation_predictions.npz'),
    report=file_ref(root/'k5/fit/training_report.json'),
    retired=r is not None,banks=banks.manifests(s,payloads=r is None)),sort_keys=True,allow_nan=False))
'''


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
    print('Authenticating completed K5/E5 and bank retirement in original source...',flush=True)
    ref,e=inspect(path,reuse)
    value=artifact('REFERENCE',parents=dict(reuse=reuse['content_hash'],spec=ref['sha256']),
        spec=ref,root=e['root'],project_dir=e['project_dir'],predictions=e['predictions'],report=e['report'])
    validate_reference(value,reuse)
    return value


def validate_reference(value, reuse):
    validate(value,'REFERENCE',dict(reuse=reuse['content_hash'],spec=value['spec']['sha256']))
    for key in ('spec','predictions','report'):
        validate_file_ref(value[key])
    old=load_json(value['spec']['path']); previous.validate(old,'SPEC')
    root=Path(value['root'])
    if (root!=Path(old['root']) or value['project_dir']!=old['project_dir']
            or value['spec']['path']!=str(root/'campaign_spec.json')
            or old['reuse']['teachers']!=reuse['teachers'] or old['reuse']['spec']!=reuse['spec']
            or old['anchor']!=reuse['anchor']
            or value['predictions']!=file_ref(root/'k5/fit/validation_predictions.npz')
            or value['report']!=file_ref(root/'k5/fit/training_report.json')):
        raise ValueError('Retained K5 reference differs')


def acquire(spec):
    """Return acquisition evidence; copy only after all source payloads were read/hashed."""
    from . import banks
    ref=spec['reference']; validate_file_ref(ref['spec'])
    _,e=inspect(ref['spec']['path'],spec['reuse'])
    if e['predictions']!=ref['predictions'] or e['report']!=ref['report']:
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
