"""Read-only original-source replay of existing frozen-campaign controls."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_frozen_adapter_hydra.contracts import validate as old_validate
from .contracts import artifact, validate, COUNTS

CONTROL_COMMIT = '4b9630fa00f137c4e48b749eb86714719b9d26cf'
CONTROL_TASKS = ('teacher_bias', 'teacher_edge', 'bias_ce_e5', 'edge_ce_e5', 'bias_ce_e0', 'edge_ce_e0')
PROGRAM = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_frozen_adapter_hydra import campaign,worker,training
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
rows={}; pending=[]
for task in ('teacher_bias','teacher_edge','bias_ce_e5','edge_ce_e5','bias_ce_e0','edge_ce_e0'):
    root=Path(s['root'])/task
    if not (root/'receipt.json').exists(): pending.append(task); continue
    receipt=worker.receipt(s,task)
    report,_=training.validate_fit(root/'fit',s,task)
    if report['parents']['prepared'] != s['anchor']['evidence']['prepared']:
        raise ValueError('Control preparation differs')
    if task.endswith('e5'):
        from hlt_classification.luka_frozen_adapter_hydra.inputs import Bundle
        if report['parents']['embedding'] != Bundle(s['bundle_root']).json('targets/u100/manifest.json')['content_hash']:
            raise ValueError('Control embedding differs')
    rows[task]=dict(report=report['content_hash'],receipt=receipt['content_hash'],
        selected_epoch=report['selected_pass'],privileged_oracle=report['privileged_oracle'],
        predictions=file_ref(root/'fit/validation_predictions.npz'))
print('JOINT_CONTROLS='+json.dumps(dict(spec=s['content_hash'],rows=rows,pending=pending),sort_keys=True))
'''


def inspect(path):
    path=Path(path).resolve(strict=True); old=load_json(path); old_validate(old,'SPEC')
    if (old['source_commit'] != CONTROL_COMMIT or old['counts'] != COUNTS
            or path != Path(old['root'])/'campaign_spec.json'):
        raise ValueError('Expected original 4b9630fa frozen-adapter control campaign')
    return old


def register(path, origin, bundle_root, delta_hash):
    old=inspect(path)
    if (any(old[k] != origin[k] for k in ('reference','imported','anchor'))
            or Path(old['bundle_root']).resolve() != Path(bundle_root).resolve()
            or old['delta_hash'] != delta_hash):
        raise ValueError('Control population, initialization, embedding or input lineage differs')
    ref, proof=replay(path,old['project_dir'],PROGRAM,'JOINT_CONTROLS=')
    if proof['spec'] != old['content_hash']: raise ValueError('Control replay differs')
    return artifact('CONTROL_REFERENCE',parents=dict(spec=old['content_hash']),spec=ref,
        source_commit=CONTROL_COMMIT,tasks=list(CONTROL_TASKS),reference_retrained=False)


def check(spec):
    value=spec['control_campaign']; validate(value,'CONTROL_REFERENCE')
    validate_file_ref(value['spec']); old=inspect(value['spec']['path'])
    if (value['parents'] != dict(spec=old['content_hash']) or value['tasks'] != list(CONTROL_TASKS)
            or value['source_commit'] != CONTROL_COMMIT or value['reference_retrained'] is not False
            or any(old[k] != spec[k] for k in ('reference','imported','anchor','counts'))
            or old['delta_hash'] != spec['delta_hash']):
        raise ValueError('Registered control evidence differs')
    return old


def completed(spec):
    old=check(spec)
    ref, proof=replay(spec['control_campaign']['spec']['path'],old['project_dir'],PROGRAM,'JOINT_CONTROLS=')
    if (ref != spec['control_campaign']['spec'] or proof['spec'] != old['content_hash']
            or set(proof['rows']) | set(proof['pending']) != set(CONTROL_TASKS)
            or set(proof['rows']) & set(proof['pending'])):
        raise ValueError('Control replay coverage differs')
    for row in proof['rows'].values(): validate_file_ref(row['predictions'])
    return proof
