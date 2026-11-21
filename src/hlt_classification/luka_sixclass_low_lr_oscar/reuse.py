"""Authenticate frozen historical teachers in their original installed source."""
import json
import os
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from . import contracts as c


def parent(path):
    from hlt_classification.luka_sixclass_balanced_oscar import campaign
    path = Path(path)
    value = load_json(path)
    campaign.check(value, live=False)
    if (str(path) != c.PARENT_SPEC or value['content_hash'] != c.PARENT_SPEC_HASH
            or value['commit'] != c.PARENT_COMMIT or path != Path(value['root'])/'campaign_spec.json'):
        raise ValueError('Wrong completed six-class parent campaign')
    return value


REPLAY = r'''
import json, sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_sixclass_balanced_oscar import campaign, artifacts, contracts, banks
s=load_json(sys.argv[1]); campaign.check(s)
memo={}; result={}
for stage in ('CONCAT','UE100','DE050'):
    r=artifacts.completed(s,stage,memo=memo)
    bank_path=validate_file_ref(r['receipt']['files']['bank'])
    bank=load_json(bank_path); contracts.validate(bank,'BANK',banks.parents(s,stage,r['refs']))
    if bank['stage']!=stage or bank['role']!='train' or bank['rows']!=contracts.DRAWS:
        raise ValueError('Historical bank scope differs')
    for ref in bank['payloads'].values():
        validate_file_ref(ref,root=Path(bank_path).parent)
    result[stage]=dict(result=file_ref(Path(s['root'])/('train_'+stage)/'result.json'),
                      receipt_sha256=r['receipt']['content_hash'],bank_sha256=bank['content_hash'])
print('LOW_LR_REUSE='+json.dumps(result,sort_keys=True,allow_nan=False))
'''


def authenticate(path):
    old = parent(path)
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH')}
    env.update(PYTHONPATH=str(Path(old['project'])/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    # Import and source checks must run in the original checkout, never relabel
    # historical artifacts with the new experiment's source or recipe.
    proc = subprocess.run([str(Path(old['environment_dir'])/'bin/python'), '-B', '-s', '-c', REPLAY, str(path)],
                          cwd=old['project'], env=env, check=True, capture_output=True, text=True)
    lines = [line.removeprefix('LOW_LR_REUSE=') for line in proc.stdout.splitlines() if line.startswith('LOW_LR_REUSE=')]
    if len(lines) != 1:
        raise ValueError('Historical authentication did not return one receipt')
    return c.artifact('REUSE', parents=dict(parent_spec=old['content_hash']),
                      spec=file_ref(path), source=old['source'], teachers=json.loads(lines[0]))


def check(value, *, payloads=False):
    c.validate(value, 'REUSE', dict(parent_spec=c.PARENT_SPEC_HASH))
    if set(value['teachers']) != set(c.EXTERNAL):
        raise ValueError('Historical teacher inventory differs')
    old = parent(value['spec']['path'])
    if value['spec'] != file_ref(c.PARENT_SPEC) or value['source'] != old['source']:
        raise ValueError('Historical spec/source binding differs')
    for stage, record in value['teachers'].items():
        expected = Path(old['root'])/('train_'+stage)/'result.json'
        if Path(record['result']['path']) != expected:
            raise ValueError('Historical teacher result path differs')
        from hlt_classification.luka_sixclass_balanced_oscar import contracts as original
        r = load_json(validate_file_ref(record['result'])); original.validate(r,'RESULT')
        if (r['content_hash'] != record['receipt_sha256'] or r['stage'] != stage
                or r['parents']['spec'] != old['content_hash']):
            raise ValueError('Historical teacher receipt differs')
    if payloads and authenticate(c.PARENT_SPEC) != value:
        raise ValueError('Frozen teacher/source replay differs')
    return old


def completed(spec, stage):
    if stage not in c.EXTERNAL:
        raise PermissionError('Only the three frozen external teachers may be reused')
    old = check(spec['reuse'])
    from hlt_classification.luka_sixclass_balanced_oscar import artifacts
    result = artifacts.completed(old, stage)
    if result['receipt']['content_hash'] != spec['reuse']['teachers'][stage]['receipt_sha256']:
        raise ValueError('Teacher changed after registration')
    bank = load_json(result['receipt']['files']['bank']['path'])
    if bank['content_hash'] != spec['reuse']['teachers'][stage]['bank_sha256']:
        raise ValueError('Historical bank binding differs')
    return result
