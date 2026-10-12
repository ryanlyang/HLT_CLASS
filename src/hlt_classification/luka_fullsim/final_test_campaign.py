"""Separate immutable final-test locks; ordinary Luka readers stay sealed."""
from pathlib import Path
import json
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, write_immutable_json, validate_content_hash, with_content_hash,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.jetclass2_delphes.reporting import metrics_contract
from . import stage2 as s, science_campaign as science, preflight_reuse

SCIENCE_COMMIT = '01a738328b267da1ecc13bbf4263c2e4a2d9e4e2'
SPLIT_HASH = '79ac6e577b92a5d2ed00d848c996812b388cab57f9d97d6b519f2bc10060c3b2'
AUTHORIZATION = 'AUTHORIZE LUKA ALL 12 FROZEN MODELS FINAL TEST EXACT PLAN'
ADDITIONS = frozenset('src/hlt_classification/luka_fullsim/final_test_'+n+'.py'
                     for n in ('campaign', 'data', 'worker', 'submission'))
REQUIRED = (*ADDITIONS, 'scripts/luka_fullsim_final_test.py',
    'scripts/queue_luka_fullsim_final_test.sh', 'sbatch/run_luka_fullsim_final_test.sh',
    'docs/plans/LUKA_FULLSIM_FINAL_TEST_PLAN.md', 'docs/contracts/LUKA_FULLSIM_FINAL_TEST.md',
    'docs/LUKA_FULLSIM_FINAL_TEST_RUNBOOK.md', 'tests/test_luka_fullsim_final_test.py')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract='LUKA_FULLSIM_FINAL_TEST_'+kind+'/v1',
        schema_version=1, parents=dict(parents or {}), **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract='LUKA_FULLSIM_FINAL_TEST_'+kind+'/v1',
                                  expected_schema_version=1)
    if parents is not None and value['parents'] != parents:
        raise ValueError('Final-test parent bindings differ')
    return digest


def validate_smoke(value, execution):
    validate(value, 'TRAIN_SMOKE', dict(execution=execution['content_hash']))
    if (value['final_test_accessed'] is not False or value['native_mask_parity'] is not True
            or value['installed_parity'].get('passed') is not True
            or len(value['models']) != len(execution['parent']['models'])):
        raise ValueError('Pre-unsealing TRAIN miniature did not pass')
    for row, model in zip(value['models'], execution['parent']['models']):
        if (row['node_id'] != model['node']['node_id'] or row['checkpoint_sha256'] != model['checkpoint']['sha256']
                or row['exact_probability_replay'] is not True or row['rows'] != 64):
            raise ValueError('TRAIN miniature checkpoint/coverage differs')


def protocol():
    return dict(role='final_test', counts=149553, files=337, split_sha256=SPLIT_HASH,
        nodes=science.nodes(), temperature=1., batch_size=256, precision='cuda_bf16_forward_fp32_softmax',
        metrics=metrics_contract(), recovery_reference=dict(zero='M0HLT', hundred='OFFLINE'),
        checkpoint_selection='existing_validation_selected_only', retraining=False,
        primary_contrast=['FINAL_BRIDGE_D000', 'DIRECT_D000'],
        secondary_contrast=['FINAL_BRIDGE_D000', 'M0HLT'],
        bootstrap=dict(unit='source_file', draws=200, seed=20261012, interval=[.025, .975]),
        test_selected_winner=False, final_test_access_authorized=True)


def source(project, commit):
    value = s.source(project, commit)
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit final-test implementation, contracts and tests first')
    if Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/final_test_campaign.py':
        raise ValueError('Final-test module escapes execution worktree')
    return value


# Executed with ORIGINAL science imports, not by weakening its source validator.
PARENT_READER = r'''
import json, sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullsim import science_campaign as c, science_worker as w
p = Path(sys.argv[1]).resolve(strict=True)
spec = load_json(p)
if p != Path(spec['root']).resolve()/'campaign_spec.json':
    raise ValueError('Noncanonical science spec')
c.validate_spec(spec)
seen = {}
done = w.completed(spec, 'complete', seen)
if done is None:
    raise ValueError('Science campaign must be fully committed before test locks')
models = []
for node in c.nodes():
    receipt = seen['train_'+node['node_id']]
    report = load_json(Path(spec['root'])/receipt['result']['training_report'])
    models.append(dict(node=node, receipt=file_ref(Path(spec['root'])/('train_'+node['node_id'])/'receipt.json'),
        checkpoint=file_ref(Path(spec['root'])/receipt['result']['checkpoint']),
        report=file_ref(Path(spec['root'])/receipt['result']['training_report']),
        report_sha256=report['content_hash'], selected_pass=report['selected_pass'], validation=report['validation']))
print(json.dumps(dict(spec=file_ref(p), source_snapshot=spec['source_snapshot'], project_dir=spec['project_dir'],
    root=spec['root'], prepared_root=spec['admission']['prepared_root'],
    preflight=spec['admission']['preflight'], completion=file_ref(Path(spec['root'])/'complete/receipt.json'),
    models=models), sort_keys=True, allow_nan=False))
'''


def parent_snapshot(path):
    path = Path(path).resolve(strict=True)
    preliminary = load_json(path)
    if preliminary['source_commit'] != SCIENCE_COMMIT:
        raise ValueError('Wrong completed science campaign source')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(preliminary['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating completed parent in its original pinned worktree...', flush=True)
    try:
        result = subprocess.run([sys.executable, '-s', '-c', PARENT_READER, str(path)], env=env,
            cwd=preliminary['project_dir'], text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Original pinned science authentication failed:\n'+(exc.stderr or '').strip()) from exc
    return json.loads(result.stdout)


def reuse(parent, active, project):
    old, new = Path(parent['project_dir']), Path(project)
    validate_source_snapshot(parent['source_snapshot'], repository=old, require_clean=True)
    validate_source_snapshot(active, repository=new, require_clean=True)
    before, after = preflight_reuse._files(old), preflight_reuse._files(new)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Final-test source additions/deletions differ')
    for name in sorted(before):
        if sha256_file(old/name) != sha256_file(new/name):
            raise ValueError('Frozen scientific source changed: '+name)
    return artifact('SOURCE_REUSE', parents=dict(science=parent['source_snapshot']['content_hash'],
        execution=active['content_hash']), added_modules=sorted(ADDITIONS), old_src_files=len(before))


def population(parent):
    p, f, inventory, splits = s.load_prepared(parent['prepared_root'])
    if (splits['content_hash'] != SPLIT_HASH or f['counts'] != dict(train=100000, validation=50000, final_test=149553)
            or sum(g['role'] == 'final_test' for g in splits['groups']) != 337):
        raise ValueError('Remote final-test population differs from frozen split')
    return p, f, inventory, splits


def finalist(parent, p, f, splits):
    if [r['node'] for r in parent['models']] != science.nodes():
        raise ValueError('All twelve registered models required, in frozen order')
    for model in parent['models']:
        for key in ('receipt', 'checkpoint', 'report'):
            validate_file_ref(model[key])
    return artifact('FINALIST_LOCK', parents=dict(science=load_json(parent['spec']['path'])['content_hash'],
        completion=load_json(parent['completion']['path'])['content_hash'], prepared=p['content_hash'],
        foundation=f['content_hash'], splits=splits['content_hash']),
        models=parent['models'], protocol=protocol(), final_test_accessed=False)


def execution_lock(root, project, active, parent, lock, reuse_record):
    return artifact('EXECUTION_LOCK', parents=dict(finalists=lock['content_hash'], source=active['content_hash'],
        reuse=reuse_record['content_hash']), root=str(Path(root).resolve()), project_dir=str(Path(project).resolve()),
        source_snapshot=active, parent=parent, protocol=protocol(), reuse=reuse_record,
        resources=dict(site=science.registration()['execution_site'], cpus=6, memory_mb=256000,
                       gpu_minutes=1386, report_minutes=240, report_memory_mb=32768),
        claim_dir=str(Path(parent['root'])/'final_test_access'), final_test_accessed=False,
        holdout_consumption='all_model_frozen_comparison_no_test_selection')


def create(parent_spec, root, project, commit):
    active = source(project, commit)
    parent = parent_snapshot(parent_spec)
    p, f, _, splits = population(parent)
    record = reuse(parent, active, project)
    if (Path(parent['root'])/'final_test_access').exists():
        raise PermissionError('This science campaign already has a final-test execution claim')
    root = s.fresh(root, project, parent['root'], parent['project_dir'], parent['prepared_root'],
                   f['input_container'], p['foundation_root'], Path(parent['preflight']['path']).parent)
    lock = finalist(parent, p, f, splits)
    execution = execution_lock(root, project, active, parent, lock, record)
    if Path(execution['claim_dir']).exists():
        raise PermissionError('This science campaign already has a final-test execution claim')
    write_immutable_json(root/'finalist_lock.json', lock)
    write_immutable_json(root/'execution_lock.json', execution)
    return execution


def authenticate(path, *, full=True):
    path = Path(path).resolve(strict=True)
    execution = load_json(path)
    validate(execution, 'EXECUTION_LOCK')
    root = Path(execution['root'])
    if path != root.resolve()/'execution_lock.json' or execution['protocol'] != protocol():
        raise ValueError('Canonical final-test execution/protocol required')
    lock = load_json(root/'finalist_lock.json')
    validate(lock, 'FINALIST_LOCK')
    if (lock['content_hash'] != execution['parents']['finalists'] or lock['protocol'] != protocol()
            or lock['models'] != execution['parent']['models'] or lock['final_test_accessed'] is not False
            or execution['final_test_accessed'] is not False):
        raise ValueError('Finalist/execution locks differ')
    if not full:
        return execution, lock
    active = source(execution['project_dir'], execution['source_snapshot']['git_commit'])
    parent = parent_snapshot(execution['parent']['spec']['path'])
    p, f, inventory, splits = population(parent)
    expected = execution_lock(root, execution['project_dir'], active, parent,
        finalist(parent, p, f, splits), reuse(parent, active, execution['project_dir']))
    if expected != execution:
        raise ValueError('Final-test frozen parents/source/protocol drifted')
    return execution, lock, p, f, inventory, splits
