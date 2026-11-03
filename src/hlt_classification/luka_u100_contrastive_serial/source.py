"""Read-only original-source replay of successful setup and completed fits."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.provenance import validate_source_snapshot
from .contracts import artifact, validate

BASE = 'c08f6df603d87a452a4814a9e954421d8731a4a7'
PARENT_COMMIT = '431341db5eced8eaa3e52b37ba047f8cd4beed50'
MODULES = ('__init__', 'contracts', 'source', 'campaign', 'training', 'submission', 'worker')
RUNTIME = {'src/hlt_classification/luka_u100_contrastive_serial/'+n+'.py' for n in MODULES} | {
    'scripts/luka_u100_contrastive_serial.py', 'scripts/queue_luka_u100_contrastive_serial.sh',
    'sbatch/run_luka_u100_contrastive_serial.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_u100_contrastive_serial.py',
    'docs/plans/LUKA_U100_CONTRASTIVE_SERIAL_PLAN.md', 'docs/contracts/LUKA_U100_CONTRASTIVE_SERIAL.md',
    'docs/LUKA_U100_CONTRASTIVE_SERIAL_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    rows = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if not REQUIRED <= set(git('ls-files')) or any(r[:2] not in ('A\t', 'M\t') or r[2:] not in RUNTIME for r in rows):
        raise ValueError('Commit complete additive serial campaign; preserve existing runtime')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_contrastive_serial/source.py':
        raise ValueError('Runtime escapes pinned source')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_contrastive import campaign,worker,training,resources,submission
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False); root=Path(s['root'])
worker.completed_setup(s)
jobs=submission.accepted_jobs(s,'science'); setup_jobs=submission.accepted_jobs(s,'setup')
result=worker.results(s)
lineage_files=[file_ref(p) for name in ('submission_setup','submission_science')
               for p in sorted((root/name).rglob('*.json'))]
completed={}
files=[file_ref(root/'setup/receipt.json')]
for output in load_json(root/'setup/receipt.json')['outputs']:
    files.append(file_ref(root/'setup'/output['path']))
bundle=load_json(root/'setup/bundle.json')
for field in ('u100_manifest','offline_manifest'):
    ref=bundle[field]; validate_file_ref(ref); files.append(ref)
    manifest=load_json(ref['path']); payload=manifest['payload']
    validate_file_ref(payload,root=Path(ref['path']).parent)
    files.append(file_ref(Path(ref['path']).parent/payload['path']))
for arm in s['arms']:
    folder=root/arm
    if not (folder/'receipt.json').exists(): continue
    receipt=worker._receipt(s,arm)
    refs=[file_ref(folder/'receipt.json')]+[file_ref(folder/r['path']) for r in receipt['outputs']]
    completed[arm]=dict(job_id=receipt['job_id'],files=refs,
        report=file_ref(folder/'fit/training_report.json'),
        predictions=file_ref(folder/'fit/validation_predictions.npz'))
print('SERIAL_PARENT='+json.dumps(dict(jobs=jobs,setup_jobs=setup_jobs,completed=completed,
    setup_files=files,lineage_files=lineage_files,acceptance=resources.accepted(s),results=result),sort_keys=True,allow_nan=False))
'''


def inspect_parent(path):
    path = Path(path).resolve(strict=True); old = load_json(path)
    if old['source_commit'] != PARENT_COMMIT or path != Path(old['root'])/'campaign_spec.json':
        raise ValueError('Expected the successful 431341db contrastive campaign')
    ref, evidence = replay(path, old['project_dir'], REPLAY, 'SERIAL_PARENT=')
    return artifact('PARENT', parents=dict(spec=old['content_hash']), spec=ref, **evidence)


def validate_parent(value):
    validate(value, 'PARENT'); validate_file_ref(value['spec']); old = load_json(value['spec']['path'])
    if old['source_commit'] != PARENT_COMMIT or value['parents'] != dict(spec=old['content_hash']):
        raise ValueError('Original source/parent identity differs')
    validate_source_snapshot(old['source_snapshot'], repository=old['project_dir'], require_clean=True)
    for ref in [*value['setup_files'], *value['lineage_files']]:
        validate_file_ref(ref)
    root = Path(old['root'])
    required = {str(root/name/file) for name in ('submission_setup','submission_science')
                for file in ('command_plan.json','submission_ledger.json')}
    if (not required <= {ref['path'] for ref in value['lineage_files']}
            or load_json(root/'setup/acceptance.json') != value['acceptance']
            or load_json(root/'submission_science/submission_ledger.json')['jobs'] != value['jobs']
            or load_json(root/'submission_setup/submission_ledger.json')['jobs'] != value['setup_jobs']):
        raise ValueError('Original accepted jobs/resource evidence differs')
    for row in value['completed'].values():
        for ref in row['files']:
            validate_file_ref(ref)
    return old
