"""Authenticate the completed control in its own source; never rerun it."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_attention_weight.source import replay
from hlt_classification.luka_u100_multiview.source import validate_import
from .contracts import artifact, validate, REFERENCE_JOB, COUNTS

BASE = '431341db5eced8eaa3e52b37ba047f8cd4beed50'
MODULES = ('__init__', 'contracts', 'source', 'targets', 'geometry', 'model', 'losses',
           'training', 'resources', 'campaign', 'submission', 'worker')
RUNTIME = {'src/hlt_classification/luka_u100_pair_correction/'+n+'.py' for n in MODULES} | {
    'scripts/luka_u100_pair_correction.py', 'scripts/queue_luka_u100_pair_correction.sh',
    'sbatch/run_luka_u100_pair_correction.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_u100_pair_correction.py',
    'docs/contracts/LUKA_U100_PAIR_CORRECTION.md', 'docs/LUKA_U100_PAIR_CORRECTION_RUNBOOK.md',
    'docs/plans/LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md'}


def compatible_changes(rows):
    return all(r[:2] in ('A\t', 'M\t') and r[2:] in RUNTIME for r in rows)


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    if not REQUIRED <= set(git('ls-files')) or not compatible_changes(
            git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')):
        raise ValueError('Commit complete additive correction implementation; preserve existing scientific runtime')
    if Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_pair_correction/source.py':
        raise ValueError('Runtime import escapes pinned source')
    return snapshot


REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.luka_u100_contrastive import campaign,worker,training,resources,contracts as c
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
r=Path(s['root']); task='u100_w5_s0'
worker.completed_setup(s,metadata_only_keys=True)
receipt=worker._receipt(s,task)
if receipt['job_id']!='21993056': raise ValueError('Expected completed control job 21993056')
report,a=training.validate_fit(r/task/'fit',s,task)
bundle=load_json(r/'setup/bundle.json'); c.validate(bundle,'BUNDLE',dict(spec=s['content_hash']))
manifest=load_json(bundle['u100_manifest']['path'])
if report['parents']!=dict(spec=s['content_hash'],recipe=s['recipe']['content_hash'],
    prepared=s['anchor']['evidence']['prepared'],u100=manifest['content_hash']):
    raise ValueError('Completed control bank lineage differs')
validate_file_ref(bundle['u100_manifest'])
validate_file_ref(manifest['payload'],root=Path(bundle['u100_manifest']['path']).parent)
print('PAIR_CONTROL='+json.dumps(dict(imported=s['imported'],counts=s['counts'],
    environment=resources.accepted(s)['environment'],
    u100_manifest=bundle['u100_manifest'],job_id=receipt['job_id'],
    files={k:file_ref(r/v) for k,v in dict(receipt=task+'/receipt.json',
    fit=task+'/fit/receipt.json',report=task+'/fit/training_report.json',
    checkpoint=task+'/fit/selected.pt',predictions=task+'/fit/validation_predictions.npz',
    bundle='setup/bundle.json',setup_receipt='setup/receipt.json',acceptance='setup/acceptance.json').items()}),sort_keys=True,allow_nan=False))
'''


def import_control(path):
    path = Path(path).resolve(strict=True); old = load_json(path)
    validate_source_snapshot(old['source_snapshot'], repository=old['project_dir'], require_clean=True)
    if path != Path(old['root'])/'campaign_spec.json':
        raise ValueError('Reference must remain at its registered location')
    ref, evidence = replay(path, old['project_dir'], REPLAY, 'PAIR_CONTROL=')
    if evidence['counts'] != COUNTS or evidence['job_id'] != REFERENCE_JOB:
        raise ValueError('Wrong completed control/population')
    result = artifact('REFERENCE', parents=dict(spec=old['content_hash'], imported=evidence['imported']['content_hash']),
        spec=ref, source=old['source_snapshot'], project=old['project_dir'], **evidence)
    validate_reference(result)
    return result


def validate_reference(value):
    validate(value, 'REFERENCE')
    validate_file_ref(value['spec']); old = load_json(value['spec']['path'])
    validate_source_snapshot(value['source'], repository=value['project'], require_clean=True)
    validate_import(value['imported'])
    if (value['job_id'] != REFERENCE_JOB or value['counts'] != COUNTS
            or value['source'] != old['source_snapshot'] or value['project'] != old['project_dir']
            or value['imported'] != old['imported'] or value['parents'] != dict(
                spec=old['content_hash'], imported=value['imported']['content_hash'])):
        raise ValueError('Reference lineage changed')
    for ref in value['files'].values():
        validate_file_ref(ref)
    if load_json(value['files']['acceptance']['path'])['environment'] != value['environment']:
        raise ValueError('Control installed environment differs')
    validate_file_ref(value['u100_manifest'])
    manifest = load_json(value['u100_manifest']['path'])
    validate_file_ref(manifest['payload'], root=Path(value['u100_manifest']['path']).parent)
    return value
