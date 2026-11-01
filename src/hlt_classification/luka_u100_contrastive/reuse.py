"""Read-only recovery of published exports from the failed v1 resource admission."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, validate_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_attention_weight.source import replay
from .contracts import artifact, validate, recipe, sampler_recipe, COUNTS

ORIGINAL_COMMIT = '755f43d874ce90ff50ae9b80fe66d36869601650'
# Executed with the immutable original source, never with the repaired validators.
REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_contrastive import campaign, submission, resources
from hlt_classification.luka_u100_contrastive.contracts import validate
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
r=Path(s['root']); folder=r/'setup'
jobs=submission.accepted_jobs(s,'setup')
if set(jobs)!={'setup'} or (r/'submission_science').exists():
    raise ValueError('Expected one failed setup with no science submission attempt')
job=jobs['setup']; state=submission.accounting(job)
if state!=['FAILED','1:0']:
    raise ValueError('Original setup must be terminal FAILED/1:0; do not reuse active output')
claim=load_json(r/'claims/setup/intent.json'); validate(claim,'RUN_CLAIM',dict(spec=s['content_hash']))
if claim['task']!='setup' or claim['job_id']!=job:
    raise ValueError('Setup ownership differs')
if (folder/'receipt.json').exists() or (folder/'acceptance.json').exists():
    raise ValueError('Expected failed admission, not a completed setup')
d=load_json(folder/'resource_diagnostic.json')
if d['status']!='complete' or d['error'] is not None or not d['baseline_native_tensor_parity']:
    raise ValueError('Incomplete technical execution')
try:
    resources.accept_diagnostic(d,s)
except ValueError as exc:
    if 'exceeds registered walltime cap' not in str(exc): raise
else:
    raise ValueError('Expected the diagnosed walltime admission failure')
print('CONTRASTIVE_EXPORT_REUSE='+json.dumps(dict(job_id=job,accounting=state)))
'''


def original_spec(path):
    path = Path(path).resolve(strict=True); old = load_json(path)
    validate_content_hash(old, expected_contract='LUKA_U100_CONTRASTIVE_SPEC/v1')
    if (old['source_commit'] != ORIGINAL_COMMIT or old['schema_version'] != 1
            or path != Path(old['root'])/'campaign_spec.json'):
        raise ValueError('Expected the original v1 contrastive campaign, at its registered path')
    return old


def import_exports(path, imported):
    old = original_spec(path)
    ref, evidence = replay(path, old['project_dir'], REPLAY, 'CONTRASTIVE_EXPORT_REUSE=')
    root = Path(old['root']); folder = root/'setup'
    bundle = load_json(folder/'bundle.json')
    readbacks = [file_ref(folder/'offline_readback.json')]
    if not bundle['reused_original_u100_payload']:
        readbacks.append(file_ref(folder/'u100_readback.json'))
    value = artifact('SETUP_REUSE', parents=dict(origin=old['content_hash'],
        source=old['source_snapshot']['content_hash'], imported=imported['content_hash']),
        origin_spec=ref, origin_bundle=file_ref(folder/'bundle.json'),
        u100_manifest=bundle['u100_manifest'], offline_manifest=bundle['offline_manifest'],
        readbacks=readbacks, diagnostic=file_ref(folder/'resource_diagnostic.json'),
        claim=file_ref(root/'claims/setup/intent.json'),
        setup_job=evidence['job_id'], terminal_accounting=evidence['accounting'],
        scope='published_targets_only_new_miniatures_and_resource_measurements_required')
    validate_reuse(value, imported)
    return value


def validate_reuse(value, imported, *, metadata_only=False):
    validate(value, 'SETUP_REUSE')
    for key in ('origin_spec', 'origin_bundle', 'u100_manifest', 'offline_manifest', 'diagnostic', 'claim'):
        validate_file_ref(value[key])
    for ref in value['readbacks']:
        validate_file_ref(ref)
    old = original_spec(value['origin_spec']['path']); folder = Path(old['root'])/'setup'
    validate(value, 'SETUP_REUSE', dict(origin=old['content_hash'],
        source=old['source_snapshot']['content_hash'], imported=imported['content_hash']))
    validate_source_snapshot(old['source_snapshot'], repository=old['project_dir'], require_clean=True)
    if (old['imported'] != imported or old['recipe'] != recipe() or old['sampler'] != sampler_recipe()
            or old['counts'] != COUNTS or value['terminal_accounting'] != ['FAILED', '1:0']
            or value['scope'] != 'published_targets_only_new_miniatures_and_resource_measurements_required'
            or (Path(old['root'])/'submission_science').exists()):
        raise ValueError('Export reuse population/teacher/recipe or execution scope differs')
    expected_paths = dict(origin_bundle=folder/'bundle.json', diagnostic=folder/'resource_diagnostic.json',
                          claim=Path(old['root'])/'claims/setup/intent.json', offline_manifest=folder/'OFFLINE/manifest.json')
    if any(Path(value[k]['path']) != p for k, p in expected_paths.items()):
        raise ValueError('Export evidence is outside its original setup')
    bundle = load_json(value['origin_bundle']['path'])
    validate_content_hash(bundle, expected_contract='LUKA_U100_CONTRASTIVE_BUNDLE/v1')
    if (bundle['parents'] != dict(spec=old['content_hash']) or bundle['teachers_retrained'] is not False
            or any(value[k] != bundle[k] for k in ('u100_manifest', 'offline_manifest'))):
        raise ValueError('Original export bundle differs')
    expected_readbacks = [folder/'offline_readback.json']
    if not bundle['reused_original_u100_payload']:
        expected_readbacks.append(folder/'u100_readback.json')
    if [Path(r['path']) for r in value['readbacks']] != expected_readbacks:
        raise ValueError('Missing original teacher readback')
    for ref, coordinate in zip(value['readbacks'], ('OFFLINE', 'U100')):
        proof = load_json(ref['path'])
        parents = (dict(spec=old['content_hash'], imported=imported['content_hash']) if coordinate == 'OFFLINE'
                   else dict(spec=old['content_hash'], teacher=imported['teacher']['content_hash']))
        validate(proof, 'TEACHER_READBACK', parents)
        if (proof['native_probability_parity'] is not True or proof['coordinate'] != coordinate
                or proof['rows'] != COUNTS['validation'] or (proof['rtol'], proof['atol']) != (1e-5, 1e-6)):
            raise ValueError('Original teacher readback failed')
    claim = load_json(value['claim']['path']); validate(claim, 'RUN_CLAIM', dict(spec=old['content_hash']))
    if claim['job_id'] != value['setup_job'] or claim['task'] != 'setup':
        raise ValueError('Original job ownership differs')
    diag = load_json(value['diagnostic']['path']); validate(diag, 'RESOURCE_DIAGNOSTIC', dict(spec=old['content_hash']))
    if diag['status'] != 'complete' or diag['error'] is not None or diag['baseline_native_tensor_parity'] is not True:
        raise ValueError('Original technical checks did not complete')
    manifest = load_json(value['offline_manifest']['path'])
    validate(manifest, 'KEYS', dict(spec=old['content_hash'], imported=imported['content_hash'],
        prepared=old['anchor']['evidence']['prepared'], sampler=sampler_recipe()['content_hash']))
    if manifest['teacher'] != old['anchor']['evidence']['models']['OFFLINE']['checkpoint']:
        raise ValueError('Original OFFLINE teacher differs')
    if not metadata_only:
        validate_file_ref(manifest['payload'], root=folder/'OFFLINE')
    return old


def publish_bundle(spec, load_seconds):
    value = spec['setup_reuse']; validate_reuse(value, spec['imported'])
    root = Path(spec['root'])/'setup'
    write_immutable_json(root/'reused_exports.json', artifact('REUSED_EXPORTS',
        parents=dict(spec=spec['content_hash'], reuse=value['content_hash']), reference=value))
    bundle = artifact('BUNDLE', parents=dict(spec=spec['content_hash']),
        u100_manifest=value['u100_manifest'], offline_manifest=value['offline_manifest'],
        reused_original_u100_payload=True, teachers_retrained=False, setup_load_seconds=load_seconds,
        reused_setup=value['content_hash'])
    write_immutable_json(root/'bundle.json', bundle)
    return bundle
