"""Versioned CONTEXT_V2 endpoint; exact V1 population and unchanged training."""
from pathlib import Path
import sys

from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json
from hlt_classification.context_v2 import dataset as d, workflow as w
from hlt_classification.context_v2.contracts import require, checked
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, file_ref, validate_file_ref, write_json
from . import context as old

KIND = 'controlled_context_v2_synthetic_proxy'
GATE_AUTHORIZATION = 'AUTHORIZE JETCLASS2 CONTEXT V2 OSCAR GATE'
SCIENCE_AUTHORIZATION = 'AUTHORIZE JETCLASS2 CONTEXT V2 OSCAR DIRECT COARSE SCIENCE'


def reader(request):
    return d.Dataset(request['study_root'])


def release_request(dataset_root):
    value = d.Dataset(dataset_root)
    return artifact('RELEASE_REQUEST', version=5, study_root=str(value.proxy_root),
        offline_root=str(value.offline_root), provenance_root=str(value.provenance_root),
        manifest_hash=value.manifest['content_hash'], counts=w.policy()['counts'],
        study_ref=file_ref(value.proxy_root/'study_spec.json'),
        manifest_ref=file_ref(value.proxy_root/'dataset_manifest.json'),
        dataset_kind=KIND, recipe='CONTEXT_V2', selection_domain=value.spec['source_request']['selection_domain'],
        labels_read=False, selection_depends_on_labels=False, allowed_roles=['train', 'validation'],
        original_release=value.spec['original_release'], workflow_hash=value.spec['content_hash'])


def validate_request(value):
    digest = validate(value, 'RELEASE_REQUEST', version=5)
    validate_file_ref(value['study_ref'])
    validate_file_ref(value['manifest_ref'])
    require(value == release_request(value['study_root']), 'V2 request/population/source differs')
    return digest


def _bindings(value):
    consumer = reader(value['request'])
    original = load_json(checked(consumer.spec['original_release']))
    receipts, blocks = [], {}
    for shard in sorted(consumer.spec['shards'], key=lambda s: s['shard_id']):
        if shard['role'] not in ('train', 'validation'):
            continue
        row = consumer._receipt(shard)
        receipts.append({k: row[k] for k in ('shard_id', 'role', 'content_hash', 'ordered_identities')})
        for b in row['blocks']:
            require(b['relative'] not in blocks, 'Duplicate V2 block')
            blocks[b['relative']] = dict(path=b['relative'], sha256=b['sha256'], bytes=b['bytes'])
    ordered = [blocks[b['path']] for b in original['proxy_blocks']]
    return consumer, original, receipts, ordered


def validate_release_source(value):
    consumer, original, receipts, blocks = _bindings(value)
    require(value['study_contract'] == 'JC2_CONTEXT_V2_STUDY/v1'
        and value['parents']['study'] == consumer.study['content_hash']
        and value['study_root'] == str(consumer.proxy_root) and value['offline_root'] == str(consumer.offline_root)
        and value['source_files'] == original['source_files'] and value['receipts'] == receipts
        and value['proxy_blocks'] == blocks and value['identity_sha256'] == original['identity_sha256']
        and value['bank'] == original['bank'], 'V2 release must preserve the original V1 index byte-for-byte')


def build_release(request, *, output_root):
    from .release import validate_release
    validate_request(request)
    root = Path(output_root).resolve()
    require(not root.exists(), 'Fresh release root required')
    consumer, original, receipts, blocks = _bindings(dict(request=request))
    original_root = Path(request['original_release']['path']).parent
    bank_path = validate_file_ref(original['bank'], root=original_root)
    for b in blocks:
        validate_file_ref(b, root=consumer.proxy_root)
    for f in original['source_files']:
        validate_file_ref(f, root=consumer.offline_root)
    root.mkdir(parents=True)
    atomic_publish_bytes(root/original['bank']['path'], bank_path.read_bytes())
    result = artifact('RELEASE', version=5,
        parents=dict(request=request['content_hash'], study=consumer.study['content_hash']),
        request=request, study_contract=consumer.study['contract'], study_root=str(consumer.proxy_root),
        offline_root=str(consumer.offline_root), scope='complete_dataset_manifest', counts=request['counts'],
        selection_domain=request['selection_domain'], labels_read=False, selection_depends_on_labels=False,
        receipts=receipts, source_files=original['source_files'], proxy_blocks=blocks,
        bank=file_ref(root/original['bank']['path'], root=root), identity_sha256=original['identity_sha256'],
        total_rows=original['total_rows'])
    validate_release(result, root=root)
    write_json(root/'release.json', result)
    return result


def view_contract():
    from .views import view_contract as original
    v = original(context=True)
    fields = {k: val for k, val in v.items() if k not in
              ('contract', 'schema_version', 'content_hash', 'parents', 'final_test_accessed')}
    fields['support'] = 'context_v2_persistent_shell_offline_tail_v1'
    fields['endpoints'] = dict(fields['endpoints'], D000='exact_frozen_context_v2_proxy')
    return artifact('VIEWS', version=5, parents=v['parents'], **fields)


def create_gate(spec):
    root = Path(spec['root'])/'gate'
    require(not root.exists(), 'Fresh gate root required')
    request = release_request(spec['dataset_root'])
    result = artifact('GATE_SPEC', version=10,
        parents=dict(source=spec['source']['content_hash'], request=request['content_hash']),
        source=spec['source'], request=request, gate_root=str(root), project_dir=spec['project_dir'],
        source_commit=spec['source_commit'], capacity=512, execution_site=execution_site('oscar_l40s'),
        tasks=old.gate_tasks(), workers=6, foundation_workers=6, scientific_branches=['DIRECT', 'COARSE'],
        dataset_kind=KIND, full_views_persisted=False, site_transfer_policy=None,
        admission='context_v2_workflow_integrity_runtime_only')
    validate_gate(result, check_source=True)
    root.mkdir()
    write_json(root/'gate_spec.json', result)
    return result


def validate_gate(spec, *, check_source=False):
    digest = validate(spec, 'GATE_SPEC', version=10, parents=dict(
        source=spec['source']['content_hash'], request=spec['request']['content_hash']))
    validate_request(spec['request'])
    value = reader(spec['request']).spec
    require(spec['source'] == value['source'] and spec['source_commit'] == value['source_commit']
        and spec['project_dir'] == value['project_dir'] and spec['gate_root'] == str(Path(value['root'])/'gate')
        and spec['execution_site'] == execution_site('oscar_l40s') and spec['tasks'] == old.gate_tasks()
        and spec['workers'] == spec['foundation_workers'] == 6 and spec['capacity'] == 512
        and spec['scientific_branches'] == ['DIRECT', 'COARSE'] and spec['dataset_kind'] == KIND
        and spec['site_transfer_policy'] is None and 'measurement_site' not in spec
        and spec['full_views_persisted'] is False
        and spec['admission'] == 'context_v2_workflow_integrity_runtime_only', 'V2 gate differs')
    if check_source:
        require(w.source_lock(Path(spec['project_dir']), spec['source_commit']) == spec['source'], 'V2 gate source drift')
    return digest


def scientific_plan(foundation, *, foundation_root=None):
    from .campaign import _build_scientific_plan, DIRECT_COARSE_BRANCHES
    require(foundation['schema_version'] == foundation['release']['schema_version'] == 5
        and foundation['role_counts'] == w.policy()['counts'], 'V2 science population differs')
    return _build_scientific_plan(foundation, registered_branches=DIRECT_COARSE_BRANCHES,
        version=7, foundation_root=foundation_root, node_prefix='CTXV2')


def validate_profile(profile, *, foundation, spec):
    return old.validate_profile(profile, foundation=foundation, spec=spec, _version=10, _plan_builder=scientific_plan)


def create_campaign(**kwargs):
    return old.create_campaign(**kwargs, _adapter=sys.modules[__name__])


def validate_campaign(spec, *, check_source=False):
    return old.validate_campaign(spec, check_source=check_source, _adapter=sys.modules[__name__])
