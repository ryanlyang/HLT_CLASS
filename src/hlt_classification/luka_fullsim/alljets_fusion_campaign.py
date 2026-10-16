"""Additive eight-fit continuation; never rewrite a parent artifact or role."""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash, validate_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from . import alljets_campaign as old, alljets_fusion_source as sources
from .science_campaign import nodes as fusion_nodes, preflight_accounting

COUNTS = dict(old.COUNTS)
MEMORY_MB = 256000
AUTHORIZATION = 'AUTHORIZE LUKA ALLJETS FUSION BOTH ROUTES EXACT PLAN'
FLAGS = dict(final_test_accessed=False, historical_test_evaluated=True,
    historical_test_repurposed_for_training=True, independent_test_available=False,
    evaluation_scope='exploratory_development_validation')


V2_KINDS = {'GATE', 'SPEC', 'PREFLIGHT', 'PLAN', 'TASK', 'RESULTS', 'DIAGNOSTICS', 'SUBMISSION_CLAIM'}


def artifact(kind, *, parents=None, version=1, **fields):
    if version != 1 and not (version in (2, 3, 4, 5, 6) and kind in V2_KINDS) and not (version in (3, 4) and kind == 'PROBE'):
        raise ValueError('Unsupported fusion artifact version')
    return with_content_hash(dict(contract=f'LUKA_ALLJETS_FUSION_{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), **FLAGS, **fields))


def bound_artifact(scope, kind, **fields):
    return artifact(kind, version=scope.get('schema_version', 1), **fields)


def validate(value, kind, parents=None):
    version = value.get('schema_version')
    if type(version) is not int or (version != 1 and not (version in (2, 3, 4, 5, 6) and kind in V2_KINDS) and not (version in (3, 4) and kind == 'PROBE')):
        raise ValueError('Unsupported fusion artifact version')
    digest = validate_content_hash(value, expected_contract=f'LUKA_ALLJETS_FUSION_{kind}/v{version}', expected_schema_version=version)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory/no-independent-test scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Fusion parent lineage differs')
    return digest


def validate_bound(value, scope, kind, parents=None):
    digest = validate(value, kind, parents)
    if value['schema_version'] != scope.get('schema_version', 1):
        raise ValueError('Fusion execution versions differ')
    return digest


def nodes():
    return fusion_nodes()[4:]


def tasks():
    rows, teachers = [], {n['teacher'] for n in nodes()}-{'U000'}
    for n in nodes():
        name = n['node_id']
        rows.append(dict(task_id='train_'+name, kind='train', node_id=name,
            dependencies=[] if n['teacher'] == 'U000' else ['reduce_'+n['teacher']]))
        if name in teachers:
            rows.append(dict(task_id='reduce_'+name, kind='reduce', node_id=name, dependencies=['train_'+name]))
    rows.append(dict(task_id='aggregate', kind='aggregate', node_id=None, dependencies=[r['task_id'] for r in rows]))
    rows.append(dict(task_id='complete', kind='complete', node_id=None, dependencies=['aggregate']))
    return rows


def scoped_nodes(scope):
    return old.nodes() + nodes() if scope.get('schema_version') in (4, 5, 6) else nodes()


def scoped_tasks(scope):
    if scope.get('schema_version') not in (4, 5, 6):
        return tasks()
    rows = []
    models = scoped_nodes(scope)
    teachers = {n['teacher'] for n in models} - {None}
    for n in models:
        name = n['node_id']
        rows.append(dict(task_id='train_'+name, kind='train', node_id=name,
            dependencies=[] if n['teacher'] is None else ['reduce_'+n['teacher']]))
        if name in teachers:
            rows.append(dict(task_id='reduce_'+name, kind='reduce', node_id=name,
                dependencies=['train_'+name]))
    rows.append(dict(task_id='aggregate', kind='aggregate', node_id=None,
                     dependencies=[t['task_id'] for t in rows]))
    rows.append(dict(task_id='complete', kind='complete', node_id=None, dependencies=['aggregate']))
    return rows


def completion(scope):
    return (dict(fits=17, reducers=11, imported_comparisons=0, performance_never_gates_completion=True)
            if scope.get('schema_version') in (4, 5, 6) else
            dict(fits=8, reducers=6, imported_comparisons=9, performance_never_gates_completion=True))


def authorization(scope):
    if scope.get('schema_version') == 6:
        from .alljets_30epoch import AUTHORIZATION as thirty_authorization
        return thirty_authorization
    if scope.get('schema_version') == 5:
        from .alljets_report_recovery import AUTHORIZATION as recovery_authorization
        return recovery_authorization
    return 'AUTHORIZE LUKA MATCHED BATCH128 EXACT PLAN' if scope.get('schema_version') == 4 else AUTHORIZATION


def registration(version=1):
    if version == 6:
        from .alljets_30epoch import POLICY
        from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe
        from copy import deepcopy
        return dict(registration(4), training_recipe=recipe(), gate_minutes=240,
            schedule_reuse_policy=deepcopy(POLICY),
            preflight_execution=dict(version='cpu_only_fixed30_resource_revalidation_v1',
                                     gpus=0, repeats_gpu_probes=False),
            scientific_comparison='all17_refitted_true128_exact30_epochs_warmup3_cosine30')
    if version == 5:
        from .alljets_report_recovery import POLICY
        from copy import deepcopy
        return dict(registration(4), report_recovery_policy=deepcopy(POLICY), gate_minutes=240)
    if version == 4:
        from .alljets_batch128 import POLICY
        from hlt_classification.jetclass2_delphes.batch128_recipe import recipe
        from copy import deepcopy
        scope = dict(schema_version=4)
        return dict(registration(3), nodes=scoped_nodes(scope), tasks=scoped_tasks(scope),
            training_recipe=recipe(), fits=17, reducers=11, jobs=30, comparison_parent_fits=0,
            imported_train_banks=[], preflight_execution=deepcopy(POLICY),
            scientific_comparison='all_models_refitted_at_true_batch128_no_accumulation')
    extra = {}
    if version in (2, 3):
        from .alljets_fusion_sync import PAIR_STORAGE
        extra['pair_storage'] = dict(PAIR_STORAGE)
        if version == 3:
            from .alljets_fusion_isolated_preflight import POLICY
            from copy import deepcopy
            extra['preflight_execution'] = deepcopy(POLICY)
    elif version != 1:
        raise ValueError('Unsupported fusion registration')
    return dict(counts=COUNTS, nodes=nodes(), tasks=tasks(), training_recipe=old.recipe(),
        fits=8, reducers=6, jobs=16, comparison_parent_fits=9, cpus=6, memory_mb=MEMORY_MB,
        execution_site=execution_site('sporc_a100_debug'), gate_minutes=480,
        alpha=1., fusion_injections=[2,4,6,8], direction='context_to_primary',
        imported_weights=[], imported_train_banks=['U000'], new_models_cold_start=True,
        validation_policy='entire_74895_checkpoint_selection_original_50000_secondary_reporting_only',
        automatic_followup=False, automatic_retry=False, **extra)


def protect(root, parent, project):
    protected = (project, parent['project_dir'], Path(parent['gate_path']).parent,
                 parent['admission']['prepared_root'])
    gate = load_json(parent['gate_path'])
    for name in (*protected, gate['input_container']):
        path = Path(name).resolve()
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise PermissionError('New campaign overlaps protected parent/source/raw inputs')


def create_gate(*, root, parent_path, project, commit):
    project, root = Path(project).resolve(), Path(root).resolve()
    active = sources.source(project, commit)
    imported, parent = sources.authenticate(parent_path, active=active, project=project)
    protect(root, parent, project)
    gate = artifact('GATE', parents=dict(source=active['content_hash'], parent=parent['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        parent_import=imported, prepared_sha256=parent['prepared_sha256'], **registration())
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate(gate):
    if gate.get('schema_version') == 6:
        from .alljets_30epoch import validate_gate as thirty_gate
        return thirty_gate(gate)
    if gate.get('schema_version') == 5:
        from .alljets_report_recovery import validate_gate as recovered_report_gate
        return recovered_report_gate(gate)
    if gate.get('schema_version') == 4:
        from .alljets_batch128 import validate_gate as batch128_gate
        return batch128_gate(gate)
    if gate.get('schema_version') == 3:
        from .alljets_fusion_isolated_recovery import validate_gate as isolated_gate
        return isolated_gate(gate)
    if gate.get('schema_version') == 2:
        from .alljets_fusion_recovery import validate_gate as recovered_gate
        return recovered_gate(gate)
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], parent=gate['parent_import']['evidence']['spec']))
    if (any(gate.get(k) != v for k,v in registration().items())
            or sources.source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Fusion gate registration/source differs')
    validate_file_ref(gate['parent_import']['spec'])
    imported, parent = sources.authenticate(gate['parent_import']['spec']['path'],
        active=gate['source_snapshot'], project=gate['project_dir'])
    if imported != gate['parent_import'] or gate['prepared_sha256'] != parent['prepared_sha256']:
        raise ValueError('Parent campaign changed')
    protect(Path(gate['root']).resolve(), parent, gate['project_dir'])
    return parent


def evidence(gate):
    if gate.get('schema_version') == 6:
        from .alljets_30epoch import evidence as thirty_evidence
        return thirty_evidence(gate)
    if gate.get('schema_version') == 5:
        from .alljets_report_recovery import evidence as recovered_evidence
        return recovered_evidence(gate)
    from .alljets_fusion_preflight import validate_evidence
    from .regular_cache import metadata
    parent = validate_gate(gate)
    m, p, _, _ = metadata(parent['admission']['prepared_root'])
    if (p['content_hash'] != gate['prepared_sha256']
            or m['content_hash'] != gate['parent_import']['evidence']['cache']):
        raise ValueError('Parent cache/preparation differs')
    output = Path(gate['root'])/'preflight'
    report = load_json(output/'preflight.json')
    validate_evidence(report, gate, p)
    for ref in report['outputs']:
        validate_file_ref(ref, root=output)
    runtime = load_json(output/'runtime.json')
    if runtime != {k:report[k] for k in ('job_id','environment','gpu','cublas_workspace_config','cudnn_v8_disabled')}:
        raise ValueError('Bound preflight runtime differs')
    from hlt_classification.jetclass2_delphes.contracts import validate as kernel_validate
    for measured in report['measurements']:
        name = measured['node']['node_id']+'_bank'
        if gate['schema_version'] in (3, 4):
            name = 'children/measure_'+measured['node']['node_id']+'/'+name
        bank = load_json(output/name/'manifest.json')
        kernel_validate(bank, 'PROBABILITY_BANK')
        expected = dict(foundation_sha256=p['content_hash'], teacher_report_sha256=measured['training']['content_hash'],
            teacher_node=measured['node']['node_id'], role='train', rows=COUNTS['train'], temperature=2.,
            class_count=11, dtype='float32', final_test_accessed=False)
        if any(bank.get(k) != v for k,v in expected.items()):
            raise ValueError('Technical TRAIN bank lineage differs')
        for shard in bank['shards']:
            if dict(path=name+'/'+shard['path'], bytes=shard['bytes'], sha256=shard['sha256']) not in report['outputs']:
                raise ValueError('Missing bound technical bank shard')
    return parent, p, report


def create_science(gate):
    from .alljets_fusion_submission import accepted_jobs
    parent, p, report = evidence(gate)
    jobs = accepted_jobs(gate)
    if jobs != {'preflight': str(report['job_id'])}:
        raise ValueError('Accepted preflight differs from exact submitted journal')
    root = Path(gate['root'])/'science'
    spec = bound_artifact(gate, 'SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'], source_snapshot=gate['source_snapshot'],
        parent_import=gate['parent_import'], prepared_sha256=p['content_hash'], preflight_sha256=report['content_hash'],
        preflight_accounting=preflight_accounting(report['job_id']),
        train_minutes=report['projected_train_minutes'], reduce_minutes=report['projected_reduce_minutes'], **registration(gate['schema_version']))
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec):
    from .alljets_fusion_submission import accepted_jobs
    gate = load_json(spec['gate_path'])
    parent, p, report = evidence(gate)
    validate_bound(spec, gate, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    expected = dict(**registration(gate['schema_version']), parent_import=gate['parent_import'],
        project_dir=gate['project_dir'], source_commit=gate['source_commit'], source_snapshot=gate['source_snapshot'],
        prepared_sha256=p['content_hash'], preflight_sha256=report['content_hash'],
        train_minutes=report['projected_train_minutes'], reduce_minutes=report['projected_reduce_minutes'],
        preflight_accounting=dict(job_id=str(report['job_id']), state='COMPLETED', exit_code='0:0'))
    if (any(spec.get(k) != v for k,v in expected.items())
            or Path(spec['root']) != Path(gate['root'])/'science'
            or accepted_jobs(gate) != {'preflight':str(report['job_id'])}
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Fusion science/admission/resources differ')
    return parent, p, report
