"""One shared gate; three independently initialized and submitted scientific fits."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from .contracts import artifact, validate, nodes, WEIGHTS, COUNTS
from . import source


def registration(parent):
    return dict(counts=COUNTS, original50k_rows=50000, nodes=nodes(parent), weights=WEIGHTS,
        execution_site=execution_site('sporc_a100'), cpus=6, memory_mb=90000,
        gate_minutes=480, science_ceiling_minutes=1440, storage_budget_bytes=2*2**30,
        automatic_retry=False, automatic_followup=False)


def create(root, parent_path, embedding_path, project, commit, *, available_quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    imported = source.import_embedding(embedding_path, anchor)
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container']),
        Path(embedding_path).resolve().parent.parent]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New root overlaps shared artifacts/source')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Reserve at least 2 personally available persistent GiB')
    gate = artifact('GATE', parents=dict(source=snapshot['content_hash'],
        anchor=anchor['content_hash'], embedding=imported['content_hash']), root=str(root),
        project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        anchor=anchor, embedding=imported, available_quota_gib=available_quota_gib, **registration(parent))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate_record(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'],
        anchor=gate['anchor']['content_hash'], embedding=gate['embedding']['content_hash']))
    from hlt_classification.luka_embedding_aux.contracts import validate as old_validate
    from hlt_classification.luka_embedding_ladder.contracts import validate as ladder_validate
    old_validate(gate['anchor'], 'ANCHOR')
    ladder_validate(gate['embedding'], 'EMBEDDING_IMPORT')
    if gate['embedding']['parents']['anchor'] != gate['anchor']['content_hash']:
        raise ValueError('Embedding import anchor differs')
    # Runtime-only reuse is stricter; compact reporting does not require old payloads.
    parent_nodes = [dict(gate['nodes']['w075'], node_id='M0HLT', branch='BASELINE')]
    parent_nodes[0].pop('auxiliary_weight')
    expected = registration(dict(nodes=parent_nodes))
    if (any(gate.get(k) != v for k, v in expected.items())
            or type(gate['available_quota_gib']) is not int or gate['available_quota_gib'] < 2):
        raise ValueError('Registered weight nodes/resources/scope differ')


def check_gate(gate):
    validate_gate_record(gate)
    if (source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Gate source or immutable record differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'])
    imported = source.import_embedding(gate['embedding']['spec']['path'], anchor)
    if anchor != gate['anchor'] or imported != gate['embedding'] or any(
            gate.get(k) != v for k, v in registration(parent).items()):
        raise ValueError('Original source/target import differs')
    return parent


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != {'preflight': report['job_id']}:
        raise ValueError('Accepted gate journal differs')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'],
        train_minutes=report['train_minutes'], arms=list(WEIGHTS))
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec_record(spec, gate, report):
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science'
            or Path(spec['gate_path']) != Path(gate['root'])/'gate_spec.json'
            or spec['project_dir'] != gate['project_dir'] or spec['source_commit'] != gate['source_commit']
            or spec['arms'] != list(WEIGHTS) or spec['train_minutes'] != report['train_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Scientific registration differs')


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate_spec_record(spec, gate, report)
    return gate, report
