"""One preflight and one science fit, with exact-accounting admission."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from .contracts import artifact, validate, recipe, node, COUNTS, AUTHORIZATION
from . import source


def registration(parent, partition):
    if partition not in ('tier3', 'debug'):
        raise ValueError('Registered SPORC tier3 or explicit debug only')
    return dict(counts=COUNTS, original50k_rows=50000, node=node(parent), recipe=recipe(),
        execution_site=execution_site('sporc_a100' if partition == 'tier3' else 'sporc_a100_debug'),
        cpus=6, memory_mb=90000, gate_minutes=480, science_ceiling_minutes=1440,
        automatic_retry=False, automatic_followup=False, storage_budget_bytes=2*2**30)


def create(root, parent_path, project, commit, *, available_quota_gib, partition='tier3'):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh root overlaps protected source/data/campaign')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Declare at least 2 personally available persistent GiB')
    gate = artifact('GATE', parents=dict(source=active['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        anchor=anchor, available_quota_gib=available_quota_gib, **registration(parent, partition))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def check_gate(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], anchor=gate['anchor']['content_hash']))
    validate(gate['anchor'], 'ANCHOR')
    if (source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate
            or type(gate['available_quota_gib']) is not int or gate['available_quota_gib'] < 2):
        raise ValueError('Gate source/record/storage declaration differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'])
    if (anchor != gate['anchor'] or any(gate.get(k) != v for k, v in
            registration(parent, gate['execution_site']['partition']).items())):
        raise ValueError('Matched scientific registration/parent differs')
    return parent


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != {'preflight': report['job_id']}:
        raise ValueError('Accepted report differs from exact gate journal')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'],
        train_minutes=report['train_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec_record(spec, gate, report):
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science'
            or Path(spec['gate_path']) != Path(gate['root'])/'gate_spec.json'
            or spec['project_dir'] != gate['project_dir'] or spec['source_commit'] != gate['source_commit']
            or spec['train_minutes'] != report['train_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Science registration differs')


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate_spec_record(spec, gate, report)
    return gate, report
