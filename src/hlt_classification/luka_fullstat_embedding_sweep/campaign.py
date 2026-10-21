"""Immutable Oscar registration and successful-gate admission."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from . import source
from .contracts import artifact, validate, recipe, nodes, tasks, COUNTS, WEIGHTS, AUTHORIZATION


def registration(parent):
    return dict(counts=dict(COUNTS), nodes=nodes(parent), tasks=tasks(), recipes={arm:recipe(w) for arm,w in WEIGHTS.items()},
        execution_site=execution_site('oscar_l40s'), cpus=6, memory_mb=180000,
        gate_minutes=480, max_minutes=2880, cpu_partition='batch',
        automatic_followup=False, automatic_retry=False, automatic_deletion=False)


def separate(root, anchor, project):
    root = Path(root).resolve()
    old = anchor['spec']['path']
    e = anchor['evidence']
    parent = load_json(old)
    for p in (project, parent['project_dir'], parent['root'], e['cache_root'],
              e['foundation_root'], e['raw_container']):
        other = Path(p).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise PermissionError('New output overlaps protected source/data/results')


def storage(gate):
    required = 12*2**30
    if gate['available_result_bytes'] < required or shutil.disk_usage(gate['root']).free < required:
        raise OSError('Need 12 GiB personally available durable result space; no files deleted')


def create(root, parent_path, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path, project)
    separate(root, anchor, project)
    if type(quota_gib) is not int or quota_gib < 12:
        raise ValueError('Declare at least 12 GiB personally available result storage')
    gate = artifact('GATE', parents=dict(source=active['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        anchor=anchor, available_result_bytes=quota_gib*2**30, **registration(parent))
    root.mkdir(parents=True, exist_ok=False)
    storage(gate)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def check_gate(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], anchor=gate['anchor']['content_hash']))
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Current source snapshot differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'], gate['project_dir'])
    if (anchor != gate['anchor'] or any(gate.get(k) != v for k, v in registration(parent).items())
            or type(gate['available_result_bytes']) is not int or gate['available_result_bytes'] < 12*2**30
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Gate/parent/scientific policy differs')
    separate(gate['root'], anchor, gate['project_dir'])
    return parent


def accepted(gate):
    from .submission import accepted_jobs
    from .preflight import validate_report
    check_gate(gate)
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != dict(preflight=report['job_id']):
        raise ValueError('Preflight differs from exact submission journal')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = accepted(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'],
        train_minutes=report['train_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = accepted(gate)
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science'
            or spec['project_dir'] != gate['project_dir'] or spec['source_commit'] != gate['source_commit']
            or spec['train_minutes'] != report['train_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Scientific execution registration differs')
    return gate, report
