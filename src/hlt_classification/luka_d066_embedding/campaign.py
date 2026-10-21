"""One target-source comparison with separate real gate and fresh science."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from .contracts import artifact, validate, recipe, node, COUNTS
from . import source


def registration(parent):
    return dict(counts=COUNTS, original50k_rows=50000, node=node(parent), recipe=recipe(),
        execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=90000,
        gate_minutes=480, science_ceiling_minutes=1440, automatic_retry=False,
        automatic_followup=False, storage_budget_bytes=2*2**30)


def create(root, parent_path, teacher_path, reference_path, project, commit, *, available_quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    teacher = source.teacher(teacher_path, anchor)
    reference = source.reference(reference_path, anchor, node(parent))
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    protected += [Path(p).resolve().parent.parent for p in (teacher_path,reference_path)]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Fresh root overlaps shared data/source/campaign')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Reserve2GiB personally available persistent storage')
    gate = artifact('GATE', parents=dict(source=snapshot['content_hash'], anchor=anchor['content_hash'],
        teacher=teacher['content_hash'], reference=reference['content_hash']), root=str(root),
        project_dir=str(project), source_commit=commit, source_snapshot=snapshot, anchor=anchor,
        teacher=teacher, reference=reference, available_quota_gib=available_quota_gib, **registration(parent))
    root.mkdir(parents=True, exist_ok=False); write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate_record(gate):
    validate(gate, 'GATE', {k: gate[k if k!='source' else 'source_snapshot']['content_hash']
                          for k in ('source','anchor','teacher','reference')})
    if type(gate['available_quota_gib']) is not int or gate['available_quota_gib'] < 2:
        raise ValueError('Storage allowance differs')
    parent = dict(nodes=[dict(gate['node'], node_id='M0HLT', branch='BASELINE')])
    parent['nodes'][0].pop('auxiliary_weight')
    if any(gate.get(k) != v for k,v in registration(parent).items()):
        raise ValueError('Registered scientific setup or resources differ')


def check_gate(gate):
    validate_gate_record(gate)
    if (source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Source or immutable gate changed')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'])
    if (anchor != gate['anchor'] or source.teacher(gate['teacher']['spec']['path'], anchor) != gate['teacher']
            or source.reference(gate['reference']['spec']['path'], anchor, node(parent)) != gate['reference']
            or any(gate.get(k) != v for k,v in registration(parent).items())):
        raise ValueError('Original-source input/reference lineage changed')
    return parent


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != {'preflight': report['job_id']}:
        raise ValueError('Exact gate submission journal differs')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate); root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'), project_dir=gate['project_dir'],
        source_commit=gate['source_commit'], train_minutes=report['train_minutes'])
    root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json', spec)
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
    gate = load_json(spec['gate_path']); report = evidence(gate)
    validate_spec_record(spec, gate, report)
    return gate, report
