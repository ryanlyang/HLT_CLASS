"""Separate Oscar gate/science registration and measured admission."""
from pathlib import Path
import math
import shutil

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from . import source
from .contracts import artifact, validate, recipe, nodes, tasks, COUNTS

AUTHORIZATION = 'AUTHORIZE FULLSTAT LAST ATTENTION EXACT PLAN'


def registration(parent):
    return dict(counts=COUNTS, nodes=nodes(parent['nodes']), tasks=tasks(), recipe=recipe(),
        execution_site=execution_site('oscar_l40s'), cpus=6, memory_mb=180000,
        gate_minutes=480, max_minutes=2880, cpu_partition='batch',
        automatic_followup=False, automatic_retry=False, automatic_deletion=False)


def separate(root, bank, protected):
    values = [Path(root).resolve(), Path(bank).resolve()]+[Path(p).resolve() for p in protected]
    for i in (0, 1):
        for other in values[i+1:]:
            if values[i].is_relative_to(other) or other.is_relative_to(values[i]):
                raise PermissionError('New output/bank overlaps source or protected artifacts')


def create(root, parent_path, bank_root, project, commit, *, quota_gib, result_quota_gib):
    root, bank_root, project = [Path(p).resolve() for p in (root, bank_root, project)]
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path, project)
    if any(type(v) is not int or v <= 0 for v in (quota_gib, result_quota_gib)):
        raise ValueError('Declare personally available bank and result quota GiB')
    e = anchor['evidence']
    separate(root, bank_root, [project, parent['project_dir'], Path(parent['root']).parent,
                              e['cache_root'], e['foundation_root'], e['raw_container']])
    if root.exists() or bank_root.exists():
        raise FileExistsError('Preserve existing outputs; use fresh explicit roots')
    gate = artifact('GATE', parents=dict(source=active['content_hash'], anchor=anchor['content_hash']),
        root=str(root), bank_root=str(bank_root), project_dir=str(project), source_commit=commit,
        source_snapshot=active, anchor=anchor, available_bank_bytes=quota_gib*2**30,
        available_result_bytes=result_quota_gib*2**30, **registration(parent))
    root.mkdir(parents=True, exist_ok=False)
    bank_root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def check_gate(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], anchor=gate['anchor']['content_hash']))
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Pinned source differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'], gate['project_dir'])
    if (anchor != gate['anchor'] or any(gate.get(k) != v for k,v in registration(parent).items())
            or load_json(Path(gate['root'])/'gate_spec.json') != gate
            or any(type(gate[k]) is not int or gate[k] <= 0 for k in ('available_bank_bytes','available_result_bytes'))):
        raise ValueError('Gate/parent registration differs')
    e = anchor['evidence']
    separate(gate['root'], gate['bank_root'], [gate['project_dir'], parent['project_dir'],
        Path(parent['root']).parent, e['cache_root'], e['foundation_root'], e['raw_container']])
    return parent


def storage(gate, sum_squared_counts, rows, *, check_space=True):
    if type(sum_squared_counts) is not int or sum_squared_counts < rows or rows != gate['counts']['train']:
        raise ValueError('Invalid measured full TRAIN attention support')
    payload = 4*sum_squared_counts
    # One shared science bank, population buffers/headers, temporary shard,
    # and reserve. Miniature banks/checkpoints live in durable result storage.
    bank_required = math.ceil(payload*1.1)+rows*128+2*2**30
    result_required = 8*2**30
    if bank_required > gate['available_bank_bytes'] or result_required > gate['available_result_bytes']:
        raise OSError(f'Need {bank_required/2**30:.2f} GiB bank and8 GiB result headroom; no files deleted')
    if check_space:
        free_bank, free_result = [shutil.disk_usage(gate[k]).free for k in ('bank_root','root')]
        if bank_required > free_bank or result_required > free_result:
            raise OSError('Insufficient physical free space; no files deleted')
        if Path(gate['bank_root']).stat().st_dev == Path(gate['root']).stat().st_dev and bank_required+result_required > min(free_bank,free_result):
            raise OSError('Combined same-filesystem space is insufficient')
    return dict(sum_squared_counts=sum_squared_counts, rows=rows, attention_bytes=payload,
        bank_required_bytes=bank_required, result_required_bytes=result_required,
        declared_bank_bytes=gate['available_bank_bytes'], declared_result_bytes=gate['available_result_bytes'])


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != {'preflight': report['job_id']}:
        raise ValueError('Preflight differs from exact submission journal')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'],
        train_minutes=report['train_minutes'], export_minutes=report['export_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science' or spec['project_dir'] != gate['project_dir']
            or spec['source_commit'] != gate['source_commit'] or spec['train_minutes'] != report['train_minutes']
            or spec['export_minutes'] != report['export_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Science registration differs')
    return gate, report
