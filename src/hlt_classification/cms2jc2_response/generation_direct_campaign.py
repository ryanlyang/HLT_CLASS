"""Direct ROOT-to-JOINT Tigris engineering study; no SPORC export required."""
from pathlib import Path
import math
import shutil

from . import dev_campaign as dev, generation_benchmark_campaign as gen
from . import generation_benchmark_data as data, generation_portable_campaign as port
from .contracts import artifact, load_json, validate, with_content_hash
from .dev_data import checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES
from .provenance import numerical_environment
from .storage import GIB, TOTAL_CAP, REPORT_CAP, usage

CONTRACT = 'CMS2JC2_RESPONSE_TG_STAGE/v1'
WORKER = port.WORKER
WORKERS = (16, 36, 72)


def protocol():
    return artifact('TG_PROTOCOL', candidate='JOINT', replica=0, jets=10000,
        gate_jets=64, cross_site_jets=64, worker_choices=list(WORKERS), repeats=2,
        chunk=32, block_jets=1000, native_threads=1, raw_root_ingestion=True,
        cross_site_atol=1e-12, cross_site_rtol=1e-10, within_site_exact=True,
        automatic_followup=False, production_qualified=False)


def tasks(stage, *, hours=8, memory_gib=128):
    if stage == 'tigris_direct_gate':
        return [dev.task('jt_gate', 'jt_gate', 4, 32, 4)]
    if stage != 'tigris_direct_screen':
        raise PermissionError('No production or export stage in direct Tigris study')
    if type(hours) is not int or not 2 <= hours <= 8 or memory_gib not in (32, 64, 128):
        raise ValueError('Unregistered direct Tigris resource envelope')
    rows = []
    for workers in WORKERS:
        previous = []
        for repeat in range(2):
            row = dev.task(f'jt_c{workers:03d}_r{repeat}', 'jt_run', workers, memory_gib,
                           hours, previous, workers=workers, repeat=repeat)
            rows.append(row)
            previous = [row['task_id']]
    return [*rows, dev.task('jt_report', 'jt_report', 1, 32, 4, [r['task_id'] for r in rows])]


def source_match(current, donor):
    # The existing execution-only list is not expanded for this route. In
    # particular the ROOT reader, engine, response/maps and key code must match.
    for name, digest in donor['files'].items():
        if name not in EXECUTION_ONLY_FILES and current['files'].get(name) != digest:
            raise ValueError('Frozen direct Tigris scientific source differs: '+name)


def import_evidence(parent_path):
    parent_ref = file_ref(parent_path)
    parent = load_json(checked_file(parent_ref))
    if parent.get('contract') != gen.CONTRACT or parent.get('stage') != 'generation_gate':
        raise PermissionError('Require completed original SPORC generation gate')
    # Full historical fit/map ancestry once at import. Subsequent reuse checks
    # the materialized bundle, immutable gate and their pinned receipts instead
    # of repeatedly reconstructing the old selection on every worker.
    original = gen.validate_stage(parent, source=False)
    gate = gen.accepted(parent)
    receipt = dev.verified_outputs(parent, 'jg_gate')
    bundle = load_json(checked_file(original['bundle']))
    return artifact('TG_IMPORT', parents={'donor': parent['content_hash'],
        'study': original['content_hash'], 'gate': gate['content_hash'],
        'receipt': receipt['content_hash'], 'bundle': bundle['content_hash']},
        donor=parent_ref, original_study=parent['study'], bundle=original['bundle'])


def inputs(study):
    evidence = load_json(checked_file(study['imported']))
    parent = load_json(checked_file(evidence['donor']))
    original = load_json(checked_file(evidence['original_study']))
    if (parent.get('contract') != gen.CONTRACT or parent.get('stage') != 'generation_gate'
            or parent['study'] != evidence['original_study'] or parent['parent_spec'] is not None):
        raise PermissionError('Direct Tigris donor gate differs')
    validate(original, 'GEN_STUDY', parents={'source': original['source']['content_hash'],
        'protocol': gen.protocol()['content_hash'], 'membership': original['membership']['content_hash']})
    gen.registration(parent, original)
    if (original['site'] != gen.site(original['site']['partition'])
            or original['particle_roles'] != ['train'] or original['native_hlt_access'] is not False
            or original['production_qualified'] is not False or evidence['bundle'] != original['bundle']):
        raise PermissionError('Imported generation boundary differs')
    data.validate_data_root(original['data_root'])
    data.validate_membership(original)
    gate = gen.accepted(parent)
    receipt = dev.verified_outputs(parent, 'jg_gate')
    bundle = load_json(checked_file(evidence['bundle']))
    validate(bundle, 'GEN_BUNDLE')
    validate(evidence, 'TG_IMPORT', parents={'donor': parent['content_hash'],
        'study': original['content_hash'], 'gate': gate['content_hash'],
        'receipt': receipt['content_hash'], 'bundle': bundle['content_hash']})
    return original, gate, bundle


def validate_study(study, *, source=True):
    validate(study, 'TG_STUDY', parents={'source': study['source']['content_hash'],
        'protocol': protocol()['content_hash'], 'import': study['import_hash']})
    if (study['protocol'] != protocol() or study['site'] != port.site('tigris')
            or study['particle_roles'] != ['train'] or study['production_qualified'] is not False):
        raise PermissionError('Direct Tigris scope differs')
    imported = load_json(checked_file(study['imported']))
    if imported['content_hash'] != study['import_hash']:
        raise ValueError('Direct Tigris import differs')
    original, _, _ = inputs(study)
    source_match(study['source'], original['source'])
    validate(study['numerical_environment'], 'NUMERICAL_ENVIRONMENT')
    for other in (original['root'], original['data_root'], original['project_dir'], study['project_dir']):
        gen.joint.audit.disjoint(study['root'], other)
    if source and dev.source_snapshot(study['project_dir'], study['source']['commit']) != study['source']:
        raise ValueError('Direct Tigris source changed')


def parents(spec):
    return {'stage': spec['content_hash'], 'protocol': protocol()['content_hash']}


def projection(row):
    reading = row['input_read_seconds']
    if not math.isfinite(reading) or reading < 0:
        raise ValueError('Invalid gate input timing')
    measured = dict(row, serial=dict(row['serial'],
        processing_seconds=row['serial']['processing_seconds']+reading))
    return port.projection(measured, 72)


def storage_check(root, projection):
    total, reports = usage(Path(root))
    if (not projection['resource_envelope_ok'] or total+projection['total_bytes'] > TOTAL_CAP
            or reports+GIB > REPORT_CAP
            or shutil.disk_usage(root).free < projection['required_free_bytes']):
        raise OSError('Direct Tigris measured storage/resource envelope insufficient; nothing deleted')


def accepted(spec, study):
    from .generation_direct_worker import compare_reference
    from .generation_benchmark_engine import parity
    row = dev.product(spec, 'jt_gate', 'result')
    validate(row, 'TG_GATE', parents=parents(spec))
    original, reference, _ = inputs(study)
    runs = [row['serial'], *row['replay_runs']]
    if (row['jets'] != 64 or len(runs) != 3 or row['within_site_exact'] is not True
            or row['compatible'] is not True or row['cross_site_jets'] != 64
            or row['environment'] != study['numerical_environment']
            or row['projection'] != projection(row) or not row['projection']['resource_envelope_ok']):
        raise PermissionError('Direct Tigris replay/resource gate not satisfied')
    parity(runs)
    checks = [compare_reference(run, spec['root'], original, reference) for run in runs]
    if checks != row['comparisons']:
        raise ValueError('Saved direct Tigris comparisons differ')
    return row


def registration(spec, study, resources=None):
    validate(spec, 'TG_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']})
    if (spec['root'] != study['root'] or spec['name'] != spec['stage']+'_r1'
            or spec['tasks'] != tasks(spec['stage'], **(resources or {}))
            or spec['scientific_qualification'] is not False):
        raise ValueError('Direct Tigris stage registration differs')


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec['study']))
    validate_study(study, source=source)
    if spec['stage'] == 'tigris_direct_screen':
        parent = load_json(checked_file(spec['parent_spec']))
        if (parent['stage'] != 'tigris_direct_gate' or parent['study'] != spec['study']
                or parent['parent_spec'] is not None):
            raise PermissionError('Wrong direct Tigris gate')
        registration(parent, study)
        registration(spec, study, port.resources(accepted(parent, study)['projection']))
    else:
        if spec['parent_spec'] is not None:
            raise PermissionError('Unexpected direct Tigris parent')
        registration(spec, study)
    return study


def publish(study, stage, parent=None):
    envelope = port.resources(accepted(load_json(checked_file(parent)), study)['projection']) if parent else {}
    spec = artifact('TG_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']},
        root=study['root'], study=file_ref(Path(study['root'])/'study_spec.json'), stage=stage,
        name=stage+'_r1', parent_spec=parent, tasks=tasks(stage, **envelope), scientific_qualification=False)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(study['root'], f'stages/{spec["name"]}/stage_spec.json', spec, 'TG_STAGE')
    dev.write(study['root'], f'stages/{spec["name"]}/command_plan.json', command_plan(spec, study), 'DEV_PLAN')
    return spec


def create(*, parent_spec, project_dir, source_commit, root):
    root, project = Path(root).resolve(), Path(project_dir).resolve(strict=True)
    if root.exists():
        raise FileExistsError('Use a fresh direct Tigris root')
    if any(any((p/n).exists() for n in ('study_spec.json', 'campaign_spec.json')) for p in root.parents):
        raise PermissionError('Do not nest another study')
    print('CMS2JC2-TG phase=authenticate_original_gate metadata_only=true', flush=True)
    evidence = import_evidence(parent_spec)
    original = load_json(checked_file(evidence['original_study']))
    source = dev.source_snapshot(project, source_commit)
    source_match(source, original['source'])
    for other in (original['root'], original['data_root'], original['project_dir'], project):
        gen.joint.audit.disjoint(root, other)
    env = numerical_environment()
    root.mkdir(parents=True, exist_ok=False)
    imported = dev.write(root, 'import_evidence.json', evidence, 'TG_IMPORT')
    study = artifact('TG_STUDY', parents={'source': source['content_hash'],
        'protocol': protocol()['content_hash'], 'import': evidence['content_hash']},
        root=str(root), project_dir=str(project), source=source, numerical_environment=env,
        imported=file_ref(imported), import_hash=evidence['content_hash'], protocol=protocol(),
        site=port.site('tigris'), particle_roles=['train'], production_qualified=False)
    validate_study(study)
    dev.write(root, 'study_spec.json', study, 'TG_STUDY')
    return publish(study, 'tigris_direct_gate')


def advance(parent_spec):
    parent = load_json(parent_spec)
    study = validate_stage(parent)
    if parent['stage'] != 'tigris_direct_gate':
        raise PermissionError('No production stage or automatic followup')
    storage_check(study['root'], accepted(parent, study)['projection'])
    return publish(study, 'tigris_direct_screen', file_ref(parent_spec))


def command_plan(spec, study):
    # Same absolute CPU-only wrapper and Tigris site semantics, new task/stage
    # contracts. max_workers only selects the plan's registered CPU ceiling.
    plan = port.command_plan(spec, dict(study, max_workers=72))
    return with_content_hash(dict(plan, cpu_upper_bound=124 if spec['stage'] == 'tigris_direct_screen' else 4))
