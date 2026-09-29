"""Independent TRAIN-only engineering campaign; no production/test capability."""
import math
from pathlib import Path
import shutil

from . import dev_campaign as dev, bdz_joint_campaign as joint
from . import generation_benchmark_data as data
from .contracts import artifact, load_json, validate
from .dev_data import file_ref, checked_file
from .dev_restart import EXECUTION_ONLY_FILES
from .provenance import numerical_environment
from .storage import GIB, TOTAL_CAP, REPORT_CAP, usage

CONTRACT = 'CMS2JC2_RESPONSE_GEN_STAGE/v1'
SETTINGS = (
    dict(id='C04_B032', cpus=4, chunk=32, compression='stored'),
    dict(id='C08_B032', cpus=8, chunk=32, compression='stored'),
    dict(id='C16_B032', cpus=16, chunk=32, compression='stored'),
    dict(id='C16_B128', cpus=16, chunk=128, compression='stored'),
    dict(id='C16_B128_Z', cpus=16, chunk=128, compression='deflate'),
)


def protocol():
    return artifact('GEN_PROTOCOL', candidate='JOINT', replica=0, trace=False,
        jets=data.COUNT, gate_jets=64, settings=list(SETTINGS), repeats=2,
        output_block_jets=1000, output_dtype='float64', native_threads=1,
        projection_jets=2_250_000, automatic_followup=False, production_qualified=False)


def tasks(stage):
    if stage == 'generation_gate':
        return [dev.task('jg_gate', 'jg_gate', 4, 32, 4)]
    if stage != 'generation_screen':
        raise ValueError('Unregistered generation stage')
    rows = [dev.task(f"jg_{s['id']}_r{r}", 'jg_run', s['cpus'], 64, 8,
                    setting=s, repeat=r) for s in SETTINGS for r in range(2)]
    return [*rows, dev.task('jg_report', 'jg_report', 1, 32, 4, [t['task_id'] for t in rows])]


def freeze(parent):
    if parent.get('contract') != joint.CONTRACT or parent.get('stage') != 'joint_compare':
        raise PermissionError('Require completed JOINT comparison')
    selected = joint.read(parent)
    if selected['selected'] != 'JOINT' or selected['guard_failures']['JOINT']:
        raise PermissionError('Only the frozen, guard-passing JOINT candidate is registered')
    from .bdz_joint_worker import inputs
    _, model, _, historical = inputs(parent)
    reg = joint.registry(parent)
    return artifact('GEN_BUNDLE', parents={'comparison': parent['content_hash'],
        'selection': selected['content_hash'], 'receipt': dev.verified_outputs(parent, 'bj_select')['content_hash'],
        'model': model['content_hash'], 'historical': historical['content_hash'], 'joint': reg['content_hash']},
        response=model['runtime_response'], historical=historical['mapping'], joint=reg['mapping'])


def site(partition):
    if partition not in ('debug', 'tier3'):
        raise ValueError('Choose debug or tier3 explicitly')
    return dict(partition=partition, account='reu-aisocial', qos='qos_tier3', nodes=1, tasks=1, gpus=0,
                conda_prefix='/home/ryreu/miniconda3/envs/atlas_kd_sporc')


def validate_study(study, *, source=True):
    validate(study, 'GEN_STUDY', parents={'source': study['source']['content_hash'],
        'protocol': protocol()['content_hash'], 'membership': study['membership']['content_hash']})
    if (study['site'] != site(study['site']['partition']) or study['protocol'] != protocol()
            or study['particle_roles'] != ['train'] or study['native_hlt_access'] is not False
            or study['production_qualified'] is not False):
        raise PermissionError('Generation study boundary differs')
    data.validate_data_root(study['data_root'])
    data.validate_membership(study)
    parent = load_json(checked_file(study['donor']))
    previous = load_json(checked_file(parent['study']))
    for k in ('review', 'numerical_environment'):
        if study[k] != previous[k]:
            raise ValueError('Frozen donor interface changed: '+k)
    for name, digest in previous['source']['files'].items():
        if name not in EXECUTION_ONLY_FILES and study['source']['files'].get(name) != digest:
            raise ValueError('Frozen scientific source changed: '+name)
    if load_json(checked_file(study['bundle'])) != freeze(parent):
        raise ValueError('Generation bundle differs from authenticated donor')
    for other in (previous['root'], study['data_root'], study['project_dir']):
        joint.audit.disjoint(study['root'], other)
    if source and dev.source_snapshot(study['project_dir'], study['source']['commit']) != study['source']:
        raise ValueError('Generation execution source changed')


def parents(spec):
    return {'stage': spec['content_hash'], 'protocol': protocol()['content_hash']}


def gate(spec):
    return spec if spec['stage'] == 'generation_gate' else load_json(checked_file(spec['parent_spec']))


def projection(row):
    if not all(math.isfinite(row[k]) and row[k] >= 0 for k in ('input_read_seconds',)):
        raise ValueError('Invalid gate input timing')
    seconds = row['serial']['processing_seconds']+row['input_read_seconds']
    rss = row['measurement']['sampled_peak_tree_rss_bytes']
    size = row['serial']['output_bytes']
    if not all(math.isfinite(x) and x > 0 for x in (seconds, rss, size)):
        raise ValueError('Invalid gate measurement')
    # Serial timing: no promised multiprocessing scaling. Reserve authentication time.
    wall = 3600 + 2*seconds*data.COUNT/64
    memory = 6*rss+2*GIB  # four to sixteen processes, plus safety margin
    disk = math.ceil(2*size/64*data.COUNT*10)+2*GIB
    return dict(seconds=wall, memory_bytes=memory, total_bytes=disk,
        required_free_bytes=2*disk+5*GIB,
        resource_envelope_ok=wall <= 8*3600 and memory <= 64*GIB and disk <= TOTAL_CAP)


def accepted(spec):
    from .generation_benchmark_engine import parity
    g = gate(spec)
    study = load_json(checked_file(g['study']))
    row = dev.product(g, 'jg_gate', 'result')
    validate(row, 'GEN_GATE', parents=parents(g))
    runs = [row['serial'], *row['replay_runs']]
    if (row['jets'] != 64 or row['parity'] is not True or row['projection'] != projection(row)
            or len(runs) != 4 or not parity(runs) or runs[0]['jets'] != 64
            or runs[0]['ordered_identities'] != study['membership']['gate_ordered_identities']
            or not row['projection']['resource_envelope_ok']):
        raise PermissionError('Real generation parity/resource gate not satisfied')
    return row


def registration(spec, study):
    validate(spec, 'GEN_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']})
    if (spec['root'] != study['root'] or spec['name'] != spec['stage']+'_r1'
            or spec['tasks'] != tasks(spec['stage']) or spec['protocol'] != protocol()
            or spec['scientific_qualification'] is not False):
        raise ValueError('Generation stage registration differs')


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec['study']))
    validate_study(study, source=source)
    registration(spec, study)
    if spec['stage'] == 'generation_gate':
        if spec['parent_spec'] is not None:
            raise ValueError('Gate must not inherit another stage')
    else:
        parent = load_json(checked_file(spec['parent_spec']))
        if (parent['stage'] != 'generation_gate' or parent['study'] != spec['study']
                or parent['parent_spec'] is not None):
            raise PermissionError('Wrong generation gate')
        # The same authenticated study has just been validated. Do not repeat
        # expensive donor/environment ancestry work for this identical parent.
        registration(parent, study)
        accepted(spec)
    return study


def publish(study, stage, parent=None):
    spec = artifact('GEN_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']},
        study=file_ref(Path(study['root'])/'study_spec.json'), root=study['root'], name=stage+'_r1',
        stage=stage, parent_spec=parent, tasks=tasks(stage), protocol=protocol(), scientific_qualification=False)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(study['root'], f"stages/{spec['name']}/stage_spec.json", spec, 'GEN_STAGE')
    dev.write(study['root'], f"stages/{spec['name']}/command_plan.json", dev.command_plan(spec, study), 'DEV_PLAN')
    return spec


def create(*, parent_spec, inventory, profile, data_root, project_dir, source_commit, root, partition='tier3'):
    root, project, data_root = Path(root).resolve(), Path(project_dir).resolve(strict=True), Path(data_root).resolve(strict=True)
    if root.exists():
        raise FileExistsError('Use a fresh benchmark root')
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    previous = load_json(checked_file(parent['study']))
    for other in (data_root, project, previous['root']):
        joint.audit.disjoint(root, other)
    if any((p/'study_spec.json').exists() or (p/'campaign_spec.json').exists() for p in root.parents):
        raise PermissionError('Do not nest an existing campaign')
    data.validate_data_root(data_root)
    inv, prof = file_ref(inventory), file_ref(profile)
    membership = data.build(load_json(checked_file(inv)), load_json(checked_file(prof)))
    source = dev.source_snapshot(project, source_commit)
    env = numerical_environment()
    if env != previous['numerical_environment']:
        raise ValueError('Use the frozen donor numerical environment')
    bundle = freeze(parent)
    root.mkdir(parents=True, exist_ok=False)
    bundle_path = dev.write(root, 'frozen_bundle.json', bundle, 'GEN_BUNDLE')
    study = artifact('GEN_STUDY', parents={'source': source['content_hash'], 'protocol': protocol()['content_hash'],
        'membership': membership['content_hash']}, project_dir=str(project), root=str(root), source=source,
        protocol=protocol(), membership=membership, numerical_environment=env, review=previous['review'],
        donor=parent_ref, bundle=file_ref(bundle_path), inventory=inv, profile=prof, data_root=str(data_root),
        site=site(partition), particle_roles=['train'], native_hlt_access=False, production_qualified=False)
    validate_study(study)
    dev.write(root, 'study_spec.json', study, 'GEN_STUDY')
    return publish(study, 'generation_gate')


def advance(parent_spec):
    parent = load_json(parent_spec)
    study = validate_stage(parent)
    if parent['stage'] != 'generation_gate':
        raise PermissionError('No production or test stage exists')
    p = accepted(parent)['projection']
    total, reports = usage(Path(parent['root']))
    if total+p['total_bytes'] > TOTAL_CAP or reports+GIB > REPORT_CAP:
        raise OSError('Benchmark would exceed root storage cap; nothing deleted')
    if shutil.disk_usage(parent['root']).free < p['required_free_bytes']:
        raise OSError('Insufficient disk headroom for measured benchmark')
    return publish(study, 'generation_screen', file_ref(parent_spec))


def read(spec):
    validate_stage(spec, source=False)
    if spec['stage'] == 'generation_gate':
        return accepted(spec)
    from .generation_benchmark_worker import report
    row = dev.product(spec, 'jg_report', 'result')
    validate(row, 'GEN_REPORT', parents=parents(spec))
    if row != report(spec):
        raise ValueError('Generation report disagrees with registered runs')
    return row


def render(spec):
    row = read(spec)
    if spec['stage'] == 'generation_gate':
        return 'Real generation gate passed. Review and separately submit screen. '+str(row['projection'])
    lines = ['Generation-only benchmark; one JOINT realization; TRAIN only.',
        'Setting             CPU  processing jets/s  end-to-end jets/s  MiB/jet']
    for r in row['settings']:
        lines.append(f"{r['id']:<20} {r['cpus']:>3} {r['processing_jets_per_second']:>18.2f} "
                     f"{r['end_to_end_jets_per_second']:>18.2f} {r['bytes_per_jet']/2**20:>8.5f}")
    lines += [f"Fastest mean single-job elapsed (two repeats): {row['fastest_elapsed']}",
              f"Best observed CPU-hour efficiency: {row['best_cpu_efficiency']}",
              'Conditional 2.25M projections (steady-state and conservative 10k-task overhead):']
    for r in row['projections']:
        lines.append(f"{r['available_cpus']} CPUs: {r['setting']}, {r['jobs']} concurrent jobs, "
                     f"{r['processing_hours']:.1f}--{r['end_to_end_hours']:.1f} hours; {r['storage_gib']:.1f} GiB")
    return '\n'.join(lines+['Not a queue forecast, scaling guarantee, or production qualification.'])
