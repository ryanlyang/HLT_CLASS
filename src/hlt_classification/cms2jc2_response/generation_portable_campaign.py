"""Separate SPORC export and Tigris replay contracts; original studies untouched."""
import math
from pathlib import Path
import shutil

from . import dev_campaign as dev, generation_benchmark_campaign as gen
from . import generation_portable_data as data
from .contracts import artifact, load_json, validate
from .dev_data import checked_file, file_ref
from .provenance import numerical_environment
from .storage import GIB, TOTAL_CAP, REPORT_CAP, usage

CONTRACT = 'CMS2JC2_RESPONSE_PORT_STAGE/v1'
WORKER = 'sbatch/run_cms2jc2_response_portable_cpu.sh'


def protocol():
    return artifact('PORT_PROTOCOL', jets=10000, gate_jets=64, candidate='JOINT', replica=0,
        cross_site_atol=data.ATOL, cross_site_rtol=data.RTOL, within_site_exact=True,
        worker_choices=[16, 36, 72, 144], chunk=32, repeats=2, block_jets=1000,
        native_threads=1, input='frozen_training_physical_packet', raw_root_benchmark=False,
        screen_resources='ceil_measured_hours_2_to_8_and_ram_32_64_128',
        automatic_followup=False, production_qualified=False)


def site(partition):
    if partition in ('debug', 'tier3'):
        return gen.site(partition)
    if partition != 'tigris':
        raise ValueError('Unknown portable site')
    return dict(partition='tigris', account='reu-aisocial', qos=None, nodes=1, tasks=1, gpus=0,
                conda_prefix='/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris')


def tasks(stage, maximum=36, *, hours=8, memory_gib=128):
    if type(maximum) is not int or maximum not in (36, 72, 144):
        raise ValueError('Register maximum workers as 36, 72 or 144')
    if stage == 'port_export':
        return [dev.task('jp_export', 'jp_export', 16, 64, 8)]
    if stage == 'port_gate':
        return [dev.task('jp_gate', 'jp_gate', 4, 32, 2)]
    if stage != 'port_screen':
        raise PermissionError('No production stage exists')
    if type(hours) is not int or not 2 <= hours <= 8 or memory_gib not in (32, 64, 128):
        raise ValueError('Unregistered portable screen resource envelope')
    rows = []
    for cpus in (n for n in (16, 36, 72, 144) if n <= maximum):
        prior = [t['task_id'] for t in rows if t['params']['repeat'] == 1] if cpus == 144 else []
        for repeat in range(2):
            task = dev.task(f'jp_c{cpus:03d}_r{repeat}', 'jp_run', cpus, memory_gib, hours, prior,
                            workers=cpus, repeat=repeat)
            rows.append(task); prior = [task['task_id']]
    return [*rows, dev.task('jp_report', 'jp_report', 1, 32, 4, [r['task_id'] for r in rows])]


def parents(spec):
    return {'stage': spec['content_hash'], 'protocol': protocol()['content_hash']}


def validate_source_match(current, previous):
    # No portability allowlist for scientific code or the export engine.
    if current['commit'] != previous['commit'] or current['files'] != previous['files']:
        raise ValueError('Portable replay requires the exact exported source commit/files')


def validate_study(study, *, source=True):
    validate(study, 'PORT_STUDY', parents={'source': study['source']['content_hash'],
                                         'protocol': protocol()['content_hash']})
    if (study['protocol'] != protocol() or study['site'] != site(study['site']['partition'])
            or study['particle_roles'] != ['train'] or study['production_qualified'] is not False):
        raise PermissionError('Portable scope differs')
    tasks('port_screen', study['max_workers'])
    if study['mode'] == 'export':
        if study['site']['partition'] not in ('debug', 'tier3') or study['packet'] is not None:
            raise PermissionError('Export requires SPORC')
        parent = load_json(checked_file(study['donor']))
        old = gen.validate_stage(parent, source=False)
        if parent['stage'] != 'generation_gate':
            raise PermissionError('Require real generation gate, not a screen')
        gen.accepted(parent)
        validate_source_match(study['source'], old['source'])
        if study['numerical_environment'] != old['numerical_environment']:
            raise ValueError('Export numerical environment differs from frozen SPORC donor')
        protected = [old['root'], old['data_root']]
    elif study['mode'] == 'replay':
        if study['site']['partition'] != 'tigris' or study['donor'] is not None:
            raise PermissionError('Replay requires Tigris')
        ref = study['packet']
        packet = data.read_packet(checked_file(ref), ref['sha256'])
        validate_source_match(study['source'], packet['source'])
        protected = [str(Path(ref['path']).parent)]
    else:
        raise PermissionError('Unregistered portable mode')
    for target in (*protected, study['project_dir']):
        gen.joint.audit.disjoint(study['root'], target)
    if source and dev.source_snapshot(study['project_dir'], study['source']['commit']) != study['source']:
        raise ValueError('Portable source changed')


def register(spec, study, screen_resources=None):
    validate(spec, 'PORT_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']})
    if (spec['root'] != study['root'] or spec['name'] != spec['stage']+'_r1'
            or spec['tasks'] != tasks(spec['stage'], study['max_workers'], **(screen_resources or {}))
            or spec['scientific_qualification'] is not False
            or (spec['stage'] == 'port_export') != (study['mode'] == 'export')):
        raise ValueError('Portable stage registration differs')


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec['study']))
    validate_study(study, source=source)
    if spec['stage'] == 'port_screen':
        parent = load_json(checked_file(spec['parent_spec']))
        register(parent, study)
        if parent['study'] != spec['study'] or parent['stage'] != 'port_gate' or parent['parent_spec'] is not None:
            raise ValueError('Wrong portable gate')
        register(spec, study, resources(accepted(parent, study)['projection']))
    elif spec['parent_spec'] is not None:
        raise ValueError('Unexpected stage parent')
    else:
        register(spec, study)
    return study


def resources(projection):
    if projection['resource_envelope_ok'] is not True:
        raise PermissionError('Measured resource envelope failed')
    return dict(hours=max(2, math.ceil(projection['seconds']/3600)),
                memory_gib=next(n for n in (32, 64, 128) if projection['memory_bytes'] <= n*GIB))


def projection(row, maximum):
    seconds = row['serial']['processing_seconds']
    rss = row['measurement']['sampled_peak_tree_rss_bytes']
    size = row['serial']['output_bytes']
    if not all(math.isfinite(v) and v > 0 for v in (seconds, rss, size)):
        raise ValueError('Invalid portability resource measurements')
    runs = len(tasks('port_screen', maximum))-1
    wall, memory = 3600+2*seconds*10000/64, 2*(maximum/4)*rss+2*GIB
    disk = math.ceil(2*size/64*10000*runs)+2*GIB
    return dict(seconds=wall, memory_bytes=memory, total_bytes=disk,
        required_free_bytes=2*disk+5*GIB,
        resource_envelope_ok=wall <= 8*3600 and memory <= 128*GIB and disk <= TOTAL_CAP)


def accepted(spec, study):
    row = dev.product(spec, 'jp_gate', 'result')
    validate(row, 'PORT_GATE', parents=parents(spec))
    if (row['compatible'] is not True or row['within_site_exact'] is not True
            or row['jets'] != 64 or row['projection'] != projection(row, study['max_workers'])
            or not row['projection']['resource_envelope_ok']
            or row['environment'] != study['numerical_environment']):
        raise PermissionError('Portable measured replay gate not satisfied')
    from .generation_benchmark_engine import parity
    parity([row['serial'], *row['replay_runs']])
    if len(row['replay_runs']) != 2:
        raise ValueError('Incomplete portability gate')
    packet = data.read_packet(study['packet']['path'], study['packet']['sha256'])
    root = Path(study['packet']['path']).parent
    checks = [data.compare_run(r, study['root'], packet, root, gate=True)
              for r in (row['serial'], *row['replay_runs'])]
    if row['comparisons'] != checks:
        raise ValueError('Portability comparisons changed')
    return row


def publish(study, stage, parent=None):
    envelope = resources(accepted(load_json(checked_file(parent)), study)['projection']) if stage == 'port_screen' else {}
    spec = artifact('PORT_STAGE', parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']},
        study=file_ref(Path(study['root'])/'study_spec.json'), root=study['root'], stage=stage,
        name=stage+'_r1', parent_spec=parent, tasks=tasks(stage, study['max_workers'], **envelope), scientific_qualification=False)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(study['root'], f'stages/{spec["name"]}/stage_spec.json', spec, 'PORT_STAGE')
    dev.write(study['root'], f'stages/{spec["name"]}/command_plan.json', dev.command_plan(spec, study), 'DEV_PLAN')
    return spec


def create(*, project_dir, source_commit, root, parent_spec=None, packet=None,
           packet_sha256=None, partition='tier3', max_workers=36):
    root, project = Path(root).resolve(), Path(project_dir).resolve(strict=True)
    if root.exists():
        raise FileExistsError('Use a fresh portable root')
    for p in root.parents:
        if any((p/n).exists() for n in ('study_spec.json', 'campaign_spec.json')):
            raise PermissionError('Do not nest another campaign')
    if (parent_spec is None) == (packet is None):
        raise ValueError('Choose exactly one export parent or imported packet')
    if packet is not None:
        data.read_packet(packet, packet_sha256 or '')
        if partition != 'tigris':
            raise PermissionError('Packet import is Tigris-only')
    source = dev.source_snapshot(project, source_commit)
    study = artifact('PORT_STUDY', parents={'source': source['content_hash'], 'protocol': protocol()['content_hash']},
        project_dir=str(project), root=str(root), source=source, protocol=protocol(),
        numerical_environment=numerical_environment(), site=site(partition), max_workers=max_workers,
        mode='export' if parent_spec else 'replay', donor=file_ref(parent_spec) if parent_spec else None,
        packet=file_ref(packet) if packet else None, particle_roles=['train'], production_qualified=False)
    validate_study(study)
    root.mkdir(parents=True, exist_ok=False)
    dev.write(root, 'study_spec.json', study, 'PORT_STUDY')
    return publish(study, 'port_export' if parent_spec else 'port_gate')


def advance(parent_spec):
    parent = load_json(parent_spec)
    study = validate_stage(parent)
    if parent['stage'] != 'port_gate':
        raise PermissionError('Only Tigris gate can advance; no production stage')
    p = accepted(parent, study)['projection']
    total, reports = usage(Path(study['root']))
    if (total+p['total_bytes'] > TOTAL_CAP or reports+GIB > REPORT_CAP
            or shutil.disk_usage(study['root']).free < p['required_free_bytes']):
        raise OSError('Insufficient measured portable screen storage; nothing deleted')
    return publish(study, 'port_screen', file_ref(parent_spec))


def command_plan(spec, study):
    root, project, s = dev.stage_dir(spec), Path(study['project_dir']), study['site']
    commands = []
    for task in spec['tasks']:
        argv = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            f'--cpus-per-task={task["cpus"]}', f'--mem={task["memory_gib"]}G', f'--time={task["hours"]:02}:00:00',
            f'--partition={s["partition"]}', f'--account={s["account"]}']
        if s['qos'] is not None:
            argv.append('--qos='+s['qos'])
        argv += [f'--job-name=c2jd_{task["task_id"]}', f'--comment=c2jd:{spec["content_hash"]}:{task["task_id"]}',
            f'--chdir={project}', f'--output={root}/slurm-%j.out', str(project/WORKER),
            str(project), str(root/'stage_spec.json'), task['task_id']]
        commands.append(dict(task_id=task['task_id'], depends_on=task['depends_on'],
                             dependency_mode=task['dependency_mode'], argv=argv))
    cap = (max(144, 124) if study['max_workers'] == 144 else sum(n for n in (16, 36, 72) if n <= study['max_workers']))
    if spec['stage'] != 'port_screen':
        cap = spec['tasks'][0]['cpus']
    return artifact('DEV_PLAN', parents={'stage': spec['content_hash']}, commands=commands, gpus=0,
        cpu_upper_bound=cap, allocated_cpu_hour_upper_bound=sum(t['cpus']*t['hours'] for t in spec['tasks']),
        live_authorization_phrase=dev.PHRASES[spec['stage']])
