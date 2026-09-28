"""Frozen candidate, separately claimed full confirmation, no next tuning round."""
import math
from pathlib import Path
import shutil

from . import dev_campaign as dev, bdz_joint_campaign as joint, frozen_joint_data as data
from . import frozen_joint_metrics as metrics
from .contracts import artifact, load_json, validate
from .dev_data import checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES
from .storage import GIB, REPORT_CAP, TOTAL_CAP, usage

KIND = 'FROZEN_STAGE'
CONTRACT = 'CMS2JC2_RESPONSE_FROZEN_STAGE/v1'
STAGES = ('frozen_gate', 'frozen_confirm')
LANES = 8


def protocol():
    return artifact('FROZEN_PROTOCOL', selected='JOINT', reference='B_DZ', replicas=[0, 1, 2],
        minimum_jets=data.MIN_JETS, maximum_shard_jets=data.SHARD_JETS,
        use_all_confirmation=True, lanes=LANES, checks=metrics.rules(),
        automatic_followup=False, production_qualified=False, transfer_authorized=False)


def tasks(stage, membership):
    if stage == 'frozen_gate':
        return [dev.task('jf_acceptance', 'jf_acceptance', 8, 32, 2)]
    if stage != 'frozen_confirm':
        raise ValueError('Unknown frozen confirmation stage')
    names = [f'jf_eval_{i:04d}' for i in range(len(membership['shards']))]
    rows = [dev.task(n, 'jf_evaluate', 16, 64, 4, [names[i-LANES]] if i >= LANES else [], shard=i)
            for i, n in enumerate(names)]
    # Depend on every shard, not merely successful lane tails.
    return [*rows, dev.task('jf_report', 'jf_report', 1, 32, 4, names)]


def gate(spec):
    return spec if spec['stage'] == 'frozen_gate' else load_json(checked_file(spec['parent_spec']))


def donor(spec):
    return load_json(checked_file(gate(spec)['parent_spec']))


def reuse(parent, study):
    if parent.get('contract') != joint.CONTRACT or parent.get('stage') != 'joint_compare':
        raise PermissionError('Require completed JOINT development comparison')
    row = joint.read(parent)
    if row['selected'] != 'JOINT' or row['guard_failures']['JOINT']:
        raise PermissionError('This registered confirmation freezes JOINT only')
    previous = load_json(checked_file(parent['study']))
    for key in ('imported', 'review', 'numerical_environment'):
        if study[key] != previous[key]:
            raise ValueError('Frozen donor interface changed: '+key)
    for name, digest in previous['source']['files'].items():
        if name not in EXECUTION_ONLY_FILES and study['source']['files'].get(name) != digest:
            raise ValueError('Frozen scientific source changed: '+name)
    joint.audit.disjoint(study['root'], previous['root'])
    from .bdz_joint_worker import inputs
    _, model, ranges, historical = inputs(parent)
    reg = joint.registry(parent)
    return artifact('FROZEN_REUSE', parents={'study': study['content_hash'], 'comparison': parent['content_hash'],
        'selection': row['content_hash'], 'receipt': dev.verified_outputs(parent, 'bj_select')['content_hash'],
        'registry': reg['content_hash'], 'model': model['content_hash'], 'ranges': ranges['content_hash'],
        'historical_registry': historical['content_hash']}, selected='JOINT', reference='B_DZ',
        old_artifacts_modified=False, refit=False)


def projection(measurement, result_bytes, jets, members):
    if type(jets) is not int or jets <= 0 or type(result_bytes) is not int or result_bytes <= 0:
        raise ValueError('Invalid acceptance counts/size')
    wall, rss = (measurement[k] for k in ('wall_seconds', 'sampled_peak_tree_rss_bytes'))
    if not all(math.isfinite(x) and x > 0 for x in (wall, rss)):
        raise ValueError('Invalid measured acceptance resources')
    # No assumed 8->16 CPU scaling. Include an hour for source/authentication.
    seconds = 3600+2*wall*max(r['jets'] for r in members['shards'])/jets
    memory = 3*rss+2*1024**3
    # Histograms/top examples are bounded per file, not per particle. Each shard
    # contains one file; reserve two additional per-file copies for the report.
    reports = 4*result_bytes*(len(members['shards'])+2*len(members['files']))
    disk = reports+2*GIB
    return dict(projected_seconds=seconds, projected_bytes=memory,
        projected_report_bytes=reports, projected_total_bytes=disk,
        required_free_bytes=math.ceil(2*disk+5*GIB),
        resource_envelope_ok=(seconds <= 4*3600 and memory <= 64*GIB
                              and reports <= REPORT_CAP and disk <= TOTAL_CAP))


def accepted(spec):
    g = gate(spec)
    row = dev.product(g, 'jf_acceptance', 'result')
    validate(row, 'FROZEN_ACCEPTANCE', parents={'stage': g['content_hash'], 'reuse': g['reuse']['content_hash'],
                                               'protocol': protocol()['content_hash']})
    expected = projection(row['measurement'], row['result_bytes'], row['jets'], spec['membership'])
    if (row['projection'] != expected or row['serial_process_parity'] is not True
            or row['frozen_worker_replay'] is not True or row['confirmation_accessed'] is not False
            or row['jets'] != min(64, joint.COUNTS['residual']) or not expected['resource_envelope_ok']):
        raise PermissionError('Frozen CPU acceptance/resource envelope not satisfied')
    return row


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec['study']))
    dev.validate_study(study, source=source)
    validate(spec, KIND, parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']})
    stage = spec['stage']
    if (stage not in STAGES or spec['name'] != stage+'_r1' or spec['root'] != study['root']
            or spec['protocol'] != protocol() or spec['tasks'] != tasks(stage, spec['membership'])
            or spec['b_threads'] != 1 or spec['untouched_asserted'] is not True
            or study['site']['partition'] not in ('debug', 'tier3')
            or spec['scientific_qualification'] is not False or spec['resources_are_development_envelopes'] is not True):
        raise ValueError('Frozen stage registration differs')
    data.validate_membership(spec['membership'], study)
    parent = load_json(checked_file(spec['parent_spec']))
    if stage == 'frozen_gate':
        if spec['reuse'] != reuse(parent, study):
            raise ValueError('Frozen reuse changed')
    else:
        validate_stage(parent, source=source)
        if (parent['stage'] != 'frozen_gate' or parent['study'] != spec['study']
                or spec['reuse'] != parent['reuse'] or spec['membership'] != parent['membership']):
            raise ValueError('Frozen confirmation ancestry differs')
        accepted(spec)
    if spec['policy'] != parent['policy']:
        raise ValueError('Association policy changed')
    return study


def publish(study, parent_ref, stage, membership, evidence):
    parent = load_json(checked_file(parent_ref))
    spec = artifact(KIND, parents={'study': study['content_hash'], 'protocol': protocol()['content_hash']},
        study=file_ref(Path(study['root'])/'study_spec.json'), root=study['root'], name=stage+'_r1',
        stage=stage, parent_spec=parent_ref, policy=parent['policy'], b_threads=1, protocol=protocol(),
        membership=membership, reuse=evidence, tasks=tasks(stage, membership), untouched_asserted=True,
        scientific_qualification=False, resources_are_development_envelopes=True)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(spec['root'], f"stages/{spec['name']}/stage_spec.json", spec, KIND)
    dev.write(spec['root'], f"stages/{spec['name']}/command_plan.json", dev.command_plan(spec, study), 'DEV_PLAN')
    if stage == 'frozen_confirm':
        dev.write(spec['root'], f"stages/{spec['name']}/confirmation_access.json", data.access_value(spec), 'FROZEN_ACCESS')
    return spec


def create(*, parent_spec, project_dir, source_commit, root, partition, assert_untouched=False):
    if partition not in ('debug', 'tier3') or assert_untouched is not True:
        raise PermissionError('Choose partition and explicitly assert unused response_confirm population')
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    joint.read(parent)
    previous = load_json(checked_file(parent['study']))
    # Capacity check is metadata-only and precedes creation of a new root.
    membership = data.build(previous)
    joint.audit.disjoint(root, previous['root'])
    if any((p/'study_spec.json').exists() or (p/'campaign_spec.json').exists() for p in Path(root).resolve().parents):
        raise PermissionError('Do not nest in an existing campaign')
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(previous['imported']['preparation_spec']), partition=partition)
    return publish(study, ref, 'frozen_gate', membership, reuse(parent, study))


def advance(parent_spec):
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    study = validate_stage(parent)
    if parent['stage'] != 'frozen_gate':
        raise PermissionError('Stop after confirmation; no tuning or transfer stage')
    projected = accepted(parent)['projection']
    total, reports = usage(Path(parent['root']))
    if (total+projected['projected_total_bytes'] > TOTAL_CAP
            or reports+projected['projected_report_bytes'] > REPORT_CAP):
        raise OSError('Projected confirmation exceeds existing campaign storage caps')
    required = projected['required_free_bytes']
    if shutil.disk_usage(parent['root']).free < required:
        raise OSError(f'Need {required/1024**3:.1f} GiB free for confirmation; no deletion performed')
    return publish(study, ref, 'frozen_confirm', parent['membership'], parent['reuse'])


def read(spec):
    validate_stage(spec, source=False)
    if spec['stage'] == 'frozen_gate':
        return accepted(spec)
    row = dev.product(spec, 'jf_report', 'result')
    validate(row, 'FROZEN_REPORT', parents=parents(spec))
    hashes = [dev.product(spec, t['task_id'], 'result')['content_hash'] for t in spec['tasks'][:-1]]
    if (row['shard_hashes'] != hashes or row['jets'] != spec['membership']['jets']
            or row['decision'] != metrics.assess(row['by_file'], row['audit'])
            or row['selected'] != 'JOINT' or row['confirmation_accessed'] is not True
            or any(row[k] is not False for k in ('production_qualified', 'transfer_authorized', 'reselection'))):
        raise ValueError('Frozen report differs')
    return row


def parents(spec):
    return {'stage': spec['content_hash'], 'protocol': protocol()['content_hash'],
            'reuse': spec['reuse']['content_hash'], 'membership': spec['membership']['content_hash']}


def render(spec, *, statistics=False):
    row = read(spec)
    if spec['stage'] == 'frozen_gate':
        return 'CPU acceptance passed. Separately create/review/submit confirmation. '+str(row['projection'])
    decision = row['decision']
    lines = [f"Frozen JOINT confirmation: {decision['status'].upper()}",
        f"Jets: {row['jets']}; files: {decision['source_files']}; replicas: 3 (dependent)",
        'No reselection, refit, JetClass2 transfer or final-test access.',
        'Collection checks only; not detector-response production qualification.',
        f"{'check':<66} {'value':>10} {'95% interval':>24} status"]
    if not decision['independent_groups_sufficient']:
        lines.insert(2, 'Too few source files for acceptance-level file-bootstrap evidence; jet/replica counts do not fix this.')
    for name, c in decision['checks'].items():
        if statistics or c['status'] != 'supported':
            lines.append(f"{name:<66} {str(c['value']):>10} {str(c['interval_95']):>24} {c['status']}")
    if statistics:
        for name in metrics.CANDIDATES:
            lines += ['', name, f"{'observable':<32} {'CMS mean +/- SD':>25} {'proxy mean +/- SD':>25} {'TV':>9}"]
            for field, c in row['charts'][name].items():
                def fmt(m):
                    return 'unavailable' if m['mean'] is None else f"{m['mean']:.6g} +/- {m['sd']:.6g}"
                lines.append(f"{field:<32} {fmt(c['real']):>25} {fmt(c['proxy']):>25} {str(c['tv']):>9}")
        lines += ['SD is width, not mean uncertainty. Proxy moments pool dependent replica exposures.',
                  f"Generator flags (jet-replica occurrences / {3*row['jets']}): {row['flags']}"]
    return '\n'.join(lines)
