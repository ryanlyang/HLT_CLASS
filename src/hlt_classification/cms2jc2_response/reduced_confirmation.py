"""Explicit time-budget truncation; never masquerades as full confirmation."""
from collections import Counter
from pathlib import Path
import time

from . import dev_campaign as dev, dev_submission as submission
from . import frozen_joint_campaign as full, frozen_joint_metrics as metrics
from . import bdz_worker as merge_worker, bdz_audit_metrics as audit
from .contracts import artifact, load_json, validate, safe_relative
from .dev_data import checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES

KIND = 'FROZEN_REDUCED_STAGE'
CONTRACT = 'CMS2JC2_RESPONSE_FROZEN_REDUCED_STAGE/v1'
NAME = 'frozen_reduced_r1'
KEEP = tuple(range(36))
OMIT = tuple(range(36, 57))
TARGETS = tuple(f'jf_eval_{i:04d}' for i in OMIT)+('jf_report',)
RETIRE_PHRASE = 'AUTHORIZE CMS2JC2 STOP REMAINING 21 AND ORIGINAL REPORT'
FULL_STATUS = 'inconclusive_incomplete_population'


def tasks():
    return [dev.task('jr_report', 'jr_report', 1, 32, 4)]


def subject(spec):
    return load_json(checked_file(spec['subject_spec']))


def donor(spec):
    return full.donor(subject(spec))


def coverage(original):
    membership = original['membership']
    shards = membership['shards']
    if len(shards) != 57 or [s['index'] for s in shards] != list(range(57)):
        raise PermissionError('This amendment is exactly the first 36 of 57 shards')
    files = []
    for f in membership['files']:
        chosen = [s for s in shards[:36] if s['sha256'] == f['sha256']]
        files.append(dict(path=f['path'], sha256=f['sha256'], source=f['source'],
            planned_jets=f['jets'], included_jets=sum(s['jets'] for s in chosen)))
    sources = sorted({f['source'] for f in files})
    per_source = {s: dict(planned_jets=sum(f['planned_jets'] for f in files if f['source'] == s),
        included_jets=sum(f['included_jets'] for f in files if f['source'] == s)) for s in sources}
    if (sum(f['planned_jets'] for f in files) != membership['jets']
            or sum(f['included_jets'] for f in files) != sum(s['jets'] for s in shards[:36])):
        raise ValueError('Confirmation membership totals differ')
    return dict(included_indices=list(KEEP), excluded_indices=list(OMIT),
        included_jets=sum(s['jets'] for s in shards[:36]), planned_jets=membership['jets'],
        files=files, sources=per_source,
        missing_sources=[s for s in sources if not per_source[s]['included_jets']],
        included_source_files=sum(f['included_jets'] > 0 for f in files),
        minimum_250k_requirement_waived=True, all_registered_jets_included=False,
        selection='fixed_file_order_prefix_0000_0035_at_user_time_budget_stop',
        representative_random_sample=False, omitted_particles_may_already_have_been_accessed=True)


def jobs(original, study):
    plan = dev.command_plan(original, study)
    value = load_json(dev.stage_dir(original)/'submission_ledger.json')
    validate(value, 'DEV_LEDGER', parents={'stage': original['content_hash'], 'plan': plan['content_hash']})
    expected = {f'jf_eval_{i:04d}' for i in range(57)} | {'jf_report'}
    if (value['dry_run'] is not False or set(value['jobs']) != expected
            or value['jobs'] != submission.submitted_jobs(original, plan)):
        raise PermissionError('Original submission ledger/journals differ')
    return value


def check_shard(original, index, row):
    expected = original['membership']['shards'][index]
    validate(row, 'FROZEN_SHARD', parents=full.parents(original))
    if (row['shard'] != index or row['jets'] != expected['jets']
            or row['ordered_identities'] != expected['ordered_identities']
            or row['source_groups'] != [expected['sha256']]
            or sorted(row['by_file']) != row['source_groups']
            or row['selected'] != 'JOINT' or row['reference'] != 'B_DZ'
            or row['replicas'] != [0, 1, 2] or row['confirmation_accessed'] is not True):
        raise ValueError('Reduced confirmation shard identity/lineage differs')
    return row


def freeze_outputs(original):
    refs = []
    for i in KEEP:
        task = f'jf_eval_{i:04d}'
        receipt = dev.verified_outputs(original, task)
        path = safe_relative(Path(original['root']), receipt['outputs']['result']['relative'])
        row = check_shard(original, i, load_json(path))
        refs.append(dict(index=i, result=file_ref(path), content_hash=row['content_hash'],
            receipt=file_ref(dev.stage_dir(original)/'receipts'/(task+'.json'))))
    return refs


def validate_refs(spec, original):
    if [r['index'] for r in spec['included_outputs']] != list(KEEP):
        raise ValueError('Included indices changed')
    for r in spec['included_outputs']:
        receipt_path = checked_file(r['receipt'])
        expected_path = dev.stage_dir(original)/'receipts'/f"jf_eval_{r['index']:04d}.json"
        if receipt_path.resolve() != expected_path.resolve():
            raise ValueError('Included receipt came from a different stage')
        receipt = load_json(receipt_path)
        validate(receipt, 'DEV_OUTPUTS', parents={'stage': original['content_hash']})
        record = receipt['outputs']['result']
        if (receipt['owner'] != f"jf_eval_{r['index']:04d}"
                or Path(r['result']['path']).resolve() != safe_relative(Path(original['root']), record['relative'])
                or r['result']['sha256'] != record['sha256']):
            raise ValueError('Frozen result differs from original receipt')


def validate_stage(spec, *, source=True):
    original = subject(spec)
    old_study = full.validate_stage(original, source=False)
    study = load_json(checked_file(spec['study']))
    dev.validate_study(study, source=source)
    ledger = load_json(checked_file(spec['subject_ledger']))
    validate(spec, KIND, parents={'study': study['content_hash'], 'subject': original['content_hash'],
        'ledger': ledger['content_hash'], 'protocol': full.protocol()['content_hash']})
    if (original['stage'] != 'frozen_confirm' or spec['stage'] != 'frozen_reduced'
            or spec['name'] != NAME or spec['root'] != study['root'] or spec['tasks'] != tasks()
            or spec['parent_spec'] != spec['subject_spec'] or spec['coverage'] != coverage(original)
            or spec['subject_jobs'] != ledger['jobs'] or spec['protocol'] != full.protocol()
            or spec['b_threads'] != 1 or spec['policy'] != original['policy']
            or spec['scientific_qualification'] is not False
            or spec['resources_are_development_envelopes'] is not True
            or spec['reason'] != 'user_time_budget_stop_no_metric_selection'
            or study['site']['partition'] not in ('debug', 'tier3')
            or ledger != jobs(original, old_study)):
        raise ValueError('Reduced confirmation registration differs')
    for key in ('imported', 'review', 'numerical_environment'):
        if study[key] != old_study[key]:
            raise ValueError('Frozen confirmation interface changed: '+key)
    for path, digest in old_study['source']['files'].items():
        if path not in EXECUTION_ONLY_FILES and study['source']['files'].get(path) != digest:
            raise ValueError('Frozen confirmation scientific source changed: '+path)
    for other in (original['root'], old_study['project_dir'], old_study['imported']['cms_root'], study['project_dir']):
        full.joint.audit.disjoint(spec['root'], other)
    validate_refs(spec, original)
    return study


def states(spec):
    status = submission.states(spec['subject_jobs'])
    for i in KEEP:
        if status.get(spec['subject_jobs'][f'jf_eval_{i:04d}']) != ('COMPLETED', '0:0'):
            raise PermissionError(f'Included shard {i} is not COMPLETED/0:0')
    rows = {t: dict(job=spec['subject_jobs'][t], state=status.get(spec['subject_jobs'][t], ('UNKNOWN', None))[0],
        exit_code=status.get(spec['subject_jobs'][t], (None, None))[1]) for t in TARGETS}
    if any(r['state'] not in submission.TERMINAL | {'PENDING', 'RUNNING', 'COMPLETING'} for r in rows.values()):
        raise PermissionError('Unknown/unsupported original job state; stop retirement')
    return rows


def create(*, subject_spec, project_dir, source_commit, root, partition='debug'):
    original_ref = file_ref(subject_spec)
    original = load_json(checked_file(original_ref))
    if original.get('stage') != 'frozen_confirm' or partition not in ('debug', 'tier3'):
        raise PermissionError('Require original frozen confirmation and explicit CPU partition')
    old_study = full.validate_stage(original, source=False)
    evidence = jobs(original, old_study)
    counts = coverage(original)
    states({'subject_jobs': evidence['jobs']})
    outputs = freeze_outputs(original)
    full.joint.audit.disjoint(root, original['root'])
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(old_study['imported']['preparation_spec']), partition=partition)
    spec = artifact(KIND, parents={'study': study['content_hash'], 'subject': original['content_hash'],
        'ledger': evidence['content_hash'], 'protocol': full.protocol()['content_hash']},
        root=study['root'], study=file_ref(Path(study['root'])/'study_spec.json'),
        subject_spec=original_ref, parent_spec=original_ref,
        subject_ledger=file_ref(dev.stage_dir(original)/'submission_ledger.json'),
        subject_jobs=evidence['jobs'], included_outputs=outputs, coverage=counts,
        stage='frozen_reduced', name=NAME, tasks=tasks(), protocol=full.protocol(),
        policy=original['policy'], b_threads=1, reason='user_time_budget_stop_no_metric_selection',
        scientific_qualification=False, resources_are_development_envelopes=True)
    validate_stage(spec)
    dev.write(root, f'stages/{NAME}/stage_spec.json', spec, KIND)
    dev.write(root, f'stages/{NAME}/command_plan.json', dev.command_plan(spec, study), 'DEV_PLAN')
    return spec


def verify_retirement(spec, *, live=False):
    plan = load_json(dev.stage_dir(spec)/'command_plan.json')
    if plan != dev.command_plan(spec, load_json(checked_file(spec['study']))):
        raise ValueError('Retirement command plan differs')
    value = load_json(dev.stage_dir(spec)/'retirement.json')
    validate(value, 'FROZEN_REDUCED_RETIREMENT', parents={'stage': spec['content_hash'], 'plan': plan['content_hash']})
    if (set(value['tasks']) != set(TARGETS) or value['preserved_indices'] != list(KEEP)
            or any(r['job'] != spec['subject_jobs'][t] or r['state'] not in submission.TERMINAL
                   for t, r in value['tasks'].items())):
        raise ValueError('Reduced retirement evidence differs')
    if live and any(r['state'] not in submission.TERMINAL for r in states(spec).values()):
        raise PermissionError('Original excluded/report jobs remain active')
    return value


def retire(spec, *, execute=False, phrase=None, plan_hash=None):
    study = validate_stage(spec)
    plan = dev.command_plan(spec, study)
    if plan != load_json(dev.stage_dir(spec)/'command_plan.json'):
        raise ValueError('Saved reduced plan changed')
    rows = states(spec)
    if not execute:
        return dict(read_only=True, tasks=rows, preserved_indices=list(KEEP),
            plan_hash=plan['content_hash'], required_phrase=RETIRE_PHRASE)
    if phrase != RETIRE_PHRASE or plan_hash != plan['content_hash']:
        raise PermissionError('Explicit exact-plan retirement authorization required')
    if (dev.stage_dir(spec)/'retirement.json').exists():
        return verify_retirement(spec, live=True)
    original = subject(spec)
    old_study = load_json(checked_file(original['study']))
    # Check EVERY target before making the first scheduler mutation.
    cancel = []
    for task, row in rows.items():
        if row['state'] in ('PENDING', 'RUNNING'):
            fields = submission.scheduler_identity(original, old_study, task, row['job'], pending=True)
            if fields.get('JobName') != 'c2jd_'+task or fields.get('JobState') not in ('PENDING', 'RUNNING'):
                raise PermissionError('Target state/identity changed; recheck before retirement')
            cancel.append(row['job'])
    if cancel:
        result = submission.scheduler(['scancel', '--ctld', '--partition='+old_study['site']['partition'],
            '--account='+old_study['site']['account'], *cancel])
        if result.returncode:
            raise RuntimeError('Cancellation incomplete; recheck the same IDs: '+result.stderr)
    for _ in range(20):
        rows = states(spec)
        if all(r['state'] in submission.TERMINAL for r in rows.values()):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Cancellation still settling; explicitly repeat retire once accounting is terminal')
    value = artifact('FROZEN_REDUCED_RETIREMENT',
        parents={'stage': spec['content_hash'], 'plan': plan['content_hash']},
        tasks=rows, preserved_indices=list(KEEP), cancellation_policy='exact_pending_and_running_targets_only')
    dev.write(spec['root'], f'stages/{NAME}/retirement.json', value, 'FROZEN_REDUCED_RETIREMENT')
    return value


def aggregate(spec):
    original = subject(spec)
    total = dict(merge_worker.empty(), audit={})
    hashes = []
    for r in spec['included_outputs']:
        row = check_shard(original, r['index'], load_json(checked_file(r['result'])))
        if row['content_hash'] != r['content_hash']:
            raise ValueError('Frozen included shard content hash changed')
        merge_worker.merge_result(total, row)
        audit.merge(total['audit'], row['audit'])
        hashes.append(row['content_hash'])
        print(f"CMS2JC2-REDUCED phase=report shards={len(hashes)}/36", flush=True)
    if total['jets'] != spec['coverage']['included_jets']:
        raise ValueError('Reduced jet count differs')
    expected_files = {f['sha256'] for f in spec['coverage']['files'] if f['included_jets']}
    if set(total['by_file']) != expected_files or set(total['audit']) != expected_files:
        raise ValueError('Reduced per-file identity differs')
    for f in spec['coverage']['files']:
        if not f['included_jets']:
            continue
        for name in metrics.CANDIDATES:
            for side in metrics.SIDES:
                if total['by_file'][f['sha256']][name]['cells'][side+'/all']['jets'] != f['included_jets']:
                    raise ValueError('Reduced per-file coverage differs')
    return total, hashes


def report(spec):
    verify_retirement(spec, live=True)
    total, hashes = aggregate(spec)
    histograms = metrics.summaries(total['by_file'])
    return artifact('FROZEN_REDUCED_REPORT', parents={'stage': spec['content_hash'],
        'subject': subject(spec)['content_hash'], 'protocol': full.protocol()['content_hash']},
        **total, shard_hashes=hashes, coverage=spec['coverage'],
        decision=metrics.assess(total['by_file'], total['audit']), decision_scope='included_36_shards_only',
        full_population_status=FULL_STATUS, all_registered_jets_included=False,
        charts={n: metrics.chart(p) for n, p in histograms.items()},
        diagnostics=audit.diagnostics(total['audit']), selected='JOINT', reference='B_DZ',
        confirmation_accessed=True, raw_particles_read=False, production_qualified=False,
        transfer_authorized=False, reselection=False, replicas_are_independent=False,
        class_labels_accessed=False,
        production_six_block_qualification_performed=False, original_association_requirement_waived=False)


def read(spec):
    validate_stage(spec, source=False)
    verify_retirement(spec)
    row = dev.product(spec, 'jr_report', 'result')
    validate(row, 'FROZEN_REDUCED_REPORT', parents={'stage': spec['content_hash'],
        'subject': subject(spec)['content_hash'], 'protocol': full.protocol()['content_hash']})
    if (row['coverage'] != spec['coverage'] or row['jets'] != spec['coverage']['included_jets']
            or row['shard_hashes'] != [r['content_hash'] for r in spec['included_outputs']]
            or row['decision_scope'] != 'included_36_shards_only' or row['full_population_status'] != FULL_STATUS
            or row['selected'] != 'JOINT' or row['reference'] != 'B_DZ'
            or row['confirmation_accessed'] is not True
            or any(row[k] is not False for k in ('all_registered_jets_included', 'raw_particles_read',
                'production_qualified', 'transfer_authorized', 'reselection', 'replicas_are_independent',
                'class_labels_accessed',
                'production_six_block_qualification_performed', 'original_association_requirement_waived'))
            or row['decision'] != metrics.assess(row['by_file'], row['audit'])):
        raise ValueError('Reduced confirmation report scope differs')
    original = subject(spec)
    for r in spec['included_outputs']:
        saved = check_shard(original, r['index'], load_json(checked_file(r['result'])))
        if saved['content_hash'] != r['content_hash']:
            raise ValueError('Reduced confirmation source content hash differs')
    return row


def render(spec):
    row = read(spec)
    c = row['coverage']
    statuses = Counter(v['status'] for v in row['decision']['checks'].values())
    lines = [f"Reduced frozen JOINT confirmation: {row['decision']['status']} (subset only)",
        f"Shards: 36/57; jets: {row['jets']}/{c['planned_jets']}; source files: {c['included_source_files']}",
        'Full-population status: '+row['full_population_status'], 'Check outcomes: '+str(dict(statuses)),
        'Per-source coverage:']
    lines += [f"  {s}: {v['included_jets']}/{v['planned_jets']} jets" for s, v in c['sources'].items()]
    lines += ['Subset checks that are rejected or inconclusive:']
    lines += [f"  {name}: {check['status']}; value={check['value']}; 95% interval={check['interval_95']}"
        for name, check in row['decision']['checks'].items() if check['status'] != 'supported']
    lines += ['Fixed file-order prefix, NOT a random representative sample.',
        'Frozen mapping unchanged. No new particles, refit, or final-test access. Not production qualified.']
    return '\n'.join(lines)
