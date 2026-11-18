"""Reuse genuine GPU qualification with an explicit, separately recorded time cap."""
from pathlib import Path
import re
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from . import core

CAP_ERROR = 'RuntimeError: Measured walltime exceeds 48h cap; preserve diagnostics'


def only_error(log, expected):
    lines = [re.sub(r'^\[rank\d+\]:\s*', '', line).strip() for line in log.read_text(errors='replace').splitlines()
             if re.search(r'\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):', line)]
    lines = [line for line in lines if 'CalledProcessError:' not in line and 'ChildFailedError:' not in line]
    if not lines or any(line != expected for line in lines):
        raise ValueError('Unrecognized failure cause: ' + str(log))


def salvage(original, r, accounting):
    jobs = r('submission').accepted_jobs(original, 'science_pascal')
    if any(jobs.get(t) != j for t, j in core.SALVAGE.items()):
        raise ValueError('Supplemental failure IDs differ')
    states = accounting(core.SALVAGE); root = Path(original['root']); records = {}
    for task, job in core.SALVAGE.items():
        if states.get(task) != ['FAILED', '1:0']:
            raise ValueError('Only terminal storage failures may be salvaged: ' + task)
        folder = root / task; log = root / f'{task}-{job}.out'
        only_error(log, 'OSError: Insufficient remaining stability output allowance plus one GiB reserve')
        if (folder / 'receipt.json').exists(): raise ValueError('Task already published: ' + task)
        fit, _ = r('training').validate_fit(folder / 'fit', original, task)
        startup, _ = r('training').validate_fit(folder / 'startup', original, task, technical=True)
        memory = load_json(folder / 'startup_memory.json')
        r('contracts').validate(memory, 'STARTUP', dict(spec=original['content_hash']))
        if memory['task'] != task or memory['miniature'] != startup['content_hash'] or memory['discarded'] is not True:
            raise ValueError('Invalid salvaged startup')
        claim = root / 'claims' / task / 'intent.json'
        if load_json(claim) != r('contracts').artifact('RUN_CLAIM', parents=dict(spec=original['content_hash']), task=task, job_id=job):
            raise ValueError('Salvaged claim differs')
        records[task] = dict(job=job, report=fit['content_hash'], references=[file_ref(p) for p in
            (log, folder / 'fit/receipt.json', folder / 'startup/receipt.json', folder / 'startup_memory.json', claim)])
    return records


def failed_setup(soft, fr):
    jobs = fr('submission').accepted_jobs(soft, 'bootstrap')
    if jobs != dict(setup_pascal=core.SETUP_JOB, release_pascal=core.RELEASE_JOB):
        raise ValueError('Exact failed SOFT bootstrap required')
    if fr('submission').accounting(core.SETUP_JOB) != ['FAILED', '1:0']:
        raise ValueError('SOFT setup must be terminal FAILED 1:0')
    status = fr('submission').accounting(core.RELEASE_JOB)
    if not status or not status[0].startswith('CANCELLED') or status[1] != '0:0':
        raise ValueError('Old SOFT release must be cancelled')
    root = Path(soft['root'])
    # No old science journal/claim is permitted, even if a partial submission
    # has not yet received an accounting row.
    if (root / 'submission_science_pascal').exists() or any((root / t).exists() or (root / 'claims' / t).exists() for t in core.TASKS):
        raise ValueError('Old SOFT science may already exist; do not duplicate it')
    if (root / 'setup_pascal/receipt.json').exists() or (root / 'setup_pascal/acceptance.json').exists():
        raise ValueError('Expected the exact pre-acceptance failure')
    log = root / f'setup_pascal-{core.SETUP_JOB}.out'; only_error(log, CAP_ERROR)
    claim = root / 'claims/setup_pascal/intent.json'
    if load_json(claim) != fr('contracts').artifact('RUN_CLAIM', parents=dict(spec=soft['content_hash']), task='setup_pascal', job_id=core.SETUP_JOB):
        raise ValueError('SOFT setup claim differs')
    return [file_ref(log), file_ref(claim)]


def soft_qualification(soft, old, fr):
    root = Path(soft['root']) / 'setup_pascal'; resource = fr('resources')
    resource.check_qualification(soft, 'pascal', load_json(root / 'qualification.json'))
    resource.check_replay(soft, old, 'pascal')
    calibration = fr('calibration').load(soft)
    membership = load_json(root / 'bn_membership.json')
    fr('contracts').validate(membership, 'BN_MEMBERSHIP', dict(spec=soft['content_hash']))
    fr('recovery').check_membership(soft, membership)
    diagnostic = load_json(root / 'resource_diagnostic.json')
    if diagnostic['parents'] != dict(spec=soft['content_hash'], soft_calibration=calibration['content_hash'], bn_membership=membership['content_hash']):
        raise ValueError('SOFT qualification parents differ')
    initials = {}
    for task in core.TASKS:
        folder = root / ('mini_' + task)
        report, _ = fr('training').validate_fit(folder, soft, task, technical=True)
        if (report['content_hash'] != diagnostic['miniatures'][task]['content_hash']
            or report['bn_membership'] != membership['content_hash'] or report['world_size'] != 4):
            raise ValueError('SOFT miniature lineage differs')
        initials[task] = load_json(folder / 'initialization.json')
        fr('recovery').check_initial(soft, task, initials[task])
    fr('identity').validate_pairs(soft, initials, complete=True)
    # All original numerical, support, memory, evolved-cost, timing and storage
    # inventory checks remain. Only the execution time ceiling is overridden.
    # This temporary validator input is NEVER published as a scientific spec or
    # original ACCEPTANCE. The override is explicit in our new contract.
    acceptance = resource.accept(dict(soft, fit_max_minutes=core.MAX_MINUTES), diagnostic, 'pascal')
    if not any(n > soft['fit_max_minutes'] for n in acceptance['minutes'].values()):
        raise ValueError('Measurements do not reproduce the time-cap failure')
    refs = [file_ref(p) for p in sorted(root.rglob('*')) if p.is_file()]
    return acceptance, refs


def qualify(values):
    original, storage, soft, old, r, sr, fr = values
    r('worker').completed_setup(original, 'pascal')
    salvaged = salvage(original, r, sr('core').accounting)
    refs = failed_setup(soft, fr)
    accepted, setup_refs = soft_qualification(soft, old, fr)
    print('Authenticated four completed fits; only three SOFT replacements need GPUs.', flush=True)
    for task, minutes in accepted['minutes'].items():
        print(f'{task}: measured request {minutes} minutes ({minutes/60:.2f} h), including original 2x safety margin.', flush=True)
    return core.artifact('QUALIFICATION', parents=dict(original=original['content_hash'], storage=storage['content_hash'], soft=soft['content_hash']),
        salvaged=salvaged, failure_references=refs, setup_references=setup_refs,
        resource_overrides=dict(fit_max_minutes=core.MAX_MINUTES), walltime_policy=core.POLICY,
        minutes=accepted['minutes'], devices=accepted['devices'], environment=accepted['environment'],
        storage=accepted['storage'], diagnostic=accepted['parents']['diagnostic'], technical_weights_discarded=True)


def check_record(value, values):
    original, storage, soft, _, _, _, _ = values
    core.validate(value, 'QUALIFICATION', dict(original=original['content_hash'], storage=storage['content_hash'], soft=soft['content_hash']))
    if (set(value['salvaged']) != set(core.SALVAGE) or set(value['minutes']) != set(core.TASKS)
        or value['resource_overrides'] != dict(fit_max_minutes=core.MAX_MINUTES) or value['walltime_policy'] != core.POLICY
        or value['environment'] != soft['environment'] or value['technical_weights_discarded'] is not True
        or any(type(n) is not int or not 0 < n <= core.MAX_MINUTES for n in value['minutes'].values())):
        raise ValueError('Qualification scope differs')
    for task, record in value['salvaged'].items():
        if record['job'] != core.SALVAGE[task]: raise ValueError('Salvage ID differs')
        for ref in record['references']: validate_file_ref(ref)
    for ref in (*value['failure_references'], *value['setup_references']): validate_file_ref(ref)
