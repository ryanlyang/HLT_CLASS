"""One view of original, salvaged, existing replacement and SOFT-v2 results."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from . import core, submission


def location(spec, values, task):
    original, storage, soft, _, r, sr, fr = values
    if task in core.TASKS:
        return Path(spec['root']) / task, soft, fr, 'soft_v2_replacement'
    if task in sr('core').RESTART:
        return Path(storage['root']) / task, original, r, 'existing_storage_replacement'
    kind = 'verified_storage_salvage' if task in core.SALVAGE or task in sr('core').SALVAGE else 'original_completed_task'
    return Path(original['root']) / task, original, r, kind


def completed(spec, values, task, qualification):
    from .runtime import receipt
    original, storage, _, _, r, sr, _ = values
    folder, scientific, rr, kind = location(spec, values, task)
    if kind == 'verified_storage_salvage':
        expected = (qualification['salvaged'][task] if task in core.SALVAGE else storage['failures'][task])['report']
    else:
        if not (folder / 'receipt.json').is_file(): return None
        if kind == 'soft_v2_replacement': receipt(spec, task, scientific, rr)
        elif kind == 'existing_storage_replacement': sr('runtime').receipt(storage, task, original, r)
        else: r('worker').receipt(original, task)
        expected = None
    report, arrays = rr('training').validate_fit(folder / 'fit', scientific, task)
    if expected is not None and report['content_hash'] != expected:
        raise ValueError('Salvaged report changed')
    return report, arrays, load_json(folder / 'fit/initialization.json'), dict(
        kind=kind, folder=str(folder), scientific_spec=scientific['content_hash'], report=report['content_hash'])


def requested(readout, constants):
    if readout not in (*constants.READOUTS, 'all'): raise ValueError('Unknown readout')
    return constants.READOUTS if readout == 'all' else (readout,)


def results(spec, readout='RAW'):
    values = core.check(spec); original, storage, _, _, r, sr, _ = values
    constants = r('contracts'); names = requested(readout, constants)
    qualification = load_json(spec['qualification']['path'])
    status = r('reporting').states(original)
    status.update(sr('core').accounting(sr('submission').accepted_jobs(storage)))
    status.update(sr('core').accounting(submission.accepted_jobs(spec)))
    reports = {}; initials = {}; provenance = {}; reference = None
    for task in original['tasks']:
        value = completed(spec, values, task, qualification)
        if value is None: continue
        report, arrays, initial, origin = value
        if reference is not None and any(not np.array_equal(reference[k], arrays[k]) for k in ('identities', 'labels')):
            raise ValueError('Different validation populations')
        reference = arrays
        reports[task] = report; initials[task] = initial; provenance[task] = origin
    def auc(task, name):
        return reports[task]['selected_readouts'][name]['validation']['macro_ovr_auc'] if task in reports else None
    rows = []
    for task, report in reports.items():
        control = constants.control(task); baseline = 'R24_s' + str(constants.cell(task)['seed_index'])
        pairing = {other: sr('reporting').differences(initials[task], initials[other]) if other in initials else {'missing': other}
                   for other in (control, baseline)}
        for name in constants.READOUTS:
            selected = report['selected_readouts'][name]; value = auc(task, name)
            match = auc(control, name); r24 = auc(baseline, name)
            rows.append(dict(task=task, readout=name, epoch=selected['epoch'], auc=value,
                recovery=recovery_percent(value, original['recovery_anchors']), control=control,
                delta=None if match is None or pairing[control] else value-match,
                delta_r24=None if r24 is None or pairing[baseline] else value-r24,
                delta_same_fit_raw_bn=value-auc(task, 'RAW_BN'), pairing_differences=pairing,
                late=r('reporting').stability(r('reporting').for_readout(report['history'], name), original['recovery_anchors']),
                provenance=provenance[task]))
    print('Combined 42-fit stability results. RAW primary; historical recovery anchors; exploratory validation.')
    print('SOFT uses numerical solver v2. Storage salvages preserve original fits; Slurm states are unchanged.')
    print('Paired deltas are withheld for missing or mismatched initialization/RNG controls.')
    for name in names:
        print('\n' + name)
        print(f'{"task":12} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta ctrl":>12} provenance')
        for row in rows:
            if row['readout'] != name: continue
            delta = '--' if row['delta'] is None else f'{row["delta"]:+.8f}'
            print(f'{row["task"]:12} {row["epoch"]:3} {row["auc"]:12.8f} {row["recovery"]:7.1f} {delta:>12} {row["provenance"]["kind"]}')
    pending = [t for t in original['tasks'] if t not in reports]
    for task in pending:
        # Do not label an unsubmitted replacement with its old failed job state.
        state = status.get(task, ['NOT_SUBMITTED', ''])
        if task in core.TASKS and task not in submission.accepted_jobs(spec): state = ['REPLACEMENT_NOT_SUBMITTED', '']
        print(task, state)
    return core.artifact('RESULTS', parents=dict(spec=spec['content_hash']), primary='RAW',
        rows=rows, summary=r('reporting').summarize(rows), states=status, pending=pending)


def progress(spec, readout='RAW'):
    values = core.check(spec); original, _, _, _, r, _, _ = values
    names = requested(readout, r('contracts')); history = {}
    print('Provisional saved epochs; read-only. RAW primary, historical anchors, no independent test.')
    for task in original['tasks']:
        folder, scientific, rr, _ = location(spec, values, task); rows = []
        for epoch, path in enumerate(sorted((folder / 'fit').glob('epoch-*.json')), 1):
            if epoch > 30 or path.name != f'epoch-{epoch:02}.json': raise ValueError('Missing or extra epochs')
            row = load_json(path)
            rr('training').validate_epoch(scientific, task, row, epoch, scientific['counts'], 4, False)
            rows.append(row)
        if rows:
            rr('identity').validate_initial(scientific, task, load_json(folder / 'fit/initialization.json'), 4)
        history[task] = rows
        for name in names:
            if not rows: continue
            print('\n' + task + ' ' + name)
            for row in rows:
                if name in row['readouts']:
                    value = row['readouts'][name]['macro_ovr_auc']
                    print(f'  epoch {row["epoch"]:2}: AUC={value:.8f} recovery={recovery_percent(value, original["recovery_anchors"]):.1f}%')
    return core.artifact('PROGRESS', parents=dict(spec=spec['content_hash']), epochs=history)
