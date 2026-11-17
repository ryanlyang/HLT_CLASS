"""Combine authenticated old fits, three salvages and seven new execution receipts."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from . import core, submission


def differences(a, b):
    fields = ('seeds', 'dropout_rank_seeds', 'epoch1_order_hash')
    out = {k: [a[k], b[k]] for k in fields if a[k] != b[k]}
    for key in ('native', 'predictor'):
        if a['components'][key] != b['components'][key]: out[key] = [a['components'][key], b['components'][key]]
    return out


def locate(spec, p, task):
    return Path(spec['root'] if task in core.RESTART else p['root']) / task


def results(spec, readout='RAW'):
    from .runtime import receipt
    p, _, r = core.check(spec); constants = r('contracts'); old_reporting = r('reporting')
    if readout not in (*constants.READOUTS, 'all'): raise ValueError('Unknown readout')
    status = old_reporting.states(p); status.update(core.accounting(submission.accepted_jobs(spec)))
    reports = {}; initials = {}; provenance = {}; reference = None
    for task in p['tasks']:
        folder = locate(spec, p, task)
        if task in core.SALVAGE:
            kind = 'verified_original_fit_after_storage_failure'
        elif not (folder / 'receipt.json').is_file(): continue
        elif task in core.RESTART:
            receipt(spec, task, p, r); kind = 'replacement_execution'
        else:
            r('worker').receipt(p, task); kind = 'original_completed_task'
        report, arrays = r('training').validate_fit(folder / 'fit', p, task)
        if task in core.SALVAGE and report['content_hash'] != spec['failures'][task]['report']:
            raise ValueError('Salvaged fit identity changed')
        if reference is not None and any(not np.array_equal(reference[k], arrays[k]) for k in ('identities', 'labels')):
            raise ValueError('Different validation populations')
        reference = arrays
        reports[task] = report; initials[task] = load_json(folder / 'fit/initialization.json')
        provenance[task] = dict(kind=kind, folder=str(folder), report=report['content_hash'])
    def auc(task, name):
        return reports[task]['selected_readouts'][name]['validation']['macro_ovr_auc'] if task in reports else None
    rows = []
    for task, report in reports.items():
        control = constants.control(task); baseline = 'R24_s' + str(constants.cell(task)['seed_index'])
        pairing = {other: differences(initials[task], initials[other]) if other in initials else {'missing': other}
                   for other in (control, baseline)}
        for name in constants.READOUTS:
            selected = report['selected_readouts'][name]; value = auc(task, name)
            match = auc(control, name); r24 = auc(baseline, name)
            rows.append(dict(task=task, readout=name, epoch=selected['epoch'], auc=value,
                recovery=recovery_percent(value, p['recovery_anchors']), control=control,
                delta=None if match is None or pairing[control] else value-match,
                delta_r24=None if r24 is None or pairing[baseline] else value-r24,
                delta_same_fit_raw_bn=value-auc(task, 'RAW_BN'), pairing_differences=pairing,
                late=old_reporting.stability(old_reporting.for_readout(report['history'], name), p['recovery_anchors']),
                provenance=provenance[task]))
    print('Original stability science; storage-only recovery. RAW primary; exploratory validation.')
    print('Three original fits are authenticated salvages; only seven tasks restart. SOFT recovery remains separate.')
    print('Paired deltas require identical native/predictor starts and RNG identities; mismatches retain individual results.')
    for name in constants.READOUTS if readout == 'all' else (readout,):
        print('\n' + name)
        print(f'{"task":12} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta ctrl":>12} status')
        for row in rows:
            if row['readout'] == name:
                delta = '--' if row['delta'] is None else f'{row["delta"]:+.8f}'
                print(f'{row["task"]:12} {row["epoch"]:3} {row["auc"]:12.8f} {row["recovery"]:7.1f} {delta:>12} {row["provenance"]["kind"]}')
    pending = [t for t in p['tasks'] if t not in reports]
    for task in pending: print(task, status.get(task, ['NOT_SUBMITTED', '']))
    return core.artifact('RESULTS', parents=dict(spec=spec['content_hash'], original=p['content_hash']),
                         primary='RAW', rows=rows, summary=old_reporting.summarize(rows), states=status, pending=pending)


def progress(spec, readout='RAW'):
    p, _, r = core.check(spec); constants = r('contracts'); history = {}
    if readout not in (*constants.READOUTS, 'all'): raise ValueError('Unknown readout')
    for task in p['tasks']:
        folder = locate(spec, p, task) / 'fit'; rows = []
        for epoch, path in enumerate(sorted(folder.glob('epoch-*.json')), 1):
            if epoch > 30 or path.name != f'epoch-{epoch:02}.json': raise ValueError('Missing or extra epochs')
            value = load_json(path); r('training').validate_epoch(p, task, value, epoch, p['counts'], 4, False); rows.append(value)
        if rows:
            r('identity').validate_initial(p, task, load_json(folder / 'initialization.json'), 4)
        history[task] = rows
    print('Provisional saved epochs. RAW primary; no inference or new writes.')
    for name in constants.READOUTS if readout == 'all' else (readout,):
        for task, rows in history.items():
            if not rows: continue
            print('\n' + task + ' ' + name)
            for row in rows:
                if name in row['readouts']:
                    auc = row['readouts'][name]['macro_ovr_auc']
                    print(f'  epoch {row["epoch"]:2}: AUC={auc:.8f} recovery={recovery_percent(auc, p["recovery_anchors"]):.1f}%')
    return core.artifact('PROGRESS', parents=dict(spec=spec['content_hash'], original=p['content_hash']), epochs=history)
