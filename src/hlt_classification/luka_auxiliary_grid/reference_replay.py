"""Read-only replay executed against the original strong-KD source, not a patched worktree.

Only the confirmed K5 post-publication JSON summation-order failure is recoverable.
All original artifact validators still run. No optimizer, export or file write occurs.
"""
from copy import deepcopy
import getpass
import json
from pathlib import Path
import re
import subprocess
import sys

FAILED_COMMIT = '234b029ec14a8d4861134e098cafbf939d325cd8'
FAILED_JOB = '21962581'
ERROR = 'Epoch loss/schedule/population evidence differs'


def ordered_report(report, teachers):
    from hlt_classification.data.cache_contracts import canonical_json_bytes
    value = deepcopy(report)
    for row in value['validation_history']:
        for key in ('logit_losses', 'embedding_losses', 'weighted_logit_losses', 'weighted_embedding_losses'):
            if set(row[key]) != set(teachers):
                raise ValueError('Per-teacher loss inventory differs')
            row[key] = {t: row[key][t] for t in teachers}
    if canonical_json_bytes(value) != canonical_json_bytes(report):
        raise ValueError('Recovery changed numerical evidence or content hash')
    return value


def reordered_epochs(report, teachers):
    value = ordered_report(report, teachers)
    def total(row):
        return row['weighted_ce_loss']+sum(row['weighted_logit_losses'].values())+sum(row['weighted_embedding_losses'].values())
    return [a['pass_number'] for a, b in zip(report['validation_history'], value['validation_history'])
            if a['total_loss'] != total(a) and a['total_loss'] == total(b)]


def accounting(spec, task, job):
    if not re.fullmatch('[1-9][0-9]*', str(job)):
        raise ValueError('Invalid original accepted job ID')
    proc = subprocess.run(['sacct', '-S', '2026-10-24', '-j', job, '-X', '-n', '-P',
        '-o', 'JobIDRaw,JobName%80,User,Account,State%40,ExitCode'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:6] for line in proc.stdout.splitlines() if line.strip()]
    rows = [r for r in rows if r[0] == job]
    if (len(rows) != 1 or len(rows[0]) != 6
            or rows[0][:4] != [job, 'lukaas_'+task, getpass.getuser(), spec['execution_site']['account']]):
        raise ValueError('Original job accounting missing/ambiguous or identity differs: '+task)
    return dict(zip(('job_id', 'name', 'user', 'account', 'state', 'exit_code'), rows[0]))


def failure_log(spec, job):
    from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
    path = Path(spec['root'])/('k5-'+job+'.out')
    ref = file_ref(path)
    text = path.read_text(encoding='utf-8')
    trace = text.rsplit('Traceback (most recent call last):', 1)
    folder = Path(spec['project_dir'])/'src/hlt_classification/luka_auxiliary_logits_strong'
    expected = [(str(folder/'worker.py'), 'validate_task'), (str(folder/'worker.py'), 'validate_fit'),
                (str(folder/'training.py'), 'validate_report')]
    if (len(trace) != 2 or not trace[1].strip().endswith('ValueError: '+ERROR)
            or re.findall(r'File "([^"]+)", line \d+, in (\w+)', trace[1])[-3:] != expected):
        raise ValueError('Failed K5 log is not the registered post-publication validation failure')
    validate_file_ref(ref)
    return ref


def main(path):
    import numpy as np
    from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
    from hlt_classification.cms_proxy_ladder.contracts import file_ref
    from hlt_classification.luka_auxiliary_logits_strong import campaign, worker, submission, banks, training
    from hlt_classification.luka_auxiliary_logits_strong.contracts import TEACHERS
    s = load_json(path); campaign.check(s)
    root = Path(s['root']); retired = worker.retired_evidence(s); jobs = submission.accepted_jobs(s)
    state = accounting(s, 'k5', jobs.get('k5'))
    mode = 'completed'; repair = None; original_validator = training.validate_report
    if (state['state'], state['exit_code']) != ('COMPLETED', '0:0'):
        if (s['source_commit'] != FAILED_COMMIT or jobs.get('k5') != FAILED_JOB
                or (state['state'], state['exit_code']) != ('FAILED', '1:0') or retired is not None):
            raise ValueError('Reference is neither completed nor the registered failed K5 execution')
        report = load_json(root/'k5/fit/training_report.json')
        bundle = load_json(root/'targets/bundle.json')
        try:
            original_validator(report, s, 'k5', bundle['content_hash'])
        except ValueError as exc:
            if str(exc) != ERROR:
                raise
        else:
            raise ValueError('Failed reference does not exhibit the registered validation bug')
        ordered = ordered_report(report, TEACHERS)
        original_validator(ordered, s, 'k5', bundle['content_hash'])
        epochs = reordered_epochs(report, TEACHERS)
        if not epochs:
            raise ValueError('No JSON-order discrepancy explains the failed reference')
        repair = dict(reordered_epochs=epochs, failure_log=failure_log(s, jobs['k5']),
                      rule='exact_original_teacher_order_no_value_or_tolerance_change')
        mode = 'recovered_postfit_json_order'

        def validate_original_order(value, spec, arm, bank_hash):
            if arm != 'k5' or value != report or spec != s or bank_hash != bundle['content_hash']:
                raise ValueError('Recovery adapter used outside the exact saved K5 report')
            return original_validator(ordered_report(value, TEACHERS), spec, arm, bank_hash)
        training.validate_report = validate_original_order
    seen = {}
    try:
        worker.validate_task(s, 'k5', seen, retired=retired)
    finally:
        training.validate_report = original_validator
    states = {'k5': state}
    for task, row in seen.items():
        if task == 'k5':
            continue
        states[task] = accounting(s, task, row['job_id'])
        if (states[task]['state'], states[task]['exit_code']) != ('COMPLETED', '0:0'):
            raise ValueError('Original target ancestor must be COMPLETED/0:0')
    saved = load_npz_arrays(root/'k5/fit/validation_predictions.npz')
    population = load_npz_arrays(root/'targets/references/M0HLT.npz')
    if any(not np.array_equal(saved[k], population[k]) for k in ('identities', 'labels')):
        raise ValueError('K5 reference prediction population differs')
    evidence = dict(mode=mode, accounting=states, repair=repair,
        files={p: file_ref(root/p) for p in ('k5/receipt.json', 'k5/costs.json', 'k5/fit/receipt.json',
            'k5/fit/selected.pt', 'k5/fit/training_report.json', 'k5/fit/validation_predictions.npz',
            'targets/receipt.json', 'targets/bundle.json', 'targets/startup.json')})
    print('AUXGRID_REFERENCE='+json.dumps(dict(root=str(root), project_dir=s['project_dir'],
        c50=s['reuse']['spec'], teachers=s['reuse']['teachers'], anchor=s['anchor'],
        predictions=file_ref(root/'k5/fit/validation_predictions.npz'),
        report=file_ref(root/'k5/fit/training_report.json'), checkpoint=file_ref(root/'k5/fit/selected.pt'),
        recovery=evidence, retired=retired is not None, banks=banks.manifests(s, payloads=retired is None)),
        sort_keys=True, allow_nan=False))
