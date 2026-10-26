"""Explicit, single-use successor to the pre-work runtime-path failure."""
from pathlib import Path
import getpass
import json
import os
import subprocess
import sys
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json
from .contracts import artifact, validate
from . import finalize_source as source

CONTRACT = 'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_SPEC/v1'
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_V1'
FAILED_COMMIT = '420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6'
FAILED_JOB = '7963448'
REPLAY = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullstat_splithead import finalize,submission
s=load_json(sys.argv[1]); finalize.check(s); finalize.verify_claim(s)
print('FAILED_FINALIZER='+json.dumps(dict(spec=s,jobs=submission.accepted_jobs(s)),sort_keys=True,allow_nan=False))
'''


def replay(path):
    path = Path(path).resolve(strict=True)
    previous = load_json(path)
    validate(previous, 'FINALIZE_SPEC')
    if (previous['source_commit'] != FAILED_COMMIT or previous['runtime_prefix'] != source.RUNTIME
            or path != Path(previous['root']).resolve()/'campaign_spec.json'):
        raise ValueError('Expected the failed 420d22f8 finalization execution')
    source.runtime(previous['reuse']['environment'])
    project = Path(previous['project_dir']).resolve(strict=True)
    env = dict(os.environ, PYTHONPATH=str(project/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    print('Authenticating failed finalizer and its preserved submission claim...', flush=True)
    result = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(path)],
        cwd=project, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Failed-finalizer original-source authentication failed:\n'+result.stderr)
    lines = [x.partition('=')[2] for x in result.stdout.splitlines() if x.startswith('FAILED_FINALIZER=')]
    if len(lines) != 1:
        raise ValueError('Missing/ambiguous failed-finalizer replay')
    result = json.loads(lines[0])
    jobs = result['jobs']
    if (result['spec'] != previous or set(jobs) != {'finalize_DE033', 'train_DE000'}
            or jobs['finalize_DE033'] != FAILED_JOB or len(set(jobs.values())) != 2
            or any(not isinstance(j, str) or not j.isdigit() or int(j) <= 0 for j in jobs.values())):
        raise ValueError('Failed finalizer spec or accepted job ledger changed')
    return previous, jobs


def accounting(previous, jobs):
    if jobs.get('finalize_DE033') != FAILED_JOB or set(jobs) != {'finalize_DE033', 'train_DE000'}:
        raise ValueError('Only diagnosed finalizer 7963448 may be superseded')
    result = subprocess.run(['sacct', '-S', '2026-10-26', '-j', ','.join(jobs.values()), '-X', '-n', '-P',
        '-o', 'JobIDRaw,JobName%80,User,Account,State%40,ExitCode,Elapsed'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:7] for line in result.stdout.splitlines() if line.strip()]
    states = {}
    for task, job in jobs.items():
        matches = [r for r in rows if r[0]==job]
        if len(matches) != 1 or matches[0][:4] != [job, 'lukafssf_'+task, getpass.getuser(), previous['execution_site']['account']]:
            raise ValueError('Finalizer accounting missing/ambiguous or owner/account/name changed')
        row = matches[0]
        valid = row[4:6] == ['FAILED', '1:0'] if task=='finalize_DE033' else (
            row[4].startswith('CANCELLED') and row[5:]==['0:0', '00:00:00'])
        if not valid:
            raise ValueError('Finalizer not failed or descendant active/started/not cancelled')
        states[task] = row
    return states


def audit(path):
    previous, jobs = replay(path)
    states = accounting(previous, jobs)
    root = Path(previous['root'])
    # The diagnosed failure is before attempt creation, not a partially run
    # readback/export/fit. Never treat another failure as this retry.
    if any((root/p).exists() for p in ('finalize_DE033', 'train_DE066', 'train_DE033',
            'train_DE000', 'train_DE000_attempt', 'bundles', 'targets', 'logit_targets')):
        raise ValueError('Finalization work already started; preserve and inspect')
    log = root/('finalize_DE033-'+FAILED_JOB+'.out')
    text = log.read_text(errors='replace')
    if not all(s in text for s in ('job = worker.execution(spec, spec[\'reuse\'])',
            'allocation(gate[\'execution_site\'])',
            'PermissionError: Worker host/Conda environment differs from the execution site')):
        raise ValueError('Log does not identify the diagnosed pre-work runtime-path failure')
    files = [file_ref(p) for p in sorted((root/'submission').rglob('*')) if p.is_file()]
    original = load_json(validate_file_ref(previous['finalization']['parent_spec']))
    files += [file_ref(log), file_ref(Path(original['root'])/'finalize_DE033_claim_v1/intent.json')]
    return artifact('FINALIZE_RETRY_EVIDENCE', parents=dict(previous=previous['content_hash']),
        failed_spec=file_ref(Path(path).resolve()), previous_root=previous['root'],
        previous_project=previous['project_dir'], original_spec=previous['finalization']['parent_spec'],
        original_evidence=previous['finalization']['evidence']['content_hash'], jobs=jobs, accounting=states,
        failure='registered_legacy_prefix_rejects_verified_replacement_before_work',
        work_started=False, files=files)


def protected(root, evidence):
    root = Path(root).resolve()
    for key in ('previous_root', 'previous_project'):
        other = Path(evidence[key]).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise PermissionError('Retry output overlaps failed finalizer or its source')
