"""Verify an already exported DE033 fit, then train only its DE000 descendant."""
from copy import deepcopy
from pathlib import Path
import gc
import getpass
import subprocess
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from .contracts import artifact, validate
from . import campaign, worker, banks, kd_banks, recovery, submission, finalize_source as source

CONTRACT = 'LUKA_FULLSTAT_SPLITHEAD_FINALIZE_SPEC/v1'
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SPLITHEAD_FINALIZE_V1'
MINIMUM_GIB = 2


def accounting(old, jobs):
    if jobs != source.JOBS:
        raise ValueError('Only the diagnosed 7938792 closure may be finalized')
    proc = subprocess.run(['sacct', '-S', '2026-10-25', '-j', ','.join(jobs.values()), '-X', '-n', '-P',
        '-o', 'JobIDRaw,JobName%80,User,Account,State%40,ExitCode,Elapsed'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:7] for line in proc.stdout.splitlines() if line.strip()]
    result = {}
    for task, job in jobs.items():
        found = [r for r in rows if r[0] == job]
        if len(found) != 1 or found[0][:4] != [job, 'lukafssr_'+task, getpass.getuser(), old['execution_site']['account']]:
            raise ValueError('Missing/ambiguous accounting or owner/account/name mismatch: '+job)
        row = found[0]
        if task == 'train_DE000':
            valid = row[4].startswith('CANCELLED') and row[5:] == ['0:0', '00:00:00']
        else:
            valid = row[4:6] == (['COMPLETED', '0:0'] if task == 'recover_DE066' else ['FAILED', '1:0'])
        if not valid:
            raise ValueError('Unexpected job state/exit or started descendant: '+job)
        result[task] = row
    return result


def audit(old, jobs, de066):
    states = accounting(old, jobs)
    root = Path(old['root'])
    if any((root/p).exists() for p in ('train_DE033/task_receipt.json', 'train_DE000',
            'train_DE000_attempt', 'bundles/DE000.json')):
        raise ValueError('Unexpected completion or started DE000; preserve and inspect')
    if de066 is None or recovery.completed_de066(old) != de066:
        raise ValueError('Completed DE066 ancestry changed')
    bundle = banks.bundle_record(old, 'DE033')
    if load_json(root/'bundles/DE033.json') != bundle:
        raise ValueError('Original DE033 target bundle changed')
    worker.validate_startup(old, 'DE033', load_json(root/'train_DE033_attempt/startup.json'), bundle['content_hash'])
    report = worker.validate_fit(root/'train_DE033', old, 'DE033', bundle['content_hash'])
    for kind in ('embedding', 'kd'):
        worker.validate_export(old, 'DE033', kind)
    log = root/'train_DE033-7938792.out'
    text = log.read_text(errors='replace')
    required = ('stage=DE033 epoch=20/20 auc=',
        'Original campaign authentication failed:',
        'Fatal Python error: init_fs_encoding:',
        "can't find encoding")
    if not all(s in text for s in required):
        raise ValueError('Log does not establish the diagnosed post-export Python startup failure')
    files = [root/p for p in worker.outputs('DE033')]
    files += [root/'train_DE066/task_receipt.json', log]
    for group in ('records', 'kd_records'):
        for ref in bundle[group].values():
            path = validate_file_ref(ref)
            files += [path, validate_file_ref(load_json(path)['payload'], root=path.parent)]
    fit = load_json(root/'train_DE033/receipt.json')
    return artifact('FINALIZE_EVIDENCE', parents=dict(original=old['content_hash'],
        de066=de066['content_hash'], fit=fit['content_hash'], bank=bundle['content_hash']),
        old_jobs=jobs, accounting=states, failure='post_export_Python_encodings_unavailable',
        selected_epoch=report['selected_pass'], selected_auc=report['validation']['macro_ovr_auc'],
        training_repeated=False, targets_regenerated=False,
        files=[file_ref(p) for p in sorted(set(files))])


def policy(old):
    p = deepcopy(recovery.policy(load_json(validate_file_ref(old['recovery']['parent_spec'])), old['recovery']['evidence']))
    root = Path(old['root'])
    for t in ('DE066', 'DE033'):
        refs = banks.refs_for(old, t)
        if t == 'DE033':
            refs = dict(refs, predictions=file_ref(root/'train_DE033/validation_predictions.npz'))
        p['external_refs'][t] = refs
        p['imported_banks'][t] = dict(embedding=file_ref(root/'targets'/t/'manifest.json'),
                                    kd=file_ref(root/'logit_targets'/t/'manifest.json'))
    p.update(tasks=[dict(task_id='finalize_DE033', dependencies=[]),
        dict(task_id='train_DE000', dependencies=['finalize_DE033'])],
        new_scientific_fits=1, reused_scientific_fits=2,
        recovery_action='verify_saved_fit_and_exports_then_finalize_only',
        runtime_prefix=source.RUNTIME, minimum_result_gib=MINIMUM_GIB)
    return p


def protected(root, old, project):
    recovery.protected(root, old, project)
    original = load_json(validate_file_ref(old['recovery']['parent_spec']))
    recovery.protected(root, original, project)
    a, b = Path(root).resolve(), Path(source.RUNTIME).resolve()
    if a.is_relative_to(b) or b.is_relative_to(a):
        raise PermissionError('Output overlaps replacement environment')


def create(root, previous_spec, project, commit, *, quota_gib, failed_finalization=None):
    root, project = Path(root).resolve(), Path(project).resolve()
    snap = source.current(project, commit)
    old, jobs, de066 = source.original(previous_spec)
    protected(root, old, project)
    if type(quota_gib) is not int or quota_gib < MINIMUM_GIB:
        raise ValueError('Declare at least 2 GiB personal durable headroom for this one-fit closure')
    evidence = audit(old, jobs, de066)
    extra = {}; parent_hashes = dict(source=snap['content_hash'], original=old['content_hash'], evidence=evidence['content_hash'])
    if failed_finalization is not None:
        from . import finalize_retry as retry
        failed = retry.audit(failed_finalization)
        if (failed['original_spec'] != file_ref(Path(previous_spec))
                or failed['original_evidence'] != evidence['content_hash']):
            raise ValueError('Retry is not bound to the same original fit evidence')
        retry.protected(root, failed)
        extra['retry'] = failed
        parent_hashes['retry'] = failed['content_hash']
    spec = artifact('FINALIZE_RETRY_SPEC' if extra else 'FINALIZE_SPEC', parents=parent_hashes,
        root=str(root), project_dir=str(project), source_commit=commit,
        source_snapshot=snap, reuse=old['reuse'], available_result_bytes=quota_gib*2**30,
        finalization=dict(parent_spec=file_ref(Path(previous_spec)), evidence=evidence), **extra, **policy(old))
    root.mkdir(parents=True, exist_ok=False)
    campaign.storage(spec)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    rec = spec['finalization']
    old, jobs, de066 = source.original(validate_file_ref(rec['parent_spec']))
    from . import finalize_retry as retry
    retrying = spec.get('contract') == retry.CONTRACT
    parent_hashes = dict(source=spec['source_snapshot']['content_hash'],
        original=old['content_hash'], evidence=rec['evidence']['content_hash'])
    if retrying:
        failed = retry.audit(validate_file_ref(spec['retry']['failed_spec']))
        if (failed != spec['retry'] or failed['original_spec'] != rec['parent_spec']
                or failed['original_evidence'] != rec['evidence']['content_hash']):
            raise ValueError('Retry evidence or original fit changed')
        retry.protected(spec['root'], failed)
        parent_hashes['retry'] = failed['content_hash']
    elif 'retry' in spec:
        raise ValueError('Retry requires its own versioned contract')
    validate(spec, 'FINALIZE_RETRY_SPEC' if retrying else 'FINALIZE_SPEC', parent_hashes)
    if (source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or spec['reuse'] != old['reuse'] or audit(old, jobs, de066) != rec['evidence']
            or any(spec.get(k) != v for k, v in policy(old).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < MINIMUM_GIB*2**30):
        raise ValueError('Finalization source/evidence/recipe/runtime changed')
    protected(spec['root'], old, spec['project_dir'])
    return old


def claim_identity(spec, plan):
    old = load_json(validate_file_ref(spec['finalization']['parent_spec']))
    path = Path(old['root'])/'finalize_DE033_claim_v1'
    kind = 'FINALIZE_CLAIM'
    parent_hashes = dict(spec=spec['content_hash'], plan=plan['content_hash'], original=old['content_hash'])
    from . import finalize_retry as retry
    if spec.get('contract') == retry.CONTRACT:
        previous = load_json(validate_file_ref(spec['retry']['failed_spec']))
        path = Path(previous['root'])/'retry_after_7963448_v1'
        kind = 'FINALIZE_RETRY_CLAIM'
        parent_hashes['previous'] = previous['content_hash']
    return path, kind, parent_hashes


def claim(spec, plan):
    path, kind, parent_hashes = claim_identity(spec, plan)
    path.mkdir(exist_ok=False)
    write_immutable_json(path/'intent.json', artifact(kind, parents=parent_hashes, recovery_root=spec['root']))


def verify_claim(spec):
    path, kind, parent_hashes = claim_identity(spec, submission.render(spec))
    record = load_json(path/'intent.json')
    validate(record, kind, parent_hashes)
    if record['recovery_root'] != spec['root']:
        raise ValueError('Wrong finalization claim')


def parents(spec):
    e = spec['finalization']['evidence']
    return dict(spec=spec['content_hash'], evidence=e['content_hash'],
                original_fit=e['parents']['fit'], original_de066=e['parents']['de066'])


def completed(spec, stage, *, accounting=True):
    if stage not in ('DE066', 'DE033'):
        raise ValueError('Only two imported stages')
    root = Path(spec['root'])
    if not (root/('train_'+stage)/'task_receipt.json').is_file():
        return None
    verify_claim(spec)
    rec = spec['finalization']; old = load_json(validate_file_ref(rec['parent_spec']))
    de066 = recovery.completed_de066(old)
    if audit(old, rec['evidence']['old_jobs'], de066) != rec['evidence']:
        raise ValueError('Finalized evidence changed')
    records = {}
    for t in ('DE066', 'DE033'):
        record = load_json(root/('train_'+t)/'task_receipt.json')
        validate(record, 'FINALIZE_TASK', parents(spec))
        expected = dict(stage=t, task_id='finalize_DE033', source_commit=spec['source_commit'],
            job_id=submission.accepted_jobs(spec).get('finalize_DE033'), training_repeated=False,
            targets_regenerated=False, outputs=[file_ref(root/'finalize_DE033/readback.json', root=root)])
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Finalized ancestor receipt differs')
        records[t] = record
    readback = load_json(root/'finalize_DE033/readback.json')
    validate(readback, 'FINALIZE_READBACK', parents(spec))
    expected = dict(native_prediction_bytes_equal=True, train_joins_verified=True,
        targets_regenerated=False, training_repeated=False, rows=spec['counts']['validation'],
        selected_refs=spec['external_refs']['DE033'], runtime_prefix=source.RUNTIME)
    if any(readback.get(k) != v for k, v in expected.items()):
        raise ValueError('Missing exact native readback or TRAIN join evidence')
    worker.limits(spec, readback['memory'])
    if accounting:
        submission.successful(spec, 'finalize_DE033', records[stage]['job_id'])
    return records[stage]


def run(spec):
    old = check(spec)
    verify_claim(spec)
    job = worker.execution(spec, spec['reuse'])
    if job != submission.accepted_jobs(spec).get('finalize_DE033'):
        raise PermissionError('Not the exact submitted finalization job')
    root = Path(spec['root']); attempt = root/'finalize_DE033'
    attempt.mkdir(exist_ok=False)
    campaign.storage(spec)
    log = PhaseLog(attempt, lambda: memory_snapshot(torch))
    meta = worker.metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Registered support changed')
    with log.phase('readback_saved_DE033_native_checkpoint_no_training'):
        val = worker.data(spec, 'validation', 'D033', authenticated=meta)
        refs = spec['external_refs']['DE033']
        native = DelphesParticleTransformer().to('cuda')
        native.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']), map_location='cpu', weights_only=True), strict=True)
        expected = load_npz_arrays(validate_file_ref(refs['predictions']))
        actual = predict(native, val, device='cuda', batch_size=128)
        if (not np.array_equal(val.identities, expected['identities']) or not np.array_equal(val.labels, expected['labels'])
                or not np.array_equal(actual, expected['probabilities'])):
            raise ValueError('Saved DE033 native prediction bytes differ')
        del val, native, expected, actual
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('verify_saved_TRAIN_targets_without_regeneration'):
        tr = worker.data(spec, 'train', 'D033', authenticated=meta)
        bundle = banks.Bundle(old, 'DE033', tr)
        if bundle.manifest['content_hash'] != spec['finalization']['evidence']['parents']['bank']:
            raise ValueError('Original TRAIN bundle changed')
        del bundle
        for module, kind in ((banks, 'embedding'), (kd_banks, 'kd')):
            _, arrays = module.arrays(spec['imported_banks']['DE033'][kind], tr,
                refs=refs, teacher_id='DE033')
            del arrays
        del tr
        gc.collect(); torch.cuda.empty_cache()
    memory = memory_snapshot(torch)
    worker.limits(spec, memory)
    # A new interpreter must still work before publishing successful provenance.
    check(spec)
    if worker.execution(spec, spec['reuse']) != job:
        raise PermissionError('Finalization environment changed')
    write_immutable_json(attempt/'readback.json', artifact('FINALIZE_READBACK', parents=parents(spec),
        native_prediction_bytes_equal=True, train_joins_verified=True, targets_regenerated=False,
        training_repeated=False, rows=spec['counts']['validation'], selected_refs=refs,
        runtime_prefix=source.RUNTIME, memory=memory))
    for stage in ('DE066', 'DE033'):
        directory = root/('train_'+stage); directory.mkdir(exist_ok=False)
        write_immutable_json(directory/'task_receipt.json', artifact('FINALIZE_TASK', parents=parents(spec),
            stage=stage, task_id='finalize_DE033', source_commit=spec['source_commit'], job_id=job,
            training_repeated=False, targets_regenerated=False,
            outputs=[file_ref(attempt/'readback.json', root=root)]))
    return completed(spec, 'DE033', accounting=False)
