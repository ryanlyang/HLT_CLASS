"""Recover one published native fit, then run two unchanged scientific rungs.

Never resumes an optimizer, retrains DE066, edits the failed execution, or
pretends that its failed scheduler job completed. New task receipts explicitly
bind the old FIT receipt and successful recovery job.
"""
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
from . import campaign, worker, banks, submission, recovery_source

FAILED_JOB = '7910169'
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SPLITHEAD_RECOVERY_V1'
RECOVERY_CONTRACT = 'LUKA_FULLSTAT_SPLITHEAD_RECOVERY_SPEC/v1'


def accounting(old, jobs):
    """Only the exact failed fit and never-started cancelled descendants."""
    if jobs.get('train_DE066') != FAILED_JOB or len(set(jobs.values())) != 3:
        raise ValueError('Recovery is registered only for failed DE066 job '+FAILED_JOB)
    if any(not str(j).isdigit() or int(j) <= 0 for j in jobs.values()):
        raise ValueError('Invalid old job ID')
    result = subprocess.run(['sacct', '-S', '2026-10-24', '-j', ','.join(jobs.values()), '-X', '-n', '-P',
        '-o', 'JobIDRaw,JobName%80,User,Account,State%40,ExitCode,Elapsed'],
        check=True, capture_output=True, text=True)
    rows = [line.strip().split('|')[:7] for line in result.stdout.splitlines() if line.strip()]
    result = {}
    for task, job in jobs.items():
        matching = [r for r in rows if r[0] == job]
        if len(matching) != 1 or matching[0][:4] != [job, 'lukafss_'+task, getpass.getuser(), old['execution_site']['account']]:
            raise ValueError('Missing/ambiguous old job or owner/account/name mismatch: '+job)
        row = matching[0]
        if task == 'train_DE066':
            if row[4:6] != ['FAILED', '1:0']:
                raise ValueError('Expected FAILED/1:0 for the saved DE066 fit')
        elif not row[4].startswith('CANCELLED') or row[5:] != ['0:0', '00:00:00']:
            raise ValueError('Descendant active, started or not cancelled; no duplicate submission: '+job)
        result[task] = row
    return result


def audit(old, jobs):
    states = accounting(old, jobs)
    root = Path(old['root']); directory = root/'train_DE066'
    forbidden = ['train_DE066/task_receipt.json', 'targets/DE066', 'logit_targets/DE066']
    forbidden += [p for stage in ('DE033', 'DE000') for p in ('train_'+stage, 'train_'+stage+'_attempt')]
    if any((root/p).exists() for p in forbidden):
        raise ValueError('Unexpected later-stage or partial export output; preserve and inspect')
    bundle = banks.bundle_record(old, 'DE066')
    if load_json(root/'bundles/DE066.json') != bundle:
        raise ValueError('Saved DE066 target bundle changed')
    worker.validate_startup(old, 'DE066', load_json(root/'train_DE066_attempt/startup.json'), bundle['content_hash'])
    report = worker.validate_fit(directory, old, 'DE066', bundle['content_hash'])
    # This repair must explain the reported failure without tolerating other
    # evidence differences. validate_fit above checks the exact registered sums.
    reordered = [x['pass_number'] for x in report['validation_history'] if x['total_loss'] !=
        x['weighted_ce_loss']+sum(x['weighted_logit_losses'].values())+sum(x['weighted_embedding_losses'].values())]
    if not reordered:
        raise ValueError('Saved report does not exhibit the registered JSON sum-order failure')
    worker.validate_export(old, 'M0HLT', 'kd')
    paths = [directory/n for n in ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json')]
    paths += [root/'bundles/DE066.json', root/'train_DE066_attempt/startup.json']
    for group in ('records', 'kd_records'):
        for ref in bundle[group].values():
            manifest = validate_file_ref(ref)
            m = load_json(manifest)
            paths += [manifest, validate_file_ref(m['payload'], root=manifest.parent)]
    receipt = load_json(directory/'receipt.json')
    return artifact('RECOVERY_EVIDENCE', parents=dict(original=old['content_hash'], fit=receipt['content_hash'],
        bank=bundle['content_hash']), failed_job=FAILED_JOB, old_jobs=jobs, accounting=states,
        reordered_epochs=reordered, training_repeated=False,
        selected_epoch=report['selected_pass'], selected_auc=report['validation']['macro_ovr_auc'],
        files=[file_ref(p) for p in sorted(set(paths))])


def policy(old, evidence):
    # Original science, rows, seeds, support, optimizer, environment and resource
    # acceptance are reused verbatim. Only execution/provenance changes.
    p = deepcopy(campaign.policy(old['reuse']))
    directory = Path(old['root'])/'train_DE066'
    p['external_refs']['DE066'] = {k: file_ref(directory/f) for k, f in
        [('checkpoint', 'selected.pt'), ('report', 'training_report.json'), ('predictions', 'validation_predictions.npz')]}
    p['imported_banks']['M0HLT']['kd'] = file_ref(Path(old['root'])/'logit_targets/M0HLT/manifest.json')
    p.update(tasks=[dict(task_id='recover_DE066', dependencies=[]),
        dict(task_id='train_DE033', dependencies=['recover_DE066']),
        dict(task_id='train_DE000', dependencies=['recover_DE066', 'train_DE033'])],
        new_scientific_fits=2, reused_scientific_fits=1,
        recovery_action='verify_native_fit_then_export_TRAIN_only',
        validation_rule='exact_registered_teacher_order',
        # Keep inherited allocation rather than claim an unmeasured short limit.
        recovery_walltime_inherited=True)
    return p


def protected(root, old, project):
    campaign.protected(root, old['reuse'], project)
    for other in (old['root'], old['project_dir']):
        a, b = Path(root).resolve(), Path(other).resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise PermissionError('Recovery output overlaps original execution')


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snap = recovery_source.current(project, commit)
    old, jobs = recovery_source.original(previous_spec)
    protected(root, old, project)
    if type(quota_gib) is not int or quota_gib < 16:
        raise ValueError('Declare at least 16 GiB personally available result quota')
    evidence = audit(old, jobs)
    rec = dict(parent_spec=file_ref(Path(previous_spec)), evidence=evidence)
    spec = artifact('RECOVERY_SPEC', parents=dict(source=snap['content_hash'], original=old['content_hash'],
        evidence=evidence['content_hash']), root=str(root), project_dir=str(project), source_commit=commit,
        source_snapshot=snap, reuse=old['reuse'], recovery=rec, available_result_bytes=quota_gib*2**30,
        **policy(old, evidence))
    root.mkdir(parents=True, exist_ok=False)
    campaign.storage(spec)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    rec = spec['recovery']
    old, jobs = recovery_source.original(validate_file_ref(rec['parent_spec']))
    validate(spec, 'RECOVERY_SPEC', dict(source=spec['source_snapshot']['content_hash'], original=old['content_hash'],
        evidence=rec['evidence']['content_hash']))
    if (recovery_source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or spec['reuse'] != old['reuse'] or audit(old, jobs) != rec['evidence']
            or any(spec.get(k) != v for k, v in policy(old, rec['evidence']).items())
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or type(spec['available_result_bytes']) is not int or spec['available_result_bytes'] < 16*2**30):
        raise ValueError('Recovery source/evidence/recipe/source fit changed')
    protected(spec['root'], old, spec['project_dir'])
    return old


def claim(spec, plan):
    # Serializes every replacement root for this one failed execution.
    old = load_json(validate_file_ref(spec['recovery']['parent_spec']))
    path = Path(old['root'])/'recovery_claim_v1'
    path.mkdir(exist_ok=False)
    write_immutable_json(path/'intent.json', artifact('RECOVERY_CLAIM',
        parents=dict(spec=spec['content_hash'], plan=plan['content_hash'], original=old['content_hash']),
        recovery_root=spec['root']))


def verify_claim(spec):
    old = load_json(validate_file_ref(spec['recovery']['parent_spec']))
    record = load_json(Path(old['root'])/'recovery_claim_v1/intent.json')
    validate(record, 'RECOVERY_CLAIM', dict(spec=spec['content_hash'],
        plan=submission.render(spec)['content_hash'], original=old['content_hash']))
    if record['recovery_root'] != spec['root']:
        raise ValueError('Wrong recovery claim')


def recovered_outputs():
    return ['recovery_DE066/readback.json'] + [f'{folder}/DE066/{name}'
        for folder, payload in [('targets', 'targets.npz'), ('logit_targets', 'probabilities.npz')]
        for name in ('manifest.json', payload)]


def completed_de066(spec, *, accounting=True):
    root = Path(spec['root']); path = root/'train_DE066/task_receipt.json'
    if not path.is_file():
        return None
    verify_claim(spec)
    rec = spec['recovery']; old = load_json(validate_file_ref(rec['parent_spec']))
    if audit(old, rec['evidence']['old_jobs']) != rec['evidence']:
        raise ValueError('Recovered fit evidence changed')
    record = load_json(path)
    parents = dict(spec=spec['content_hash'], original_fit=rec['evidence']['parents']['fit'], evidence=rec['evidence']['content_hash'])
    validate(record, 'RECOVERY_TASK', parents)
    if (record['task_id'] != 'recover_DE066' or record['stage'] != 'DE066'
            or record['source_commit'] != spec['source_commit'] or record['training_repeated'] is not False
            or record['job_id'] != submission.accepted_jobs(spec).get('recover_DE066')
            or [r['path'] for r in record['outputs']] != recovered_outputs()):
        raise ValueError('Recovery task identity differs')
    for ref in record['outputs']:
        validate_file_ref(ref, root=root)
    readback = load_json(root/'recovery_DE066/readback.json')
    validate(readback, 'RECOVERY_READBACK', parents)
    if (readback['native_prediction_bytes_equal'] is not True or readback['rows'] != spec['counts']['validation']
            or readback['selected_refs'] != spec['external_refs']['DE066']):
        raise ValueError('Missing exact native checkpoint validation readback')
    for kind in ('embedding', 'kd'):
        worker.validate_export(spec, 'DE066', kind)
    if accounting:
        submission.successful(spec, 'recover_DE066', record['job_id'])
    return record


def run_de066(spec):
    old = check(spec)
    verify_claim(spec)
    job = worker.execution(spec, spec['reuse'])
    if job != submission.accepted_jobs(spec).get('recover_DE066'):
        raise PermissionError('Not the exact submitted recovery job')
    root = Path(spec['root']); attempt = root/'recovery_DE066'
    attempt.mkdir(exist_ok=False)
    campaign.storage(spec)
    log = PhaseLog(attempt, lambda: memory_snapshot(torch))
    parents = dict(spec=spec['content_hash'], original_fit=spec['recovery']['evidence']['parents']['fit'],
        evidence=spec['recovery']['evidence']['content_hash'])
    meta = worker.metadata(spec)
    if meta[0]['summaries'] != spec['reuse']['summaries']:
        raise ValueError('Cache support metadata changed')
    with log.phase('verify_saved_DE066_native_checkpoint_no_training'):
        val = worker.data(spec, 'validation', 'D066', authenticated=meta)
        refs = spec['external_refs']['DE066']
        native = DelphesParticleTransformer().to('cuda')
        native.load_state_dict(torch.load(validate_file_ref(refs['checkpoint']), map_location='cpu', weights_only=True), strict=True)
        expected = load_npz_arrays(validate_file_ref(refs['predictions']))
        actual = predict(native, val, device='cuda', batch_size=128)
        if (not np.array_equal(val.identities, expected['identities']) or not np.array_equal(val.labels, expected['labels'])
                or not np.array_equal(actual, expected['probabilities'])):
            raise ValueError('Recovered native validation prediction bytes differ')
        write_immutable_json(attempt/'readback.json', artifact('RECOVERY_READBACK', parents=parents,
            rows=len(val), selected_refs=refs, native_prediction_bytes_equal=True))
        del val, native, actual, expected
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('verify_original_TRAIN_target_join_no_training'):
        tr = worker.data(spec, 'train', 'D066', authenticated=meta)
        bundle = banks.Bundle(old, 'DE066', tr)
        if bundle.manifest['content_hash'] != spec['recovery']['evidence']['parents']['bank']:
            raise ValueError('Original TRAIN bundle changed')
        del bundle
        gc.collect(); torch.cuda.empty_cache()
    for kind in ('embedding', 'kd'):
        with log.phase('export_recovered_DE066_TRAIN_'+kind):
            worker.export_target(spec, 'DE066', kind, tr, device='cuda')
    worker.limits(spec, memory_snapshot(torch))
    check(spec)
    if worker.execution(spec, spec['reuse']) != job:
        raise PermissionError('Recovery execution environment changed')
    record = artifact('RECOVERY_TASK', parents=parents, task_id='recover_DE066', stage='DE066',
        source_commit=spec['source_commit'], job_id=job, training_repeated=False,
        outputs=[file_ref(root/p, root=root) for p in recovered_outputs()])
    (root/'train_DE066').mkdir(exist_ok=False)
    write_immutable_json(root/'train_DE066/task_receipt.json', record)
    return completed_de066(spec, accounting=False)
