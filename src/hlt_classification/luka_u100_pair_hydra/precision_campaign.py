"""Separate one-job diagnostic lifecycle; no access to science submission."""
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag, load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from . import campaign, inputs, site, precision
from .contracts import artifact, validate, SITE, COUNTS, PARITY, recipe

AUTHORIZATION = 'AUTHORIZE_LUKA_HYDRA_PRECISION_DIAGNOSTIC_V1'


def registration():
    return dict(execution_site=SITE, cpus=6, memory_mb=120000, minutes=60,
        storage_budget_bytes=256*2**20, counts=COUNTS, numerical_threads=1,
        batch_size=128, complete_probe_batches=4, include_partial_tail=True,
        modes=list(precision.MODES), original_parity_limits=PARITY,
        scientific_fits=0, reference_fits=0, teacher_fits=0, admission=False,
        automatic_science=False, automatic_retry=False, role='validation', coordinate='D000')


def storage(spec):
    root = Path(spec['root'])
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root if root.exists() else root.parent).free < remaining+2**30:
        raise OSError('Diagnostic needs 256 MiB output allowance plus one GiB filesystem headroom')


def _origin(bundle):
    origin = bundle.json('origin/campaign_spec.json'); proof = bundle.manifest['origin']
    if (origin['content_hash'] != proof['campaign'] or origin['reference']['content_hash'] != proof['reference']
            or origin['reference']['environment'] != proof['environment'] or origin['counts'] != COUNTS
            or origin['source_commit'] != proof['source_commit'] or origin['recipe'] != recipe()
            or origin['reference']['job_id'] != '21993056'):
        raise ValueError('Diagnostic origin does not bind the registered saved control')
    return origin


def create(root, bundle_root, project, commit):
    root = Path(root).absolute(); project = Path(project).resolve()
    bundle = inputs.Bundle(bundle_root); origin = _origin(bundle)
    env = Path(sys.prefix).resolve()
    if (root.resolve() != root or root.exists() or not root.parent.is_dir()
            or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in (bundle.root, project, env))):
        raise PermissionError('Use a fresh diagnostic root separate from source, inputs and environment')
    site.environment(origin['reference']['environment'])
    snapshot = campaign.current(project, commit)
    spec = artifact('PRECISION_SPEC', parents=dict(source=snapshot['content_hash'], origin=origin['content_hash'],
        transfer=bundle.manifest['content_hash']), root=str(root), project_dir=str(project), source_commit=commit,
        source_snapshot=snapshot, bundle_root=str(bundle.root), environment_dir=str(env),
        reference=origin['reference'], **registration())
    storage(spec); root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    bundle = inputs.Bundle(spec['bundle_root']); origin = _origin(bundle)
    validate(spec, 'PRECISION_SPEC', dict(source=spec['source_snapshot']['content_hash'],
        origin=origin['content_hash'], transfer=bundle.manifest['content_hash']))
    if (any(spec.get(k) != v for k, v in registration().items()) or spec['reference'] != origin['reference']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or campaign.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or Path(sys.prefix).resolve() != Path(spec['environment_dir'])):
        raise ValueError('Immutable diagnostic registration/source/environment differs')
    site.environment(spec['reference']['environment'])
    if check_storage:
        storage(spec)
    return spec


def plan(spec):
    root = Path(spec['root']); project = spec['project_dir']
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
        '--partition='+SITE['partition'], '--account='+SITE['account'], '--qos='+SITE['qos'],
        '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M', '--gres='+SITE['gres'],
        '--time=01:00:00', '--job-name=lukaph_precision', '--comment=lukaph_precision:'+spec['content_hash'],
        '--chdir='+project, '--output='+str(root/'precision-%j.out'),
        project+'/sbatch/run_luka_u100_pair_hydra.sh', project, str(root/'campaign_spec.json'), 'precision']
    return artifact('PRECISION_PLAN', parents=dict(spec=spec['content_hash']), jobs=1, scientific_fits=0,
        automatic_science=False, authorization=AUTHORIZATION,
        commands=[dict(task_id='precision', dependencies=[], command=command)])


def review(spec):
    check(spec); value = plan(spec); site.check_site(value)
    root = Path(spec['root'])/'submission'; root.mkdir(exist_ok=True)
    write_immutable_json(root/'command_plan.json', value)
    submit_exact_dag(identity=value['content_hash'], plan=value, output=root/'dry_run_submission_ledger.json',
        canonical_dry_run=root/'dry_run_submission_ledger.json', execute=False, environment=clean_environment())
    return value


def submit(spec, plan_hash, authorization):
    check(spec); root = Path(spec['root'])/'submission'; value = plan(spec)
    if (authorization != AUTHORIZATION or plan_hash != value['content_hash']
            or load_json(root/'command_plan.json') != value):
        raise PermissionError('Exact reviewed single diagnostic plan required')
    site.check_site(value)
    (root/'live_claim').mkdir(exist_ok=False)
    write_immutable_json(root/'live_claim/intent.json', artifact('PRECISION_SUBMIT_CLAIM', parents=dict(plan=value['content_hash'])))
    return submit_exact_dag(identity=value['content_hash'], plan=value, output=root/'submission_ledger.json',
        canonical_dry_run=root/'dry_run_submission_ledger.json', execute=True, environment=clean_environment())


def accepted_job(spec):
    root = Path(spec['root'])/'submission'; value = plan(spec)
    if load_json(root/'command_plan.json') != value:
        raise ValueError('Diagnostic command plan differs')
    events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=value['content_hash'], plan=value)
    if (root/'submission_ledger.json').exists() and load_json(root/'submission_ledger.json') != assemble_submission_ledger(
            campaign_spec_sha256=value['content_hash'], events=events):
        raise ValueError('Diagnostic ledger/journal differs')
    return jobs.get('precision')


def evaluate(spec, folder, *, device='cuda'):
    """Called only inside the authenticated diagnostic job; no training reads."""
    import torch
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.jetclass2_delphes.execution import gpu_identity
    from hlt_classification.luka_embedding_weight.worker import validate_predictions
    folder = Path(folder); bundle = inputs.Bundle(spec['bundle_root'])
    view = inputs.load_view(spec, 'validation', 'D000')
    saved_path = bundle.path('reference/matched/predictions.npz')
    checkpoint = bundle.path('reference/matched/checkpoint.pt')
    saved = load_npz_arrays(saved_path); validate_predictions(saved, len(view))
    if not np.array_equal(saved['identities'], view.identities) or not np.array_equal(saved['labels'], view.labels):
        raise ValueError('Diagnostic validation population/order differs')
    model = DelphesParticleTransformer().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    settings = precision.backend_settings()
    print('PRECISION full validation using original BF16 path', flush=True)
    actual = predict(model, view, device=device, batch_size=128)
    full = precision.full_comparison(view.labels, actual, saved['probabilities'])
    full['settings'] = settings
    atomic_publish_bytes(folder/'hydra_validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=view.identities, labels=view.labels, probabilities=actual)))
    write_immutable_json(folder/'full_validation.json', artifact('PRECISION_FULL', parents=dict(spec=spec['content_hash']), **full))
    probe, arrays = precision.probe(model, view, actual, saved['probabilities'])
    atomic_publish_bytes(folder/'fixed_batch_predictions.npz', deterministic_npz_bytes(arrays))
    result = artifact('PRECISION_REPORT', parents=dict(spec=spec['content_hash']),
        checkpoint=file_ref(checkpoint), saved_predictions=file_ref(saved_path),
        environment=installed_environment(), gpu=gpu_identity(), full_validation=full, probe=probe,
        admission=False, scientific_fits_released=False, reference_retrained=False,
        outputs=[file_ref(folder/name, root=folder) for name in (
            'hydra_validation_predictions.npz', 'full_validation.json', 'fixed_batch_predictions.npz')])
    write_immutable_json(folder/'report.json', result)
    return result


def run(spec):
    from hlt_classification.luka_fullsim.preflight_v2 import require_backend
    check(spec); require_backend(); job = site.allocation(spec)
    for _ in range(12):
        if accepted_job(spec) == job:
            break
        time.sleep(5)
    else:
        raise PermissionError('Not the exact submitted diagnostic job')
    folder = Path(spec['root'])/'precision'; folder.mkdir(exist_ok=False)
    write_immutable_json(folder/'intent.json', artifact('PRECISION_RUN_CLAIM', parents=dict(spec=spec['content_hash']), job_id=job))
    result = evaluate(spec, folder)
    check(spec)
    write_immutable_json(folder/'receipt.json', artifact('PRECISION_RECEIPT', parents=dict(spec=spec['content_hash'],
        report=result['content_hash']), job_id=job, report=file_ref(folder/'report.json', root=folder)))
    precision.print_report(result)
    return result


def results(spec):
    check(spec, check_storage=False)
    folder = Path(spec['root'])/'precision'; receipt = load_json(folder/'receipt.json')
    validate_file_ref(receipt['report'], root=folder)
    value = load_json(folder/'report.json')
    validate(value, 'PRECISION_REPORT', dict(spec=spec['content_hash']))
    validate(receipt, 'PRECISION_RECEIPT', dict(spec=spec['content_hash'], report=value['content_hash']))
    bundle = inputs.Bundle(spec['bundle_root'])
    if (receipt['job_id'] != accepted_job(spec) or receipt['report']['path'] != 'report.json'
            or value['admission'] is not False or value['scientific_fits_released'] is not False
            or value['reference_retrained'] is not False or value['environment'] != spec['reference']['environment']
            or value['checkpoint'] != file_ref(bundle.path('reference/matched/checkpoint.pt'))
            or value['saved_predictions'] != file_ref(bundle.path('reference/matched/predictions.npz'))
            or [r['path'] for r in value['outputs']] != [
                'hydra_validation_predictions.npz', 'full_validation.json', 'fixed_batch_predictions.npz']):
        raise ValueError('Diagnostic completion/lineage differs')
    for ref in value['outputs']:
        validate_file_ref(ref, root=folder)
    return value
