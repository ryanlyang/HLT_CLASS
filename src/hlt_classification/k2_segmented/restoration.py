"""Restore the native 320000-MiB envelope without another GPU preflight."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json
from . import campaign, memory_migration as migration

ABANDONED_COMMIT = 'c640942f8a4a91086016fa4bd151ae5d3bc9c31a'


def abandoned(path):
    """An unused v2 attempt is a retirement target, never a science donor."""
    path = Path(path).resolve()
    value = load_json(path)
    if (value.get('contract') != 'K2_SEGMENTED_CAMPAIGN_SPEC/v2'
            or value['source_commit'] != ABANDONED_COMMIT
            or path != Path(value['campaign_root']).resolve() / 'campaign_spec.json'):
        raise ValueError('Expected the pinned c640942f 128-GiB attempt at its registered location')
    campaign.validate_spec(value)
    root = Path(value['campaign_root'])
    science = root / 'submissions_science'
    if ((science / 'submission_ledger.json').exists()
            or (science / 'submission_in_progress.claim').exists()
            or (science / 'submission_ledger_journal').exists()
            or (science / 'submission_intents').exists()):
        raise PermissionError('128-GiB science submission started; restoration requires review')
    # A dry plan is allowed. Any worker execution, including the gate, is not.
    for folder in ('claims', 'execution', 'tasks'):
        if (root / folder).exists() and any((root / folder).iterdir()):
            raise PermissionError('128-GiB attempt started; do not discard its progress')
    ledger = migration.ledger(value, 'gate')
    if set(ledger['jobs']) != {'preflight', 'after_gate'}:
        raise ValueError('Unexpected 128-GiB gate jobs')
    result = campaign.artifact('ABANDONED_EXECUTION', spec=migration.descriptor(path),
        ledger=migration.descriptor(root / 'submissions_gate/submission_ledger.json'),
        jobs=ledger['jobs'], final_test_accessed=False)
    return value, result


def restored_import(imported):
    value = {k: deepcopy(v) for k, v in imported.items() if k not in ('content_hash', 'contract', 'schema_version')}
    value['memory_mb'] = 320000
    return campaign.artifact('RESTORE_RESUME_IMPORT', **value)


def retirement_jobs(imported, obsolete):
    jobs = dict(imported['pending_jobs'])
    jobs.update({'abandoned_' + name: job for name, job in obsolete['jobs'].items()})
    if len(set(jobs.values())) != len(jobs):
        raise ValueError('Retirement job IDs overlap')
    return jobs


def validate_import(spec):
    imported = spec['resume_import']
    campaign.validate(imported, 'RESTORE_RESUME_IMPORT')
    unused, obsolete = abandoned(spec['abandoned_execution']['spec']['path'])
    if obsolete != spec['abandoned_execution'] or imported != restored_import(unused['resume_import']):
        raise ValueError('Restoration source snapshot differs')
    donor = migration.read_descriptor(imported['donor_spec'])
    migration.compatible_training(spec['project_dir'], donor)
    if (spec['old_jobs'] != retirement_jobs(imported, obsolete)
            or spec['source_spec'] != donor['source_spec']
            or spec['imported_receipts'] != donor['imported_receipts']
            or spec['resources'] != donor['resources']
            or spec.get('reuse_native_gate') is not True):
        raise ValueError('Restoration must retain the original native resources and science')
    return donor


def create(*, abandoned_spec, campaign_root, project_dir, source_commit):
    unused, obsolete = abandoned(abandoned_spec)
    imported = restored_import(unused['resume_import'])
    migration.verify_completed_accounting(imported)
    donor = migration.read_descriptor(imported['donor_spec'])
    project, root = Path(project_dir).resolve(), Path(campaign_root).resolve()
    campaign._source(project, source_commit)
    migration.compatible_training(project, donor)
    source = campaign.original(donor)
    protected = [Path(p).resolve() for p in (source['campaign_root'], source['project_dir'], source['data_root'],
        donor['campaign_root'], donor['project_dir'], unused['campaign_root'], unused['project_dir'], project)]
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Restoration root overlaps protected artifacts/source/data')
    if root.exists():
        raise FileExistsError('Use a fresh restoration root')
    jobs = retirement_jobs(imported, obsolete)
    states = campaign.accounting(list(jobs.values()))
    if any(states.get(j, ('UNKNOWN',))[0] != 'CANCELLED' for j in imported['pending_jobs'].values()):
        raise PermissionError('Original remainder must remain cancelled; do not create competing writers')
    if any(states.get(j, ('UNKNOWN',))[0] not in ('PENDING', 'CANCELLED') for j in obsolete['jobs'].values()):
        raise PermissionError('128-GiB gates must be pending or cancelled, never running')
    value = deepcopy(donor)
    value.pop('content_hash')
    value.update(contract='K2_SEGMENTED_CAMPAIGN_SPEC/v3', schema_version=3,
        project_dir=str(project), source_commit=source_commit, campaign_root=str(root),
        resume_import=imported, abandoned_execution=obsolete, old_jobs=jobs, reuse_native_gate=True)
    value.update(tasks=campaign.graph(value), resources=campaign.resources(value))
    spec = with_content_hash(value)
    campaign.validate_spec(spec)
    root.mkdir(parents=True)
    write_immutable_json(root / 'campaign_spec.json', spec)
    prepare(spec)
    for stage in ('full', 'science'):
        campaign.submit(spec, stage=stage)
    return spec


def gate_record(spec, native, copy):
    return campaign.artifact('REUSED_NATIVE_GATE', campaign_sha256=spec['content_hash'],
        donor_spec=spec['resume_import']['donor_spec'],
        donor_acceptance_sha256=native['content_hash'],
        donor_preflight_receipt_sha256=spec['resume_import']['receipt_hashes']['preflight'],
        resume_copy_sha256=copy['content_hash'], resources=spec['resources'],
        native_gate_reused=True, new_gpu_preflight_run=False, final_test_accessed=False)


def prepare(spec):
    """Read/copy authentic artifacts on CPU; no caches, training, or Slurm calls."""
    donor = validate_import(spec)
    native = campaign.gate(donor)
    migration.verify_completed_accounting(spec['resume_import'])
    migration.materialize(spec)
    copy = migration.verify_copy(spec)
    write_immutable_json(Path(spec['campaign_root']) / 'reused_native_gate.json', gate_record(spec, native, copy))
    return reused_gate(spec)


def reused_gate(spec):
    # Called also while saving epochs: do not repeatedly scan historical state
    # payloads here. validate_spec/fit_segment authenticate them before training.
    donor = migration.read_descriptor(spec['resume_import']['donor_spec'])
    native = campaign.gate(donor)
    if (spec['resources'] != donor['resources'] or spec['resources'] != campaign.RESOURCES
            or native['content_hash'] != spec['resume_import']['acceptance_sha256']):
        raise ValueError('Native acceptance/resources differ from restoration')
    root = Path(spec['campaign_root'])
    copy = load_json(root / 'resume_import.json')
    campaign.validate(copy, 'RESUME_COPY')
    if (copy['campaign_sha256'] != spec['content_hash']
            or copy['resume_import_sha256'] != spec['resume_import']['content_hash']
            or copy['files'] != spec['resume_import']['checkpoint']['files']
            or copy['resume_sha256'] != spec['resume_import']['checkpoint']['resume_sha256']
            or copy['pass_number'] != spec['resume_import']['checkpoint']['pass_number']
            or copy['final_test_accessed']):
        raise ValueError('Restored checkpoint copy provenance differs')
    evidence = load_json(root / 'reused_native_gate.json')
    if evidence != gate_record(spec, native, copy):
        raise ValueError('Reused native gate provenance differs')
    return native  # Keep its original identity; never manufacture a fresh measurement.
