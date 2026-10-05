"""Read-only v1 import into a fresh 128-GiB continuation, never a cold restart."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import shutil

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.jetclass2_delphes.contracts import relative_file
from . import campaign, training

EXECUTOR_DONOR = '71c1bde2d759b11cddb1f5cfbc1d199847bd6eef'
PREFIX = ('preflight', 'train_CONCAT_K2_D025_part1', 'train_CONCAT_K2_D025_part2')
NODE = 'CONCAT_K2_D025'
FIRST_TASK = 'train_CONCAT_K2_D025_part3'


def descriptor(path):
    path = Path(path).resolve()
    value = load_json(path)
    return dict(path=str(path), sha256=sha256_file(path), content_hash=value['content_hash'])


def read_descriptor(item):
    path = Path(item['path'])
    if sha256_file(path) != item['sha256']:
        raise ValueError('Migration donor bytes changed')
    value = load_json(path)
    if value['content_hash'] != item['content_hash']:
        raise ValueError('Migration donor identity changed')
    return value


def ledger(spec, stage):
    value = load_json(Path(spec['campaign_root']) / f'submissions_{stage}/submission_ledger.json')
    campaign.validate_submission_ledger(value)
    commands = campaign.plan(spec, stage)['commands']
    jobs = value['jobs']
    if (value['dry_run'] or value['campaign_spec_sha256'] != spec['content_hash']
            or set(jobs) != {r['task_id'] for r in commands}
            or len(set(jobs.values())) != len(jobs)
            or any(value['commands'][r['task_id']] != campaign._resolved(r, jobs) for r in commands)):
        raise ValueError('Migration donor submission mapping differs')
    return value


def checkpoint_snapshot(donor, receipt):
    """Authenticate every committed epoch; require the clean part2 boundary."""
    from . import runtime
    root = Path(donor['campaign_root'])
    result = receipt['result']
    directory = root / 'checkpoints' / NODE
    if (result['node_id'] != NODE
            or result['resume_directory'] != f'checkpoints/{NODE}'
            or result['spare_segment']):
        raise ValueError('Expected D025 at a clean part2 boundary')
    commits = sorted(directory.glob('epoch_*.json'))
    if not commits:
        raise ValueError('No committed D025 state')
    last = load_json(commits[-1])
    if (last['content_hash'] != result['resume_sha256'] or last['pass_number'] != result['passes']
            or last['complete'] != result['fit_complete'] or not 1 <= last['pass_number'] <= 100):
        raise ValueError('Donor checkpoint advanced beyond part2 or is inconsistent')
    state, latest = training.load_checkpoint(directory, last['binding'])
    source = campaign.original(donor)
    node = runtime.node_for(source, NODE)
    teacher = campaign.legacy.completed(source, 'reduce_CONCAT_K2_D050')
    lineage = dict(task_sha256=teacher['content_hash'], teacher_node='CONCAT_K2_D050',
                   bank_manifest_sha256=teacher['result']['bank_manifest_sha256'])
    expected = dict(runtime.binding_for(donor, node, lineage), node=node, maximum=100, acceptance=False,
                    identities=last['binding']['identities'])
    if (last['binding'] != expected or len(state['history']) != last['pass_number']
            or state['update'] != last['pass_number'] * ((source['role_counts']['train'] + 127) // 128)
            or state['selected_pass'] not in range(1, last['pass_number'] + 1)):
        raise ValueError('Imported state is not the registered scientific D025 fit')
    identities = last['binding']['identities']
    if set(identities) != {'train', 'checkpoint'} or any(
            not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in identities.values()):
        raise ValueError('Imported population binding differs')
    files = []
    for path in commits:
        commit = load_json(path)
        payload = relative_file(directory, commit['payload'])
        if sha256_file(payload) != commit['payload_sha256'] or payload.stat().st_size != commit['payload_bytes']:
            raise ValueError('Historical checkpoint bytes differ')
        for item in (payload, path):
            files.append(dict(path=item.relative_to(root).as_posix(), sha256=sha256_file(item),
                              bytes=item.stat().st_size))
    known = {row['path'] for row in files}
    for item in receipt['outputs']:
        if item['path'].startswith('execution/') or item['path'] in known:
            continue
        path = relative_file(root, item['path'])
        files.append(dict(path=item['path'], sha256=item['sha256'], bytes=path.stat().st_size))
    return dict(files=files, pass_number=latest['pass_number'], resume_sha256=latest['content_hash'],
                binding=latest['binding'], fit_complete=latest['complete'])


def snapshot(donor_path):
    from . import runtime
    donor = load_json(donor_path)
    if (donor.get('contract') != 'K2_SEGMENTED_CAMPAIGN_SPEC/v1'
            or donor['source_commit'] != EXECUTOR_DONOR):
        raise ValueError('Only the original 71c1bde2 segmented execution is a migration donor')
    campaign.validate_spec(donor)
    acceptance = campaign.gate(donor)
    gates, science = ledger(donor, 'gate'), ledger(donor, 'science')
    receipts = {}
    for name in PREFIX:
        receipt = runtime.completed(donor, name)
        if receipt is None:
            raise PermissionError('Migration needs completed ' + name)
        receipts[name] = receipt
    if receipts[PREFIX[1]]['result']['passes'] >= receipts[PREFIX[2]]['result']['passes']:
        raise ValueError('Completed segments must advance D025')
    remainder = [r['task_id'] for r in campaign.graph(donor)
                 if r['kind'] != 'preflight' and r['task_id'] not in PREFIX]
    root = Path(donor['campaign_root'])
    for name in remainder:
        if any((root / folder / (name + suffix)).exists()
               for folder, suffix in (('tasks', '.json'), ('claims', '.claim'), ('execution', '.json'))):
            raise PermissionError('Donor remainder started/advanced; do not cancel or duplicate it: ' + name)
    return campaign.artifact('MEMORY_RESUME_IMPORT', donor_spec=descriptor(donor_path),
        acceptance_sha256=acceptance['content_hash'], receipt_hashes={k: v['content_hash'] for k, v in receipts.items()},
        gate_ledger_sha256=gates['content_hash'], science_ledger_sha256=science['content_hash'],
        completed_jobs={name: (gates if name == 'preflight' else science)['jobs'][name] for name in PREFIX},
        pending_jobs={name: science['jobs'][name] for name in remainder},
        checkpoint=checkpoint_snapshot(donor, receipts[PREFIX[2]]),
        memory_mb=131072, donor_artifacts_read_only=True, final_test_accessed=False)


def compatible_training(project, donor):
    relative = Path('src/hlt_classification/k2_segmented/training.py')
    if sha256_file(Path(project) / relative) != sha256_file(Path(donor['project_dir']) / relative):
        raise ValueError('Training kernel changed; memory migration cannot reuse this resume evidence')


def validate_import(spec):
    if campaign.restored(spec):
        from .restoration import validate_import as validate_restoration
        return validate_restoration(spec)
    imported = spec['resume_import']
    campaign.validate(imported, 'MEMORY_RESUME_IMPORT')
    donor = read_descriptor(imported['donor_spec'])
    compatible_training(spec['project_dir'], donor)
    if snapshot(imported['donor_spec']['path']) != imported:
        raise ValueError('Migration source snapshot changed')
    if (spec['old_jobs'] != imported['pending_jobs'] or spec['source_spec'] != donor['source_spec']
            or spec['imported_receipts'] != donor['imported_receipts']):
        raise ValueError('Migration source/pending mapping differs')
    return donor


def verify_completed_accounting(imported):
    jobs = imported['completed_jobs']
    states = campaign.accounting(list(jobs.values()))
    if any(states.get(job) != ('COMPLETED', '0:0') for job in jobs.values()):
        raise PermissionError('Donor preflight and both D025 segments must have completed successfully')


def create(*, donor_spec, campaign_root, project_dir, source_commit):
    """No Slurm writes. Registration refuses running/advanced donor remainder."""
    donor_path = Path(donor_spec).resolve()
    imported = snapshot(donor_path)
    donor = read_descriptor(imported['donor_spec'])
    project, root = Path(project_dir).resolve(), Path(campaign_root).resolve()
    campaign._source(project, source_commit)
    compatible_training(project, donor)
    source = campaign.original(donor)
    protected = [Path(v).resolve() for v in (source['campaign_root'], source['project_dir'], source['data_root'],
                 donor['campaign_root'], donor['project_dir'], str(project))]
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Migration root overlaps protected source/data/project')
    if root.exists():
        raise FileExistsError('Use a fresh 128-GiB continuation root')
    verify_completed_accounting(imported)
    states = campaign.accounting(list(imported['pending_jobs'].values()))
    if any(states.get(job, ('UNKNOWN',))[0] not in ('PENDING', 'CANCELLED')
           for job in imported['pending_jobs'].values()):
        raise PermissionError('Donor remainder must still be pending or cancelled; no running-job cutover')
    spec = deepcopy(donor)
    spec.pop('content_hash')
    spec.update(contract='K2_SEGMENTED_CAMPAIGN_SPEC/v2', schema_version=2,
        project_dir=str(project), source_commit=source_commit, campaign_root=str(root),
        old_jobs=imported['pending_jobs'], resume_import=imported)
    spec.update(tasks=campaign.graph(spec), resources=campaign.resources(spec))
    spec = with_content_hash(spec)
    campaign.validate_spec(spec)
    root.mkdir(parents=True)
    write_immutable_json(root / 'campaign_spec.json', spec)
    for stage in ('full', 'gate', 'science'):
        campaign.submit(spec, stage=stage)
    return spec


def materialize(spec):
    """Copy committed states byte-for-byte, independently owned; never rewrite binding."""
    donor = validate_import(spec)
    imported = spec['resume_import']
    root, old = Path(spec['campaign_root']), Path(donor['campaign_root'])
    files = imported['checkpoint']['files']
    needed = sum(row['bytes'] for row in files if not relative_file(root, row['path']).exists())
    if (sum(row['bytes'] for row in files) > spec['maximum_checkpoint_bytes']
            or shutil.disk_usage(root).free < needed + spec['minimum_free_bytes'] + max(r['bytes'] for r in files)):
        raise OSError('Insufficient space to preserve imported full-state checkpoints')
    for row in files:
        source, target = relative_file(old, row['path']), relative_file(root, row['path'])
        if source.stat().st_size != row['bytes'] or sha256_file(source) != row['sha256']:
            raise ValueError('Checkpoint changed during migration')
        if not target.exists():
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Checkpoint changed while reading')
            atomic_publish_bytes(target, payload)
        if target.stat().st_size != row['bytes'] or sha256_file(target) != row['sha256']:
            raise ValueError('Destination checkpoint conflict; never overwrite')
    state, receipt = training.load_checkpoint(root / 'checkpoints' / NODE, imported['checkpoint']['binding'])
    if receipt['content_hash'] != imported['checkpoint']['resume_sha256']:
        raise ValueError('Imported checkpoint is not the part2 endpoint')
    path = root / 'resume_import.json'
    record = campaign.artifact('RESUME_COPY', campaign_sha256=spec['content_hash'],
        resume_import_sha256=imported['content_hash'], files=files,
        resume_sha256=receipt['content_hash'], pass_number=state['pass_number'], final_test_accessed=False)
    write_immutable_json(path, record)
    return path


def verify_copy(spec):
    path = Path(spec['campaign_root']) / 'resume_import.json'
    record = load_json(path)
    campaign.validate(record, 'RESUME_COPY')
    imported = spec['resume_import']
    if (record['campaign_sha256'] != spec['content_hash'] or record['resume_import_sha256'] != imported['content_hash']
            or record['files'] != imported['checkpoint']['files'] or record['final_test_accessed']
            or record['resume_sha256'] != imported['checkpoint']['resume_sha256']
            or record['pass_number'] != imported['checkpoint']['pass_number']):
        raise ValueError('Resume copy provenance differs')
    for row in record['files']:
        if sha256_file(relative_file(Path(spec['campaign_root']), row['path'])) != row['sha256']:
            raise ValueError('Imported checkpoint copy changed')
    return record
