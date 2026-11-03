"""Small checksum-bound delta; no raw input/weight transfer or scheduler mutation."""
from pathlib import Path, PurePosixPath
import os
import re
import shutil
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, validate_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot, validate_source_snapshot_payload
from hlt_classification.luka_fullsim.regular_submission import clean_environment
from hlt_classification.luka_u100_contrastive import contracts as original
from hlt_classification.luka_u100_contrastive_serial.source import inspect_parent
from hlt_classification.luka_u100_contrastive_serial.contracts import validate as validate_parent
from .contracts import artifact, validate, ARMS, COUNTS, PARENT_COMMIT

MAX_BYTES = 1024**3
COMPLETED_FILES = ('report.json', 'predictions.npz', 'task_receipt.json', 'fit_receipt.json')


def relative(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe transfer member')
    return name


def accounting(jobs):
    ids = set(map(str, jobs.values()))
    if not ids or any(not re.fullmatch('[1-9][0-9]*', j) for j in ids):
        raise ValueError('Expected exact original job IDs')
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(ids)),
        '-o', 'JobIDRaw,State,ExitCode'], check=True, capture_output=True, text=True, env=clean_environment())
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if fields[0] in ids:
            if fields[0] in states: raise ValueError('Ambiguous original accounting')
            states[fields[0]] = fields[1:3]
    if set(states) != ids: raise ValueError('Missing original accounting; no transfer published')
    return states


def validate_manifest(value):
    validate(value, 'TRANSFER')
    old = value['origin_spec']; owner = value['key_owner']; proof = value['original_replay']
    original.validate(old, 'SPEC')
    owner_version = 1 if old.get('setup_reuse') is not None else 2
    validate_content_hash(owner, expected_contract=f'LUKA_U100_CONTRASTIVE_SPEC/v{owner_version}',
                          expected_schema_version=owner_version)
    expected_owner = old['setup_reuse']['parents']['origin'] if owner_version == 1 else old['content_hash']
    if (owner['content_hash'] != expected_owner or owner['imported'] != old['imported']
            or owner['counts'] != old['counts'] or owner['recipe'] != old['recipe']):
        raise ValueError('Wrong original key owner')
    validate_source_snapshot_payload(old['source_snapshot'])
    validate_parent(proof, 'PARENT', dict(spec=old['content_hash']))
    if (old['source_commit'] != PARENT_COMMIT or old['source_snapshot']['git_commit'] != PARENT_COMMIT
            or old['counts'] != COUNTS or old['arms'] != ARMS or old['recipe'] != original.recipe()
            or value['completed'] != sorted(proof['completed']) or set(proof['jobs']) != {*ARMS, 'report'}
            or value['parents'] != dict(origin=old['content_hash'], replay=proof['content_hash'])):
        raise ValueError('Wrong contrastive transfer lineage')
    for task, job in proof['jobs'].items():
        state = value['accounting'][job]
        if task in proof['completed']:
            if state != ['COMPLETED', '0:0'] or proof['completed'][task]['job_id'] != job:
                raise ValueError('Reused fit is not a successful exact completed job')
        elif task in ARMS and state[0] == 'COMPLETED':
            raise ValueError('Completed student has no authenticated receipt')
    if value['accounting'][proof['setup_jobs']['setup']] != ['COMPLETED', '0:0']:
        raise ValueError('Original setup did not complete')
    expected = {'keys/manifest.json', 'keys/keys.npz'} | {
        f'completed/{arm}/{name}' for arm in value['completed'] for name in COMPLETED_FILES}
    rows = value['files']
    if ({relative(r['path']) for r in rows} != expected or len(rows) != len(expected)
            or any(type(r['bytes']) is not int or r['bytes'] <= 0 or
                   not re.fullmatch('[0-9a-f]{64}', r['sha256']) for r in rows)
            or sum(r['bytes'] for r in rows) != value['payload_bytes'] or value['payload_bytes'] > MAX_BYTES):
        raise ValueError('Unexpected transfer payload or size')
    return value


def export(spec_path, output):
    path = Path(spec_path).resolve(strict=True); old = load_json(path)
    if old['source_commit'] != PARENT_COMMIT or path != Path(old['root'])/'campaign_spec.json':
        raise ValueError('Use the successful original 431341db campaign')
    validate_source_snapshot(old['source_snapshot'], repository=old['project_dir'], require_clean=True)
    print('Authenticating original contrastive setup and completed fits; no inference or scheduler changes.', flush=True)
    proof = inspect_parent(path)
    states = accounting({**proof['jobs'], **proof['setup_jobs']})
    bundle = load_json(Path(old['root'])/'setup/bundle.json')
    key_path = Path(bundle['offline_manifest']['path']); validate_file_ref(bundle['offline_manifest'])
    key = load_json(key_path)
    owner = old
    if old.get('setup_reuse') is not None:
        ref = old['setup_reuse']['origin_spec']; validate_file_ref(ref); owner = load_json(ref['path'])
    if key['parents']['spec'] != owner['content_hash']: raise ValueError('Key owner differs')
    sources = {'keys/manifest.json': key_path, 'keys/keys.npz': key_path.parent/'keys.npz'}
    expected = {'keys/manifest.json':bundle['offline_manifest'], 'keys/keys.npz':key['payload']}
    for arm, evidence in proof['completed'].items():
        folder = Path(old['root'])/arm
        refs = {r['path']:r for r in evidence['files']}
        for name, source in zip(COMPLETED_FILES, (
                folder/'fit/training_report.json', folder/'fit/validation_predictions.npz',
                folder/'receipt.json', folder/'fit/receipt.json')):
            destination = f'completed/{arm}/{name}'; sources[destination] = source
            expected[destination] = refs[str(source)]
    rows = []
    for name, source in sources.items():
        ref = file_ref(source)
        if any(ref[k] != expected[name][k] for k in ('bytes', 'sha256')):
            raise ValueError('Authenticated source changed before export')
        rows.append(dict(path=name, bytes=ref['bytes'], sha256=ref['sha256']))
    manifest = artifact('TRANSFER', parents=dict(origin=old['content_hash'], replay=proof['content_hash']),
        origin_spec=old, key_owner=owner, original_replay=proof, accounting=states,
        completed=sorted(proof['completed']), files=sorted(rows, key=lambda r:r['path']),
        payload_bytes=sum(r['bytes'] for r in rows))
    validate_manifest(manifest)
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Use a fresh transfer directory')
    if any(output.is_relative_to(p.parent) or p.is_relative_to(output) for p in sources.values()):
        raise ValueError('Transfer overlaps source artifacts')
    if shutil.disk_usage(output.parent).free < manifest['payload_bytes']+2**30:
        raise OSError('Transfer needs payload space plus one GiB headroom')
    output.mkdir()
    for row in rows:
        target = output/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        with sources[row['path']].open('rb') as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest, 2**20); dest.flush(); os.fsync(dest.fileno())
        validate_file_ref(row, root=output)
    # Manifest is the completion marker; no partial transfer is accepted.
    write_immutable_json(output/'manifest.json', manifest)
    return dict(directory=str(output), manifest_hash=manifest['content_hash'],
        payload_bytes=manifest['payload_bytes'], reused=manifest['completed'],
        remaining=[a for a in ARMS if a not in manifest['completed']])


class Delta:
    def __init__(self, root, expected_hash):
        self.root = Path(root).resolve(strict=True)
        self.manifest = validate_manifest(load_json(self.root/'manifest.json'))
        if self.manifest['content_hash'] != expected_hash:
            raise ValueError('Delta differs from the personally verified export hash')
        self.rows = {r['path']:r for r in self.manifest['files']}

    def path(self, name):
        name = relative(name)
        if name not in self.rows: raise PermissionError('Not a contrastive delta member')
        path = self.root/name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError('Redirected delta member')
        validate_file_ref(self.rows[name], root=self.root)
        return path

    def verify_all(self):
        for name in self.rows: self.path(name)
