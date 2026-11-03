"""Explicit file selection and streaming export; no scheduler or network actions.

Verification uses only the standard library. Old manifests remain byte-identical;
their absolute paths are evidence, not destinations to create on the new site.
This transfer contract does not admit training on a new execution site.
"""
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

CONTRACT = 'LUKA_U100_PAIR_TRANSFER/v1'
COUNTS = dict(train=299234, validation=74895)
MAX_JSON = 64 * 2**20
MAX_PAYLOAD = 6 * 2**30
HEADROOM = 2**30
VIEWS = (('train', 'D000'), ('validation', 'D000'), ('train', 'U100'))
POLICY = dict(raw_particles=False, final_test_payloads=False,
              validation_richer_views=False, validation_assignments=False,
              unrelated_campaign_payloads=False, environment_binaries=False,
              reference_retrained=False, original_manifests_rewritten=False,
              readiness='transfer_only_not_hydra_training_admission')

# Executed by the existing SPORC Python in the source registered by the old spec.
# No full cache or raw ROOT loading, forward pass, export, or scheduler mutation.
REPLAY = '''
import json,sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_u100_pair_correction import campaign,source,targets
s=load_json(sys.argv[1]); campaign.check(s,check_storage=False)
c=source.import_control(s['reference']['spec']['path'])
if c!=s['reference']: raise ValueError('Fresh original-source reference replay differs')
m,p,inventory,splits=targets.metadata(s)
print('PAIR_TRANSFER_ORIGIN='+json.dumps(dict(
    campaign=s['content_hash'], reference=c['content_hash'],
    prepared=p['content_hash'], cache=m['content_hash'],
    source_commit=s['source_commit'], control_job=c['job_id'],
    counts=s['counts'], environment=c['environment'],
    original_source_replay=True),sort_keys=True,allow_nan=False))
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    return dict(value, content_hash=digest(value))


def checked(value):
    bare = dict(value)
    supplied = bare.pop('content_hash')
    if supplied != digest(bare):
        raise ValueError('JSON content hash differs')
    return value


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'Missing, redirected or nonregular source: {path}')
    return path


def read_json(path):
    path = regular(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError('Metadata exceeds 64 MiB bound')
    return json.loads(path.read_bytes())


def relative(name):
    if not isinstance(name, str):
        raise ValueError('Relative path must be a string')
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or
            p.as_posix() != name or any(v in ('.', '..') for v in p.parts) or
            '\\' in name or ':' in name or any(ord(c) < 32 for c in name)):
        raise ValueError(f'Unsafe relative bundle path: {name!r}')
    return name


def authenticate(spec_path, spec):
    project = Path(spec['project_dir'])
    env = dict(os.environ, PYTHONPATH=str(project/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable, '-B', '-s', '-c', REPLAY, str(spec_path)],
                         cwd=project, env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original-source authentication failed:\n'+run.stderr)
    rows = [line.removeprefix('PAIR_TRANSFER_ORIGIN=') for line in run.stdout.splitlines()
            if line.startswith('PAIR_TRANSFER_ORIGIN=')]
    if len(rows) != 1:
        raise ValueError('Missing unique origin authentication evidence')
    return json.loads(rows[0])


def inventory(spec_path):
    """Read-only inventory. Array byte hashes are checked while exporting."""
    spec_path = regular(spec_path)
    spec = checked(read_json(spec_path))
    if (spec['contract'] != 'LUKA_U100_PAIR_CORRECTION_SPEC/v1' or
            spec['counts'] != COUNTS or spec['reference']['job_id'] != '21993056' or
            spec_path != Path(spec['root'])/'campaign_spec.json'):
        raise ValueError('Expected original four-fit campaign and completed control 21993056')
    print('Authenticating original campaign and completed control; no jobs changed.', flush=True)
    proof = authenticate(spec_path, spec)
    anchor, ref, imported = spec['anchor']['evidence'], spec['reference'], spec['imported']
    expected = dict(campaign=spec['content_hash'], reference=ref['content_hash'],
                    prepared=anchor['prepared'], cache=anchor['cache'],
                    source_commit=spec['source_commit'], control_job='21993056',
                    counts=COUNTS, environment=ref['environment'], original_source_replay=True)
    if proof != expected:
        raise ValueError('Origin proof differs from selected campaign')
    rows = []

    def add(path, destination, purpose, record=None):
        path = regular(path)
        size = path.stat().st_size
        if record is None:
            record = dict(bytes=size, sha256=sha256(path))
        if size != record['bytes'] or not re.fullmatch('[a-f0-9]{64}', record['sha256']):
            raise ValueError(f'Source size/hash record differs: {path}')
        rows.append(dict(path=relative(destination), source_path=str(path),
                         bytes=size, sha256=record['sha256'], purpose=purpose))

    def file_ref(record, destination, purpose):
        path = Path(record['path'])
        if not path.is_absolute():
            raise ValueError('Reference evidence requires an absolute source path')
        add(path, destination, purpose, record)

    def group(records, keys, prefix, purpose):
        for key in keys:
            record = records[key]
            suffix = Path(record['path']).suffix
            if suffix not in ('.json', '.pt', '.npz'):
                raise ValueError('Unexpected reference file type')
            file_ref(record, f'{prefix}/{key}{suffix}', purpose)

    add(spec_path, 'origin/campaign_spec.json', 'origin')
    base = Path(anchor['prepared_root'])
    cache, prepared = (checked(read_json(base/n)) for n in ('cache_manifest.json', 'prepared.json'))
    if cache['content_hash'] != anchor['cache'] or prepared['content_hash'] != anchor['prepared']:
        raise ValueError('Prepared cache identity changed')
    for name in ('cache_manifest.json', 'prepared.json'):
        add(base/name, 'prepared/'+name, 'prepared_metadata')
    foundation = Path(prepared['foundation_root'])
    foundation_meta = checked(read_json(foundation/'foundation.json'))
    if foundation_meta['content_hash'] != prepared['parents']['foundation']:
        raise ValueError('Foundation lineage changed')
    add(foundation/'foundation.json', 'foundation/foundation.json', 'prepared_metadata')
    for name in ('inventory.json', 'split_manifest.json'):
        add(foundation/name, 'foundation/'+name, 'prepared_metadata', foundation_meta['artifacts'][name])
    for role, coordinate in VIEWS:
        for record in cache['blocks'][role][coordinate]:
            name = f'inputs/{role}/{coordinate}/{record["file_index"]:04d}.npz'
            if record['path'] != name:
                raise ValueError('Unexpected prepared view path')
            add(base/name, 'prepared/'+name, f'view_{role}_{coordinate}', record)
    for record in prepared['assignments']['train']:
        name = f'assignments/train_{record["file_index"]:04d}.npz'
        if record['path'] != name:
            raise ValueError('Unexpected TRAIN assignment path')
        add(base/name, 'prepared/'+name, 'train_assignments', record)
    bank_ref = ref['u100_manifest']
    bank = checked(read_json(bank_ref['path']))
    if (bank['role'] != 'train' or bank['coordinate'] != 'U100' or
            bank['rows'] != COUNTS['train'] or bank['payload']['path'] != 'targets.npz' or
            bank['parents']['prepared'] != anchor['prepared'] or
            bank['parents']['teacher'] != imported['teacher']['content_hash']):
        raise ValueError('U100 embedding bank binding differs')
    file_ref(bank_ref, 'targets/u100/manifest.json', 'u100_embeddings')
    add(Path(bank_ref['path']).parent/'targets.npz', 'targets/u100/targets.npz',
        'u100_embeddings', bank['payload'])

    # One native control checkpoint enables later cross-site parity. No teacher
    # or historical student weights are needed to reuse frozen bank/predictions.
    group(ref['files'], ('receipt', 'fit', 'report', 'checkpoint', 'predictions',
                        'bundle', 'setup_receipt', 'acceptance'), 'reference/matched', 'matched_control')
    group(imported['files'], ('receipt', 'report', 'predictions', 'subset', 'bank_manifest',
                             'readback', 'fit_receipt', 'results'),
          'reference/historical', 'historical_comparison')
    group(imported['teacher']['files'], ('report', 'receipt'), 'reference/teacher', 'teacher_evidence')
    for name in ('M0HLT', 'OFFLINE'):
        group(anchor['models'][name], ('report', 'receipt', 'predictions'),
              'reference/'+name, 'recovery_anchors')
    for name, record in (('matched', ref['spec']), ('historical', imported['spec']),
                         ('fixed30', spec['anchor']['spec'])):
        file_ref(record, f'origin/{name}_spec.json', 'origin')
    rows.sort(key=lambda r: r['path'])
    total = sum(r['bytes'] for r in rows)
    if len({r['path'] for r in rows}) != len(rows) or total > MAX_PAYLOAD:
        raise ValueError('Duplicate destinations or six-GiB transfer scope exceeded')
    if read_json(spec_path) != spec:
        raise ValueError('Source campaign changed during inventory')
    result = seal(dict(contract=CONTRACT, schema_version=1, origin=proof,
                       exporter_sha256=sha256(Path(__file__)),
                       policy=POLICY, files=rows, payload_bytes=total))
    validate_manifest(result)
    return result


def validate_manifest(value):
    checked(value)
    if (value['contract'] != CONTRACT or value['schema_version'] != 1 or value['policy'] != POLICY or
            value['origin']['counts'] != COUNTS or value['origin']['control_job'] != '21993056' or
            value['origin']['original_source_replay'] is not True):
        raise ValueError('Transfer contract or population differs')
    rows = value['files']
    names = [relative(r['path']) for r in rows]
    if len(names) != len(set(names)) or 'manifest.json' in names or not rows:
        raise ValueError('Duplicate/reserved/empty bundle file list')
    for row in rows:
        if (type(row['bytes']) is not int or row['bytes'] < 0 or
                not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
            raise ValueError('Invalid payload size or digest')
    total = sum(r['bytes'] for r in rows)
    if total != value['payload_bytes'] or not 0 < total <= MAX_PAYLOAD:
        raise ValueError('Transfer size differs or exceeds scope')
    return value


class HashReader:
    def __init__(self, handle):
        self.handle = handle
        self.hash = hashlib.sha256()
        self.size = 0

    def read(self, size):
        data = self.handle.read(size)
        self.hash.update(data)
        self.size += len(data)
        return data


def member(name, size):
    info = tarfile.TarInfo(relative(name))
    info.size, info.mode, info.mtime = size, 0o600, 0
    return info


def export(spec_path, output):
    """Exclusive new directory; publish archive only after all source hashes pass."""
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Output must be a fresh directory with no symlink components')
    plan = inventory(spec_path)
    for row in plan['files']:
        source = Path(row['source_path'])
        if output == source or output in source.parents or source.parent in output.parents:
            raise ValueError('Transfer output overlaps an input directory')
    # One tar, no staging copy of each payload and no compressed duplicate.
    required = plan['payload_bytes'] + len(canonical(plan)) + len(plan['files'])*2048 + HEADROOM
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < required:
        raise OSError('Insufficient transfer space including one GiB headroom')
    output.mkdir(exist_ok=False)
    (output/'inventory.json').write_bytes(canonical(plan)+b'\n')
    partial, final = output/'campaign-inputs.tar.partial', output/'campaign-inputs.tar'
    with partial.open('xb') as handle:
        with tarfile.open(fileobj=handle, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            metadata = canonical(plan)
            tar.addfile(member('manifest.json', len(metadata)), io.BytesIO(metadata))
            for index, row in enumerate(plan['files'], 1):
                source = regular(row['source_path'])
                with source.open('rb') as raw:
                    reader = HashReader(raw)
                    tar.addfile(member(row['path'], row['bytes']), reader)
                    if (reader.size != row['bytes'] or reader.hash.hexdigest() != row['sha256']
                            or raw.read(1)):
                        raise ValueError(f'Source bytes differ during export: {source}')
                if index % 200 == 0:
                    print(f'Verified and packed {index}/{len(plan["files"])} files', flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    # Hard-link publication is atomic and cannot overwrite an existing path.
    os.link(partial, final)
    partial.unlink()
    checksum = sha256(final)
    with (output/'SHA256SUMS').open('x', encoding='ascii', newline='\n') as out:
        out.write(checksum+'  campaign-inputs.tar\n')
    return dict(archive=str(final), sha256=checksum, manifest=plan['content_hash'],
                files=len(plan['files']), payload_bytes=plan['payload_bytes'])


def verify(archive, expected_sha256):
    """Read archive in place, without extraction or loading any model/arrays."""
    archive = regular(archive)
    if not re.fullmatch('[a-f0-9]{64}', expected_sha256) or sha256(archive) != expected_sha256:
        raise ValueError('Archive SHA256 differs from the origin value')
    with tarfile.open(archive, 'r|') as tar:
        first = tar.next()
        if first is None or first.name != 'manifest.json' or not first.isfile() or first.size > MAX_JSON:
            raise ValueError('Missing bounded regular manifest as first member')
        plan = validate_manifest(json.load(tar.extractfile(first)))
        expected = {r['path']: r for r in plan['files']}
        seen = set()
        while True:
            info = tar.next()
            if info is None:
                break
            name = relative(info.name)
            if name not in expected or name in seen or not info.isfile() or info.issparse():
                raise ValueError('Unexpected/duplicate/nonregular archive member')
            row = expected[name]
            if info.size != row['bytes']:
                raise ValueError('Archive member size differs')
            reader = HashReader(tar.extractfile(info))
            while reader.read(2**20):
                pass
            if reader.size != row['bytes'] or reader.hash.hexdigest() != row['sha256']:
                raise ValueError('Archive member SHA256 differs')
            seen.add(name)
        if seen != set(expected):
            raise ValueError('Incomplete archive')
    return plan


def summary(plan):
    sizes, counts = Counter(), Counter()
    for row in plan['files']:
        sizes[row['purpose']] += row['bytes']
        counts[row['purpose']] += 1
    for purpose in sorted(sizes):
        print(f'{purpose:24} {sizes[purpose]/2**30:8.3f} GiB  files={counts[purpose]}')
    print(f'Total payload: {plan["payload_bytes"]/2**30:.3f} GiB; manifest={plan["content_hash"]}')
    print('Transfer only. Original absolute paths are retained as evidence; Hydra training is not admitted.')
