"""Explicit file-level retirement; never recursively remove user directories."""
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact, validate


def strict_path(base, relative):
    base = Path(base)
    if '\\' in str(relative) or ':' in str(relative):
        raise PermissionError('Retirement paths use portable relative components only')
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or any(x in ('..', '.') for x in relative.parts):
        raise PermissionError('Retirement path is not an exact relative file')
    path = base.joinpath(*relative.parts)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise PermissionError('No symlink traversal for owned artifacts')
    if not path.resolve().is_relative_to(base.resolve()):
        raise PermissionError('Retirement path escapes ownership')
    return path


def create_owned(path, owner):
    path = Path(path)
    # Validate ancestors before creation, including Windows junction resolution.
    if path.absolute() != path.resolve():
        raise PermissionError('Owned directory must have an exact canonical path')
    path.mkdir(parents=True, exist_ok=False)
    record = artifact('OWNER', parents=dict(owner=owner), root=str(path.resolve()))
    write_immutable_json(path/'owner.json', record)
    return path


def scratch(gate, job):
    base = Path(gate['scratch_base'])
    if not base.is_dir() or base.absolute() != base.resolve():
        raise PermissionError('Explicit canonical scratch base is unavailable')
    if os.name == 'posix':
        # Linux mount identity check: reject NFS/Ceph/other shared persistent storage.
        mounts = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            mount = Path(left.split()[4].replace('\\040', ' '))
            if base.is_relative_to(mount):
                mounts.append((len(mount.parts), right.split()[0]))
        if not mounts or max(mounts)[1] not in ('ext4', 'xfs', 'tmpfs', 'btrfs', 'overlay'):
            raise PermissionError('Bank scratch must be an explicitly node-local filesystem')
    path = Path(tempfile.mkdtemp(prefix='luka-swap-'+str(job)+'-', dir=base))
    # mkdtemp already made the exact private directory; no fallback to home.
    write_immutable_json(path/'owner.json', artifact('OWNER',
        parents=dict(owner=gate['content_hash']), root=str(path)))
    return path


def budget(lengths, persistent, scratch_root, quota_gib):
    rows = len(lengths)
    bank_bytes = 4*sum(int(n)**2 for n in lengths)
    # One bank + probability/identity payloads, publication buffers/checkpoint,
    # measured model state well below the conservative 1GiB allowance, 2GiB reserve.
    scratch_required = bank_bytes + rows*128 + 3*2**30
    persistent_required = 2**30  # compact six technical/science prediction bundles + reserve
    free_scratch = shutil.disk_usage(scratch_root).free
    free_persistent = shutil.disk_usage(persistent).free
    if scratch_required > free_scratch or persistent_required > min(free_persistent, quota_gib*2**30):
        raise OSError('Insufficient declared persistent or node-local scratch storage; no fallback')
    return dict(attention_bank_bytes=bank_bytes, scratch_required_bytes=scratch_required,
        persistent_required_bytes=persistent_required, scratch_free_bytes=free_scratch,
        persistent_free_bytes=free_persistent, declared_quota_gib=quota_gib)


def retire(base, refs, *, owner, consumer, output):
    """Consumer is an already authenticated durable receipt, not a path-exists flag."""
    base, output = Path(base), Path(output)
    owned = load_json(base/'owner.json')
    validate(owned, 'OWNER', dict(owner=owner))
    if owned['root'] != str(base.resolve()):
        raise PermissionError('Ownership root differs')
    validate_file_ref(consumer)
    if not refs or len({r['path'] for r in refs}) != len(refs):
        raise ValueError('Empty/duplicate retirement inventory')
    paths = []
    for ref in refs:
        path = strict_path(base, ref['path'])
        if path.name == 'owner.json' or not path.is_file() or path.stat().st_nlink != 1:
            raise PermissionError('Retirement requires exclusively owned regular payloads')
        validate_file_ref(ref, root=base)
        paths.append(path)
    output.mkdir(parents=True, exist_ok=False)
    intent = artifact('RETIREMENT_INTENT', parents=dict(owner=owner, consumer=consumer['sha256']),
        base=str(base.resolve()), files=refs, consumer=consumer,
        recovery='re-export or retrain; no trash copy')
    write_immutable_json(output/'intent.json', intent)
    # Validate the entire inventory before ANY unlink; no recursive deletion.
    validate_file_ref(consumer)
    for ref, path in zip(refs, paths):
        strict_path(base, ref['path'])
        validate_file_ref(ref, root=base)
        path.unlink()
    result = artifact('RETIREMENT', parents=dict(intent=intent['content_hash']),
        deleted_bytes=sum(r['bytes'] for r in refs), deleted_files=len(refs), complete=True)
    write_immutable_json(output/'receipt.json', result)
    print(f'Retired {len(refs)} owned files ({result["deleted_bytes"]/2**30:.3f} GiB); no trash backup.', flush=True)
    return result


def validate_retirement(output, owner):
    output = Path(output)
    intent, result = load_json(output/'intent.json'), load_json(output/'receipt.json')
    validate(intent, 'RETIREMENT_INTENT', dict(owner=owner, consumer=intent['consumer']['sha256']))
    validate_file_ref(intent['consumer'])
    validate(result, 'RETIREMENT', dict(intent=intent['content_hash']))
    refs = intent['files']
    if (not refs or len({r['path'] for r in refs}) != len(refs) or result['complete'] is not True
            or result['deleted_files'] != len(refs) or result['deleted_bytes'] != sum(r['bytes'] for r in refs)):
        raise ValueError('Incomplete retirement')
    for ref in refs:
        if strict_path(Path(intent['base']), ref['path']).exists():
            raise ValueError('Retired payload reappeared')
    return result


def bank_refs(root, manifest):
    refs = [manifest['population']]+[s['payload'] for s in manifest['shards']]
    expected = {r['path'] for r in refs} | {'manifest.json'}
    root = Path(root)
    actual = {p.name for p in root.iterdir()}
    if actual != expected:
        raise ValueError('Unknown bank files; refuse cleanup')
    return refs+[file_ref(root/'manifest.json', root=root)]
