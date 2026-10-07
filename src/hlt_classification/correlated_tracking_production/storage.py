"""Bounded durable output; conservative reservations survive failed attempts."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import time

from .contracts import GIB, artifact, validate, load_json, write, safe, atomic_publish_bytes

METADATA_ALLOWANCE = 2*GIB
FREE_HEADROOM = 2*GIB


def usage(root):
    total = 0
    for path in Path(root).rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in production root')
        if path.is_file():
            total += path.stat().st_size
    return total


def check_space(root, required=0):
    if shutil.disk_usage(root).free < required+FREE_HEADROOM:
        raise OSError('Insufficient free filesystem headroom; no files deleted')


@contextmanager
def lock(root, name='publication'):
    directory = safe(root, f'locks/{name}.active')
    directory.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(600):
        try:
            directory.mkdir()
            break
        except FileExistsError:
            time.sleep(.1)
    else:
        raise RuntimeError('Publication lock busy/stale; inspect owner before manual recovery')
    try:
        yield
    finally:
        directory.rmdir()


def reserve(study, attempt, shard_id, allowance):
    if type(allowance) is not int or allowance <= 0:
        raise ValueError('Positive integer byte reservation required')
    root = Path(study['root'])
    relative = f'attempts/{attempt}/shards/{shard_id}'
    value = artifact('RESERVATION', parents={'study': study['content_hash']},
        relative=relative, allowance=int(allowance))
    path = safe(root, f'reservations/{attempt}_{shard_id}.json')
    with lock(root):
        rows = [load_json(p) for p in safe(root, 'reservations').glob('*.json')]
        for row in rows:
            validate(row, 'RESERVATION', parents={'study': study['content_hash']}, test=False)
            if type(row['allowance']) is not int or row['allowance'] <= 0:
                raise ValueError('Invalid saved byte reservation')
        extra = 0 if path.exists() and load_json(path) == value else allowance
        if sum(r['allowance'] for r in rows)+extra+METADATA_ALLOWANCE > study['storage']['budget_bytes']:
            raise OSError('Output reservation budget exhausted; failed attempts remain preserved')
        check_space(root, allowance)
        write(path, value)
    return value


def publish_block(study, reservation, filename, blob):
    root = Path(study['root'])
    validate(reservation, 'RESERVATION', parents={'study': study['content_hash']}, test=False)
    directory = safe(root, reservation['relative'])
    path = safe(directory, filename)
    if usage(directory)+(0 if path.exists() else len(blob)) > reservation['allowance']:
        raise OSError('Shard output exceeds its reserved allowance')
    check_space(root, len(blob))
    atomic_publish_bytes(path, blob)
    return path
