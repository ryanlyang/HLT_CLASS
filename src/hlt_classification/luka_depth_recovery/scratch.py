"""Linux node-local conservative reservations for concurrent token exporters."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import stat


def budget(free, reserved, required):
    if min(free, reserved) < 0 or required <= 0 or free - reserved < required:
        raise OSError('Insufficient unreserved node-local token scratch; no home fallback')


@contextmanager
def reserve(base, required):
    import fcntl  # Production is Linux. No silent cross-platform no-lock fallback.
    base = Path(base)
    if base != Path('/tmp') or base.resolve(strict=True) != base:
        raise PermissionError('Only canonical node-local /tmp is allowed')
    directory = base/('lukasr-reservations-' + str(os.getuid()))
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError('Scratch reservation directory is not private/owned')
    path = directory/'leases.json'; key = str(os.getpid())

    def change(add):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'r+', encoding='utf-8') as handle:
            meta = os.fstat(handle.fileno())
            if meta.st_uid != os.getuid() or meta.st_nlink != 1 or not stat.S_ISREG(meta.st_mode):
                raise PermissionError('Unowned scratch lease file')
            fcntl.flock(handle, fcntl.LOCK_EX)
            text = handle.read(); leases = json.loads(text) if text else {}
            if not isinstance(leases, dict):
                raise ValueError('Invalid lease metadata; inspect before retry')
            alive = {}
            for pid, size in leases.items():
                if not pid.isdigit() or int(pid) <= 0 or type(size) != int or size <= 0:
                    raise ValueError('Invalid scratch lease')
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    continue
                alive[pid] = size
            if add:
                if key in alive:
                    raise ValueError('Duplicate scratch lease')
                # Conservative: existing payloads already reduce free space;
                # charging their entire reservations again cannot overbook.
                budget(shutil.disk_usage(base).free, sum(alive.values()), required)
                alive[key] = required
            else:
                alive.pop(key, None)
            handle.seek(0); handle.truncate(); json.dump(alive, handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
    change(True)
    try:
        yield
    finally:
        change(False)
