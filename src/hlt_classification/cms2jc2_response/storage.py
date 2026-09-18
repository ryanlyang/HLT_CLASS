"""Atomic publication under a shared cap that includes interrupted attempts."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time

from hlt_classification.data.cache_contracts import atomic_publish_bytes
from .contracts import validate

GIB = 1024**3
TOTAL_CAP = 12*GIB
REPORT_CAP = 2*GIB


def usage(root: Path):
    total, reports = 0, 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError("Response output roots may not contain symlink escape routes")
        if path.is_file():
            size = path.stat().st_size
            total += size
            if path.relative_to(root).parts[0] in {"reports", "figures", "examples"}:
                reports += size
    return total, reports


@contextmanager
def publication_lock(root: Path):
    root = root.resolve(strict=True)
    lock = root/".publication_lock"
    for attempt in range(100):
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if attempt == 99:
                raise RuntimeError("Publication lock is busy/stale; inspect exact writer, do not delete automatically")
            time.sleep(.05)
    try:
        yield
    finally:
        lock.rmdir()  # Exact empty lock created above; no recursive removal.


def publish_json(root: Path, relative: str, value: dict, kind: str, *, remaining_bytes: int):
    validate(value, kind)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    return publish_bytes(root, relative, data, remaining_bytes=remaining_bytes)


def publish_bytes(root: Path, relative: str, data: bytes, *, remaining_bytes: int):
    from .contracts import safe_relative
    root = root.resolve(strict=True)
    path = safe_relative(root, relative)
    if not isinstance(data, bytes):
        raise TypeError("Publication requires immutable bytes")
    if type(remaining_bytes) is not int or remaining_bytes < 0:
        raise ValueError("Storage plan must declare remaining writes")
    with publication_lock(root):
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Existing response output has different immutable content")
            return path
        total, reports = usage(root)
        if total+len(data) > TOTAL_CAP:
            raise OSError("CMS2JC2 campaign exceeds 12 GiB, including interrupted artifacts")
        is_report = relative.split("/")[0] in {"reports", "figures", "examples"}
        if is_report and reports+len(data) > REPORT_CAP:
            raise OSError("CMS2JC2 reports/examples exceed 2 GiB")
        if shutil.disk_usage(root).free < 2*max(remaining_bytes, len(data))+5*GIB:
            raise OSError("Insufficient headroom for declared remaining response writes")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_publish_bytes(path, data)
    return path
