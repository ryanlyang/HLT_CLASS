"""Frozen v1 metadata semantics without the training packages' eager imports.

Adapted from cache_contracts, cms_proxy_ladder/contracts, FullStat contracts
and the Delphes label map at 1fc128dca7dc7ef547d242a1e2941e9a74e25ac8.
Parity tests bind these small adapters to the original implementations.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

CLASS_NAMES = ("QCD", "X_bb", "X_cc", "X_ss", "X_qq", "X_gg", "X_ee", "X_mm",
               "X_tauhtaue", "X_tauhtaum", "X_tauhtauh")
SOURCE = "ryang@lxplus.cern.ch:/eos/user/l/llambrec/jetclass/output_20261015_fullsim_fullstat/merged"
COUNTS = dict(train=3000000, validation=500000, final_test=1500000, unused=105325)
ROLES = tuple(COUNTS)


def split_policy():
    return dict(seed=20261016, block_rows=2048,
        algorithm="shared_block_hash_order_stratum_exact_quotas_v1",
        strata="signal_class_or_QCD_source_bin", boundary_blocks_may_split=True,
        event_independence_verified=False, old_trial_overlap_verified=False,
        additional_pt_cut=None, final_test_accessed=False)


def map_labels(raw):
    raw = np.asarray(raw)
    mapping = {n:i+1 for i,n in enumerate((0,1,2,3,9,10,11,12,13,14))}
    mapping.update({n:0 for n in range(161,188)})
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or set(np.unique(raw))-mapping.keys():
        raise ValueError("Invalid/unregistered raw TRAIN label")
    out = np.empty(len(raw), np.int64)
    for n in np.unique(raw):
        out[raw == n] = mapping[n]
    return out


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def with_content_hash(value):
    body = {k:v for k,v in value.items() if k != "content_hash"}
    return {**body, "content_hash": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def validate_content_hash(value, *, expected_contract, expected_schema_version=1):
    if (not isinstance(expected_contract, str) or value.get("contract") != expected_contract
            or value.get("schema_version") != expected_schema_version
            or value.get("content_hash") != with_content_hash(value)["content_hash"]):
        raise ValueError("Metadata contract or content hash differs")
    return value["content_hash"]


def load_json(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path, *, root=None):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path) if root is None else path.relative_to(Path(root).resolve()).as_posix(),
                sha256=sha256_file(path), bytes=path.stat().st_size)


def validate_file_ref(ref, *, root=None):
    if root is None:
        path = Path(ref["path"])
        if not path.is_absolute():
            raise ValueError("Expected absolute file reference")
    else:
        relative = ref["path"]
        if any(p in {"", ".", ".."} for p in relative.split("/")) or ":" in relative or "\\" in relative:
            raise ValueError("Unsafe relative file reference")
        path = Path(root).resolve()
        for part in relative.split("/"):
            path /= part
            if path.is_symlink():
                raise ValueError("Symlink file reference")
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("File reference escaped root")
    if path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
        raise ValueError(f"File bytes differ: {path}")
    return path


def array_sha256(name, array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    for part in (name.encode(), b"\0", value.dtype.str.encode("ascii"), b"\0",
                 canonical_bytes(list(value.shape)), b"\0", value.tobytes(order="C")):
        digest.update(part)
    return digest.hexdigest()


def atomic_publish_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # no overwrite; completion marker is published last
    finally:
        os.unlink(temp)


def write_immutable_json(path, value):
    atomic_publish_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
