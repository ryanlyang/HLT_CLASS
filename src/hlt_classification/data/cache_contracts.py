"""Versioned, deterministic, immutable cache shard and manifest contracts."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .schema import MAX_CONSTITUENTS, RAW_TOKEN_DIM

CACHE_SHARD_CONTRACT = "hlt_classification_cache_shard_v1"
OFFLINE_CACHE_MANIFEST_CONTRACT = "hlt_classification_offline_cache_manifest_v1"
HLT_CACHE_MANIFEST_CONTRACT = "hlt_classification_hlt_cache_manifest_v1"
CACHE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SHARD_DIRECTORY = "shards"
OFFLINE_ARRAY_NAMES = ("tokens", "mask", "labels", "identity_keys")
HLT_ARRAY_NAMES = (
    "tokens",
    "mask",
    "labels",
    "identity_keys",
    "measurement_states",
)


class CacheBuildInterrupted(RuntimeError):
    """Intentional interruption after durable shards, used to prove resume."""


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def with_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_hash", None)
    result["content_hash"] = canonical_sha256(result)
    return result


def validate_content_hash(
    payload: Mapping[str, Any],
    *,
    expected_contract: str,
) -> str:
    if payload.get("contract") != expected_contract:
        raise ValueError(
            f"contract mismatch: expected {expected_contract!r}, "
            f"got {payload.get('contract')!r}"
        )
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema version mismatch")
    supplied = require_sha256(payload.get("content_hash"), name="content_hash")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    calculated = canonical_sha256(unhashed)
    if supplied != calculated:
        raise ValueError("cache content hash mismatch")
    return supplied


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_files_sha256(files: Mapping[str, str | Path]) -> str:
    digest = hashlib.sha256()
    for logical_name in sorted(files):
        path = Path(files[logical_name])
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def array_sha256(name: str, array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def identity_order_sha256_iter(identity_keys: Any) -> str:
    """Hash a canonical JSON string list without retaining the whole list."""

    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for raw_value in identity_keys:
        if not first:
            digest.update(b",")
        digest.update(
            json.dumps(
                str(raw_value),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def identity_order_sha256(identity_keys: Sequence[str]) -> str:
    return identity_order_sha256_iter(identity_keys)


def identity_key_array(identity_keys: Sequence[str]) -> np.ndarray:
    keys = [str(value) for value in identity_keys]
    width = max((len(value) for value in keys), default=1)
    return np.asarray(keys, dtype=f"<U{width}")


def cache_spec_sha256(payload: Mapping[str, Any]) -> str:
    """Hash immutable build intent, excluding produced artifacts and this hash."""

    specification = dict(payload)
    for key in (
        "aggregate_diagnostics",
        "cache_spec_sha256",
        "content_hash",
        "shards",
    ):
        specification.pop(key, None)
    return canonical_sha256(specification)


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize arrays with stable names, NPY headers, ZIP metadata, and order."""

    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"unsafe array name {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"object array {name!r} is forbidden")
            npy = BytesIO()
            np.lib.format.write_array(npy, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, npy.getvalue())
    return output.getvalue()


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def atomic_publish_bytes(path: str | Path, data: bytes) -> str:
    """Publish complete bytes without ever replacing different existing bytes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise FileExistsError(
                f"immutable artifact already exists with different bytes: {destination}"
            )
        return "reused"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(handle, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if sha256_file(destination) != digest:
                raise FileExistsError(
                    f"immutable artifact race produced different bytes: {destination}"
                )
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return "published"


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return atomic_publish_bytes(path, serialized)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _expected_array_names(cache_kind: str) -> tuple[str, ...]:
    if cache_kind == "offline":
        return OFFLINE_ARRAY_NAMES
    if cache_kind == "hlt":
        return HLT_ARRAY_NAMES
    raise ValueError(f"unknown cache kind {cache_kind!r}")


def validate_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    cache_kind: str,
    expected_rows: int | None = None,
    expected_identity_keys: Sequence[str] | None = None,
) -> None:
    expected_names = _expected_array_names(cache_kind)
    if set(arrays) != set(expected_names):
        raise ValueError(
            f"{cache_kind} shard arrays differ: "
            f"expected={sorted(expected_names)}, actual={sorted(arrays)}"
        )
    tokens = np.asarray(arrays["tokens"])
    mask = np.asarray(arrays["mask"])
    labels = np.asarray(arrays["labels"])
    identities = np.asarray(arrays["identity_keys"])
    if (
        tokens.ndim != 3
        or tokens.shape[1:] != (MAX_CONSTITUENTS, RAW_TOKEN_DIM)
        or tokens.dtype != np.float32
    ):
        raise ValueError(
            "cache tokens must be float32 "
            f"[rows,{MAX_CONSTITUENTS},{RAW_TOKEN_DIM}]"
        )
    rows, particles = tokens.shape[:2]
    if mask.shape != (rows, particles) or mask.dtype != np.bool_:
        raise ValueError("cache mask shape or dtype differs")
    if labels.shape != (rows,) or labels.dtype != np.int64:
        raise ValueError("cache labels shape or dtype differs")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("cache label lies outside 0..9")
    if identities.shape != (rows,) or identities.dtype.kind != "U":
        raise ValueError("cache identity keys must be a one-dimensional Unicode array")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"cache row count {rows} differs from {expected_rows}")
    if not bool(np.isfinite(tokens).all()):
        raise ValueError("cache tokens contain nonfinite values")
    if np.any(tokens[~mask] != 0):
        raise ValueError("cache padding tokens are nonzero")
    keys = [str(value) for value in identities.tolist()]
    if len(keys) != len(set(keys)):
        raise ValueError("cache shard contains duplicate identities")
    if expected_identity_keys is not None and keys != list(expected_identity_keys):
        raise ValueError("cache identity order differs")
    if cache_kind == "hlt":
        states = np.asarray(arrays["measurement_states"])
        if states.shape != (rows, particles) or states.dtype != np.int8:
            raise ValueError("HLT measurement states shape or dtype differs")
        if np.any((states < 0) | (states > 2)):
            raise ValueError("HLT measurement state lies outside 0..2")
        if np.any(states[~mask] != 0):
            raise ValueError("HLT padding measurement states are nonzero")


def build_shard_record(
    *,
    cache_kind: str,
    logical_role: str,
    shard_index: int,
    row_start: int,
    row_stop: int,
    shard_filename: str,
    shard_file_sha256: str,
    arrays: Mapping[str, np.ndarray],
    lineage: Mapping[str, Any],
    parents: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if shard_index < 0 or row_start < 0 or row_stop < row_start:
        raise ValueError("invalid cache shard range")
    validate_cache_arrays(
        arrays,
        cache_kind=cache_kind,
        expected_rows=row_stop - row_start,
    )
    identities = [str(value) for value in arrays["identity_keys"].tolist()]
    return with_content_hash(
        {
            "contract": CACHE_SHARD_CONTRACT,
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": cache_kind,
            "logical_role": logical_role,
            "shard_index": shard_index,
            "row_start": row_start,
            "row_stop": row_stop,
            "row_count": row_stop - row_start,
            "shard_filename": shard_filename,
            "shard_file_sha256": require_sha256(
                shard_file_sha256, name="shard_file_sha256"
            ),
            "identity_order_sha256": identity_order_sha256(identities),
            "array_sha256": {
                name: array_sha256(name, arrays[name])
                for name in sorted(arrays)
            },
            "array_shapes": {
                name: list(np.asarray(arrays[name]).shape)
                for name in sorted(arrays)
            },
            "array_dtypes": {
                name: np.asarray(arrays[name]).dtype.str
                for name in sorted(arrays)
            },
            "lineage": json.loads(json.dumps(lineage, allow_nan=False)),
            "parents": (
                {}
                if parents is None
                else json.loads(json.dumps(parents, allow_nan=False))
            ),
            "diagnostics": (
                {} if diagnostics is None else json.loads(
                    json.dumps(diagnostics, allow_nan=False)
                )
            ),
        }
    )


def shard_paths(
    cache_root: str | Path,
    shard_index: int,
) -> tuple[Path, Path]:
    if shard_index < 0:
        raise ValueError("shard index must be nonnegative")
    base = Path(cache_root) / SHARD_DIRECTORY / f"shard_{shard_index:06d}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def load_completed_shard_record(
    cache_root: str | Path,
    shard_index: int,
) -> dict[str, Any] | None:
    """Load an authenticated sidecar and attach its own artifact digest."""

    _, metadata_path = shard_paths(cache_root, shard_index)
    if not metadata_path.exists():
        return None
    record = load_json(metadata_path)
    validate_content_hash(record, expected_contract=CACHE_SHARD_CONTRACT)
    if int(record.get("shard_index", -1)) != shard_index:
        raise ValueError("cache shard sidecar index differs from its filename")
    record["metadata_filename"] = metadata_path.relative_to(cache_root).as_posix()
    record["metadata_file_sha256"] = sha256_file(metadata_path)
    return record


def publish_shard(
    *,
    cache_root: str | Path,
    cache_kind: str,
    logical_role: str,
    shard_index: int,
    row_start: int,
    row_stop: int,
    arrays: Mapping[str, np.ndarray],
    lineage: Mapping[str, Any],
    parents: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Atomically publish a deterministic NPZ followed by its authenticated sidecar."""

    validate_cache_arrays(
        arrays,
        cache_kind=cache_kind,
        expected_rows=row_stop - row_start,
    )
    shard_path, metadata_path = shard_paths(cache_root, shard_index)
    shard_bytes = deterministic_npz_bytes(arrays)
    shard_status = atomic_publish_bytes(shard_path, shard_bytes)
    record = build_shard_record(
        cache_kind=cache_kind,
        logical_role=logical_role,
        shard_index=shard_index,
        row_start=row_start,
        row_stop=row_stop,
        shard_filename=shard_path.relative_to(cache_root).as_posix(),
        shard_file_sha256=sha256_bytes(shard_bytes),
        arrays=arrays,
        lineage=lineage,
        parents=parents,
        diagnostics=diagnostics,
    )
    metadata_status = write_immutable_json(metadata_path, record)
    record["metadata_filename"] = metadata_path.relative_to(cache_root).as_posix()
    record["metadata_file_sha256"] = sha256_file(metadata_path)
    status = (
        "published"
        if "published" in (shard_status, metadata_status)
        else "reused"
    )
    return record, status


def validate_shard_record(
    record: Mapping[str, Any],
    *,
    cache_root: str | Path,
    expected_cache_kind: str,
    expected_role: str,
    expected_lineage: Mapping[str, Any],
    expected_identity_keys: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    semantic_record = dict(record)
    semantic_record.pop("metadata_filename", None)
    semantic_record.pop("metadata_file_sha256", None)
    validate_content_hash(semantic_record, expected_contract=CACHE_SHARD_CONTRACT)
    if record.get("cache_kind") != expected_cache_kind:
        raise ValueError("cache shard kind differs")
    if record.get("logical_role") != expected_role:
        raise ValueError("cache shard logical role differs")
    if record.get("lineage") != dict(expected_lineage):
        raise ValueError("cache shard lineage differs")
    root = Path(cache_root).resolve()
    metadata_filename = record.get("metadata_filename")
    metadata_hash = record.get("metadata_file_sha256")
    if (metadata_filename is None) != (metadata_hash is None):
        raise ValueError("cache shard metadata lineage is incomplete")
    if metadata_filename is not None:
        metadata_path = (root / str(metadata_filename)).resolve()
        try:
            metadata_path.relative_to(root)
        except ValueError as error:
            raise ValueError("cache shard metadata path escapes cache root") from error
        if (
            not metadata_path.is_file()
            or sha256_file(metadata_path)
            != require_sha256(metadata_hash, name="metadata_file_sha256")
        ):
            raise ValueError("cache shard sidecar is absent or corrupt")
        metadata_record = load_json(metadata_path)
        if metadata_record != semantic_record:
            raise ValueError("cache shard manifest entry differs from its sidecar")
    shard_path = (root / str(record["shard_filename"])).resolve()
    try:
        shard_path.relative_to(root)
    except ValueError as error:
        raise ValueError("cache shard path escapes cache root") from error
    expected_file_hash = require_sha256(
        record.get("shard_file_sha256"), name="shard_file_sha256"
    )
    if not shard_path.is_file() or sha256_file(shard_path) != expected_file_hash:
        raise ValueError(f"cache shard file is absent or corrupt: {shard_path}")
    arrays = load_npz_arrays(shard_path)
    validate_cache_arrays(
        arrays,
        cache_kind=expected_cache_kind,
        expected_rows=int(record["row_count"]),
        expected_identity_keys=expected_identity_keys,
    )
    actual_identity_keys = [
        str(value) for value in arrays["identity_keys"].tolist()
    ]
    for name, array in arrays.items():
        if record["array_sha256"].get(name) != array_sha256(name, array):
            raise ValueError(f"cache shard array hash differs for {name}")
        if record["array_shapes"].get(name) != list(array.shape):
            raise ValueError(f"cache shard array shape differs for {name}")
        if record["array_dtypes"].get(name) != array.dtype.str:
            raise ValueError(f"cache shard array dtype differs for {name}")
    if record.get("identity_order_sha256") != identity_order_sha256(
        actual_identity_keys
    ):
        raise ValueError("cache shard identity hash differs")
    return arrays


def build_cache_manifest(
    *,
    cache_kind: str,
    logical_role: str,
    shard_size: int,
    total_rows: int,
    identity_keys: Sequence[str] | None,
    identity_order_digest: str | None = None,
    lineage: Mapping[str, Any],
    shard_records: Sequence[Mapping[str, Any]],
    aggregate_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if shard_size <= 0 or total_rows < 0:
        raise ValueError("invalid cache manifest sizes")
    contract = (
        OFFLINE_CACHE_MANIFEST_CONTRACT
        if cache_kind == "offline"
        else HLT_CACHE_MANIFEST_CONTRACT
        if cache_kind == "hlt"
        else None
    )
    if contract is None:
        raise ValueError(f"unknown cache kind {cache_kind!r}")
    records = [dict(record) for record in shard_records]
    expected_start = 0
    for index, record in enumerate(records):
        semantic_record = dict(record)
        semantic_record.pop("metadata_filename", None)
        semantic_record.pop("metadata_file_sha256", None)
        validate_content_hash(semantic_record, expected_contract=CACHE_SHARD_CONTRACT)
        if int(record["shard_index"]) != index:
            raise ValueError("cache shard indices are not contiguous")
        if int(record["row_start"]) != expected_start:
            raise ValueError("cache shard row ranges are not contiguous")
        expected_start = int(record["row_stop"])
    if expected_start != total_rows:
        raise ValueError("cache shard ranges do not cover the population")
    if identity_keys is not None:
        if len(identity_keys) != total_rows or len(set(identity_keys)) != total_rows:
            raise ValueError("cache manifest identities are incomplete or duplicated")
        identity_digest = identity_order_sha256(identity_keys)
    else:
        identity_digest = require_sha256(
            identity_order_digest, name="identity_order_digest"
        )
    return with_content_hash(
        {
            "contract": contract,
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": cache_kind,
            "logical_role": logical_role,
            "shard_size": shard_size,
            "total_rows": total_rows,
            "shard_count": len(records),
            "array_names": list(_expected_array_names(cache_kind)),
            "identity_order_sha256": identity_digest,
            "lineage": json.loads(json.dumps(lineage, allow_nan=False)),
            "shards": records,
            "aggregate_diagnostics": (
                {}
                if aggregate_diagnostics is None
                else json.loads(json.dumps(aggregate_diagnostics, allow_nan=False))
            ),
        }
    )


def validate_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    cache_root: str | Path,
    expected_cache_kind: str,
    expected_role: str | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
    expected_identity_keys: Sequence[str] | None = None,
    validate_shards: bool = True,
) -> str:
    contract = (
        OFFLINE_CACHE_MANIFEST_CONTRACT
        if expected_cache_kind == "offline"
        else HLT_CACHE_MANIFEST_CONTRACT
    )
    digest = validate_content_hash(manifest, expected_contract=contract)
    if manifest.get("cache_kind") != expected_cache_kind:
        raise ValueError("cache manifest kind differs")
    if manifest.get("array_names") != list(_expected_array_names(expected_cache_kind)):
        raise ValueError("cache manifest array schema differs")
    if int(manifest.get("shard_size", 0)) <= 0:
        raise ValueError("cache manifest shard size must be positive")
    role = str(manifest.get("logical_role"))
    if expected_role is not None and role != expected_role:
        raise ValueError("cache manifest logical role differs")
    lineage = dict(manifest.get("lineage", {}))
    if expected_lineage is not None and lineage != dict(expected_lineage):
        raise ValueError("cache manifest lineage differs")
    total_rows = int(manifest["total_rows"])
    if expected_identity_keys is not None:
        keys = list(expected_identity_keys)
        if len(keys) != total_rows:
            raise ValueError("cache manifest population size differs")
        if manifest.get("identity_order_sha256") != identity_order_sha256(keys):
            raise ValueError("cache manifest identity order differs")
    else:
        keys = []
    streamed_identity_digest = hashlib.sha256()
    streamed_identity_digest.update(b"[")
    streamed_identity_first = True
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != int(manifest["shard_count"]):
        raise ValueError("cache manifest shard registry differs")
    expected_start = 0
    for index, record in enumerate(records):
        if int(record["shard_index"]) != index:
            raise ValueError("cache manifest shard indices differ")
        start, stop = int(record["row_start"]), int(record["row_stop"])
        if start != expected_start or stop < start:
            raise ValueError("cache manifest shard ranges differ")
        if int(record.get("row_count", -1)) != stop - start:
            raise ValueError("cache manifest shard row count differs")
        if stop - start > int(manifest["shard_size"]):
            raise ValueError("cache manifest shard exceeds its bounded size")
        expected_start = stop
        if validate_shards:
            arrays = validate_shard_record(
                record,
                cache_root=cache_root,
                expected_cache_kind=expected_cache_kind,
                expected_role=role,
                expected_lineage=lineage,
                expected_identity_keys=(
                    keys[start:stop] if expected_identity_keys is not None else None
                ),
            )
            if expected_identity_keys is None:
                for value in arrays["identity_keys"].tolist():
                    if not streamed_identity_first:
                        streamed_identity_digest.update(b",")
                    streamed_identity_digest.update(
                        json.dumps(
                            str(value),
                            ensure_ascii=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    streamed_identity_first = False
    if expected_start != total_rows:
        raise ValueError("cache manifest does not cover every row")
    if validate_shards:
        if expected_identity_keys is None:
            streamed_identity_digest.update(b"]")
            actual_identity_digest = streamed_identity_digest.hexdigest()
        else:
            actual_identity_digest = identity_order_sha256(keys)
        if manifest.get("identity_order_sha256") != actual_identity_digest:
            raise ValueError("cache manifest global identity order differs")
    return digest


__all__ = [
    "CACHE_SHARD_CONTRACT",
    "CacheBuildInterrupted",
    "HLT_CACHE_MANIFEST_CONTRACT",
    "OFFLINE_CACHE_MANIFEST_CONTRACT",
    "array_sha256",
    "atomic_publish_bytes",
    "build_cache_manifest",
    "build_shard_record",
    "cache_spec_sha256",
    "canonical_sha256",
    "deterministic_npz_bytes",
    "identity_order_sha256",
    "identity_order_sha256_iter",
    "identity_key_array",
    "MANIFEST_FILENAME",
    "load_json",
    "load_completed_shard_record",
    "load_npz_arrays",
    "publish_shard",
    "require_sha256",
    "sha256_file",
    "source_files_sha256",
    "shard_paths",
    "validate_cache_arrays",
    "validate_cache_manifest",
    "validate_shard_record",
    "with_content_hash",
    "write_immutable_json",
]
