"""Crash-safe immutable publication primitives for HCWDL-RKD artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from hlt_classification.data.cache_contracts import (
    canonical_json_bytes,
    load_json,
    require_sha256,
    sha256_bytes,
    sha256_file,
    validate_content_hash,
)

from .hcwdl_representation_contracts import (
    SHARED_BINARY_ENVELOPE_CONTRACT,
    build_versioned_artifact,
    derive_envelope_id,
    derive_envelope_owner_id,
    validate_parent_hashes,
    validate_versioned_artifact,
)


FailureHook = Callable[[str], None]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ).encode("utf-8") + b"\n"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe HCWDL-RKD member path {value!r}")
    return path.as_posix()


def _fsync_directory_windows(path: Path) -> None:
    # FlushFileBuffers on a directory handle is the Windows equivalent of
    # fsync(dirfd).  This is exercised locally; Tigris still has its separate
    # real-filesystem acceptance gate.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path), 0x80000000, 0x00000001 | 0x00000002, None, 3,
        0x02000000, None,  # FILE_FLAG_BACKUP_SEMANTICS
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), f"cannot open directory for flush: {path}")
    try:
        if not kernel32.FlushFileBuffers(handle):
            error = ctypes.get_last_error()
            # Some local Windows filesystems return ACCESS_DENIED even with a
            # valid directory handle.  The rename remains atomic, but the real
            # Tigris acceptance probe must establish durable directory fsync.
            if error not in {5, 87}:
                raise OSError(error, f"cannot flush directory: {path}")
    finally:
        kernel32.CloseHandle(handle)


def fsync_directory(path: str | Path) -> None:
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if os.name == "nt":
        _fsync_directory_windows(directory)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(path: str | Path) -> None:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    directories = [root]
    for current, child_directories, filenames in os.walk(root):
        here = Path(current)
        directories.extend(here / name for name in child_directories)
        for name in filenames:
            # Windows only accepts FlushFileBuffers for a writable handle;
            # opening an immutable staged member r+b does not mutate it.
            with (here / name).open("r+b") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(set(directories), key=lambda item: len(item.parts), reverse=True):
        fsync_directory(directory)


def write_staged_immutable_bytes(path: str | Path, data: bytes) -> str:
    """Write one owner-scoped staged member without replacing different bytes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise FileExistsError(f"conflicting staged HCWDL-RKD member: {destination}")
        return "reused"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if not destination.is_file() or sha256_file(destination) != digest:
                raise FileExistsError(f"staged HCWDL-RKD member race differs: {destination}")
        os.unlink(temporary_name)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            # A successful competing cleanup may already have removed the
            # owner-scoped temporary; there is no remaining artifact to act on.
            temporary_name = ""
        raise
    return "published"


def write_staged_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    return write_staged_immutable_bytes(path, _json_bytes(value))


def commit_staged_directory(
    staging: str | Path,
    destination: str | Path,
    *,
    validate_committed: Callable[[Path], Any],
    failure_hook: FailureHook | None = None,
) -> str:
    """Atomically expose a complete nonempty directory as one commit point."""

    source = Path(staging)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_committed(target)
        return "reused"
    if not source.is_dir() or not any(source.iterdir()):
        raise FileNotFoundError("HCWDL-RKD staging directory is absent or empty")
    fsync_tree(source)
    if failure_hook is not None:
        failure_hook("before_directory_rename")
    try:
        os.rename(source, target)
    except FileExistsError:
        validate_committed(target)
        return "reused"
    fsync_directory(target.parent)
    # Also persist removal of the source entry from its old parent.
    if source.parent.exists():
        fsync_directory(source.parent)
    if failure_hook is not None:
        failure_hook("after_directory_rename")
    validate_committed(target)
    return "published"


@dataclass(frozen=True)
class CommittedBinaryEnvelope:
    root: Path
    envelope_id: str
    owner_id: str
    sidecar: Mapping[str, Any]
    commit: Mapping[str, Any]

    @property
    def directory(self) -> Path:
        return self.root / "committed" / self.envelope_id


def _normalize_member_metadata(
    payloads: Mapping[str, bytes], member_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(payloads) != set(member_metadata) or not payloads:
        raise ValueError("HCWDL-RKD payload/member-metadata registries differ")
    result: dict[str, dict[str, Any]] = {}
    for raw_name in sorted(payloads):
        name = _safe_relative_path(str(raw_name))
        if name in result:
            raise ValueError("HCWDL-RKD payload member paths collide after normalization")
        if name in {"sidecar.json", "branch_access.json", "commit.json"}:
            raise ValueError("HCWDL-RKD payload uses a reserved member name")
        data = payloads[raw_name]
        if not isinstance(data, bytes):
            raise TypeError("HCWDL-RKD payload members must be bytes")
        metadata = dict(member_metadata[raw_name])
        if set(metadata) not in (
            {"logical_sha256"},
            {"logical_sha256", "dtype", "shape"},
        ):
            raise ValueError("HCWDL-RKD member metadata fields differ")
        metadata["logical_sha256"] = require_sha256(
            metadata.get("logical_sha256"), name=f"logical hash for {name}",
        )
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if (dtype is None) != (shape is None):
            raise ValueError("HCWDL-RKD member dtype/shape must appear together")
        if shape is not None:
            if (
                not isinstance(dtype, str) or not dtype
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                )
            ):
                raise ValueError("HCWDL-RKD member shape/dtype differs")
            metadata["shape"] = list(shape)
        result[name] = metadata
    return result


def _assert_committed_request_equals(
    envelope: CommittedBinaryEnvelope,
    *,
    desired_members: Mapping[str, bytes],
    expected_sidecar: Mapping[str, Any],
) -> None:
    """Reject a same-identity publication request with different bytes.

    ``envelope_id`` intentionally identifies the registered output row and its
    immutable parents, not bytes that do not exist when the task is
    registered.  Consequently an already committed envelope must be compared
    with the complete requested payload before it can be called idempotent.
    Path existence plus self-consistent old metadata is not enough.
    """

    expected = dict(desired_members)
    expected["sidecar.json"] = _json_bytes(expected_sidecar)
    actual_names = {
        path.relative_to(envelope.directory).as_posix()
        for path in envelope.directory.rglob("*")
        if path.is_file() and path.name != "commit.json"
    }
    if actual_names != set(expected):
        raise FileExistsError("committed HCWDL-RKD envelope member set differs")
    for name, data in expected.items():
        path = envelope.directory / Path(name)
        if not path.is_file() or sha256_file(path) != sha256_bytes(data):
            raise FileExistsError(
                f"committed HCWDL-RKD envelope request differs: {name}"
            )


def validate_binary_envelope(
    root: str | Path,
    envelope_id: str,
    *,
    expected_contract: str,
    expected_parents: Mapping[str, Any],
    expected_owner_id: str | None = None,
) -> CommittedBinaryEnvelope:
    base = Path(root)
    identity = require_sha256(envelope_id, name="HCWDL-RKD envelope ID")
    directory = base / "committed" / identity
    if not directory.is_dir():
        raise FileNotFoundError(f"committed HCWDL-RKD envelope is absent: {directory}")
    commit = load_json(directory / "commit.json")
    validate_versioned_artifact(
        commit, expected_contract=SHARED_BINARY_ENVELOPE_CONTRACT,
        expected_parents=expected_parents,
        required_payload_keys=(
            "envelope_id", "envelope_owner_id", "artifact_contract",
            "producer_task_id", "schema", "registered_output_row",
            "campaign_or_recovery_owner", "members",
        ),
    )
    payload = commit["payload"]
    if payload["envelope_id"] != identity or payload["artifact_contract"] != expected_contract:
        raise ValueError("HCWDL-RKD committed envelope identity/contract differs")
    if (
        not isinstance(payload["producer_task_id"], str)
        or not payload["producer_task_id"]
        or not isinstance(payload["schema"], Mapping)
        or not payload["schema"]
        or not isinstance(payload["registered_output_row"], Mapping)
        or not payload["registered_output_row"]
        or not isinstance(payload["campaign_or_recovery_owner"], Mapping)
        or not payload["campaign_or_recovery_owner"]
    ):
        raise ValueError("HCWDL-RKD committed envelope identity inputs differ")
    owner = require_sha256(payload["envelope_owner_id"], name="HCWDL-RKD envelope owner ID")
    if expected_owner_id is not None and owner != require_sha256(
        expected_owner_id, name="expected HCWDL-RKD owner ID",
    ):
        raise ValueError("HCWDL-RKD committed envelope owner differs")
    members = payload["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("HCWDL-RKD committed member registry is empty")
    expected_names = {"commit.json"}
    seen: set[str] = set()
    for record in members:
        if not isinstance(record, Mapping) or set(record) != {
            "path", "byte_sha256", "logical_sha256", "dtype", "shape",
            "immutable_parent_hashes",
        }:
            raise ValueError("HCWDL-RKD committed member row differs")
        name = _safe_relative_path(str(record.get("path", "")))
        if name in seen or name == "commit.json":
            raise ValueError("HCWDL-RKD committed member registry repeats a path")
        seen.add(name)
        path = directory / Path(name)
        if not path.is_file() or sha256_file(path) != require_sha256(
            record.get("byte_sha256"), name=f"byte hash for {name}",
        ):
            raise ValueError(f"HCWDL-RKD committed member bytes differ: {name}")
        require_sha256(record.get("logical_sha256"), name=f"logical hash for {name}")
        if record["immutable_parent_hashes"] != validate_parent_hashes(expected_parents):
            raise ValueError("HCWDL-RKD committed member parent lineage differs")
        if (record["dtype"] is None) != (record["shape"] is None):
            raise ValueError("HCWDL-RKD committed member dtype/shape differs")
        if record["shape"] is not None and (
            not isinstance(record["dtype"], str) or not record["dtype"]
            or not isinstance(record["shape"], list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in record["shape"]
            )
        ):
            raise ValueError("HCWDL-RKD committed member dtype/shape differs")
        expected_names.add(name)
    actual_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
    }
    if actual_names != expected_names:
        raise ValueError("HCWDL-RKD committed envelope contains unregistered members")
    if "sidecar.json" not in seen:
        raise ValueError("HCWDL-RKD committed envelope lacks its sidecar")
    sidecar = load_json(directory / "sidecar.json")
    validate_versioned_artifact(
        sidecar, expected_contract=expected_contract,
        expected_parents=expected_parents,
        required_payload_keys=(
            "envelope_id", "envelope_owner_id", "producer_task_id", "schema",
            "registered_output_row", "campaign_or_recovery_owner", "payload_members",
        ),
    )
    sidecar_payload = sidecar["payload"]
    identity_fields = (
        "envelope_id", "envelope_owner_id", "producer_task_id", "schema",
        "registered_output_row", "campaign_or_recovery_owner",
    )
    if any(sidecar_payload[name] != payload[name] for name in identity_fields):
        raise ValueError("HCWDL-RKD envelope identity inputs differ")
    derived_identity = derive_envelope_id(
        contract=expected_contract,
        producer_task_id=str(payload["producer_task_id"]),
        schema=payload["schema"],
        immutable_parent_hashes=expected_parents,
        registered_output_row=payload["registered_output_row"],
    )
    if derived_identity != identity:
        raise ValueError("HCWDL-RKD envelope identity is not derivable from its inputs")
    derived_owner = derive_envelope_owner_id(
        envelope_id=identity,
        campaign_or_recovery_owner=payload["campaign_or_recovery_owner"],
    )
    if derived_owner != owner:
        raise ValueError("HCWDL-RKD envelope owner is not derivable from its inputs")
    sidecar_record = next(row for row in members if row["path"] == "sidecar.json")
    if (
        sidecar_record["logical_sha256"] != sidecar["content_hash"]
        or sidecar_record["dtype"] is not None
        or sidecar_record["shape"] is not None
    ):
        raise ValueError("HCWDL-RKD sidecar logical hash differs")
    payload_metadata = sidecar_payload["payload_members"]
    if not isinstance(payload_metadata, Mapping):
        raise ValueError("HCWDL-RKD sidecar payload-member registry differs")
    auxiliary = {"sidecar.json"}
    if "branch_access.json" in seen:
        auxiliary.add("branch_access.json")
    if set(payload_metadata) != seen - auxiliary:
        raise ValueError("HCWDL-RKD sidecar payload-member registry differs")
    by_name = {_safe_relative_path(str(row["path"])): row for row in members}
    for name, raw_metadata in payload_metadata.items():
        if not isinstance(raw_metadata, Mapping) or set(raw_metadata) not in (
            {"logical_sha256"},
            {"logical_sha256", "dtype", "shape"},
        ):
            raise ValueError("HCWDL-RKD sidecar member metadata differs")
        metadata = dict(raw_metadata)
        logical = require_sha256(
            metadata.get("logical_sha256"), name=f"sidecar logical hash for {name}",
        )
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        if by_name[name]["logical_sha256"] != logical or (
            by_name[name]["dtype"], by_name[name]["shape"]
        ) != (dtype, shape):
            raise ValueError("HCWDL-RKD commit/sidecar member metadata differs")
    if "branch_access.json" in seen:
        branch_access = load_json(directory / "branch_access.json")
        contract = branch_access.get("contract")
        if not isinstance(contract, str) or not contract:
            raise ValueError("HCWDL-RKD branch-access contract differs")
        branch_hash = validate_content_hash(
            branch_access,
            expected_contract=contract,
            expected_schema_version=1,
        )
        branch_record = by_name["branch_access.json"]
        if (
            branch_record["logical_sha256"] != branch_hash
            or branch_record["dtype"] is not None
            or branch_record["shape"] is not None
            or validate_parent_hashes(expected_parents).get("branch_access") != branch_hash
        ):
            raise ValueError("HCWDL-RKD branch-access lineage differs")
    return CommittedBinaryEnvelope(base, identity, owner, sidecar, commit)


def publish_binary_envelope(
    root: str | Path,
    *,
    artifact_contract: str,
    producer_task_id: str,
    schema: Mapping[str, Any],
    immutable_parent_hashes: Mapping[str, Any],
    registered_output_row: Mapping[str, Any],
    campaign_or_recovery_owner: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    member_metadata: Mapping[str, Mapping[str, Any]],
    sidecar_payload: Mapping[str, Any],
    branch_access: Mapping[str, Any] | None = None,
    failure_hook: FailureHook | None = None,
) -> CommittedBinaryEnvelope:
    """Publish or idempotently validate a shared immutable binary envelope."""

    parents = validate_parent_hashes(immutable_parent_hashes)
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("HCWDL-RKD envelope schema is empty")
    if not isinstance(registered_output_row, Mapping) or not registered_output_row:
        raise ValueError("HCWDL-RKD registered output row is empty")
    if not isinstance(campaign_or_recovery_owner, Mapping) or not campaign_or_recovery_owner:
        raise ValueError("HCWDL-RKD envelope owner payload is empty")
    if branch_access is not None:
        if not isinstance(branch_access, Mapping):
            raise TypeError("HCWDL-RKD branch-access record must be a mapping")
        branch_contract = branch_access.get("contract")
        if not isinstance(branch_contract, str) or not branch_contract:
            raise ValueError("HCWDL-RKD branch-access contract differs")
        branch_hash = validate_content_hash(
            branch_access,
            expected_contract=branch_contract,
            expected_schema_version=1,
        )
        if parents.get("branch_access") != branch_hash:
            raise ValueError("HCWDL-RKD branch-access parent lineage differs")
    envelope_id = derive_envelope_id(
        contract=artifact_contract, producer_task_id=producer_task_id,
        schema=schema, immutable_parent_hashes=parents,
        registered_output_row=registered_output_row,
    )
    owner_id = derive_envelope_owner_id(
        envelope_id=envelope_id,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
    )
    normalized_metadata = _normalize_member_metadata(payloads, member_metadata)
    sidecar = build_versioned_artifact(
        artifact_contract, parents=parents,
        payload={
            **dict(sidecar_payload),
            "envelope_id": envelope_id,
            "envelope_owner_id": owner_id,
            "producer_task_id": producer_task_id,
            "schema": dict(schema),
            "registered_output_row": dict(registered_output_row),
            "campaign_or_recovery_owner": dict(campaign_or_recovery_owner),
            "payload_members": normalized_metadata,
        },
    )
    desired: dict[str, bytes] = {}
    for raw_name, data in payloads.items():
        name = _safe_relative_path(str(raw_name))
        if name in desired:
            raise ValueError("HCWDL-RKD payload member paths collide after normalization")
        desired[name] = data
    if branch_access is not None:
        desired["branch_access.json"] = _json_bytes(branch_access)
    base = Path(root)
    committed = base / "committed" / envelope_id
    if committed.exists():
        envelope = validate_binary_envelope(
            base, envelope_id, expected_contract=artifact_contract,
            expected_parents=parents, expected_owner_id=owner_id,
        )
        _assert_committed_request_equals(
            envelope, desired_members=desired, expected_sidecar=sidecar,
        )
        return envelope
    envelope_staging_root = base / "staging" / envelope_id
    if envelope_staging_root.exists():
        foreign = [path for path in envelope_staging_root.iterdir() if path.name != owner_id]
        if foreign:
            raise PermissionError("another owner has staged this HCWDL-RKD envelope")
    staging = envelope_staging_root / owner_id
    staging.mkdir(parents=True, exist_ok=True)
    desired["sidecar.json"] = _json_bytes(sidecar)
    member_rows = []
    for name in sorted(desired):
        data = desired[name]
        write_staged_immutable_bytes(staging / Path(name), data)
        if name in normalized_metadata:
            metadata = normalized_metadata[name]
            logical_hash = metadata["logical_sha256"]
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
        elif name == "sidecar.json":
            logical_hash = sidecar["content_hash"]
            dtype = shape = None
        else:
            parsed = json.loads(data.decode("utf-8"))
            logical_hash = require_sha256(
                parsed.get("content_hash"), name="branch-access content hash",
            )
            dtype = shape = None
        member_rows.append({
            "path": name,
            "byte_sha256": sha256_bytes(data),
            "logical_sha256": logical_hash,
            "dtype": dtype,
            "shape": shape,
            "immutable_parent_hashes": parents,
        })
        if failure_hook is not None:
            failure_hook(f"after_member:{name}")
    commit = build_versioned_artifact(
        SHARED_BINARY_ENVELOPE_CONTRACT, parents=parents,
        payload={
            "envelope_id": envelope_id,
            "envelope_owner_id": owner_id,
            "artifact_contract": artifact_contract,
            "producer_task_id": producer_task_id,
            "schema": dict(schema),
            "registered_output_row": dict(registered_output_row),
            "campaign_or_recovery_owner": dict(campaign_or_recovery_owner),
            "members": member_rows,
        },
    )
    write_staged_immutable_json(staging / "commit.json", commit)
    if failure_hook is not None:
        failure_hook("after_commit")

    def validate(path: Path) -> CommittedBinaryEnvelope:
        del path
        return validate_binary_envelope(
            base, envelope_id, expected_contract=artifact_contract,
            expected_parents=parents, expected_owner_id=owner_id,
        )

    commit_staged_directory(
        staging, committed, validate_committed=validate, failure_hook=failure_hook,
    )
    return validate(committed)


__all__ = [
    "CommittedBinaryEnvelope", "FailureHook", "commit_staged_directory",
    "fsync_directory", "fsync_tree", "publish_binary_envelope",
    "validate_binary_envelope", "write_staged_immutable_bytes",
    "write_staged_immutable_json",
]
