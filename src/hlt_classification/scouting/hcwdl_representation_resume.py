"""Crash-safe immutable resume generations for HCWDL representation training.

Generation ``q`` consists of ``state_q.pt``, ``state_q.json``, and
``commit_q.json``.  The state and sidecar may exist after a crash, but only the
last atomically published commit makes a generation eligible for resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_bytes,
    sha256_file,
)

from .hcwdl_representation_contracts import (
    REPRESENTATION_RESUME_STATE_CONTRACT,
    build_versioned_artifact,
    logical_array_sha256_from_byte_hash,
    validate_parent_hashes,
    validate_versioned_artifact,
)
from .hcwdl_representation_graph_registry import validate_registered_graph_sha256


RESUME_STATE_CONTRACT: Final = REPRESENTATION_RESUME_STATE_CONTRACT
RESUME_SCHEMA_VERSION: Final = 1

REQUIRED_STATE_NAMESPACES: Final = frozenset({
    "calibration",
    "cursor",
    "deployable_model",
    "interval_aggregates",
    "model_runtime",
    "optimizer",
    "representation_heads",
    "rng",
    "rng_streams",
    "sampler",
    "scheduler",
    "selection_state",
    "target_bindings",
    "trimmer",
    "validation_history",
    "producer_runtime_signature",
})
REQUIRED_LINEAGE_KEYS: Final = frozenset({
    "ascent_graph",
    "execution",
    "producer_runtime_signature",
    "representation_recipe",
    "target_generation",
    "target_logical",
})

_STATE_PATTERN = re.compile(r"^state_(0|[1-9][0-9]*)\.pt$")
_SIDECAR_PATTERN = re.compile(r"^state_(0|[1-9][0-9]*)\.json$")
_COMMIT_PATTERN = re.compile(r"^commit_(0|[1-9][0-9]*)\.json$")
_CRASH_POINTS: Final = frozenset({
    "before_state",
    "after_state",
    "after_sidecar",
    "before_commit",
    "after_commit",
    "after_validation_before_cleanup",
    "during_pruning_after_commit_delete",
    "during_pruning_after_sidecar_delete",
})


class ResumePublicationInterrupted(RuntimeError):
    """Deliberate failure injection used to verify publication recovery."""


@dataclass(frozen=True)
class InvalidCommittedGeneration:
    sequence: int
    reason: str


@dataclass(frozen=True)
class LoadedResumeGeneration:
    sequence: int
    state: Mapping[str, Any]
    sidecar: Mapping[str, Any]
    commit: Mapping[str, Any]
    state_path: Path
    sidecar_path: Path
    commit_path: Path


@dataclass(frozen=True)
class ResumeScan:
    valid_generations: tuple[LoadedResumeGeneration, ...]
    invalid_commits: tuple[InvalidCommittedGeneration, ...]
    orphan_files: tuple[str, ...]

    @property
    def highest_valid(self) -> LoadedResumeGeneration | None:
        return self.valid_generations[-1] if self.valid_generations else None


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on platforms that permit directory FDs."""

    path.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not expose portable directory fsync.  Every file itself
        # is still fsynced by atomic_publish_bytes.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, data: bytes) -> str:
    status = atomic_publish_bytes(path, data)
    _fsync_directory(path.parent)
    return status


def _maybe_interrupt(crash_after: str | None, point: str) -> None:
    if crash_after == point:
        raise ResumePublicationInterrupted(f"interrupted at {point}")


def _tensor_bytes(value: Any) -> bytes:
    import torch

    tensor = value.detach().cpu().contiguous()
    if tensor.layout != torch.strided:
        raise TypeError("resume tensors must use strided layout")
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _array_record(path: str, value: Any, *, kind: str) -> dict[str, Any]:
    if kind == "torch_tensor":
        dtype = str(value.dtype)
        shape = list(value.shape)
        data = _tensor_bytes(value)
        requires_grad = bool(value.requires_grad)
    else:
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError(f"object array is forbidden in resume state at {path}")
        dtype = array.dtype.str
        shape = list(array.shape)
        data = array.tobytes(order="C")
        requires_grad = False
    logical_hash = logical_array_sha256_from_byte_hash(
        name=path,
        dtype=dtype,
        shape=shape,
        c_order_byte_sha256=sha256_bytes(data),
        byte_length=len(data),
    )
    return {
        "path": path,
        "kind": kind,
        "dtype": dtype,
        "shape": shape,
        "requires_grad": requires_grad,
        "logical_sha256": logical_hash,
    }


def _inventory_value(value: Any, path: str, records: list[dict[str, Any]]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        record = _array_record(path, value, kind="torch_tensor")
        records.append(record)
        return {key: record[key] for key in (
            "kind", "dtype", "shape", "requires_grad", "logical_sha256"
        )}
    if isinstance(value, np.ndarray):
        record = _array_record(path, value, kind="numpy_array")
        records.append(record)
        return {key: record[key] for key in (
            "kind", "dtype", "shape", "requires_grad", "logical_sha256"
        )}
    if isinstance(value, np.generic):
        return _inventory_value(value.item(), path, records)
    if isinstance(value, Mapping):
        normalized: list[dict[str, Any]] = []
        keyed: list[tuple[str, str, Any]] = []
        for raw_key in value:
            if isinstance(raw_key, bool) or not isinstance(raw_key, (str, int)):
                raise TypeError(f"resume mapping keys must be strings or integers at {path}")
            if isinstance(raw_key, str) and not raw_key:
                raise TypeError(f"resume mapping string key is empty at {path}")
            keyed.append((type(raw_key).__name__, str(raw_key), raw_key))
        for key_kind, key_text, raw_key in sorted(keyed):
            key_record: dict[str, Any] = {"kind": key_kind, "value": raw_key}
            child = f"{path}[{key_kind}:{key_text}]"
            normalized.append({
                "key": key_record,
                "value": _inventory_value(value[raw_key], child, records),
            })
        return {"kind": "mapping", "items": normalized}
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _inventory_value(item, f"{path}[{index}]", records)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes", "length": len(value),
            "logical_sha256": canonical_sha256({
                "byte_length": len(value),
                "byte_sha256": sha256_bytes(value),
                "contract": "HCWDL_REPRESENTATION_RESUME_BYTES/v1",
                "path": path,
            }),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"nonfinite float in resume state at {path}")
        return {"kind": "float", "hex": value.hex()}
    raise TypeError(f"unsupported resume value {type(value).__name__} at {path}")


def build_state_inventory(state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Inventory every namespace/leaf and return its logical state hash."""

    if not isinstance(state, Mapping) or set(state) != REQUIRED_STATE_NAMESPACES:
        raise ValueError(
            "resume state namespaces differ; expected exactly "
            f"{sorted(REQUIRED_STATE_NAMESPACES)}"
        )
    namespaces: dict[str, Any] = {}
    for namespace in sorted(state):
        records: list[dict[str, Any]] = []
        tree = _inventory_value(state[namespace], namespace, records)
        namespace_hash = canonical_sha256(tree)
        namespaces[namespace] = {
            "logical_sha256": namespace_hash,
            "tensor_and_array_leaves": records,
            "tree": tree,
        }
    logical_hash = canonical_sha256({
        name: namespaces[name]["logical_sha256"] for name in sorted(namespaces)
    })
    return namespaces, logical_hash


def _torch_bytes(state: Mapping[str, Any]) -> bytes:
    import torch

    stream = BytesIO()
    torch.save(dict(state), stream)
    return stream.getvalue()


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ValueError("resume PT payload is not a mapping")
    return loaded


def _normalize_lineage(lineage: Mapping[str, Any]) -> dict[str, str]:
    normalized = validate_parent_hashes(lineage)
    if set(normalized) != REQUIRED_LINEAGE_KEYS:
        raise ValueError("resume lineage keys differ")
    try:
        validate_registered_graph_sha256(normalized["ascent_graph"])
    except ValueError as error:
        raise ValueError("resume ascent graph differs") from error
    return normalized


def _validate_cursor(
    state: Mapping[str, Any], *, completed_pass: int, completed_update: int,
    next_canonical_batch: int,
) -> None:
    cursor = state["cursor"]
    if not isinstance(cursor, Mapping) or set(cursor) != {
        "completed_pass", "completed_update", "next_canonical_batch"
    }:
        raise ValueError("resume cursor namespace differs")
    expected = {
        "completed_pass": completed_pass,
        "completed_update": completed_update,
        "next_canonical_batch": next_canonical_batch,
    }
    if dict(cursor) != expected:
        raise ValueError("resume cursor metadata differs from serialized state")


def _validate_state_bindings(
    state: Mapping[str, Any], *, lineage: Mapping[str, str],
    active_projections: Sequence[str], calibration_artifact_hashes: Mapping[str, str],
) -> None:
    target_bindings = state["target_bindings"]
    if not isinstance(target_bindings, Mapping) or dict(target_bindings) != {
        "generation_sha256": lineage["target_generation"],
        "logical_target_sha256": lineage["target_logical"],
    }:
        raise ValueError("resume target bindings differ")
    heads = state["representation_heads"]
    if not isinstance(heads, Mapping) or tuple(sorted(heads)) != tuple(active_projections):
        raise ValueError("resume active projection names differ")
    calibration = state["calibration"]
    if not isinstance(calibration, Mapping) or calibration.get("artifact_hashes") != dict(
        calibration_artifact_hashes
    ):
        raise ValueError("resume calibration artifact bindings differ")
    runtime = state["producer_runtime_signature"]
    if not isinstance(runtime, Mapping) or runtime.get("content_hash") != lineage[
        "producer_runtime_signature"
    ]:
        raise ValueError("resume producer runtime signature differs")


def publish_resume_generation(
    root: str | Path,
    *,
    sequence: int,
    state: Mapping[str, Any],
    lineage: Mapping[str, Any],
    completed_pass: int,
    completed_update: int,
    next_canonical_batch: int,
    active_projections: Sequence[str],
    calibration_artifact_hashes: Mapping[str, Any],
    retain_generations: int = 2,
    validated_prior_generations: Sequence[LoadedResumeGeneration] | None = None,
    crash_after: str | None = None,
) -> LoadedResumeGeneration:
    """Publish and validate one immutable resume generation.

    ``crash_after`` is a test-only fault-injection point.  It never changes the
    bytes or lineage of a successfully committed generation.
    """

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("resume sequence must be a nonnegative integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (completed_pass, completed_update, next_canonical_batch)
    ):
        raise ValueError("resume pass/update/batch cursor must be nonnegative integers")
    if isinstance(retain_generations, bool) or retain_generations < 2:
        raise ValueError("resume cleanup must retain at least two generations")
    if crash_after is not None and crash_after not in _CRASH_POINTS:
        raise ValueError("unknown resume crash-injection point")

    normalized_lineage = _normalize_lineage(lineage)
    projection_names = tuple(sorted(str(name) for name in active_projections))
    if len(projection_names) != len(set(projection_names)) or any(not name for name in projection_names):
        raise ValueError("active projection names must be unique and nonempty")
    calibration_hashes = validate_parent_hashes(
        calibration_artifact_hashes, allow_empty=True,
    )
    _validate_cursor(
        state, completed_pass=completed_pass, completed_update=completed_update,
        next_canonical_batch=next_canonical_batch,
    )
    _validate_state_bindings(
        state, lineage=normalized_lineage, active_projections=projection_names,
        calibration_artifact_hashes=calibration_hashes,
    )
    inventory, logical_state_sha256 = build_state_inventory(state)
    state_bytes = _torch_bytes(state)
    state_byte_sha256 = sha256_bytes(state_bytes)

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    state_name = f"state_{sequence}.pt"
    sidecar_name = f"state_{sequence}.json"
    commit_name = f"commit_{sequence}.json"
    state_path = directory / state_name
    sidecar_path = directory / sidecar_name
    commit_path = directory / commit_name

    _maybe_interrupt(crash_after, "before_state")
    _publish(state_path, state_bytes)
    _maybe_interrupt(crash_after, "after_state")

    sidecar = build_versioned_artifact(
        RESUME_STATE_CONTRACT,
        parents=normalized_lineage,
        payload={
            "record_kind": "state_sidecar",
            "sequence": sequence,
            "state_filename": state_name,
            "state_serialized_byte_sha256": state_byte_sha256,
            "state_logical_sha256": logical_state_sha256,
            "required_state_namespaces": sorted(REQUIRED_STATE_NAMESPACES),
            "namespace_inventory": inventory,
            "completed_pass": completed_pass,
            "completed_update": completed_update,
            "next_canonical_batch": next_canonical_batch,
            "active_projections": list(projection_names),
            "calibration_artifact_hashes": calibration_hashes,
            "target_generation_sha256": normalized_lineage["target_generation"],
            "target_logical_sha256": normalized_lineage["target_logical"],
            "producer_runtime_signature_sha256": normalized_lineage[
                "producer_runtime_signature"
            ],
        },
    )
    sidecar_bytes = _json_file_bytes(sidecar)
    _publish(sidecar_path, sidecar_bytes)
    _maybe_interrupt(crash_after, "after_sidecar")
    _maybe_interrupt(crash_after, "before_commit")

    commit = build_versioned_artifact(
        RESUME_STATE_CONTRACT,
        parents={
            **normalized_lineage,
            "state_sidecar": sidecar["content_hash"],
        },
        payload={
            "record_kind": "commit",
            "sequence": sequence,
            "state_relative_path": state_name,
            "state_serialized_byte_sha256": state_byte_sha256,
            "state_logical_sha256": logical_state_sha256,
            "sidecar_relative_path": sidecar_name,
            "sidecar_serialized_byte_sha256": sha256_bytes(sidecar_bytes),
            "sidecar_content_hash": sidecar["content_hash"],
        },
    )
    _publish(commit_path, _json_file_bytes(commit))
    _maybe_interrupt(crash_after, "after_commit")

    if validated_prior_generations is None:
        generation = validate_resume_generation(
            directory, sequence=sequence, expected_lineage=normalized_lineage,
        )
    else:
        # The state was just inventoried, serialized, hashed, atomically
        # linked, and fsynced by this process.  Verify the durable bytes once,
        # then retain the already authenticated in-memory state instead of
        # immediately torch-loading and reinventoring the same payload.  A
        # fresh process always takes the exhaustive validator above through
        # load_highest_valid_resume().
        if sha256_file(state_path) != state_byte_sha256:
            raise ValueError("published resume state bytes differ")
        if sha256_file(sidecar_path) != sha256_bytes(sidecar_bytes):
            raise ValueError("published resume sidecar bytes differ")
        commit_bytes = _json_file_bytes(commit)
        if sha256_file(commit_path) != sha256_bytes(commit_bytes):
            raise ValueError("published resume commit bytes differ")
        generation = LoadedResumeGeneration(
            sequence=sequence,
            state=state,
            sidecar=sidecar,
            commit=commit,
            state_path=state_path,
            sidecar_path=sidecar_path,
            commit_path=commit_path,
        )
    _maybe_interrupt(crash_after, "after_validation_before_cleanup")
    fast_priors = None
    if validated_prior_generations is not None:
        supplied = tuple(validated_prior_generations)
        supplied_sequences = tuple(item.sequence for item in supplied)
        if (
            supplied_sequences != tuple(sorted(set(supplied_sequences)))
            or any(item.sequence >= sequence for item in supplied)
            or any(item.state_path.parent.resolve() != directory.resolve() for item in supplied)
            or any(
                _normalize_lineage(item.sidecar.get("parents", {}))
                != normalized_lineage
                for item in supplied
            )
            or any(
                not path.is_file()
                for item in supplied
                for path in (item.state_path, item.sidecar_path, item.commit_path)
            )
        ):
            raise ValueError("validated prior resume generations differ")
        files, _ = _sequence_files(directory)
        committed = {
            item_sequence for item_sequence, kinds in files.items()
            if "commit" in kinds
        }
        expected_committed = {*supplied_sequences, sequence}
        if committed == expected_committed:
            fast_priors = supplied
    if fast_priors is None:
        prune_resume_generations(
            directory, retain=retain_generations,
            expected_lineage=normalized_lineage, crash_after=crash_after,
        )
    else:
        _prune_validated_resume_generations(
            directory,
            generations=(*fast_priors, generation),
            retain=retain_generations,
            crash_after=crash_after,
        )
    return generation


def validate_resume_generation(
    root: str | Path,
    *,
    sequence: int,
    expected_lineage: Mapping[str, Any] | None = None,
) -> LoadedResumeGeneration:
    """Validate a committed PT/sidecar/commit triple and load its state."""

    directory = Path(root)
    state_path = directory / f"state_{sequence}.pt"
    sidecar_path = directory / f"state_{sequence}.json"
    commit_path = directory / f"commit_{sequence}.json"
    if not commit_path.is_file():
        raise FileNotFoundError(f"resume generation {sequence} has no commit")
    commit = load_json(commit_path)
    validate_versioned_artifact(
        commit,
        expected_contract=RESUME_STATE_CONTRACT,
        required_payload_keys=(
            "record_kind", "sequence", "state_relative_path",
            "state_serialized_byte_sha256", "state_logical_sha256",
            "sidecar_relative_path", "sidecar_serialized_byte_sha256",
            "sidecar_content_hash",
        ),
    )
    commit_payload = commit["payload"]
    expected_commit_fields = {
        "record_kind", "sequence", "state_relative_path",
        "state_serialized_byte_sha256", "state_logical_sha256",
        "sidecar_relative_path", "sidecar_serialized_byte_sha256",
        "sidecar_content_hash",
    }
    if set(commit_payload) != expected_commit_fields:
        raise ValueError("resume commit fields differ")
    if (
        commit_payload["record_kind"] != "commit"
        or commit_payload["sequence"] != sequence
        or commit_payload["state_relative_path"] != state_path.name
        or commit_payload["sidecar_relative_path"] != sidecar_path.name
    ):
        raise ValueError("resume commit identity or paths differ")
    if not state_path.is_file() or not sidecar_path.is_file():
        raise ValueError("resume commit references a missing state member")
    if sha256_file(state_path) != require_sha256(
        commit_payload["state_serialized_byte_sha256"], name="resume state byte SHA-256",
    ):
        raise ValueError("resume state bytes differ from commit")
    if sha256_file(sidecar_path) != require_sha256(
        commit_payload["sidecar_serialized_byte_sha256"], name="resume sidecar byte SHA-256",
    ):
        raise ValueError("resume sidecar bytes differ from commit")

    sidecar = load_json(sidecar_path)
    sidecar_lineage = _normalize_lineage(sidecar.get("parents", {}))
    normalized_expected = (
        None if expected_lineage is None else _normalize_lineage(expected_lineage)
    )
    if normalized_expected is not None and sidecar_lineage != normalized_expected:
        raise ValueError("resume sidecar lineage differs")
    validate_versioned_artifact(
        sidecar,
        expected_contract=RESUME_STATE_CONTRACT,
        expected_parents=sidecar_lineage,
        required_payload_keys=(
            "record_kind", "sequence", "state_filename",
            "state_serialized_byte_sha256", "state_logical_sha256",
            "required_state_namespaces", "namespace_inventory",
            "completed_pass", "completed_update", "next_canonical_batch",
            "active_projections", "calibration_artifact_hashes",
            "target_generation_sha256", "target_logical_sha256",
            "producer_runtime_signature_sha256",
        ),
    )
    sidecar_payload = sidecar["payload"]
    expected_sidecar_fields = {
        "record_kind", "sequence", "state_filename",
        "state_serialized_byte_sha256", "state_logical_sha256",
        "required_state_namespaces", "namespace_inventory",
        "completed_pass", "completed_update", "next_canonical_batch",
        "active_projections", "calibration_artifact_hashes",
        "target_generation_sha256", "target_logical_sha256",
        "producer_runtime_signature_sha256",
    }
    if set(sidecar_payload) != expected_sidecar_fields:
        raise ValueError("resume sidecar fields differ")
    if (
        sidecar_payload["record_kind"] != "state_sidecar"
        or sidecar_payload["sequence"] != sequence
        or sidecar_payload["state_filename"] != state_path.name
        or sidecar_payload["required_state_namespaces"]
        != sorted(REQUIRED_STATE_NAMESPACES)
        or sidecar["content_hash"] != commit_payload["sidecar_content_hash"]
        or commit["parents"].get("state_sidecar") != sidecar["content_hash"]
    ):
        raise ValueError("resume sidecar identity differs")
    commit_base_lineage = {
        name: digest for name, digest in commit["parents"].items()
        if name != "state_sidecar"
    }
    if commit_base_lineage != sidecar_lineage:
        raise ValueError("resume commit lineage differs from sidecar")
    if (
        sidecar_payload["state_serialized_byte_sha256"]
        != commit_payload["state_serialized_byte_sha256"]
        or sidecar_payload["state_logical_sha256"]
        != commit_payload["state_logical_sha256"]
    ):
        raise ValueError("resume sidecar and commit state hashes differ")
    if (
        sidecar_payload["target_generation_sha256"]
        != sidecar_lineage["target_generation"]
        or sidecar_payload["target_logical_sha256"]
        != sidecar_lineage["target_logical"]
        or sidecar_payload["producer_runtime_signature_sha256"]
        != sidecar_lineage["producer_runtime_signature"]
    ):
        raise ValueError("resume sidecar target/runtime lineage differs")

    state = _torch_load(state_path)
    for cursor_name in ("completed_pass", "completed_update", "next_canonical_batch"):
        cursor_value = sidecar_payload[cursor_name]
        if isinstance(cursor_value, bool) or not isinstance(cursor_value, int) or cursor_value < 0:
            raise ValueError("resume sidecar cursor fields differ")
    projection_names = sidecar_payload["active_projections"]
    if (
        not isinstance(projection_names, list)
        or any(not isinstance(name, str) or not name for name in projection_names)
        or projection_names != sorted(set(projection_names))
    ):
        raise ValueError("resume sidecar active projections differ")
    _validate_cursor(
        state,
        completed_pass=sidecar_payload["completed_pass"],
        completed_update=sidecar_payload["completed_update"],
        next_canonical_batch=sidecar_payload["next_canonical_batch"],
    )
    calibration_hashes = validate_parent_hashes(
        sidecar_payload["calibration_artifact_hashes"], allow_empty=True,
    )
    _validate_state_bindings(
        state,
        lineage=sidecar_lineage,
        active_projections=tuple(projection_names),
        calibration_artifact_hashes=calibration_hashes,
    )
    inventory, logical_hash = build_state_inventory(state)
    if (
        inventory != sidecar_payload["namespace_inventory"]
        or logical_hash != sidecar_payload["state_logical_sha256"]
    ):
        raise ValueError("resume logical state differs from sidecar")
    return LoadedResumeGeneration(
        sequence=sequence,
        state=state,
        sidecar=sidecar,
        commit=commit,
        state_path=state_path,
        sidecar_path=sidecar_path,
        commit_path=commit_path,
    )


def _sequence_files(root: Path) -> tuple[dict[int, set[str]], list[str]]:
    files: dict[int, set[str]] = {}
    unrelated: list[str] = []
    if not root.exists():
        return files, unrelated
    for path in root.iterdir():
        if not path.is_file():
            continue
        matched = False
        for kind, pattern in (
            ("state", _STATE_PATTERN), ("sidecar", _SIDECAR_PATTERN),
            ("commit", _COMMIT_PATTERN),
        ):
            match = pattern.fullmatch(path.name)
            if match is not None:
                files.setdefault(int(match.group(1)), set()).add(kind)
                matched = True
                break
        if not matched and not path.name.startswith("."):
            unrelated.append(path.name)
    return files, unrelated


def scan_resume_generations(
    root: str | Path, *, expected_lineage: Mapping[str, Any] | None = None,
) -> ResumeScan:
    """Audit every sequence and return valid generations in ascending order."""

    directory = Path(root)
    files, unrelated = _sequence_files(directory)
    valid: list[LoadedResumeGeneration] = []
    invalid: list[InvalidCommittedGeneration] = []
    orphans = list(unrelated)
    for sequence in sorted(files):
        kinds = files[sequence]
        if "commit" not in kinds:
            for kind in sorted(kinds):
                suffix = "pt" if kind == "state" else "json"
                orphans.append(f"state_{sequence}.{suffix}")
            continue
        try:
            valid.append(validate_resume_generation(
                directory, sequence=sequence, expected_lineage=expected_lineage,
            ))
        except Exception as error:  # preserve the corruption audit; try older states
            invalid.append(InvalidCommittedGeneration(sequence, f"{type(error).__name__}: {error}"))
    return ResumeScan(
        valid_generations=tuple(valid),
        invalid_commits=tuple(invalid),
        orphan_files=tuple(sorted(set(orphans))),
    )


def load_highest_valid_resume(
    root: str | Path, *, expected_lineage: Mapping[str, Any] | None = None,
) -> tuple[LoadedResumeGeneration | None, ResumeScan]:
    """Load the highest fully valid commit; never treat an orphan pair as state."""

    scan = scan_resume_generations(root, expected_lineage=expected_lineage)
    return scan.highest_valid, scan


def prune_resume_generations(
    root: str | Path, *, retain: int = 2,
    expected_lineage: Mapping[str, Any] | None = None,
    crash_after: str | None = None,
) -> tuple[int, ...]:
    """Delete only valid generations older than the newest ``retain`` states."""

    if isinstance(retain, bool) or not isinstance(retain, int) or retain < 2:
        raise ValueError("resume cleanup must retain at least two generations")
    if crash_after not in {
        None, "during_pruning_after_commit_delete",
        "during_pruning_after_sidecar_delete",
    }:
        raise ValueError("unknown resume pruning crash-injection point")
    directory = Path(root)
    scan = scan_resume_generations(directory, expected_lineage=expected_lineage)
    removable = scan.valid_generations[:-retain]
    removed: list[int] = []
    for generation in removable:
        # Commit is removed first so a crash during cleanup cannot leave a
        # partially deleted generation advertised as recoverable.
        generation.commit_path.unlink()
        _fsync_directory(directory)
        _maybe_interrupt(crash_after, "during_pruning_after_commit_delete")
        generation.sidecar_path.unlink()
        _maybe_interrupt(crash_after, "during_pruning_after_sidecar_delete")
        generation.state_path.unlink()
        _fsync_directory(directory)
        removed.append(generation.sequence)
    return tuple(removed)


def _prune_validated_resume_generations(
    root: Path,
    *,
    generations: Sequence[LoadedResumeGeneration],
    retain: int,
    crash_after: str | None,
) -> tuple[int, ...]:
    """Prune a caller-owned, already authenticated in-process generation set.

    Recovery/startup always performs the exhaustive independent disk scan.
    During one uninterrupted training process, every supplied generation was
    either authenticated by that startup scan or by the immediately preceding
    publication.  Reusing those handles avoids reloading and reinventoring the
    same multi-gigabyte states solely to decide which old triple to delete.
    """

    if isinstance(retain, bool) or not isinstance(retain, int) or retain < 2:
        raise ValueError("resume cleanup must retain at least two generations")
    ordered = tuple(generations)
    sequences = tuple(item.sequence for item in ordered)
    if sequences != tuple(sorted(set(sequences))):
        raise ValueError("validated resume generation order differs")
    removable = ordered[:-retain]
    removed: list[int] = []
    for generation in removable:
        if any(
            path.parent.resolve() != root.resolve()
            for path in (
                generation.commit_path,
                generation.sidecar_path,
                generation.state_path,
            )
        ):
            raise ValueError("validated resume generation root differs")
        generation.commit_path.unlink()
        _fsync_directory(root)
        _maybe_interrupt(crash_after, "during_pruning_after_commit_delete")
        generation.sidecar_path.unlink()
        _maybe_interrupt(crash_after, "during_pruning_after_sidecar_delete")
        generation.state_path.unlink()
        _fsync_directory(root)
        removed.append(generation.sequence)
    return tuple(removed)


__all__ = [
    "InvalidCommittedGeneration",
    "LoadedResumeGeneration",
    "REQUIRED_LINEAGE_KEYS",
    "REQUIRED_STATE_NAMESPACES",
    "RESUME_STATE_CONTRACT",
    "ResumePublicationInterrupted",
    "ResumeScan",
    "build_state_inventory",
    "load_highest_valid_resume",
    "prune_resume_generations",
    "publish_resume_generation",
    "scan_resume_generations",
    "validate_resume_generation",
]
