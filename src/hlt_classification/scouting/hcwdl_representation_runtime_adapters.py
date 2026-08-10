"""Closed HCWDL-RKD task adapters.

The campaign runtime is deliberately split in two:

* production adapters below call repository-owned scientific/artifact APIs and
  authenticate the exact registered output paths; and
* the local planning adapter performs bounded, real synthetic work without
  opening a final-population capability or claiming scientific authority.

There is no generic Python entry point, import string, shell command, or
"success" fallback.  A task whose concrete data-plane assembly has not been
bound by the immutable runtime row raises :class:`ProductionConfigurationError`
before publishing anything.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import copy
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PRODUCTION_ADAPTER_CONTRACT: Final = "HCWDL_REPRESENTATION_PRODUCTION_ADAPTER/v1"
LOCAL_PLANNING_WORK_CONTRACT: Final = "HCWDL_REPRESENTATION_LOCAL_TASK_WORK/v1"


# This is deliberately an explicit, closed registry rather than a category
# fallback.  The local smoke runner checks it against TASK_KINDS before doing
# any work, and every result records the exact semantic surface exercised.
LOCAL_SEMANTIC_COVERAGE: Final[Mapping[str, str]] = {
    "tap_schema": "canonical_tap_schema_builder",
    "surface_parity": "local_model_surface_forward_backward_non_authorizing",
    "architecture_attestation": "architecture_schema_and_local_surface_gate",
    "parent_loss_attestation": "corrected_parent_loss_forward_backward",
    "parent_import": "parent_import_contract_validator",
    "dense_teacher_import": "historical_toff_training_teacher_file_authority",
    "kernel_resources": "spectral_resource_builder_and_weighted_mean",
    "representation_recipe": "frozen_recipe_builder_and_validator",
    "numerical_acceptance": "spectral_numerical_acceptance_math",
    "target_build": "ordinary_and_toff_target_generation_load_join_cleanup",
    "control_registry": "ascent_graph_and_control_registry_build_validate",
    "cache_miniature_bank": "target_bank_generation_and_cleanup_authorization",
    "cache_miniature": "ordinary_and_toff_cache_miniature_lifecycle",
    "smoke_probe": "all_node_full_representation_loss_backward",
    "zero_coefficient_acceptance": "zero_coefficient_gradient_disconnect_math",
    "reservation": "synthetic_shared_final_reservation_pipeline",
    "train_node": "registered_primary_full_representation_loss_backward",
    "train_control": "registered_control_full_representation_loss_backward",
    "shuffle_map": "within_class_derangement_builder_validator",
    "target_cleanup": "authorized_target_cleanup_and_recovery_validator",
    "screen_aggregate": "screen_reporting_builder",
    "confirmation_registry": "confirmation_registry_builder",
    "confirmation": "registered_confirmation_full_representation_loss_backward",
    "confirmation_aggregate": "five_seed_confirmation_aggregate_builder",
    "finalist_lock": "synthetic_shared_final_finalist_gate",
    "shared_final_claim": "synthetic_shared_final_claim_pipeline",
    "final_selection": "synthetic_shared_final_selection_pipeline",
    "assignment_shard": "synthetic_shared_final_assignment_shard_pipeline",
    "assignment_finalize": "synthetic_shared_final_assignment_finalize_pipeline",
    "data_attestation": "synthetic_shared_final_data_attestation_pipeline",
    "execution_lock": "synthetic_shared_final_execution_lock_pipeline",
    "prediction_shard": "synthetic_shared_final_prediction_shard_pipeline",
    "prediction_finalize": "synthetic_shared_final_prediction_finalize_pipeline",
    "metric_join": "synthetic_shared_final_metric_join_pipeline",
    "final_aggregate": "synthetic_shared_final_aggregate_pipeline",
    "validation_only_aggregate": "validation_only_reporting_aggregate_builder",
    "dense_training_aggregate": "dense_training_only_aggregate_builder",
}


class ProductionConfigurationError(RuntimeError):
    """An immutable row lacks inputs needed by its fixed production adapter."""


class RegisteredInputPath(str):
    """Internal provenance marker for a path resolved from ``runtime_row.inputs``."""


_REGISTERED_ARGUMENT_TAGS: Final = frozenset({
    "registered_json",
    "registered_reference",
    "registered_path",
    "registered_member",
    "registered_field",
})


def _registered_directory_identities(path: Path) -> set[str]:
    """Return the complete byte-inventory identity for a registered directory."""

    inventory = _directory_inventory(path)
    return {str(inventory["inventory_sha256"])}


def _registered_input_reference(
    runtime_row: Mapping[str, Any], logical_name: object, *, location: str,
) -> dict[str, str]:
    """Resolve and re-authenticate one exact runtime-row input reference."""

    logical = str(logical_name)
    inputs = runtime_row.get("inputs")
    if not isinstance(inputs, Mapping) or logical not in inputs:
        raise ProductionConfigurationError(
            f"{location} names an unregistered input {logical!r}"
        )
    raw = inputs[logical]
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise ProductionConfigurationError(
            f"registered input reference differs for {logical!r}"
        )
    digest = require_sha256(
        raw["sha256"], name=f"{logical} registered-input SHA-256",
    )
    path = Path(str(raw["path"]))
    if not path.is_absolute():
        raise ProductionConfigurationError(
            f"registered input path is not absolute: {logical}"
        )
    if not path.exists():
        raise FileNotFoundError(f"registered input is absent: {logical} -> {path}")
    if path.is_symlink():
        raise PermissionError(f"registered input is a symlink: {logical}")
    if path.is_file():
        identities = {sha256_file(path)}
    elif path.is_dir():
        identities = _registered_directory_identities(path)
    else:
        raise ValueError(f"registered input is not a file/directory: {logical}")
    if digest not in identities:
        raise ValueError(f"registered input byte/hash identity differs: {logical}")
    return {"path": str(path), "sha256": digest}


def _load_registered_json(reference: Mapping[str, str], *, location: str) -> Any:
    path = Path(reference["path"])
    if not path.is_file():
        raise ProductionConfigurationError(
            f"{location} requires a registered JSON file"
        )
    value = load_json(path)
    if isinstance(value, Mapping) and "content_hash" in value:
        _validate_declared_content_hash(value)
    return value


def _registered_member(
    runtime_row: Mapping[str, Any], value: object, *, location: str,
) -> Any:
    if not isinstance(value, Mapping) or set(value) not in (
        {"input", "relative"}, {"input", "relative", "mode"},
    ):
        raise ProductionConfigurationError(
            f"{location}.registered_member fields differ"
        )
    reference = _registered_input_reference(
        runtime_row, value["input"], location=location,
    )
    root = Path(reference["path"])
    if not root.is_dir():
        raise ProductionConfigurationError(
            f"{location}.registered_member requires a registered directory"
        )
    relative_text = str(value["relative"])
    relative = PurePosixPath(relative_text)
    if (
        not relative_text or relative_text in {".", ".."}
        or "\\" in relative_text or relative.is_absolute()
        or (relative.parts and ":" in relative.parts[0])
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PermissionError(f"{location}.registered_member path is unsafe")
    member = root.joinpath(*relative.parts)
    try:
        member.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PermissionError(
            f"{location}.registered_member escapes its registered input"
        ) from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PermissionError(
                f"{location}.registered_member crosses a symlink"
            )
    if not member.exists():
        raise FileNotFoundError(f"{location}.registered_member is absent: {member}")
    mode = str(value.get("mode", "path"))
    if mode == "path":
        return RegisteredInputPath(str(member))
    if not member.is_file():
        raise ProductionConfigurationError(
            f"{location}.registered_member mode {mode!r} requires a file"
        )
    member_reference = {"path": str(member), "sha256": sha256_file(member)}
    if mode == "reference":
        return member_reference
    if mode == "json":
        return _load_registered_json(member_reference, location=location)
    raise ProductionConfigurationError(
        f"{location}.registered_member mode must be json, reference, or path"
    )


def _registered_field(
    runtime_row: Mapping[str, Any], value: object, *, location: str,
) -> Any:
    """Extract one scalar from authenticated JSON bytes or a JSON descendant."""

    if not isinstance(value, Mapping) or set(value) not in (
        {"input", "field"}, {"input", "relative", "field"},
    ):
        raise ProductionConfigurationError(
            f"{location}.registered_field fields differ"
        )
    field = value["field"]
    if (
        not isinstance(field, list) or not field
        or any(not isinstance(part, str) or not part for part in field)
    ):
        raise ProductionConfigurationError(
            f"{location}.registered_field path differs"
        )
    if "relative" in value:
        source = _registered_member(
            runtime_row,
            {"input": value["input"], "relative": value["relative"], "mode": "json"},
            location=location,
        )
    else:
        reference = _registered_input_reference(
            runtime_row, value["input"], location=location,
        )
        source = _load_registered_json(reference, location=location)
    cursor = source
    for part in field:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"{location}.registered_field is absent at {part!r}")
        cursor = cursor[part]
    if cursor is not None and not isinstance(cursor, (str, int, float, bool)):
        raise ProductionConfigurationError(
            f"{location}.registered_field must resolve to a JSON scalar"
        )
    return cursor


def resolve_registered_arguments(
    value: object,
    runtime_row: Mapping[str, Any],
    *,
    location: str = "scientific_arguments",
) -> Any:
    """Resolve tagged inputs and reject unauthenticated artifact injection.

    Five JSON-safe tags are accepted:

    ``{"registered_json": logical}``
        Load the exact registered JSON bytes.
    ``{"registered_reference": logical}``
        Return the exact authenticated ``path``/``sha256`` reference.
    ``{"registered_path": logical}``
        Return the exact registered file or directory path.
    ``{"registered_member": {"input": logical, "relative": path, "mode": ...}}``
        Resolve a symlink-free descendant of a registered directory.  ``mode``
        is ``path`` by default and may be ``reference`` or ``json``.
    ``{"registered_field": {"input": logical, "field": [key, ...]}}``
        Extract one scalar from authenticated JSON bytes.  An optional safe
        ``relative`` member path supports committed directory inputs.

    Raw ``{path, sha256}`` objects and inline content-hashed artifacts are
    rejected even when their self-declared hash is internally consistent.
    This makes the immutable runtime-row input registry the sole authority for
    scientific artifact bytes.
    """

    if isinstance(value, Mapping):
        keys = set(value)
        present_tags = keys & _REGISTERED_ARGUMENT_TAGS
        if present_tags:
            if len(present_tags) != 1 or len(keys) != 1:
                raise ProductionConfigurationError(
                    f"{location} registered-input tag fields differ"
                )
            tag = next(iter(present_tags))
            payload = value[tag]
            if tag == "registered_member":
                return _registered_member(runtime_row, payload, location=location)
            if tag == "registered_field":
                return _registered_field(runtime_row, payload, location=location)
            reference = _registered_input_reference(
                runtime_row, payload, location=location,
            )
            if tag == "registered_reference":
                if not Path(reference["path"]).is_file():
                    raise ProductionConfigurationError(
                        f"{location}.registered_reference requires a file"
                    )
                return reference
            if tag == "registered_path":
                return RegisteredInputPath(reference["path"])
            return _load_registered_json(reference, location=location)
        if {"path", "sha256"} <= keys:
            raise PermissionError(
                f"{location} contains a raw artifact reference; use a registered-input tag"
            )
        if "content_hash" in keys:
            raise PermissionError(
                f"{location} contains an inline self-hashed artifact; register its bytes"
            )
        return {
            str(key): resolve_registered_arguments(
                item, runtime_row, location=f"{location}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            resolve_registered_arguments(
                item, runtime_row, location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{location} contains a non-JSON value")


def _resolve_registered_path_argument(
    value: object, runtime_row: Mapping[str, Any], *, location: str,
) -> str:
    """Require one path-valued argument to use a path/member tag."""

    if not isinstance(value, Mapping) or len(value) != 1:
        raise ProductionConfigurationError(
            f"{location} must be a registered_path or registered_member tag"
        )
    tag = next(iter(value))
    if tag not in {"registered_path", "registered_member"}:
        raise ProductionConfigurationError(
            f"{location} must be a registered_path or registered_member tag"
        )
    result = resolve_registered_arguments(value, runtime_row, location=location)
    if not isinstance(result, RegisteredInputPath):
        raise ProductionConfigurationError(f"{location} did not resolve to a path")
    return str(result)


def _validate_declared_content_hash(value: Mapping[str, Any]) -> str:
    """Validate a repository artifact against its explicit versioned contract."""

    contract = value.get("contract")
    schema_version = value.get("schema_version")
    if not isinstance(contract, str) or not contract:
        raise ValueError("repository artifact contract is absent")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("repository artifact schema version is absent")
    return validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=schema_version,
    )


def _task_identity(task: Any) -> tuple[str, str]:
    key = str(getattr(task, "task_key", ""))
    kind = str(getattr(task, "kind", ""))
    if not key or not kind:
        raise ValueError("HCWDL-RKD runtime task identity is empty")
    return key, kind


def _parameters(task: Any, runtime_row: Mapping[str, Any]) -> Mapping[str, Any]:
    key, kind = _task_identity(task)
    value = runtime_row.get("parameters")
    if not isinstance(value, Mapping):
        raise ProductionConfigurationError(
            f"{key} ({kind}) requires an immutable JSON parameter object"
        )
    missing = [name for name in ("adapter_contract", "task_kind") if name not in value]
    if missing:
        raise ProductionConfigurationError(
            f"{key} ({kind}) production binding is incomplete; required fields: "
            "adapter_contract, task_kind"
        )
    if value.get("adapter_contract") != PRODUCTION_ADAPTER_CONTRACT:
        raise ProductionConfigurationError(
            f"{key} ({kind}) adapter_contract must be {PRODUCTION_ADAPTER_CONTRACT}"
        )
    if value.get("task_kind") != kind:
        raise ValueError(f"{key} production adapter task kind differs")
    # Parameters are already inside the content-hashed runtime binding.  The
    # separate runtime-signature field authenticates the Weaver/accelerator
    # runtime and must not be overloaded as a second parameter hash.
    return value


def _require_exact_parameters(
    task: Any,
    runtime_row: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> Mapping[str, Any]:
    value = _parameters(task, runtime_row)
    expected_required = {"adapter_contract", "task_kind", *required}
    allowed = expected_required | set(optional)
    missing = expected_required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        key, kind = _task_identity(task)
        raise ProductionConfigurationError(
            f"{key} ({kind}) production parameter schema differs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}, "
            f"required={sorted(expected_required)}"
        )
    return value


def _outputs(task: Any, runtime_row: Mapping[str, Any]) -> dict[str, Path]:
    raw = runtime_row.get("outputs")
    registered = tuple(getattr(task, "registered_outputs", ()))
    if not isinstance(raw, Mapping) or set(raw) != set(registered):
        raise ValueError("HCWDL-RKD production output registry differs")
    from .hcwdl_representation_runtime_binding import IMMUTABLE_OUTPUT_ROOT_BINDING

    result = {}
    for name in registered:
        value = raw[name]
        if isinstance(value, Mapping) and set(value) == {
            IMMUTABLE_OUTPUT_ROOT_BINDING,
        }:
            descriptor = value[IMMUTABLE_OUTPUT_ROOT_BINDING]
            if not isinstance(descriptor, Mapping) or "root" not in descriptor:
                raise ValueError("HCWDL-RKD deferred output descriptor differs")
            result[name] = Path(str(descriptor["root"]))
        else:
            result[name] = Path(str(value))
    if len(set(result.values())) != len(result):
        raise ValueError("HCWDL-RKD production output paths collide")
    return result


def _published_path_matches_output(
    task: Any, runtime_row: Mapping[str, Any], logical: str, published: Path,
) -> bool:
    """Accept an exact leaf, or the sole commit under an exact deferred root."""

    from .hcwdl_representation_runtime_binding import IMMUTABLE_OUTPUT_ROOT_BINDING

    raw = runtime_row["outputs"][logical]
    if isinstance(raw, Mapping) and set(raw) == {IMMUTABLE_OUTPUT_ROOT_BINDING}:
        root = Path(str(raw[IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]))
        try:
            relative = published.resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return (
            len(relative.parts) == 3
            and relative.parts[0] == "committed"
            and len(relative.parts[1]) == 64
            and relative.parts[2] == "commit.json"
        ) or (
            len(relative.parts) == 2
            and relative.parts[0] == "committed"
            and len(relative.parts[1]) == 64
        )
    return published.resolve() == Path(str(raw)).resolve()


def _publish_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish once, or require an already published value to be identical."""

    if path.exists():
        if path.is_symlink():
            raise PermissionError(f"registered immutable output is a symlink: {path}")
        if not path.is_file() or load_json(path) != dict(value):
            raise FileExistsError(f"registered immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(path, value)


def _directory_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise PermissionError(f"registered output directory is a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"registered output directory is absent: {path}")
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise PermissionError(
                f"registered output directory contains a symlink: {item}"
            )
        if item.is_file():
            rows.append({
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            })
    if not rows:
        raise ValueError(f"registered output directory is empty: {path}")
    return {"files": rows, "inventory_sha256": canonical_sha256(rows)}


def _validate_registered_outputs(
    task: Any, runtime_row: Mapping[str, Any], *, operation: str,
) -> dict[str, Any]:
    proof: dict[str, Any] = {}
    from .hcwdl_representation_runtime_binding import IMMUTABLE_OUTPUT_ROOT_BINDING
    from .hcwdl_representation_task_runtime import _resolve_immutable_output_root

    for logical, path in _outputs(task, runtime_row).items():
        raw = runtime_row["outputs"][logical]
        if isinstance(raw, Mapping) and set(raw) == {IMMUTABLE_OUTPUT_ROOT_BINDING}:
            descriptor = raw[IMMUTABLE_OUTPUT_ROOT_BINDING]
            path, digest = _resolve_immutable_output_root(descriptor)
            proof[logical] = {
                "path": str(path), **_directory_inventory(path),
                "resolved_from_deferred_root": str(descriptor["root"]),
                "resolved_inventory_sha256": digest,
            }
            continue
        if path.is_file():
            row: dict[str, Any] = {
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix == ".json":
                artifact = load_json(path)
                if isinstance(artifact, Mapping) and "content_hash" in artifact:
                    _validate_declared_content_hash(artifact)
                    row["logical_sha256"] = artifact["content_hash"]
            proof[logical] = row
        elif path.is_dir():
            proof[logical] = {"path": str(path), **_directory_inventory(path)}
        else:
            raise FileNotFoundError(
                f"{operation} did not publish registered output {logical}: {path}"
            )
    return {
        "operation": operation,
        "registered_outputs": proof,
        "registered_outputs_sha256": canonical_sha256(proof),
    }


def _single_output(task: Any, runtime_row: Mapping[str, Any]) -> Path:
    outputs = _outputs(task, runtime_row)
    if len(outputs) != 1:
        raise ValueError("task adapter requires exactly one registered output")
    return next(iter(outputs.values()))


def _tap_schema_adapter(spec, task, index, runtime_row):
    del spec, index
    _require_exact_parameters(task, runtime_row)
    from hlt_classification.models.hcwdl_surfaces import tap_schema

    _publish_exact_json(_single_output(task, runtime_row), tap_schema())
    return _validate_registered_outputs(task, runtime_row, operation="tap_schema")


def _surface_fixture(*, seed: int, device: str):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def cloud(channels: int, tokens: int):
        features = torch.randn((2, channels, tokens), generator=generator).to(device)
        vectors = torch.randn((2, 4, tokens), generator=generator)
        # Weaver's standard-four input is (px, py, pz, E).  Independent
        # Gaussian energy components create mostly spacelike vectors and can
        # make the installed pair-feature path nonfinite.  A fixed unit mass
        # keeps this deterministic fixture physical without constraining its
        # directions or momentum scales.
        vectors[:, 3] = vectors[:, :3].square().sum(1).add(1.0).sqrt()
        vectors = vectors.to(device)
        mask = torch.ones((2, 1, tokens), dtype=torch.bool, device=device)
        visible = torch.arange(tokens, dtype=torch.int64, device=device).repeat(2, 1)
        family = (visible % 2).to(torch.int8)
        return features, vectors, mask, visible, family

    ordinary = cloud(21, 6)
    charged = cloud(19, 5)
    neutral = cloud(7, 4)
    native = (
        charged[0], charged[1], charged[2], neutral[0], neutral[1],
        neutral[2], charged[3], neutral[3],
    )
    return ordinary, native


def build_installed_weaver_surface_parity_artifact(
    *, fixture_seed: int = 1337, device: str = "cpu",
) -> dict[str, Any]:
    """Build the authoritative installed-Weaver parity artifact.

    This reusable path is shared by the pre-campaign evidence CLI and the
    registered campaign gate, preventing the bootstrap step from depending on
    a campaign that already requires the resulting parent-import hash.
    """

    from hlt_classification.models.hcwdl_surfaces import (
        build_surface_parity_report, validate_surface_parity_report,
    )
    from hlt_classification.models.scouting_particle_transformer import (
        build_native_offline_particle_transformer,
        build_scouting_particle_transformer,
    )

    ordinary = build_scouting_particle_transformer()
    native = build_native_offline_particle_transformer()
    ordinary_inputs, native_inputs = _surface_fixture(
        seed=int(fixture_seed), device=str(device),
    )
    ordinary.to(device)
    native.to(device)
    report = build_surface_parity_report(
        ordinary_model=ordinary, native_offline_model=native,
        ordinary_inputs=ordinary_inputs, native_offline_inputs=native_inputs,
        runtime_kind="installed_weaver",
    )
    validate_surface_parity_report(report)
    return report


def _surface_parity_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row, required=("fixture_seed", "runtime_kind"),
    )
    if parameters["runtime_kind"] != "installed_weaver":
        raise ProductionConfigurationError(
            "production surface parity requires runtime_kind='installed_weaver'"
        )
    report = build_installed_weaver_surface_parity_artifact(
        fixture_seed=int(parameters["fixture_seed"]),
        device=str(runtime_row["device"]),
    )
    _publish_exact_json(_single_output(task, runtime_row), report)
    return _validate_registered_outputs(task, runtime_row, operation="surface_parity")


def _prebuilt_validated_adapter(
    spec: Mapping[str, Any], task: Any, index: int | None,
    runtime_row: Mapping[str, Any], *, validator: Callable[[Mapping[str, Any]], Any],
    operation: str, additional_required: Sequence[str] = (),
    pre_publish_check: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Authenticate an immutable artifact produced during campaign assembly.

    Campaign creation already binds parent-import, graph, recipe, and final
    disposition hashes.  These gate tasks re-open and validate those exact
    bytes; they do not relabel an unvalidated path as a completed task.
    """

    del index
    parameters = _require_exact_parameters(
        task, runtime_row, required=("artifact", *additional_required),
    )
    reference = resolve_registered_arguments(
        parameters["artifact"], runtime_row, location=f"{operation}.artifact",
    )
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ProductionConfigurationError(
            f"{operation}.artifact must use registered_reference"
        )
    source = Path(str(reference["path"]))
    value = load_json(source)
    digest = validator(value)
    if value.get("content_hash") is not None and digest != value["content_hash"]:
        raise ValueError(f"{operation} logical artifact hash differs")
    if pre_publish_check is not None:
        pre_publish_check(value)
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, value)
    return _validate_registered_outputs(task, runtime_row, operation=operation)


def _architecture_attestation_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=(
            "tap_schema_path", "surface_parity_path", "parent_report_paths",
            "model_source_paths",
        ),
    )
    raw_parent_reports = parameters["parent_report_paths"]
    raw_model_sources = parameters["model_source_paths"]
    if not isinstance(raw_parent_reports, Mapping) or not raw_parent_reports:
        raise ProductionConfigurationError(
            "architecture attestation requires the full parent-report path registry"
        )
    if not isinstance(raw_model_sources, Mapping) or not raw_model_sources:
        raise ProductionConfigurationError(
            "architecture attestation requires canonical model-source paths"
        )
    parent_reports = {
        str(key): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"architecture_attestation.parent_report_paths.{key}",
        )
        for key, value in raw_parent_reports.items()
    }
    model_sources = {
        str(key): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"architecture_attestation.model_source_paths.{key}",
        )
        for key, value in raw_model_sources.items()
    }
    tap_schema_path = _resolve_registered_path_argument(
        parameters["tap_schema_path"], runtime_row,
        location="architecture_attestation.tap_schema_path",
    )
    surface_parity_path = _resolve_registered_path_argument(
        parameters["surface_parity_path"], runtime_row,
        location="architecture_attestation.surface_parity_path",
    )
    from hlt_classification.models.hcwdl_surfaces import (
        build_architecture_attestation_from_files,
        validate_architecture_attestation,
    )
    artifact = build_architecture_attestation_from_files(
        tap_schema_path=tap_schema_path,
        surface_parity_path=surface_parity_path,
        parent_reports=parent_reports,
        model_source_paths=model_sources,
    )
    validate_architecture_attestation(artifact, require_authorized=True)
    _publish_exact_json(_single_output(task, runtime_row), artifact)
    return _validate_registered_outputs(
        task, runtime_row, operation="architecture_attestation",
    )


def _parent_loss_attestation_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=(
            "parent_campaign_spec_path", "parent_recipe_path",
            "parent_report_paths", "runtime_source_paths",
        ),
    )
    raw_reports = parameters["parent_report_paths"]
    raw_sources = parameters["runtime_source_paths"]
    if (
        not isinstance(raw_reports, Mapping) or not raw_reports
        or not isinstance(raw_sources, Mapping) or not raw_sources
    ):
        raise ProductionConfigurationError(
            "parent-loss attestation requires report and runtime-source path registries"
        )
    reports = {
        str(key): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"parent_loss_attestation.parent_report_paths.{key}",
        )
        for key, value in raw_reports.items()
    }
    sources = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"parent_loss_attestation.runtime_source_paths.{name}",
        )
        for name, value in raw_sources.items()
    }
    parent_recipe_path = _resolve_registered_path_argument(
        parameters["parent_recipe_path"], runtime_row,
        location="parent_loss_attestation.parent_recipe_path",
    )
    parent_campaign_spec_path = _resolve_registered_path_argument(
        parameters["parent_campaign_spec_path"], runtime_row,
        location="parent_loss_attestation.parent_campaign_spec_path",
    )
    from .hcwdl_parent_loss import (
        build_parent_loss_attestation_from_reports,
        validate_parent_loss_attestation,
    )
    artifact = build_parent_loss_attestation_from_reports(
        parent_recipe_path=parent_recipe_path,
        parent_campaign_spec_path=parent_campaign_spec_path,
        parent_reports=reports,
        runtime_source_paths=sources,
    )
    validate_parent_loss_attestation(
        artifact, parent_recipe=load_json(Path(parent_recipe_path)),
    )
    _publish_exact_json(_single_output(task, runtime_row), artifact)
    return _validate_registered_outputs(
        task, runtime_row, operation="parent_loss_attestation",
    )


def _validate_parent_import_registered_bundles(
    imported: Mapping[str, Any], *, architecture: Mapping[str, Any],
    parent_report_paths: Mapping[str, str], model_source_paths: Mapping[str, str],
) -> None:
    """Tie the persisted import to every registered parent-bundle member."""

    from .hcwdl_representation_locks import (
        IMPORTED_LOGIT_CONTROLS, IMPORTED_TEACHERS,
    )

    imported_rows = {
        str(row["node_id"]): row
        for group in (imported["payload"]["teachers"], imported["payload"]["logit_controls"])
        for row in group
    }
    expected_nodes = set(IMPORTED_TEACHERS) | set(IMPORTED_LOGIT_CONTROLS)
    if set(imported_rows) != expected_nodes or set(parent_report_paths) != expected_nodes:
        raise ValueError("parent import registered report bundle differs")
    architecture_rows = {
        str(row["node_id"]): row for row in architecture["checkpoint_audits"]
    }
    if set(architecture_rows) != expected_nodes:
        raise ValueError("parent import architecture registry differs")
    for node_id in sorted(expected_nodes):
        registered_path = Path(parent_report_paths[node_id]).resolve()
        imported_path = Path(str(imported_rows[node_id]["report_path"])).resolve()
        architecture_path = Path(str(architecture_rows[node_id]["report_path"])).resolve()
        if registered_path != imported_path or registered_path != architecture_path:
            raise ValueError(f"parent import registered report path differs for {node_id}")
        report = load_json(registered_path)
        contract = report.get("contract") if isinstance(report, Mapping) else None
        schema = report.get("schema_version") if isinstance(report, Mapping) else None
        if not isinstance(contract, str) or not isinstance(schema, int):
            raise ValueError(f"parent import registered report is not versioned: {node_id}")
        digest = validate_content_hash(
            report, expected_contract=contract, expected_schema_version=schema,
        )
        if digest != imported_rows[node_id]["report_sha256"]:
            raise ValueError(f"parent import registered report hash differs for {node_id}")

    attested_sources = {
        str(row["logical_name"]): row for row in architecture["model_source_files"]
    }
    if set(model_source_paths) != set(attested_sources):
        raise ValueError("parent import registered model-source bundle differs")
    for logical_name, raw_path in model_source_paths.items():
        registered_path = Path(raw_path).resolve()
        attested = attested_sources[logical_name]
        if (
            registered_path != Path(str(attested["path"])).resolve()
            or not registered_path.is_file()
            or sha256_file(registered_path) != attested["sha256"]
        ):
            raise ValueError(
                f"parent import registered model-source bytes differ for {logical_name}"
            )
    d0w_path = Path(model_source_paths["D0w"]).resolve()
    d0w_architecture = architecture_rows["D0w"]
    if d0w_path != Path(str(d0w_architecture["engine_report_path"])).resolve():
        raise ValueError("parent D0w model source is not its attested engine report")


def _validate_parent_import_fresh_evidence(
    imported: Mapping[str, Any], *, architecture_sha256: str,
    parent_loss_sha256: str,
) -> None:
    """Require the just-produced evidence, not only static authority copies."""

    parents = imported.get("parents")
    if (
        not isinstance(parents, Mapping)
        or require_sha256(
            architecture_sha256, name="fresh architecture attestation",
        ) != parents.get("architecture_attestation")
        or require_sha256(
            parent_loss_sha256, name="fresh parent-loss attestation",
        ) != parents.get("parent_loss_attestation")
    ):
        raise PermissionError(
            "fresh parent evidence differs from the prebuilt parent import"
        )


def _parent_import_adapter(spec, task, index, runtime_row):
    del index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=(
            "artifact", "architecture_attestation", "parent_loss_attestation",
            "parent_report_paths", "model_source_paths", "authority_files",
            "qualifier_report_paths", "confirmation_report_paths",
        ),
    )
    artifact_reference = resolve_registered_arguments(
        parameters["artifact"], runtime_row, location="parent_import.artifact",
    )
    if not isinstance(artifact_reference, Mapping) or set(artifact_reference) != {
        "path", "sha256",
    }:
        raise ProductionConfigurationError(
            "parent_import.artifact must use registered_reference"
        )
    imported = load_json(Path(str(artifact_reference["path"])))
    architecture_reference = resolve_registered_arguments(
        parameters["architecture_attestation"], runtime_row,
        location="parent_import.architecture_attestation",
    )
    loss_reference = resolve_registered_arguments(
        parameters["parent_loss_attestation"], runtime_row,
        location="parent_import.parent_loss_attestation",
    )
    if any(
        not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
        for reference in (architecture_reference, loss_reference)
    ):
        raise ProductionConfigurationError(
            "parent import fresh evidence must use registered_reference"
        )
    architecture = load_json(Path(str(architecture_reference["path"])))
    loss = load_json(Path(str(loss_reference["path"])))
    raw_reports = parameters["parent_report_paths"]
    raw_sources = parameters["model_source_paths"]
    if not isinstance(raw_reports, Mapping) or not isinstance(raw_sources, Mapping):
        raise ProductionConfigurationError("parent import bundle registries differ")
    parent_report_paths = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row, location=f"parent_import.parent_report_paths.{name}",
        )
        for name, value in raw_reports.items()
    }
    model_source_paths = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row, location=f"parent_import.model_source_paths.{name}",
        )
        for name, value in raw_sources.items()
    }
    raw_authority = parameters["authority_files"]
    raw_qualifiers = parameters["qualifier_report_paths"]
    raw_confirmations = parameters["confirmation_report_paths"]
    if (
        not isinstance(raw_authority, Mapping)
        or not isinstance(raw_qualifiers, Mapping)
        or not isinstance(raw_confirmations, Mapping)
    ):
        raise ProductionConfigurationError("parent import authority registries differ")
    authority_files = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row, location=f"parent_import.authority_files.{name}",
        )
        for name, value in raw_authority.items()
    }
    qualifier_report_paths = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"parent_import.qualifier_report_paths.{name}",
        )
        for name, value in raw_qualifiers.items()
    }
    confirmation_report_paths = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"parent_import.confirmation_report_paths.{name}",
        )
        for name, value in raw_confirmations.items()
    }
    from .hcwdl_representation_locks import (
        validate_parent_import_against_authority_files,
    )
    from .hcwdl_parent_loss import validate_parent_loss_attestation
    from hlt_classification.models.hcwdl_surfaces import (
        validate_architecture_attestation,
    )

    architecture_sha256 = validate_architecture_attestation(
        architecture, require_authorized=True,
    )
    parent_recipe = load_json(Path(authority_files["recipe"]))
    parent_loss_sha256 = validate_parent_loss_attestation(
        loss, parent_recipe=parent_recipe,
    )
    _validate_parent_import_fresh_evidence(
        imported, architecture_sha256=architecture_sha256,
        parent_loss_sha256=parent_loss_sha256,
    )

    digest = validate_parent_import_against_authority_files(
        imported, authority_files=authority_files,
        qualifier_report_paths=qualifier_report_paths,
        confirmation_report_paths=confirmation_report_paths,
    )
    if digest != spec["parent_import_sha256"]:
        raise ValueError("parent import differs from campaign identity")
    _validate_parent_import_registered_bundles(
        imported, architecture=architecture,
        parent_report_paths=parent_report_paths,
        model_source_paths=model_source_paths,
    )
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, imported)
    return _validate_registered_outputs(task, runtime_row, operation="parent_import")


def _dense_teacher_import_adapter(spec, task, index, runtime_row):
    del index
    parameters = _require_exact_parameters(
        task, runtime_row, required=(
            "artifact", "authority_files", "historical_project_dir",
        ),
    )
    artifact_reference = resolve_registered_arguments(
        parameters["artifact"], runtime_row,
        location="dense_teacher_import.artifact",
    )
    if not isinstance(artifact_reference, Mapping) or set(artifact_reference) != {
        "path", "sha256",
    }:
        raise ProductionConfigurationError(
            "dense teacher artifact must use registered_reference"
        )
    imported = load_json(Path(str(artifact_reference["path"])))
    raw_files = parameters["authority_files"]
    if not isinstance(raw_files, Mapping):
        raise ProductionConfigurationError("dense teacher authority files differ")
    authority_files = {
        str(name): _resolve_registered_path_argument(
            value, runtime_row,
            location=f"dense_teacher_import.authority_files.{name}",
        )
        for name, value in raw_files.items()
    }
    historical_project = _resolve_registered_path_argument(
        parameters["historical_project_dir"], runtime_row,
        location="dense_teacher_import.historical_project_dir",
    )
    from .hcwdl_representation_dense_teacher import (
        validate_dense_teacher_import_against_files,
    )
    digest = validate_dense_teacher_import_against_files(
        imported, authority_files=authority_files,
        historical_project_dir=historical_project,
    )
    if digest != spec["parent_import_sha256"]:
        raise PermissionError("dense teacher import differs from campaign identity")
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, imported)
    return _validate_registered_outputs(
        task, runtime_row, operation="dense_teacher_import",
    )


def _kernel_resources_adapter(spec, task, index, runtime_row):
    del index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=(
            "root", "producer_task_id", "immutable_parent_hashes",
            "registered_output_row", "campaign_or_recovery_owner", "recipe",
        ),
    )
    from .hcwdl_representation_kernels import (
        generate_spectral_resource_bundle, publish_spectral_resources,
        spectral_resource_logical_hashes,
    )
    from .hcwdl_representation_recipe import validate_representation_recipe

    recipe_reference = resolve_registered_arguments(
        parameters["recipe"], runtime_row, location="kernel_resources.recipe",
    )
    if not isinstance(recipe_reference, Mapping) or set(recipe_reference) != {
        "path", "sha256",
    }:
        raise ProductionConfigurationError(
            "kernel_resources.recipe must use registered_reference"
        )
    recipe = load_json(Path(str(recipe_reference["path"])))
    validate_representation_recipe(recipe)
    if recipe["content_hash"] != spec["representation_recipe_sha256"]:
        raise ValueError("kernel resource recipe differs from campaign identity")
    resources = generate_spectral_resource_bundle()
    if (
        recipe["parents"].get("kernel_resources") != resources.content_hash
        or recipe["payload"].get("kernel_array_logical_hashes")
        != spectral_resource_logical_hashes(resources)
    ):
        raise ValueError("kernel resources differ from the prebuilt recipe")

    published = publish_spectral_resources(
        resources,
        root=parameters["root"],
        producer_task_id=str(parameters["producer_task_id"]),
        immutable_parent_hashes=parameters["immutable_parent_hashes"],
        registered_output_row=parameters["registered_output_row"],
        campaign_or_recovery_owner=parameters["campaign_or_recovery_owner"],
    )
    logical = next(iter(getattr(task, "registered_outputs", ())))
    if not _published_path_matches_output(
        task, runtime_row, logical, published.envelope.directory,
    ):
        raise ValueError("kernel resource envelope differs from registered output")
    return _validate_registered_outputs(task, runtime_row, operation="kernel_resources")


def _representation_recipe_adapter(spec, task, index, runtime_row):
    from .hcwdl_representation_graph import (
        validate_ascent_graph_artifact, validate_control_registry_artifact,
    )
    from .hcwdl_representation_recipe import validate_representation_recipe

    lineage_names = (
        "representation_graph", "control_registry", "parent_import",
        "assignment_manifest",
    )
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=("artifact", "producer_source_sha256", *lineage_names),
    )
    producer_source_sha256 = require_sha256(
        parameters["producer_source_sha256"],
        name="runtime representation recipe producer source",
    )

    lineage_artifacts: dict[str, Mapping[str, Any]] = {}
    lineage_paths: dict[str, Path] = {}
    for name in lineage_names:
        reference = resolve_registered_arguments(
            parameters[name], runtime_row,
            location=f"representation_recipe.{name}",
        )
        if not isinstance(reference, Mapping) or set(reference) != {
            "path", "sha256",
        }:
            raise ProductionConfigurationError(
                f"representation_recipe.{name} must use registered_reference"
            )
        lineage_paths[name] = Path(str(reference["path"]))
        lineage_artifacts[name] = load_json(lineage_paths[name])

    parent_import = lineage_artifacts["parent_import"]
    dense_teacher = parent_import.get("contract") == (
        "HCWDL_REPRESENTATION_DENSE_TEACHER_IMPORT/v1"
    )
    if dense_teacher:
        from .hcwdl_representation_dense_teacher import validate_dense_teacher_import
        parent_import_sha256 = validate_dense_teacher_import(parent_import)
        parent_graph_sha256 = parent_import["payload"][
            "historical_parent_graph_sha256"
        ]
        source_sha256 = parent_import["parents"]["historical_source_manifest"]
        split_sha256 = parent_import["parents"]["historical_split_manifest"]
    else:
        from .hcwdl_representation_locks import validate_parent_import
        parent_import_sha256 = validate_parent_import(parent_import)
        parent_graph_sha256 = parent_import["parents"]["parent_graph"]
        source_sha256 = parent_import["parents"]["source_manifest"]
        split_sha256 = parent_import["parents"]["split_manifest"]
    graph = lineage_artifacts["representation_graph"]
    graph_sha256 = validate_ascent_graph_artifact(
        graph,
        expected_parents={
            "parent_graph": parent_graph_sha256,
            "parent_import": parent_import_sha256,
        },
    )
    control_registry_sha256 = validate_control_registry_artifact(
        lineage_artifacts["control_registry"],
        ascent_graph_artifact_sha256=graph_sha256,
    )
    from .hcwdl_assignment import validate_train_assignment_authority
    from .hcwdl_representation_kernels import generate_spectral_resource_bundle
    assignment = lineage_artifacts["assignment_manifest"]
    expected_assignment_rows = assignment.get("expected_mapped_jets")
    assignment_sha256 = validate_train_assignment_authority(
        lineage_paths["assignment_manifest"],
        split_manifest_sha256=split_sha256,
        row_selection_sha256=(
            parent_import["parents"]["historical_row_selection"]
            if dense_teacher else parent_import["parents"]["row_selection"]
        ),
        expected_mapped_jets=expected_assignment_rows,
    )
    kernel_bundle_sha256 = generate_spectral_resource_bundle().content_hash
    if (
        parent_import_sha256 != require_sha256(
            spec["parent_import_sha256"], name="campaign parent import",
        )
        or graph_sha256 != require_sha256(
            spec["graph_sha256"], name="campaign representation graph",
        )
        or source_sha256
        != require_sha256(
            spec["source_manifest_sha256"], name="campaign source manifest",
        )
        or split_sha256
        != require_sha256(
            spec["split_manifest_sha256"], name="campaign split manifest",
        )
    ):
        raise PermissionError(
            "representation recipe registered lineage differs from campaign identity"
        )

    def pre_publish_check(recipe: Mapping[str, Any]) -> None:
        if recipe["content_hash"] != spec["representation_recipe_sha256"]:
            raise ValueError("representation recipe differs from campaign identity")
        expected_parent_links = ({
            "assignment_manifest": assignment_sha256,
            "dense_teacher_import": parent_import_sha256,
            "historical_parent_graph": parent_graph_sha256,
            "parent_recipe": parent_import["parents"]["historical_recipe"],
            "row_selection": parent_import["parents"]["historical_row_selection"],
            "source_manifest": source_sha256,
            "split_manifest": split_sha256,
            "representation_ascent_graph": graph_sha256,
            "representation_control_registry": control_registry_sha256,
            "kernel_resources": kernel_bundle_sha256,
        } if dense_teacher else {
            "architecture_attestation": parent_import["parents"][
                "architecture_attestation"
            ],
            "assignment_manifest": parent_import["parents"][
                "train_assignment_manifest"
            ],
            "parent_graph": parent_import["parents"]["parent_graph"],
            "parent_loss_attestation": parent_import["parents"][
                "parent_loss_attestation"
            ],
            "parent_recipe": parent_import["parents"]["parent_recipe"],
            "representation_ascent_graph": graph_sha256,
            "representation_control_registry": control_registry_sha256,
            "row_selection": parent_import["parents"]["row_selection"],
            "source_manifest": parent_import["parents"]["source_manifest"],
            "split_manifest": parent_import["parents"]["split_manifest"],
            "teacher_import": parent_import_sha256,
        })
        if any(
            recipe["parents"].get(name) != digest
            for name, digest in expected_parent_links.items()
        ):
            raise PermissionError(
                "representation recipe parents differ from registered lineage"
            )
        if recipe["parents"].get("producer_source") != producer_source_sha256:
            raise PermissionError(
                "representation recipe producer source differs from measured runtime source"
            )

    result = _prebuilt_validated_adapter(
        spec, task, index, runtime_row,
        validator=validate_representation_recipe,
        operation="representation_recipe",
        additional_required=("producer_source_sha256", *lineage_names),
        pre_publish_check=pre_publish_check,
    )
    return result


def _numerical_acceptance_adapter(spec, task, index, runtime_row):
    del index
    parameters = _require_exact_parameters(
        task, runtime_row, required=("representation_recipe",),
    )
    from .hcwdl_numerical_acceptance import build_numerical_acceptance
    from .hcwdl_representation_recipe import validate_representation_recipe

    reference = resolve_registered_arguments(
        parameters["representation_recipe"], runtime_row,
        location="numerical_acceptance.representation_recipe",
    )
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ProductionConfigurationError(
            "numerical_acceptance recipe must use registered_reference"
        )
    recipe = load_json(Path(str(reference["path"])))
    validate_representation_recipe(recipe)
    if recipe["content_hash"] != spec["representation_recipe_sha256"]:
        raise ValueError("numerical acceptance recipe differs from campaign identity")
    result = build_numerical_acceptance()
    expected = recipe["payload"]["acceptance_evidence"]
    if any(
        expected[name] != result["content_hash"]
        for name in ("analytic_gradient", "diagnostic_reference", "finite_kernel")
    ):
        raise ValueError("numerical acceptance differs from the prebuilt recipe")

    _publish_exact_json(_single_output(task, runtime_row), result)
    return _validate_registered_outputs(task, runtime_row, operation="numerical_acceptance")


def _smoke_probe_adapter(spec, task, index, runtime_row):
    del spec, index
    _require_exact_parameters(task, runtime_row)
    from .hcwdl_representation_smoke import run_scientific_full_loss_probe

    report = run_scientific_full_loss_probe(device=str(runtime_row["device"]))
    _publish_exact_json(_single_output(task, runtime_row), report)
    return _validate_registered_outputs(task, runtime_row, operation="smoke_probe")


def _mapping_builder_adapter(
    spec: Mapping[str, Any], task: Any, index: int | None,
    runtime_row: Mapping[str, Any], *, builder: Callable[..., Mapping[str, Any]],
    operation: str,
) -> dict[str, Any]:
    """Call a fixed builder using only inline scalars and registered artifacts."""

    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row, required=("builder_arguments",),
    )
    raw_arguments = parameters["builder_arguments"]
    if not isinstance(raw_arguments, Mapping):
        raise ProductionConfigurationError(
            f"{operation} builder_arguments must be an immutable JSON object"
        )
    arguments = resolve_registered_arguments(
        raw_arguments, runtime_row, location=f"{operation}.builder_arguments",
    )
    if not isinstance(arguments, Mapping):
        raise ProductionConfigurationError(
            f"{operation} resolved builder arguments are not an object"
        )
    result = builder(**dict(arguments))
    if not isinstance(result, Mapping):
        raise TypeError(f"{operation} repository builder returned a non-artifact")
    _validate_declared_content_hash(result)
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, result)
    return _validate_registered_outputs(task, runtime_row, operation=operation)


def _screen_aggregate_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=("parent_import", "architecture_attestation", "builder_arguments"),
    )
    parent_import_reference = resolve_registered_arguments(
        parameters["parent_import"], runtime_row,
        location="screen_aggregate.parent_import",
    )
    architecture_reference = resolve_registered_arguments(
        parameters["architecture_attestation"], runtime_row,
        location="screen_aggregate.architecture_attestation",
    )
    if any(
        not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
        for reference in (parent_import_reference, architecture_reference)
    ):
        raise ProductionConfigurationError(
            "screen aggregate lineage must use registered references"
        )
    parent_import = load_json(Path(str(parent_import_reference["path"])))
    architecture = load_json(Path(str(architecture_reference["path"])))
    raw_arguments = parameters["builder_arguments"]
    if not isinstance(raw_arguments, Mapping):
        raise ProductionConfigurationError(
            "screen_aggregate builder_arguments must be an immutable JSON object"
        )
    arguments = resolve_registered_arguments(
        raw_arguments, runtime_row, location="screen_aggregate.builder_arguments",
    )
    if not isinstance(arguments, Mapping):
        raise ProductionConfigurationError(
            "screen_aggregate resolved builder arguments are not an object"
        )
    raw_parent_reports = arguments.get("parent_reports")
    dense_teacher = parent_import.get("contract") == (
        "HCWDL_REPRESENTATION_DENSE_TEACHER_IMPORT/v1"
    )
    if dense_teacher:
        from .hcwdl_representation_dense_teacher import validate_dense_teacher_import
        validate_dense_teacher_import(parent_import)
        if architecture != parent_import or raw_parent_reports != {}:
            raise PermissionError("dense screen imported obsolete parent authority")
        parent_reports = {}
    else:
        from .hcwdl_representation_locks import IMPORTED_LOGIT_CONTROLS
        if not isinstance(raw_parent_reports, Mapping) or set(raw_parent_reports) != set(
            IMPORTED_LOGIT_CONTROLS
        ):
            raise ValueError("screen aggregate parent-report registry differs")
        from .hcwdl_representation_production import _validate_imported_pmard_source

        parent_reports = {
            str(node_id): _validate_imported_pmard_source(
                reference, node_id=str(node_id), parent_import=parent_import,
                architecture=architecture, name=f"{node_id} screen parent",
            )["engine"]
            for node_id, reference in raw_parent_reports.items()
        }
    from .hcwdl_representation_reporting import build_screen_aggregate

    result = build_screen_aggregate(
        **{**dict(arguments), "parent_reports": parent_reports},
    )
    _validate_declared_content_hash(result)
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, result)
    return _validate_registered_outputs(task, runtime_row, operation="screen_aggregate")


def _confirmation_registry_adapter(spec, task, index, runtime_row):
    from .hcwdl_representation_reporting import build_confirmation_registry

    return _mapping_builder_adapter(
        spec, task, index, runtime_row, builder=build_confirmation_registry,
        operation="confirmation_registry",
    )


def _confirmation_aggregate_adapter(spec, task, index, runtime_row):
    from .hcwdl_representation_reporting import build_confirmation_aggregate

    return _mapping_builder_adapter(
        spec, task, index, runtime_row, builder=build_confirmation_aggregate,
        operation="confirmation_aggregate",
    )


def _dense_training_aggregate_adapter(spec, task, index, runtime_row):
    from .hcwdl_representation_reporting import build_dense_training_aggregate
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=("builder_arguments", "dense_training_disposition"),
    )
    raw_arguments = parameters["builder_arguments"]
    if not isinstance(raw_arguments, Mapping):
        raise ProductionConfigurationError(
            "dense_training_aggregate builder_arguments must be an object"
        )
    arguments = resolve_registered_arguments(
        raw_arguments, runtime_row,
        location="dense_training_aggregate.builder_arguments",
    )
    disposition = resolve_registered_arguments(
        parameters["dense_training_disposition"], runtime_row,
        location="dense_training_aggregate.dense_training_disposition",
    )
    if not isinstance(arguments, Mapping) or not isinstance(disposition, Mapping):
        raise ProductionConfigurationError(
            "dense_training_aggregate registered arguments differ"
        )
    result = build_dense_training_aggregate(
        **dict(arguments), dense_training_disposition=disposition,
    )
    _validate_declared_content_hash(result)
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, result)
    return _validate_registered_outputs(
        task, runtime_row, operation="dense_training_aggregate",
    )


def _finalist_lock_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row, required=("builder_arguments",),
    )
    raw_arguments = parameters["builder_arguments"]
    if not isinstance(raw_arguments, Mapping):
        raise ProductionConfigurationError(
            "finalist_lock builder_arguments must be an immutable JSON object"
        )
    arguments = resolve_registered_arguments(
        raw_arguments, runtime_row, location="finalist_lock.builder_arguments",
    )
    if not isinstance(arguments, Mapping):
        raise ProductionConfigurationError(
            "finalist_lock resolved builder arguments are not an object"
        )
    from .hcwdl_representation_final_policy import (
        build_pretraining_finalist_policy_commitment,
    )
    parent_finalist_lock = arguments.get("parent_finalist_lock")
    commitment = build_pretraining_finalist_policy_commitment(
        parent_finalists=arguments["parent_finalists"],
        parent_finalist_lock=parent_finalist_lock,
    )
    reservation = arguments.get("reservation")
    if (
        not isinstance(reservation, Mapping)
        or reservation.get("finalist_registry_commitment_sha256")
        != commitment["content_hash"]
    ):
        raise PermissionError(
            "finalist-lock parent union/policy differs from pretraining reservation"
        )
    from .hcwdl_representation_final import build_finalist_lock
    builder_arguments = dict(arguments)
    del builder_arguments["parent_finalist_lock"]
    result = build_finalist_lock(**builder_arguments)
    _validate_declared_content_hash(result)
    for path in _outputs(task, runtime_row).values():
        _publish_exact_json(path, result)
    return _validate_registered_outputs(task, runtime_row, operation="finalist_lock")


def _data_attestation_adapter(spec, task, index, runtime_row):
    from .hcwdl_representation_final import build_final_data_attestation

    return _mapping_builder_adapter(
        spec, task, index, runtime_row, builder=build_final_data_attestation,
        operation="data_attestation",
    )


def _reservation_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row, required=(
            "builder_arguments", "assignment_source_partitions", "parent_finalists",
            "matcher_resources", "parent_finalist_lock",
        ),
    )
    raw_arguments = parameters["builder_arguments"]
    if not isinstance(raw_arguments, Mapping):
        raise ProductionConfigurationError(
            "reservation builder_arguments must be an immutable JSON object"
        )
    arguments = resolve_registered_arguments(
        raw_arguments, runtime_row, location="reservation.builder_arguments",
    )
    if not isinstance(arguments, Mapping):
        raise ProductionConfigurationError(
            "reservation resolved builder arguments are not an object"
        )
    matcher_resources = resolve_registered_arguments(
        parameters["matcher_resources"], runtime_row,
        location="reservation.matcher_resources",
    )
    parent_finalist_lock = resolve_registered_arguments(
        parameters["parent_finalist_lock"], runtime_row,
        location="reservation.parent_finalist_lock",
    )
    source_partitions = parameters["assignment_source_partitions"]
    parent_finalists = parameters["parent_finalists"]
    if not all(isinstance(value, Mapping) for value in (
        matcher_resources, parent_finalist_lock,
    )) or not isinstance(source_partitions, list) or not isinstance(
        parent_finalists, list,
    ):
        raise ProductionConfigurationError(
            "reservation scientific commitments are not immutable JSON objects"
        )
    from .hcwdl_representation_final_policy import (
        build_final_assignment_spec,
        build_pretraining_finalist_policy_commitment,
        validate_final_assignment_spec,
        validate_pretraining_finalist_policy_commitment,
    )
    assignment_spec = build_final_assignment_spec(
        matcher_resources=matcher_resources,
        source_partitions=source_partitions,
    )
    finalist_policy = build_pretraining_finalist_policy_commitment(
        parent_finalists=parent_finalists,
        parent_finalist_lock=parent_finalist_lock,
    )
    assignment_digest = validate_final_assignment_spec(
        assignment_spec, matcher_resources=matcher_resources,
        source_partitions=source_partitions,
    )
    policy_digest = validate_pretraining_finalist_policy_commitment(
        finalist_policy,
        parent_finalists=parent_finalists,
        parent_finalist_lock=parent_finalist_lock,
    )
    if (
        arguments.get("assignment_spec_sha256") != assignment_digest
        or arguments.get("finalist_registry_commitment_sha256") != policy_digest
    ):
        raise PermissionError(
            "reservation arguments differ from published assignment/finalist commitments"
        )
    from .hcwdl_shared_final import register_final_population

    reservation = register_final_population(**dict(arguments))
    _validate_declared_content_hash(reservation)
    # The registrar owns its population-scoped publication.  Requiring the
    # exact registered path here prevents a successful call from being
    # credited to a different campaign row.
    outputs = _outputs(task, runtime_row)
    published = {"reservation": False, "assignment": False, "policy": False}
    for logical, path in outputs.items():
        if logical.endswith("/reservation.json"):
            artifact, key = reservation, "reservation"
        elif logical == "final/assignment/specification.json":
            artifact, key = assignment_spec, "assignment"
        elif logical == "final/pretraining_finalist_policy.json":
            artifact, key = finalist_policy, "policy"
        else:
            raise ValueError(f"reservation registered output differs: {logical}")
        _publish_exact_json(path, artifact)
        published[key] = True
    if not all(published.values()):
        raise ValueError("reservation commitment output registry is incomplete")
    return _validate_registered_outputs(task, runtime_row, operation="reservation")


def _shared_final_claim_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=("task_registry_arguments", "claim_arguments"),
    )
    raw_registry_arguments = parameters["task_registry_arguments"]
    raw_claim_arguments = parameters["claim_arguments"]
    if not isinstance(raw_registry_arguments, Mapping) or not isinstance(
        raw_claim_arguments, Mapping,
    ):
        raise ProductionConfigurationError(
            "shared final claim requires immutable task-registry and claim arguments"
        )
    registry_arguments = resolve_registered_arguments(
        raw_registry_arguments, runtime_row,
        location="shared_final_claim.task_registry_arguments",
    )
    claim_arguments = resolve_registered_arguments(
        raw_claim_arguments, runtime_row,
        location="shared_final_claim.claim_arguments",
    )
    if not isinstance(registry_arguments, Mapping) or not isinstance(
        claim_arguments, Mapping,
    ):
        raise ProductionConfigurationError(
            "shared final claim resolved arguments are not objects"
        )
    from .hcwdl_shared_final import build_final_task_registry, claim_final_execution

    registry = build_final_task_registry(**dict(registry_arguments))
    claim = claim_final_execution(task_registry=registry, **dict(claim_arguments))
    outputs = _outputs(task, runtime_row)
    for logical, path in outputs.items():
        artifact = registry if "task_registry" in logical else claim
        _publish_exact_json(path, artifact)
    return _validate_registered_outputs(task, runtime_row, operation="shared_final_claim")


def _target_cleanup_adapter(spec, task, index, runtime_row):
    del spec, index
    parameters = _require_exact_parameters(
        task, runtime_row,
        required=(
            "cleanup_root", "consumer_report_paths",
            "exact_reconstruction_authorized",
        ),
        optional=("bank_root", "generation_id", "committed_directory"),
    )
    raw_report_paths = parameters["consumer_report_paths"]
    if not isinstance(raw_report_paths, Mapping) or not raw_report_paths:
        raise ProductionConfigurationError(
            "target cleanup requires a nonempty immutable consumer-report path registry"
        )
    reports = {
        str(execution): resolve_registered_arguments(
            reference, runtime_row,
            location=f"target_cleanup.consumer_report_paths.{execution}",
        )
        for execution, reference in raw_report_paths.items()
    }
    if any(not isinstance(report, Mapping) for report in reports.values()):
        raise ProductionConfigurationError(
            "target cleanup consumer reports must use registered_json tags"
        )
    if "committed_directory" in parameters:
        if "bank_root" in parameters or "generation_id" in parameters:
            raise ProductionConfigurationError(
                "target cleanup cannot mix committed and legacy target paths"
            )
        committed = Path(_resolve_registered_path_argument(
            parameters["committed_directory"], runtime_row,
            location="target_cleanup.committed_directory",
        ))
        if (
            committed.parent.name != "generations" or len(committed.name) != 64
            or any(character not in "0123456789abcdef" for character in committed.name)
        ):
            raise ProductionConfigurationError(
                "target cleanup committed generation path differs"
            )
        bank_root = str(committed.parent.parent)
        generation_id = committed.name
    else:
        if "bank_root" not in parameters or "generation_id" not in parameters:
            raise ProductionConfigurationError(
                "target cleanup requires one exact target generation"
            )
        bank_root = _resolve_registered_path_argument(
            parameters["bank_root"], runtime_row,
            location="target_cleanup.bank_root",
        )
        generation_id = str(parameters["generation_id"])
    reconstruction_authorized = parameters["exact_reconstruction_authorized"]
    if not isinstance(reconstruction_authorized, bool):
        raise ProductionConfigurationError(
            "target cleanup exact_reconstruction_authorized must be a JSON boolean"
        )
    from .hcwdl_representation_target_recovery import (
        authorize_target_cleanup, complete_target_cleanup,
    )

    authorization = authorize_target_cleanup(
        bank_root, parameters["cleanup_root"],
        generation_id=generation_id,
        consumer_reports=reports,
        exact_reconstruction_authorized=reconstruction_authorized,
    )
    completion = complete_target_cleanup(
        bank_root, parameters["cleanup_root"],
        generation_id=generation_id,
    )
    for logical, path in _outputs(task, runtime_row).items():
        artifact = authorization if "authorization" in logical else completion
        _publish_exact_json(path, artifact)
    return _validate_registered_outputs(task, runtime_row, operation="target_cleanup")


def _local_digest(name: str) -> str:
    return canonical_sha256({"local_non_authorizing_fixture": str(name)})


def _local_parent_loss() -> dict[str, Any]:
    from .hcwdl_parent_loss import parent_loss_runtime_fingerprint

    value = parent_loss_runtime_fingerprint()
    return {
        "work_kind": "corrected_parent_loss_forward_backward",
        "work_sha256": canonical_sha256(value),
    }


def _local_kernel_math() -> dict[str, Any]:
    from .hcwdl_representation_kernels import (
        generate_spectral_resources, weighted_feature_mean,
    )

    resources = generate_spectral_resources("relation")
    values = np.asarray([[-0.4], [0.2], [1.3]], dtype=np.float32)
    weights = np.asarray([1.0, 2.0, 1.0], dtype=np.float32)
    mean = weighted_feature_mean(values, weights, resources)
    mean_array = (
        mean.detach().cpu().numpy() if hasattr(mean, "detach") else np.asarray(mean)
    )
    return {
        "work_kind": "fixed_spectral_feature_mean",
        "resource_sha256": resources.content_hash,
        "mean_sha256": canonical_sha256(mean_array.astype(float).tolist()),
        "finite": bool(np.isfinite(mean_array).all()),
    }


def _local_target_lifecycle(task: Any, array_index: int | None) -> dict[str, Any]:
    """Use the real ordinary+TOFF target builders, loaders, joins and cleanup."""

    from .hcwdl_representation_smoke import run_local_target_lifecycle_probe

    probe = run_local_target_lifecycle_probe()
    logical_bank = str(getattr(task, "logical_bank", "") or "")
    selected = [
        row for row in probe["banks"]
        if not logical_bank
        or row["bank_kind"] == ("toff" if logical_bank == "TOFF" else "ordinary")
    ]
    if not selected:
        raise RuntimeError("local target lifecycle did not cover the registered bank")
    return {
        "work_kind": "real_target_generation_load_join_cleanup",
        "logical_bank": logical_bank or "ordinary_and_toff",
        "array_index": array_index,
        "banks": copy.deepcopy(selected),
        "ordinary_and_toff_probe_sha256": probe["content_hash"],
        "teacher_forward_calls": sum(row["teacher_forward_calls"] for row in selected),
        "cleanup_validated": all(row["cleanup_completed"] for row in selected),
    }


@lru_cache(maxsize=1)
def _local_parent_import_gate() -> Mapping[str, Any]:
    from .hcwdl_representation_locks import (
        IMPORTED_LOGIT_CONTROLS, IMPORTED_TEACHERS,
        PARENT_AUTHORITY_PARENT_KEYS, PARENT_IMPORT_CONTRACT,
        validate_parent_import,
    )

    def row(node_id: str, *, teacher: bool) -> dict[str, Any]:
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
        domain = "native_offline" if node_id == "TOFF" else (
            "hlt" if not teacher or node_id.startswith("D0") else
            f"d{node_id[1:].rstrip('cw').lower()}"
        )
        digest = _local_digest(f"parent-checkpoint:{node_id}")
        return {
            "node_id": node_id, "domain": domain, "track": track,
            "report_path": f"/non-authorizing/reports/{node_id}.json",
            "report_sha256": _local_digest(f"parent-report:{node_id}"),
            "checkpoint_path": f"/non-authorizing/checkpoints/{node_id}.pt",
            "checkpoint_sha256": digest, "checkpoint_byte_sha256": digest,
        }

    teachers = {
        node: row(node, teacher=True) for node in IMPORTED_TEACHERS
    }
    controls = {
        node: row(node, teacher=False) for node in IMPORTED_LOGIT_CONTROLS
    }
    artifact = with_content_hash({
        "contract": PARENT_IMPORT_CONTRACT,
        "schema_version": 2,
        "parents": {
            name: _local_digest(f"parent-import:{name}")
            for name in PARENT_AUTHORITY_PARENT_KEYS
        },
        "payload": {
            "parent_source_commit": "0" * 40,
            "parent_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v8",
            "parent_campaign_mode": "pilot",
            "parent_execution_scope": "parent_prefix_through_finalist_lock",
            "parent_recipe_contract": "HCWDL_RECIPE/v4",
            "endpoint_continuation": "preauthorized_automatic",
            "training_passes": 60,
            "validation_every_passes": 1,
            "parent_train_rows": 300000,
            "terminal_task_id": "finalist_lock",
            "execution_lock_authorized": False,
            "final_test_access_authorized": False,
            "registered_final_test_tasks": 0,
            "teachers": [teachers[node] for node in sorted(teachers)],
            "logit_controls": [controls[node] for node in sorted(controls)],
            "authority_derived_from_registered_files": True,
            "complete": True,
        },
    })
    digest = validate_parent_import(artifact)
    return {
        "work_kind": "parent_import_contract_gate",
        "parent_import_sha256": digest,
        "parent_import_contract": PARENT_IMPORT_CONTRACT,
        "parent_import_schema_version": 2,
        "teacher_count": len(IMPORTED_TEACHERS),
        "logit_control_count": len(IMPORTED_LOGIT_CONTROLS),
        "nonauthorizing_synthetic_v3_fixture": True,
        "authority_files_reopened": False,
    }


@lru_cache(maxsize=1)
def _local_graph_control_gate() -> Mapping[str, Any]:
    from .hcwdl_representation_graph import (
        CONTROL_REGISTRY, NODE_REGISTRY, ascent_graph_artifact,
        control_registry_artifact, validate_ascent_graph_artifact,
        validate_control_registry_artifact,
    )

    graph = ascent_graph_artifact(parents={
        "parent_graph": _local_digest("parent-graph"),
        "parent_import": _local_digest("parent-import"),
    })
    graph_hash = validate_ascent_graph_artifact(graph)
    controls = control_registry_artifact(ascent_graph_artifact_sha256=graph_hash)
    control_hash = validate_control_registry_artifact(
        controls, ascent_graph_artifact_sha256=graph_hash,
    )
    return {
        "work_kind": "dense_descent_graph_and_empty_control_registry_gate",
        "graph_sha256": graph_hash, "control_registry_sha256": control_hash,
        "node_count": len(NODE_REGISTRY), "control_count": len(CONTROL_REGISTRY),
    }


@lru_cache(maxsize=1)
def _local_recipe_gate() -> Mapping[str, Any]:
    from .hcwdl_representation_recipe import (
        example_representation_recipe, validate_representation_recipe,
    )

    recipe = example_representation_recipe()
    return {
        "work_kind": "frozen_representation_recipe_gate",
        "recipe_sha256": validate_representation_recipe(recipe),
        "payload_sha256": canonical_sha256(recipe["payload"]),
    }


@lru_cache(maxsize=1)
def _local_shuffle_gate() -> Mapping[str, Any]:
    from .hcwdl_representation_controls import (
        build_within_class_shuffle_map, validate_within_class_shuffle_map,
    )

    identities = tuple(_local_digest(f"shuffle-row:{index}") for index in range(30))
    labels = np.repeat(np.arange(15, dtype=np.int64), 2)
    artifact, mapping = build_within_class_shuffle_map(
        identity_sha256=identities, labels=labels,
        split_manifest_sha256=_local_digest("shuffle-split"),
        row_selection_sha256=_local_digest("shuffle-selection"),
        parent_hashes={"local_fixture": _local_digest("shuffle-parent")},
    )
    digest = validate_within_class_shuffle_map(
        artifact, mapping, identity_sha256=identities, labels=labels,
    )
    return {
        "work_kind": "within_class_shuffle_map_gate",
        "shuffle_map_sha256": digest, "rows": len(mapping),
        "no_fixed_points": bool(np.all(mapping != np.arange(len(mapping)))),
        "all_classes_preserved": bool(np.array_equal(labels, labels[mapping])),
    }


@lru_cache(maxsize=4)
def _local_reporting_pipeline(campaign_sha256: str) -> Mapping[str, Any]:
    """Build every validation reporting artifact through the real builders."""

    from .hcwdl_paired_bootstrap import BASE_METRICS
    from .hcwdl_representation_graph import CONTROL_REGISTRY, NODE_REGISTRY
    from .hcwdl_representation_reporting import (
        build_confirmation_aggregate, build_confirmation_registry,
        build_screen_aggregate, build_validation_only_aggregate,
    )

    graph_sha = _local_graph_control_gate()["graph_sha256"]
    recipe_sha = _local_recipe_gate()["recipe_sha256"]

    def metrics(offset: float = 0.0) -> dict[str, float]:
        values = {
            "cross_entropy": 0.70 - offset,
            "accuracy": 0.70 + offset,
            "balanced_accuracy": 0.60 + offset,
            "macro_ovr_auc": 0.90 + offset,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 5.0 + offset,
            "multiclass_brier": 0.18 - offset,
            "top_label_ece_15_bin": 0.03 - offset / 10.0,
        }
        if set(values) != set(BASE_METRICS):
            raise RuntimeError("local reporting metric registry differs")
        return values

    primary_ids = tuple(sorted(NODE_REGISTRY))
    control_ids = tuple(sorted(CONTROL_REGISTRY))
    primary_reports = []
    for index, node_id in enumerate(primary_ids):
        node = NODE_REGISTRY[node_id]
        primary_reports.append(with_content_hash({
            "contract": "HCWDL_LOCAL_EXACT_REPORT_FIXTURE/v1", "schema_version": 1,
            "node_id": node_id, "complete": True,
            "graph_sha256": graph_sha, "recipe_sha256": recipe_sha,
            "parent_counterpart": node.parent_counterpart,
            "validation": metrics(0.001 + index * 0.00001),
        }))
    control_reports = []
    for index, control_id in enumerate(control_ids):
        control = CONTROL_REGISTRY[control_id]
        control_reports.append(with_content_hash({
            "contract": "HCWDL_LOCAL_EXACT_REPORT_FIXTURE/v1", "schema_version": 1,
            "node_id": control_id, "complete": True,
            "graph_sha256": graph_sha, "recipe_sha256": recipe_sha,
            "control_counterpart": control.paired_primary_node,
            "validation": metrics(0.0005 + index * 0.00001),
        }))
    parent_ids = {
        node.parent_counterpart for node in NODE_REGISTRY.values()
    } | {"M0", "D0c", "D0w", "D100", "TOFF"}
    parent_reports = {}
    for node_id in sorted(parent_ids):
        offset = 0.02 if node_id in {"D100", "TOFF"} else 0.0
        parent_reports[node_id] = with_content_hash({
            "contract": "HCWDL_LOCAL_EXACT_PARENT_REPORT_FIXTURE/v1",
            "schema_version": 1, "node_id": node_id,
            "validation": metrics(offset),
        })
    screen = build_screen_aggregate(
        primary_reports=primary_reports, control_reports=control_reports,
        parent_reports=parent_reports, graph_sha256=graph_sha,
        recipe_sha256=recipe_sha, campaign_spec_sha256=campaign_sha256,
        expected_primary_ids=primary_ids, expected_control_ids=control_ids,
    )
    target_sha = _local_digest("reporting-toff-logical-bank")
    registry = build_confirmation_registry(
        screen_sha256=screen["content_hash"], campaign_sha256=campaign_sha256,
        recipe_sha256=recipe_sha,
        target_logical_bank_sha256s={
            objective: target_sha
            for objective in ("RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w")
        },
        objectives=("RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w"),
    )
    confirmation_reports = [
        {
            **dict(row),
            "validation": metrics(0.002 + int(row["seed"]) * 0.000001),
        }
        for row in registry["rows"]
    ]
    confirmation = build_confirmation_aggregate(
        registry=registry, reports=confirmation_reports,
    )
    validation_only = build_validation_only_aggregate(
        screen_aggregate=screen, confirmation_aggregate=confirmation,
        campaign_spec_sha256=campaign_sha256,
        final_disposition_sha256=_local_digest("validation-only-disposition"),
    )
    return {
        "screen": screen, "confirmation_registry": registry,
        "confirmation_aggregate": confirmation,
        "validation_only_aggregate": validation_only,
    }


def _local_zero_coefficient_math() -> dict[str, Any]:
    import torch

    base = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    head = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    base_loss = (2.0 * base - 1.0).square()
    representation_loss = (3.0 * head + 0.5).square()
    (base_loss + 0.0 * representation_loss).backward()
    reference = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    (2.0 * reference - 1.0).square().backward()
    shared_difference = abs(float(base.grad) - float(reference.grad))
    head_gradient = abs(float(head.grad))
    if shared_difference > 1.0e-12 or head_gradient != 0.0:
        raise RuntimeError("zero-coefficient local gradient gate differs")
    return {
        "work_kind": "zero_coefficient_gradient_disconnect_gate",
        "shared_gradient_max_abs": shared_difference,
        "representation_head_gradient_max_abs": head_gradient,
        "installed_weaver_available": False,
        "authorization_capable": False,
        "local_math_surface_exercised": True,
    }


@lru_cache(maxsize=1)
def _local_full_loss_probe() -> Mapping[str, Any]:
    from .hcwdl_representation_smoke import run_scientific_full_loss_probe

    return run_scientific_full_loss_probe(device="cpu")


def local_scientific_probe() -> dict[str, Any]:
    """Return the one cached bounded scientific probe for local smoke reports."""

    return copy.deepcopy(dict(_local_full_loss_probe()))


def _local_training_work(task: Any, array_index: int | None) -> dict[str, Any]:
    key, kind = _task_identity(task)
    execution_id = str(getattr(task, "graph_node", ""))
    if not execution_id:
        raise ValueError("local representation training row lacks its graph node")
    probe = _local_full_loss_probe()
    cases = {row["execution_id"]: row for row in probe["cases"]}
    if execution_id not in cases:
        raise ValueError("local representation training row is not in the frozen graph")
    case = cases[execution_id]
    return {
        "work_kind": "full_representation_loss_backward",
        "execution_id": execution_id,
        "array_index": array_index,
        "task_kind": kind,
        "total_loss": case["total_loss"],
        "representation_loss": case["representation_loss"],
        "head_gradient_norms": copy.deepcopy(case["head_gradient_norms"]),
        "probe_sha256": probe["content_hash"],
    }


def execute_local_planning_work(
    spec: Mapping[str, Any], task: Any, array_index: int | None,
) -> dict[str, Any]:
    """Perform bounded real work for one non-final registered row."""

    from .hcwdl_representation_smoke import FINAL_ROLE_KINDS

    key, kind = _task_identity(task)
    forbidden = set(FINAL_ROLE_KINDS) | {"reservation", "finalist_lock"}
    if kind in forbidden:
        raise PermissionError(
            f"local planning fixture cannot invoke final/reservation task {key} ({kind})"
        )
    if kind in {"train_node", "train_control", "confirmation"}:
        work = _local_training_work(task, array_index)
    elif kind in {
        "target_build", "cache_miniature", "target_cleanup",
        "cache_miniature_bank",
    }:
        work = _local_target_lifecycle(task, array_index)
    elif kind == "parent_loss_attestation":
        work = _local_parent_loss()
    elif kind == "parent_import":
        work = copy.deepcopy(dict(_local_parent_import_gate()))
    elif kind in {"kernel_resources", "numerical_acceptance"}:
        work = _local_kernel_math()
    elif kind == "representation_recipe":
        work = copy.deepcopy(dict(_local_recipe_gate()))
    elif kind == "control_registry":
        work = copy.deepcopy(dict(_local_graph_control_gate()))
    elif kind == "shuffle_map":
        work = copy.deepcopy(dict(_local_shuffle_gate()))
    elif kind in {
        "screen_aggregate", "confirmation_registry", "confirmation_aggregate",
        "validation_only_aggregate",
    }:
        reporting = _local_reporting_pipeline(str(spec["content_hash"]))
        report_key = {
            "screen_aggregate": "screen",
            "confirmation_registry": "confirmation_registry",
            "confirmation_aggregate": "confirmation_aggregate",
            "validation_only_aggregate": "validation_only_aggregate",
        }[kind]
        artifact = reporting[report_key]
        work = {
            "work_kind": f"{kind}_builder_gate",
            "artifact_contract": artifact["contract"],
            "artifact_sha256": artifact["content_hash"],
        }
    elif kind == "tap_schema":
        from hlt_classification.models.hcwdl_surfaces import tap_schema

        work = {
            "work_kind": "tap_schema_materialization",
            "tap_schema_sha256": canonical_sha256(tap_schema()),
        }
    elif kind == "zero_coefficient_acceptance":
        work = _local_zero_coefficient_math()
    elif kind in {"surface_parity", "architecture_attestation", "smoke_probe"}:
        probe = _local_full_loss_probe()
        work = {
            "work_kind": f"{kind}_local_model_surface_gate",
            "probe_sha256": probe["content_hash"],
            "primary_count": probe["primary_count"],
            "control_count": probe["control_count"],
            "installed_weaver_available": False,
            "authorization_capable": False,
            "local_validator_math_exercised": True,
        }
    else:
        raise KeyError(f"local semantic coverage has no task-specific dispatch for {kind!r}")
    semantic_surface = LOCAL_SEMANTIC_COVERAGE.get(kind)
    if not semantic_surface:
        raise KeyError(f"local semantic coverage registry omits {kind!r}")
    result = with_content_hash({
        "contract": LOCAL_PLANNING_WORK_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "task_key": key,
        "task_kind": kind,
        "array_index": array_index,
        "semantic_surface": semantic_surface,
        "semantic_fixture_executed": True,
        "generic_fallback": False,
        "work": work,
        "work_sha256": canonical_sha256(work),
        "scientific_authorization": False,
        "production_handler_invoked": False,
        "final_role_accessed": False,
    })
    validate_content_hash(
        result, expected_contract=LOCAL_PLANNING_WORK_CONTRACT,
        expected_schema_version=1,
    )
    return result


@lru_cache(maxsize=1)
def _local_synthetic_final_probe() -> Mapping[str, Any]:
    """Run the complete shared-final implementation over temporary arrays."""

    from .hcwdl_representation_synthetic_final import run_synthetic_final_pipeline

    with tempfile.TemporaryDirectory(prefix="hcwdl-rkd-synthetic-final-") as directory:
        report = run_synthetic_final_pipeline(Path(directory))
    return copy.deepcopy(dict(report))


def execute_local_synthetic_final_work(
    spec: Mapping[str, Any], task: Any, array_index: int | None,
) -> dict[str, Any]:
    """Bind one registered row to a real nonauthorizing synthetic final run."""

    from .hcwdl_representation_smoke import FINAL_ROLE_KINDS

    key, kind = _task_identity(task)
    if kind not in set(FINAL_ROLE_KINDS) | {"reservation", "finalist_lock"}:
        raise ValueError("local synthetic-final handler received a nonfinal task")
    registered_inputs = tuple(getattr(task, "registered_inputs", ()))
    registered_outputs = tuple(getattr(task, "registered_outputs", ()))
    if not registered_inputs or not registered_outputs:
        raise ValueError("local final structural row lacks registered I/O")
    if len(set(registered_inputs)) != len(registered_inputs) or len(
        set(registered_outputs)
    ) != len(registered_outputs):
        raise ValueError("local final structural row repeats registered I/O")
    probe = _local_synthetic_final_probe()
    proof = {
        "task_key": key,
        "task_kind": kind,
        "array_index": array_index,
        "registered_inputs_sha256": canonical_sha256(list(registered_inputs)),
        "registered_outputs_sha256": canonical_sha256(list(registered_outputs)),
        "array_registry": getattr(task, "array_registry", None),
        "synthetic_final_probe_sha256": probe["content_hash"],
        "synthetic_final_evidence_sha256": probe["evidence_sha256"],
        "full_shared_final_semantics_exercised": probe[
            "full_shared_final_semantics_exercised"
        ],
        "structural_only": False,
        "synthetic_final_pipeline": True,
        "semantic_surface": LOCAL_SEMANTIC_COVERAGE[kind],
        "semantic_fixture_executed": True,
        "generic_fallback": False,
        "final_role_accessed": False,
        "scientific_authorization": False,
        "production_handler_invoked": False,
    }
    return with_content_hash({
        "contract": LOCAL_PLANNING_WORK_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        **proof,
    })


def build_local_planning_handlers(
    task_kinds: Sequence[str],
) -> dict[str, Callable[[Mapping[str, Any], Any, int | None], Mapping[str, Any]]]:
    """Return the closed real-work/nonauthorizing local handler registry."""

    from .hcwdl_representation_smoke import FINAL_ROLE_KINDS
    from .hcwdl_representation_task_runtime import TASK_KINDS

    if set(LOCAL_SEMANTIC_COVERAGE) != set(TASK_KINDS):
        missing = sorted(set(TASK_KINDS) - set(LOCAL_SEMANTIC_COVERAGE))
        extra = sorted(set(LOCAL_SEMANTIC_COVERAGE) - set(TASK_KINDS))
        raise RuntimeError(
            f"local semantic coverage registry differs; missing={missing}, extra={extra}"
        )
    forbidden_labels = ("roundtrip", "structural", "mock", "schema_only")
    contaminated = {
        kind: surface for kind, surface in LOCAL_SEMANTIC_COVERAGE.items()
        if any(label in surface for label in forbidden_labels)
    }
    if contaminated:
        raise RuntimeError(
            f"local semantic coverage contains generic/mock surfaces: {contaminated}"
        )

    kinds = {str(value) for value in task_kinds}
    if not kinds or any(not value for value in kinds):
        raise ValueError("local planning task-kind registry is empty or invalid")
    unknown = kinds - set(TASK_KINDS)
    if unknown:
        raise KeyError(f"local planning task kinds are not built in: {sorted(unknown)}")
    forbidden = set(FINAL_ROLE_KINDS) | {"reservation", "finalist_lock"}
    return {
        kind: (
            execute_local_synthetic_final_work if kind in forbidden
            else execute_local_planning_work
        )
        for kind in sorted(kinds)
    }


from .hcwdl_representation_campaign_adapters import (
    cache_miniature_adapter,
    cache_miniature_bank_adapter,
    control_registry_adapter,
    shuffle_map_adapter,
    zero_coefficient_acceptance_adapter,
)


from .hcwdl_representation_production import (
    assignment_finalize_adapter, assignment_shard_adapter,
    execution_lock_adapter, final_aggregate_adapter, final_selection_adapter,
    metric_join_adapter, prediction_finalize_adapter, prediction_shard_adapter,
    target_build_adapter, training_adapter, validation_only_aggregate_adapter,
)


PRODUCTION_ADAPTERS: Final[Mapping[str, Callable[..., Any]]] = {
    "tap_schema": _tap_schema_adapter,
    "surface_parity": _surface_parity_adapter,
    "architecture_attestation": _architecture_attestation_adapter,
    "parent_loss_attestation": _parent_loss_attestation_adapter,
    "parent_import": _parent_import_adapter,
    "dense_teacher_import": _dense_teacher_import_adapter,
    "control_registry": control_registry_adapter,
    "kernel_resources": _kernel_resources_adapter,
    "representation_recipe": _representation_recipe_adapter,
    "numerical_acceptance": _numerical_acceptance_adapter,
    "cache_miniature_bank": cache_miniature_bank_adapter,
    "cache_miniature": cache_miniature_adapter,
    "target_build": target_build_adapter,
    "smoke_probe": _smoke_probe_adapter,
    "zero_coefficient_acceptance": zero_coefficient_acceptance_adapter,
    "reservation": _reservation_adapter,
    "train_node": training_adapter,
    "train_control": training_adapter,
    "shuffle_map": shuffle_map_adapter,
    "target_cleanup": _target_cleanup_adapter,
    "screen_aggregate": _screen_aggregate_adapter,
    "confirmation_registry": _confirmation_registry_adapter,
    "confirmation": training_adapter,
    "confirmation_aggregate": _confirmation_aggregate_adapter,
    "finalist_lock": _finalist_lock_adapter,
    "shared_final_claim": _shared_final_claim_adapter,
    "final_selection": final_selection_adapter,
    "assignment_shard": assignment_shard_adapter,
    "assignment_finalize": assignment_finalize_adapter,
    "data_attestation": _data_attestation_adapter,
    "execution_lock": execution_lock_adapter,
    "prediction_shard": prediction_shard_adapter,
    "prediction_finalize": prediction_finalize_adapter,
    "metric_join": metric_join_adapter,
    "final_aggregate": final_aggregate_adapter,
    "validation_only_aggregate": validation_only_aggregate_adapter,
    "dense_training_aggregate": _dense_training_aggregate_adapter,
}


__all__ = [
    "LOCAL_PLANNING_WORK_CONTRACT", "LOCAL_SEMANTIC_COVERAGE", "PRODUCTION_ADAPTERS",
    "PRODUCTION_ADAPTER_CONTRACT", "ProductionConfigurationError",
    "build_installed_weaver_surface_parity_artifact",
    "build_local_planning_handlers", "execute_local_planning_work",
    "execute_local_synthetic_final_work", "local_scientific_probe",
]
