"""Bounded validation-only execution proxy for HCWDL-RKD acceptance.

This module deliberately does not reuse the shared-final implementation.  A
validation proxy has one label-bearing selection read, followed by independent
label-free assignment and model-input reads.  Validation labels remain
in-process: the immutable record contains only their logical digest and class
counts, never the raw values.  It does contain actual selected source
identities, FP32 logits, metrics, and deterministic bootstrap sidecars.

The proxy is acceptance mechanics only.  It cannot grant pilot authority and
never consumes a final-role capability, reservation, claim, escrow, finalist
lock, or execution lock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256,
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_paired_bootstrap import (
    BOOTSTRAP_SEED,
    DEFAULT_METRICS,
    PAIRED_BOOTSTRAP_CONTRACT,
    paired_classification_bootstrap,
)
from .hcwdl_representation_contracts import (
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
    VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
)
from .hcwdl_representation_worker_runtime import validate_live_worker_runtime
from .identity import normalize_source_path
from .repair import full_endpoint_required_branches
from .schema import (
    BASELINE_BRANCHES,
    LABEL_BRANCHES,
    TREE_NAME,
    hlt_required_branches,
    matching_required_branches,
    native_offline_required_branches,
)


VALIDATION_PROXY_PATHS: Final = ("hlt", "shell_exact", "native_offline")
VALIDATION_PROXY_VIEW_REGISTRY: Final = {
    "hlt": "D0c",
    "shell_exact": "D100",
    "native_offline": "TOFF",
}
VALIDATION_PROXY_MODEL_IDS: Final = {
    "hlt": "D0c",
    "shell_exact": "D100",
    "native_offline": "TOFF",
}
VALIDATION_PROXY_REGISTERED_INPUTS: Final = (
    "matcher_resources",
    "parent_campaign_spec",
    "parent_import",
    "parent_reports",
    "parent_source_manifest",
    "split_manifest",
    "validation_assignment_manifest",
)
VALIDATION_PROXY_BOOTSTRAP_REPLICATES: Final = 2_000
VALIDATION_PROXY_BOOTSTRAP_METRICS: Final = (
    *DEFAULT_METRICS,
    "acceptance_proxy_nonfinal_marker",
)
VALIDATION_PROXY_COMPARISONS: Final = (
    ("D100_minus_D0c", "shell_exact", "hlt"),
    ("TOFF_minus_D0c", "native_offline", "hlt"),
)
VALIDATION_SELECTION_BRANCHES: Final = frozenset(BASELINE_BRANCHES) | frozenset(
    LABEL_BRANCHES
)
VALIDATION_ASSIGNMENT_BRANCHES: Final = matching_required_branches()
VALIDATION_HLT_BRANCHES: Final = hlt_required_branches()
VALIDATION_SHELL_EXACT_BRANCHES: Final = (
    hlt_required_branches() | full_endpoint_required_branches()
)
VALIDATION_NATIVE_OFFLINE_BRANCHES: Final = native_offline_required_branches()
VALIDATION_BRANCH_ALLOWLISTS: Final = {
    "selection": VALIDATION_SELECTION_BRANCHES,
    "assignment": VALIDATION_ASSIGNMENT_BRANCHES,
    "hlt": VALIDATION_HLT_BRANCHES,
    "shell_exact": VALIDATION_SHELL_EXACT_BRANCHES,
    "native_offline": VALIDATION_NATIVE_OFFLINE_BRANCHES,
}

_ROLE_CAPS: Final = {"train": 512, "validation": 256, "final_test": 0}
_VALIDATION_ACTION_BASE: Final = {
    "action_id": "validation_proxy",
    "kind": "validation_proxy",
    "dependencies": [],
    "worker_role": "deterministic",
    "resource_class": "gpu_final_prediction",
    "scalar_only": True,
    "array": None,
    "train_rows": 0,
    "validation_rows": 256,
    "final_rows": 0,
    "replicate_seed": None,
    "effective_batch_size": None,
    "maximum_optimizer_updates": 0,
    "execution_id": None,
    "target_identity": None,
    "mode": "acceptance",
    "campaign_task_kind": None,
    "final_role_access_authorized": False,
}
_VALIDATION_ACTION: Final = {
    **_VALIDATION_ACTION_BASE,
    "action_spec_sha256": canonical_sha256(_VALIDATION_ACTION_BASE),
}
_LABEL_KEYS: Final = frozenset(
    {"label", "labels", "target", "targets", *LABEL_BRANCHES}
)
_FORBIDDEN_AUTHORITY_FIELDS: Final = frozenset({
    "capability", "claim", "escrow", "execution_lock", "finalist_lock",
    "final_selection",
})
_FORBIDDEN_ROUTE_COMPONENTS: Final = frozenset({
    "final", "final_test", "shared_final", "reservation", "capability",
    "escrow", "execution_lock", "finalist_lock", "final_selection",
})

if any(
    set(branches) & set(LABEL_BRANCHES)
    for name, branches in VALIDATION_BRANCH_ALLOWLISTS.items()
    if name != "selection"
):
    raise RuntimeError("validation-only label-free allow-list contains a label branch")


AuthorityValidator = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class ValidationAccessRequest:
    """The complete input given to a validation reader or predictor."""

    authority_sha256: str
    role: str
    phase: str
    path: str
    projected_branches: tuple[str, ...]
    row_limit: int
    selected_identities: tuple[str, ...]
    labels_allowed: bool
    view_id: str | None
    model_id: str | None
    checkpoint_sha256: str | None
    model_source_lineage_sha256: str | None


@dataclass(frozen=True)
class ValidationReadResult:
    """Rows plus the concrete source ranges opened by one reader."""

    rows: Sequence[Mapping[str, Any]]
    source_accesses: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ValidationModelRow:
    """A label-free model input passed to a predictor."""

    identity_digest: str
    model_inputs: Any


SelectionReader = Callable[[ValidationAccessRequest], ValidationReadResult]
AssignmentReader = Callable[[ValidationAccessRequest], ValidationReadResult]
ModelReader = Callable[
    [ValidationAccessRequest, Mapping[str, Any] | None], ValidationReadResult
]
Predictor = Callable[[ValidationAccessRequest, Sequence[ValidationModelRow]], np.ndarray]


def _source_commit(value: object) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("validation proxy source commit must be a full lowercase Git SHA")
    return commit


def _validate_authority(
    authority: Mapping[str, Any], *, authority_validator: AuthorityValidator,
) -> dict[str, Any]:
    if not callable(authority_validator):
        raise TypeError("validation proxy requires an authority validator")
    digest = validate_content_hash(
        authority,
        expected_contract=NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        expected_schema_version=1,
    )
    if authority_validator(authority) != digest:
        raise ValueError("validation proxy authority validator returned a different digest")
    required = {
        "contract", "schema_version", "content_hash", "source_commit",
        "representation_recipe_sha256", "role_caps", "actions",
        "effective_batch_size", "maximum_optimizer_updates",
        "bounded_action_execution_authorized", "action_inputs_sha256",
    }
    if not required <= set(authority):
        raise ValueError("validation proxy authority fields are incomplete")
    if set(authority) & _FORBIDDEN_AUTHORITY_FIELDS:
        raise PermissionError("validation proxy authority contains a final-only key")
    caps = authority.get("role_caps")
    if not isinstance(caps, Mapping) or dict(caps) != _ROLE_CAPS:
        raise PermissionError("validation proxy authority role caps differ")
    registry = authority.get("actions")
    if not isinstance(registry, Mapping) or registry.get("validation_proxy") != _VALIDATION_ACTION:
        raise PermissionError("validation proxy action authority differs")
    action = registry["validation_proxy"]
    if not isinstance(action, Mapping) or set(action) != set(_VALIDATION_ACTION):
        raise PermissionError("validation proxy action fields differ")
    false_authority_claims = (
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
    )
    if any(authority.get(name) is not False for name in false_authority_claims):
        raise PermissionError("validation proxy authority boundary differs")
    if (
        authority.get("effective_batch_size") != 256
        or authority.get("maximum_optimizer_updates") != 2
        or authority.get("bounded_action_execution_authorized") is not True
    ):
        raise PermissionError("validation proxy authority execution bounds differ")
    return {
        "authority_sha256": digest,
        "source_commit": _source_commit(authority["source_commit"]),
        "representation_recipe_sha256": require_sha256(
            authority["representation_recipe_sha256"],
            name="validation proxy representation recipe",
        ),
        "rows": int(caps["validation"]),
        "maximum_rows": 4096,
        "action_spec_sha256": action["action_spec_sha256"],
        "action_inputs_sha256": require_sha256(
            authority["action_inputs_sha256"], name="validation action inputs",
        ),
    }


def _request(
    authority_view: Mapping[str, Any], *, phase: str, path: str,
    selected_identities: Sequence[str] = (),
    model_binding: Mapping[str, str] | None = None,
) -> ValidationAccessRequest:
    if path not in VALIDATION_BRANCH_ALLOWLISTS:
        raise ValueError("unknown validation proxy input path")
    labels_allowed = path == "selection"
    if path in VALIDATION_PROXY_PATHS:
        if model_binding is None:
            raise ValueError("validation prediction request lacks a model binding")
        view_id = model_binding["view_id"]
        model_id = model_binding["model_id"]
        checkpoint_sha256 = model_binding["checkpoint_sha256"]
        model_source_lineage_sha256 = model_binding[
            "model_source_lineage_sha256"
        ]
    else:
        if model_binding is not None:
            raise ValueError("nonprediction validation request received a model binding")
        view_id = model_id = checkpoint_sha256 = model_source_lineage_sha256 = None
    return ValidationAccessRequest(
        authority_sha256=str(authority_view["authority_sha256"]),
        role="validation",
        phase=phase,
        path=path,
        projected_branches=tuple(sorted(VALIDATION_BRANCH_ALLOWLISTS[path])),
        row_limit=int(authority_view["rows"]),
        selected_identities=tuple(selected_identities),
        labels_allowed=labels_allowed,
        view_id=view_id,
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha256,
        model_source_lineage_sha256=model_source_lineage_sha256,
    )


def _model_bindings(value: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(VALIDATION_PROXY_PATHS):
        raise ValueError("validation proxy model-binding registry differs")
    normalized: dict[str, dict[str, str]] = {}
    for path in VALIDATION_PROXY_PATHS:
        row = value[path]
        if not isinstance(row, Mapping) or set(row) != {
            "view_id", "model_id", "checkpoint_sha256",
            "model_source_lineage_sha256",
        }:
            raise ValueError("validation proxy model-binding fields differ")
        view_id = str(row["view_id"])
        model_id = str(row["model_id"])
        if view_id != VALIDATION_PROXY_VIEW_REGISTRY[path] or not model_id:
            raise ValueError("validation proxy view/model identity differs")
        normalized[path] = {
            "view_id": view_id,
            "model_id": model_id,
            "checkpoint_sha256": require_sha256(
                row["checkpoint_sha256"], name=f"{path} checkpoint",
            ),
            "model_source_lineage_sha256": require_sha256(
                row["model_source_lineage_sha256"],
                name=f"{path} model-source lineage",
            ),
        }
    return normalized


def build_validation_proxy_input_lineage(
    *, authority_sha256: str, action_inputs_sha256: str,
    source_runtime_row_sha256: str, action_assembly_sha256: str,
    bounded_row_selection_sha256: str, parent_campaign_spec_sha256: str,
    source_manifest_sha256: str, split_manifest_sha256: str,
    matcher_resources_sha256: str, validation_assignment_manifest_sha256: str,
    parent_import_sha256: str,
    registered_input_bytes_sha256: Mapping[str, str],
    model_report_sha256: Mapping[str, str],
    model_source_lineage: Mapping[str, Mapping[str, Any]],
    live_worker_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every reopened non-final input used by the production proxy."""

    registered = {
        str(name): require_sha256(value, name=f"validation input bytes {name}")
        for name, value in sorted(registered_input_bytes_sha256.items())
    }
    if tuple(registered) != VALIDATION_PROXY_REGISTERED_INPUTS:
        raise ValueError("validation proxy registered input registry differs")
    reports = {
        str(name): require_sha256(value, name=f"validation model report {name}")
        for name, value in sorted(model_report_sha256.items())
    }
    if set(reports) != set(VALIDATION_PROXY_MODEL_IDS.values()):
        raise ValueError("validation proxy model report registry differs")
    source_lineage: dict[str, dict[str, Any]] = {}
    if not isinstance(model_source_lineage, Mapping) or set(
        model_source_lineage
    ) != set(VALIDATION_PROXY_MODEL_IDS.values()):
        raise ValueError("validation proxy model-source registry differs")
    for node_id, row in sorted(model_source_lineage.items()):
        if not isinstance(row, Mapping):
            raise ValueError("validation proxy model-source row differs")
        _validate_nested_record(row, expected_kind="validation_proxy_model_source")
        expected_source_fields = {
            "kind", "schema_version", "node_id", "source_kind",
            "parent_import_row_sha256", "wrapper_report_content_sha256",
            "wrapper_report_byte_sha256", "engine_report_content_sha256",
            "engine_report_byte_sha256", "wrapper_execution_config_sha256",
            "engine_execution_config_sha256", "engine_config_sha256",
            "model_extraction_sha256",
            "checkpoint_sha256", "content_hash",
        }
        if (
            set(row) != expected_source_fields
            or row.get("node_id") != node_id
            or row.get("source_kind") != "authenticated_pmard_parent_v1"
            or row.get("wrapper_report_content_sha256") != reports[node_id]
        ):
            raise ValueError("validation proxy model-source lineage differs")
        for name in expected_source_fields - {
            "kind", "schema_version", "node_id", "source_kind", "content_hash",
        }:
            require_sha256(row[name], name=f"{node_id} {name}")
        source_lineage[node_id] = dict(row)
    runtime_sha256 = validate_live_worker_runtime(live_worker_runtime)
    if (
        live_worker_runtime.get("resource_class") != "gpu_final_prediction"
        or live_worker_runtime.get("row_device") != "cuda"
        or live_worker_runtime.get("deterministic_worker") is not True
    ):
        raise PermissionError("validation proxy live worker route differs")
    return with_content_hash({
        "kind": "validation_proxy_authenticated_inputs",
        "schema_version": 1,
        "authority_sha256": require_sha256(
            authority_sha256, name="validation authority",
        ),
        "action_inputs_sha256": require_sha256(
            action_inputs_sha256, name="validation action inputs",
        ),
        "source_runtime_row_sha256": require_sha256(
            source_runtime_row_sha256, name="validation source runtime row",
        ),
        "action_assembly_sha256": require_sha256(
            action_assembly_sha256, name="validation action assembly",
        ),
        "bounded_row_selection_sha256": require_sha256(
            bounded_row_selection_sha256, name="validation bounded selection",
        ),
        "parent_campaign_spec_sha256": require_sha256(
            parent_campaign_spec_sha256, name="validation parent campaign",
        ),
        "source_manifest_sha256": require_sha256(
            source_manifest_sha256, name="validation source manifest",
        ),
        "split_manifest_sha256": require_sha256(
            split_manifest_sha256, name="validation split manifest",
        ),
        "matcher_resources_sha256": require_sha256(
            matcher_resources_sha256, name="validation matcher resources",
        ),
        "validation_assignment_manifest_sha256": require_sha256(
            validation_assignment_manifest_sha256,
            name="validation assignment manifest",
        ),
        "parent_import_sha256": require_sha256(
            parent_import_sha256, name="validation parent import",
        ),
        "registered_input_bytes_sha256": registered,
        "model_report_sha256": reports,
        "model_source_lineage": source_lineage,
        "live_worker_runtime": dict(live_worker_runtime),
        "live_worker_runtime_sha256": runtime_sha256,
        "derivation_kind": "canonical_bounded_validation_projection_v1",
        "role": "validation",
        "rows": 256,
        "final_role_accessed": False,
        "shared_final_accessed": False,
    })


def validate_validation_proxy_input_lineage(
    value: Mapping[str, Any], *, authority_sha256: str,
    action_inputs_sha256: str,
) -> str:
    digest = _validate_nested_record(
        value, expected_kind="validation_proxy_authenticated_inputs",
    )
    expected_fields = {
        "kind", "schema_version", "authority_sha256", "action_inputs_sha256",
        "source_runtime_row_sha256", "action_assembly_sha256",
        "bounded_row_selection_sha256", "parent_campaign_spec_sha256",
        "source_manifest_sha256", "split_manifest_sha256",
        "matcher_resources_sha256", "validation_assignment_manifest_sha256",
        "parent_import_sha256", "registered_input_bytes_sha256",
        "model_report_sha256", "model_source_lineage", "live_worker_runtime",
        "live_worker_runtime_sha256", "derivation_kind", "role", "rows",
        "final_role_accessed", "shared_final_accessed", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("validation proxy input-lineage fields differ")
    hash_fields = (
        "source_runtime_row_sha256", "action_assembly_sha256",
        "bounded_row_selection_sha256", "parent_campaign_spec_sha256",
        "source_manifest_sha256", "split_manifest_sha256",
        "matcher_resources_sha256", "validation_assignment_manifest_sha256",
        "parent_import_sha256",
    )
    for name in hash_fields:
        require_sha256(value[name], name=f"validation lineage {name}")
    registered = value.get("registered_input_bytes_sha256")
    reports = value.get("model_report_sha256")
    source_lineage = value.get("model_source_lineage")
    if (
        not isinstance(registered, Mapping)
        or tuple(registered) != VALIDATION_PROXY_REGISTERED_INPUTS
        or not isinstance(reports, Mapping)
        or set(reports) != set(VALIDATION_PROXY_MODEL_IDS.values())
        or not isinstance(source_lineage, Mapping)
        or set(source_lineage) != set(VALIDATION_PROXY_MODEL_IDS.values())
    ):
        raise ValueError("validation proxy input-lineage registries differ")
    for name, sha256 in (*registered.items(), *reports.items()):
        require_sha256(sha256, name=f"validation lineage {name}")
    for node_id, row in source_lineage.items():
        if not isinstance(row, Mapping):
            raise ValueError("validation proxy model-source lineage row differs")
        _validate_nested_record(row, expected_kind="validation_proxy_model_source")
        expected_source_fields = {
            "kind", "schema_version", "node_id", "source_kind",
            "parent_import_row_sha256", "wrapper_report_content_sha256",
            "wrapper_report_byte_sha256", "engine_report_content_sha256",
            "engine_report_byte_sha256", "wrapper_execution_config_sha256",
            "engine_execution_config_sha256", "engine_config_sha256",
            "model_extraction_sha256",
            "checkpoint_sha256", "content_hash",
        }
        if (
            set(row) != expected_source_fields
            or row.get("node_id") != node_id
            or row.get("source_kind") != "authenticated_pmard_parent_v1"
            or row.get("wrapper_report_content_sha256") != reports[node_id]
        ):
            raise PermissionError("validation proxy model-source lineage differs")
        for name in expected_source_fields - {
            "kind", "schema_version", "node_id", "source_kind", "content_hash",
        }:
            require_sha256(row[name], name=f"{node_id} {name}")
    runtime = value.get("live_worker_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("validation proxy live worker runtime is absent")
    runtime_sha256 = validate_live_worker_runtime(runtime)
    if (
        value.get("authority_sha256") != require_sha256(
            authority_sha256, name="validation authority",
        )
        or value.get("action_inputs_sha256") != require_sha256(
            action_inputs_sha256, name="validation action inputs",
        )
        or value.get("live_worker_runtime_sha256") != runtime_sha256
        or runtime.get("resource_class") != "gpu_final_prediction"
        or runtime.get("row_device") != "cuda"
        or runtime.get("deterministic_worker") is not True
        or value.get("derivation_kind")
        != "canonical_bounded_validation_projection_v1"
        or value.get("role") != "validation"
        or value.get("rows") != 256
        or value.get("final_role_accessed") is not False
        or value.get("shared_final_accessed") is not False
    ):
        raise PermissionError("validation proxy input-lineage authority differs")
    return digest


def _json_value(value: Any, *, name: str) -> Any:
    """Normalize actual bounded payloads without accepting digest placeholders."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value), name=name)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.str_)
        ):
            raise TypeError(f"{name} contains an unsupported array")
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise FloatingPointError(f"{name} contains a nonfinite array")
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise TypeError(f"{name} mapping keys must be nonempty strings")
            normalized[raw_key] = _json_value(item, name=f"{name}.{raw_key}")
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name=name) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise FloatingPointError(f"{name} contains a nonfinite scalar")
        return result
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}")


def _validate_nested_record(
    value: Mapping[str, Any], *, expected_kind: str, schema_version: int = 1,
) -> str:
    """Validate a typed subrecord without minting a standalone contract."""

    if not isinstance(value, Mapping) or value.get("kind") != expected_kind:
        raise ValueError(f"validation proxy nested kind differs: {expected_kind}")
    if value.get("schema_version") != schema_version:
        raise ValueError("validation proxy nested schema version differs")
    supplied = require_sha256(value.get("content_hash"), name="nested content hash")
    unhashed = dict(value)
    unhashed.pop("content_hash", None)
    if canonical_sha256(unhashed) != supplied:
        raise ValueError("validation proxy nested content hash mismatch")
    return supplied


def _contains_label(value: Any) -> bool:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        if any(str(key).lower() in _LABEL_KEYS for key in value):
            return True
        return any(_contains_label(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_label(item) for item in value)
    return False


def _source_accesses(
    raw: Sequence[Mapping[str, Any]], *, request: ValidationAccessRequest,
) -> tuple[dict[str, Any], ...]:
    if not raw:
        raise ValueError(f"validation proxy {request.path} reader reported no source access")
    accesses: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != {
            "source_path", "source_file_sha256", "tree", "entry_start", "entry_stop",
        }:
            raise ValueError("validation proxy source access fields differ")
        start, stop = value["entry_start"], value["entry_stop"]
        if (
            isinstance(start, bool) or not isinstance(start, Integral)
            or isinstance(stop, bool) or not isinstance(stop, Integral)
            or int(start) < 0 or int(stop) <= int(start)
            or value["tree"] != TREE_NAME
        ):
            raise ValueError("validation proxy source access range differs")
        path = normalize_source_path(str(value["source_path"]))
        if any(
            part.lower() in _FORBIDDEN_ROUTE_COMPONENTS
            for part in path.split("/")
        ):
            raise PermissionError("validation proxy source access names a final route")
        accesses.append({
            "source_path": path,
            "source_file_sha256": require_sha256(
                value["source_file_sha256"], name="validation proxy source",
            ),
            "tree": TREE_NAME,
            "entry_start": int(start),
            "entry_stop": int(stop),
        })
    return tuple(accesses)


def _access_artifact(
    result: ValidationReadResult, *, request: ValidationAccessRequest,
) -> dict[str, Any]:
    if not isinstance(result, ValidationReadResult):
        raise TypeError("validation reader must return ValidationReadResult")
    accesses = _source_accesses(result.source_accesses, request=request)
    return with_content_hash({
        "contract": VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
        "schema_version": 1,
        "authority_sha256": request.authority_sha256,
        "role": "validation",
        "phase": request.phase,
        "path": request.path,
        "view_id": request.view_id,
        "model_id": request.model_id,
        "checkpoint_sha256": request.checkpoint_sha256,
        "model_source_lineage_sha256": request.model_source_lineage_sha256,
        "projected_branches": list(request.projected_branches),
        "labels_allowed": request.labels_allowed,
        "label_free": not request.labels_allowed,
        "row_limit": request.row_limit,
        "selected_identity_order_sha256": (
            canonical_sha256(list(request.selected_identities))
            if request.selected_identities else None
        ),
        "source_accesses": list(accesses),
    })


def _validate_access_artifact(
    value: Mapping[str, Any], *, authority_sha256: str, phase: str, path: str,
    rows: int, model_binding: Mapping[str, str] | None = None,
    selected_identities: Sequence[str] = (),
) -> str:
    digest = validate_content_hash(
        value, expected_contract=VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
        expected_schema_version=1,
    )
    expected_request = ValidationAccessRequest(
        authority_sha256=authority_sha256,
        role="validation",
        phase=phase,
        path=path,
        projected_branches=tuple(sorted(VALIDATION_BRANCH_ALLOWLISTS[path])),
        row_limit=rows,
        selected_identities=tuple(selected_identities),
        labels_allowed=path == "selection",
        view_id=None if model_binding is None else model_binding["view_id"],
        model_id=None if model_binding is None else model_binding["model_id"],
        checkpoint_sha256=(
            None if model_binding is None else model_binding["checkpoint_sha256"]
        ),
        model_source_lineage_sha256=(
            None
            if model_binding is None
            else model_binding["model_source_lineage_sha256"]
        ),
    )
    expected = with_content_hash({
        "contract": VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "phase": phase,
        "path": path,
        "view_id": expected_request.view_id,
        "model_id": expected_request.model_id,
        "checkpoint_sha256": expected_request.checkpoint_sha256,
        "model_source_lineage_sha256": (
            expected_request.model_source_lineage_sha256
        ),
        "projected_branches": list(expected_request.projected_branches),
        "labels_allowed": path == "selection",
        "label_free": path != "selection",
        "row_limit": rows,
        "selected_identity_order_sha256": (
            canonical_sha256(list(expected_request.selected_identities))
            if expected_request.selected_identities else None
        ),
        "source_accesses": list(_source_accesses(
            value.get("source_accesses", ()), request=expected_request,
        )),
    })
    if dict(value) != expected:
        raise ValueError("validation proxy access artifact is not canonical")
    return digest


def _selection_artifact(
    rows: Sequence[Mapping[str, Any]], *, authority_sha256: str, expected_rows: int,
) -> tuple[dict[str, Any], tuple[str, ...], np.ndarray]:
    if len(rows) != expected_rows or expected_rows <= 0 or expected_rows > 4096:
        raise ValueError("validation proxy selection size differs from authority")
    selected: list[dict[str, Any]] = []
    for value in rows:
        if not isinstance(value, Mapping) or set(value) != {
            "identity_digest", "label", "source_path", "source_file_sha256",
            "source_entry",
        }:
            raise ValueError("validation proxy selection row fields differ")
        identity = require_sha256(value["identity_digest"], name="validation identity")
        source_path = normalize_source_path(str(value["source_path"]))
        source_sha256 = require_sha256(
            value["source_file_sha256"], name="validation source file",
        )
        entry = value["source_entry"]
        if (
            isinstance(entry, bool) or not isinstance(entry, Integral)
            or int(entry) < 0
        ):
            raise ValueError("validation proxy selection source entry differs")
        if identity != canonical_sha256({
            "source_file_sha256": source_sha256,
            "source_entry": int(entry),
        }):
            raise ValueError("validation proxy identity is not source-derived")
        label = value["label"]
        if isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= int(label) < 15:
            raise ValueError("validation proxy selection label lies outside 0..14")
        selected.append({
            "identity_digest": identity,
            "source_path": source_path,
            "source_file_sha256": source_sha256,
            "source_entry": int(entry),
        })
    identities = tuple(row["identity_digest"] for row in selected)
    source_identities = tuple(
        (row["source_path"], row["source_file_sha256"], row["source_entry"])
        for row in selected
    )
    if (
        len(identities) != len(set(identities))
        or len(source_identities) != len(set(source_identities))
    ):
        raise ValueError("validation proxy selection repeats an identity")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    counts = np.bincount(labels, minlength=15)
    if np.any(counts == 0):
        raise ValueError("validation proxy requires all fifteen classes")
    artifact = with_content_hash({
        "kind": "validation_selection",
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "rows": len(selected),
        "class_counts": counts.tolist(),
        "selected_rows": selected,
        "identity_order_sha256": canonical_sha256(list(identities)),
        "source_membership_sha256": canonical_sha256([
            {
                "source_path": row["source_path"],
                "source_file_sha256": row["source_file_sha256"],
                "source_entry": row["source_entry"],
            }
            for row in selected
        ]),
        "label_vector_logical_sha256": array_sha256(
            "validation_labels", labels,
        ),
        "raw_labels_published": False,
        "labels_opened_by_selection_only": True,
    })
    return artifact, identities, labels


def _validate_selection_artifact(
    value: Mapping[str, Any], *, authority_sha256: str, expected_rows: int,
) -> tuple[str, tuple[str, ...], str]:
    digest = _validate_nested_record(value, expected_kind="validation_selection")
    expected_fields = {
        "kind", "schema_version", "authority_sha256", "role", "rows",
        "class_counts", "selected_rows", "identity_order_sha256",
        "source_membership_sha256", "label_vector_logical_sha256",
        "raw_labels_published", "labels_opened_by_selection_only",
        "content_hash",
    }
    rows = value.get("selected_rows")
    counts = value.get("class_counts")
    if (
        set(value) != expected_fields
        or value.get("authority_sha256") != authority_sha256
        or value.get("role") != "validation"
        or value.get("rows") != expected_rows
        or not isinstance(rows, list)
        or len(rows) != expected_rows
        or not isinstance(counts, list)
        or len(counts) != 15
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
               for item in counts)
        or sum(counts) != expected_rows
        or value.get("raw_labels_published") is not False
        or value.get("labels_opened_by_selection_only") is not True
    ):
        raise ValueError("validation proxy selection artifact fields differ")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "identity_digest", "source_path", "source_file_sha256", "source_entry",
        }:
            raise ValueError("validation proxy durable selection row differs")
        source_path = normalize_source_path(str(row["source_path"]))
        source_sha256 = require_sha256(
            row["source_file_sha256"], name="validation source file",
        )
        entry = row["source_entry"]
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
            raise ValueError("validation proxy durable source entry differs")
        identity = require_sha256(row["identity_digest"], name="validation identity")
        expected_identity = canonical_sha256({
            "source_file_sha256": source_sha256, "source_entry": entry,
        })
        if identity != expected_identity:
            raise ValueError("validation proxy durable identity is not source-derived")
        normalized.append({
            "identity_digest": identity, "source_path": source_path,
            "source_file_sha256": source_sha256, "source_entry": entry,
        })
    identities = tuple(row["identity_digest"] for row in normalized)
    source_rows = [
        {name: row[name] for name in (
            "source_path", "source_file_sha256", "source_entry",
        )}
        for row in normalized
    ]
    label_sha256 = require_sha256(
        value.get("label_vector_logical_sha256"), name="validation label vector",
    )
    if (
        len(set(identities)) != len(identities)
        or len({tuple(row.values()) for row in source_rows}) != len(source_rows)
        or value.get("identity_order_sha256")
        != canonical_sha256(list(identities))
        or value.get("source_membership_sha256") != canonical_sha256(source_rows)
    ):
        raise ValueError("validation proxy durable selection derivation differs")
    return digest, identities, label_sha256


def _validate_selection_access_coverage(
    selection: Mapping[str, Any], access: Mapping[str, Any],
) -> None:
    """Prove every selected source identity lies in an authenticated read range."""

    rows = selection.get("selected_rows")
    source_accesses = access.get("source_accesses")
    if not isinstance(rows, list) or not isinstance(source_accesses, list):
        raise ValueError("validation proxy selection/access coverage is absent")
    for row in rows:
        if not any(
            source.get("source_path") == row["source_path"]
            and source.get("source_file_sha256") == row["source_file_sha256"]
            and int(source.get("entry_start", -1)) <= int(row["source_entry"])
            < int(source.get("entry_stop", -1))
            for source in source_accesses
        ):
            raise PermissionError(
                "validation proxy selected identity lacks exact source access"
            )


def _assignment_artifact(
    rows: Sequence[Mapping[str, Any]], *, authority_sha256: str,
    identities: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != len(identities):
        raise ValueError("validation proxy assignment coverage differs")
    normalized_rows: list[dict[str, Any]] = []
    assignments: dict[str, Any] = {}
    for expected, value in zip(identities, rows, strict=True):
        if not isinstance(value, Mapping) or set(value) != {"identity_digest", "assignment"}:
            raise ValueError("validation proxy assignment row fields differ")
        identity = require_sha256(value["identity_digest"], name="assignment identity")
        if identity != expected:
            raise ValueError("validation proxy assignment identity order differs")
        if _contains_label(value["assignment"]):
            raise PermissionError("validation proxy assignment stream emitted labels")
        payload = _json_value(value["assignment"], name="assignment payload")
        normalized_rows.append({"identity_digest": identity, "assignment": payload})
        assignments[identity] = value["assignment"]
    artifact = with_content_hash({
        "kind": "validation_assignment",
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "rows": len(normalized_rows),
        "identity_order_sha256": canonical_sha256(list(identities)),
        "assignment_rows": normalized_rows,
        "label_free": True,
    })
    return artifact, assignments


def _validate_assignment_artifact(
    value: Mapping[str, Any], *, authority_sha256: str, identities: tuple[str, ...],
) -> str:
    digest = _validate_nested_record(value, expected_kind="validation_assignment")
    rows = value.get("assignment_rows")
    if not isinstance(rows, list):
        raise ValueError("validation proxy assignment rows differ")
    rebuilt, _ = _assignment_artifact(
        rows, authority_sha256=authority_sha256, identities=identities,
    )
    if dict(value) != rebuilt:
        raise ValueError("validation proxy assignment artifact is not canonical")
    return digest


def _model_rows(
    rows: Sequence[Mapping[str, Any]], *, identities: tuple[str, ...], path: str,
) -> tuple[ValidationModelRow, ...]:
    if len(rows) != len(identities):
        raise ValueError(f"validation proxy {path} model-input coverage differs")
    result: list[ValidationModelRow] = []
    for expected, value in zip(identities, rows, strict=True):
        if not isinstance(value, Mapping) or set(value) != {"identity_digest", "model_inputs"}:
            raise ValueError("validation proxy model-input row fields differ")
        identity = require_sha256(value["identity_digest"], name=f"{path} identity")
        if identity != expected:
            raise ValueError(f"validation proxy {path} identity order differs")
        if _contains_label(value["model_inputs"]):
            raise PermissionError(f"validation proxy {path} model stream emitted labels")
        result.append(ValidationModelRow(identity, value["model_inputs"]))
    return tuple(result)


def _prediction_artifact(
    *, path: str, logits: np.ndarray, authority_sha256: str,
    identities: tuple[str, ...], model_binding: Mapping[str, str],
) -> dict[str, Any]:
    value = np.asarray(logits)
    if value.dtype != np.float32 or value.shape != (len(identities), 15):
        raise ValueError(f"validation proxy {path} logits must be FP32 [rows,15]")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"validation proxy {path} logits are nonfinite")
    return with_content_hash({
        "kind": "validation_prediction",
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "path": path,
        "view_id": model_binding["view_id"],
        "model_id": model_binding["model_id"],
        "checkpoint_sha256": model_binding["checkpoint_sha256"],
        "model_source_lineage_sha256": model_binding[
            "model_source_lineage_sha256"
        ],
        "rows": len(identities),
        "identity_digests": list(identities),
        "identity_order_sha256": canonical_sha256(list(identities)),
        "logits_dtype": "float32",
        "logits_shape": [len(identities), 15],
        "logits_logical_sha256": array_sha256(f"{path}_logits", value),
        "logits": value.tolist(),
        "label_free": True,
    })


def _validate_prediction_artifact(
    value: Mapping[str, Any], *, path: str, authority_sha256: str,
    identities: tuple[str, ...], model_binding: Mapping[str, str],
) -> tuple[str, np.ndarray]:
    digest = _validate_nested_record(value, expected_kind="validation_prediction")
    logits = np.asarray(value.get("logits"), dtype=np.float32)
    rebuilt = _prediction_artifact(
        path=path, logits=logits, authority_sha256=authority_sha256,
        identities=identities, model_binding=model_binding,
    )
    if dict(value) != rebuilt:
        raise ValueError(f"validation proxy {path} prediction is not canonical")
    return digest, logits


def _join_artifact(
    *, authority_sha256: str, selection_sha256: str,
    identities: tuple[str, ...], label_vector_sha256: str,
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prediction_hashes = {
        path: require_sha256(predictions[path]["content_hash"], name=f"{path} prediction")
        for path in VALIDATION_PROXY_PATHS
    }
    return with_content_hash({
        "kind": "validation_identity_join",
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "rows": len(identities),
        "identity_digests": list(identities),
        "identity_order_sha256": canonical_sha256(list(identities)),
        "label_vector_logical_sha256": require_sha256(
            label_vector_sha256, name="validation label vector",
        ),
        "raw_labels_published": False,
        "selection_sha256": require_sha256(selection_sha256, name="validation selection"),
        "prediction_sha256s": prediction_hashes,
        "join_kind": "exact_identity_order",
        "all_prediction_streams_label_free": True,
        "labels_joined_after_prediction": True,
    })


def _metric_artifact(
    *, path: str, logits: np.ndarray, labels: np.ndarray,
    authority_sha256: str, join_sha256: str, prediction_sha256: str,
    model_binding: Mapping[str, str],
) -> dict[str, Any]:
    metrics = _json_value(
        classification_metrics(logits, labels), name=f"{path} classification metrics",
    )
    return with_content_hash({
        "kind": "validation_classification_metrics",
        "schema_version": 1,
        "authority_sha256": authority_sha256,
        "role": "validation",
        "path": path,
        "view_id": model_binding["view_id"],
        "model_id": model_binding["model_id"],
        "checkpoint_sha256": model_binding["checkpoint_sha256"],
        "model_source_lineage_sha256": model_binding[
            "model_source_lineage_sha256"
        ],
        "rows": len(labels),
        "label_vector_logical_sha256": array_sha256(
            "validation_labels", labels,
        ),
        "join_sha256": require_sha256(join_sha256, name="validation join"),
        "prediction_sha256": require_sha256(
            prediction_sha256, name=f"{path} validation prediction",
        ),
        "classification_metrics": metrics,
        "scientific_authorization": False,
    })


def _identity_bytes(identities: Sequence[str]) -> np.ndarray:
    return np.stack([
        np.frombuffer(bytes.fromhex(identity), dtype=np.uint8)
        for identity in identities
    ])


def _acceptance_proxy_metrics(
    logits: np.ndarray, labels: np.ndarray,
) -> Mapping[str, Any]:
    """Full frozen metrics plus a domain separator that forbids promotion."""

    result = classification_metrics(logits, labels)
    result["acceptance_proxy_nonfinal_marker"] = 0.0
    return result


def _contains_raw_label_payload(value: Any) -> bool:
    """Detect persisted raw labels while allowing lineage/role metadata."""

    if isinstance(value, Mapping):
        if any(str(key).lower() in {"label", "labels"} for key in value):
            return True
        return any(_contains_raw_label_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_label_payload(item) for item in value)
    return False


def _validate_finite_json(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_json(item, name=name)
    elif isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"{name} contains a nonfinite value")


def _validate_metric_artifact_without_labels(
    value: Mapping[str, Any], *, path: str, authority_sha256: str,
    rows: int, join_sha256: str, prediction_sha256: str,
    label_vector_sha256: str, model_binding: Mapping[str, str],
) -> str:
    digest = _validate_nested_record(
        value, expected_kind="validation_classification_metrics",
    )
    expected_fields = {
        "kind", "schema_version", "authority_sha256", "role", "path",
        "view_id", "model_id", "checkpoint_sha256",
        "model_source_lineage_sha256", "rows",
        "label_vector_logical_sha256", "join_sha256", "prediction_sha256",
        "classification_metrics", "scientific_authorization", "content_hash",
    }
    metrics = value.get("classification_metrics")
    if (
        set(value) != expected_fields
        or value.get("authority_sha256") != authority_sha256
        or value.get("role") != "validation"
        or value.get("path") != path
        or value.get("view_id") != model_binding["view_id"]
        or value.get("model_id") != model_binding["model_id"]
        or value.get("checkpoint_sha256") != model_binding["checkpoint_sha256"]
        or value.get("model_source_lineage_sha256")
        != model_binding["model_source_lineage_sha256"]
        or value.get("rows") != rows
        or value.get("label_vector_logical_sha256") != label_vector_sha256
        or value.get("join_sha256") != join_sha256
        or value.get("prediction_sha256") != prediction_sha256
        or not isinstance(metrics, Mapping)
        or not metrics
        or value.get("scientific_authorization") is not False
    ):
        raise ValueError(f"validation proxy {path} metric lineage differs")
    _validate_finite_json(metrics, name=f"{path} classification metrics")
    if _contains_raw_label_payload(value):
        raise PermissionError("validation proxy metric artifact persists raw labels")
    return digest


def _bootstrap_sidecars(
    *, identities: tuple[str, ...], labels: np.ndarray,
    predictions: Mapping[str, Mapping[str, Any]], logits: Mapping[str, np.ndarray],
    selection_sha256: str, label_vector_sha256: str,
    model_bindings: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    identity_digests = _identity_bytes(identities)
    results: list[dict[str, Any]] = []
    for comparison_id, left, right in VALIDATION_PROXY_COMPARISONS:
        sidecar, _ = paired_classification_bootstrap(
            left_logits=logits[left], right_logits=logits[right], labels=labels,
            identity_digests=identity_digests,
            left_id=model_bindings[left]["view_id"],
            right_id=model_bindings[right]["view_id"],
            comparison_id=comparison_id,
            parent_hashes={
                "selection": selection_sha256,
                "validation_label_vector": require_sha256(
                    label_vector_sha256, name="validation label vector",
                ),
                "left_prediction": predictions[left]["content_hash"],
                "right_prediction": predictions[right]["content_hash"],
                "left_checkpoint": model_bindings[left]["checkpoint_sha256"],
                "right_checkpoint": model_bindings[right]["checkpoint_sha256"],
            },
            metrics=VALIDATION_PROXY_BOOTSTRAP_METRICS,
            replicates=VALIDATION_PROXY_BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
            metric_function=_acceptance_proxy_metrics,
        )
        if sidecar.get("scientific_authorization") is not False:
            raise PermissionError("validation proxy bootstrap became scientifically authorizing")
        results.append(sidecar)
    return results


def run_validation_proxy_action(
    *,
    authority: Mapping[str, Any],
    authority_validator: AuthorityValidator,
    selection_reader: SelectionReader,
    assignment_reader: AssignmentReader,
    stream_readers: Mapping[str, ModelReader],
    predictors: Mapping[str, Predictor],
    model_bindings: Mapping[str, Mapping[str, Any]],
    input_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the closed validation-only proxy pipeline in memory.

    Readers receive exact branch projections and never receive labels outside
    the selection phase.  Predictors receive only ``ValidationModelRow``
    instances.  The returned object is immutable-by-content and can be
    published with :func:`publish_validation_proxy_action_result`.
    """

    authority_view = _validate_authority(
        authority, authority_validator=authority_validator,
    )
    expected_paths = set(VALIDATION_PROXY_PATHS)
    if set(stream_readers) != expected_paths or set(predictors) != expected_paths:
        raise ValueError("validation proxy stream/predictor registry differs")
    bindings = _model_bindings(model_bindings)
    lineage_sha256 = validate_validation_proxy_input_lineage(
        input_lineage,
        authority_sha256=authority_view["authority_sha256"],
        action_inputs_sha256=authority_view["action_inputs_sha256"],
    )
    if any(
        bindings[path]["model_id"] != VALIDATION_PROXY_MODEL_IDS[path]
        or input_lineage["model_report_sha256"].get(
            VALIDATION_PROXY_MODEL_IDS[path]
        ) is None
        or bindings[path]["model_source_lineage_sha256"]
        != input_lineage["model_source_lineage"][
            VALIDATION_PROXY_MODEL_IDS[path]
        ]["content_hash"]
        for path in VALIDATION_PROXY_PATHS
    ):
        raise PermissionError("validation proxy production model identity differs")

    selection_request = _request(authority_view, phase="selection", path="selection")
    selection_result = selection_reader(selection_request)
    selection_access = _access_artifact(selection_result, request=selection_request)
    selection, identities, labels = _selection_artifact(
        selection_result.rows,
        authority_sha256=authority_view["authority_sha256"],
        expected_rows=authority_view["rows"],
    )
    _validate_selection_access_coverage(selection, selection_access)
    if len(identities) > authority_view["maximum_rows"]:
        raise PermissionError("validation proxy exceeds its hard row cap")
    label_vector_sha256 = selection["label_vector_logical_sha256"]

    assignment_request = _request(
        authority_view, phase="assignment", path="assignment",
        selected_identities=identities,
    )
    assignment_result = assignment_reader(assignment_request)
    assignment_access = _access_artifact(assignment_result, request=assignment_request)
    assignment, assignments = _assignment_artifact(
        assignment_result.rows,
        authority_sha256=authority_view["authority_sha256"],
        identities=identities,
    )

    access_records: dict[str, dict[str, Any]] = {
        "selection": selection_access,
        "assignment": assignment_access,
    }
    predictions: dict[str, dict[str, Any]] = {}
    logits_by_path: dict[str, np.ndarray] = {}
    for path in VALIDATION_PROXY_PATHS:
        request = _request(
            authority_view, phase="prediction", path=path,
            selected_identities=identities,
            model_binding=bindings[path],
        )
        read_result = stream_readers[path](
            request, assignments if path == "shell_exact" else None,
        )
        access_records[path] = _access_artifact(read_result, request=request)
        model_rows = _model_rows(read_result.rows, identities=identities, path=path)
        predicted = predictors[path](request, model_rows)
        prediction = _prediction_artifact(
            path=path, logits=predicted,
            authority_sha256=authority_view["authority_sha256"],
            identities=identities, model_binding=bindings[path],
        )
        predictions[path] = prediction
        logits_by_path[path] = np.asarray(predicted)

    join = _join_artifact(
        authority_sha256=authority_view["authority_sha256"],
        selection_sha256=selection["content_hash"],
        identities=identities, label_vector_sha256=label_vector_sha256,
        predictions=predictions,
    )
    metrics = {
        path: _metric_artifact(
            path=path, logits=logits_by_path[path], labels=labels,
            authority_sha256=authority_view["authority_sha256"],
            join_sha256=join["content_hash"],
            prediction_sha256=predictions[path]["content_hash"],
            model_binding=bindings[path],
        )
        for path in VALIDATION_PROXY_PATHS
    }
    bootstraps = _bootstrap_sidecars(
        identities=identities, labels=labels, predictions=predictions,
        logits=logits_by_path, selection_sha256=selection["content_hash"],
        label_vector_sha256=label_vector_sha256,
        model_bindings=bindings,
    )
    return with_content_hash({
        "contract": VALIDATION_PROXY_PROOF_CONTRACT,
        "schema_version": 1,
        "authority_sha256": authority_view["authority_sha256"],
        "action_id": "validation_proxy",
        "action_spec_sha256": authority_view["action_spec_sha256"],
        "action_inputs_sha256": authority_view["action_inputs_sha256"],
        "source_commit": authority_view["source_commit"],
        "representation_recipe_sha256": authority_view[
            "representation_recipe_sha256"
        ],
        "input_lineage": dict(input_lineage),
        "input_lineage_sha256": lineage_sha256,
        "role": "validation",
        "rows": len(identities),
        "view_registry": dict(VALIDATION_PROXY_VIEW_REGISTRY),
        "model_bindings": bindings,
        "selection": selection,
        "access_records": access_records,
        "assignment": assignment,
        "predictions": predictions,
        "identity_join": join,
        "classification_metrics": metrics,
        "paired_bootstraps": bootstraps,
        "completed_phases": [
            "label_bearing_selection", "label_free_assignment",
            "label_free_hlt_prediction", "label_free_shell_exact_prediction",
            "label_free_native_offline_prediction", "exact_identity_join",
            "classification_metrics", "paired_bootstrap",
        ],
        "all_fifteen_classes_represented": True,
        "all_model_streams_label_free": True,
        "labels_opened_only_by_validation_selection": True,
        "labels_joined_only_after_prediction": True,
        "raw_validation_labels_published": False,
        "final_role_accessed": False,
        "pilot_submission_authorized": False,
        "scientific_authorization": False,
        "scheduler_mutated": False,
    })


def validate_validation_proxy_action_result(
    value: Mapping[str, Any], *, authority: Mapping[str, Any],
    authority_validator: AuthorityValidator,
) -> str:
    """Deeply rebuild all deterministic validation-proxy derivatives."""

    digest = validate_content_hash(
        value, expected_contract=VALIDATION_PROXY_PROOF_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "authority_sha256", "action_id",
        "action_spec_sha256", "action_inputs_sha256", "source_commit",
        "representation_recipe_sha256", "role", "rows", "view_registry",
        "input_lineage", "input_lineage_sha256", "model_bindings",
        "selection", "access_records", "assignment",
        "predictions", "identity_join", "classification_metrics",
        "paired_bootstraps", "completed_phases",
        "all_fifteen_classes_represented", "all_model_streams_label_free",
        "labels_opened_only_by_validation_selection",
        "labels_joined_only_after_prediction", "raw_validation_labels_published",
        "final_role_accessed",
        "pilot_submission_authorized", "scientific_authorization",
        "scheduler_mutated", "content_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("validation proxy proof fields differ")
    authority_view = _validate_authority(
        authority, authority_validator=authority_validator,
    )
    if (
        value.get("authority_sha256") != authority_view["authority_sha256"]
        or value.get("action_id") != "validation_proxy"
        or value.get("action_spec_sha256") != authority_view["action_spec_sha256"]
        or value.get("action_inputs_sha256") != authority_view["action_inputs_sha256"]
        or value.get("source_commit") != authority_view["source_commit"]
        or value.get("representation_recipe_sha256")
        != authority_view["representation_recipe_sha256"]
        or value.get("role") != "validation"
        or value.get("rows") != authority_view["rows"]
    ):
        raise PermissionError("validation proxy result authority lineage differs")
    if value.get("view_registry") != VALIDATION_PROXY_VIEW_REGISTRY:
        raise ValueError("validation proxy D0c/D100/TOFF registry differs")
    bindings = _model_bindings(value.get("model_bindings", {}))
    lineage = value.get("input_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("validation proxy input lineage is absent")
    lineage_sha256 = validate_validation_proxy_input_lineage(
        lineage,
        authority_sha256=authority_view["authority_sha256"],
        action_inputs_sha256=authority_view["action_inputs_sha256"],
    )
    if value.get("input_lineage_sha256") != lineage_sha256 or any(
        bindings[path]["model_id"] != VALIDATION_PROXY_MODEL_IDS[path]
        or lineage["model_report_sha256"].get(VALIDATION_PROXY_MODEL_IDS[path])
        is None
        or bindings[path]["model_source_lineage_sha256"]
        != lineage["model_source_lineage"][
            VALIDATION_PROXY_MODEL_IDS[path]
        ]["content_hash"]
        for path in VALIDATION_PROXY_PATHS
    ):
        raise PermissionError("validation proxy model/input lineage differs")
    selection_sha, identities, label_vector_sha256 = _validate_selection_artifact(
        value.get("selection", {}),
        authority_sha256=authority_view["authority_sha256"],
        expected_rows=authority_view["rows"],
    )
    if len(identities) > authority_view["maximum_rows"]:
        raise PermissionError("validation proxy result exceeds its hard row cap")

    access = value.get("access_records")
    expected_access = {"selection", "assignment", *VALIDATION_PROXY_PATHS}
    if not isinstance(access, Mapping) or set(access) != expected_access:
        raise ValueError("validation proxy access registry differs")
    _validate_access_artifact(
        access["selection"], authority_sha256=authority_view["authority_sha256"],
        phase="selection", path="selection", rows=len(identities),
    )
    _validate_selection_access_coverage(value["selection"], access["selection"])
    _validate_access_artifact(
        access["assignment"], authority_sha256=authority_view["authority_sha256"],
        phase="assignment", path="assignment", rows=len(identities),
        selected_identities=identities,
    )
    for path in VALIDATION_PROXY_PATHS:
        _validate_access_artifact(
            access[path], authority_sha256=authority_view["authority_sha256"],
            phase="prediction", path=path, rows=len(identities),
            model_binding=bindings[path],
            selected_identities=identities,
        )
    _validate_assignment_artifact(
        value.get("assignment", {}),
        authority_sha256=authority_view["authority_sha256"], identities=identities,
    )

    raw_predictions = value.get("predictions")
    if not isinstance(raw_predictions, Mapping) or set(raw_predictions) != set(
        VALIDATION_PROXY_PATHS
    ):
        raise ValueError("validation proxy prediction registry differs")
    logits: dict[str, np.ndarray] = {}
    for path in VALIDATION_PROXY_PATHS:
        _, logits[path] = _validate_prediction_artifact(
            raw_predictions[path], path=path,
            authority_sha256=authority_view["authority_sha256"], identities=identities,
            model_binding=bindings[path],
        )
    expected_join = _join_artifact(
        authority_sha256=authority_view["authority_sha256"],
        selection_sha256=selection_sha,
        identities=identities, label_vector_sha256=label_vector_sha256,
        predictions=raw_predictions,
    )
    join = value.get("identity_join")
    if not isinstance(join, Mapping):
        raise ValueError("validation proxy identity join is absent")
    _validate_nested_record(join, expected_kind="validation_identity_join")
    if dict(join) != expected_join:
        raise ValueError("validation proxy exact identity join differs")

    raw_metrics = value.get("classification_metrics")
    if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(
        VALIDATION_PROXY_PATHS
    ):
        raise ValueError("validation proxy metric registry differs")
    for path in VALIDATION_PROXY_PATHS:
        _validate_metric_artifact_without_labels(
            raw_metrics[path], path=path,
            authority_sha256=authority_view["authority_sha256"],
            rows=len(identities), join_sha256=expected_join["content_hash"],
            prediction_sha256=raw_predictions[path]["content_hash"],
            label_vector_sha256=label_vector_sha256,
            model_binding=bindings[path],
        )

    bootstraps = value.get("paired_bootstraps")
    if not isinstance(bootstraps, list) or len(bootstraps) != len(
        VALIDATION_PROXY_COMPARISONS
    ):
        raise ValueError("validation proxy paired bootstrap registry differs")
    for sidecar, (comparison_id, left, right) in zip(
        bootstraps, VALIDATION_PROXY_COMPARISONS, strict=True,
    ):
        validate_content_hash(
            sidecar, expected_contract=PAIRED_BOOTSTRAP_CONTRACT,
            expected_schema_version=1,
        )
        expected_parents = {
            "selection": selection_sha,
            "validation_label_vector": label_vector_sha256,
            "left_prediction": raw_predictions[left]["content_hash"],
            "right_prediction": raw_predictions[right]["content_hash"],
            "left_checkpoint": bindings[left]["checkpoint_sha256"],
            "right_checkpoint": bindings[right]["checkpoint_sha256"],
        }
        counts = value["selection"]["class_counts"]
        if (
            sidecar.get("comparison_id") != comparison_id
            or sidecar.get("left_id") != bindings[left]["view_id"]
            or sidecar.get("right_id") != bindings[right]["view_id"]
            or sidecar.get("rows") != len(identities)
            or sidecar.get("joined_identity_order_sha256")
            != canonical_sha256(list(identities))
            or sidecar.get("class_counts") != counts
            or sidecar.get("replicates") != VALIDATION_PROXY_BOOTSTRAP_REPLICATES
            or sidecar.get("seed") != BOOTSTRAP_SEED
            or sidecar.get("bit_generator") != "PCG64"
            or sidecar.get("metric_order")
            != list(VALIDATION_PROXY_BOOTSTRAP_METRICS)
            or sidecar.get("parent_hashes") != dict(sorted(expected_parents.items()))
            or sidecar.get("scientific_authorization") is not False
        ):
            raise PermissionError("validation proxy bootstrap lineage differs")
        _validate_finite_json(sidecar.get("intervals"), name=comparison_id)
        if _contains_raw_label_payload(sidecar):
            raise PermissionError("validation proxy bootstrap persists raw labels")

    expected_claims = {
        "completed_phases": [
            "label_bearing_selection", "label_free_assignment",
            "label_free_hlt_prediction", "label_free_shell_exact_prediction",
            "label_free_native_offline_prediction", "exact_identity_join",
            "classification_metrics", "paired_bootstrap",
        ],
        "all_fifteen_classes_represented": True,
        "all_model_streams_label_free": True,
        "labels_opened_only_by_validation_selection": True,
        "labels_joined_only_after_prediction": True,
        "raw_validation_labels_published": False,
        "final_role_accessed": False,
        "pilot_submission_authorized": False,
        "scientific_authorization": False,
        "scheduler_mutated": False,
    }
    if any(value.get(key) != expected for key, expected in expected_claims.items()):
        raise PermissionError("validation proxy result claims differ")
    if _contains_raw_label_payload(value):
        raise PermissionError("validation proxy result persists raw labels")
    return digest


def _nonfinal_artifact_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise PermissionError("validation proxy artifact path cannot be a symlink")
    destination = raw.resolve()
    if any(
        part.lower() in _FORBIDDEN_ROUTE_COMPONENTS
        for part in destination.parts
    ):
        raise PermissionError("validation proxy artifact cannot use a final route")
    return destination


def _canonical_validation_result_path(authority: Mapping[str, Any]) -> Path:
    reference = authority.get("action_inputs")
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise PermissionError("validation proxy authority lacks canonical action inputs")
    action_inputs = _nonfinal_artifact_path(str(reference["path"]))
    if (
        not action_inputs.is_absolute()
        or not action_inputs.is_file()
        or action_inputs.is_symlink()
        or action_inputs.name != "action_inputs.json"
        or sha256_file(action_inputs)
        != require_sha256(reference["sha256"], name="validation action-input bytes")
    ):
        raise PermissionError("validation proxy action-input route differs")
    return action_inputs.parent / "validation_proxy" / "result.json"


def publish_validation_proxy_action_result(
    path: str | Path, *, result: Mapping[str, Any], authority: Mapping[str, Any],
    authority_validator: AuthorityValidator,
) -> dict[str, str]:
    """Publish an already validated action result without replacing bytes."""

    validate_validation_proxy_action_result(
        result, authority=authority, authority_validator=authority_validator,
    )
    if _contains_raw_label_payload(result):
        raise PermissionError("validation proxy publisher refuses raw labels")
    destination = _nonfinal_artifact_path(path)
    if destination != _canonical_validation_result_path(authority):
        raise PermissionError("validation proxy proof route differs")
    access_root = destination.parent / "access"
    if destination.exists() or access_root.exists():
        raise FileExistsError(
            "validation proxy semantic/access route already exists"
        )
    for stage, record in result["access_records"].items():
        write_immutable_json(access_root / f"{stage}.json", record)
    write_immutable_json(destination, result)
    return {"path": str(destination), "sha256": sha256_file(destination)}


def _load_result_reference(reference: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError("validation proxy result reference fields differ")
    path = _nonfinal_artifact_path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != require_sha256(reference["sha256"], name="validation proxy result bytes"):
        raise ValueError("validation proxy result bytes differ")
    return load_json(path), {"path": str(path), "sha256": observed}


def build_validation_proxy_proof_v2(
    *, result_reference: Mapping[str, Any], authority: Mapping[str, Any],
    authority_validator: AuthorityValidator,
) -> dict[str, Any]:
    """Reopen the immutable proof and every separately published access row."""

    authority_validator(authority)
    if not isinstance(result_reference, Mapping):
        raise ValueError("validation proxy result reference fields differ")
    supplied_path = _nonfinal_artifact_path(str(result_reference.get("path", "")))
    if supplied_path != _canonical_validation_result_path(authority):
        raise PermissionError("validation proxy proof route differs")
    result, reference = _load_result_reference(result_reference)
    validate_validation_proxy_action_result(
        result, authority=authority, authority_validator=authority_validator,
    )
    root = Path(reference["path"]).parent
    access_root = root / "access"
    expected_names = {f"{stage}.json" for stage in result["access_records"]}
    actual_names = (
        {path.name for path in access_root.iterdir()}
        if access_root.is_dir() and not access_root.is_symlink()
        else set()
    )
    if actual_names != expected_names:
        raise ValueError("validation proxy access publication inventory differs")
    for stage, expected in result["access_records"].items():
        path = access_root / f"{stage}.json"
        if not path.is_file() or path.is_symlink() or load_json(path) != expected:
            raise ValueError(f"validation proxy access publication differs: {stage}")
    return result


def validate_validation_proxy_proof_v2(
    value: Mapping[str, Any], *, authority: Mapping[str, Any],
    authority_validator: AuthorityValidator,
) -> str:
    """Deeply validate every actual dataflow product in proof-v2."""

    return validate_validation_proxy_action_result(
        value, authority=authority, authority_validator=authority_validator,
    )


def validate_validation_proxy_branch_access(
    value: Mapping[str, Any], *, authority_sha256: str, stage: str, rows: int,
    model_binding: Mapping[str, str] | None = None,
    selected_identities: Sequence[str] = (),
) -> str:
    """Public contextual validator for one canonical access-stage artifact."""

    phases = {
        "selection": "selection",
        "assignment": "assignment",
        "hlt": "prediction",
        "shell_exact": "prediction",
        "native_offline": "prediction",
    }
    if stage not in phases:
        raise ValueError("validation proxy access stage differs")
    return _validate_access_artifact(
        value, authority_sha256=require_sha256(
            authority_sha256, name="validation proxy authority",
        ),
        phase=phases[stage], path=stage, rows=rows,
        model_binding=model_binding,
        selected_identities=selected_identities,
    )


# Canonical public names used by the route and evidence registries.
build_validation_proxy_proof = build_validation_proxy_proof_v2
validate_validation_proxy_proof = validate_validation_proxy_proof_v2


__all__ = [
    "NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT",
    "VALIDATION_BRANCH_ALLOWLISTS",
    "VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT",
    "VALIDATION_PROXY_BOOTSTRAP_METRICS",
    "VALIDATION_PROXY_BOOTSTRAP_REPLICATES",
    "VALIDATION_PROXY_COMPARISONS",
    "VALIDATION_PROXY_MODEL_IDS",
    "VALIDATION_PROXY_PATHS",
    "VALIDATION_PROXY_REGISTERED_INPUTS",
    "VALIDATION_PROXY_PROOF_CONTRACT",
    "VALIDATION_PROXY_VIEW_REGISTRY",
    "ValidationAccessRequest",
    "ValidationModelRow",
    "ValidationReadResult",
    "build_validation_proxy_input_lineage",
    "build_validation_proxy_proof_v2",
    "build_validation_proxy_proof",
    "publish_validation_proxy_action_result",
    "run_validation_proxy_action",
    "validate_validation_proxy_action_result",
    "validate_validation_proxy_branch_access",
    "validate_validation_proxy_input_lineage",
    "validate_validation_proxy_proof",
    "validate_validation_proxy_proof_v2",
]
