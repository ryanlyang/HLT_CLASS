"""Concrete execution binding for the HCWDL-RKD command plan.

The scientific campaign registry deliberately contains logical input/output
routes.  This module binds every logical route, including every array row, to
one concrete immutable path.  Exogenous inputs carry their known byte hash.
Campaign-produced inputs instead freeze an exact transitive producer/output
route and are byte/inventory authenticated only after that producer publishes;
pre-campaign construction never invents a future hash.  The binding has its
own operational contract because it is a durable artifact at a path distinct
from the symbolic command plan.  This separation does not create a new
scientific configuration surface.

No Python module/function name, shell command, or arbitrary entry point is
accepted.  The production dispatcher selects a built-in adapter solely from
the already frozen campaign task kind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_campaign import CampaignTask
from .hcwdl_representation_contracts import (
    DENSE_TEACHER_IMPORT_CONTRACT, PARENT_IMPORT_CONTRACT,
    REPRESENTATION_RECIPE_CONTRACT,
    RUNTIME_BINDING_CONTRACT,
    contract_schema_version,
)
from .hcwdl_representation_workflow import array_indices


RUNTIME_BINDING_KIND: Final = "runtime_binding_v2"
UPSTREAM_OUTPUT_BINDING: Final = "upstream_output"
CAMPAIGN_ARTIFACT_BINDING: Final = "campaign_artifact"
IMMUTABLE_OUTPUT_ROOT_BINDING: Final = "immutable_output_root"
PREPUBLISHED_OUTPUT_BINDING: Final = "prepublished_output"
UPSTREAM_ARTIFACT_KINDS: Final = frozenset({"json", "file", "directory"})
RUNTIME_FACT_KEYS: Final = frozenset({
    "conda_environment",
    "data_root",
    "device",
    "project_dir",
    "python_no_user_site",
    "source_snapshot_sha256",
    "weaver_runtime_sha256",
})
_FORBIDDEN_DYNAMIC_KEYS: Final = frozenset({
    "argv", "callable", "command", "entry_point", "function", "import",
    "module", "python", "script", "shell",
})
_SAFE_DYNAMIC_KEY_METADATA_PATHS: Final = frozenset({
    ("parameters.assembly.runtime_environment.producer.packages", "python"),
})

# These are the only campaign rows allowed to read an immutable pre-campaign
# artifact at a designated campaign output.  The owner gate reopens and
# validates those exact bytes, then publication is an identical-file no-op.
# Kernel-resource generation is the one closed read-only consumer required
# before the recipe owner gate runs.  All other static-input/output path
# overlap remains forbidden.
_PREPUBLISHED_OUTPUT_ROUTES: Final = {
    ("parent_import", "dense_teacher_import", "${prebuilt_parent_import}"): {
        "owner_task_key": "parent_import",
        "owner_task_kind": "dense_teacher_import",
        "registered_output": "import/dense_teacher_import.json",
        "expected_contract": DENSE_TEACHER_IMPORT_CONTRACT,
        "expected_schema_version": contract_schema_version(
            DENSE_TEACHER_IMPORT_CONTRACT,
        ),
        "campaign_hash_field": "parent_import_sha256",
    },
    ("parent_import", "parent_import", "${prebuilt_parent_import}"): {
        "owner_task_key": "parent_import",
        "owner_task_kind": "parent_import",
        "registered_output": "import/parent_import.json",
        "expected_contract": PARENT_IMPORT_CONTRACT,
        "expected_schema_version": contract_schema_version(
            PARENT_IMPORT_CONTRACT,
        ),
        "campaign_hash_field": "parent_import_sha256",
    },
    (
        "representation_recipe", "representation_recipe",
        "${prebuilt_representation_recipe}",
    ): {
        "owner_task_key": "representation_recipe",
        "owner_task_kind": "representation_recipe",
        "registered_output": "recipes/representation_recipe.json",
        "expected_contract": REPRESENTATION_RECIPE_CONTRACT,
        "expected_schema_version": contract_schema_version(
            REPRESENTATION_RECIPE_CONTRACT,
        ),
        "campaign_hash_field": "representation_recipe_sha256",
    },
    (
        "kernel_resources", "kernel_resources",
        "${prebuilt_representation_recipe}",
    ): {
        "owner_task_key": "representation_recipe",
        "owner_task_kind": "representation_recipe",
        "registered_output": "recipes/representation_recipe.json",
        "expected_contract": REPRESENTATION_RECIPE_CONTRACT,
        "expected_schema_version": contract_schema_version(
            REPRESENTATION_RECIPE_CONTRACT,
        ),
        "campaign_hash_field": "representation_recipe_sha256",
    },
}


def _campaign_tasks(spec: Mapping[str, Any]) -> tuple[CampaignTask, ...]:
    return tuple(
        CampaignTask(
            **{
                **row,
                "dependencies": tuple(row["dependencies"]),
                "registered_inputs": tuple(row["registered_inputs"]),
                "registered_outputs": tuple(row["registered_outputs"]),
            }
        )
        for row in spec["tasks"]
    )


def runtime_campaign_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the acyclic campaign identity to which a binding is attached.

    The binding hash is later embedded in the executable campaign, so the
    identity intentionally omits only that hash and post-review authorization
    evidence.  It includes the entire task/resource/path topology.  The
    campaign artifact itself is independently content-hash validated by every
    worker before this identity is considered.
    """

    return {
        key: spec.get(key)
        for key in (
            "mode", "campaign_root", "checkpoint_namespace",
            "project_dir", "source_commit", "source_manifest_sha256",
            "split_manifest_sha256", "parent_import_sha256",
            "representation_recipe_sha256", "graph_sha256",
            "disposition_sha256", "disposition", "role_counts",
            "final_source_partitions", "combined_finalist_count",
            "artifact_paths", "resources", "array_concurrency_limits",
            "resource_request_sha256", "tasks",
        )
    }


def _absolute_tigris_path(value: object, *, name: str) -> str:
    path = str(value)
    # Runtime bindings execute on Tigris even when reviewed on Windows.  Use
    # POSIX path semantics explicitly rather than the host platform's Path.
    if not path or not PurePosixPath(path).is_absolute() or "\x00" in path:
        raise ValueError(f"HCWDL-RKD runtime {name} is not an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in PurePosixPath(path).parts[1:]):
        raise ValueError(f"HCWDL-RKD runtime {name} is not normalized")
    return path


def _artifact_reference(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"HCWDL-RKD runtime input reference differs: {name}")
    return {
        "path": _absolute_tigris_path(value["path"], name=f"{name} path"),
        "sha256": require_sha256(value["sha256"], name=f"{name} SHA-256"),
    }


def _prepublished_output_reference(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "sha256", PREPUBLISHED_OUTPUT_BINDING,
    }:
        raise ValueError(f"HCWDL-RKD prepublished output reference differs: {name}")
    raw = value[PREPUBLISHED_OUTPUT_BINDING]
    required = {
        "consumer_task_key", "consumer_task_kind", "owner_task_key",
        "owner_task_kind", "registered_input", "registered_output",
        "expected_contract", "expected_schema_version",
        "expected_content_hash",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"HCWDL-RKD prepublished output fields differ: {name}")
    schema = raw["expected_schema_version"]
    normalized = {
        "path": _absolute_tigris_path(
            value["path"], name=f"{name} prepublished path",
        ),
        "sha256": require_sha256(
            value["sha256"], name=f"{name} prepublished byte SHA-256",
        ),
        PREPUBLISHED_OUTPUT_BINDING: {
            "consumer_task_key": str(raw["consumer_task_key"]),
            "consumer_task_kind": str(raw["consumer_task_kind"]),
            "owner_task_key": str(raw["owner_task_key"]),
            "owner_task_kind": str(raw["owner_task_kind"]),
            "registered_input": str(raw["registered_input"]),
            "registered_output": str(raw["registered_output"]),
            "expected_contract": str(raw["expected_contract"]),
            "expected_schema_version": schema,
            "expected_content_hash": require_sha256(
                raw["expected_content_hash"],
                name=f"{name} prepublished content hash",
            ),
        },
    }
    descriptor = normalized[PREPUBLISHED_OUTPUT_BINDING]
    if (
        not descriptor["consumer_task_key"]
        or not descriptor["consumer_task_kind"]
        or not descriptor["owner_task_key"]
        or not descriptor["owner_task_kind"]
        or not descriptor["registered_input"]
        or not descriptor["registered_output"]
        or not descriptor["expected_contract"]
        or isinstance(schema, bool) or not isinstance(schema, int) or schema < 1
    ):
        raise ValueError(f"HCWDL-RKD prepublished output identity differs: {name}")
    return normalized


def _expected_prepublished_output_binding(
    *, task_key: str, task_kind: str, logical_input: str,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    route = _PREPUBLISHED_OUTPUT_ROUTES.get((task_key, task_kind, logical_input))
    if route is None:
        return None
    return {
        "consumer_task_key": task_key,
        "consumer_task_kind": task_kind,
        "owner_task_key": str(route["owner_task_key"]),
        "owner_task_kind": str(route["owner_task_kind"]),
        "registered_input": logical_input,
        "registered_output": str(route["registered_output"]),
        "expected_contract": str(route["expected_contract"]),
        "expected_schema_version": int(route["expected_schema_version"]),
        "expected_content_hash": require_sha256(
            spec.get(str(route["campaign_hash_field"])),
            name=f"{task_key} prepublished campaign content hash",
        ),
    }


def validate_prepublished_output_binding(
    value: object, *, logical_input: str, spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one allowed in-place prepublication route.

    The descriptor is self-identifying so a production worker can recheck the
    exact task/input/output route, JSON contract/schema, and campaign-bound
    logical content hash before an adapter opens the file.
    """

    if not isinstance(value, Mapping):
        raise ValueError("HCWDL-RKD prepublished output descriptor is not an object")
    task_key = str(value.get("consumer_task_key", ""))
    task_kind = str(value.get("consumer_task_kind", ""))
    expected = _expected_prepublished_output_binding(
        task_key=task_key, task_kind=task_kind,
        logical_input=logical_input, spec=spec,
    )
    if expected is None or dict(value) != expected:
        raise PermissionError(
            "HCWDL-RKD prepublished input is not its designated task output"
        )
    matching_consumers = [
        row for row in spec.get("tasks", ())
        if isinstance(row, Mapping) and row.get("task_key") == task_key
    ]
    owner_key = expected["owner_task_key"]
    matching_owners = [
        row for row in spec.get("tasks", ())
        if isinstance(row, Mapping) and row.get("task_key") == owner_key
    ]
    if (
        len(matching_consumers) != 1
        or matching_consumers[0].get("kind") != task_kind
        or logical_input not in matching_consumers[0].get("registered_inputs", ())
        or len(matching_owners) != 1
        or matching_owners[0].get("kind") != expected["owner_task_kind"]
        or expected["registered_output"] not in matching_owners[0].get(
            "registered_outputs", ()
        )
    ):
        raise PermissionError(
            "HCWDL-RKD prepublished input route is absent from the campaign task"
        )
    return expected


def _upstream_output_reference(value: object, *, name: str) -> dict[str, Any]:
    """Normalize a future producer output without requiring future bytes."""

    if not isinstance(value, Mapping) or set(value) not in (
        {UPSTREAM_OUTPUT_BINDING}, {UPSTREAM_OUTPUT_BINDING, "path"},
        {UPSTREAM_OUTPUT_BINDING, IMMUTABLE_OUTPUT_ROOT_BINDING},
    ):
        raise ValueError(f"HCWDL-RKD runtime input reference differs: {name}")
    raw = value[UPSTREAM_OUTPUT_BINDING]
    required = {
        "task_key", "array_index", "registered_output", "artifact_kind",
        "expected_contract", "expected_schema_version",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"HCWDL-RKD upstream-output reference fields differ: {name}")
    task_key = str(raw["task_key"])
    output = str(raw["registered_output"])
    artifact_kind = str(raw["artifact_kind"])
    array_index = raw["array_index"]
    if (
        not task_key or not output or artifact_kind not in UPSTREAM_ARTIFACT_KINDS
        or isinstance(array_index, bool)
        or (array_index is not None and not isinstance(array_index, int))
    ):
        raise ValueError(f"HCWDL-RKD upstream-output identity differs: {name}")
    contract = raw["expected_contract"]
    schema = raw["expected_schema_version"]
    if artifact_kind == "json":
        if (
            not isinstance(contract, str) or not contract
            or isinstance(schema, bool) or not isinstance(schema, int) or schema < 1
        ):
            raise ValueError(
                f"HCWDL-RKD upstream JSON contract/schema differs: {name}"
            )
    elif contract is not None or schema is not None:
        raise ValueError(
            f"HCWDL-RKD non-JSON upstream output cannot declare a JSON contract: {name}"
        )
    normalized: dict[str, Any] = {
        UPSTREAM_OUTPUT_BINDING: {
            "task_key": task_key,
            "array_index": array_index,
            "registered_output": output,
            "artifact_kind": artifact_kind,
            "expected_contract": contract,
            "expected_schema_version": schema,
        }
    }
    if "path" in value:
        normalized["path"] = _absolute_tigris_path(
            value["path"], name=f"{name} frozen upstream path",
        )
    if IMMUTABLE_OUTPUT_ROOT_BINDING in value:
        normalized[IMMUTABLE_OUTPUT_ROOT_BINDING] = _immutable_output_root_reference(
            value[IMMUTABLE_OUTPUT_ROOT_BINDING], name=f"{name} frozen immutable root",
            route=None,
        )
    return normalized


def _campaign_artifact_reference(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) not in (
        {CAMPAIGN_ARTIFACT_BINDING}, {CAMPAIGN_ARTIFACT_BINDING, "path"},
    ):
        raise ValueError(f"HCWDL-RKD campaign-artifact reference differs: {name}")
    raw = value[CAMPAIGN_ARTIFACT_BINDING]
    required = {
        "artifact_role", "expected_contract", "expected_schema_version",
        "campaign_identity_sha256", "command_plan_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError(f"HCWDL-RKD campaign-artifact fields differ: {name}")
    if str(raw["artifact_role"]) != "submission_ledger":
        raise ValueError("HCWDL-RKD unsupported campaign-produced artifact role")
    normalized = {
        CAMPAIGN_ARTIFACT_BINDING: {
            "artifact_role": "submission_ledger",
            "expected_contract": str(raw["expected_contract"]),
            "expected_schema_version": int(raw["expected_schema_version"]),
            "campaign_identity_sha256": require_sha256(
                raw["campaign_identity_sha256"], name="campaign artifact identity",
            ),
            "command_plan_sha256": require_sha256(
                raw["command_plan_sha256"], name="campaign artifact command plan",
            ),
        },
    }
    if not normalized[CAMPAIGN_ARTIFACT_BINDING]["expected_contract"] or (
        isinstance(raw["expected_schema_version"], bool)
        or normalized[CAMPAIGN_ARTIFACT_BINDING]["expected_schema_version"] < 1
    ):
        raise ValueError("HCWDL-RKD campaign artifact contract/schema differs")
    if "path" in value:
        normalized["path"] = _absolute_tigris_path(
            value["path"], name=f"{name} frozen campaign artifact path",
        )
    return normalized


def _immutable_output_root_reference(
    value: object, *, name: str, route: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required = {
        "root", "artifact_contract", "producer_task_id",
        "registered_output_row", "campaign_identity_sha256",
        "expected_publication_owner", "expected_parent_sources",
    }
    canonical_required = required | {"registered_route", "template"}
    if not isinstance(value, Mapping) or set(value) not in (required, canonical_required):
        raise ValueError(f"HCWDL-RKD immutable output-root fields differ: {name}")
    row = value["registered_output_row"]
    if not isinstance(row, Mapping) or not row:
        raise ValueError("HCWDL-RKD immutable output registered row differs")
    contract = str(value["artifact_contract"])
    producer = str(value["producer_task_id"])
    if not contract or not producer:
        raise ValueError("HCWDL-RKD immutable output identity differs")
    owner = value["expected_publication_owner"]
    if not isinstance(owner, Mapping) or set(owner) != {
        "campaign_task", "array_index", "owner_kind", "campaign_identity_sha256",
    }:
        raise ValueError("HCWDL-RKD immutable output publication owner differs")
    if (
        not str(owner["campaign_task"])
        or owner["owner_kind"] != "initial_campaign"
        or isinstance(owner["array_index"], bool)
        or (
            owner["array_index"] is not None
            and not isinstance(owner["array_index"], int)
        )
    ):
        raise ValueError("HCWDL-RKD immutable output publication owner differs")
    normalized_owner = {
        "campaign_task": str(owner["campaign_task"]),
        "array_index": owner["array_index"],
        "owner_kind": "initial_campaign",
        "campaign_identity_sha256": require_sha256(
            owner["campaign_identity_sha256"],
            name="immutable output campaign identity",
        ),
    }
    raw_sources = value["expected_parent_sources"]
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError("HCWDL-RKD immutable output parent sources differ")
    parent_sources = {
        str(parent): _parent_source_reference(
            source, name=f"{name} parent {parent}",
        )
        for parent, source in sorted(raw_sources.items())
    }
    if any(not parent for parent in parent_sources):
        raise ValueError("HCWDL-RKD immutable output parent name is empty")
    normalized: dict[str, Any] = {
        "root": _absolute_tigris_path(value["root"], name=f"{name} root"),
        "artifact_contract": contract,
        "producer_task_id": producer,
        "registered_output_row": dict(row),
        "campaign_identity_sha256": require_sha256(
            value["campaign_identity_sha256"],
            name="immutable output campaign identity",
        ),
        "expected_publication_owner": normalized_owner,
        "expected_parent_sources": parent_sources,
        "template": "committed/${envelope_id}",
    }
    if route is None:
        if set(value) != canonical_required:
            raise ValueError("HCWDL-RKD upstream immutable root lacks its frozen route")
        frozen_route = value["registered_route"]
    else:
        frozen_route = route
    if not isinstance(frozen_route, Mapping) or set(frozen_route) != {
        "task_key", "array_index", "registered_output",
    }:
        raise ValueError("HCWDL-RKD immutable output route differs")
    normalized["registered_route"] = dict(frozen_route)
    if (
        normalized_owner["campaign_task"] != frozen_route["task_key"]
        or normalized_owner["array_index"] != frozen_route["array_index"]
    ):
        raise ValueError("HCWDL-RKD immutable output owner/route lineage differs")
    if set(value) == canonical_required and (
        value["template"] != normalized["template"]
        or dict(value["registered_route"]) != normalized["registered_route"]
    ):
        raise ValueError("HCWDL-RKD immutable output canonical route differs")
    return normalized


def _parent_source_reference(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"HCWDL-RKD immutable parent source differs: {name}")
    kind = str(value.get("source_kind", ""))
    if kind == "literal_sha256":
        if set(value) != {"source_kind", "sha256"}:
            raise ValueError(f"HCWDL-RKD literal parent source differs: {name}")
        return {
            "source_kind": kind,
            "sha256": require_sha256(value["sha256"], name=f"{name} SHA-256"),
        }
    common = {
        "source_kind", "expected_contract", "expected_schema_version",
    }
    if kind == "json_artifact":
        required = common | {"path"}
    elif kind == "envelope_json_member":
        required = common | {"member"}
    elif kind == "final_task_registry_field":
        required = common | {"path", "task_id", "field"}
    else:
        raise ValueError(f"HCWDL-RKD immutable parent source kind differs: {name}")
    if set(value) != required:
        raise ValueError(f"HCWDL-RKD immutable parent source fields differ: {name}")
    contract = str(value["expected_contract"])
    schema = value["expected_schema_version"]
    if (
        not contract or isinstance(schema, bool)
        or not isinstance(schema, int) or schema < 1
    ):
        raise ValueError(f"HCWDL-RKD immutable parent contract differs: {name}")
    result: dict[str, Any] = {
        "source_kind": kind,
        "expected_contract": contract,
        "expected_schema_version": schema,
    }
    if kind in {"json_artifact", "final_task_registry_field"}:
        result["path"] = _absolute_tigris_path(
            value["path"], name=f"{name} path",
        )
    if kind == "envelope_json_member":
        member = str(value["member"])
        member_path = PurePosixPath(member)
        if (
            not member or member_path.is_absolute()
            or len(member_path.parts) != 1 or member in {".", "..", "commit.json"}
        ):
            raise ValueError(f"HCWDL-RKD immutable parent member differs: {name}")
        result["member"] = member
    if kind == "final_task_registry_field":
        task_id = str(value["task_id"])
        if not task_id or value["field"] != "checkpoint_sha256":
            raise ValueError(f"HCWDL-RKD final-task parent selector differs: {name}")
        result.update(task_id=task_id, field="checkpoint_sha256")
    return result


def _input_reference(value: object, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and PREPUBLISHED_OUTPUT_BINDING in value:
        return _prepublished_output_reference(value, name=name)
    if isinstance(value, Mapping) and UPSTREAM_OUTPUT_BINDING in value:
        return _upstream_output_reference(value, name=name)
    if isinstance(value, Mapping) and CAMPAIGN_ARTIFACT_BINDING in value:
        return _campaign_artifact_reference(value, name=name)
    return _artifact_reference(value, name=name)


def _reject_dynamic_dispatch(value: object, *, location: str = "parameters") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.lower()
            if (
                normalized_key in _FORBIDDEN_DYNAMIC_KEYS
                and (location, normalized_key)
                not in _SAFE_DYNAMIC_KEY_METADATA_PATHS
            ):
                raise PermissionError(
                    f"dynamic dispatch field is forbidden in runtime binding: {location}.{key}"
                )
            _reject_dynamic_dispatch(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_dynamic_dispatch(item, location=f"{location}[{index}]")
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and value.startswith(("python:", "module:", "callable:")):
            raise PermissionError(f"dynamic dispatch value is forbidden at {location}")
        return
    raise TypeError(f"runtime binding contains a non-JSON parameter at {location}")


def _normalize_row_binding(
    value: object, *, task: CampaignTask, array_index: int | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "array_index", "device", "inputs", "outputs", "parameters",
        "runtime_signature_sha256",
    }:
        raise ValueError(f"HCWDL-RKD runtime row fields differ for {task.task_key}")
    if value["array_index"] != array_index:
        raise ValueError(f"HCWDL-RKD runtime array index differs for {task.task_key}")
    device = str(value["device"])
    if device not in {"cpu", "cuda"}:
        raise ValueError("HCWDL-RKD runtime row device differs")
    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(task.registered_inputs):
        raise ValueError(f"HCWDL-RKD concrete inputs differ for {task.task_key}")
    inputs = {
        name: _input_reference(raw_inputs[name], name=f"{task.task_key}:{name}")
        for name in task.registered_inputs
    }
    raw_outputs = value["outputs"]
    if not isinstance(raw_outputs, Mapping) or set(raw_outputs) != set(
        task.registered_outputs
    ):
        raise ValueError(f"HCWDL-RKD concrete outputs differ for {task.task_key}")
    outputs: dict[str, Any] = {}
    output_identities: list[str] = []
    for logical in task.registered_outputs:
        raw_output = raw_outputs[logical]
        if isinstance(raw_output, Mapping) and IMMUTABLE_OUTPUT_ROOT_BINDING in raw_output:
            if set(raw_output) != {IMMUTABLE_OUTPUT_ROOT_BINDING}:
                raise ValueError("HCWDL-RKD immutable output binding wrapper differs")
            if "${envelope_id}" not in logical:
                raise ValueError(
                    "HCWDL-RKD immutable output root is only valid for an envelope route"
                )
            descriptor = _immutable_output_root_reference(
                raw_output[IMMUTABLE_OUTPUT_ROOT_BINDING],
                name=f"{task.task_key}:{logical}",
                route={
                    "task_key": task.task_key, "array_index": array_index,
                    "registered_output": logical,
                },
            )
            outputs[logical] = {IMMUTABLE_OUTPUT_ROOT_BINDING: descriptor}
            output_identities.append(f"root:{descriptor['root']}")
        else:
            path = _absolute_tigris_path(
                raw_output, name=f"{task.task_key}:{logical}",
            )
            outputs[logical] = path
            output_identities.append(f"path:{path}")
    if len(set(output_identities)) != len(output_identities):
        raise ValueError(f"HCWDL-RKD concrete outputs collide for {task.task_key}")
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("HCWDL-RKD runtime parameters must be a JSON object")
    _reject_dynamic_dispatch(parameters)
    return {
        "array_index": array_index,
        "device": device,
        "inputs": inputs,
        "outputs": outputs,
        "parameters": dict(parameters),
        "runtime_signature_sha256": require_sha256(
            value["runtime_signature_sha256"],
            name=f"{task.task_key} runtime signature",
        ),
    }


def build_runtime_binding(
    *,
    spec: Mapping[str, Any],
    runtime_facts: Mapping[str, Any],
    task_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the exact concrete binding for all task and array rows.

    ``task_rows[task_key]`` uses ``"single"`` for a scalar task and the
    decimal array index for an array task.  The builder is intentionally
    exhaustive: a missing future row cannot inherit a nearby row's paths.
    """

    if set(runtime_facts) != RUNTIME_FACT_KEYS:
        raise ValueError("HCWDL-RKD runtime fact registry differs")
    normalized_facts = dict(runtime_facts)
    normalized_facts["project_dir"] = _absolute_tigris_path(
        runtime_facts["project_dir"], name="project directory",
    )
    normalized_facts["data_root"] = _absolute_tigris_path(
        runtime_facts["data_root"], name="data root",
    )
    if normalized_facts["project_dir"] != spec["project_dir"]:
        raise ValueError("HCWDL-RKD runtime project directory differs from campaign")
    if normalized_facts["device"] not in {"cuda", "cpu"}:
        raise ValueError("HCWDL-RKD runtime device differs")
    if not isinstance(normalized_facts["conda_environment"], str) or not normalized_facts[
        "conda_environment"
    ]:
        raise ValueError("HCWDL-RKD runtime Conda environment differs")
    if normalized_facts["python_no_user_site"] is not True:
        raise ValueError("HCWDL-RKD runtime must isolate the user site")
    for name in ("source_snapshot_sha256", "weaver_runtime_sha256"):
        normalized_facts[name] = require_sha256(
            normalized_facts[name], name=f"runtime {name}",
        )

    campaign_identity = runtime_campaign_identity(spec)
    campaign_identity_sha256 = canonical_sha256(campaign_identity)
    tasks = _campaign_tasks(spec)
    task_by_key = {task.task_key: task for task in tasks}
    if set(task_rows) != {task.task_key for task in tasks}:
        raise ValueError("HCWDL-RKD runtime task registry differs from campaign")
    normalized_tasks = []
    all_output_paths: set[str] = set()
    output_routes: dict[tuple[str, int | None, str], str] = {}
    for task in tasks:
        raw_rows = task_rows[task.task_key]
        if not isinstance(raw_rows, Mapping):
            raise ValueError("HCWDL-RKD runtime task rows differ")
        indices = array_indices(task.array)
        expected_keys = {
            "single" if index is None else str(index) for index in indices
        }
        if set(raw_rows) != expected_keys:
            raise ValueError(f"HCWDL-RKD runtime row registry differs for {task.task_key}")
        rows = []
        for index in indices:
            key = "single" if index is None else str(index)
            row = _normalize_row_binding(raw_rows[key], task=task, array_index=index)
            for output in row["outputs"].values():
                if not isinstance(output, Mapping) or (
                    IMMUTABLE_OUTPUT_ROOT_BINDING not in output
                ):
                    continue
                descriptor = output[IMMUTABLE_OUTPUT_ROOT_BINDING]
                if (
                    descriptor["campaign_identity_sha256"]
                    != campaign_identity_sha256
                    or descriptor["expected_publication_owner"][
                        "campaign_identity_sha256"
                    ] != campaign_identity_sha256
                ):
                    raise PermissionError(
                        "HCWDL-RKD immutable output campaign lineage differs"
                    )
            campaign_reference = row["inputs"].get("${campaign_spec}")
            if campaign_reference is not None:
                if (
                    UPSTREAM_OUTPUT_BINDING in campaign_reference
                    or campaign_reference["sha256"] != campaign_identity_sha256
                ):
                    raise ValueError(
                        "HCWDL-RKD runtime campaign-spec reference must use the acyclic "
                        "campaign identity SHA-256"
                    )
            output_locations = {
                (
                    value[IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
                    if isinstance(value, Mapping)
                    and IMMUTABLE_OUTPUT_ROOT_BINDING in value
                    else str(value)
                )
                for value in row["outputs"].values()
            }
            overlap = all_output_paths & output_locations
            if overlap:
                raise ValueError(
                    f"HCWDL-RKD runtime output path reused across rows: {sorted(overlap)}"
                )
            all_output_paths.update(output_locations)
            for logical_output, output_path in row["outputs"].items():
                route = (task.task_key, index, logical_output)
                if route in output_routes:
                    raise ValueError("HCWDL-RKD runtime output route repeats")
                output_routes[route] = output_path
            rows.append(row)
        normalized_tasks.append({
            "task_key": task.task_key,
            "kind": task.kind,
            "deterministic_worker": task.deterministic_worker,
            "rows": rows,
        })

    def transitive_dependencies(task_key: str) -> set[str]:
        found: set[str] = set()
        pending = list(task_by_key[task_key].dependencies)
        while pending:
            dependency = pending.pop()
            if dependency in found:
                continue
            if dependency not in task_by_key:
                raise ValueError(
                    f"HCWDL-RKD campaign dependency is absent: {dependency}"
                )
            found.add(dependency)
            pending.extend(task_by_key[dependency].dependencies)
        return found

    # Future producer bytes cannot have a pre-campaign SHA.  Freeze their exact
    # producer row and output route now; the worker authenticates the published
    # bytes and materializes a transient path/SHA reference at consumption.
    for task_binding in normalized_tasks:
        consumer = str(task_binding["task_key"])
        ancestors = transitive_dependencies(consumer)
        for row in task_binding["rows"]:
            for logical_input, reference in tuple(row["inputs"].items()):
                if CAMPAIGN_ARTIFACT_BINDING in reference:
                    from .hcwdl_representation_contracts import SUBMISSION_LEDGER_CONTRACT

                    if (
                        task_by_key[consumer].kind != "shared_final_claim"
                        or logical_input != "${submission_ledger}"
                    ):
                        raise PermissionError(
                            "campaign-produced submission ledger is consumable only by "
                            "shared_final_claim"
                        )
                    campaign_artifact = reference[CAMPAIGN_ARTIFACT_BINDING]
                    exact_path = str(PurePosixPath(str(spec["campaign_root"])) / "submission_ledger.json")
                    if (
                        campaign_artifact["expected_contract"] != SUBMISSION_LEDGER_CONTRACT
                        or campaign_artifact["expected_schema_version"] != 1
                        or campaign_artifact["campaign_identity_sha256"]
                        != campaign_identity_sha256
                        or campaign_artifact["command_plan_sha256"]
                        != spec.get("command_plan_sha256")
                        or reference.get("path", exact_path) != exact_path
                    ):
                        raise ValueError(
                            "submission-ledger late binding campaign/plan lineage differs"
                        )
                    row["inputs"][logical_input] = {
                        CAMPAIGN_ARTIFACT_BINDING: dict(campaign_artifact),
                        "path": exact_path,
                    }
                    continue
                prepublished = reference.get(PREPUBLISHED_OUTPUT_BINDING)
                if prepublished is not None:
                    descriptor = validate_prepublished_output_binding(
                        prepublished, logical_input=logical_input, spec=spec,
                    )
                    route = (
                        descriptor["owner_task_key"], None,
                        descriptor["registered_output"],
                    )
                    expected_output = output_routes.get(route)
                    if (
                        isinstance(expected_output, Mapping)
                        or reference.get("path") != expected_output
                        or reference.get("path") not in all_output_paths
                    ):
                        raise PermissionError(
                            "prepublished input path differs from its exact own output"
                        )
                    continue
                if UPSTREAM_OUTPUT_BINDING not in reference:
                    if reference["path"] in all_output_paths:
                        current_task = task_by_key[consumer]
                        descriptor = _expected_prepublished_output_binding(
                            task_key=current_task.task_key,
                            task_kind=current_task.kind,
                            logical_input=logical_input,
                            spec=spec,
                        )
                        if descriptor is None:
                            raise PermissionError(
                                "campaign-produced inputs must use an upstream_output binding"
                            )
                        route = (
                            descriptor["owner_task_key"], None,
                            descriptor["registered_output"],
                        )
                        expected_output = output_routes.get(route)
                        if (
                            isinstance(expected_output, Mapping)
                            or reference["path"] != expected_output
                        ):
                            raise PermissionError(
                                "prepublished input path differs from its exact own output"
                            )
                        reference[PREPUBLISHED_OUTPUT_BINDING] = descriptor
                    continue
                upstream = reference[UPSTREAM_OUTPUT_BINDING]
                producer = upstream["task_key"]
                if producer not in ancestors:
                    raise PermissionError(
                        f"late-bound producer {producer!r} is not a transitive "
                        f"dependency of {consumer!r}"
                    )
                route = (
                    producer, upstream["array_index"],
                    upstream["registered_output"],
                )
                if route not in output_routes:
                    raise ValueError(
                        f"late-bound upstream output route is absent: {route!r}"
                    )
                expected_output = output_routes[route]
                declared_path = reference.get("path")
                declared_root = reference.get(IMMUTABLE_OUTPUT_ROOT_BINDING)
                if isinstance(expected_output, Mapping) and (
                    IMMUTABLE_OUTPUT_ROOT_BINDING in expected_output
                ):
                    expected_root = expected_output[IMMUTABLE_OUTPUT_ROOT_BINDING]
                    if declared_path is not None or (
                        declared_root is not None and declared_root != expected_root
                    ):
                        raise PermissionError(
                            "late-bound immutable root differs from the exact producer output"
                        )
                    row["inputs"][logical_input] = {
                        UPSTREAM_OUTPUT_BINDING: dict(upstream),
                        IMMUTABLE_OUTPUT_ROOT_BINDING: dict(expected_root),
                    }
                else:
                    if declared_root is not None or (
                        declared_path is not None and declared_path != expected_output
                    ):
                        raise PermissionError(
                            "late-bound input path differs from the exact producer output"
                        )
                    row["inputs"][logical_input] = {
                        UPSTREAM_OUTPUT_BINDING: dict(upstream),
                        "path": expected_output,
                    }
    return with_content_hash({
        "contract": RUNTIME_BINDING_CONTRACT,
        "schema_version": 1,
        "command_plan_kind": RUNTIME_BINDING_KIND,
        "campaign_identity": campaign_identity,
        "campaign_identity_sha256": campaign_identity_sha256,
        "source_commit": spec["source_commit"],
        "runtime_facts": normalized_facts,
        "tasks": normalized_tasks,
    })


def validate_runtime_binding(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=RUNTIME_BINDING_CONTRACT, expected_schema_version=1,
    )
    if value.get("command_plan_kind") != RUNTIME_BINDING_KIND:
        raise ValueError("HCWDL-RKD runtime binding subtype differs")
    identity = runtime_campaign_identity(spec)
    if (
        value.get("campaign_identity") != identity
        or value.get("campaign_identity_sha256") != canonical_sha256(identity)
        or value.get("source_commit") != spec.get("source_commit")
    ):
        raise ValueError("HCWDL-RKD runtime binding campaign/source lineage differs")
    rows = value.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("HCWDL-RKD runtime binding task rows differ")
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "task_key", "kind", "deterministic_worker", "rows",
        }:
            raise ValueError("HCWDL-RKD runtime task binding fields differ")
        key = str(row["task_key"])
        if key in by_key:
            raise ValueError("HCWDL-RKD runtime binding repeats a task")
        by_key[key] = row
    raw_task_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for task in _campaign_tasks(spec):
        row = by_key.get(task.task_key)
        if row is None or row["kind"] != task.kind or row[
            "deterministic_worker"
        ] is not task.deterministic_worker:
            raise ValueError("HCWDL-RKD runtime task identity differs")
        row_values = row["rows"]
        if not isinstance(row_values, list):
            raise ValueError("HCWDL-RKD runtime task array rows differ")
        mapped: dict[str, Mapping[str, Any]] = {}
        for item in row_values:
            if not isinstance(item, Mapping):
                raise ValueError("HCWDL-RKD runtime row differs")
            index = item.get("array_index")
            key = "single" if index is None else str(index)
            if key in mapped:
                raise ValueError("HCWDL-RKD runtime repeats an array row")
            mapped[key] = item
        raw_task_rows[task.task_key] = mapped
    expected = build_runtime_binding(
        spec=spec,
        runtime_facts=value.get("runtime_facts", {}),
        task_rows=raw_task_rows,
    )
    if dict(value) != expected:
        raise ValueError("HCWDL-RKD runtime binding differs from canonical form")
    return digest


def resolve_runtime_row(
    binding: Mapping[str, Any], *, spec: Mapping[str, Any],
    task_key: str, array_index: int | None,
) -> Mapping[str, Any]:
    validate_runtime_binding(binding, spec=spec)
    matches = [row for row in binding["tasks"] if row["task_key"] == task_key]
    if len(matches) != 1:
        raise KeyError("HCWDL-RKD runtime task row is absent")
    rows = [row for row in matches[0]["rows"] if row["array_index"] == array_index]
    if len(rows) != 1:
        raise KeyError("HCWDL-RKD runtime array row is absent")
    return rows[0]


def load_runtime_binding(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Load and authenticate the campaign's one concrete runtime binding."""

    from hlt_classification.data.cache_contracts import load_json

    path = Path(str(spec["artifact_paths"]["runtime_binding"]))
    value = load_json(path)
    digest = validate_runtime_binding(value, spec=spec)
    if digest != spec.get("runtime_binding_sha256"):
        raise ValueError("HCWDL-RKD runtime-binding hash differs from campaign")
    return value


__all__ = [
    "CAMPAIGN_ARTIFACT_BINDING", "IMMUTABLE_OUTPUT_ROOT_BINDING",
    "PREPUBLISHED_OUTPUT_BINDING", "RUNTIME_BINDING_CONTRACT",
    "RUNTIME_BINDING_KIND", "RUNTIME_FACT_KEYS", "UPSTREAM_OUTPUT_BINDING",
    "build_runtime_binding",
    "load_runtime_binding", "resolve_runtime_row", "runtime_campaign_identity",
    "validate_prepublished_output_binding", "validate_runtime_binding",
]
