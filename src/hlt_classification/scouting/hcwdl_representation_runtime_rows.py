"""Deterministic construction of every HCWDL-RKD production runtime row.

The campaign registry deliberately freezes logical routes, while
``hcwdl_representation_runtime_binding`` freezes their concrete paths.  This
module is the only bridge between those layers.  It derives every scalar and
array row from one campaign specification plus one closed operator
prerequisite registry; callers cannot supply a hand-written per-task
``assembly`` object.

Exogenous bytes are literal ``path``/``sha256`` references.  Campaign products
are always exact ``upstream_output`` references.  Values which do not exist
until an upstream JSON artifact is published use ``registered_field`` tags and
are resolved by the fixed production adapter at execution time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file,
    validate_content_hash, with_content_hash,
)

from .hcwdl_representation_campaign import (
    CampaignTask,
    PARENT_IMPORT_AUTHORITY_ROUTES,
    PARENT_QUALIFIER_REPORT_ROUTES,
)
from .hcwdl_representation_contracts import (
    ARCHITECTURE_ATTESTATION_CONTRACT, PARENT_IMPORT_CONTRACT,
    PARENT_LOSS_ATTESTATION_CONTRACT,
    RUNTIME_DRY_RUN_AUDIT_CONTRACT, RUNTIME_PREREQUISITES_CONTRACT,
    SURFACE_PARITY_CONTRACT, contract_schema_version,
)
from .hcwdl_representation_graph import CONTROL_REGISTRY, NODE_REGISTRY
from .hcwdl_representation_locks import IMPORTED_LOGIT_CONTROLS, IMPORTED_TEACHERS
from .hcwdl_representation_reporting import CONFIRMATION_SEEDS
from .hcwdl_representation_runtime_adapters import PRODUCTION_ADAPTER_CONTRACT
from .hcwdl_representation_runtime_binding import (
    CAMPAIGN_ARTIFACT_BINDING, IMMUTABLE_OUTPUT_ROOT_BINDING,
    UPSTREAM_OUTPUT_BINDING, runtime_campaign_identity,
)
from .hcwdl_representation_targets import derive_target_generation_id
from .hcwdl_representation_workflow import array_indices


def _required_final_comparisons() -> list[dict[str, str]]:
    pairs = (
        ("RSET_M6c", "M6c"),
        ("RREL_M6c", "M6c"),
        ("RREL_M6c", "RSET_M6c"),
        ("RSET_M6w", "M6w"),
        ("RREL_M6w", "M6w"),
        ("RREL_M6w", "RSET_M6w"),
        ("M6w", "M6c"),
        ("RSET_M6w", "RSET_M6c"),
        ("RREL_M6w", "RREL_M6c"),
    )
    return [
        {
            "comparison_id": f"{left}-minus-{right}",
            "left_id": left, "right_id": right, "sign": "left_minus_right",
        }
        for left, right in pairs
    ]


def _proportional_quotas(counts: Sequence[object], total: int) -> list[int]:
    values = [int(value) for value in counts]
    population = sum(values)
    if (
        len(values) != 15 or any(value < 0 for value in values)
        or total <= 0 or total > population
    ):
        raise ValueError("final-test class population/budget differs")
    numerators = [total * value for value in values]
    quotas = [value // population for value in numerators]
    remainder = total - sum(quotas)
    order = sorted(
        range(15), key=lambda index: (-(numerators[index] % population), index),
    )
    for index in order[:remainder]:
        quotas[index] += 1
    if (
        sum(quotas) != total
        or any(quota <= 0 or quota > count for quota, count in zip(quotas, values))
    ):
        raise RuntimeError("final-test proportional quota derivation failed")
    return quotas


def _final_selection_policy(
    spec: Mapping[str, Any], split_manifest: Mapping[str, Any],
) -> tuple[list[int], str]:
    digest = validate_content_hash(
        split_manifest,
        expected_contract="hlt_classification_scouting_file_split_v2",
        expected_schema_version=2,
    )
    if digest != require_sha256(spec["split_manifest_sha256"], name="split manifest"):
        raise ValueError("runtime split manifest differs from campaign")
    roles = split_manifest.get("roles")
    final_role = roles.get("final_test") if isinstance(roles, Mapping) else None
    if not isinstance(final_role, Mapping):
        raise ValueError("runtime split lacks final-test role")
    quotas = _proportional_quotas(
        final_role.get("class_counts", ()), int(spec["role_counts"]["final_test"]),
    )
    selection_sha256 = canonical_sha256({
        "split_manifest_sha256": digest,
        "role": "final_test",
        "rows": int(spec["role_counts"]["final_test"]),
        "rule": "per_class_smallest_identity_sha256_rank_v1",
        "seed": 1337,
    })
    return quotas, selection_sha256


def _expected_view_cache_gib(spec: Mapping[str, Any]) -> float:
    resources = spec.get("resources")
    request = resources.get("gpu_representation") if isinstance(resources, Mapping) else None
    memory = str(request.get("memory", "")) if isinstance(request, Mapping) else ""
    if not memory.endswith("G") or not memory[:-1].isdigit() or int(memory[:-1]) <= 20:
        raise ValueError("representation resource memory cannot bound the RAM view cache")
    return float(int(memory[:-1]) - 20)


_SETTINGS_KEYS: Final = frozenset({
    "kernel_parent_hashes", "shuffle_parent_hashes", "target_budgets",
    "target_runtime_environment", "miniature_row_limit", "view_cache_max_gib",
    "synthetic_passes", "training_mode", "final",
})
_FINAL_SETTINGS_KEYS: Final = frozenset({
    "population_sha256", "selection_rule_sha256", "assignment_spec_sha256",
    "assignment_spec", "finalist_policy",
    "finalist_registry_commitment_sha256", "legacy_jobs_present",
    "parent_campaign_sha256", "parent_finalist_lock",
    "parent_finalist_registry_relative", "parent_finalist_registry_sha256",
    "parent_confirmation_report_keys",
    "rows_per_class", "step_size",
    "source_partitions", "finalists", "comparison_registry",
})
def _expected_kernel_parents(spec: Mapping[str, Any]) -> dict[str, str]:
    return {
        "parent_import": require_sha256(spec["parent_import_sha256"], name="parent import"),
        "representation_graph": require_sha256(spec["graph_sha256"], name="graph"),
        "representation_recipe": require_sha256(
            spec["representation_recipe_sha256"], name="representation recipe",
        ),
    }


def _expected_shuffle_parents(
    spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
) -> dict[str, str]:
    hashes = _content_hashes(prerequisites)
    return {
        "representation_graph": require_sha256(spec["graph_sha256"], name="graph"),
        "representation_recipe": require_sha256(
            spec["representation_recipe_sha256"], name="representation recipe",
        ),
        "split_manifest": require_sha256(spec["split_manifest_sha256"], name="split"),
        "train_row_selection": require_sha256(
            hashes["${train_row_selection}"], name="train row selection",
        ),
    }


def _derived_target_runtime_environment(
    target_forward_specs: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(target_forward_specs, Mapping) or not target_forward_specs:
        raise ValueError("target-forward authority registry is empty")
    expected = None
    from .hcwdl_representation_contracts import TARGET_FORWARD_SPEC_CONTRACT
    for logical, artifact in sorted(target_forward_specs.items()):
        if not isinstance(logical, str) or not logical.startswith("${target_forward_spec:"):
            raise ValueError("target-forward authority logical name differs")
        if not isinstance(artifact, Mapping):
            raise ValueError("target-forward authority artifact differs")
        validate_content_hash(
            artifact, expected_contract=TARGET_FORWARD_SPEC_CONTRACT,
            expected_schema_version=1,
        )
        payload = artifact.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("target-forward authority payload differs")
        environment = {
            name: payload[name]
            for name in ("producer", "device", "precision", "determinism")
            if name in payload
        }
        if set(environment) != {"producer", "device", "precision", "determinism"}:
            raise ValueError("target-forward runtime commitment is incomplete")
        if expected is None:
            expected = environment
        elif environment != expected:
            raise ValueError("target-forward runtime commitments disagree")
    assert expected is not None
    return dict(expected)


def _derived_final_authority(
    *, spec: Mapping[str, Any], runtime_facts: Mapping[str, Any],
    artifact_content_hashes: Mapping[str, Any], split_manifest: Mapping[str, Any],
    final_authorities: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "shared_final_population", "parent_final_state", "final_disposition",
        "parent_import", "parent_finalist_registry", "locked_model_members",
        "parent_confirmation_registry", "paired_screen_model_members",
        "matcher_resources",
    }
    if not isinstance(final_authorities, Mapping) or set(final_authorities) != required:
        raise ValueError("final authority registry differs")
    hashes = _content_hashes({"artifact_content_hashes": artifact_content_hashes})

    population = final_authorities["shared_final_population"]
    if not isinstance(population, Mapping):
        raise ValueError("shared final population authority differs")
    population_artifact = validate_content_hash(
        population, expected_contract="HCWDL_SHARED_FINAL_POPULATION/v1",
        expected_schema_version=1,
    )
    population_sha256 = require_sha256(
        population.get("population_sha256"), name="final population",
    )
    if (
        hashes.get("${shared_final_population}") != population_artifact
        or population.get("split_manifest_sha256") != spec["split_manifest_sha256"]
        or population.get("source_snapshot_sha256")
        != runtime_facts["source_snapshot_sha256"]
        or population.get("role") != "final_test"
        or population.get("row_count") != int(spec["role_counts"]["final_test"])
    ):
        raise ValueError("shared final population authority differs from campaign")

    from .hcwdl_shared_final import validate_parent_final_state

    parent_state = final_authorities["parent_final_state"]
    if not isinstance(parent_state, Mapping):
        raise ValueError("parent final-state authority differs")
    parent_state_sha256 = validate_parent_final_state(parent_state)
    if hashes.get("${parent_final_state}") != parent_state_sha256:
        raise ValueError("parent final-state authority differs from registered input")

    from .hcwdl_representation_locks import validate_parent_import

    parent_import = final_authorities["parent_import"]
    if not isinstance(parent_import, Mapping):
        raise ValueError("parent-import final authority differs")
    parent_import_sha256 = validate_parent_import(parent_import)
    if (
        parent_import_sha256 != spec["parent_import_sha256"]
        or hashes.get("${prebuilt_parent_import}") != parent_import_sha256
        or parent_import["parents"].get("parent_campaign_spec")
        != parent_state.get("parent_campaign_sha256")
        or parent_import["parents"].get("split_manifest")
        != spec["split_manifest_sha256"]
    ):
        raise ValueError("parent import/final authority lineage differs")

    from .hcwdl_locks import validate_lock

    confirmation_lock = final_authorities["parent_confirmation_registry"]
    if not isinstance(confirmation_lock, Mapping):
        raise ValueError("parent confirmation-registry authority differs")
    confirmation_lock_sha256 = validate_lock(
        confirmation_lock, expected_level="confirmation_registry",
    )
    confirmation_payload = confirmation_lock.get("payload")
    confirmation_registry = (
        confirmation_payload.get("registry")
        if isinstance(confirmation_payload, Mapping) else None
    )
    if (
        confirmation_lock_sha256
        != parent_import["parents"].get("confirmation_registry_lock")
        or confirmation_lock.get("campaign_spec_sha256")
        != parent_import["parents"].get("parent_campaign_spec")
        or not isinstance(confirmation_registry, list)
        or not confirmation_registry
    ):
        raise ValueError("parent confirmation registry differs from parent import")
    confirmation_report_keys: list[str] = []
    for index, row in enumerate(confirmation_registry):
        node_id = row.get("node_id") if isinstance(row, Mapping) else None
        seed = row.get("seed") if isinstance(row, Mapping) else None
        if (
            not isinstance(node_id, str) or not node_id
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("parent confirmation registry row identity differs")
        confirmation_report_keys.append(f"{index:03d}:{node_id}:{seed}")
    if len(confirmation_report_keys) != len(set(confirmation_report_keys)):
        raise ValueError("parent confirmation report keys are not unique")

    disposition = final_authorities["final_disposition"]
    if not isinstance(disposition, Mapping):
        raise ValueError("final disposition authority differs")
    disposition_sha256 = validate_content_hash(
        disposition, expected_contract="HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1",
        expected_schema_version=1,
    )
    if (
        hashes.get("${final_disposition}") != disposition_sha256
        or disposition_sha256 != spec["disposition_sha256"]
        or disposition.get("parent_final_state_sha256") != parent_state_sha256
        or disposition.get("disposition") != spec["disposition"]
        or bool(disposition.get("final_tasks_registered"))
        != (spec["disposition"] == "combined_confirmatory")
    ):
        raise ValueError("final disposition authority differs from campaign")

    parent_lock = final_authorities["parent_finalist_registry"]
    if not isinstance(parent_lock, Mapping):
        raise ValueError("parent finalist-registry authority differs")

    roles = split_manifest.get("roles")
    final_role = roles.get("final_test") if isinstance(roles, Mapping) else None
    files = final_role.get("files") if isinstance(final_role, Mapping) else None
    if not isinstance(files, list) or not files:
        raise ValueError("split final-test source inventory is absent")
    source_partitions = []
    for row in files:
        path = row.get("path") if isinstance(row, Mapping) else None
        if not isinstance(path, str) or not path:
            raise ValueError("split final-test source identity differs")
        source_partitions.append(path)
    if (
        source_partitions != sorted(source_partitions)
        or len(source_partitions) != len(set(source_partitions))
        or len(source_partitions) != int(spec["final_source_partitions"])
    ):
        raise ValueError("final source partitions differ from authenticated split")

    from .hcwdl_representation_final_policy import (
        build_final_assignment_spec,
        build_pretraining_finalist_policy_commitment,
        project_parent_finalists,
    )

    parent_finalists = project_parent_finalists(
        parent_finalist_lock=parent_lock,
        locked_model_members=final_authorities["locked_model_members"],
        paired_screen_model_members=final_authorities[
            "paired_screen_model_members"
        ],
    )
    parent_registry_sha256 = require_sha256(
        parent_lock.get("content_hash"), name="parent finalist lock",
    )
    if (
        parent_lock.get("campaign_spec_sha256")
        != parent_state["parent_campaign_sha256"]
        or parent_registry_sha256 != parent_import["parents"].get("finalist_lock")
    ):
        raise ValueError("parent finalist lock belongs to another parent campaign")

    matcher_resources = final_authorities["matcher_resources"]
    if not isinstance(matcher_resources, Mapping):
        raise ValueError("matcher resource authority differs")
    if hashes.get("${matcher_resources}") != matcher_resources.get("content_hash"):
        raise ValueError("matcher resource authority differs from registered input")
    assignment_spec = build_final_assignment_spec(
        matcher_resources=matcher_resources, source_partitions=source_partitions,
    )
    finalist_policy = build_pretraining_finalist_policy_commitment(
        parent_finalists=parent_finalists,
        parent_finalist_lock=parent_lock,
    )

    return {
        "population_sha256": population_sha256,
        "assignment_spec_sha256": assignment_spec["content_hash"],
        "assignment_spec": assignment_spec,
        "finalist_registry_commitment_sha256": finalist_policy["content_hash"],
        "finalist_policy": finalist_policy,
        "parent_campaign_sha256": require_sha256(
            parent_state.get("parent_campaign_sha256"), name="parent campaign",
        ),
        "parent_finalist_registry_sha256": parent_registry_sha256,
        "parent_finalist_lock": dict(parent_lock),
        "parent_confirmation_report_keys": sorted(confirmation_report_keys),
        "legacy_jobs_present": bool(parent_state.get("legacy_jobs")),
        "source_partitions": source_partitions,
        "finalists": list(parent_finalists),
    }

_EXACT_OUTPUT = re.compile(
    r"^\$\{task_output:([A-Za-z0-9_]+)(?:\[([0-9]+)\])?:([0-9]+)\}$"
)
_LEGACY_OUTPUT = re.compile(r"^\$\{task_output:([A-Za-z0-9_]+)\}$")

_ASSEMBLY_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "target_build": (
        "bank_root", "logical_bank", "consumer_registry", "forward_spec",
        "split_manifest", "row_selection", "data_root", "teacher_view",
        "source_partitions", "assignment_manifest", "teacher_report",
        "parent_import", "architecture_attestation",
        "kernel_envelope", "build_owner", "budgets", "storage_estimate",
        "resource_profile", "runtime_environment",
    ),
    "train_node": (
        "parent_recipe", "representation_recipe", "split_manifest",
        "row_selection", "data_root", "train_rows", "validation_rows",
        "target", "kernel_envelope", "execution_id", "registered_execution_id",
        "replicate_seed", "mode", "synthetic_passes", "resume_lineage",
        "producer_runtime_signature", "architecture_attestation",
        "parent_import", "model_sources", "shuffle_map", "view_cache_max_gib",
        "registered_output_row", "publication_owner", "confirmation_registry",
    ),
    "final_selection": (
        "split_manifest", "population", "claim", "task_registry", "task_id",
        "data_root", "rows_per_class", "selection_rule_sha256", "step_size",
        "escrow_root", "producer_task_id", "registered_output_row",
        "publication_owner",
    ),
    "assignment_shard": (
        "split_manifest", "selection", "matcher_resources", "claim",
        "task_registry", "task_id", "data_root", "source_path", "step_size",
        "assignment_spec_sha256",
        "envelope_root", "producer_task_id", "registered_output_row",
        "publication_owner",
    ),
    "assignment_finalize": (
        "split_manifest", "selection", "matcher_resources", "claim",
        "task_registry", "assignment_spec_sha256", "shards",
        "expected_mapped_jets", "require_sub10pct_dustbins",
    ),
    "prediction_shard": (
        "split_manifest", "selection", "prediction_spec", "finalist_lock",
        "execution_lock", "claim", "task_registry", "task_id", "data_root",
        "source_partition", "finalist_id", "model_source",
        "assignment_manifest", "step_size", "producer_runtime_signature",
        "envelope_root", "producer_task_id", "registered_output_row",
        "publication_owner",
    ),
    "prediction_finalize": (
        "selection", "prediction_spec", "execution_lock", "finalist",
        "shards", "expected_source_partitions",
    ),
    "metric_join": (
        "selection", "label_escrow", "prediction_spec", "execution_lock",
        "finalist_lock", "data_attestation", "claim", "task_registry",
        "task_id", "prediction_manifests", "prediction_shards",
        "evaluation_output_paths", "comparison_registry", "bootstrap_root",
        "bootstrap_output_rows", "publication_owner", "producer_task_id_prefix",
    ),
    "validation_only_aggregate": (
        "screen_aggregate", "confirmation_aggregate", "final_disposition",
    ),
    "execution_lock": (
        "finalist_lock", "data_attestation", "claim", "task_registry",
        "row_selection", "prediction_runtime_signature", "source_partitions",
    ),
    "final_aggregate": (
        "metric_join", "evaluations", "finalist_lock", "execution_lock",
        "confirmation_aggregate", "comparison_registry", "bootstrap_root",
    ),
}

_ASSEMBLY_CONTRACT: Final = {
    "target_build": "HCWDL_REPRESENTATION_TARGET_ASSEMBLY/v1",
    "train_node": "HCWDL_REPRESENTATION_TRAINING_ASSEMBLY/v1",
    "final_selection": "HCWDL_REPRESENTATION_FINAL_SELECTION_ASSEMBLY/v1",
    "assignment_shard": "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_ASSEMBLY/v1",
    "assignment_finalize": "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_ASSEMBLY/v1",
    "prediction_shard": "HCWDL_REPRESENTATION_FINAL_PREDICTION_ASSEMBLY/v1",
    "prediction_finalize": "HCWDL_REPRESENTATION_FINAL_PREDICTION_ASSEMBLY/v1",
    "metric_join": "HCWDL_REPRESENTATION_FINAL_JOIN_ASSEMBLY/v1",
    "validation_only_aggregate": "HCWDL_REPRESENTATION_VALIDATION_ONLY_ASSEMBLY/v1",
    "execution_lock": "HCWDL_REPRESENTATION_FINAL_EXECUTION_ASSEMBLY/v1",
    "final_aggregate": "HCWDL_REPRESENTATION_FINAL_AGGREGATE_ASSEMBLY/v1",
}

_OUTER_REQUIRED: Final[Mapping[str, tuple[str, ...]]] = {
    "tap_schema": (), "surface_parity": ("fixture_seed", "runtime_kind"),
    "architecture_attestation": (
        "tap_schema_path", "surface_parity_path", "parent_report_paths",
        "model_source_paths",
    ),
    "parent_loss_attestation": (
        "parent_campaign_spec_path", "parent_recipe_path", "parent_report_paths",
        "runtime_source_paths",
    ),
    "parent_import": (
        "artifact", "architecture_attestation", "parent_loss_attestation",
        "parent_report_paths", "model_source_paths", "authority_files",
        "qualifier_report_paths", "confirmation_report_paths",
    ),
    "control_registry": (),
    "kernel_resources": (
        "root", "producer_task_id", "immutable_parent_hashes",
        "registered_output_row", "campaign_or_recovery_owner", "recipe",
    ),
    "representation_recipe": (
        "artifact", "producer_source_sha256", "representation_graph",
        "control_registry", "parent_import",
    ),
    "numerical_acceptance": ("representation_recipe",),
    "cache_miniature_bank": (
        "bank_root", "generation_id", "bounded_row_limit", "cleanup_root",
    ),
    "cache_miniature": (), "smoke_probe": (),
    "zero_coefficient_acceptance": (),
    "reservation": (
        "builder_arguments", "assignment_source_partitions", "parent_finalists",
        "matcher_resources", "parent_finalist_lock",
    ),
    "shuffle_map": (
        "data_root", "parent_hashes", "output_root", "registered_output_row",
        "owner",
    ),
    "target_cleanup": (
        "committed_directory", "cleanup_root", "consumer_report_paths",
        "exact_reconstruction_authorized",
    ),
    "screen_aggregate": (
        "parent_import", "architecture_attestation", "builder_arguments",
    ),
    "confirmation_registry": ("builder_arguments",),
    "confirmation_aggregate": ("builder_arguments",),
    "finalist_lock": ("builder_arguments",),
    "shared_final_claim": ("task_registry_arguments", "claim_arguments"),
    "data_attestation": ("builder_arguments",),
}


def _tasks(spec: Mapping[str, Any]) -> tuple[CampaignTask, ...]:
    rows = spec.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("HCWDL-RKD runtime-row campaign task registry differs")
    return tuple(CampaignTask(**{
        **row,
        "dependencies": tuple(row["dependencies"]),
        "registered_inputs": tuple(row["registered_inputs"]),
        "registered_outputs": tuple(row["registered_outputs"]),
    }) for row in rows)


def _absolute(value: object, *, name: str) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if not text or not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError(f"{name} is not an absolute normalized POSIX path")
    return text


def _registered_json(logical: str) -> dict[str, str]:
    return {"registered_json": logical}


def _registered_reference(logical: str) -> dict[str, str]:
    return {"registered_reference": logical}


def _registered_path(logical: str) -> dict[str, str]:
    return {"registered_path": logical}


def _registered_member(
    logical: str, relative: str, *, mode: str = "path",
) -> dict[str, Any]:
    return {"registered_member": {
        "input": logical, "relative": relative, "mode": mode,
    }}


def _registered_field(
    logical: str, *field: str, relative: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"input": logical, "field": list(field)}
    if relative is not None:
        payload["relative"] = relative
    return {"registered_field": payload}


def _row_key(index: int | None) -> str:
    return "single" if index is None else str(index)


def _registered_output_row(
    task: CampaignTask, index: int | None, logical: str,
) -> dict[str, Any]:
    return {
        "task_key": task.task_key,
        "array_index": index,
        "registered_output": logical,
    }


def _producer_owner(
    spec: Mapping[str, Any], task: CampaignTask, index: int | None,
) -> dict[str, Any]:
    return {
        "campaign_task": task.task_key,
        "array_index": index,
        "owner_kind": "initial_campaign",
        "campaign_identity_sha256": canonical_sha256(
            runtime_campaign_identity(spec),
        ),
    }


def _content_hashes(prerequisites: Mapping[str, Any]) -> Mapping[str, str]:
    value = prerequisites.get("artifact_content_hashes")
    if not isinstance(value, Mapping):
        raise ValueError("runtime prerequisites lack artifact content hashes")
    return {str(key): require_sha256(raw, name=f"artifact content hash {key}")
            for key, raw in value.items()}


def _output_contract(task: CampaignTask, ordinal: int) -> str | None:
    """Return the exact JSON contract for one registered producer output."""

    kind = task.kind
    fixed: Mapping[str, tuple[str | None, ...]] = {
        "tap_schema": ("HCWDL_REPRESENTATION_TAP/v1",),
        "surface_parity": (SURFACE_PARITY_CONTRACT,),
        "architecture_attestation": (
            ARCHITECTURE_ATTESTATION_CONTRACT,
        ),
        "parent_loss_attestation": (
            PARENT_LOSS_ATTESTATION_CONTRACT,
        ),
        "parent_import": (PARENT_IMPORT_CONTRACT,),
        "control_registry": ("HCWDL_REPRESENTATION_CONTROL_REGISTRY/v1",),
        "kernel_resources": (None,),
        "representation_recipe": (
            "HCWDL_REPRESENTATION_RECIPE/v2",
        ),
        "numerical_acceptance": (
            "HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1",
        ),
        "cache_miniature_bank": (
            "HCWDL_REPRESENTATION_CACHE_MINIATURE_BANK/v1",
            "HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1",
            "HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1",
        ),
        "cache_miniature": (
            "HCWDL_REPRESENTATION_CACHE_MINIATURE/v1",
        ),
        "target_build": (None,),
        "smoke_probe": ("HCWDL_REPRESENTATION_SMOKE_PROBE/v1",),
        "zero_coefficient_acceptance": (
            "HCWDL_REPRESENTATION_ZERO_COEFFICIENT_ACCEPTANCE/v1",
        ),
        "reservation": (
            "HCWDL_SHARED_FINAL_RESERVATION/v1",
            "HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_SPEC/v1",
            "HCWDL_REPRESENTATION_PRETRAINING_FINALIST_POLICY/v1",
        ),
        "shuffle_map": (None,),
        "target_cleanup": (
            "HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1",
            "HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1",
        ),
        "screen_aggregate": ("HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v1",),
        "confirmation_registry": (
            "HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v1",
        ),
        "confirmation_aggregate": (
            "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1",
        ),
        "finalist_lock": ("HCWDL_REPRESENTATION_FINALIST_LOCK/v1",),
        "shared_final_claim": (
            "HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1",
            "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1",
        ),
        "final_selection": (
            "HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
            "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1",
            None,
            "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1",
        ),
        "assignment_shard": (
            None, "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1",
        ),
        "assignment_finalize": (
            "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2",
            "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1",
        ),
        "data_attestation": (
            "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1",
        ),
        "execution_lock": (
            "HCWDL_REPRESENTATION_EXECUTION_LOCK/v1",
            "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1",
        ),
        "prediction_shard": (
            None, "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1",
        ),
        "prediction_finalize": (
            "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v1",
        ),
        "metric_join": (
            "HCWDL_REPRESENTATION_METRIC_JOIN/v1", None, None,
            "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1",
        ),
        "final_aggregate": (
            "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v1",
        ),
        "validation_only_aggregate": (
            "HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1",
        ),
    }
    if kind in {"train_node", "train_control", "confirmation"}:
        values: tuple[str | None, ...] = (
            "HCWDL_REPRESENTATION_TRAINING_REPORT/v1",
            "HCWDL_REPRESENTATION_CHECKPOINT_SELECTION/v1",
            "HCWDL_REPRESENTATION_DEPLOYABLE_EXTRACTION/v1",
            None,
        )
        if kind == "confirmation":
            values += ("HCWDL_REPRESENTATION_CONFIRMATION_RUN/v1",)
    else:
        values = fixed[kind]
    if len(values) != len(task.registered_outputs):
        raise RuntimeError(f"output-contract registry differs for {task.task_key}")
    return values[ordinal]


def _upstream_reference(
    producer: CampaignTask, *, ordinal: int, array_index: int | None,
) -> dict[str, Any]:
    logical = producer.registered_outputs[ordinal]
    contract = _output_contract(producer, ordinal)
    kind = "json" if contract is not None else "directory"
    return {UPSTREAM_OUTPUT_BINDING: {
        "task_key": producer.task_key,
        "array_index": array_index,
        "registered_output": logical,
        "artifact_kind": kind,
        "expected_contract": contract,
        "expected_schema_version": (
            2
            if contract == "HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2"
            else (
                contract_schema_version(contract)
                if contract is not None else None
            )
        ),
    }}


def _alias_route(
    logical: str, *, task: CampaignTask, by_key: Mapping[str, CampaignTask],
) -> tuple[CampaignTask, int, int | None] | None:
    match = _EXACT_OUTPUT.fullmatch(logical)
    if match is not None:
        key, raw_index, raw_ordinal = match.groups()
        return by_key[key], int(raw_ordinal), (
            None if raw_index is None else int(raw_index)
        )
    match = _LEGACY_OUTPUT.fullmatch(logical)
    if match is not None:
        producer = by_key[match.group(1)]
        if producer.array is not None or len(producer.registered_outputs) != 1:
            raise ValueError("legacy producer alias is not one scalar output")
        return producer, 0, None
    aliases: dict[str, tuple[str, int]] = {
        "${parent_import}": ("parent_import", 0),
        "${representation_recipe}": ("representation_recipe", 0),
        "${control_registry}": ("control_registry", 0),
        "${kernel_resources}": ("kernel_resources", 0),
        "${zero_coefficient_acceptance}": ("zero_coefficient_acceptance", 0),
        "${task_output:architecture_attestation}": (
            "architecture_attestation", 0,
        ),
        "${task_output:parent_loss_attestation}": (
            "parent_loss_attestation", 0,
        ),
        "${cache_miniature:D100:evidence}": (
            "miniature_D100_verify_cleanup", 0,
        ),
        "${cache_miniature:D100:cleanup_authorization}": (
            "miniature_D100_verify_cleanup", 1,
        ),
        "${cache_miniature:D100:cleanup_completion}": (
            "miniature_D100_verify_cleanup", 2,
        ),
        "${cache_miniature:TOFF:evidence}": (
            "miniature_TOFF_verify_cleanup", 0,
        ),
        "${cache_miniature:TOFF:cleanup_authorization}": (
            "miniature_TOFF_verify_cleanup", 1,
        ),
        "${cache_miniature:TOFF:cleanup_completion}": (
            "miniature_TOFF_verify_cleanup", 2,
        ),
    }
    if logical.startswith("${array_registry:confirmation/"):
        aliases[logical] = ("confirmation_registry", 0)
    if logical.startswith("${array_registry:final/"):
        aliases[logical] = ("shared_claim_gate", 1)
    row = aliases.get(logical)
    if row is None:
        return None
    producer = by_key[row[0]]
    if producer.task_key not in task.dependencies and producer.task_key not in _ancestors(
        task.task_key, by_key,
    ):
        raise PermissionError("registered producer alias is not task ancestry")
    return producer, row[1], None


def _ancestors(key: str, by_key: Mapping[str, CampaignTask]) -> set[str]:
    found: set[str] = set()
    pending = list(by_key[key].dependencies)
    while pending:
        value = pending.pop()
        if value not in found:
            found.add(value)
            pending.extend(by_key[value].dependencies)
    return found


def _runtime_inputs(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, by_key: Mapping[str, CampaignTask],
) -> dict[str, Any]:
    static = prerequisites.get("static_inputs")
    if not isinstance(static, Mapping):
        raise ValueError("runtime prerequisites lack the static input registry")
    campaign_identity_sha256 = canonical_sha256(runtime_campaign_identity(spec))
    result: dict[str, Any] = {}
    for logical in task.registered_inputs:
        route = _alias_route(logical, task=task, by_key=by_key)
        if route is not None:
            producer, ordinal, array_index = route
            result[logical] = _upstream_reference(
                producer, ordinal=ordinal, array_index=array_index,
            )
        elif logical == "${campaign_spec}":
            result[logical] = {
                "path": _absolute(
                    PurePosixPath(str(spec["campaign_root"])) / "campaign_spec.json",
                    name="campaign-spec path",
                ),
                "sha256": campaign_identity_sha256,
            }
        elif logical == "${submission_ledger}":
            result[logical] = {CAMPAIGN_ARTIFACT_BINDING: {
                "artifact_role": "submission_ledger",
                "expected_contract": (
                    "HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1"
                ),
                "expected_schema_version": 1,
                "campaign_identity_sha256": campaign_identity_sha256,
                "command_plan_sha256": require_sha256(
                    spec.get("command_plan_sha256"), name="command plan",
                ),
            }}
        else:
            reference = static.get(logical)
            if not isinstance(reference, Mapping) or set(reference) != {
                "path", "sha256",
            }:
                raise ValueError(
                    f"exogenous registered input is absent: {logical}"
                )
            result[logical] = {
                "path": _absolute(reference["path"], name=f"{logical} path"),
                "sha256": require_sha256(
                    reference["sha256"], name=f"{logical} byte hash",
                ),
            }
    return result


def _target_key(task: CampaignTask) -> str:
    if task.logical_bank is None or task.target_purpose is None:
        raise ValueError("task does not own a target generation")
    return f"{task.logical_bank}:{task.target_purpose}"


def _target_record(
    prerequisites: Mapping[str, Any], task: CampaignTask,
) -> Mapping[str, Any]:
    targets = prerequisites.get("target_generations")
    key = _target_key(task)
    if not isinstance(targets, Mapping) or not isinstance(targets.get(key), Mapping):
        raise ValueError(f"runtime prerequisites lack target generation {key}")
    value = targets[key]
    required = {
        "generation_id", "generation_parent_sha256", "source_partitions",
        "execution_ids", "execution_campaign_sha256",
    }
    if set(value) != required:
        raise ValueError(f"target-generation prerequisite schema differs: {key}")
    generation = require_sha256(value["generation_id"], name=f"{key} generation")
    generation_parent = require_sha256(
        value["generation_parent_sha256"], name=f"{key} generation parent",
    )
    require_sha256(
        value["execution_campaign_sha256"],
        name=f"{key} execution campaign",
    )
    hashes = _content_hashes(prerequisites)
    bank_logical = f"${{logical_bank:{task.logical_bank}}}"
    registry_logical = (
        f"${{target_consumer_registry:{task.logical_bank}:{task.target_purpose}}}"
    )
    if bank_logical not in hashes or registry_logical not in hashes:
        raise ValueError(f"target identity content hashes are absent: {key}")
    expected = derive_target_generation_id(
        hashes[bank_logical], hashes[registry_logical],
        purpose=str(task.target_purpose),
        generation_parent_sha256=generation_parent,
    )
    if generation != expected:
        raise ValueError(f"target generation ID differs: {key}")
    partitions = value["source_partitions"]
    if not isinstance(partitions, Mapping) or not partitions:
        raise ValueError(f"target source partitions are empty: {key}")
    execution_ids = value["execution_ids"]
    if not isinstance(execution_ids, Mapping) or not execution_ids:
        raise ValueError(f"target execution registry is empty: {key}")
    return value


def _training_seed(task: CampaignTask, index: int | None) -> int:
    if task.kind == "confirmation":
        if index is None or index not in range(len(CONFIRMATION_SEEDS)):
            raise ValueError("confirmation array index differs")
        return int(CONFIRMATION_SEEDS[index])
    if index is not None:
        raise ValueError("nonconfirmation training row unexpectedly has an index")
    return 1337


def _registered_execution_id(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> str:
    record = _target_record(prerequisites, task)
    node = str(task.graph_node)
    seed = _training_seed(task, index)
    key = f"{node}:{seed}"
    ids = record["execution_ids"]
    if key not in ids:
        raise ValueError(f"target consumer execution is absent: {_target_key(task)}:{key}")
    if (NODE_REGISTRY.get(node) or CONTROL_REGISTRY.get(node)) is None:
        raise ValueError(f"training graph node is unregistered: {node}")
    # The exact ID is owned by the authenticated target-consumer registry.
    # Re-deriving it here would require duplicating the registry's campaign
    # identity payload and could accidentally authorize a nearby execution.
    return require_sha256(ids[key], name=f"{key} target consumer")


def _final_settings(prerequisites: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = prerequisites.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("runtime prerequisites lack settings")
    value = settings.get("final")
    if not isinstance(value, Mapping):
        raise ValueError("runtime prerequisites lack final settings")
    return value


def _finalists(
    prerequisites: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return parent finalists plus four deferred representation endpoints."""

    raw = _final_settings(prerequisites).get("finalists")
    if not isinstance(raw, list) or not raw:
        raise ValueError("parent finalist runtime registry is empty")
    parent_rows = tuple(raw)
    ids: set[str] = set()
    for row in parent_rows:
        required = {
            "finalist_id", "checkpoint_sha256", "report_sha256", "domain",
            "deployable", "extraction_sha256", "source_campaign",
            "screening_seed", "execution_id", "checkpoint_selection_sha256",
            "model_kind", "model_relative",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("finalist runtime row schema differs")
        finalist = str(row["finalist_id"])
        if (
            not finalist or finalist in ids
            or finalist in {"RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"}
            or row["source_campaign"] == "representation"
            or row["domain"] not in {
            "hlt", "shell_exact_d100", "native_offline",
            }
        ):
            raise ValueError("parent finalist runtime identity/domain differs")
        ids.add(finalist)
        require_sha256(row["checkpoint_sha256"], name=f"{finalist} checkpoint")
        require_sha256(row["report_sha256"], name=f"{finalist} report")
        if not isinstance(row["deployable"], bool):
            raise ValueError("parent finalist deployment flag differs")
        if row["deployable"]:
            require_sha256(row["extraction_sha256"], name=f"{finalist} extraction")
            if row["domain"] != "hlt":
                raise ValueError("deployable parent finalist must consume HLT")
        elif row["extraction_sha256"] is not None:
            raise ValueError("nondeployable parent finalist claims an extraction")
        if row["model_kind"] not in {"pmard", "hcwdl"}:
            raise ValueError("finalist model kind differs")
    endpoints = []
    by_key = {task.task_key: task for task in _tasks(spec)}
    for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
        task = by_key[f"train_{node}"]
        report = f"${{task_output:{task.task_key}:0}}"
        selection = f"${{task_output:{task.task_key}:1}}"
        extraction = f"${{task_output:{task.task_key}:2}}"
        endpoints.append({
            "finalist_id": node,
            "checkpoint_sha256": _registered_field(
                extraction, "payload", "deployable_state_sha256",
            ),
            "report_sha256": _registered_field(report, "content_hash"),
            "domain": "hlt", "deployable": True,
            "extraction_sha256": _registered_field(extraction, "content_hash"),
            "source_campaign": "representation", "screening_seed": 1337,
            "execution_id": _registered_execution_id(
                spec=spec, prerequisites=prerequisites, task=task, index=None,
            ),
            "checkpoint_selection_sha256": _registered_field(
                selection, "content_hash",
            ),
            "model_kind": "hcwdl", "model_relative": None,
        })
    return (*parent_rows, *endpoints)


def _finalist_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key] for key in (
            "finalist_id", "checkpoint_sha256", "report_sha256", "domain",
            "deployable", "extraction_sha256", "source_campaign",
            "screening_seed", "execution_id", "checkpoint_selection_sha256",
        )
    }


def _final_task_ids(
    prerequisites: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> dict[str, Any]:
    final = _final_settings(prerequisites)
    partitions = final.get("source_partitions")
    if not isinstance(partitions, list) or not partitions or len(partitions) != len(
        set(map(str, partitions))
    ):
        raise ValueError("final source-partition registry differs")
    finalists = _finalists(prerequisites, spec=spec)
    parent_finalists = [
        row for row in finalists if row["source_campaign"] != "representation"
    ]
    finalist_members = _bundle_members(prerequisites, "finalist_models")
    if not isinstance(finalist_members, Mapping) or set(finalist_members) != {
        str(row["finalist_id"]) for row in parent_finalists
    }:
        raise ValueError("parent finalist model-source bundle is incomplete or expanded")
    for row in parent_finalists:
        raw = finalist_members[str(row["finalist_id"])]
        relative = raw.get("relative") if isinstance(raw, Mapping) else raw
        if relative != row["model_relative"]:
            raise ValueError("parent finalist model-source member differs")
    return {
        "selection": "row_selection",
        "assignment_shards": {
            str(source): f"assignment_shard:{position:04d}"
            for position, source in enumerate(partitions)
        },
        "assignment_finalize": "assignment_finalize",
        "data_attestation": "data_attestation",
        "execution_lock": "execution_lock",
        "prediction_shards": {
            (str(row["finalist_id"]), str(source)):
            f"prediction_shard:{row['finalist_id']}:{position:04d}"
            for row in finalists for position, source in enumerate(partitions)
        },
        "prediction_manifests": {
            str(row["finalist_id"]): f"prediction_finalize:{row['finalist_id']}"
            for row in finalists
        },
        "metric_join": "metric_join",
        "final_aggregate": "final_aggregate",
    }


def _substitutions(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> Mapping[str, str]:
    values: dict[str, str] = {}
    if task.logical_bank is not None:
        values["generation_id"] = str(_target_record(prerequisites, task)["generation_id"])
    if task.kind in {"train_node", "train_control", "confirmation"}:
        values["execution_id"] = _registered_execution_id(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )
    if task.kind in {
        "reservation", "shared_final_claim",
    }:
        population = require_sha256(
            _final_settings(prerequisites).get("population_sha256"),
            name="final population",
        )
        values["population_namespace"] = str(
            PurePosixPath(str(spec["checkpoint_namespace"]))
            / "final_claims" / population
        )
    if task.kind in {
        "assignment_shard", "prediction_shard", "prediction_finalize",
    }:
        ids = _final_task_ids(prerequisites, spec=spec)
        final = _final_settings(prerequisites)
        partitions = [str(value) for value in final["source_partitions"]]
        finalists = _finalists(prerequisites, spec=spec)
        if task.kind == "assignment_shard":
            assert index is not None
            source = partitions[index]
            values.update(source_partition=source, task_id=ids["assignment_shards"][source])
        elif task.kind == "prediction_shard":
            assert index is not None
            partition_index = index % len(partitions)
            finalist_index = index // len(partitions)
            finalist = str(finalists[finalist_index]["finalist_id"])
            source = partitions[partition_index]
            values.update(
                source_partition=source, finalist_id=finalist,
                task_id=ids["prediction_shards"][(finalist, source)],
            )
        else:
            assert index is not None
            finalist = str(finalists[index]["finalist_id"])
            values["finalist_id"] = finalist
    if task.kind == "metric_join":
        values["task_id"] = _final_task_ids(
            prerequisites, spec=spec,
        )["metric_join"]
    return values


def _replace_template(template: str, values: Mapping[str, str]) -> str:
    result = template
    for name, value in values.items():
        result = result.replace("${" + name + "}", value)
    return result


def _output_values(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> dict[str, Any]:
    root = PurePosixPath(_absolute(spec["campaign_root"], name="campaign root"))
    substitutions = _substitutions(
        spec=spec, prerequisites=prerequisites, task=task, index=index,
    )
    values: dict[str, Any] = {}
    for ordinal, logical in enumerate(task.registered_outputs):
        if "${envelope_id}" in logical:
            prefix = logical.removesuffix("/committed/${envelope_id}")
            if prefix == logical:
                raise ValueError("envelope output route is not canonical")
            envelope_root = _absolute(
                root / _replace_template(prefix, substitutions),
                name=f"{task.task_key} immutable output root",
            )
            values[logical] = {IMMUTABLE_OUTPUT_ROOT_BINDING: {
                "root": envelope_root,
                "artifact_contract": _binary_contract(task, ordinal),
                "producer_task_id": (
                    substitutions.get("task_id") or task.task_key
                ),
                "registered_output_row": _registered_output_row(
                    task, index, logical,
                ),
                "campaign_identity_sha256": canonical_sha256(
                    runtime_campaign_identity(spec),
                ),
                "expected_publication_owner": _producer_owner(
                    spec, task, index,
                ),
            }}
            continue
        relative = _replace_template(logical, substitutions)
        if "${" in relative:
            raise ValueError(
                f"unresolved output placeholder for {task.task_key}: {relative}"
            )
        path = PurePosixPath(relative)
        values[logical] = _absolute(
            path if path.is_absolute() else root / path,
            name=f"{task.task_key} output",
        )
    for logical, wrapped in values.items():
        if not isinstance(wrapped, Mapping) or IMMUTABLE_OUTPUT_ROOT_BINDING not in wrapped:
            continue
        wrapped[IMMUTABLE_OUTPUT_ROOT_BINDING]["expected_parent_sources"] = (
            _immutable_parent_sources(
                spec=spec, prerequisites=prerequisites, task=task,
                index=index, outputs=values,
            )
        )
    return values


def _binary_contract(task: CampaignTask, ordinal: int) -> str:
    if task.kind == "kernel_resources":
        return "HCWDL_REPRESENTATION_KERNEL_RESOURCES/v1"
    if task.kind == "shuffle_map":
        return "HCWDL_REPRESENTATION_SHUFFLE_MAP/v1"
    if task.kind == "final_selection" and ordinal == 2:
        return "HCWDL_SHARED_FINAL_LABEL_ESCROW/v1"
    if task.kind == "assignment_shard" and ordinal == 0:
        return "HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1"
    if task.kind == "prediction_shard" and ordinal == 0:
        return "HCWDL_REPRESENTATION_PREDICTION_SHARD/v1"
    raise ValueError(f"binary-envelope contract is absent for {task.task_key}:{ordinal}")


def _literal_parent(value: object) -> dict[str, Any]:
    return {
        "source_kind": "literal_sha256",
        "sha256": require_sha256(value, name="immutable envelope parent"),
    }


def _json_parent(path: object, contract: str) -> dict[str, Any]:
    return {
        "source_kind": "json_artifact",
        "path": _absolute(path, name="immutable envelope parent artifact"),
        "expected_contract": contract,
        "expected_schema_version": 1,
    }


def _embedded_json_parent(member: str, contract: str) -> dict[str, Any]:
    return {
        "source_kind": "envelope_json_member",
        "member": member,
        "expected_contract": contract,
        "expected_schema_version": 1,
    }


def _immutable_parent_sources(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None, outputs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Freeze how every future envelope parent hash must be derived.

    Future final-task JSON hashes do not exist while the runtime binding is
    assembled.  Their exact registered paths and contracts do, so the binding
    freezes typed derivations rather than accepting the envelope's own parent
    registry as its authority at consumption time.
    """

    root = PurePosixPath(_absolute(spec["campaign_root"], name="campaign root"))
    settings = _settings(prerequisites)
    if task.kind == "kernel_resources":
        return {
            name: _literal_parent(digest)
            for name, digest in sorted(settings["kernel_parent_hashes"].items())
        }
    if task.kind == "shuffle_map":
        return {
            name: _literal_parent(digest)
            for name, digest in sorted(settings["shuffle_parent_hashes"].items())
        }
    capability_contract = "HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1"
    branch_contract = "HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1"
    if task.kind == "final_selection":
        return {
            "selection": _json_parent(
                outputs[task.registered_outputs[0]],
                "HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
            ),
            "population": _literal_parent(
                _content_hashes(prerequisites)["${shared_final_population}"],
            ),
            "capability": _json_parent(
                outputs[task.registered_outputs[3]], capability_contract,
            ),
        }
    if task.kind == "assignment_shard":
        return {
            "split_manifest": _literal_parent(
                _content_hashes(prerequisites)["${split_manifest}"],
            ),
            "selection": _json_parent(
                root / "final/selection/row_selection.json",
                "HCWDL_SHARED_FINAL_ROW_SELECTION/v1",
            ),
            "matcher_resources": _literal_parent(
                _content_hashes(prerequisites)["${matcher_resources}"],
            ),
            "capability": _json_parent(
                outputs[task.registered_outputs[1]], capability_contract,
            ),
            "branch_access": _embedded_json_parent(
                "branch_access.json", branch_contract,
            ),
        }
    if task.kind == "prediction_shard":
        return {
            "prediction_spec": _json_parent(
                root / "final/prediction_spec.json",
                "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1",
            ),
            "execution_lock": _json_parent(
                root / "locks/07_execution.json",
                "HCWDL_REPRESENTATION_EXECUTION_LOCK/v1",
            ),
            "checkpoint": {
                "source_kind": "final_task_registry_field",
                "path": _absolute(
                    root / "final/task_registry.json",
                    name="final task registry",
                ),
                "expected_contract": "HCWDL_SHARED_FINAL_TASK_REGISTRY/v1",
                "expected_schema_version": 1,
                "task_id": str(
                    _substitutions(
                        spec=spec, prerequisites=prerequisites,
                        task=task, index=index,
                    )["task_id"]
                ),
                "field": "checkpoint_sha256",
            },
            "branch_access": _embedded_json_parent(
                "branch_access.json", branch_contract,
            ),
        }
    raise RuntimeError(
        f"immutable parent-source registry is absent for {task.task_key}"
    )


def _settings(prerequisites: Mapping[str, Any]) -> Mapping[str, Any]:
    value = prerequisites.get("settings")
    if not isinstance(value, Mapping):
        raise ValueError("runtime prerequisites lack settings")
    return value


def _bundle_members(
    prerequisites: Mapping[str, Any], name: str,
) -> Mapping[str, Any] | list[Any]:
    bundles = prerequisites.get("bundle_members")
    if not isinstance(bundles, Mapping) or name not in bundles:
        raise ValueError(f"runtime prerequisites lack bundle members: {name}")
    value = bundles[name]
    if not isinstance(value, (Mapping, list)) or not value:
        raise ValueError(f"runtime bundle member registry differs: {name}")
    return value


def _member_registry(
    prerequisites: Mapping[str, Any], *, bundle: str, logical: str,
    mode: str,
) -> dict[str, Any]:
    raw = _bundle_members(prerequisites, bundle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"runtime bundle {bundle} must be a mapping")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        relative = value["relative"] if isinstance(value, Mapping) else value
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"runtime bundle relative path differs: {bundle}:{key}")
        result[str(key)] = _registered_member(logical, relative, mode=mode)
    return result


def _parent_engine_report_registry(
    prerequisites: Mapping[str, Any], *, logical: str,
) -> dict[str, Any]:
    """Resolve engine reports through the registered parent wrapper bundle.

    The parent import and both attestations bind the HCWDL wrapper report.  The
    validation aggregate consumes the sibling PMARD engine report because that
    is where the selected validation metrics live.  Deriving the sibling name
    here keeps both files under one authenticated directory input and avoids a
    second, independently supplied report path.
    """

    raw = _bundle_members(prerequisites, "parent_reports")
    if not isinstance(raw, Mapping):
        raise ValueError("runtime parent-report bundle must be a mapping")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        relative = value["relative"] if isinstance(value, Mapping) else value
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"runtime parent report relative path differs: {key}")
        wrapper = PurePosixPath(relative)
        if wrapper.name == "training_report.json":
            raise ValueError("parent wrapper report cannot alias its engine report")
        engine = wrapper.parent / "training_report.json"
        result[str(key)] = _registered_member(
            logical, engine.as_posix(), mode="reference",
        )
    return result


def _runtime_source_list(prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    raw = _bundle_members(prerequisites, "parent_runtime_sources")
    from .hcwdl_parent_loss import PARENT_LOSS_RUNTIME_SOURCE_FILES

    if (
        not isinstance(raw, Mapping)
        or set(raw) != set(PARENT_LOSS_RUNTIME_SOURCE_FILES)
        or any(not isinstance(value, str) or not value for value in raw.values())
    ):
        raise ValueError("parent runtime-source member registry differs")
    return {
        name: _registered_member(
            "${parent_runtime_sources}", raw[name], mode="path",
        )
        for name in sorted(PARENT_LOSS_RUNTIME_SOURCE_FILES)
    }


def _parameters_base(task: CampaignTask) -> dict[str, Any]:
    return {
        "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
        "task_kind": task.kind,
    }


def _target_teacher_view(task: CampaignTask) -> str:
    bank = str(task.logical_bank)
    base = bank[:-1] if bank.endswith(("c", "w")) else bank
    mapping = {
        "D0": "hlt", "D25": "d25", "D50": "d50",
        "D75": "d75", "D100": "d100", "TOFF": "toff",
    }
    if base not in mapping:
        raise ValueError(f"target teacher view is unknown: {bank}")
    return mapping[base]


def _target_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> dict[str, Any]:
    del index
    record = _target_record(prerequisites, task)
    logical_bank = f"${{logical_bank:{task.logical_bank}}}"
    registry = (
        f"${{target_consumer_registry:{task.logical_bank}:{task.target_purpose}}}"
    )
    forward = f"${{target_forward_spec:{task.logical_bank}:{task.target_purpose}}}"
    assignment = None
    base = str(task.logical_bank).rstrip("cw")
    if base in {"D25", "D50", "D75", "D100"}:
        assignment = _registered_reference("${assignment_manifest:train}")
    kernel = "${kernel_resources}"
    settings = _settings(prerequisites)
    assembly = {
        "contract": _ASSEMBLY_CONTRACT["target_build"],
        "bank_root": str(
            PurePosixPath(str(spec["campaign_root"]))
            / "targets" / str(task.logical_bank)
        ),
        "logical_bank": _registered_reference(logical_bank),
        "consumer_registry": _registered_reference(registry),
        "forward_spec": _registered_reference(forward),
        "split_manifest": _registered_reference("${split_manifest}"),
        "row_selection": _registered_reference("${train_row_selection}"),
        "data_root": _absolute(
            prerequisites["runtime_facts"]["data_root"], name="data root",
        ),
        "teacher_view": _target_teacher_view(task),
        "source_partitions": dict(record["source_partitions"]),
        "assignment_manifest": assignment,
        "teacher_report": _registered_reference(
            f"${{teacher_report:{task.logical_bank}}}"
        ),
        "parent_import": _registered_reference("${task_output:parent_import:0}"),
        "architecture_attestation": _registered_reference(
            "${task_output:architecture_attestation:0}"
        ),
        "kernel_envelope": {
            "committed_directory": _registered_path(kernel),
        },
        "build_owner": _producer_owner(spec, task, None),
        "budgets": dict(settings["target_budgets"]),
        "storage_estimate": _registered_reference("${storage_estimate}"),
        "resource_profile": _registered_reference("${resource_profile}"),
        "runtime_environment": dict(settings["target_runtime_environment"]),
    }
    return {**_parameters_base(task), "assembly": assembly}


def _model_sources(
    *, prerequisites: Mapping[str, Any], task: CampaignTask,
) -> dict[str, Any]:
    execution = NODE_REGISTRY.get(str(task.graph_node)) or CONTROL_REGISTRY.get(
        str(task.graph_node)
    )
    if execution is None:
        raise ValueError("training execution graph node is absent")
    names = {
        value for value in (
            execution.initialization_parent,
            execution.predecessor_logit_teacher,
        ) if value is not None
    }
    parent_members = _bundle_members(prerequisites, "parent_model_sources")
    if not isinstance(parent_members, Mapping):
        raise ValueError("parent model-source member registry differs")
    result: dict[str, Any] = {}
    for name in sorted(names):
        if name in NODE_REGISTRY:
            producer = f"train_{name}"
            logical = f"${{task_output:{producer}:3}}"
            if logical not in task.registered_inputs:
                raise ValueError(
                    f"training predecessor route is absent: {task.task_key}:{name}"
                )
            result[name] = {
                "kind": "hcwdl",
                "execution_directory": _registered_path(logical),
            }
            continue
        if name not in parent_members:
            raise ValueError(f"parent model source is absent: {name}")
        raw = parent_members[name]
        if isinstance(raw, Mapping):
            if set(raw) != {"relative", "kind"} or raw["kind"] != "pmard":
                raise ValueError(f"parent model source schema differs: {name}")
            relative = raw["relative"]
        else:
            relative = raw
        result[name] = {
            "kind": "pmard",
            "report": _registered_member(
                "${parent_model_sources}", str(relative), mode="reference",
            ),
        }
    return result


def _training_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> dict[str, Any]:
    execution_id = _registered_execution_id(
        spec=spec, prerequisites=prerequisites, task=task, index=index,
    )
    target_task = f"target_{task.logical_bank}_{task.target_purpose}"
    target_logical = f"${{task_output:{target_task}:0}}"
    settings = _settings(prerequisites)
    mode = "smoke" if spec.get("mode") == "smoke" else "scientific"
    passes = 1 if mode == "smoke" else 60
    if settings.get("training_mode", mode) != mode or int(
        settings.get("synthetic_passes", passes)
    ) != passes:
        raise PermissionError(
            "campaign training mode/pass count cannot be overridden by runtime inputs"
        )
    shuffle_logical = "${task_output:shuffle_map:0}"
    shuffled = bool(
        getattr(
            CONTROL_REGISTRY.get(str(task.graph_node)),
            "shuffled_representation_targets", False,
        )
    )
    confirmation = (
        _registered_reference("${task_output:confirmation_registry:0}")
        if task.kind == "confirmation" else None
    )
    assembly = {
        "contract": _ASSEMBLY_CONTRACT["train_node"],
        "parent_recipe": _registered_reference("${parent_recipe}"),
        "representation_recipe": _registered_reference("${representation_recipe}"),
        "split_manifest": _registered_reference("${split_manifest}"),
        "row_selection": _registered_reference("${train_validation_row_selection}"),
        "data_root": _absolute(
            prerequisites["runtime_facts"]["data_root"], name="data root",
        ),
        "train_rows": int(spec["role_counts"]["train"]),
        "validation_rows": int(spec["role_counts"]["validation"]),
        "target": {"committed_directory": _registered_path(target_logical)},
        "kernel_envelope": {
            "committed_directory": _registered_path("${kernel_resources}"),
        },
        "execution_id": str(task.graph_node),
        "registered_execution_id": execution_id,
        "replicate_seed": _training_seed(task, index),
        "mode": mode,
        "synthetic_passes": passes,
        "resume_lineage": {
            "ascent_graph": require_sha256(spec["graph_sha256"], name="graph"),
            "execution": execution_id,
            "producer_runtime_signature": _registered_field(
                "${producer_runtime_signature}", "content_hash",
            ),
            "representation_recipe": require_sha256(
                spec["representation_recipe_sha256"], name="representation recipe",
            ),
        },
        "producer_runtime_signature": _registered_reference(
            "${producer_runtime_signature}"
        ),
        "architecture_attestation": _registered_reference(
            "${task_output:architecture_attestation:0}"
        ),
        "parent_import": _registered_reference("${task_output:parent_import:0}"),
        "model_sources": _model_sources(
            prerequisites=prerequisites, task=task,
        ),
        "shuffle_map": (
            {"committed_directory": _registered_path(shuffle_logical)}
            if shuffled else None
        ),
        "view_cache_max_gib": float(settings["view_cache_max_gib"]),
        "registered_output_row": _registered_output_row(
            task, index, task.registered_outputs[3],
        ),
        "publication_owner": _producer_owner(spec, task, index),
        "confirmation_registry": confirmation,
    }
    return {**_parameters_base(task), "assembly": assembly}


def _training_report_routes(task: CampaignTask) -> dict[str, str]:
    result: dict[str, str] = {}
    if task.kind == "confirmation":
        for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
            result[f"{task.graph_node}:{seed}"] = (
                f"${{task_output:{task.task_key}[{seed_index}]:0}}"
            )
    else:
        result[str(task.graph_node)] = f"${{task_output:{task.task_key}:0}}"
    return result


def _simple_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None, by_key: Mapping[str, CampaignTask],
) -> dict[str, Any]:
    del index
    base = _parameters_base(task)
    kind = task.kind
    settings = _settings(prerequisites)
    if kind in {
        "tap_schema", "control_registry",
        "cache_miniature", "smoke_probe", "zero_coefficient_acceptance",
    }:
        return base
    if kind == "surface_parity":
        return {**base, "fixture_seed": 1337, "runtime_kind": "installed_weaver"}
    if kind == "architecture_attestation":
        return {
            **base,
            "tap_schema_path": _registered_path("${task_output:tap_schema:0}"),
            "surface_parity_path": _registered_path(
                "${task_output:surface_parity:0}"
            ),
            "parent_report_paths": _member_registry(
                prerequisites, bundle="parent_reports",
                logical="${parent_reports}", mode="path",
            ),
            "model_source_paths": _member_registry(
                prerequisites, bundle="parent_model_sources",
                logical="${parent_model_sources}", mode="path",
            ),
        }
    if kind == "parent_loss_attestation":
        return {
            **base,
            "parent_campaign_spec_path": _registered_path("${parent_campaign_spec}"),
            "parent_recipe_path": _registered_path("${parent_recipe}"),
            "parent_report_paths": _member_registry(
                prerequisites, bundle="parent_reports",
                logical="${parent_reports}", mode="path",
            ),
            "runtime_source_paths": _runtime_source_list(prerequisites),
        }
    if kind == "parent_import":
        final = settings.get("final")
        if not isinstance(final, Mapping):
            raise ValueError("runtime settings lack the final registry")
        parent_finalist_relative = final.get("parent_finalist_registry_relative")
        if not isinstance(parent_finalist_relative, str) or not parent_finalist_relative:
            raise ValueError("parent finalist-registry member path differs")
        authority_files = {}
        for name, logical in PARENT_IMPORT_AUTHORITY_ROUTES.items():
            authority_files[name] = (
                _registered_member(
                    logical, parent_finalist_relative, mode="path",
                )
                if name == "finalist_lock"
                else _registered_path(logical)
            )
        authority_files.update({
            "architecture_attestation": _registered_path(
                "${task_output:architecture_attestation:0}"
            ),
            "parent_loss_attestation": _registered_path(
                "${task_output:parent_loss_attestation:0}"
            ),
        })
        return {
            **base,
            "artifact": _registered_reference("${prebuilt_parent_import}"),
            "architecture_attestation": _registered_reference(
                "${task_output:architecture_attestation:0}"
            ),
            "parent_loss_attestation": _registered_reference(
                "${task_output:parent_loss_attestation:0}"
            ),
            "parent_report_paths": _member_registry(
                prerequisites, bundle="parent_reports",
                logical="${parent_reports}", mode="path",
            ),
            "model_source_paths": _member_registry(
                prerequisites, bundle="parent_model_sources",
                logical="${parent_model_sources}", mode="path",
            ),
            "authority_files": authority_files,
            "qualifier_report_paths": {
                name: _registered_path(logical)
                for name, logical in PARENT_QUALIFIER_REPORT_ROUTES.items()
            },
            "confirmation_report_paths": _member_registry(
                prerequisites, bundle="parent_confirmation_reports",
                logical="${parent_confirmation_reports}", mode="path",
            ),
        }
    if kind == "representation_recipe":
        return {
            **base,
            "artifact": _registered_reference("${prebuilt_representation_recipe}"),
            "producer_source_sha256": require_sha256(
                prerequisites["runtime_facts"]["source_snapshot_sha256"],
                name="representation recipe producer source snapshot",
            ),
            "representation_graph": _registered_reference(
                "${representation_graph}"
            ),
            "control_registry": _registered_reference("${control_registry}"),
            "parent_import": _registered_reference("${parent_import}"),
        }
    if kind == "numerical_acceptance":
        return {
            **base,
            "representation_recipe": _registered_reference(
                "${task_output:representation_recipe:0}"
            ),
        }
    if kind == "kernel_resources":
        logical = task.registered_outputs[0]
        root = _output_values(
            spec=spec, prerequisites=prerequisites, task=task, index=None,
        )[logical][IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
        return {
            **base, "root": root, "producer_task_id": task.task_key,
            "immutable_parent_hashes": dict(settings["kernel_parent_hashes"]),
            "registered_output_row": _registered_output_row(task, None, logical),
            "campaign_or_recovery_owner": _producer_owner(spec, task, None),
            "recipe": _registered_reference("${prebuilt_representation_recipe}"),
        }
    if kind == "cache_miniature_bank":
        build_task = by_key[f"miniature_{task.logical_bank}_build"]
        record = _target_record(prerequisites, build_task)
        return {
            **base,
            "bank_root": str(
                PurePosixPath(str(spec["campaign_root"]))
                / "targets" / str(task.logical_bank)
            ),
            "generation_id": str(record["generation_id"]),
            "bounded_row_limit": int(settings["miniature_row_limit"]),
            "cleanup_root": str(PurePosixPath(str(spec["campaign_root"])) / "cleanup"),
        }
    if kind == "reservation":
        final = _final_settings(prerequisites)
        return {
            **base,
            # These are frozen scientific inputs, not pre-built artifacts.
            # The worker rebuilds and publishes both commitments from these
            # values plus the authenticated matcher/parent-lock inputs.
            "assignment_source_partitions": list(final["source_partitions"]),
            "parent_finalists": list(final["finalists"]),
            "matcher_resources": _registered_json("${matcher_resources}"),
            "parent_finalist_lock": _registered_member(
                "${parent_finalist_registry}",
                str(final["parent_finalist_registry_relative"]), mode="json",
            ),
            "builder_arguments": {
            "checkpoint_namespace": _absolute(
                spec["checkpoint_namespace"], name="checkpoint namespace",
            ),
            "population": _registered_json("${shared_final_population}"),
            "campaign_spec_sha256": _registered_field(
                "${campaign_spec}", "content_hash",
            ),
            "final_disposition": _registered_json("${final_disposition}"),
            "parent_final_state": _registered_json("${parent_final_state}"),
            "selection_rule_sha256": require_sha256(
                final["selection_rule_sha256"], name="selection rule",
            ),
            "assignment_spec_sha256": require_sha256(
                final["assignment_spec_sha256"], name="assignment spec",
            ),
            "finalist_registry_commitment_sha256": require_sha256(
                final["finalist_registry_commitment_sha256"],
                name="finalist registry commitment",
            ),
            "registration_owner_id": None,
            "legacy_jobs_present": bool(final["legacy_jobs_present"]),
            },
        }
    if kind == "shuffle_map":
        logical = task.registered_outputs[0]
        root = _output_values(
            spec=spec, prerequisites=prerequisites, task=task, index=None,
        )[logical][IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
        return {
            **base,
            "data_root": _absolute(
                prerequisites["runtime_facts"]["data_root"], name="data root",
            ),
            "parent_hashes": dict(settings["shuffle_parent_hashes"]),
            "output_root": root,
            "registered_output_row": _registered_output_row(task, None, logical),
            "owner": _producer_owner(spec, task, None),
        }
    if kind == "target_cleanup":
        build_key = f"target_{task.logical_bank}_{task.target_purpose}"
        build = by_key[build_key]
        logical = f"${{task_output:{build_key}:0}}"
        reports: dict[str, Any] = {}
        for consumer_key in task.dependencies:
            consumer = by_key[consumer_key]
            reports.update({
                execution: _registered_json(route)
                for execution, route in _training_report_routes(consumer).items()
            })
        return {
            **base,
            "committed_directory": _registered_path(logical),
            "cleanup_root": str(PurePosixPath(str(spec["campaign_root"])) / "cleanup"),
            "consumer_report_paths": reports,
            "exact_reconstruction_authorized": True,
        }
    raise KeyError(f"no simple parameter builder for {task.task_key} ({kind})")


def _reporting_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, by_key: Mapping[str, CampaignTask],
) -> dict[str, Any]:
    base = _parameters_base(task)
    kind = task.kind
    if kind == "screen_aggregate":
        primary = [
            _registered_json(f"${{task_output:train_{node}:0}}")
            for node in sorted(NODE_REGISTRY)
        ]
        controls = [
            _registered_json(f"${{task_output:control_{node}:0}}")
            for node in sorted(CONTROL_REGISTRY)
        ]
        return {
            **base,
            "parent_import": _registered_reference(
                "${task_output:parent_import:0}"
            ),
            "architecture_attestation": _registered_reference(
                "${task_output:architecture_attestation:0}"
            ),
            "builder_arguments": {
            "primary_reports": primary,
            "control_reports": controls,
            "parent_reports": _parent_engine_report_registry(
                prerequisites, logical="${parent_reports}",
            ),
            "graph_sha256": require_sha256(spec["graph_sha256"], name="graph"),
            "recipe_sha256": require_sha256(
                spec["representation_recipe_sha256"], name="recipe",
            ),
            "campaign_spec_sha256": _registered_field(
                "${campaign_spec}", "content_hash",
            ),
            "expected_primary_ids": sorted(NODE_REGISTRY),
            "expected_control_ids": sorted(CONTROL_REGISTRY),
        }}
    if kind == "confirmation_registry":
        target = next(
            value for key, value in prerequisites["target_generations"].items()
            if key == "TOFF:confirmation"
        )
        return {**base, "builder_arguments": {
            "screen_sha256": _registered_field(
                "${task_output:screen_aggregate:0}", "content_hash",
            ),
            "campaign_sha256": require_sha256(
                target["execution_campaign_sha256"],
                name="confirmation execution campaign",
            ),
            "recipe_sha256": require_sha256(
                spec["representation_recipe_sha256"], name="recipe",
            ),
            "target_logical_bank_sha256": _content_hashes(prerequisites)[
                "${logical_bank:TOFF}"
            ],
            "objectives": ["RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"],
            "seeds": list(CONFIRMATION_SEEDS),
        }}
    if kind == "confirmation_aggregate":
        reports = []
        for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
            reports.extend(
                _registered_json(
                    f"${{task_output:confirm_{node}[{seed_index}]:4}}"
                )
                for seed_index in range(len(CONFIRMATION_SEEDS))
            )
        return {**base, "builder_arguments": {
            "registry": _registered_json("${task_output:confirmation_registry:0}"),
            "reports": reports,
        }}
    if kind == "validation_only_aggregate":
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[kind],
            "screen_aggregate": _registered_reference(
                "${task_output:screen_aggregate:0}"
            ),
            "confirmation_aggregate": _registered_reference(
                "${task_output:confirmation_aggregate:0}"
            ),
            "final_disposition": _registered_reference("${final_disposition}"),
        }}
    raise KeyError(f"no reporting parameter builder for {task.task_key}")


def _final_resource_signature(
    spec: Mapping[str, Any], campaign_task_key: str,
) -> dict[str, Any]:
    tasks = {str(row["task_key"]): row for row in spec["tasks"]}
    task = tasks[campaign_task_key]
    resource_class = str(task["resource_class"])
    request = spec["resources"][resource_class]
    return {
        "campaign_task_key": campaign_task_key,
        "resource_class": resource_class,
        "request_sha256": canonical_sha256(request),
        "campaign_identity_sha256": canonical_sha256(
            runtime_campaign_identity(spec),
        ),
    }


def _final_task_registry_rows(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
) -> list[dict[str, Any]]:
    final = _final_settings(prerequisites)
    partitions = [str(value) for value in final["source_partitions"]]
    finalists = _finalists(prerequisites, spec=spec)
    ids = _final_task_ids(prerequisites, spec=spec)
    rows: list[dict[str, Any]] = []

    def add(
        task_id: str, kind: str, *, outputs: Sequence[str],
        dependencies: Sequence[str] = (), branch: str | None = None,
        source: str | None = None, finalist: str | None = None,
        checkpoint: object | None = None, campaign_task: str,
    ) -> None:
        rows.append({
            "task_id": task_id, "kind": kind, "purpose": kind,
            "branch_family": branch, "source_partition": source,
            "finalist_id": finalist, "checkpoint_sha256": checkpoint,
            "registered_outputs": list(outputs),
            "dependencies": list(dependencies),
            "resource_signature": _final_resource_signature(
                spec, campaign_task,
            ),
        })

    add(
        ids["selection"], "row_selection", branch="selection",
        outputs=(
            "final/selection/row_selection.json",
            "final/selection/branch_access.json",
            "final/selection/label_escrow",
            "final/capabilities/row_selection.json",
        ), campaign_task="final_selection",
    )
    for source in partitions:
        task_id = ids["assignment_shards"][source]
        add(
            task_id, "assignment_shard", branch="assignment", source=source,
            dependencies=(ids["selection"],),
            outputs=(
                f"final/assignment/shards/{source}",
                f"final/capabilities/{task_id}.json",
            ), campaign_task="final_assignment_shards",
        )
    add(
        ids["assignment_finalize"], "assignment_finalize",
        dependencies=tuple(ids["assignment_shards"].values()),
        outputs=("final/assignment/manifest.json", "final/assignment/audit.json"),
        campaign_task="final_assignment_manifest",
    )
    add(
        ids["data_attestation"], "data_attestation",
        dependencies=(ids["assignment_finalize"],),
        outputs=("locks/06_final_data_attestation.json",),
        campaign_task="final_data_attestation",
    )
    add(
        ids["execution_lock"], "execution_lock",
        dependencies=(ids["data_attestation"],),
        outputs=("locks/07_execution.json", "final/prediction_spec.json"),
        campaign_task="representation_execution_lock",
    )
    for finalist_row in finalists:
        finalist = str(finalist_row["finalist_id"])
        branch = {
            "hlt": "hlt", "shell_exact_d100": "shell_exact",
            "native_offline": "native_offline",
        }[str(finalist_row["domain"])]
        for source in partitions:
            task_id = ids["prediction_shards"][(finalist, source)]
            add(
                task_id, "prediction_shard", branch=branch, source=source,
                finalist=finalist,
                checkpoint=finalist_row["checkpoint_sha256"],
                dependencies=(ids["execution_lock"],),
                outputs=(
                    f"final/predictions/{finalist}/shards/{source}",
                    f"final/capabilities/{task_id}.json",
                ), campaign_task="final_prediction_shards",
            )
        manifest_id = ids["prediction_manifests"][finalist]
        add(
            manifest_id, "prediction_finalize", finalist=finalist,
            dependencies=tuple(
                ids["prediction_shards"][(finalist, source)]
                for source in partitions
            ),
            outputs=(f"final/predictions/{finalist}/manifest.json",),
            campaign_task="final_prediction_manifests",
        )
    add(
        ids["metric_join"], "metric_join", branch="label_escrow",
        dependencies=tuple(ids["prediction_manifests"].values()),
        outputs=(
            "final/metric_join.json", "final/evaluations",
            "reports/paired_bootstrap", "final/capabilities/metric_join.json",
        ), campaign_task="locked_metric_join",
    )
    add(
        ids["final_aggregate"], "final_aggregate",
        dependencies=(ids["metric_join"],),
        outputs=("reports/final_aggregate.json",),
        campaign_task="final_aggregate",
    )
    prerequisite_hashes = _content_hashes(prerequisites)
    for row in rows:
        if row["kind"] == "assignment_shard":
            row["resource_signature"].update({
                "split_manifest_sha256": prerequisite_hashes["${split_manifest}"],
                "matcher_resources_sha256": prerequisite_hashes[
                    "${matcher_resources}"
                ],
            })
    return rows


def _representation_endpoint_rows(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    by_key = {task.task_key: task for task in _tasks(spec)}
    for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
        task = by_key[f"train_{node}"]
        execution = _registered_execution_id(
            spec=spec, prerequisites=prerequisites, task=task, index=None,
        )
        report = f"${{task_output:{task.task_key}:0}}"
        selection = f"${{task_output:{task.task_key}:1}}"
        extraction = f"${{task_output:{task.task_key}:2}}"
        rows.append({
            "finalist_id": node,
            "checkpoint_sha256": _registered_field(
                extraction, "payload", "deployable_state_sha256",
            ),
            "report_sha256": _registered_field(report, "content_hash"),
            "domain": "hlt", "deployable": True,
            "extraction_sha256": _registered_field(extraction, "content_hash"),
            "source_campaign": "representation", "screening_seed": 1337,
            "execution_id": execution,
            "checkpoint_selection_sha256": _registered_field(
                selection, "content_hash",
            ),
        })
    return rows


def _finalist_lock_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any], task: CampaignTask,
) -> dict[str, Any]:
    final = _final_settings(prerequisites)
    legacy = (
        _registered_field("${legacy_cancellation}", "content_hash")
        if bool(final["legacy_jobs_present"]) else None
    )
    return {**_parameters_base(task), "builder_arguments": {
        "parent_finalists": list(final["finalists"]),
        "parent_finalist_lock": _registered_member(
            "${parent_finalist_registry}",
            str(final["parent_finalist_registry_relative"]), mode="json",
        ),
        "representation_endpoints": _representation_endpoint_rows(
            spec=spec, prerequisites=prerequisites,
        ),
        "parent_campaign_sha256": require_sha256(
            final["parent_campaign_sha256"], name="parent campaign",
        ),
        "representation_campaign_sha256": _registered_field(
            "${campaign_spec}", "content_hash",
        ),
        "reservation": _registered_json("${task_output:pretraining_reservation:0}"),
        "parent_recipe_sha256": _content_hashes(prerequisites)["${parent_recipe}"],
        "representation_recipe_sha256": require_sha256(
            spec["representation_recipe_sha256"], name="representation recipe",
        ),
        "loss_attestation_sha256": _registered_field(
            "${task_output:parent_loss_attestation:0}", "content_hash",
        ),
        "architecture_attestation_sha256": _registered_field(
            "${task_output:architecture_attestation:0}", "content_hash",
        ),
        "selection_rule_sha256": require_sha256(
            final["selection_rule_sha256"], name="selection rule",
        ),
        "assignment_spec_sha256": require_sha256(
            final["assignment_spec_sha256"], name="assignment spec",
        ),
        "legacy_cancellation_sha256": legacy,
        "parent_finalist_registry_sha256": _registered_field(
            "${parent_finalist_registry}", "content_hash",
            relative=str(final["parent_finalist_registry_relative"]),
        ),
        "screen_aggregate": _registered_json("${task_output:screen_aggregate:0}"),
        "confirmation_registry": _registered_json(
            "${task_output:confirmation_registry:0}"
        ),
        "confirmation_aggregate": _registered_json(
            "${task_output:confirmation_aggregate:0}"
        ),
    }}


def _shared_claim_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any], task: CampaignTask,
) -> dict[str, Any]:
    final = _final_settings(prerequisites)
    return {
        **_parameters_base(task),
        "task_registry_arguments": {
            "population_sha256": require_sha256(
                final["population_sha256"], name="final population",
            ),
            "campaign_spec_sha256": _registered_field(
                "${campaign_spec}", "content_hash",
            ),
            "tasks": _final_task_registry_rows(
                spec=spec, prerequisites=prerequisites,
            ),
            "finalist_lock_sha256": _registered_field(
                "${task_output:finalist_lock:0}", "content_hash",
            ),
            "submission_ledger_sha256": _registered_field(
                "${submission_ledger}", "content_hash",
            ),
        },
        "claim_arguments": {
            "checkpoint_namespace": _absolute(
                spec["checkpoint_namespace"], name="checkpoint namespace",
            ),
            "reservation": _registered_json(
                "${task_output:pretraining_reservation:0}"
            ),
            "finalist_lock_sha256": _registered_field(
                "${task_output:finalist_lock:0}", "content_hash",
            ),
            "source_commit": str(spec["source_commit"]),
            "claim_owner_id": None,
        },
    }


def _finalist_model_source(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["source_campaign"] == "representation":
        node = str(row["finalist_id"])
        return {
            "kind": "hcwdl",
            "execution_directory": _registered_path(
                f"${{task_output:train_{node}:3}}"
            ),
        }
    relative = str(row["model_relative"])
    if row["model_kind"] == "pmard":
        return {
            "kind": "pmard",
            "report": _registered_member(
                "${finalist_models}", relative, mode="reference",
            ),
        }
    return {
        "kind": "hcwdl",
        "execution_directory": _registered_member(
            "${finalist_models}", relative, mode="path",
        ),
    }


def _final_parameters(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None,
) -> dict[str, Any]:
    base = _parameters_base(task)
    final = _final_settings(prerequisites)
    ids = _final_task_ids(prerequisites, spec=spec)
    partitions = [str(value) for value in final["source_partitions"]]
    finalists = _finalists(prerequisites, spec=spec)
    data_root = _absolute(
        prerequisites["runtime_facts"]["data_root"], name="data root",
    )
    claim = _registered_reference("${task_output:shared_claim_gate:0}")
    registry = _registered_reference("${task_output:shared_claim_gate:1}")
    selection = _registered_reference("${task_output:final_selection:0}")
    execution_lock = _registered_reference(
        "${task_output:representation_execution_lock:0}"
    )
    prediction_spec = _registered_reference(
        "${task_output:representation_execution_lock:1}"
    )
    finalist_lock = _registered_reference("${task_output:finalist_lock:0}")
    owner = _producer_owner(spec, task, index)

    if task.kind == "final_selection":
        envelope_logical = task.registered_outputs[2]
        envelope_root = _output_values(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )[envelope_logical][IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "split_manifest": _registered_reference("${split_manifest}"),
            "population": _registered_reference("${shared_final_population}"),
            "claim": claim, "task_registry": registry,
            "task_id": ids["selection"], "data_root": data_root,
            "rows_per_class": list(final["rows_per_class"]),
            "selection_rule_sha256": require_sha256(
                final["selection_rule_sha256"], name="selection rule",
            ),
            "step_size": int(final["step_size"]),
            "escrow_root": envelope_root,
            "producer_task_id": ids["selection"],
            "registered_output_row": _registered_output_row(
                task, index, envelope_logical,
            ),
            "publication_owner": owner,
        }}
    if task.kind == "assignment_shard":
        assert index is not None
        source = partitions[index]
        task_id = ids["assignment_shards"][source]
        envelope_logical = task.registered_outputs[0]
        envelope_root = _output_values(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )[envelope_logical][IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "split_manifest": _registered_reference("${split_manifest}"),
            "selection": selection,
            "matcher_resources": _registered_reference("${matcher_resources}"),
            "claim": claim, "task_registry": registry, "task_id": task_id,
            "data_root": data_root, "source_path": source,
            "step_size": int(final["step_size"]),
            "assignment_spec_sha256": require_sha256(
                final["assignment_spec_sha256"], name="assignment spec",
            ),
            "envelope_root": envelope_root, "producer_task_id": task_id,
            "registered_output_row": _registered_output_row(
                task, index, envelope_logical,
            ),
            "publication_owner": owner,
        }}
    if task.kind == "assignment_finalize":
        shards = [
            {"committed_directory": _registered_path(
                f"${{task_output:final_assignment_shards[{position}]:0}}"
            )}
            for position in range(len(partitions))
        ]
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "split_manifest": _registered_reference("${split_manifest}"),
            "selection": selection,
            "matcher_resources": _registered_reference("${matcher_resources}"),
            "claim": claim, "task_registry": registry,
            "assignment_spec_sha256": require_sha256(
                final["assignment_spec_sha256"], name="assignment spec",
            ),
            "shards": shards,
            "expected_mapped_jets": sum(int(value) for value in final["rows_per_class"]),
            "require_sub10pct_dustbins": True,
        }}
    if task.kind == "data_attestation":
        return {**base, "builder_arguments": {
            "selection": _registered_json("${task_output:final_selection:0}"),
            "assignment_audit": _registered_json(
                "${task_output:final_assignment_manifest:1}"
            ),
            "assignment_manifest": _registered_json(
                "${task_output:final_assignment_manifest:0}"
            ),
            "assignment_spec": _registered_json(
                "${task_output:pretraining_reservation:1}"
            ),
            "matcher_resources": _registered_json("${matcher_resources}"),
            "label_escrow": _registered_member(
                "${task_output:final_selection:2}", "sidecar.json", mode="json",
            ),
            "task_registry": _registered_json("${task_output:shared_claim_gate:1}"),
            "claim": _registered_json("${task_output:shared_claim_gate:0}"),
        }}
    if task.kind == "execution_lock":
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "finalist_lock": finalist_lock,
            "data_attestation": _registered_reference(
                "${task_output:final_data_attestation:0}"
            ),
            "claim": claim, "task_registry": registry,
            "row_selection": selection,
            "prediction_runtime_signature": _registered_json(
                "${prediction_runtime_signature}"
            ),
            "source_partitions": partitions,
        }}
    if task.kind == "prediction_shard":
        assert index is not None
        partition_index = index % len(partitions)
        finalist_index = index // len(partitions)
        source = partitions[partition_index]
        finalist_row = finalists[finalist_index]
        finalist = str(finalist_row["finalist_id"])
        task_id = ids["prediction_shards"][(finalist, source)]
        envelope_logical = task.registered_outputs[0]
        envelope_root = _output_values(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )[envelope_logical][IMMUTABLE_OUTPUT_ROOT_BINDING]["root"]
        assignment = (
            _registered_reference("${task_output:final_assignment_manifest:0}")
            if finalist_row["domain"] == "shell_exact_d100" else None
        )
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "split_manifest": _registered_reference("${split_manifest}"),
            "selection": selection, "prediction_spec": prediction_spec,
            "finalist_lock": finalist_lock, "execution_lock": execution_lock,
            "claim": claim, "task_registry": registry, "task_id": task_id,
            "data_root": data_root, "source_partition": source,
            "finalist_id": finalist,
            "model_source": _finalist_model_source(finalist_row),
            "assignment_manifest": assignment,
            "step_size": int(final["step_size"]),
            "producer_runtime_signature": _registered_json(
                "${prediction_runtime_signature}"
            ),
            "envelope_root": envelope_root, "producer_task_id": task_id,
            "registered_output_row": _registered_output_row(
                task, index, envelope_logical,
            ),
            "publication_owner": owner,
        }}
    if task.kind == "prediction_finalize":
        assert index is not None
        finalist_row = finalists[index]
        finalist = str(finalist_row["finalist_id"])
        shards = []
        for partition_index, _source in enumerate(partitions):
            row = index * len(partitions) + partition_index
            shards.append({"committed_directory": _registered_path(
                f"${{task_output:final_prediction_shards[{row}]:0}}"
            )})
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind], "selection": selection,
            "prediction_spec": prediction_spec, "execution_lock": execution_lock,
            "finalist": _finalist_payload(finalist_row), "shards": shards,
            "expected_source_partitions": partitions,
        }}
    if task.kind == "metric_join":
        manifest_refs = {
            str(row["finalist_id"]): _registered_reference(
                f"${{task_output:final_prediction_manifests[{position}]:0}}"
            ) for position, row in enumerate(finalists)
        }
        shard_refs: dict[str, list[Any]] = {}
        for finalist_index, row in enumerate(finalists):
            finalist = str(row["finalist_id"])
            shard_refs[finalist] = [
                {"committed_directory": _registered_path(
                    "${task_output:final_prediction_shards["
                    + str(finalist_index * len(partitions) + partition_index)
                    + "]:0}"
                )}
                for partition_index in range(len(partitions))
            ]
        evaluation_root = str(
            PurePosixPath(str(spec["campaign_root"])) / "final" / "evaluations"
        )
        comparisons = list(final["comparison_registry"])
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind], "selection": selection,
            "label_escrow": {"committed_directory": _registered_path(
                "${task_output:final_selection:2}"
            )},
            "prediction_spec": prediction_spec, "execution_lock": execution_lock,
            "finalist_lock": finalist_lock,
            "data_attestation": _registered_reference(
                "${task_output:final_data_attestation:0}"
            ),
            "claim": claim, "task_registry": registry,
            "task_id": ids["metric_join"],
            "prediction_manifests": manifest_refs,
            "prediction_shards": shard_refs,
            "evaluation_output_paths": {
                str(row["finalist_id"]): f"{evaluation_root}/{row['finalist_id']}.json"
                for row in finalists
            },
            "comparison_registry": comparisons,
            "bootstrap_root": str(
                PurePosixPath(str(spec["campaign_root"]))
                / "reports" / "paired_bootstrap"
            ),
            "bootstrap_output_rows": {
                str(row["comparison_id"]): {
                    "comparison_id": str(row["comparison_id"]),
                    "task_id": ids["metric_join"],
                } for row in comparisons
            },
            "publication_owner": owner,
            "producer_task_id_prefix": ids["metric_join"],
        }}
    if task.kind == "final_aggregate":
        # The exact bootstrap-root consumer is intentionally handled by the
        # production adapter: envelope IDs depend on the future metric-join
        # content hash, so pre-campaign code may not invent them.
        return {**base, "assembly": {
            "contract": _ASSEMBLY_CONTRACT[task.kind],
            "metric_join": _registered_reference(
                "${task_output:locked_metric_join:0}"
            ),
            "evaluations": {
                str(row["finalist_id"]): _registered_member(
                    "${task_output:locked_metric_join:1}",
                    f"{row['finalist_id']}.json", mode="reference",
                ) for row in finalists
            },
            "finalist_lock": finalist_lock, "execution_lock": execution_lock,
            "confirmation_aggregate": _registered_reference(
                "${task_output:confirmation_aggregate:0}"
            ),
            "comparison_registry": list(final["comparison_registry"]),
            "bootstrap_root": _registered_path(
                "${task_output:locked_metric_join:2}"
            ),
        }}
    raise KeyError(f"no final parameter builder for {task.task_key}")


def _parameters_for(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task: CampaignTask, index: int | None, by_key: Mapping[str, CampaignTask],
) -> dict[str, Any]:
    if task.kind == "target_build":
        return _target_parameters(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )
    if task.kind in {"train_node", "train_control", "confirmation"}:
        return _training_parameters(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )
    if task.kind in {
        "screen_aggregate", "confirmation_registry", "confirmation_aggregate",
        "validation_only_aggregate",
    }:
        return _reporting_parameters(
            spec=spec, prerequisites=prerequisites, task=task, by_key=by_key,
        )
    if task.kind == "finalist_lock":
        return _finalist_lock_parameters(
            spec=spec, prerequisites=prerequisites, task=task,
        )
    if task.kind == "shared_final_claim":
        return _shared_claim_parameters(
            spec=spec, prerequisites=prerequisites, task=task,
        )
    if task.kind in {
        "final_selection", "assignment_shard", "assignment_finalize",
        "data_attestation", "execution_lock", "prediction_shard",
        "prediction_finalize", "metric_join", "final_aggregate",
    }:
        return _final_parameters(
            spec=spec, prerequisites=prerequisites, task=task, index=index,
        )
    return _simple_parameters(
        spec=spec, prerequisites=prerequisites, task=task, index=index,
        by_key=by_key,
    )


def _tag_inputs(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for tag in (
            "registered_json", "registered_reference", "registered_path",
        ):
            if tag in value:
                found.add(str(value[tag]))
        if "registered_member" in value:
            payload = value["registered_member"]
            if isinstance(payload, Mapping):
                found.add(str(payload.get("input")))
        if "registered_field" in value:
            payload = value["registered_field"]
            if isinstance(payload, Mapping):
                found.add(str(payload.get("input")))
        for item in value.values():
            found.update(_tag_inputs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_tag_inputs(item))
    return found


def _reject_raw_artifacts(value: object, *, location: str = "parameters") -> None:
    if isinstance(value, Mapping):
        keys = set(value)
        if {"path", "sha256"} <= keys:
            raise PermissionError(
                f"runtime parameter contains a raw artifact reference: {location}"
            )
        if "content_hash" in keys:
            raise PermissionError(
                f"runtime parameter contains an inline artifact: {location}"
            )
        for key, item in value.items():
            _reject_raw_artifacts(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _reject_raw_artifacts(item, location=f"{location}[{position}]")


def _schema_kind(kind: str) -> str:
    return "train_node" if kind in {"train_node", "train_control", "confirmation"} else kind


def _validate_parameter_schema(task: CampaignTask, parameters: Mapping[str, Any]) -> None:
    kind = _schema_kind(task.kind)
    base = {"adapter_contract", "task_kind"}
    if kind in _ASSEMBLY_FIELDS:
        if set(parameters) != base | {"assembly"}:
            raise ValueError(f"adapter outer parameter schema differs: {task.task_key}")
        assembly = parameters["assembly"]
        if not isinstance(assembly, Mapping) or set(assembly) != {
            "contract", *_ASSEMBLY_FIELDS[kind],
        } or assembly.get("contract") != _ASSEMBLY_CONTRACT[kind]:
            raise ValueError(f"adapter assembly schema differs: {task.task_key}")
    else:
        required = _OUTER_REQUIRED.get(kind)
        if required is None or set(parameters) != base | set(required):
            raise ValueError(f"adapter parameter schema differs: {task.task_key}")
    if parameters.get("adapter_contract") != PRODUCTION_ADAPTER_CONTRACT:
        raise ValueError("production adapter contract differs")
    if parameters.get("task_kind") != task.kind:
        raise ValueError("production adapter task kind differs")
    tags = _tag_inputs(parameters)
    missing = tags - set(task.registered_inputs)
    if missing:
        raise PermissionError(
            f"parameter tags name unregistered inputs for {task.task_key}: {sorted(missing)}"
        )
    _reject_raw_artifacts(parameters)


def _validate_prerequisites(
    *, spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    tasks: Sequence[CampaignTask], by_key: Mapping[str, CampaignTask],
) -> None:
    required = {
        "contract", "schema_version", "runtime_facts", "runtime_signatures",
        "static_inputs", "artifact_content_hashes", "target_generations",
        "bundle_members", "settings", "content_hash",
    }
    if set(prerequisites) != required or prerequisites.get(
        "contract"
    ) != RUNTIME_PREREQUISITES_CONTRACT or prerequisites.get("schema_version") != 1:
        raise ValueError("HCWDL-RKD runtime prerequisite schema differs")
    facts = prerequisites["runtime_facts"]
    expected_fact_keys = {
        "conda_environment", "data_root", "device", "project_dir",
        "python_no_user_site", "source_snapshot_sha256",
        "weaver_runtime_sha256",
    }
    if not isinstance(facts, Mapping) or set(facts) != expected_fact_keys:
        raise ValueError("runtime fact registry differs")
    if facts["project_dir"] != spec["project_dir"]:
        raise ValueError("runtime prerequisite project directory differs")
    signatures = prerequisites["runtime_signatures"]
    resource_classes = {task.resource_class for task in tasks}
    if not isinstance(signatures, Mapping) or set(signatures) != resource_classes:
        raise ValueError("runtime signature resource registry differs")
    for name, digest in signatures.items():
        require_sha256(digest, name=f"runtime signature {name}")
    static = prerequisites["static_inputs"]
    if not isinstance(static, Mapping):
        raise ValueError("static input registry differs")
    expected_static = set()
    for task in tasks:
        for logical in task.registered_inputs:
            if logical in {"${campaign_spec}", "${submission_ledger}"}:
                continue
            if _alias_route(logical, task=task, by_key=by_key) is None:
                expected_static.add(logical)
    if set(static) != expected_static:
        raise ValueError(
            "exogenous static input registry differs; "
            f"missing={sorted(expected_static - set(static))}, "
            f"extra={sorted(set(static) - expected_static)}"
        )
    for logical, reference in static.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"static input reference differs: {logical}")
        _absolute(reference["path"], name=f"{logical} path")
        require_sha256(reference["sha256"], name=f"{logical} byte hash")
    _content_hashes(prerequisites)
    parent_reports = _bundle_members(prerequisites, "parent_reports")
    parent_models = _bundle_members(prerequisites, "parent_model_sources")
    parent_confirmations = _bundle_members(
        prerequisites, "parent_confirmation_reports",
    )
    expected_parent_nodes = set(IMPORTED_TEACHERS) | set(IMPORTED_LOGIT_CONTROLS)
    if not isinstance(parent_reports, Mapping) or set(parent_reports) != expected_parent_nodes:
        raise ValueError("parent report bundle is incomplete or expanded")
    if not isinstance(parent_models, Mapping) or set(parent_models) != {
        "D0w", "hcwdl_surfaces", "scouting_particle_transformer",
    }:
        raise ValueError("parent model-source bundle is incomplete or expanded")
    expected_confirmation_keys = _final_settings(prerequisites).get(
        "parent_confirmation_report_keys"
    )
    if (
        not isinstance(parent_confirmations, Mapping)
        or not isinstance(expected_confirmation_keys, list)
        or not expected_confirmation_keys
        or expected_confirmation_keys != sorted(expected_confirmation_keys)
        or set(parent_confirmations) != set(expected_confirmation_keys)
    ):
        raise ValueError("parent confirmation-report bundle is incomplete or expanded")
    d0w_source = parent_models["D0w"]
    if (
        not isinstance(d0w_source, Mapping)
        or set(d0w_source) != {"relative", "kind"}
        or d0w_source.get("kind") != "pmard"
    ):
        raise ValueError("parent D0w model-source bundle member differs")
    settings = _settings(prerequisites)
    final = _final_settings(prerequisites)
    if set(settings) != _SETTINGS_KEYS or set(final) != _FINAL_SETTINGS_KEYS:
        raise ValueError("runtime settings schema differs")
    if (
        settings["kernel_parent_hashes"] != _expected_kernel_parents(spec)
        or settings["shuffle_parent_hashes"]
        != _expected_shuffle_parents(spec, prerequisites)
    ):
        raise ValueError("kernel/shuffle parent lineage differs")
    budgets = settings["target_budgets"]
    budget_keys = {
        "target_storage_cap_bytes", "container_overhead_bytes",
        "staging_recovery_reserve_bytes", "quarantine_reserve_bytes",
        "filesystem_headroom_bytes", "peak_runtime_bytes",
        "slurm_mem_per_node_bytes", "filesystem_available_bytes",
    }
    if (
        not isinstance(budgets, Mapping) or set(budgets) != budget_keys
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in budgets.values())
        or budgets["slurm_mem_per_node_bytes"]
        != int(str(spec["resources"]["gpu_target"]["memory"])[:-1]) * 1024**3
    ):
        raise ValueError("target budget/resource lineage differs")
    expected_mode = "smoke" if spec.get("mode") == "smoke" else "scientific"
    expected_passes = 1 if expected_mode == "smoke" else 60
    if (
        settings.get("training_mode") != expected_mode
        or settings.get("synthetic_passes") != expected_passes
        or settings.get("miniature_row_limit") != 4096
        or float(settings.get("view_cache_max_gib", -1.0))
        != _expected_view_cache_gib(spec)
    ):
        raise PermissionError("runtime settings differ from frozen campaign/resource policy")
    finalists = _finalists(prerequisites, spec=spec)
    finalist_ids = {str(row["finalist_id"]) for row in finalists}
    required_finalists = {
        "M0", "D100", "TOFF", "M6c", "M6w",
        "RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w",
    }
    if (
        len(finalists) != int(spec["combined_finalist_count"])
        or not required_finalists <= finalist_ids
    ):
        raise ValueError("combined finalist registry lacks frozen logit/representation endpoints")
    partitions = final.get("source_partitions")
    if (
        not isinstance(partitions, list) or not partitions
        or partitions != sorted(partitions)
        or len(partitions) != len(set(partitions))
        or len(partitions) != int(spec["final_source_partitions"])
    ):
        raise ValueError("final source-partition registry differs")
    from .hcwdl_representation_final_policy import (
        build_final_assignment_spec,
        build_pretraining_finalist_policy_commitment,
    )
    from .highcov_resources import resource_validation_report
    expected_assignment = build_final_assignment_spec(
        matcher_resources=resource_validation_report(),
        source_partitions=partitions, step_size=8192,
    )["content_hash"]
    expected_finalists = build_pretraining_finalist_policy_commitment(
        parent_finalists=final["finalists"],
        parent_finalist_lock=final["parent_finalist_lock"],
    )["content_hash"]
    if (
        final.get("assignment_spec_sha256") != expected_assignment
        or final.get("finalist_registry_commitment_sha256") != expected_finalists
    ):
        raise PermissionError("final assignment/finalist policy commitment differs")
    if final.get("step_size") != 8192:
        raise ValueError("final streaming step size differs from frozen parent selection")
    quotas = final.get("rows_per_class")
    if (
        not isinstance(quotas, list) or len(quotas) != 15
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in quotas)
        or sum(quotas) != int(spec["role_counts"]["final_test"])
    ):
        raise ValueError("final selection quotas do not cover the frozen final-test budget")
    expected_selection = canonical_sha256({
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "role": "final_test", "rows": int(spec["role_counts"]["final_test"]),
        "rule": "per_class_smallest_identity_sha256_rank_v1", "seed": 1337,
    })
    if final.get("selection_rule_sha256") != expected_selection:
        raise ValueError("final selection-rule commitment differs")
    if final.get("comparison_registry") != _required_final_comparisons():
        raise ValueError("final paired-comparison registry is incomplete or reordered")
    validate_content_hash(
        prerequisites, expected_contract=RUNTIME_PREREQUISITES_CONTRACT,
        expected_schema_version=1,
    )


def build_runtime_prerequisites(
    spec: Mapping[str, Any], *, runtime_facts: Mapping[str, Any],
    runtime_signatures: Mapping[str, Any], static_inputs: Mapping[str, Any],
    artifact_content_hashes: Mapping[str, Any],
    target_generations: Mapping[str, Any], bundle_members: Mapping[str, Any],
    settings: Mapping[str, Any], split_manifest: Mapping[str, Any],
    final_authorities: Mapping[str, Any],
    target_forward_specs: Mapping[str, Any],
    representation_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble and validate the sole explicit operator-input document.

    Each argument is a closed immutable registry with a distinct operational
    role.  The constructor prevents callers and runbooks from manually
    assembling the versioned prerequisite envelope or silently omitting a
    registry before exhaustive row construction.
    """

    if set(settings) != _SETTINGS_KEYS:
        raise ValueError("operator runtime settings schema differs")
    from .hcwdl_representation_recipe import validate_representation_recipe

    recipe_sha256 = validate_representation_recipe(representation_recipe)
    if recipe_sha256 != require_sha256(
        spec["representation_recipe_sha256"], name="campaign representation recipe",
    ):
        raise ValueError("runtime representation recipe differs from campaign identity")
    runtime_source_sha256 = require_sha256(
        runtime_facts.get("source_snapshot_sha256"),
        name="runtime source snapshot",
    )
    if representation_recipe["parents"].get(
        "producer_source"
    ) != runtime_source_sha256:
        raise PermissionError(
            "representation recipe producer source differs from measured runtime source"
        )
    quotas, selection_rule = _final_selection_policy(spec, split_manifest)
    normalized_settings = dict(settings)
    expected_target_runtime = _derived_target_runtime_environment(
        target_forward_specs,
    )
    if settings.get("target_runtime_environment") not in (
        None, expected_target_runtime,
    ):
        raise PermissionError(
            "operator target runtime environment differs from target-forward authority"
        )
    normalized_settings["target_runtime_environment"] = expected_target_runtime
    expected_parent_registries = {
        "kernel_parent_hashes": _expected_kernel_parents(spec),
        "shuffle_parent_hashes": _expected_shuffle_parents(
            spec, {"artifact_content_hashes": artifact_content_hashes},
        ),
    }
    for name, expected in expected_parent_registries.items():
        if settings.get(name) not in (None, expected):
            raise PermissionError(
                f"operator runtime setting {name!r} differs from campaign lineage"
            )
        normalized_settings[name] = expected
    expected_mode = "smoke" if spec.get("mode") == "smoke" else "scientific"
    expected_passes = 1 if expected_mode == "smoke" else 60
    frozen_outer = {
        "training_mode": expected_mode,
        "synthetic_passes": expected_passes,
        "miniature_row_limit": 4096,
        "view_cache_max_gib": _expected_view_cache_gib(spec),
    }
    for name, expected in frozen_outer.items():
        if settings.get(name) not in (None, expected):
            raise PermissionError(f"operator runtime setting {name!r} differs from frozen policy")
        normalized_settings[name] = expected
    raw_final = normalized_settings.get("final")
    if not isinstance(raw_final, Mapping):
        raise ValueError("runtime prerequisite settings lack final registry")
    if set(raw_final) != _FINAL_SETTINGS_KEYS:
        raise ValueError("operator final runtime settings schema differs")
    derived_final = _derived_final_authority(
        spec=spec, runtime_facts=runtime_facts,
        artifact_content_hashes=artifact_content_hashes,
        split_manifest=split_manifest, final_authorities=final_authorities,
    )
    for name, expected in derived_final.items():
        if raw_final.get(name) not in (None, expected):
            raise PermissionError(
                f"operator final setting {name!r} differs from authenticated authority"
            )
    supplied_quotas = raw_final.get("rows_per_class")
    supplied_rule = raw_final.get("selection_rule_sha256")
    supplied_comparisons = raw_final.get("comparison_registry")
    comparisons = _required_final_comparisons()
    if (
        supplied_quotas not in (None, quotas)
        or supplied_rule not in (None, selection_rule)
        or supplied_comparisons not in (None, comparisons)
        or raw_final.get("step_size") not in (None, 8192)
    ):
        raise PermissionError(
            "operator final selection/comparison settings differ from frozen policy"
        )
    normalized_settings["final"] = {
        **raw_final, **derived_final,
        "rows_per_class": quotas,
        "selection_rule_sha256": selection_rule,
        "comparison_registry": comparisons,
        "step_size": 8192,
    }
    value = with_content_hash({
        "contract": RUNTIME_PREREQUISITES_CONTRACT,
        "schema_version": 1,
        "runtime_facts": dict(runtime_facts),
        "runtime_signatures": dict(runtime_signatures),
        "static_inputs": dict(static_inputs),
        "artifact_content_hashes": dict(artifact_content_hashes),
        "target_generations": dict(target_generations),
        "bundle_members": dict(bundle_members),
        "settings": normalized_settings,
    })
    tasks = _tasks(spec)
    by_key = {task.task_key: task for task in tasks}
    _validate_prerequisites(
        spec=spec, prerequisites=value, tasks=tasks, by_key=by_key,
    )
    return value


def build_runtime_task_rows(
    spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build every scalar/array row required by ``build_runtime_binding``."""

    tasks = _tasks(spec)
    by_key = {task.task_key: task for task in tasks}
    if len(by_key) != len(tasks):
        raise ValueError("campaign task keys repeat")
    _validate_prerequisites(
        spec=spec, prerequisites=prerequisites, tasks=tasks, by_key=by_key,
    )
    signatures = prerequisites["runtime_signatures"]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        rows: dict[str, dict[str, Any]] = {}
        for index in array_indices(task.array):
            parameters = _parameters_for(
                spec=spec, prerequisites=prerequisites, task=task,
                index=index, by_key=by_key,
            )
            _validate_parameter_schema(task, parameters)
            rows[_row_key(index)] = {
                "array_index": index,
                "device": (
                    "cuda" if task.resource_class.startswith("gpu_") else "cpu"
                ),
                "inputs": _runtime_inputs(
                    spec=spec, prerequisites=prerequisites, task=task,
                    by_key=by_key,
                ),
                "outputs": _output_values(
                    spec=spec, prerequisites=prerequisites, task=task, index=index,
                ),
                "parameters": parameters,
                "runtime_signature_sha256": require_sha256(
                    signatures[task.resource_class],
                    name=f"{task.resource_class} runtime signature",
                ),
            }
        result[task.task_key] = rows
    return result


def validate_runtime_task_rows(
    spec: Mapping[str, Any], prerequisites: Mapping[str, Any],
    task_rows: Mapping[str, Any],
) -> None:
    """Require byte-for-JSON equality with the deterministic all-row build."""

    expected = build_runtime_task_rows(spec, prerequisites)
    if dict(task_rows) != expected:
        raise ValueError("HCWDL-RKD runtime task rows differ from deterministic assembly")


def validate_bound_runtime_task_rows(
    spec: Mapping[str, Any], runtime_binding: Mapping[str, Any],
) -> None:
    """Validate closed adapter schemas in an already canonical runtime binding.

    Candidate review has the immutable binding, not the operator's transient
    prerequisite document.  This function therefore authenticates the binding
    first and then proves that every embedded row is consumable by exactly one
    fixed adapter schema and that every tagged artifact is a registered input.
    """

    from .hcwdl_representation_runtime_binding import validate_runtime_binding

    validate_runtime_binding(runtime_binding, spec=spec)
    tasks = {task.task_key: task for task in _tasks(spec)}
    rows = runtime_binding.get("tasks")
    if not isinstance(rows, list) or {str(row.get("task_key")) for row in rows} != set(tasks):
        raise ValueError("bound runtime task inventory differs")
    bound_by_key = {str(row["task_key"]): row for row in rows}
    for task_binding in rows:
        task = tasks[str(task_binding["task_key"])]
        expected_indices = set(array_indices(task.array))
        bound_rows = task_binding.get("rows")
        if not isinstance(bound_rows, list) or {
            row.get("array_index") for row in bound_rows
        } != expected_indices:
            raise ValueError(f"bound runtime row inventory differs: {task.task_key}")
        for row in bound_rows:
            parameters = row.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError("bound runtime parameters differ")
            _validate_parameter_schema(task, parameters)

    if "final_selection" not in bound_by_key:
        return
    selection_row = bound_by_key["final_selection"]["rows"][0]
    split_reference = selection_row["inputs"].get("${split_manifest}")
    if not isinstance(split_reference, Mapping) or set(split_reference) != {
        "path", "sha256",
    }:
        raise ValueError("bound final selection lacks literal split-manifest bytes")
    split_path = Path(str(split_reference["path"]))
    split_bytes = require_sha256(split_reference["sha256"], name="bound split bytes")
    if sha256_file(split_path) != split_bytes:
        raise ValueError("bound split-manifest bytes differ")
    quotas, selection_rule = _final_selection_policy(spec, load_json(split_path))
    selection_assembly = selection_row["parameters"]["assembly"]
    if (
        selection_assembly.get("rows_per_class") != quotas
        or selection_assembly.get("selection_rule_sha256") != selection_rule
        or selection_assembly.get("step_size") != 8192
    ):
        raise PermissionError("bound final-selection policy differs from authenticated split")

    comparisons = _required_final_comparisons()
    for task_key in ("locked_metric_join", "final_aggregate"):
        assembly = bound_by_key[task_key]["rows"][0]["parameters"]["assembly"]
        if assembly.get("comparison_registry") != comparisons:
            raise PermissionError("bound paired-comparison registry is incomplete or reordered")

    expected_signature_tag = {"registered_json": "${prediction_runtime_signature}"}
    execution_assembly = bound_by_key["representation_execution_lock"]["rows"][0][
        "parameters"
    ]["assembly"]
    if execution_assembly.get("prediction_runtime_signature") != expected_signature_tag:
        raise PermissionError("execution lock runtime signature tag differs")
    for row in bound_by_key["final_prediction_shards"]["rows"]:
        if row["parameters"]["assembly"].get(
            "producer_runtime_signature"
        ) != expected_signature_tag:
            raise PermissionError("prediction shard runtime signature tag differs")


def build_runtime_dry_run_audit(
    spec: Mapping[str, Any], runtime_binding: Mapping[str, Any],
    command_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a nonmutating command plan has closed adapter rows."""

    from .hcwdl_representation_campaign import (
        COMMAND_PLAN_CONTRACT, validate_command_plan,
    )

    validate_command_plan(command_plan, spec=spec)
    validate_bound_runtime_task_rows(spec, runtime_binding)
    runtime_hash = require_sha256(
        runtime_binding.get("content_hash"), name="runtime binding",
    )
    if require_sha256(
        spec.get("runtime_binding_sha256"), name="campaign runtime binding",
    ) != runtime_hash:
        raise ValueError("dry-run runtime binding differs from campaign specification")
    task_bindings = runtime_binding.get("tasks")
    assert isinstance(task_bindings, list)
    return with_content_hash({
        "contract": RUNTIME_DRY_RUN_AUDIT_CONTRACT,
        "schema_version": 1,
        "campaign_identity_sha256": runtime_binding["campaign_identity_sha256"],
        "runtime_binding_sha256": runtime_hash,
        "command_plan_sha256": require_sha256(
            command_plan.get("content_hash"), name="command plan",
        ),
        "validated_task_count": len(task_bindings),
        "validated_runtime_row_count": sum(
            len(task_binding["rows"]) for task_binding in task_bindings
        ),
        "all_registered_runtime_rows_adapter_validated": True,
        "scheduler_mutated": False,
    })


__all__ = [
    "RUNTIME_DRY_RUN_AUDIT_CONTRACT", "RUNTIME_PREREQUISITES_CONTRACT",
    "build_runtime_dry_run_audit", "build_runtime_prerequisites",
    "build_runtime_task_rows", "validate_bound_runtime_task_rows",
    "validate_runtime_task_rows",
]
