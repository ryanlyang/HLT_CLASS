"""Combined HCWDL/HCWDL-RKD final registry and label-isolated evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import platform
from pathlib import Path
from typing import Any, Final

import numpy as np
import scipy

from hlt_classification.data.cache_contracts import (
    canonical_sha256, deterministic_npz_bytes, load_npz_arrays,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .evaluation import classification_metrics
from .hcwdl_final_stream import FINAL_LABEL_ESCROW_CONTRACT, FINAL_ROW_SELECTION_CONTRACT
from .hcwdl_shared_final import (
    FINAL_EXECUTION_CLAIM_CONTRACT,
    FINAL_RESERVATION_CONTRACT,
    FINAL_TASK_REGISTRY_CONTRACT, validate_role_capability,
)
from .hcwdl_representation_artifacts import (
    CommittedBinaryEnvelope, publish_binary_envelope, validate_binary_envelope,
)
from .hcwdl_representation_contracts import (
    SHARED_BINARY_ENVELOPE_CONTRACT, logical_array_sha256,
)
from .hcwdl_paired_bootstrap import (
    PAIRED_BOOTSTRAP_CONTRACT, publish_paired_bootstrap_envelope,
)
from .hcwdl_representation_reporting import (
    CONFIRMATION_AGGREGATE_CONTRACT, CONFIRMATION_REGISTRY_CONTRACT,
    CONFIRMATION_SEEDS, SCREEN_CONTRACT,
)


FINALIST_LOCK_CONTRACT: Final = "HCWDL_REPRESENTATION_FINALIST_LOCK/v1"
FINAL_ASSIGNMENT_AUDIT_CONTRACT: Final = "HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1"
FINAL_DATA_ATTESTATION_CONTRACT: Final = "HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1"
EXECUTION_LOCK_CONTRACT: Final = "HCWDL_REPRESENTATION_EXECUTION_LOCK/v1"
PREDICTION_SPEC_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1"
PREDICTION_SHARD_CONTRACT: Final = "HCWDL_REPRESENTATION_PREDICTION_SHARD/v1"
PREDICTION_MANIFEST_CONTRACT: Final = "HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v1"
FINAL_EVALUATION_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_EVALUATION/v1"
METRIC_JOIN_CONTRACT: Final = "HCWDL_REPRESENTATION_METRIC_JOIN/v1"
FINAL_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v1"
REPRESENTATION_ENDPOINTS: Final = (
    "RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w",
)
METRIC_RUNTIME_DOMAIN: Final = "HCWDL_REPRESENTATION_METRIC_RUNTIME_SIGNATURE/v1"


def build_metric_runtime_signature() -> dict[str, Any]:
    """Freeze the exact local implementation/runtime used for final metrics."""

    source_path = Path(__file__).with_name("evaluation.py")
    schema_path = Path(__file__).with_name("schema.py")
    from .schema import CLASS_NAMES
    value = {
        "domain": METRIC_RUNTIME_DOMAIN,
        "schema_version": 1,
        "evaluation_source_sha256": sha256_file(source_path),
        "schema_source_sha256": sha256_file(schema_path),
        "class_names_sha256": canonical_sha256(list(CLASS_NAMES)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "logit_input": "finite_float32_promoted_to_float64/v1",
        "metric_entry_point": "classification_metrics",
    }
    return {**value, "signature_sha256": canonical_sha256(value)}


def validate_metric_runtime_signature(
    value: Mapping[str, Any], *, require_live: bool = False,
) -> str:
    required = {
        "domain", "schema_version", "evaluation_source_sha256",
        "schema_source_sha256", "class_names_sha256",
        "python_version", "numpy_version", "scipy_version", "logit_input",
        "metric_entry_point", "signature_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("final metric runtime signature schema differs")
    if (
        value.get("domain") != METRIC_RUNTIME_DOMAIN
        or value.get("schema_version") != 1
        or value.get("logit_input") != "finite_float32_promoted_to_float64/v1"
        or value.get("metric_entry_point") != "classification_metrics"
        or any(
            not isinstance(value.get(name), str) or not value.get(name)
            for name in ("python_version", "numpy_version", "scipy_version")
        )
    ):
        raise ValueError("final metric runtime signature differs")
    require_sha256(
        value.get("evaluation_source_sha256"), name="metric evaluation source",
    )
    require_sha256(value.get("schema_source_sha256"), name="metric schema source")
    require_sha256(value.get("class_names_sha256"), name="metric class names")
    supplied = require_sha256(
        value.get("signature_sha256"), name="metric runtime signature",
    )
    unhashed = dict(value)
    unhashed.pop("signature_sha256")
    if supplied != canonical_sha256(unhashed):
        raise ValueError("final metric runtime signature hash differs")
    if require_live and dict(value) != build_metric_runtime_signature():
        raise PermissionError("live final metric runtime/source differs from execution lock")
    return supplied


def _selection_attestation_fields(selection: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(selection, expected_contract=FINAL_ROW_SELECTION_CONTRACT)
    row_count = selection.get("row_count")
    identities = selection.get("identity_digests")
    selected_rows = selection.get("selected_rows")
    class_counts = selection.get("class_counts")
    if (
        isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0
        or not isinstance(identities, list) or len(identities) != row_count
        or not isinstance(selected_rows, list) or len(selected_rows) != row_count
        or not isinstance(class_counts, list) or len(class_counts) != 15
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in class_counts)
        or sum(class_counts) != row_count
    ):
        raise ValueError("final selection row/class inventory differs")
    normalized_identities = [
        require_sha256(value, name="selected identity") for value in identities
    ]
    if (
        len(normalized_identities) != len(set(normalized_identities))
        or selection.get("identity_order_sha256")
        != canonical_sha256(normalized_identities)
    ):
        raise ValueError("final selection identity order differs")
    source_counts: dict[str, int] = {}
    order_keys: list[tuple[str, int]] = []
    for position, (row, identity) in enumerate(
        zip(selected_rows, normalized_identities, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError("final selection source row differs")
        source = str(row.get("source_path", ""))
        source_file_sha256 = require_sha256(
            row.get("source_file_sha256"), name=f"selected source {position}",
        )
        source_entry = row.get("source_entry")
        if (
            not source or isinstance(source_entry, bool)
            or not isinstance(source_entry, int) or source_entry < 0
            or row.get("identity_digest") != identity
            or canonical_sha256({
                "source_file_sha256": source_file_sha256,
                "source_entry": source_entry,
            }) != identity
        ):
            raise ValueError("final selection source-derived identity differs")
        order_keys.append((source, source_entry))
        source_counts[source] = source_counts.get(source, 0) + 1
    if order_keys != sorted(order_keys) or len(order_keys) != len(set(order_keys)):
        raise ValueError("final selection source order/uniqueness differs")
    return {
        "population_sha256": require_sha256(
            selection.get("population_sha256"), name="selection population",
        ),
        "selection_sha256": selection["content_hash"],
        "selection_rule_sha256": require_sha256(
            selection.get("selection_rule_sha256"), name="selection rule",
        ),
        "row_count": row_count,
        "class_counts": list(class_counts),
        "identity_order_sha256": canonical_sha256(normalized_identities),
        "source_counts": dict(sorted(source_counts.items())),
    }


def _assignment_manifest_attestation_fields(
    *, assignment_manifest: Mapping[str, Any], assignment_spec: Mapping[str, Any],
    selection_fields: Mapping[str, Any],
) -> dict[str, Any]:
    from .highcov_cache import MANIFEST_CONTRACT, SCHEMA_VERSION

    validate_content_hash(
        assignment_manifest, expected_contract=MANIFEST_CONTRACT,
        expected_schema_version=SCHEMA_VERSION,
    )
    assignment_spec_sha256 = require_sha256(
        assignment_spec.get("content_hash"), name="assignment specification",
    )
    sources = assignment_spec.get("source_partitions")
    shards = assignment_manifest.get("shards")
    if (
        assignment_manifest.get("role") != "final_test"
        or not isinstance(sources, list) or not sources
        or sources != sorted(sources) or len(sources) != len(set(sources))
        or not isinstance(shards, list) or len(shards) != len(sources)
        or assignment_manifest.get("expected_mapped_jets")
        != selection_fields["row_count"]
        or assignment_manifest.get("scanned_mapped_jets")
        != selection_fields["row_count"]
    ):
        raise ValueError("final assignment manifest coverage/source inventory differs")
    manifest_source_counts: dict[str, int] = {}
    for row in shards:
        if not isinstance(row, Mapping) or set(row) != {
            "source_path", "metadata_path", "metadata_sha256", "data_sha256", "rows",
        }:
            raise ValueError("final assignment manifest shard row differs")
        source = str(row.get("source_path", ""))
        rows = row.get("rows")
        if (
            not source or source in manifest_source_counts
            or not isinstance(row.get("metadata_path"), str) or not row["metadata_path"]
            or isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
        ):
            raise ValueError("final assignment manifest shard inventory differs")
        require_sha256(row.get("metadata_sha256"), name=f"{source} assignment metadata")
        require_sha256(row.get("data_sha256"), name=f"{source} assignment data")
        manifest_source_counts[source] = rows
    if list(manifest_source_counts) != sources:
        raise ValueError("final assignment manifest/spec source order differs")
    expected_source_counts = {
        source: int(selection_fields["source_counts"].get(source, 0))
        for source in sources
    }
    if (
        manifest_source_counts != expected_source_counts
        or sum(manifest_source_counts.values()) != selection_fields["row_count"]
    ):
        raise ValueError("final assignment manifest source counts differ from selection")
    parents = assignment_manifest.get("parents")
    expected_parent_keys = {
        "selection", "assignment_spec",
        *(f"assignment_shard_{position:04d}" for position in range(len(sources))),
    }
    if not isinstance(parents, Mapping) or set(parents) != expected_parent_keys:
        raise ValueError("final assignment manifest parent inventory differs")
    normalized_parents = {
        str(name): require_sha256(digest, name=f"assignment parent {name}")
        for name, digest in parents.items()
    }
    if (
        normalized_parents["selection"] != selection_fields["selection_sha256"]
        or normalized_parents["assignment_spec"] != assignment_spec_sha256
    ):
        raise ValueError("final assignment manifest selection/spec parents differ")
    visible = assignment_manifest.get("visible_hlt_tokens")
    assigned = assignment_manifest.get("assigned_hlt_tokens")
    dustbin = assignment_manifest.get("dustbin_fraction")
    if (
        isinstance(visible, bool) or not isinstance(visible, int) or visible <= 0
        or isinstance(assigned, bool) or not isinstance(assigned, int)
        or assigned < 0 or assigned > visible
        or isinstance(dustbin, bool) or not isinstance(dustbin, (int, float))
        or not math.isfinite(float(dustbin))
        or abs(float(dustbin) - (visible - assigned) / visible) > 1e-15
        or not float(dustbin) < 0.10
    ):
        raise ValueError("final assignment manifest token/dustbin totals differ")
    visible_by_category = assignment_manifest.get("visible_by_category")
    assigned_by_category = assignment_manifest.get("assigned_by_category")
    unclassified = assignment_manifest.get("unclassified_hlt_tokens")
    if (
        not isinstance(visible_by_category, list) or len(visible_by_category) != 5
        or not isinstance(assigned_by_category, list) or len(assigned_by_category) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (*visible_by_category, *assigned_by_category))
        or isinstance(unclassified, bool) or not isinstance(unclassified, int)
        or unclassified < 0
        or sum(visible_by_category) + unclassified != visible
        or sum(assigned_by_category) != assigned
        or any(right > left for left, right in zip(
            visible_by_category, assigned_by_category, strict=True,
        ))
    ):
        raise ValueError("final assignment manifest category totals differ")
    return {
        "assignment_manifest_sha256": assignment_manifest["content_hash"],
        "assignment_spec_sha256": assignment_spec_sha256,
        "assignment_manifest_parents": dict(sorted(normalized_parents.items())),
        "source_counts": expected_source_counts,
    }


def _finalist_row(row: Mapping[str, Any]) -> dict[str, Any]:
    finalist_id = str(row.get("finalist_id", ""))
    domain = str(row.get("domain", ""))
    if not finalist_id or domain not in {"hlt", "shell_exact_d100", "native_offline"}:
        raise ValueError("finalist identity or input domain differs")
    deployable = bool(row.get("deployable", False))
    extraction = row.get("extraction_sha256")
    if deployable:
        require_sha256(extraction, name=f"{finalist_id} extraction")
        if domain != "hlt":
            raise ValueError("deployable finalist must consume HLT only")
    elif extraction is not None:
        raise ValueError("oracle/control finalist cannot claim deployable extraction")
    execution_id = row.get("execution_id")
    selection_sha256 = row.get("checkpoint_selection_sha256")
    source_campaign = str(row.get("source_campaign", ""))
    if not source_campaign:
        raise ValueError("finalist source campaign is empty")
    if source_campaign == "representation":
        require_sha256(execution_id, name=f"{finalist_id} execution")
        require_sha256(selection_sha256, name=f"{finalist_id} checkpoint selection")
    elif execution_id is not None or selection_sha256 is not None:
        require_sha256(execution_id, name=f"{finalist_id} execution")
        require_sha256(selection_sha256, name=f"{finalist_id} checkpoint selection")
    return {
        "finalist_id": finalist_id,
        "checkpoint_sha256": require_sha256(row.get("checkpoint_sha256"), name=f"{finalist_id} checkpoint"),
        "report_sha256": require_sha256(row.get("report_sha256"), name=f"{finalist_id} report"),
        "domain": domain,
        "deployable": deployable,
        "extraction_sha256": extraction,
        "source_campaign": source_campaign,
        "screening_seed": int(row.get("screening_seed", 1337)),
        "execution_id": execution_id,
        "checkpoint_selection_sha256": selection_sha256,
    }


def build_finalist_lock(
    *,
    parent_finalists: Sequence[Mapping[str, Any]],
    representation_endpoints: Sequence[Mapping[str, Any]],
    parent_campaign_sha256: str,
    representation_campaign_sha256: str,
    reservation: Mapping[str, Any],
    parent_recipe_sha256: str,
    representation_recipe_sha256: str,
    loss_attestation_sha256: str,
    architecture_attestation_sha256: str,
    selection_rule_sha256: str,
    assignment_spec_sha256: str,
    legacy_cancellation_sha256: str | None,
    parent_finalist_registry_sha256: str,
    screen_aggregate: Mapping[str, Any],
    confirmation_registry: Mapping[str, Any],
    confirmation_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(reservation, expected_contract=FINAL_RESERVATION_CONTRACT)
    validate_content_hash(screen_aggregate, expected_contract=SCREEN_CONTRACT)
    validate_content_hash(
        confirmation_registry, expected_contract=CONFIRMATION_REGISTRY_CONTRACT,
    )
    validate_content_hash(
        confirmation_aggregate, expected_contract=CONFIRMATION_AGGREGATE_CONTRACT,
    )
    if reservation.get("allows_final_execution") is not True:
        raise PermissionError("validation-only campaign cannot freeze final finalists")
    parent_rows = [_finalist_row(row) for row in parent_finalists]
    endpoint_rows = [_finalist_row(row) for row in representation_endpoints]
    if {row["finalist_id"] for row in endpoint_rows} != set(REPRESENTATION_ENDPOINTS):
        raise ValueError("representation finalist registry must contain the four seed-1337 M6 endpoints")
    if any(row["screening_seed"] != 1337 for row in endpoint_rows):
        raise ValueError("confirmation seeds cannot enter the finalist registry")
    if any(row["source_campaign"] != "representation" for row in endpoint_rows):
        raise ValueError("representation endpoint source campaign differs")
    screen_endpoints = {
        str(row.get("node_id")): row
        for row in screen_aggregate.get("primary_rows", ())
        if str(row.get("node_id")) in REPRESENTATION_ENDPOINTS
    }
    if set(screen_endpoints) != set(REPRESENTATION_ENDPOINTS) or any(
        screen_endpoints[row["finalist_id"]].get("report_sha256") != row["report_sha256"]
        for row in endpoint_rows
    ):
        raise ValueError("representation endpoints differ from the completed screen registry")
    rows = parent_rows + endpoint_rows
    if len(rows) != len({row["finalist_id"] for row in rows}) or not parent_rows:
        raise ValueError("combined finalist union is empty or repeats a finalist")
    if reservation.get("campaign_spec_sha256") != require_sha256(
        representation_campaign_sha256, name="representation campaign",
    ):
        raise ValueError("finalist lock campaign differs from pre-training reservation")
    if reservation.get("selection_rule_sha256") != require_sha256(
        selection_rule_sha256, name="selection rule",
    ) or reservation.get("assignment_spec_sha256") != require_sha256(
        assignment_spec_sha256, name="assignment spec",
    ):
        raise ValueError("finalist lock selection/assignment commitments differ")
    if reservation.get("legacy_jobs_present") is True and legacy_cancellation_sha256 is None:
        raise PermissionError("legacy final jobs require cancellation proof before finalists")
    confirmation_rows = confirmation_registry.get("rows")
    if not isinstance(confirmation_rows, list):
        raise ValueError("confirmation registry row inventory differs")
    confirmation_pairs = {
        (str(row.get("objective_id")), int(row.get("seed", -1)))
        for row in confirmation_rows
    }
    expected_confirmation_pairs = {
        (objective, seed)
        for objective in REPRESENTATION_ENDPOINTS
        for seed in CONFIRMATION_SEEDS
    }
    if (
        confirmation_registry.get("screen_sha256") != screen_aggregate["content_hash"]
        or confirmation_registry.get("campaign_sha256")
        != require_sha256(representation_campaign_sha256, name="representation campaign")
        or confirmation_pairs != expected_confirmation_pairs
        or len(confirmation_rows) != len(expected_confirmation_pairs)
        or confirmation_aggregate.get("registry_sha256")
        != confirmation_registry["content_hash"]
        or set(confirmation_aggregate.get("objectives", {}))
        != set(REPRESENTATION_ENDPOINTS)
        or confirmation_aggregate.get("used_for_finalist_selection") is not False
    ):
        raise ValueError("confirmation registry/aggregate lineage differs")
    result = with_content_hash(
        {
            "contract": FINALIST_LOCK_CONTRACT,
            "schema_version": 1,
            "population_sha256": reservation["population_sha256"],
            "reservation_sha256": reservation["content_hash"],
            "parent_campaign_sha256": require_sha256(parent_campaign_sha256, name="parent campaign"),
            "representation_campaign_sha256": require_sha256(
                representation_campaign_sha256, name="representation campaign"
            ),
            "parent_recipe_sha256": require_sha256(parent_recipe_sha256, name="parent recipe"),
            "representation_recipe_sha256": require_sha256(
                representation_recipe_sha256, name="representation recipe"
            ),
            "loss_attestation_sha256": require_sha256(loss_attestation_sha256, name="loss attestation"),
            "architecture_attestation_sha256": require_sha256(
                architecture_attestation_sha256, name="architecture attestation"
            ),
            "selection_rule_sha256": require_sha256(selection_rule_sha256, name="selection rule"),
            "assignment_spec_sha256": require_sha256(assignment_spec_sha256, name="assignment spec"),
            "legacy_cancellation_sha256": (
                None
                if legacy_cancellation_sha256 is None
                else require_sha256(legacy_cancellation_sha256, name="legacy cancellation")
            ),
            "parent_finalist_registry_sha256": require_sha256(
                parent_finalist_registry_sha256, name="parent finalist registry"
            ),
            "pretraining_finalist_registry_commitment_sha256": reservation[
                "finalist_registry_commitment_sha256"
            ],
            "screen_aggregate_sha256": screen_aggregate["content_hash"],
            "confirmation_registry_sha256": confirmation_registry["content_hash"],
            "confirmation_aggregate_sha256": confirmation_aggregate["content_hash"],
            "parent_finalist_count": len(parent_rows),
            "representation_endpoint_count": 4,
            "finalists": sorted(rows, key=lambda row: row["finalist_id"]),
            "confirmation_seeds_are_not_finalists": True,
            "m5_controls_are_validation_only": True,
        }
    )
    return result


def build_assignment_audit(
    *, selection: Mapping[str, Any], assignment_manifest: Mapping[str, Any],
    assignment_spec: Mapping[str, Any],
    assigned_identity_digests: Sequence[str], population_sha256: str,
) -> dict[str, Any]:
    selection_fields = _selection_attestation_fields(selection)
    from .hcwdl_representation_final_policy import FINAL_ASSIGNMENT_SPEC_CONTRACT
    validate_content_hash(
        assignment_spec, expected_contract=FINAL_ASSIGNMENT_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    manifest_fields = _assignment_manifest_attestation_fields(
        assignment_manifest=assignment_manifest, assignment_spec=assignment_spec,
        selection_fields=selection_fields,
    )
    actual = tuple(str(value) for value in assigned_identity_digests)
    expected = tuple(str(value) for value in selection["identity_digests"])
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("final assignment coverage/order differs from selected identities")
    if require_sha256(
        population_sha256, name="assignment population",
    ) != selection_fields["population_sha256"]:
        raise ValueError("final assignment population differs from selection")
    result = with_content_hash(
        {
            "contract": FINAL_ASSIGNMENT_AUDIT_CONTRACT,
            "schema_version": 1,
            "population_sha256": selection_fields["population_sha256"],
            "selection_sha256": selection_fields["selection_sha256"],
            "selection_rule_sha256": selection_fields["selection_rule_sha256"],
            **manifest_fields,
            "rows": selection_fields["row_count"],
            "class_counts": selection_fields["class_counts"],
            "identity_order_sha256": selection_fields["identity_order_sha256"],
            "complete": True,
        }
    )
    return result


def validate_assignment_audit(
    value: Mapping[str, Any], *, selection: Mapping[str, Any] | None = None,
    assignment_manifest: Mapping[str, Any] | None = None,
    assignment_spec: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_ASSIGNMENT_AUDIT_CONTRACT,
    )
    required = {
        "contract", "schema_version", "population_sha256", "selection_sha256",
        "selection_rule_sha256", "assignment_manifest_sha256",
        "assignment_spec_sha256", "assignment_manifest_parents", "source_counts",
        "rows", "class_counts", "identity_order_sha256", "complete", "content_hash",
    }
    if set(value) != required or value.get("complete") is not True:
        raise ValueError("final assignment audit schema/completeness differs")
    for name in (
        "population_sha256", "selection_sha256", "selection_rule_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "identity_order_sha256",
    ):
        require_sha256(value.get(name), name=f"assignment audit {name}")
    if selection is not None:
        fields = _selection_attestation_fields(selection)
        if any((
            value["population_sha256"] != fields["population_sha256"],
            value["selection_sha256"] != fields["selection_sha256"],
            value["selection_rule_sha256"] != fields["selection_rule_sha256"],
            value["rows"] != fields["row_count"],
            value["class_counts"] != fields["class_counts"],
            value["identity_order_sha256"] != fields["identity_order_sha256"],
        )):
            raise ValueError("final assignment audit differs from selection")
    if (assignment_manifest is None) != (assignment_spec is None):
        raise ValueError("assignment audit validation requires manifest and specification")
    if assignment_manifest is not None and assignment_spec is not None:
        fields = _selection_attestation_fields(selection) if selection is not None else {
            "selection_sha256": value["selection_sha256"],
            "row_count": value["rows"],
            "source_counts": value["source_counts"],
        }
        manifest_fields = _assignment_manifest_attestation_fields(
            assignment_manifest=assignment_manifest, assignment_spec=assignment_spec,
            selection_fields=fields,
        )
        if any(value[name] != expected for name, expected in manifest_fields.items()):
            raise ValueError("final assignment audit differs from manifest/specification")
    return digest


def build_final_data_attestation(
    *, selection: Mapping[str, Any], assignment_audit: Mapping[str, Any],
    assignment_manifest: Mapping[str, Any], assignment_spec: Mapping[str, Any],
    matcher_resources: Mapping[str, Any],
    label_escrow: Mapping[str, Any], task_registry: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    selection_fields = _selection_attestation_fields(selection)
    from .hcwdl_representation_final_policy import validate_final_assignment_spec
    source_partitions = assignment_spec.get("source_partitions")
    if not isinstance(source_partitions, list):
        raise ValueError("final assignment specification source inventory differs")
    validate_final_assignment_spec(
        assignment_spec, matcher_resources=matcher_resources,
        source_partitions=source_partitions, step_size=8192,
    )
    validate_assignment_audit(
        assignment_audit, selection=selection,
        assignment_manifest=assignment_manifest, assignment_spec=assignment_spec,
    )
    validate_content_hash(label_escrow, expected_contract=FINAL_LABEL_ESCROW_CONTRACT)
    validate_content_hash(task_registry, expected_contract=FINAL_TASK_REGISTRY_CONTRACT)
    validate_content_hash(claim, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT)
    hashes = {
        selection["population_sha256"],
        assignment_audit["population_sha256"],
        task_registry["population_sha256"],
        claim["population_sha256"],
    }
    if len(hashes) != 1 or assignment_audit["selection_sha256"] != selection["content_hash"]:
        raise ValueError("final data attestation lineage differs")
    escrow_payload = label_escrow.get("payload", label_escrow)
    if (
        escrow_payload.get("selection_sha256") != selection["content_hash"]
        or escrow_payload.get("population_sha256") != selection["population_sha256"]
        or escrow_payload.get("rows") != int(selection["row_count"])
    ):
        raise ValueError("final data attestation label-escrow lineage differs")
    return with_content_hash(
        {
            "contract": FINAL_DATA_ATTESTATION_CONTRACT,
            "schema_version": 1,
            "population_sha256": hashes.pop(),
            "selection_sha256": selection_fields["selection_sha256"],
            "selection_rule_sha256": selection_fields["selection_rule_sha256"],
            "label_escrow_sha256": label_escrow["content_hash"],
            "assignment_audit_sha256": assignment_audit["content_hash"],
            "assignment_manifest_sha256": assignment_audit[
                "assignment_manifest_sha256"
            ],
            "assignment_spec_sha256": assignment_audit["assignment_spec_sha256"],
            "assignment_manifest_parents": assignment_audit[
                "assignment_manifest_parents"
            ],
            "task_registry_sha256": task_registry["content_hash"],
            "execution_claim_sha256": claim["content_hash"],
            "row_count": selection_fields["row_count"],
            "class_counts": selection_fields["class_counts"],
            "source_counts": assignment_audit["source_counts"],
            "identity_order_sha256": selection_fields["identity_order_sha256"],
            "complete": True,
        }
    )


def validate_final_data_attestation(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_DATA_ATTESTATION_CONTRACT,
    )
    required = {
        "contract", "schema_version", "population_sha256", "selection_sha256",
        "selection_rule_sha256", "label_escrow_sha256", "assignment_audit_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "assignment_manifest_parents", "task_registry_sha256",
        "execution_claim_sha256", "row_count", "class_counts", "source_counts",
        "identity_order_sha256", "complete", "content_hash",
    }
    if set(value) != required or value.get("complete") is not True:
        raise ValueError("final data attestation schema/completeness differs")
    for name in (
        "population_sha256", "selection_sha256", "selection_rule_sha256",
        "label_escrow_sha256", "assignment_audit_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "task_registry_sha256", "execution_claim_sha256", "identity_order_sha256",
    ):
        require_sha256(value.get(name), name=f"final data {name}")
    parents = value.get("assignment_manifest_parents")
    sources = value.get("source_counts")
    classes = value.get("class_counts")
    rows = value.get("row_count")
    if not isinstance(parents, Mapping) or not parents:
        raise ValueError("final data attestation count/parent inventory differs")
    for name, parent_digest in parents.items():
        if not isinstance(name, str) or not name:
            raise ValueError("final data attestation assignment parent differs")
        require_sha256(parent_digest, name=f"assignment parent {name}")
    if (
        not isinstance(sources, Mapping) or not sources
        or any(not isinstance(name, str) or not name or isinstance(count, bool)
               or not isinstance(count, int) or count < 0
               for name, count in sources.items())
        or list(sources) != sorted(sources)
        or isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
        or sum(sources.values()) != rows
        or not isinstance(classes, list) or len(classes) != 15
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
               for count in classes)
        or sum(classes) != rows
    ):
        raise ValueError("final data attestation count/parent inventory differs")
    return digest


def build_execution_lock(
    *, finalist_lock: Mapping[str, Any], data_attestation: Mapping[str, Any],
    claim: Mapping[str, Any], task_registry: Mapping[str, Any],
) -> dict[str, Any]:
    for value, contract in (
        (finalist_lock, FINALIST_LOCK_CONTRACT),
        (data_attestation, FINAL_DATA_ATTESTATION_CONTRACT),
        (claim, FINAL_EXECUTION_CLAIM_CONTRACT),
        (task_registry, FINAL_TASK_REGISTRY_CONTRACT),
    ):
        if contract == FINAL_DATA_ATTESTATION_CONTRACT:
            validate_final_data_attestation(value)
        else:
            validate_content_hash(value, expected_contract=contract)
    population = finalist_lock["population_sha256"]
    if any(value["population_sha256"] != population for value in (data_attestation, claim, task_registry)):
        raise ValueError("representation execution lock population differs")
    if claim.get("task_registry_sha256") != task_registry["content_hash"]:
        raise ValueError("execution claim task registry differs")
    if claim.get("finalist_lock_sha256") != finalist_lock["content_hash"]:
        raise ValueError("execution claim finalist lock differs")
    if data_attestation.get("task_registry_sha256") != task_registry["content_hash"]:
        raise ValueError("final data attestation registry differs")
    if data_attestation.get("execution_claim_sha256") != claim["content_hash"]:
        raise ValueError("final data attestation claim differs")
    if (
        data_attestation.get("selection_rule_sha256")
        != finalist_lock.get("selection_rule_sha256")
        or data_attestation.get("assignment_spec_sha256")
        != finalist_lock.get("assignment_spec_sha256")
    ):
        raise PermissionError("execution lock selection/assignment policy differs from finalists")
    metric_runtime_signature = build_metric_runtime_signature()
    validate_metric_runtime_signature(metric_runtime_signature, require_live=True)
    result = with_content_hash(
        {
            "contract": EXECUTION_LOCK_CONTRACT,
            "schema_version": 1,
            "population_sha256": population,
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "data_attestation_sha256": data_attestation["content_hash"],
            "execution_claim_sha256": claim["content_hash"],
            "task_registry_sha256": task_registry["content_hash"],
            "selection_sha256": data_attestation["selection_sha256"],
            "selection_rule_sha256": data_attestation["selection_rule_sha256"],
            "assignment_audit_sha256": data_attestation["assignment_audit_sha256"],
            "assignment_manifest_sha256": data_attestation[
                "assignment_manifest_sha256"
            ],
            "assignment_spec_sha256": data_attestation["assignment_spec_sha256"],
            "assignment_manifest_parents": data_attestation[
                "assignment_manifest_parents"
            ],
            "row_count": data_attestation["row_count"],
            "class_counts": data_attestation["class_counts"],
            "source_counts": data_attestation["source_counts"],
            "identity_order_sha256": data_attestation["identity_order_sha256"],
            "metric_runtime_signature": metric_runtime_signature,
            "prediction_and_metric_registry_frozen": True,
        }
    )
    validate_execution_lock(
        result, finalist_lock=finalist_lock, data_attestation=data_attestation,
        claim=claim, task_registry=task_registry, require_live_metric=True,
    )
    return result


def validate_execution_lock(
    value: Mapping[str, Any], *,
    finalist_lock: Mapping[str, Any] | None = None,
    data_attestation: Mapping[str, Any] | None = None,
    claim: Mapping[str, Any] | None = None,
    task_registry: Mapping[str, Any] | None = None,
    require_live_metric: bool = False,
) -> str:
    digest = validate_content_hash(value, expected_contract=EXECUTION_LOCK_CONTRACT)
    required = {
        "contract", "schema_version", "population_sha256", "finalist_lock_sha256",
        "data_attestation_sha256", "execution_claim_sha256", "task_registry_sha256",
        "selection_sha256", "selection_rule_sha256", "assignment_audit_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "assignment_manifest_parents", "row_count", "class_counts", "source_counts",
        "identity_order_sha256", "metric_runtime_signature",
        "prediction_and_metric_registry_frozen", "content_hash",
    }
    if set(value) != required or value.get(
        "prediction_and_metric_registry_frozen"
    ) is not True:
        raise ValueError("representation execution lock schema/freeze differs")
    for name in (
        "population_sha256", "finalist_lock_sha256", "data_attestation_sha256",
        "execution_claim_sha256", "task_registry_sha256", "selection_sha256",
        "selection_rule_sha256", "assignment_audit_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "identity_order_sha256",
    ):
        require_sha256(value.get(name), name=f"execution lock {name}")
    validate_metric_runtime_signature(
        value.get("metric_runtime_signature", {}), require_live=require_live_metric,
    )
    parents = value.get("assignment_manifest_parents")
    source_counts = value.get("source_counts")
    class_counts = value.get("class_counts")
    row_count = value.get("row_count")
    if (
        not isinstance(parents, Mapping) or not parents
        or not isinstance(source_counts, Mapping) or not source_counts
        or list(source_counts) != sorted(source_counts)
        or any(not isinstance(source, str) or not source or isinstance(count, bool)
               or not isinstance(count, int) or count < 0
               for source, count in source_counts.items())
        or isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0
        or sum(source_counts.values()) != row_count
        or not isinstance(class_counts, list) or len(class_counts) != 15
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
               for count in class_counts)
        or sum(class_counts) != row_count
    ):
        raise ValueError("representation execution lock count/parent inventory differs")
    for name, parent_digest in parents.items():
        if not isinstance(name, str) or not name:
            raise ValueError("representation execution lock assignment parent differs")
        require_sha256(parent_digest, name=f"execution assignment parent {name}")
    if finalist_lock is not None:
        validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
        if (
            value["finalist_lock_sha256"] != finalist_lock["content_hash"]
            or value["population_sha256"] != finalist_lock["population_sha256"]
            or value["selection_rule_sha256"]
            != finalist_lock.get("selection_rule_sha256")
            or value["assignment_spec_sha256"]
            != finalist_lock.get("assignment_spec_sha256")
        ):
            raise ValueError("representation execution lock finalist lineage differs")
    if data_attestation is not None:
        validate_final_data_attestation(data_attestation)
        copied = {
            "population_sha256": "population_sha256",
            "selection_sha256": "selection_sha256",
            "selection_rule_sha256": "selection_rule_sha256",
            "assignment_audit_sha256": "assignment_audit_sha256",
            "assignment_manifest_sha256": "assignment_manifest_sha256",
            "assignment_spec_sha256": "assignment_spec_sha256",
            "assignment_manifest_parents": "assignment_manifest_parents",
            "row_count": "row_count", "class_counts": "class_counts",
            "source_counts": "source_counts",
            "identity_order_sha256": "identity_order_sha256",
        }
        if (
            value["data_attestation_sha256"] != data_attestation["content_hash"]
            or any(value[target] != data_attestation[source]
                   for target, source in copied.items())
        ):
            raise ValueError("representation execution lock data lineage differs")
    if claim is not None:
        validate_content_hash(claim, expected_contract=FINAL_EXECUTION_CLAIM_CONTRACT)
        if value["execution_claim_sha256"] != claim["content_hash"]:
            raise ValueError("representation execution lock claim lineage differs")
    if task_registry is not None:
        validate_content_hash(task_registry, expected_contract=FINAL_TASK_REGISTRY_CONTRACT)
        if value["task_registry_sha256"] != task_registry["content_hash"]:
            raise ValueError("representation execution lock registry lineage differs")
    return digest


def build_prediction_spec(
    *, finalist_lock: Mapping[str, Any], execution_lock: Mapping[str, Any],
    row_selection: Mapping[str, Any], runtime_signature: Mapping[str, Any],
    source_partitions: Sequence[str],
) -> dict[str, Any]:
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    validate_execution_lock(execution_lock, finalist_lock=finalist_lock)
    selection_fields = _selection_attestation_fields(row_selection)
    if (
        finalist_lock["population_sha256"] != execution_lock["population_sha256"]
        or row_selection["population_sha256"] != execution_lock["population_sha256"]
        or execution_lock["finalist_lock_sha256"] != finalist_lock["content_hash"]
        or row_selection["content_hash"] != execution_lock.get("selection_sha256")
        or selection_fields["selection_rule_sha256"]
        != execution_lock.get("selection_rule_sha256")
        or selection_fields["row_count"] != execution_lock.get("row_count")
        or selection_fields["class_counts"] != execution_lock.get("class_counts")
        or selection_fields["identity_order_sha256"]
        != execution_lock.get("identity_order_sha256")
    ):
        raise ValueError("final prediction spec lineage differs")
    partitions = tuple(str(value) for value in source_partitions)
    if not partitions or len(partitions) != len(set(partitions)) or partitions != tuple(sorted(partitions)):
        raise ValueError("prediction source partitions differ")
    expected_sources = tuple(execution_lock.get("source_counts", ()))
    if partitions != expected_sources or {
        source: int(selection_fields["source_counts"].get(source, 0))
        for source in partitions
    } != execution_lock.get("source_counts"):
        raise ValueError("prediction source partitions/counts differ from execution lock")
    validate_prediction_runtime_signature(runtime_signature)
    return with_content_hash(
        {
            "contract": PREDICTION_SPEC_CONTRACT,
            "schema_version": 1,
            "population_sha256": execution_lock["population_sha256"],
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "execution_lock_sha256": execution_lock["content_hash"],
            "row_selection_sha256": row_selection["content_hash"],
            "selection_rule_sha256": execution_lock["selection_rule_sha256"],
            "assignment_manifest_sha256": execution_lock[
                "assignment_manifest_sha256"
            ],
            "assignment_spec_sha256": execution_lock["assignment_spec_sha256"],
            "row_count": execution_lock["row_count"],
            "class_counts": execution_lock["class_counts"],
            "source_counts": execution_lock["source_counts"],
            "identity_order_sha256": execution_lock["identity_order_sha256"],
            "architecture_attestation_sha256": require_sha256(
                finalist_lock["architecture_attestation_sha256"],
                name="final prediction architecture attestation",
            ),
            "finalists": finalist_lock["finalists"],
            "source_partitions": list(partitions),
            "runtime_signature": dict(runtime_signature),
            "output": "finite_fp32_c_order_logits_only",
        }
    )


def validate_prediction_runtime_signature(value: Mapping[str, Any]) -> str:
    """Validate every execution choice frozen for final GPU inference."""

    required = {
        "device", "device_signature", "software_signature", "model_mode",
        "parameter_dtype", "input_dtype", "forward_dtype", "batch_size",
        "batch_partition_policy", "final_short_batch_policy", "autocast",
        "tf32", "deterministic_algorithms", "backend_flags",
        "feature_identity_streamer_sha256", "row_runtime_signature_sha256",
        "output_dtype", "output_order", "softmax_location",
    }
    backend = value.get("backend_flags") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or any(not isinstance(value[name], str) or not value[name] for name in (
            "device", "device_signature", "software_signature",
        ))
        or value["model_mode"] != "eval"
        or any(value[name] != "float32" for name in (
            "parameter_dtype", "input_dtype", "forward_dtype", "output_dtype",
        ))
        or isinstance(value["batch_size"], bool)
        or value["batch_size"] != 256
        or value["batch_partition_policy"]
        != "per_source_contiguous_no_cross_source/v1"
        or value["final_short_batch_policy"]
        != "exact_remainder_no_padding/v1"
        or value["autocast"] is not False
        or value["tf32"] is not False
        or value["deterministic_algorithms"] is not True
        or not isinstance(backend, Mapping)
        or dict(backend) != {
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        or value["output_order"] != "C"
        or value["softmax_location"] != "locked_metric_join"
    ):
        raise ValueError("final prediction runtime signature differs")
    digest = require_sha256(
        value["feature_identity_streamer_sha256"],
        name="final feature/identity streamer",
    )
    require_sha256(
        value["row_runtime_signature_sha256"], name="final prediction row runtime",
    )
    from .hcwdl_final_stream import feature_identity_streamer_sha256
    if digest != feature_identity_streamer_sha256():
        raise ValueError("final feature/identity streamer source differs")
    return digest


def validate_prediction_spec(
    value: Mapping[str, Any], *, finalist_lock: Mapping[str, Any] | None = None,
    execution_lock: Mapping[str, Any] | None = None,
    row_selection: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=PREDICTION_SPEC_CONTRACT,
    )
    required = {
        "contract", "schema_version", "population_sha256", "finalist_lock_sha256",
        "execution_lock_sha256", "row_selection_sha256", "selection_rule_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256", "row_count",
        "class_counts", "source_counts", "identity_order_sha256",
        "architecture_attestation_sha256", "finalists", "source_partitions",
        "runtime_signature", "output", "content_hash",
    }
    if set(value) != required:
        raise ValueError("final prediction specification schema differs")
    for name in (
        "population_sha256", "finalist_lock_sha256", "execution_lock_sha256",
        "row_selection_sha256", "selection_rule_sha256",
        "assignment_manifest_sha256", "assignment_spec_sha256",
        "identity_order_sha256", "architecture_attestation_sha256",
    ):
        require_sha256(value.get(name), name=f"prediction specification {name}")
    validate_prediction_runtime_signature(value.get("runtime_signature", {}))
    if value.get("output") != "finite_fp32_c_order_logits_only":
        raise ValueError("final prediction output contract differs")
    partitions = value.get("source_partitions")
    if (
        not isinstance(partitions, list) or not partitions
        or partitions != sorted(partitions) or len(partitions) != len(set(partitions))
        or partitions != list(value.get("source_counts", ()))
    ):
        raise ValueError("prediction source partitions differ")
    if finalist_lock is not None:
        validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
        if (
            value.get("finalist_lock_sha256") != finalist_lock["content_hash"]
            or value.get("finalists") != finalist_lock["finalists"]
            or value.get("architecture_attestation_sha256")
            != finalist_lock.get("architecture_attestation_sha256")
        ):
            raise ValueError("final prediction finalist/architecture lineage differs")
    if execution_lock is not None:
        validate_content_hash(execution_lock, expected_contract=EXECUTION_LOCK_CONTRACT)
        if (
            value.get("execution_lock_sha256") != execution_lock["content_hash"]
            or value.get("population_sha256") != execution_lock["population_sha256"]
            or value.get("row_selection_sha256")
            != execution_lock.get("selection_sha256")
            or value.get("selection_rule_sha256")
            != execution_lock.get("selection_rule_sha256")
            or value.get("assignment_manifest_sha256")
            != execution_lock.get("assignment_manifest_sha256")
            or value.get("assignment_spec_sha256")
            != execution_lock.get("assignment_spec_sha256")
            or value.get("row_count") != execution_lock.get("row_count")
            or value.get("class_counts") != execution_lock.get("class_counts")
            or value.get("source_counts") != execution_lock.get("source_counts")
            or value.get("identity_order_sha256")
            != execution_lock.get("identity_order_sha256")
        ):
            raise ValueError("final prediction execution lineage differs")
    if row_selection is not None:
        validate_content_hash(row_selection, expected_contract=FINAL_ROW_SELECTION_CONTRACT)
        if value.get("row_selection_sha256") != row_selection["content_hash"]:
            raise ValueError("final prediction row-selection lineage differs")
    return digest


def prediction_shard_sidecar(
    *, finalist: Mapping[str, Any], source_partition: str,
    identity_digests: np.ndarray, logits: np.ndarray,
    prediction_spec_sha256: str, execution_lock_sha256: str,
    producer_runtime_signature: Mapping[str, Any], branch_access_sha256: str,
) -> dict[str, Any]:
    identities = np.asarray(identity_digests)
    value = np.asarray(logits)
    if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1:] != (32,):
        raise ValueError("prediction shard identities must be uint8 [rows,32] digests")
    if value.shape != (identities.shape[0], 15) or value.dtype != np.float32:
        raise ValueError("prediction shard must contain FP32 [rows,15] logits")
    if not value.flags.c_contiguous:
        raise ValueError("prediction shard logits must be C-contiguous")
    validate_prediction_runtime_signature(producer_runtime_signature)
    identity_hex = [bytes(row).hex() for row in identities]
    if not np.isfinite(value).all() or len(set(identity_hex)) != len(identity_hex):
        raise ValueError("prediction shard logits/identities differ")
    if identity_hex != sorted(identity_hex):
        raise ValueError("prediction shard identities must be sorted")
    row = _finalist_row(finalist)
    return with_content_hash(
        {
            "contract": PREDICTION_SHARD_CONTRACT,
            "schema_version": 1,
            "finalist": row,
            "source_partition": str(source_partition),
            "rows": len(identities),
            "identity_digests": identity_hex,
            "identity_order_sha256": canonical_sha256(identity_hex),
            "array_sha256": {
                "identity_digests": logical_array_sha256("identity_digests", identities),
                "logits": logical_array_sha256("logits", value),
            },
            "prediction_spec_sha256": require_sha256(prediction_spec_sha256, name="prediction spec"),
            "execution_lock_sha256": require_sha256(execution_lock_sha256, name="execution lock"),
            "branch_access_sha256": require_sha256(branch_access_sha256, name="branch access"),
            "producer_runtime_signature": dict(producer_runtime_signature),
            "contains_labels": False,
            "contains_probabilities": False,
        }
    )


def publish_prediction_shard(
    root: str | Path, *, finalist: Mapping[str, Any], source_partition: str,
    identity_digests: np.ndarray, logits: np.ndarray,
    prediction_spec_sha256: str, execution_lock_sha256: str,
    producer_runtime_signature: Mapping[str, Any], branch_access: Mapping[str, Any],
    producer_task_id: str, registered_output_row: Mapping[str, Any],
    campaign_or_recovery_owner: Mapping[str, Any],
    failure_hook: Callable[[str], None] | None = None,
) -> CommittedBinaryEnvelope:
    """Publish logits-only arrays through the shared immutable envelope."""

    from .hcwdl_final_stream import BRANCH_ACCESS_CONTRACT
    validate_content_hash(branch_access, expected_contract=BRANCH_ACCESS_CONTRACT)
    identities = np.asarray(identity_digests); values = np.asarray(logits)
    payload = prediction_shard_sidecar(
        finalist=finalist, source_partition=source_partition,
        identity_digests=identities, logits=values,
        prediction_spec_sha256=prediction_spec_sha256,
        execution_lock_sha256=execution_lock_sha256,
        producer_runtime_signature=producer_runtime_signature,
        branch_access_sha256=branch_access["content_hash"],
    )
    arrays = {"identity_digests": identities, "logits": values}
    return publish_binary_envelope(
        root, artifact_contract=PREDICTION_SHARD_CONTRACT,
        producer_task_id=producer_task_id,
        schema={"identity_dtype": "uint8", "identity_width": 32, "logit_dtype": "float32", "classes": 15},
        immutable_parent_hashes={
            "prediction_spec": prediction_spec_sha256,
            "execution_lock": execution_lock_sha256,
            "checkpoint": payload["finalist"]["checkpoint_sha256"],
            "branch_access": branch_access["content_hash"],
        },
        registered_output_row=registered_output_row,
        campaign_or_recovery_owner=campaign_or_recovery_owner,
        payloads={"logits.npz": deterministic_npz_bytes(arrays)},
        member_metadata={"logits.npz": {
            "logical_sha256": canonical_sha256(payload["array_sha256"]),
            "dtype": "npz", "shape": [len(values), 15],
        }},
        sidecar_payload=payload, branch_access=branch_access,
        failure_hook=failure_hook,
    )


def load_prediction_shard(
    root: str | Path, envelope_id: str, *, prediction_spec_sha256: str,
    execution_lock_sha256: str, checkpoint_sha256: str,
    branch_access_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    parents = {
        "prediction_spec": prediction_spec_sha256,
        "execution_lock": execution_lock_sha256,
        "checkpoint": checkpoint_sha256,
        "branch_access": branch_access_sha256,
    }
    envelope = validate_binary_envelope(
        root, envelope_id, expected_contract=PREDICTION_SHARD_CONTRACT,
        expected_parents=parents,
    )
    arrays = load_npz_arrays(envelope.directory / "logits.npz")
    payload = envelope.sidecar["payload"]
    expected = prediction_shard_sidecar(
        finalist=payload["finalist"], source_partition=payload["source_partition"],
        identity_digests=arrays["identity_digests"], logits=arrays["logits"],
        prediction_spec_sha256=prediction_spec_sha256,
        execution_lock_sha256=execution_lock_sha256,
        producer_runtime_signature=payload["producer_runtime_signature"],
        branch_access_sha256=branch_access_sha256,
    )
    if any(payload.get(key) != value for key, value in expected.items() if key not in {"contract", "schema_version", "content_hash"}):
        raise ValueError("prediction arrays differ from sidecar")
    return envelope.sidecar, arrays


def build_prediction_manifest(
    *, finalist: Mapping[str, Any], shard_records: Sequence[Mapping[str, Any]],
    shard_arrays: Sequence[Mapping[str, np.ndarray]],
    selected_identity_digests: Sequence[str], prediction_spec_sha256: str,
    execution_lock_sha256: str, expected_source_partitions: Sequence[str],
) -> dict[str, Any]:
    row = _finalist_row(finalist)
    expected = tuple(str(value) for value in selected_identity_digests)
    if len(shard_records) != len(shard_arrays):
        raise ValueError("prediction manifest shard arrays/sidecars differ")
    by_partition: dict[str, tuple[Mapping[str, Any], Mapping[str, np.ndarray]]] = {}
    for shard, arrays in zip(shard_records, shard_arrays, strict=True):
        validate_content_hash(shard, expected_contract=PREDICTION_SHARD_CONTRACT)
        payload = shard.get("payload", shard)
        if payload["finalist"] != row:
            raise ValueError("prediction manifest mixes finalists")
        if payload.get("prediction_spec_sha256") != require_sha256(
            prediction_spec_sha256, name="prediction spec"
        ) or payload.get("execution_lock_sha256") != require_sha256(
            execution_lock_sha256, name="execution lock"
        ):
            raise ValueError("prediction shard lineage differs")
        partition = str(payload["source_partition"])
        if partition in by_partition:
            raise ValueError("prediction manifest repeats a source partition")
        identities = np.asarray(arrays.get("identity_digests"))
        logits = np.asarray(arrays.get("logits"))
        if (
            identities.dtype != np.uint8 or identities.ndim != 2
            or identities.shape[1:] != (32,)
        ):
            raise ValueError("prediction shard identity arrays differ from committed sidecar")
        identity_hex = [bytes(value).hex() for value in identities]
        if (
            logits.dtype != np.float32 or logits.shape != (len(identities), 15)
            or not np.isfinite(logits).all()
            or identity_hex != payload.get("identity_digests")
            or payload.get("array_sha256", {}).get("identity_digests")
            != logical_array_sha256("identity_digests", identities)
            or payload.get("array_sha256", {}).get("logits")
            != logical_array_sha256("logits", logits)
        ):
            raise ValueError("prediction shard arrays differ from committed sidecar")
        by_partition[partition] = (shard, arrays)
    partition_order = tuple(str(value) for value in expected_source_partitions)
    if (
        not partition_order or partition_order != tuple(sorted(partition_order))
        or len(partition_order) != len(set(partition_order))
        or set(by_partition) != set(partition_order)
    ):
        raise ValueError("prediction manifest source partition inventory differs")
    seen: list[str] = []
    identity_blocks: list[np.ndarray] = []
    logit_blocks: list[np.ndarray] = []
    shard_rows = []
    for partition in partition_order:
        shard, arrays = by_partition[partition]
        payload = shard.get("payload", shard)
        identity_blocks.append(np.asarray(arrays["identity_digests"]))
        logit_blocks.append(np.asarray(arrays["logits"]))
        seen.extend(payload["identity_digests"])
        shard_rows.append({
            "source_partition": partition, "sidecar_sha256": shard["content_hash"],
            "rows": int(payload["rows"]), "array_sha256": dict(payload["array_sha256"]),
        })
    if len(seen) != len(set(seen)) or tuple(sorted(seen)) != tuple(sorted(expected)):
        raise ValueError("prediction manifest coverage is incomplete, duplicate, or unexpected")
    joined_identities = np.concatenate(identity_blocks, axis=0)
    joined_logits = np.concatenate(logit_blocks, axis=0)
    return with_content_hash(
        {
            "contract": PREDICTION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "finalist": row,
            "prediction_spec_sha256": require_sha256(prediction_spec_sha256, name="prediction spec"),
            "execution_lock_sha256": require_sha256(execution_lock_sha256, name="execution lock"),
            "rows": len(seen),
            "identity_set_sha256": canonical_sha256(sorted(seen)),
            "identity_order_sha256": canonical_sha256(seen),
            "array_sha256": {
                "identity_digests": logical_array_sha256("identity_digests", joined_identities),
                "logits": logical_array_sha256("logits", joined_logits),
            },
            "shards": shard_rows,
            "shard_sha256": [row["sidecar_sha256"] for row in shard_rows],
            "complete_disjoint_coverage": True,
        }
    )


def locked_metric_join(
    *, label_escrow_sidecar: Mapping[str, Any], label_arrays: Mapping[str, np.ndarray],
    finalists: Sequence[Mapping[str, Any]], prediction_arrays: Mapping[str, Mapping[str, np.ndarray]],
    prediction_manifests: Mapping[str, Mapping[str, Any]], execution_lock: Mapping[str, Any],
    finalist_lock: Mapping[str, Any], prediction_spec: Mapping[str, Any],
    data_attestation: Mapping[str, Any], capability: Mapping[str, Any], task_id: str,
    execution_claim: Mapping[str, Any], task_registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    validate_content_hash(label_escrow_sidecar, expected_contract=FINAL_LABEL_ESCROW_CONTRACT)
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    validate_final_data_attestation(data_attestation)
    validate_execution_lock(
        execution_lock, finalist_lock=finalist_lock,
        data_attestation=data_attestation, claim=execution_claim,
        task_registry=task_registry, require_live_metric=True,
    )
    validate_prediction_spec(
        prediction_spec, finalist_lock=finalist_lock,
        execution_lock=execution_lock,
    )
    validate_role_capability(
        capability, execution_claim=execution_claim,
        task_registry=task_registry,
        expected_population_sha256=execution_lock["population_sha256"],
        expected_task_id=task_id, allowed_kinds=("metric_join",),
        expected_execution_lock_sha256=execution_lock["content_hash"],
        expected_branch_family="label_escrow",
    )
    if (
        execution_lock["finalist_lock_sha256"] != finalist_lock["content_hash"]
        or execution_lock["data_attestation_sha256"] != data_attestation["content_hash"]
        or prediction_spec["execution_lock_sha256"] != execution_lock["content_hash"]
        or prediction_spec["finalist_lock_sha256"] != finalist_lock["content_hash"]
        or data_attestation.get("label_escrow_sha256") != label_escrow_sidecar["content_hash"]
    ):
        raise ValueError("locked metric join lineage differs")
    normalized_finalists = [_finalist_row(row) for row in finalists]
    if normalized_finalists != finalist_lock["finalists"] or prediction_spec["finalists"] != finalist_lock["finalists"]:
        raise ValueError("locked metric join finalist registry differs")
    escrow_payload = label_escrow_sidecar.get("payload", label_escrow_sidecar)
    escrow_ids = np.asarray(label_arrays["identity_digests"])
    labels = np.asarray(label_arrays["labels"])
    if escrow_ids.dtype != np.uint8 or escrow_ids.ndim != 2 or escrow_ids.shape[1:] != (32,):
        raise ValueError("locked metric join escrow identities must be uint8 [rows,32]")
    if labels.dtype != np.uint8 or labels.shape != (len(escrow_ids),):
        raise ValueError("locked metric join label escrow differs")
    if (
        len(labels) != execution_lock.get("row_count")
        or np.bincount(labels.astype(np.int64), minlength=15).tolist()
        != execution_lock.get("class_counts")
    ):
        raise ValueError("locked metric join label class counts differ from execution lock")
    escrow_hex = [bytes(row).hex() for row in escrow_ids]
    if escrow_payload.get("array_sha256", {}).get("identity_digests") != logical_array_sha256(
        "identity_digests", escrow_ids
    ) or escrow_payload.get("array_sha256", {}).get("labels") != logical_array_sha256("labels", labels):
        raise ValueError("locked metric join escrow logical hash differs")
    if escrow_payload.get("selection_sha256") != data_attestation["selection_sha256"]:
        raise ValueError("locked metric join escrow selection differs")
    label_by_id = {identity: int(label) for identity, label in zip(escrow_hex, labels, strict=True)}
    if len(label_by_id) != len(labels):
        raise ValueError("locked metric join repeats an escrow identity")
    evaluations: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    joined_order_sha256: str | None = None
    for finalist in normalized_finalists:
        row = finalist
        finalist_id = row["finalist_id"]
        manifest = prediction_manifests.get(finalist_id)
        arrays = prediction_arrays.get(finalist_id)
        if manifest is None or arrays is None:
            raise ValueError("locked metric join lacks a finalist prediction")
        validate_content_hash(manifest, expected_contract=PREDICTION_MANIFEST_CONTRACT)
        if (
            manifest.get("finalist") != row
            or manifest.get("prediction_spec_sha256") != prediction_spec["content_hash"]
            or manifest.get("execution_lock_sha256") != execution_lock["content_hash"]
        ):
            raise ValueError("locked metric join prediction manifest lineage differs")
        identities = np.asarray(arrays["identity_digests"])
        logits = np.asarray(arrays["logits"])
        if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1:] != (32,):
            raise ValueError("locked metric join prediction identities differ")
        if logits.dtype != np.float32 or logits.shape != (len(identities), 15):
            raise ValueError("locked metric join accepts finite FP32 logits only")
        identity_hex = [bytes(item).hex() for item in identities]
        if not np.isfinite(logits).all() or set(identity_hex) != set(label_by_id):
            raise ValueError("locked metric join prediction coverage differs from escrow")
        if manifest.get("identity_set_sha256") != canonical_sha256(sorted(identity_hex)):
            raise ValueError("locked metric join prediction identity hash differs")
        order_sha256 = canonical_sha256(identity_hex)
        if (
            manifest.get("identity_order_sha256") != order_sha256
            or manifest.get("array_sha256", {}).get("identity_digests")
            != logical_array_sha256("identity_digests", identities)
            or manifest.get("array_sha256", {}).get("logits")
            != logical_array_sha256("logits", logits)
        ):
            raise ValueError("locked metric join arrays differ from prediction manifest")
        if joined_order_sha256 is None:
            joined_order_sha256 = order_sha256
        elif joined_order_sha256 != order_sha256:
            raise ValueError("paired final predictions use different identity order")
        joined_labels = np.asarray([label_by_id[identity] for identity in identity_hex], dtype=np.int64)
        metrics = classification_metrics(logits, joined_labels)
        for value in (
            metrics["cross_entropy"], metrics["macro_ovr_auc"],
            metrics["macro_mean_log_qcd_rejection_at_50pct_signal"],
        ):
            if value is None or not math.isfinite(float(value)):
                raise FloatingPointError("locked final metric is nonfinite")
        evaluations[finalist_id] = with_content_hash(
            {
                "contract": FINAL_EVALUATION_CONTRACT,
                "schema_version": 1,
                "finalist": row,
                "prediction_manifest_sha256": manifest["content_hash"],
                "label_escrow_sha256": label_escrow_sidecar["content_hash"],
                "execution_lock_sha256": execution_lock["content_hash"],
                "metrics": metrics,
                "metric_runtime_signature_sha256": execution_lock[
                    "metric_runtime_signature"
                ]["signature_sha256"],
                "test_used_for_selection": False,
                "joined_identity_order_sha256": order_sha256,
            }
        )
        manifest_hashes[finalist_id] = manifest["content_hash"]
    join = with_content_hash(
        {
            "contract": METRIC_JOIN_CONTRACT,
            "schema_version": 1,
            "execution_lock_sha256": execution_lock["content_hash"],
            "label_escrow_sha256": label_escrow_sidecar["content_hash"],
            "prediction_manifests": dict(sorted(manifest_hashes.items())),
            "evaluation_sha256": {
                key: value["content_hash"] for key, value in sorted(evaluations.items())
            },
            "single_label_join": True,
            "population_sha256": execution_lock["population_sha256"],
            "data_attestation_sha256": data_attestation["content_hash"],
            "finalist_lock_sha256": finalist_lock["content_hash"],
            "prediction_spec_sha256": prediction_spec["content_hash"],
            "metric_runtime_signature_sha256": execution_lock[
                "metric_runtime_signature"
            ]["signature_sha256"],
            "metric_join_capability_sha256": capability["content_hash"],
            "joined_identity_order_sha256": joined_order_sha256,
        }
    )
    return join, evaluations


def build_final_aggregate(
    *, metric_join: Mapping[str, Any], evaluations: Mapping[str, Mapping[str, Any]],
    finalist_lock: Mapping[str, Any], execution_lock: Mapping[str, Any],
    paired_bootstrap_envelopes: Sequence[Mapping[str, Any]],
    paired_comparison_registry: Sequence[Mapping[str, Any]],
    confirmation_aggregate_sha256: str,
) -> dict[str, Any]:
    """Bind point estimates and committed paired statistics without reselection."""

    validate_content_hash(metric_join, expected_contract=METRIC_JOIN_CONTRACT)
    validate_content_hash(finalist_lock, expected_contract=FINALIST_LOCK_CONTRACT)
    validate_content_hash(execution_lock, expected_contract=EXECUTION_LOCK_CONTRACT)
    if (
        metric_join.get("execution_lock_sha256") != execution_lock["content_hash"]
        or metric_join.get("finalist_lock_sha256") != finalist_lock["content_hash"]
        or finalist_lock.get("confirmation_aggregate_sha256") != require_sha256(
            confirmation_aggregate_sha256, name="confirmation aggregate"
        )
    ):
        raise ValueError("final aggregate lock/confirmation lineage differs")
    expected_ids = [row["finalist_id"] for row in finalist_lock["finalists"]]
    if set(evaluations) != set(expected_ids):
        raise ValueError("final aggregate evaluation registry is incomplete")
    evaluation_hashes = {}
    for finalist_id in expected_ids:
        report = evaluations[finalist_id]
        validate_content_hash(report, expected_contract=FINAL_EVALUATION_CONTRACT)
        if (
            report["finalist"]["finalist_id"] != finalist_id
            or report["execution_lock_sha256"] != execution_lock["content_hash"]
            or report["joined_identity_order_sha256"] != metric_join["joined_identity_order_sha256"]
        ):
            raise ValueError("final aggregate evaluation lineage differs")
        evaluation_hashes[finalist_id] = report["content_hash"]
    comparison_rows: dict[str, dict[str, Any]] = {}
    for raw in paired_comparison_registry:
        if not isinstance(raw, Mapping) or set(raw) != {
            "comparison_id", "left_id", "right_id", "sign",
        }:
            raise ValueError("paired comparison registry row differs")
        comparison_id = str(raw["comparison_id"])
        left_id = str(raw["left_id"])
        right_id = str(raw["right_id"])
        sign = str(raw["sign"])
        if (
            not comparison_id or comparison_id in comparison_rows
            or left_id not in expected_ids or right_id not in expected_ids
            or left_id == right_id or sign != "left_minus_right"
        ):
            raise ValueError("paired comparison registry identity/sign differs")
        comparison_rows[comparison_id] = {
            "comparison_id": comparison_id,
            "left_id": left_id,
            "right_id": right_id,
            "sign": sign,
        }
    if not comparison_rows:
        raise ValueError("paired comparison registry is empty")
    bootstrap_rows = []
    seen: set[str] = set()
    for record in paired_bootstrap_envelopes:
        sidecar = record.get("sidecar")
        commit = record.get("commit")
        if not isinstance(sidecar, Mapping) or not isinstance(commit, Mapping):
            raise ValueError("final aggregate requires committed bootstrap envelopes")
        validate_content_hash(sidecar, expected_contract=PAIRED_BOOTSTRAP_CONTRACT)
        validate_content_hash(commit, expected_contract=SHARED_BINARY_ENVELOPE_CONTRACT)
        payload = sidecar.get("payload", sidecar)
        if payload.get("scientific_authorization") is not True:
            raise PermissionError("nonauthoritative paired bootstrap entered final aggregate")
        if payload.get("joined_identity_order_sha256") != metric_join["joined_identity_order_sha256"]:
            raise ValueError("paired bootstrap joined population differs")
        comparison_id = str(payload.get("comparison_id", ""))
        if not comparison_id or comparison_id in seen:
            raise ValueError("paired bootstrap comparison inventory differs")
        seen.add(comparison_id)
        expected_comparison = comparison_rows.get(comparison_id)
        if expected_comparison is None or (
            payload.get("left_id") != expected_comparison["left_id"]
            or payload.get("right_id") != expected_comparison["right_id"]
        ):
            raise ValueError("paired bootstrap differs from the frozen comparison registry")
        commit_payload = commit.get("payload", {})
        if commit_payload.get("artifact_contract") != PAIRED_BOOTSTRAP_CONTRACT:
            raise ValueError("paired bootstrap commit contract differs")
        if (
            commit.get("parents") != sidecar.get("parents")
            or commit_payload.get("envelope_id") != sidecar.get("payload", {}).get("envelope_id")
            or commit_payload.get("envelope_owner_id")
            != sidecar.get("payload", {}).get("envelope_owner_id")
        ):
            raise ValueError("paired bootstrap envelope identity/parent lineage differs")
        sidecar_members = [
            row for row in commit_payload.get("members", ())
            if isinstance(row, Mapping) and row.get("path") == "sidecar.json"
        ]
        if len(sidecar_members) != 1 or sidecar_members[0].get("logical_sha256") != sidecar["content_hash"]:
            raise ValueError("paired bootstrap commit does not bind its sidecar")
        bootstrap_rows.append({
            "comparison_id": comparison_id,
            "sidecar_sha256": sidecar["content_hash"],
            "commit_sha256": commit["content_hash"],
        })
    if not bootstrap_rows:
        raise ValueError("final aggregate lacks paired bootstrap envelopes")
    if seen != set(comparison_rows):
        raise ValueError("paired bootstrap comparison inventory is incomplete")
    return with_content_hash({
        "contract": FINAL_AGGREGATE_CONTRACT, "schema_version": 1,
        "population_sha256": execution_lock["population_sha256"],
        "finalist_lock_sha256": finalist_lock["content_hash"],
        "execution_lock_sha256": execution_lock["content_hash"],
        "metric_join_sha256": metric_join["content_hash"],
        "confirmation_aggregate_sha256": confirmation_aggregate_sha256,
        "evaluation_sha256": evaluation_hashes,
        "paired_bootstrap_envelopes": sorted(bootstrap_rows, key=lambda row: row["comparison_id"]),
        "paired_comparison_registry": [
            comparison_rows[key] for key in sorted(comparison_rows)
        ],
        "paired_comparison_registry_sha256": canonical_sha256(
            [comparison_rows[key] for key in sorted(comparison_rows)]
        ),
        "joined_identity_order_sha256": metric_join["joined_identity_order_sha256"],
        "test_used_for_selection": False,
        "complete_parent_union_evaluated": True,
    })


__all__ = [name for name in globals() if name.endswith("_CONTRACT")] + [
    "REPRESENTATION_ENDPOINTS",
    "build_assignment_audit",
    "build_execution_lock",
    "build_final_data_attestation",
    "build_finalist_lock",
    "build_metric_runtime_signature",
    "build_prediction_manifest",
    "build_prediction_spec",
    "build_final_aggregate",
    "load_prediction_shard",
    "locked_metric_join",
    "publish_paired_bootstrap_envelope",
    "publish_prediction_shard",
    "prediction_shard_sidecar",
    "validate_assignment_audit",
    "validate_execution_lock",
    "validate_final_data_attestation",
    "validate_metric_runtime_signature",
    "validate_prediction_runtime_signature",
    "validate_prediction_spec",
]
