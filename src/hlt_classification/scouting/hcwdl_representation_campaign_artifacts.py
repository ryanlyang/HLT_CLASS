"""Small campaign-gate artifacts that make the HCWDL-RKD DAG executable.

These artifacts deliberately contain no scientific model selection.  They
bind implementation controls and short-lived cache lifecycle evidence to
their actual immutable parents so that a definition in the plan cannot be
mistaken for an executed campaign gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_representation_contracts import (
    CACHE_MINIATURE_CONTRACT,
    CACHE_MINIATURE_BANK_CONTRACT,
    ZERO_COEFFICIENT_ACCEPTANCE_CONTRACT,
)


ZERO_TOLERANCES: Final = {
    "logits_max_abs": 1.0e-6,
    "base_loss_max_abs": 1.0e-7,
    "shared_gradient_max_abs": 1.0e-6,
    "optimizer_state_max_abs": 1.0e-6,
}


def build_cache_miniature_bank_evidence(
    *, bank_kind: str, logical_bank_sha256: str, generation_id: str,
    generation_manifest_sha256: str, rows: int, bounded_row_limit: int,
    identity_join_rows: int, loaded_array_logical_sha256: str,
    ram_bytes: int,
) -> dict[str, Any]:
    if bank_kind not in {"ordinary", "toff"}:
        raise ValueError("cache-miniature bank kind differs")
    if (
        isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
        or isinstance(bounded_row_limit, bool)
        or not isinstance(bounded_row_limit, int)
        or not rows <= bounded_row_limit <= 4096
        or identity_join_rows != rows
        or isinstance(ram_bytes, bool) or not isinstance(ram_bytes, int)
        or ram_bytes <= 0
    ):
        raise ValueError("cache-miniature bank coverage/resource evidence differs")
    return with_content_hash({
        "contract": CACHE_MINIATURE_BANK_CONTRACT,
        "schema_version": 1,
        "bank_kind": bank_kind,
        "logical_bank_sha256": require_sha256(
            logical_bank_sha256, name="cache miniature logical bank",
        ),
        "generation_id": require_sha256(
            generation_id, name="cache miniature generation",
        ),
        "generation_manifest_sha256": require_sha256(
            generation_manifest_sha256, name="cache miniature manifest",
        ),
        "rows": rows,
        "bounded_row_limit": bounded_row_limit,
        "ram_loaded": True,
        "identity_join_rows": identity_join_rows,
        "loaded_array_logical_sha256": require_sha256(
            loaded_array_logical_sha256, name="cache miniature loaded arrays",
        ),
        "ram_bytes": ram_bytes,
        "scientific_authorization": False,
        "training_consumer_authorized": False,
        "final_role_accessed": False,
    })


def validate_cache_miniature_bank_evidence(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=CACHE_MINIATURE_BANK_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_cache_miniature_bank_evidence(
        bank_kind=value["bank_kind"],
        logical_bank_sha256=value["logical_bank_sha256"],
        generation_id=value["generation_id"],
        generation_manifest_sha256=value["generation_manifest_sha256"],
        rows=value["rows"], bounded_row_limit=value["bounded_row_limit"],
        identity_join_rows=value["identity_join_rows"],
        loaded_array_logical_sha256=value["loaded_array_logical_sha256"],
        ram_bytes=value["ram_bytes"],
    )
    if dict(value) != rebuilt:
        raise ValueError("cache-miniature bank evidence semantics differ")
    if manifest is not None:
        payload = manifest.get("payload", {})
        consumers = payload.get("authorized_consumers")
        if (
            manifest.get("content_hash") != value["generation_manifest_sha256"]
            or payload.get("generation_id") != value["generation_id"]
            or payload.get("logical_target_sha256")
            != value["loaded_array_logical_sha256"]
            or int(payload.get("rows", -1)) != value["rows"]
            or payload.get("bank_kind") != value["bank_kind"]
            or not isinstance(consumers, list) or len(consumers) != 1
            or consumers[0].get("strategy") != "CACHE_MINIATURE"
            or consumers[0].get("execution_identity_payload", {}).get(
                "bounded_row_limit"
            ) != value["bounded_row_limit"]
        ):
            raise ValueError("cache-miniature bank manifest lineage differs")
    return digest


def build_zero_coefficient_acceptance(
    *,
    architecture_attestation_sha256: str,
    parent_loss_attestation_sha256: str,
    representation_recipe_sha256: str,
    runtime_signature_sha256: str,
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a real wrapper/base parity measurement to the frozen tolerances."""

    required = {
        "logits_max_abs", "base_loss_max_abs", "shared_gradient_max_abs",
        "optimizer_state_max_abs", "ce_equal", "hlt_kd_equal",
        "privileged_kd_equal", "shared_parameter_names_equal",
        "representation_heads_have_no_logit_path", "rng_state_equal",
        "trimmer_progression_equal", "optimizer_update_equal",
        "installed_weaver", "normal_training_trimming",
    }
    if not isinstance(measurements, Mapping) or set(measurements) != required:
        raise ValueError("zero-coefficient measurement registry differs")
    for name, tolerance in ZERO_TOLERANCES.items():
        raw = measurements[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"zero-coefficient measurement {name} differs")
        if not 0.0 <= float(raw) <= tolerance:
            raise ValueError(f"zero-coefficient measurement {name} exceeds tolerance")
    boolean_fields = required - set(ZERO_TOLERANCES)
    if any(measurements[name] is not True for name in boolean_fields):
        raise ValueError("zero-coefficient parity evidence is incomplete")
    return with_content_hash({
        "contract": ZERO_COEFFICIENT_ACCEPTANCE_CONTRACT,
        "schema_version": 1,
        "architecture_attestation_sha256": require_sha256(
            architecture_attestation_sha256, name="zero-coefficient architecture",
        ),
        "parent_loss_attestation_sha256": require_sha256(
            parent_loss_attestation_sha256, name="zero-coefficient parent loss",
        ),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="zero-coefficient recipe",
        ),
        "runtime_signature_sha256": require_sha256(
            runtime_signature_sha256, name="zero-coefficient runtime",
        ),
        "tolerances": dict(ZERO_TOLERANCES),
        "measurements": dict(measurements),
        "rho_representation": 0.0,
        "authorizes_scientific_training": True,
        "final_role_accessed": False,
    })


def validate_zero_coefficient_acceptance(
    value: Mapping[str, Any], *,
    architecture_attestation_sha256: str | None = None,
    parent_loss_attestation_sha256: str | None = None,
    representation_recipe_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=ZERO_COEFFICIENT_ACCEPTANCE_CONTRACT,
        expected_schema_version=1,
    )
    rebuilt = build_zero_coefficient_acceptance(
        architecture_attestation_sha256=value["architecture_attestation_sha256"],
        parent_loss_attestation_sha256=value["parent_loss_attestation_sha256"],
        representation_recipe_sha256=value["representation_recipe_sha256"],
        runtime_signature_sha256=value["runtime_signature_sha256"],
        measurements=value["measurements"],
    )
    if dict(value) != rebuilt or value.get("tolerances") != ZERO_TOLERANCES:
        raise ValueError("zero-coefficient acceptance semantics differ")
    expected = {
        "architecture_attestation_sha256": architecture_attestation_sha256,
        "parent_loss_attestation_sha256": parent_loss_attestation_sha256,
        "representation_recipe_sha256": representation_recipe_sha256,
    }
    if any(wanted is not None and value[name] != wanted for name, wanted in expected.items()):
        raise ValueError("zero-coefficient acceptance lineage differs")
    return digest


def build_cache_miniature_acceptance(
    *, representation_recipe_sha256: str,
    runtime_signature_sha256: str,
    bank_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require ordinary and TOFF build/load/join/cleanup lifecycle evidence."""

    rows = [dict(row) for row in bank_rows]
    if [row.get("bank_kind") for row in rows] != ["ordinary", "toff"]:
        raise ValueError("cache miniature must contain ordered ordinary and TOFF rows")
    required = {
        "bank_kind", "logical_bank_sha256", "generation_id",
        "generation_manifest_sha256", "rows", "bounded_row_limit",
        "ram_loaded", "identity_join_rows", "loaded_array_logical_sha256",
        "cleanup_authorization_sha256", "cleanup_completion_sha256",
        "committed_payload_absent_after_cleanup", "scientific_authorization",
    }
    for row in rows:
        if set(row) != required:
            raise ValueError("cache-miniature bank evidence fields differ")
        count = row["rows"]
        limit = row["bounded_row_limit"]
        joined = row["identity_join_rows"]
        if (
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or isinstance(limit, bool) or not isinstance(limit, int)
            or not count <= limit <= 4096
            or joined != count
        ):
            raise ValueError("cache-miniature bounded coverage differs")
        for name in (
            "logical_bank_sha256", "generation_id", "generation_manifest_sha256",
            "loaded_array_logical_sha256", "cleanup_authorization_sha256",
            "cleanup_completion_sha256",
        ):
            require_sha256(row[name], name=f"cache miniature {name}")
        if (
            row["ram_loaded"] is not True
            or row["committed_payload_absent_after_cleanup"] is not True
            or row["scientific_authorization"] is not False
        ):
            raise ValueError("cache-miniature lifecycle evidence is incomplete")
    return with_content_hash({
        "contract": CACHE_MINIATURE_CONTRACT,
        "schema_version": 1,
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="cache miniature recipe",
        ),
        "runtime_signature_sha256": require_sha256(
            runtime_signature_sha256, name="cache miniature runtime",
        ),
        "banks": rows,
        "ordinary_and_toff_built": True,
        "ordinary_and_toff_ram_loaded_and_joined": True,
        "ordinary_and_toff_cleanup_completed": True,
        "bounded_rows_only": True,
        "authorizes_scientific_training": True,
        "final_role_accessed": False,
    })


def validate_cache_miniature_acceptance(
    value: Mapping[str, Any], *, representation_recipe_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=CACHE_MINIATURE_CONTRACT, expected_schema_version=1,
    )
    rebuilt = build_cache_miniature_acceptance(
        representation_recipe_sha256=value["representation_recipe_sha256"],
        runtime_signature_sha256=value["runtime_signature_sha256"],
        bank_rows=value["banks"],
    )
    if dict(value) != rebuilt:
        raise ValueError("cache-miniature acceptance semantics differ")
    if (
        representation_recipe_sha256 is not None
        and value["representation_recipe_sha256"] != representation_recipe_sha256
    ):
        raise ValueError("cache-miniature recipe lineage differs")
    return digest


__all__ = [
    "ZERO_TOLERANCES", "build_cache_miniature_acceptance",
    "build_cache_miniature_bank_evidence", "build_zero_coefficient_acceptance",
    "validate_cache_miniature_acceptance",
    "validate_cache_miniature_bank_evidence",
    "validate_zero_coefficient_acceptance",
]
