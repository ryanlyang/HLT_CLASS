"""Canonical pre-final commitments for the HCWDL-RKD campaign.

These artifacts remove two otherwise free operator hashes from the shared
final reservation.  Both are deterministic projections of authenticated
inputs and are independently recomputed by the workers that consume them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    require_sha256, validate_content_hash, with_content_hash,
)
from .hcwdl_representation_contracts import (
    FINAL_ASSIGNMENT_SPEC_CONTRACT, PRETRAINING_FINALIST_POLICY_CONTRACT,
)


REPRESENTATION_ENDPOINTS: Final = (
    "RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w",
)
CHECKPOINT_SELECTOR: Final = (
    "highest_macro_ovr_auc", "lowest_cross_entropy",
    "highest_macro_mean_log_qcd_rejection_at_50pct_signal",
    "earliest_optimizer_update", "lexicographically_smallest_checkpoint_identity",
)
PAIRED_SCREEN_PARENT_FINALISTS: Final = ("M0", "M1c", "M1w")
REQUIRED_LOCK_NODE_IDS: Final = (
    "D100", "M0", "M6c", "M6w", "NULL_M1_SELF_KD",
    "NULL_M6_PREDECESSOR_ONLY", "TOFF",
)
_PARENT_ROW_FIELDS: Final = frozenset({
    "finalist_id", "checkpoint_sha256", "report_sha256", "domain",
    "deployable", "extraction_sha256", "source_campaign", "screening_seed",
    "execution_id", "checkpoint_selection_sha256", "model_kind", "model_relative",
})


def _parent_rows(
    parent_finalists: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(parent_finalists, Sequence) or isinstance(
        parent_finalists, (str, bytes),
    ) or not parent_finalists:
        raise ValueError("parent finalist policy is empty")
    rows = []
    seen: set[str] = set()
    for raw in parent_finalists:
        if not isinstance(raw, Mapping) or set(raw) != _PARENT_ROW_FIELDS:
            raise ValueError("parent finalist policy row schema differs")
        row = dict(raw)
        finalist_id = str(row["finalist_id"])
        domain = str(row["domain"])
        deployable = row["deployable"]
        seed = row["screening_seed"]
        if (
            not finalist_id or finalist_id in seen
            or finalist_id in REPRESENTATION_ENDPOINTS
            or row["source_campaign"] == "representation"
            or domain not in {"hlt", "shell_exact_d100", "native_offline"}
            or not isinstance(deployable, bool)
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("parent finalist policy identity differs")
        seen.add(finalist_id)
        for name in ("checkpoint_sha256", "report_sha256"):
            require_sha256(row[name], name=f"{finalist_id} {name}")
        extraction = row["extraction_sha256"]
        if deployable:
            require_sha256(extraction, name=f"{finalist_id} extraction")
            if domain != "hlt":
                raise ValueError("deployable parent finalist must consume HLT")
        elif extraction is not None:
            raise ValueError("nondeployable parent finalist claims an extraction")
        for name in ("execution_id", "checkpoint_selection_sha256"):
            if row[name] is not None:
                require_sha256(row[name], name=f"{finalist_id} {name}")
        if row["model_kind"] not in {"pmard", "hcwdl"}:
            raise ValueError("parent finalist model kind differs")
        relative = row["model_relative"]
        if not isinstance(relative, str) or not relative:
            raise ValueError("parent finalist model member differs")
        rows.append(row)
    rows.sort(key=lambda row: str(row["finalist_id"]))
    if not {"D100", "M0", "M1c", "M1w", "TOFF"} <= seen:
        raise ValueError("parent finalist registry omits a mandatory parent endpoint")
    return rows


def _parent_lock_rows(parent_finalist_lock: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Validate and return every row authorized by the real parent finalist lock."""

    from .hcwdl_locks import validate_lock

    lock_sha256 = validate_lock(parent_finalist_lock, expected_level="finalist")
    payload = parent_finalist_lock.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {
        "confirmation_report_sha256", "finalists", "selection_used_validation_only",
    } or payload["selection_used_validation_only"] is not True:
        raise ValueError("parent HCWDL finalist-lock payload differs")
    require_sha256(
        payload["confirmation_report_sha256"], name="parent confirmation aggregate",
    )
    raw_rows = payload["finalists"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("parent HCWDL finalist lock is empty")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "node_id", "seed", "checkpoint_sha256", "report_sha256", "report_path",
        }:
            raise ValueError("parent HCWDL finalist-lock row schema differs")
        node_id = str(raw["node_id"])
        seed = int(raw["seed"])
        if not node_id or isinstance(raw["seed"], bool) or seed < 0:
            raise ValueError("parent HCWDL finalist identity differs")
        identity = (node_id, seed)
        if identity in identities:
            raise ValueError("parent HCWDL finalist lock repeats a node/seed")
        identities.add(identity)
        counts[node_id] = counts.get(node_id, 0) + 1
        rows.append({
            "node_id": node_id, "seed": seed,
            "checkpoint_sha256": require_sha256(
                raw["checkpoint_sha256"], name=f"{node_id}/{seed} checkpoint",
            ),
            "report_sha256": require_sha256(
                raw["report_sha256"], name=f"{node_id}/{seed} report",
            ),
            "report_path": str(raw["report_path"]),
        })
    if not set(REQUIRED_LOCK_NODE_IDS) <= set(counts):
        raise ValueError("parent HCWDL finalist lock omits a required endpoint/control")
    for row in rows:
        row["finalist_id"] = (
            f"{row['node_id']}__seed{row['seed']}"
            if counts[row["node_id"]] > 1 else row["node_id"]
        )
    return lock_sha256, rows


def _member(
    value: object, *, name: str, checkpoint_sha256: str | None = None,
    report_sha256: str | None = None,
) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"relative", "report"}:
        raise ValueError(f"{name} authenticated model member differs")
    relative = value["relative"]
    report = value["report"]
    if (
        not isinstance(relative, str) or not relative or relative.startswith(("/", "\\"))
        or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))
        or not isinstance(report, Mapping)
    ):
        raise ValueError(f"{name} authenticated model member differs")
    contract = report.get("contract")
    schema_version = report.get("schema_version")
    if not isinstance(contract, str) or not isinstance(schema_version, int):
        raise ValueError(f"{name} report lacks a versioned contract")
    digest = validate_content_hash(
        report, expected_contract=contract, expected_schema_version=schema_version,
    )
    selected = require_sha256(
        report.get("selected_checkpoint_sha256"), name=f"{name} selected checkpoint",
    )
    if report_sha256 is not None and digest != report_sha256:
        raise ValueError(f"{name} report differs from the parent finalist lock")
    if checkpoint_sha256 is not None and selected != checkpoint_sha256:
        raise ValueError(f"{name} checkpoint differs from the parent finalist lock")
    return relative, report


def project_parent_finalists(
    *, parent_finalist_lock: Mapping[str, Any],
    locked_model_members: Mapping[str, Any],
    paired_screen_model_members: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project the complete lock plus three authenticated seed-1337 comparators."""

    _, lock_rows = _parent_lock_rows(parent_finalist_lock)
    expected_member_keys = {
        f"{row['node_id']}:{row['seed']}" for row in lock_rows
    }
    if not isinstance(locked_model_members, Mapping) or set(locked_model_members) != expected_member_keys:
        raise ValueError("parent finalist model members omit or add a lock row")
    if not isinstance(paired_screen_model_members, Mapping) or set(
        paired_screen_model_members
    ) != set(PAIRED_SCREEN_PARENT_FINALISTS):
        raise ValueError("parent screening comparator registry differs")
    rows: list[dict[str, Any]] = []
    for lock_row in lock_rows:
        node_id = lock_row["node_id"]
        relative, _ = _member(
            locked_model_members[f"{node_id}:{lock_row['seed']}"],
            name=f"{node_id}/{lock_row['seed']}",
            checkpoint_sha256=lock_row["checkpoint_sha256"],
            report_sha256=lock_row["report_sha256"],
        )
        oracle = node_id in {"D100", "TOFF"}
        rows.append({
            "finalist_id": lock_row["finalist_id"],
            "checkpoint_sha256": lock_row["checkpoint_sha256"],
            "report_sha256": lock_row["report_sha256"],
            "domain": (
                "shell_exact_d100" if node_id == "D100"
                else "native_offline" if node_id == "TOFF" else "hlt"
            ),
            "deployable": not oracle,
            # A parent PMARD report is its authenticated selected-checkpoint
            # extraction record; the raw checkpoint bytes are verified on load.
            "extraction_sha256": None if oracle else lock_row["report_sha256"],
            "source_campaign": "parent", "screening_seed": lock_row["seed"],
            "execution_id": None, "checkpoint_selection_sha256": None,
            "model_kind": "pmard", "model_relative": relative,
        })
    from .training import derive_seed
    for node_id in PAIRED_SCREEN_PARENT_FINALISTS:
        relative, report = _member(
            paired_screen_model_members[node_id], name=f"{node_id}/seed1337",
        )
        config = report.get("config")
        if (
            not isinstance(config, Mapping)
            or config.get("master_seed") != derive_seed(1337, f"hcwdl/{node_id}")
        ):
            raise ValueError(f"{node_id} comparator is not the parent screening-seed run")
        rows.append({
            "finalist_id": node_id,
            "checkpoint_sha256": require_sha256(
                report["selected_checkpoint_sha256"], name=f"{node_id} checkpoint",
            ),
            "report_sha256": require_sha256(
                report["content_hash"], name=f"{node_id} report",
            ),
            "domain": "hlt", "deployable": True,
            "extraction_sha256": report["content_hash"],
            "source_campaign": "parent", "screening_seed": 1337,
            "execution_id": None, "checkpoint_selection_sha256": None,
            "model_kind": "pmard", "model_relative": relative,
        })
    return _parent_rows(rows)


def build_final_assignment_spec(
    *, matcher_resources: Mapping[str, Any], source_partitions: Sequence[str],
    step_size: int = 8192,
) -> dict[str, Any]:
    """Bind the exact canonical matcher and final-test sharding semantics."""

    from .highcov_resources import RESOURCE_CONTRACT, resource_validation_report

    matcher_sha256 = validate_content_hash(
        matcher_resources, expected_contract=RESOURCE_CONTRACT,
        expected_schema_version=1,
    )
    if dict(matcher_resources) != resource_validation_report():
        raise PermissionError("final assignment matcher resources are not canonical")
    partitions = [str(value) for value in source_partitions]
    if (
        not partitions or partitions != sorted(partitions)
        or len(partitions) != len(set(partitions))
    ):
        raise ValueError("final assignment source partitions differ")
    if isinstance(step_size, bool) or step_size != 8192:
        raise ValueError("final assignment streaming step differs")
    return with_content_hash({
        "contract": FINAL_ASSIGNMENT_SPEC_CONTRACT, "schema_version": 1,
        "matcher_resources_sha256": matcher_sha256,
        "matcher_semantic_hashes": dict(matcher_resources["semantic_hashes"]),
        "repair_family": "HIGHCOV_SHELL_EXACT/v1", "alpha": 1.0,
        "role": "final_test", "label_projection_allowed": False,
        "source_partitions": partitions,
        "shard_policy": "one_shard_per_authenticated_final_split_source/v1",
        "selection_policy": "shared_population_class_stratified_identity_rank/v1",
        "streaming_step_size": 8192,
    })


def validate_final_assignment_spec(
    value: Mapping[str, Any], *, matcher_resources: Mapping[str, Any],
    source_partitions: Sequence[str], step_size: int = 8192,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=FINAL_ASSIGNMENT_SPEC_CONTRACT,
        expected_schema_version=1,
    )
    if dict(value) != build_final_assignment_spec(
        matcher_resources=matcher_resources, source_partitions=source_partitions,
        step_size=step_size,
    ):
        raise ValueError("final assignment specification is not canonical")
    return digest


def build_pretraining_finalist_policy_commitment(
    *, parent_finalists: Sequence[Mapping[str, Any]],
    parent_finalist_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit the complete parent union and four fixed seed-1337 endpoints."""

    rows = _parent_rows(parent_finalists)
    lock_sha256, lock_rows = _parent_lock_rows(parent_finalist_lock)
    expected_lock_rows = {
        str(row["finalist_id"]): row for row in lock_rows
    }
    actual_rows = {str(row["finalist_id"]): row for row in rows}
    if set(actual_rows) != set(expected_lock_rows) | set(PAIRED_SCREEN_PARENT_FINALISTS):
        raise ValueError("parent finalist projection omits or adds an authorized row")
    for finalist_id, raw in expected_lock_rows.items():
        row = actual_rows[finalist_id]
        node_id = str(raw["node_id"])
        oracle = node_id in {"D100", "TOFF"}
        expected = {
            "checkpoint_sha256": raw["checkpoint_sha256"],
            "report_sha256": raw["report_sha256"],
            "domain": (
                "shell_exact_d100" if node_id == "D100"
                else "native_offline" if node_id == "TOFF" else "hlt"
            ),
            "deployable": not oracle,
            "extraction_sha256": None if oracle else raw["report_sha256"],
            "source_campaign": "parent", "screening_seed": raw["seed"],
            "execution_id": None, "checkpoint_selection_sha256": None,
            "model_kind": "pmard",
        }
        if any(row[name] != value for name, value in expected.items()):
            raise ValueError("parent finalist row differs from the real parent lock")
    for finalist_id in PAIRED_SCREEN_PARENT_FINALISTS:
        row = actual_rows[finalist_id]
        if any((
            row["screening_seed"] != 1337, row["domain"] != "hlt",
            row["deployable"] is not True, row["source_campaign"] != "parent",
            row["model_kind"] != "pmard",
        )):
            raise ValueError("paired parent screening comparator differs")
    return with_content_hash({
        "contract": PRETRAINING_FINALIST_POLICY_CONTRACT, "schema_version": 1,
        "parent_finalist_registry_sha256": require_sha256(
            lock_sha256, name="parent finalist registry",
        ),
        "parent_campaign_sha256": parent_finalist_lock["campaign_spec_sha256"],
        "parent_confirmation_lock_sha256": parent_finalist_lock["parent_lock_sha256"],
        "parent_confirmation_report_sha256": parent_finalist_lock["payload"][
            "confirmation_report_sha256"
        ],
        "parent_finalists": rows,
        "representation_endpoints": [
            {"finalist_id": finalist_id, "screening_seed": 1337}
            for finalist_id in REPRESENTATION_ENDPOINTS
        ],
        "checkpoint_selector": list(CHECKPOINT_SELECTOR),
        "confirmation_seeds_are_not_finalists": True,
        "parent_union_may_not_be_pruned": True,
    })


def validate_pretraining_finalist_policy_commitment(
    value: Mapping[str, Any], *, parent_finalists: Sequence[Mapping[str, Any]],
    parent_finalist_lock: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        value, expected_contract=PRETRAINING_FINALIST_POLICY_CONTRACT,
        expected_schema_version=1,
    )
    if dict(value) != build_pretraining_finalist_policy_commitment(
        parent_finalists=parent_finalists,
        parent_finalist_lock=parent_finalist_lock,
    ):
        raise ValueError("pretraining finalist policy is not canonical")
    return digest


__all__ = [
    "CHECKPOINT_SELECTOR", "FINAL_ASSIGNMENT_SPEC_CONTRACT",
    "PAIRED_SCREEN_PARENT_FINALISTS", "PRETRAINING_FINALIST_POLICY_CONTRACT",
    "REPRESENTATION_ENDPOINTS", "REQUIRED_LOCK_NODE_IDS",
    "build_final_assignment_spec", "project_parent_finalists",
    "build_pretraining_finalist_policy_commitment",
    "validate_final_assignment_spec",
    "validate_pretraining_finalist_policy_commitment",
]
