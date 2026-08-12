"""Explicit human authorization to carry v1 operational smoke evidence into v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_homotopy_contracts import (
    ENDPOINT_LOCK_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT,
    OPERATIONAL_WAIVER_CONTRACT, WEAVER_PARITY_CONTRACT,
)
from .hcwdl_homotopy_graph import GRAPH_SHA256
from .hcwdl_homotopy_locks import (
    validate_endpoint_equality_lock, validate_graph_recipe_lock,
)


AUTHORIZATION_PHRASE: Final = (
    "AUTHORIZE HCWDL UJ V1 OPERATIONAL EVIDENCE FOR THINNED V2 PILOT"
)
V1_COMPLETION_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_CAMPAIGN_COMPLETE/v1"
V1_FIT_COUNT: Final = 80
REQUIRED_V2_TASKS: Final = (
    "authenticate", "upper_calibration", "train_base", "train_base_manifest",
    "switch_calibration", "train_switch", "validation_base",
    "validation_base_manifest", "validation_switch", "train_manifest",
    "validation_manifest", "coupling_audit", "coupling_lock",
    "cache_miniature", "endpoint_equality_lock", "toff_target_shards",
    "toff_target_manifest", "toff_target_lock", "graph_recipe_lock",
)
AUTHORIZED_REQUESTS: Final = {
    "cpu_calibration": {"cpus": 8, "memory": "64G", "walltime": "02:00:00", "gpu": None},
    "cpu_coupling": {"cpus": 8, "memory": "96G", "walltime": "04:00:00", "gpu": None},
    "cpu_finalize": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
    "gpu_targets": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "gpu_training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "cpu_report": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
}


def _commit(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError("HCWDL-UJ waiver source commit differs")
    return text


def build_operational_waiver(
    *, v1_campaign_spec: Mapping[str, Any],
    v1_campaign_completion: Mapping[str, Any],
    v2_campaign_spec: Mapping[str, Any],
    v2_endpoint_lock: Mapping[str, Any],
    v2_graph_recipe_lock: Mapping[str, Any],
    v2_weaver_parity: Mapping[str, Any],
    completed_v2_task_ids: Sequence[str],
    authorized_source_commit: str,
    authorized_semantic_source_sha256: Mapping[str, str],
    authorization_phrase: str,
) -> dict[str, Any]:
    if authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("HCWDL-UJ operational waiver phrase differs")
    v1_spec_hash = validate_content_hash(
        v1_campaign_spec, expected_contract="HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v1",
        expected_schema_version=1,
    )
    v1_completion_hash = validate_content_hash(
        v1_campaign_completion, expected_contract=V1_COMPLETION_CONTRACT,
        expected_schema_version=1,
    )
    v1_tasks = v1_campaign_spec.get("tasks")
    if (
        v1_campaign_spec.get("mode") != "smoke"
        or v1_campaign_spec.get("role_counts")
           != {"train": 4096, "validation": 4096, "final_test": 0}
        or v1_campaign_spec.get("final_test_accessed") is not False
        or not isinstance(v1_tasks, list)
        or sum(row.get("kind") == "train_node" for row in v1_tasks) != V1_FIT_COUNT
        or v1_campaign_completion.get("campaign_spec_sha256") != v1_spec_hash
        or v1_campaign_completion.get("fit_count") != V1_FIT_COUNT
        or v1_campaign_completion.get("mode") != "smoke"
        or v1_campaign_completion.get("validation_only") is not True
        or v1_campaign_completion.get("final_test_accessed") is not False
    ):
        raise ValueError("completed v1 HCWDL-UJ smoke evidence differs")

    v2_spec_hash = validate_content_hash(
        v2_campaign_spec, expected_contract="HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v2",
        expected_schema_version=1,
    )
    if (
        v2_campaign_spec.get("mode") != "smoke"
        or v2_campaign_spec.get("graph_sha256") != GRAPH_SHA256
        or v2_campaign_spec.get("role_counts")
           != {"train": 4096, "validation": 4096, "final_test": 0}
        or v2_campaign_spec.get("final_test_accessed") is not False
    ):
        raise ValueError("partial v2 HCWDL-UJ smoke evidence differs")
    endpoint_hash = validate_endpoint_equality_lock(
        v2_endpoint_lock, campaign_spec_sha256=v2_spec_hash,
    )
    graph_lock_hash = validate_graph_recipe_lock(
        v2_graph_recipe_lock, campaign_spec_sha256=v2_spec_hash,
    )
    parity_hash = validate_content_hash(
        v2_weaver_parity, expected_contract=WEAVER_PARITY_CONTRACT,
        expected_schema_version=1,
    )
    if (
        v2_graph_recipe_lock.get("endpoint_equality_lock_sha256") != endpoint_hash
        or v2_graph_recipe_lock.get("weaver_parity_sha256") != parity_hash
        or v2_graph_recipe_lock.get("graph_semantic_sha256") != GRAPH_SHA256
        or v2_weaver_parity.get("source_commit") != v2_campaign_spec.get("source_commit")
        or v2_weaver_parity.get("device") != "cuda"
        or v2_weaver_parity.get("unified_factory", {}).get("passed") is not True
        or v2_weaver_parity.get("native_teacher_factory", {}).get("passed") is not True
        or v2_weaver_parity.get("final_test_accessed") is not False
    ):
        raise ValueError("v2 production-worker acceptance evidence differs")
    semantic = v2_campaign_spec.get("semantic_source_sha256")
    if not isinstance(semantic, Mapping) or any(
        not isinstance(name, str)
        or require_sha256(value, name=f"v2 semantic source {name}") != value
        for name, value in semantic.items()
    ):
        raise ValueError("v2 smoke semantic-source lineage differs")
    completed = sorted(set(map(str, completed_v2_task_ids)))
    if not set(REQUIRED_V2_TASKS) <= set(completed):
        raise ValueError("v2 operational waiver lacks required completed tasks")
    return with_content_hash({
        "contract": OPERATIONAL_WAIVER_CONTRACT,
        "schema_version": 1,
        "authorization_phrase": authorization_phrase,
        "human_decision": (
            "carry_completed_v1_production_worker_smoke_and_completed_v2_"
            "infrastructure_acceptance_into_thinned_v2_300k_pilot"
        ),
        "v1_campaign_spec_sha256": v1_spec_hash,
        "v1_campaign_completion_sha256": v1_completion_hash,
        "v1_graph_sha256": require_sha256(
            v1_campaign_spec.get("graph_sha256"), name="v1 graph SHA-256",
        ),
        "v1_fit_count": V1_FIT_COUNT,
        "v2_smoke_campaign_spec_sha256": v2_spec_hash,
        "v2_graph_sha256": GRAPH_SHA256,
        "v2_endpoint_equality_lock_sha256": endpoint_hash,
        "v2_graph_recipe_lock_sha256": graph_lock_hash,
        "v2_weaver_parity_sha256": parity_hash,
        "completed_v2_task_ids": completed,
        "required_v2_task_ids": list(REQUIRED_V2_TASKS),
        "v2_smoke_source_commit": _commit(v2_campaign_spec["source_commit"]),
        "v2_smoke_semantic_source_sha256": dict(sorted(semantic.items())),
        "source_commit": _commit(authorized_source_commit),
        "semantic_source_sha256": {
            str(name): require_sha256(value, name=f"semantic source {name}")
            for name, value in sorted(authorized_semantic_source_sha256.items())
        },
        "authorized_requests": AUTHORIZED_REQUESTS,
        "v2_full_smoke_completed": False,
        "scientific_graph_waived": False,
        "validation_only": True,
        "final_test_accessed": False,
    })


def validate_operational_waiver(
    value: Mapping[str, Any], *, source_commit: str | None = None,
    semantic_source_sha256: Mapping[str, str] | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=OPERATIONAL_WAIVER_CONTRACT,
        expected_schema_version=1,
    )
    for name in (
        "v1_campaign_spec_sha256", "v1_campaign_completion_sha256",
        "v1_graph_sha256", "v2_smoke_campaign_spec_sha256", "v2_graph_sha256",
        "v2_endpoint_equality_lock_sha256", "v2_graph_recipe_lock_sha256",
        "v2_weaver_parity_sha256",
    ):
        require_sha256(value.get(name), name=name)
    if (
        value.get("authorization_phrase") != AUTHORIZATION_PHRASE
        or value.get("v1_fit_count") != V1_FIT_COUNT
        or value.get("v2_graph_sha256") != GRAPH_SHA256
        or value.get("required_v2_task_ids") != list(REQUIRED_V2_TASKS)
        or not set(REQUIRED_V2_TASKS) <= set(value.get("completed_v2_task_ids", ()))
        or value.get("v2_full_smoke_completed") is not False
        or value.get("scientific_graph_waived") is not False
        or value.get("validation_only") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("HCWDL-UJ operational waiver semantics differ")
    for field in ("semantic_source_sha256", "v2_smoke_semantic_source_sha256"):
        semantic = value.get(field)
        if not isinstance(semantic, Mapping) or any(
            not isinstance(name, str)
            or require_sha256(digest, name=f"{field} {name}") != digest
            for name, digest in semantic.items()
        ):
            raise ValueError(f"HCWDL-UJ waiver {field} differs")
    _commit(value.get("source_commit"))
    _commit(value.get("v2_smoke_source_commit"))
    if source_commit is not None and value.get("source_commit") != _commit(source_commit):
        raise ValueError("HCWDL-UJ operational waiver source differs")
    if semantic_source_sha256 is not None and value.get("semantic_source_sha256") != dict(semantic_source_sha256):
        raise ValueError("HCWDL-UJ operational waiver semantic source differs")
    requests = value.get("authorized_requests")
    if requests != AUTHORIZED_REQUESTS:
        raise ValueError("HCWDL-UJ operational waiver resources differ")
    return digest


def load_and_build_operational_waiver(
    *, v1_campaign_root: str | Path, v2_campaign_root: str | Path,
    completed_v2_task_ids: Sequence[str], authorization_phrase: str,
    authorized_source_commit: str,
    authorized_semantic_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    v1 = Path(v1_campaign_root); v2 = Path(v2_campaign_root)
    spec = load_json(v2 / "campaign_spec.json")
    return build_operational_waiver(
        v1_campaign_spec=load_json(v1 / "campaign_spec.json"),
        v1_campaign_completion=load_json(v1 / "reports/campaign_complete.json"),
        v2_campaign_spec=spec,
        v2_endpoint_lock=load_json(v2 / "locks/endpoint_equality_lock.json"),
        v2_graph_recipe_lock=load_json(v2 / "locks/graph_recipe_lock.json"),
        v2_weaver_parity=load_json(spec["weaver_parity"]["path"]),
        completed_v2_task_ids=completed_v2_task_ids,
        authorized_source_commit=authorized_source_commit,
        authorized_semantic_source_sha256=authorized_semantic_source_sha256,
        authorization_phrase=authorization_phrase,
    )


__all__ = [
    "AUTHORIZATION_PHRASE", "AUTHORIZED_REQUESTS", "REQUIRED_V2_TASKS",
    "build_operational_waiver", "load_and_build_operational_waiver",
    "validate_operational_waiver",
]
