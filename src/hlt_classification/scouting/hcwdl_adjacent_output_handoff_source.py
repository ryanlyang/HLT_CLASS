"""Explicit authentication of the full-cardinality U100 handoff source."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash,
)

from .hcwdl_adjacent_output_handoff_contracts import (
    CONTROL_LOCK_CONTRACT, SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_mhpe_tri60_ce_control_contracts import (
    TRAINING_REPORT_CONTRACT as CE60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_mhpe_tri60_contracts import (
    TRAINING_REPORT_CONTRACT as TRI60_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_bottleneck_campaign import (
    validate_campaign as validate_fullcard_campaign,
)
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    TRAINING_REPORT_CONTRACT as FULLCARD_TRAINING_REPORT_CONTRACT,
)
from .hcwdl_tri100_spine4_bottleneck_graph import NODE_REGISTRY as FULLCARD_NODES


def _report(path: str | Path, *, contract: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    value = load_json(resolved)
    digest = validate_content_hash(
        value, expected_contract=contract,
        expected_schema_version=int(value.get("schema_version", -1)),
    )
    if value.get("final_test_accessed") is not False:
        raise PermissionError("output-handoff source report accessed final test")
    return resolved, value, digest


def build_source_lock(
    *, source_campaign_spec: str | Path, u100_training_report: str | Path,
    u100_selected_checkpoint: str | Path,
) -> dict[str, Any]:
    spec_path = Path(source_campaign_spec).resolve()
    spec = load_json(spec_path)
    spec_hash = validate_fullcard_campaign(spec)
    report_path, report, report_hash = _report(
        u100_training_report, contract=FULLCARD_TRAINING_REPORT_CONTRACT,
    )
    node_id = str(report.get("node_id", ""))
    node = FULLCARD_NODES.get(node_id)
    checkpoint = Path(u100_selected_checkpoint).resolve()
    selected_name = str(report.get("selected_checkpoint", ""))
    expected_checkpoint = (report_path.parent / selected_name).resolve()
    if (
        node is None or node.coordinate_name != "U100"
        or report.get("campaign_spec_sha256") != spec_hash
        or report_path.parent.parent.parent.resolve() != Path(spec["campaign_root"]).resolve()
        or checkpoint != expected_checkpoint or not checkpoint.is_file()
        or sha256_file(checkpoint) != report.get("selected_checkpoint_sha256")
        or report.get("resume_policy") != "disabled_restart_from_zero_v1"
        or report.get("rolling_resume_published") is not False
    ):
        raise ValueError("output-handoff U100 source lineage differs")
    foundation_path = Path(spec["artifact_paths"]["foundation_spec"]).resolve()
    support_path = Path(spec["artifact_paths"]["support_audit"]).resolve()
    if not foundation_path.is_file() or not support_path.is_file():
        raise ValueError("output-handoff full-cardinality foundation is incomplete")
    return artifact({
        "parents": {
            "source_campaign": spec_hash,
            "source_graph": spec["parents"]["graph"],
            "source_recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
            "assignment_lock": spec["parents"]["assignment_lock"],
            "support_audit": validate_content_hash(load_json(support_path)),
            "u100_report": report_hash,
            "u100_checkpoint": report["selected_checkpoint_sha256"],
        },
        "source_campaign_spec_path": str(spec_path),
        "source_campaign_root": str(Path(spec["campaign_root"]).resolve()),
        "foundation_spec_path": str(foundation_path),
        "support_audit_path": str(support_path),
        "u100_node_id": node_id, "u100_report_path": str(report_path),
        "u100_checkpoint_path": str(checkpoint),
        "u100_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "replicate_seed": int(spec["replicate_seed"]),
        "role_counts": dict(spec["role_counts"]),
        "coordinate": "U100", "support_policy": spec["support_policy"],
        "source_campaign_completion_required": False,
        "source_outputs_mutated": False, "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)


def validate_source_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=SOURCE_LOCK_CONTRACT)
    expected = build_source_lock(
        source_campaign_spec=value["source_campaign_spec_path"],
        u100_training_report=value["u100_report_path"],
        u100_selected_checkpoint=value["u100_checkpoint_path"],
    )
    if dict(value) != expected:
        raise ValueError("output-handoff source lock changed")
    return digest


def build_control_lock(
    *, m0ce60_training_report: str | Path,
    pure_offline_u000_training_report: str | Path,
) -> dict[str, Any]:
    m0_path, m0, m0_hash = _report(
        m0ce60_training_report, contract=CE60_TRAINING_REPORT_CONTRACT,
    )
    u000_path, u000, u000_hash = _report(
        pure_offline_u000_training_report, contract=TRI60_TRAINING_REPORT_CONTRACT,
    )
    m0_checkpoint = (m0_path.parent / str(m0.get("selected_checkpoint", ""))).resolve()
    u000_checkpoint = (u000_path.parent / str(u000.get("selected_checkpoint", ""))).resolve()
    u000_spec_path = (u000_path.parent.parent.parent / "campaign_spec.json").resolve()
    if not u000_spec_path.is_file():
        raise ValueError("output-handoff U000 control campaign spec is unavailable")
    u000_spec = load_json(u000_spec_path)
    u000_spec_hash = validate_content_hash(
        u000_spec, expected_contract=str(u000_spec.get("contract", "")),
        expected_schema_version=int(u000_spec.get("schema_version", -1)),
    )
    if (
        m0.get("node_id") != "M0CE60" or u000.get("node_id") != "U000"
        or not isinstance(m0.get("validation"), Mapping)
        or not isinstance(u000.get("validation"), Mapping)
        or not m0_checkpoint.is_file() or not u000_checkpoint.is_file()
        or sha256_file(m0_checkpoint) != m0.get("selected_checkpoint_sha256")
        or sha256_file(u000_checkpoint) != u000.get("selected_checkpoint_sha256")
        or u000.get("campaign_spec_sha256") != u000_spec_hash
    ):
        raise ValueError("output-handoff reporting controls differ")
    return artifact({
        "parents": {
            "m0ce60_report": m0_hash,
            "m0ce60_checkpoint": m0["selected_checkpoint_sha256"],
            "pure_offline_u000_report": u000_hash,
            "pure_offline_u000_checkpoint": u000["selected_checkpoint_sha256"],
            "pure_offline_u000_campaign": u000_spec_hash,
        },
        "m0ce60_report_path": str(m0_path),
        "m0ce60_checkpoint_path": str(m0_checkpoint),
        "pure_offline_u000_report_path": str(u000_path),
        "pure_offline_u000_checkpoint_path": str(u000_checkpoint),
        "pure_offline_u000_campaign_spec_path": str(u000_spec_path),
        "training_teacher_use": False, "reporting_only": True,
        "final_test_accessed": False,
    }, contract=CONTROL_LOCK_CONTRACT)


def validate_control_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=CONTROL_LOCK_CONTRACT)
    expected = build_control_lock(
        m0ce60_training_report=value["m0ce60_report_path"],
        pure_offline_u000_training_report=value["pure_offline_u000_report_path"],
    )
    if dict(value) != expected:
        raise ValueError("output-handoff control lock changed")
    return digest


__all__ = [
    "build_control_lock", "build_source_lock", "validate_control_lock",
    "validate_source_lock",
]
