"""Authenticated retirement of completed HCWDL-U-RKD rolling resumes."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from hlt_classification.data.cache_contracts import (
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_homotopy_representation_campaign import validate_campaign
from .hcwdl_homotopy_representation_contracts import (
    RESUME_RETIREMENT_AUTHORIZATION_CONTRACT,
    RESUME_RETIREMENT_COMPLETION_CONTRACT,
    RESUME_RETIREMENT_PHRASE,
    TRAINING_REPORT_CONTRACT,
    validate_artifact,
)
from .hcwdl_homotopy_representation_graph import (
    NODE_REGISTRY,
    TRAINING_PASSES,
    target_bank_registry,
)
from .hcwdl_homotopy_representation_training import node_output_dir
from .hcwdl_representation_artifacts import validate_binary_envelope
from .hcwdl_representation_training import (
    FINAL_TRAINING_CHECKPOINT_CONTRACT,
    SELECTED_TRAINING_CHECKPOINT_CONTRACT,
    validate_representation_training_report,
)


_STATE = re.compile(r"state_([0-9]+)\.pt")
_SIDECAR = re.compile(r"state_([0-9]+)\.json")
_COMMIT = re.compile(r"commit_([0-9]+)\.json")


def _artifact(
    contract: str, *, parents: Mapping[str, str], **payload: Any,
) -> dict[str, Any]:
    return with_content_hash({
        "contract": contract,
        "schema_version": 1,
        "parents": {
            str(name): require_sha256(value, name=f"{contract} parent {name}")
            for name, value in sorted(parents.items())
        },
        **payload,
        "final_test_accessed": False,
    })


def validate_retirement_artifact(
    value: Mapping[str, Any], *, contract: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=1,
    )
    if value.get("final_test_accessed") is not False:
        raise PermissionError("resume retirement may not access final test")
    parents = value.get("parents")
    if not isinstance(parents, Mapping):
        raise ValueError("resume retirement lacks immutable parents")
    for name, parent in parents.items():
        require_sha256(parent, name=f"resume retirement parent {name}")
    return digest


def _validated_envelope(
    report: Mapping[str, Any], output: Path, kind: str,
) -> str:
    row = report["checkpoint_envelopes"][kind]
    envelope_id = require_sha256(row["envelope_id"], name=f"{kind} envelope ID")
    root = output / "checkpoints" / kind
    commit = load_json(root / "committed" / envelope_id / "commit.json")
    members = commit.get("payload", {}).get("members", [])
    if not members or not isinstance(members[0], Mapping):
        raise ValueError("terminal checkpoint envelope has no members")
    parents = members[0].get("immutable_parent_hashes")
    if not isinstance(parents, Mapping):
        raise ValueError("terminal checkpoint envelope lacks member parents")
    expected_contract = (
        SELECTED_TRAINING_CHECKPOINT_CONTRACT
        if kind == "selected" else FINAL_TRAINING_CHECKPOINT_CONTRACT
    )
    envelope = validate_binary_envelope(
        root,
        envelope_id,
        expected_contract=expected_contract,
        expected_parents=parents,
    )
    digest = require_sha256(row["commit_sha256"], name=f"{kind} envelope commit")
    if envelope.commit["content_hash"] != digest:
        raise ValueError("terminal checkpoint envelope report binding differs")
    return digest


def _resume_members(
    resume_root: Path, *, campaign_root: Path, node_id: str,
) -> list[dict[str, Any]]:
    if not resume_root.is_dir():
        raise FileNotFoundError(f"completed node has no resume directory: {resume_root}")
    grouped: dict[int, dict[str, Path]] = {}
    unrelated = []
    for path in resume_root.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        matched = False
        for kind, pattern in (
            ("state", _STATE), ("sidecar", _SIDECAR), ("commit", _COMMIT),
        ):
            match = pattern.fullmatch(path.name)
            if match:
                grouped.setdefault(int(match.group(1)), {})[kind] = path
                matched = True
                break
        if not matched:
            unrelated.append(path.name)
    if unrelated or not grouped:
        raise ValueError(f"resume directory contains unrelated or no files: {unrelated}")
    rows = []
    for sequence, paths in sorted(grouped.items()):
        if set(paths) != {"state", "sidecar", "commit"}:
            raise ValueError(f"resume generation {sequence} is incomplete")
        commit = load_json(paths["commit"])
        validate_content_hash(
            commit,
            expected_contract=str(commit["contract"]),
            expected_schema_version=int(commit["schema_version"]),
        )
        payload = commit.get("payload", {})
        if (
            payload.get("sequence") != sequence
            or payload.get("state_relative_path") != paths["state"].name
            or payload.get("sidecar_relative_path") != paths["sidecar"].name
        ):
            raise ValueError("resume commit path identity differs")
        sidecar = load_json(paths["sidecar"])
        validate_content_hash(
            sidecar,
            expected_contract=str(sidecar["contract"]),
            expected_schema_version=int(sidecar["schema_version"]),
        )
        if sidecar["content_hash"] != payload.get("sidecar_content_hash"):
            raise ValueError("resume commit sidecar binding differs")
        expected = {
            "state": require_sha256(
                payload.get("state_serialized_byte_sha256"),
                name="retired resume state SHA-256",
            ),
            "sidecar": require_sha256(
                payload.get("sidecar_serialized_byte_sha256"),
                name="retired resume sidecar SHA-256",
            ),
            # A content hash authenticates semantic JSON content, while the
            # retirement registry binds the exact serialized member bytes.
            "commit": sha256_file(paths["commit"]),
        }
        for kind, path in paths.items():
            digest = sha256_file(path)
            if digest != expected[kind]:
                raise ValueError(f"resume {kind} bytes differ before retirement")
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(campaign_root.resolve())
            except ValueError as error:
                raise PermissionError("resume retirement escaped campaign root") from error
            rows.append({
                "node_id": node_id,
                "sequence": sequence,
                "kind": kind,
                "path": str(resolved),
                "relative_path": relative.as_posix(),
                "byte_sha256": digest,
                "bytes": path.stat().st_size,
            })
    return rows


def build_resume_retirement_authorization(
    spec: Mapping[str, Any], *, node_ids: Sequence[str], source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("resume-retirement source commit differs")
    campaign_hash = validate_campaign(spec, executable=False)
    requested = tuple(sorted(set(node_ids)))
    if not requested or len(requested) != len(node_ids):
        raise ValueError("resume retirement node list is empty or repeated")
    root = Path(spec["campaign_root"]).resolve()
    consumers = target_bank_registry()
    parents = {
        "campaign_spec": campaign_hash,
        "retirement_module": sha256_file(Path(__file__)),
    }
    files = []
    nodes = []
    for node_id in requested:
        if node_id not in NODE_REGISTRY:
            raise ValueError(f"unknown resume-retirement node: {node_id}")
        direct_consumers = consumers.get(node_id)
        if not direct_consumers:
            raise PermissionError(
                "terminal node resumes require completed-campaign retirement"
            )
        output = node_output_dir(root, node_id)
        report = load_json(output / "training_report.json")
        report_hash = validate_representation_training_report(
            report,
            expected_execution_id=node_id,
            expected_recipe_sha256=spec["representation_recipe_sha256"],
        )
        if (
            report.get("complete") is not True
            or report.get("scientific_complete") is not True
            or int(report.get("completed_natural_population_passes", -1))
            != TRAINING_PASSES
        ):
            raise PermissionError("resume retirement requires a complete scientific node")
        wrapper = load_json(output / "combined_training_report.json")
        wrapper_hash = validate_artifact(
            wrapper,
            contract=TRAINING_REPORT_CONTRACT,
            required_parents=("campaign_spec", "engine_report"),
            required_fields=("node_id", "completed_passes"),
        )
        if (
            wrapper["node_id"] != node_id
            or wrapper["parents"]["engine_report"] != report_hash
            or int(wrapper["completed_passes"]) != TRAINING_PASSES
        ):
            raise ValueError("completed training wrapper differs")
        selected_hash = _validated_envelope(report, output, "selected")
        final_hash = _validated_envelope(report, output, "final")
        consumer_hashes = {}
        for consumer_id in direct_consumers:
            consumer_output = node_output_dir(root, consumer_id)
            consumer = load_json(consumer_output / "combined_training_report.json")
            consumer_hash = validate_artifact(
                consumer,
                contract=TRAINING_REPORT_CONTRACT,
                required_parents=("campaign_spec", "engine_report"),
                required_fields=("node_id", "completed_passes"),
            )
            if (
                consumer["node_id"] != consumer_id
                or int(consumer["completed_passes"]) != TRAINING_PASSES
            ):
                raise PermissionError("direct consumer has not completed")
            consumer_hashes[consumer_id] = consumer_hash
            parents[f"consumer_{consumer_id}"] = consumer_hash
        parents[f"report_{node_id}"] = report_hash
        parents[f"wrapper_{node_id}"] = wrapper_hash
        parents[f"selected_{node_id}"] = selected_hash
        parents[f"final_{node_id}"] = final_hash
        node_files = _resume_members(
            output / "resume", campaign_root=root, node_id=node_id,
        )
        files.extend(node_files)
        nodes.append({
            "node_id": node_id,
            "report_sha256": report_hash,
            "wrapper_sha256": wrapper_hash,
            "selected_envelope_sha256": selected_hash,
            "final_envelope_sha256": final_hash,
            "completed_consumers": consumer_hashes,
            "resume_generation_count": len(node_files) // 3,
            "retired_bytes": sum(row["bytes"] for row in node_files),
        })
    files.sort(key=lambda row: (row["node_id"], row["sequence"], row["kind"]))
    return _artifact(
        RESUME_RETIREMENT_AUTHORIZATION_CONTRACT,
        parents=parents,
        authorization_phrase=RESUME_RETIREMENT_PHRASE,
        source_commit=source_commit,
        campaign_root=str(root),
        nodes=nodes,
        files=files,
        retired_bytes=sum(row["bytes"] for row in files),
        selected_and_final_checkpoints_retained=True,
        reports_retained=True,
    )


def execute_resume_retirement(
    authorization: Mapping[str, Any], *, authorization_phrase: str,
) -> dict[str, Any]:
    authorization_hash = validate_retirement_artifact(
        authorization, contract=RESUME_RETIREMENT_AUTHORIZATION_CONTRACT,
    )
    if (
        authorization_phrase != RESUME_RETIREMENT_PHRASE
        or authorization.get("authorization_phrase") != RESUME_RETIREMENT_PHRASE
    ):
        raise PermissionError("resume-retirement phrase differs")
    files = authorization.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("resume-retirement file registry is empty")
    campaign_root_value = Path(str(authorization.get("campaign_root", "")))
    if not campaign_root_value.is_absolute():
        raise ValueError("resume-retirement campaign root is not absolute")
    campaign_root = campaign_root_value.resolve()
    for row in files:
        path = Path(str(row["path"]))
        try:
            relative = path.resolve().relative_to(campaign_root)
        except ValueError as error:
            raise PermissionError("resume-retirement path escaped campaign root") from error
        if relative.as_posix() != row.get("relative_path") or path.name.startswith("."):
            raise PermissionError("resume-retirement relative path binding differs")
        if path.exists() and sha256_file(path) != require_sha256(
            row["byte_sha256"], name="authorized retired member SHA-256",
        ):
            raise ValueError("resume member changed after authorization")
    order = {"commit": 0, "sidecar": 1, "state": 2}
    for row in sorted(
        files,
        key=lambda item: (
            item["node_id"], item["sequence"], order[item["kind"]],
        ),
    ):
        Path(row["path"]).unlink(missing_ok=True)
    if any(Path(row["path"]).exists() for row in files):
        raise RuntimeError("authorized resume retirement is incomplete")
    return _artifact(
        RESUME_RETIREMENT_COMPLETION_CONTRACT,
        parents={"authorization": authorization_hash},
        retired_nodes=[row["node_id"] for row in authorization["nodes"]],
        retired_bytes=int(authorization["retired_bytes"]),
        retired_member_count=len(files),
        all_authorized_paths_absent=True,
        selected_and_final_checkpoints_retained=True,
        reports_retained=True,
    )


__all__ = [
    "build_resume_retirement_authorization",
    "execute_resume_retirement",
    "validate_retirement_artifact",
]
