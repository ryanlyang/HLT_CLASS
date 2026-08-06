#!/usr/bin/env python3
"""Hard-link and re-attest a compatible completed PMARD pilot prefix."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.provenance import validate_source_snapshot  # noqa: E402
from hlt_classification.scouting.campaign import (  # noqa: E402
    PMARD_LEDGER_CONTRACT, validate_pmard_campaign_spec,
)
from hlt_classification.scouting.config_contracts import validate_vendored_preprocessing  # noqa: E402
from hlt_classification.scouting.engine import (  # noqa: E402
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
)
from hlt_classification.scouting.recovery import (  # noqa: E402
    IMPORTED_TASKS, PMARD_PREFIX_IMPORT_CONTRACT, PMARD_PREFIX_IMPORT_VERSION,
    PREFIX_TASKS, REBUILT_TASKS, validate_prefix_import_compatibility,
)
from hlt_classification.scouting.locks import create_lock  # noqa: E402
from hlt_classification.scouting.workflow import Workflow, write_task_attestation  # noqa: E402


def _array_ids(task: dict[str, object]) -> list[str | None]:
    value = task.get("array")
    if value is None:
        return [None]
    match = re.fullmatch(r"(\d+)-(\d+)(?:%\d+)?", str(value))
    if match is None:
        raise ValueError(f"unsupported recovery array expression {value!r}")
    return [str(index) for index in range(int(match.group(1)), int(match.group(2)) + 1)]


def _attestation_path(root: Path, task: str, array_id: str | None) -> Path:
    suffix = "" if array_id is None else f"_{array_id}"
    return root / "task_attestations" / f"{task}{suffix}.json"


def _hardlink(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"recovery source is absent: {source}")
    if target.exists():
        raise FileExistsError(f"recovery target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    if sha256_file(target) != sha256_file(source):
        raise RuntimeError("hard-linked recovery artifact hash differs")


def _import_attested_outputs(
    *, source_root: Path, target_root: Path, source_spec: dict[str, object],
    target_spec: dict[str, object], task: str, array_id: str | None,
) -> tuple[list[Path], list[dict[str, str]]]:
    attestation = load_json(_attestation_path(source_root, task, array_id))
    validate_content_hash(
        attestation, expected_contract="hlt_classification_pmard_task_attestation_v1",
        expected_schema_version=1,
    )
    if (
        attestation.get("campaign_spec_sha256") != source_spec["content_hash"]
        or attestation.get("task") != task
        or attestation.get("array_task_id") != array_id
        or attestation.get("complete") is not True
    ):
        raise ValueError(f"source attestation lineage differs for {task}_{array_id}")
    outputs = []
    imported = []
    for row in attestation["outputs"]:
        source = Path(row["path"]).resolve()
        try:
            relative = source.relative_to(source_root.resolve())
        except ValueError as error:
            raise ValueError("source attestation output escapes its campaign") from error
        if sha256_file(source) != row["sha256"]:
            raise ValueError("source attestation output hash differs")
        target = target_root / relative
        _hardlink(source, target); outputs.append(target)
        imported.append({"source": str(source), "target": str(target), "sha256": row["sha256"]})
        if target.name == "training_report.json":
            report = load_json(target)
            validate_content_hash(
                report, expected_contract=PMARD_TRAINING_REPORT_CONTRACT,
                expected_schema_version=PMARD_TRAINING_REPORT_VERSION,
            )
            checkpoint_source = source.parent / report["selected_checkpoint"]
            if sha256_file(checkpoint_source) != report["selected_checkpoint_sha256"]:
                raise ValueError("source selected checkpoint hash differs")
            checkpoint_target = target.parent / report["selected_checkpoint"]
            _hardlink(checkpoint_source, checkpoint_target)
            imported.append({
                "source": str(checkpoint_source), "target": str(checkpoint_target),
                "sha256": report["selected_checkpoint_sha256"],
            })
    write_task_attestation(
        spec=target_spec, task=task, array_id=array_id, outputs=outputs,
        campaign_root=target_root,
    )
    return outputs, imported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign-spec", type=Path, required=True)
    parser.add_argument("--source-submission-ledger", type=Path, required=True)
    parser.add_argument("--target-campaign-spec", type=Path, required=True)
    parser.add_argument("--output-monitor", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    source_spec = load_json(args.source_campaign_spec)
    target_spec = load_json(args.target_campaign_spec)
    validate_pmard_campaign_spec(source_spec); validate_pmard_campaign_spec(target_spec)
    validate_source_snapshot(
        target_spec["source_snapshot"], repository=args.repository.resolve(), require_clean=True,
    )
    proof = validate_prefix_import_compatibility(
        source_spec, target_spec, repository=args.repository.resolve(),
    )
    source_ledger = load_json(args.source_submission_ledger)
    source_ledger_hash = validate_content_hash(
        source_ledger, expected_contract=PMARD_LEDGER_CONTRACT, expected_schema_version=1,
    )
    if (
        source_ledger.get("campaign_spec_sha256") != source_spec["content_hash"]
        or source_ledger.get("dry_run") is not False
        or set(source_ledger.get("jobs", {})) != {task["name"] for task in source_spec["tasks"]}
        or any(not re.fullmatch(r"[1-9][0-9]*", str(job)) for job in source_ledger.get("jobs", {}).values())
    ):
        raise ValueError("source live ledger differs from the source campaign")
    source_root = Path(source_spec["campaign_root"]).resolve()
    target_root = Path(target_spec["campaign_root"]).resolve()
    if source_root == target_root:
        raise ValueError("prefix import requires a new campaign root")
    try:
        args.target_campaign_spec.resolve().relative_to(target_root)
        args.output_monitor.resolve().relative_to(target_root)
    except ValueError as error:
        raise ValueError("target spec and recovery monitor must stay inside the target campaign") from error
    existing = {path.resolve() for path in target_root.rglob("*") if path.is_file()}
    if existing != {args.target_campaign_spec.resolve()}:
        raise FileExistsError("target campaign root must contain only its immutable campaign spec")
    target_root.mkdir(parents=True, exist_ok=True)
    imported_rows: list[dict[str, str]] = []
    tasks = {task["name"]: task for task in target_spec["tasks"]}
    for task_name in IMPORTED_TASKS:
        for array_id in _array_ids(tasks[task_name]):
            _, rows = _import_attested_outputs(
                source_root=source_root, target_root=target_root,
                source_spec=source_spec, target_spec=target_spec,
                task=task_name, array_id=array_id,
            )
            imported_rows.extend(rows)
    import_path = target_root / "recovery/prefix_import.json"
    import_report = with_content_hash({
        "contract": PMARD_PREFIX_IMPORT_CONTRACT,
        "schema_version": PMARD_PREFIX_IMPORT_VERSION,
        **proof,
        "source_submission_ledger_sha256": source_ledger_hash,
        "prefix_tasks": list(PREFIX_TASKS),
        "imported_tasks": list(IMPORTED_TASKS),
        "rebuilt_tasks": list(REBUILT_TASKS),
        "hardlink_only": True,
        "imported_files": imported_rows,
        "scientific_training_reused": True,
        "assignment_cache_reused": False,
        "assignment_cache_rebuild_required": True,
        "failed_teacher_or_descendant_output_reused": False,
    })
    write_immutable_json(import_path, import_report)
    workflow = Workflow(target_spec, repository=args.repository.resolve())
    for task_name in REBUILT_TASKS:
        if task_name == "data_lock":
            source = load_json(workflow.source)
            split = load_json(workflow.split)
            audit = load_json(workflow.audit)
            output = workflow.data_lock
            write_immutable_json(output, create_lock(
                "data", campaign_spec_sha256=target_spec["content_hash"],
                payload={
                    "source_manifest_sha256": source["content_hash"],
                    "split_manifest_sha256": split["content_hash"],
                    "feature_audit_sha256": audit["content_hash"],
                    "preprocessing_contract": validate_vendored_preprocessing(args.repository),
                    "prefix_import_sha256": import_report["content_hash"],
                },
            ))
            outputs = [output]
        else:
            outputs = workflow.run(task_name)
        write_task_attestation(
            spec=target_spec, task=task_name, array_id=None, outputs=outputs,
            campaign_root=target_root,
        )
    monitor_rows = []
    for task in target_spec["tasks"]:
        task_name = task["name"]
        attestations = []
        if task_name in PREFIX_TASKS:
            for array_id in _array_ids(task):
                payload = load_json(_attestation_path(target_root, task_name, array_id))
                attestations.append(payload["content_hash"])
        monitor_rows.append({
            "task": task_name,
            "job_id": source_ledger["jobs"][task_name],
            "state": "COMPLETED" if task_name in PREFIX_TASKS else "NOT_REUSED",
            "attestations": attestations,
            "reusable": task_name in PREFIX_TASKS,
        })
    monitor = with_content_hash({
        "contract": "hlt_classification_pmard_monitor_v1", "schema_version": 1,
        "campaign_spec_sha256": target_spec["content_hash"],
        "ledger_sha256": source_ledger_hash,
        "prefix_import_sha256": import_report["content_hash"],
        "jobs": monitor_rows,
    })
    write_immutable_json(args.output_monitor, monitor)
    print(json.dumps({
        "prefix_import": str(import_path),
        "prefix_import_sha256": import_report["content_hash"],
        "monitor": str(args.output_monitor),
        "reused_training_artifacts": ["budget_grid", "temperature_grid"],
        "next_task": "assignment_cache",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
