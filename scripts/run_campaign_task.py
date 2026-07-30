#!/usr/bin/env python3
"""Execute one production campaign task and publish its artifact attestation."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import (  # noqa: E402
    build_task_attestation,
    task_by_name,
    validate_campaign_spec,
)
from hlt_classification.data.cache_contracts import (  # noqa: E402
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    sha256_file,
    write_immutable_json,
)
from hlt_classification.data.hlt_v3 import (  # noqa: E402
    build_hlt_v3_profile_contract,
)
from hlt_classification.data.replicas import (  # noqa: E402
    build_hlt_replica_manifest,
)
from hlt_classification.data.schema import schema_payload  # noqa: E402
from hlt_classification.data.splits import load_split_manifest  # noqa: E402
from hlt_classification.provenance import validate_campaign_source  # noqa: E402
from hlt_classification.training.engine import TrainingConfig  # noqa: E402


def _run(
    command: list[str],
    *,
    report_path: Path | None = None,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if result.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(result.returncode, command)
    if report_path is not None:
        atomic_publish_bytes(report_path, result.stdout)


def _python(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        "-s",
        str(REPO_ROOT / "scripts" / script),
        *(str(value) for value in arguments),
    ]


def _relative_artifacts(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required task artifact is absent: {path}")
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def _all_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(files))


def _write_split_parents(spec: dict, root: Path) -> list[Path]:
    split_path = root / "inputs" / "split_manifest.json.gz"
    manifest = load_split_manifest(split_path)
    validation_hash = canonical_sha256(
        [identity.key for identity in manifest.splits["model_val"]]
    )
    train_hash = canonical_sha256(
        [identity.key for identity in manifest.splits["model_train"]]
    )
    replica = build_hlt_replica_manifest(
        split_manifest_sha256=manifest.content_hash,
        validation_partition_sha256=validation_hash,
        scale_train_manifest_sha256=train_hash,
    )
    replica_path = root / "inputs" / "replica_manifest.json"
    write_immutable_json(replica_path, replica)
    profile = build_hlt_v3_profile_contract(
        raw_input_schema_sha256=canonical_sha256(schema_payload()),
        hlt_replica_manifest_sha256=replica["content_hash"],
    )
    profile_path = root / "inputs" / "hlt_profile.json"
    write_immutable_json(profile_path, profile)
    config_fields = {field.name for field in fields(TrainingConfig)}
    config = {
        key: value
        for key, value in spec["training_config"].items()
        if key in config_fields
    }
    config_path = root / "inputs" / "training_config.json"
    write_immutable_json(config_path, config)
    return [split_path, replica_path, profile_path, config_path]


def _role() -> str:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    index = int(raw)
    roles = ("model_train", "model_val")
    if index not in range(len(roles)):
        raise ValueError("campaign cache array index lies outside [0,1]")
    return roles[index]


def _execute(task: str, spec: dict, root: Path) -> list[Path]:
    source = spec["source_snapshot_sha256"]
    data = spec["data"]
    if task == "preflight":
        output = root / "reports" / "preflight.json"
        required = sum(data["split_sizes"].values()) // 10
        _run(
            _python(
                "preflight_data.py",
                "--data-root",
                spec["site"]["data_dir"],
                "--required-per-class",
                required,
            ),
            report_path=output,
        )
        return [output]
    if task == "splits":
        output = root / "inputs" / "split_manifest.json.gz"
        command = _python(
            "build_splits.py",
            "--data-root",
            spec["site"]["data_dir"],
            "--output",
            output,
            "--base-seed",
            data["base_seed"],
        )
        for role, size in data["split_sizes"].items():
            command.extend([f"--{role.replace('_', '-')}", str(size)])
        _run(command, report_path=root / "reports" / "split_build.json")
        return [
            *_write_split_parents(spec, root),
            root / "reports" / "split_build.json",
        ]
    if task == "offline_cache":
        role = _role()
        cache = root / "caches" / "offline" / role
        _run(
            _python(
                "build_offline_cache.py",
                "--split-manifest",
                root / "inputs" / "split_manifest.json.gz",
                "--role",
                role,
                "--output-dir",
                cache,
                "--data-root",
                spec["site"]["data_dir"],
                "--shard-size",
                data["cache_shard_size"],
                "--source-snapshot-sha256",
                source,
            ),
            report_path=root / "reports" / f"offline_cache_{role}.json",
        )
        cache_roots = [
            root / "caches" / "offline" / current
            for current in ("model_train", "model_val")
        ]
        if all((path / "manifest.json").is_file() for path in cache_roots):
            return _all_files(*cache_roots)
        return []
    if task == "hlt_cache":
        role = _role()
        cache = root / "caches" / "hlt" / role / "replica_0"
        _run(
            _python(
                "build_hlt_cache.py",
                "--offline-cache-dir",
                root / "caches" / "offline" / role,
                "--output-dir",
                cache,
                "--profile-contract",
                root / "inputs" / "hlt_profile.json",
                "--replica-manifest",
                root / "inputs" / "replica_manifest.json",
                "--role",
                role,
                "--replica-id",
                0,
                "--realization-policy",
                data["hlt_realization_policy"],
                "--profile-id",
                data["degradation_profile_id"],
                "--source-snapshot-sha256",
                source,
                "--shard-size",
                data["cache_shard_size"],
            ),
            report_path=root / "reports" / f"hlt_cache_{role}.json",
        )
        cache_roots = [
            root / "caches" / "hlt" / current / "replica_0"
            for current in ("model_train", "model_val")
        ]
        if all((path / "manifest.json").is_file() for path in cache_roots):
            for current in ("model_train", "model_val"):
                _run(
                    _python(
                        "audit_hlt_cache.py",
                        "--cache-dir",
                        root / "caches" / "hlt" / current / "replica_0",
                        "--expected-degradation-profile-id",
                        data["degradation_profile_id"],
                    ),
                    report_path=root / "reports" / f"hlt_audit_{current}.json",
                )
            return _all_files(*cache_roots) + [
                root / "reports" / f"hlt_audit_{current}.json"
                for current in ("model_train", "model_val")
            ]
        return []
    if task == "weaver_parity":
        output = root / "reports" / "weaver_parity.json"
        _run(
            _python("validate_weaver_parity.py", "--device", "cpu"),
            report_path=output,
        )
        return [output]
    if task == "train_interrupt":
        output = root / "models" / "baseline"
        report = root / "reports" / "train_interrupt_command.json"
        _run(
            _python(
                "train_part.py",
                "--train-cache",
                f"0={root / 'caches' / 'hlt' / 'model_train' / 'replica_0'}",
                "--validation-cache",
                root / "caches" / "hlt" / "model_val" / "replica_0",
                "--config",
                root / "inputs" / "training_config.json",
                "--output-dir",
                output,
                "--source-snapshot-sha256",
                source,
                "--device",
                "cuda",
                "--stop-after-update",
                1,
            ),
            report_path=report,
            allowed_returncodes=(3,),
        )
        return [output / "last.pt", report]
    if task == "train":
        output = root / "models" / "baseline"
        _run(
            _python(
                "train_part.py",
                "--train-cache",
                f"0={root / 'caches' / 'hlt' / 'model_train' / 'replica_0'}",
                "--validation-cache",
                root / "caches" / "hlt" / "model_val" / "replica_0",
                "--config",
                root / "inputs" / "training_config.json",
                "--output-dir",
                output,
                "--source-snapshot-sha256",
                source,
                "--device",
                "cuda",
            ),
            report_path=root / "reports" / "train_command.json",
        )
        return [
            output / "training_report.json",
            output / "last.pt",
            output / "best_model_val.pt",
        ]
    if task == "evaluate_model_val":
        model = root / "models" / "baseline"
        predictions = root / "predictions" / "model_val"
        metrics = root / "reports" / "model_val_metrics.json"
        _run(
            _python(
                "evaluate_part.py",
                "--training-report",
                model / "training_report.json",
                "--hlt-cache",
                root / "caches" / "hlt" / "model_val" / "replica_0",
                "--prediction-dir",
                predictions,
                "--metrics-output",
                metrics,
                "--source-snapshot-sha256",
                source,
                "--device",
                "cuda",
            ),
            report_path=root / "reports" / "evaluate_command.json",
        )
        return _all_files(predictions) + [metrics]
    raise ValueError(f"unknown campaign task {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_spec(spec)
    validate_campaign_source(spec, repository=REPO_ROOT)
    task_by_name(spec, args.task)
    root = Path(spec["site"]["campaign_root"])
    artifacts = _execute(args.task, spec, root)
    # Array elements publish only after the complete array's artifact set exists.
    if artifacts:
        attestation = build_task_attestation(
            campaign_spec=spec,
            task_name=args.task,
            artifacts=_relative_artifacts(root, artifacts),
        )
        write_immutable_json(
            root / "task_attestations" / f"{args.task}.json",
            attestation,
        )
    print(json.dumps({"task": args.task, "artifacts": len(artifacts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
