from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_npz_arrays,
    with_content_hash,
)
from hlt_classification.data.identity import FileRecord
from hlt_classification.prad.cache import build_prad_array_cache
from hlt_classification.prad.campaign import (
    PRAD_VARIANTS,
    build_prad_task_attestation,
    build_prad_resource_evidence,
    build_prad_storage_evidence,
    create_prad_campaign_spec,
    estimate_prad_peak_storage_bytes,
    prad_tasks,
    render_prad_submission_plan,
    submit_prad_plan,
    validate_prad_task_attestation,
    validate_prad_campaign_spec,
)
from hlt_classification.prad.splits import build_prad_split_manifest
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT


def _snapshot():
    core = {
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "tracked_files_sha256": "3" * 64,
    }
    return with_content_hash(
        {
            "contract": SOURCE_SNAPSHOT_CONTRACT,
            "schema_version": 1,
            **core,
            "tracked_file_count": 20,
            "worktree_clean": True,
            "source_snapshot_sha256": canonical_sha256(core),
        }
    )


def test_prad_smoke_dag_uses_exact_tigris_account_and_dependencies() -> None:
    spec = create_prad_campaign_spec(
        source_snapshot=_snapshot(),
        mode="smoke",
        campaign_root="/home/ryreu/atlas/HLT_Classification/artifacts/prad/smoke",
    )
    validate_prad_campaign_spec(spec)
    assert tuple(spec["required_full_split"]["sizes"].values()) == (500_000, 150_000, 500_000)
    assert tuple(spec["split"]["sizes"].values()) == (200, 100, 100)
    assert set(spec["run_registry"]["core"]) == {f"E{i}" for i in range(11)}
    assert set(spec["run_registry"]["variants"]) == set(PRAD_VARIANTS)
    plan = render_prad_submission_plan(
        campaign_spec_path="/tmp/prad.json", spec=spec
    )
    assert all("--account=reu-aisocial" in row["command"] for row in plan)
    assert all("--partition=tigris" in row["command"] for row in plan)
    final = next(row for row in plan if row["task"] == "final_test")
    dependency = next(value for value in final["command"] if value.startswith("--dependency"))
    assert "${finalist_lock_JOB_ID}" in dependency
    baseline = next(row for row in spec["tasks"] if row["name"] == "E0_baseline")
    assert "prad_runtime" in baseline["dependencies"]
    finalist = next(row for row in spec["tasks"] if row["name"] == "finalist_lock")
    assert finalist["dependencies"] == ["confirmation", "split"]
    oracle = next(row for row in spec["tasks"] if row["name"] == "E2_oracle")
    assert oracle["dependencies"] == ["E1_teacher"]
    assert len(spec["tasks"]) == 15
    assert not {
        "paired_train",
        "paired_val",
        "paired_test_inputs",
        "targets_train",
        "targets_val",
        "targets_test_inputs",
        "teacher_val_outputs",
    } & {row["name"] for row in spec["tasks"]}
    assert spec["storage_policy"]["large_intermediates_durable"] is False
    assert spec["storage_policy"]["paired_views"] == (
        "slurm_job_local_ephemeral_recomputed"
    )
    assert 28 * 1024**3 < estimate_prad_peak_storage_bytes() < 30 * 1024**3


def test_prad_task_attestation_is_bound_to_exact_array_element() -> None:
    attestation = build_prad_task_attestation(
        campaign_spec_sha256="a" * 64,
        task="confirmation",
        array_task_id="3",
        result={"seed": 44},
    )
    validate_prad_task_attestation(
        attestation,
        campaign_spec_sha256="a" * 64,
        task="confirmation",
        array_task_id="3",
    )
    with pytest.raises(ValueError, match="lineage"):
        validate_prad_task_attestation(
            attestation,
            campaign_spec_sha256="a" * 64,
            task="confirmation",
            array_task_id="2",
        )


def test_weaver_parity_worker_does_not_require_split_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_prad_task.py"
    module_spec = importlib.util.spec_from_file_location(
        "run_prad_task_regression", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    commands = []
    monkeypatch.setattr(module, "_run", lambda command: commands.append(command))
    result = module._dispatch(
        "weaver_parity",
        {
            "source_snapshot": {"source_snapshot_sha256": "a" * 64},
            "site": {"project_dir": "/project"},
        },
        {"split": tmp_path / "missing.json"},
    )
    assert result == {"validated": True}
    assert commands and commands[0][-2:] == ["--device", "cpu"]


def test_prad_runtime_worker_authenticates_v3_report_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_prad_task.py"
    module_spec = importlib.util.spec_from_file_location(
        "run_prad_task_runtime_v3", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    reports = tmp_path / "reports"
    expected = with_content_hash(
        {
            "contract": "hlt_classification_prad_runtime_validation_v3",
            "schema_version": 3,
            "passed": True,
        }
    )

    def successful_runtime(command):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(module, "_run", successful_runtime)
    result = module._dispatch(
        "prad_runtime",
        {
            "source_snapshot": {"source_snapshot_sha256": "a" * 64},
            "site": {"project_dir": "/project"},
        },
        {"reports": reports},
    )

    assert result == {"report_sha256": expected["content_hash"]}


def test_resource_capture_includes_distinct_slurm_array_child_raw_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "capture_prad_resources.py"
    module_spec = importlib.util.spec_from_file_location(
        "capture_prad_array_resources", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    stdout = "\n".join(
        (
            "42825|COMPLETED|1053||8",
            "42825.batch|COMPLETED|1053|2668800K|8",
            # RIT Slurm returns a distinct JobIDRaw for array element 1.
            "42992|COMPLETED|1074||8",
            "42992.batch|COMPLETED|1074|12769856K|8",
        )
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
    )

    usage = module._query_job("42825")

    assert usage == {
        "state": "COMPLETED",
        "elapsed_seconds": 1074,
        "max_rss_bytes": 12769856 * 1024,
        "allocated_cpus": 8,
    }


def test_minimum_storage_worker_keeps_large_caches_outside_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_prad_task.py"
    module_spec = importlib.util.spec_from_file_location(
        "run_prad_task_storage", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = create_prad_campaign_spec(
        source_snapshot=_snapshot(), mode="smoke", campaign_root="/campaign"
    )
    ephemeral_base = tmp_path / "slurm_job"
    monkeypatch.setenv("SLURM_TMPDIR", str(ephemeral_base))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    paths = module._paths(spec)
    assert paths["ephemeral_root"].parent == ephemeral_base.resolve()
    assert paths["paired"].is_relative_to(paths["ephemeral_root"])
    assert paths["targets"].is_relative_to(paths["ephemeral_root"])
    assert paths["teacher_outputs"].is_relative_to(paths["ephemeral_root"])
    assert not paths["runs"].is_relative_to(paths["ephemeral_root"])


def test_minimum_storage_worker_rejects_ephemeral_project_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_prad_task.py"
    module_spec = importlib.util.spec_from_file_location(
        "run_prad_task_unsafe_storage", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    project = tmp_path / "project"
    spec = create_prad_campaign_spec(
        source_snapshot=_snapshot(), mode="smoke", campaign_root=str(project / "run")
    )
    spec["site"]["project_dir"] = str(project)
    monkeypatch.setenv("SLURM_TMPDIR", str(project / "tmp"))
    with pytest.raises(RuntimeError, match="unsafe PRAD ephemeral base"):
        module._paths(spec)
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="unsafe PRAD ephemeral base"):
        module._paths(spec)


def test_teacher_test_predictions_are_compact_and_durable(
    tmp_path: Path,
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_prad_task.py"
    module_spec = importlib.util.spec_from_file_location(
        "run_prad_task_teacher_predictions", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    split = build_prad_split_manifest(
        tuple(FileRecord(f"class_{label}/sample.root", label, 4) for label in range(10)),
        data_root=str(tmp_path),
        output_dir=tmp_path / "splits",
        split_sizes={"train": 20, "val": 10, "test": 10},
    )

    def outputs(start, stop, _identities):
        rows = stop - start
        logits = np.zeros((rows, 10), dtype=np.float32)
        logits[:, start % 10] = 1.0
        return {
            "teacher_logits": logits,
            "teacher_true_class_confidence": np.full(rows, 0.5, np.float32),
        }

    teacher_root = tmp_path / "ephemeral_teacher"
    build_prad_array_cache(
        split.identities("test"),
        cache_kind="teacher_outputs",
        logical_role="test",
        output_dir=teacher_root,
        parents={
            "split_manifest_sha256": split.content_hash,
            "teacher_checkpoint_sha256": "b" * 64,
        },
        shard_builder=outputs,
        shard_size=4,
    )
    paths = {
        "split": tmp_path / "splits" / "split_manifest.json",
        "results": tmp_path / "durable_results",
    }
    report = module._evaluate_teacher_test_cache(
        paths, teacher_root, "a" * 64
    )
    prediction_root = paths["results"] / "test" / "E1" / "predictions"
    arrays = load_npz_arrays(prediction_root / "teacher_logits.npz")
    assert set(arrays) == {"logits"}
    assert arrays["logits"].shape == (10, 10)
    assert report["teacher_prediction_manifest_sha256"]


def test_prad_production_requires_all_prior_evidence() -> None:
    with pytest.raises(PermissionError, match="explicit authorization"):
        create_prad_campaign_spec(
            source_snapshot=_snapshot(),
            mode="production",
            campaign_root="/campaign",
        )
    with pytest.raises(ValueError, match="dry_run_report_sha256"):
        create_prad_campaign_spec(
            source_snapshot=_snapshot(),
            mode="production",
            campaign_root="/campaign",
            production_authorized=True,
        )


def test_prad_submission_uses_numeric_ids_and_afterok_graph() -> None:
    spec = create_prad_campaign_spec(
        source_snapshot=_snapshot(),
        mode="smoke",
        campaign_root="/campaign",
    )
    counter = iter(range(1000, 1000 + len(spec["tasks"])))
    ledger = submit_prad_plan(
        campaign_spec_path="/campaign/campaign_spec.json",
        spec=spec,
        executor=lambda _: f"{next(counter)};tigris",
    )
    assert len(ledger["jobs"]) == len(spec["tasks"])
    assert all(row["job_id"].isdigit() for row in ledger["jobs"])
    oracle = next(row for row in ledger["jobs"] if row["task"] == "E2_oracle")
    jobs_by_task = {row["task"]: row for row in ledger["jobs"]}
    oracle_spec = next(row for row in spec["tasks"] if row["name"] == "E2_oracle")
    expected = (
        "--dependency=afterok:"
        + ":".join(
            jobs_by_task[name]["job_id"] for name in oracle_spec["dependencies"]
        )
    )
    assert expected in oracle["command"]

    resumed_counter = iter(range(3002, 3002 + len(spec["tasks"])))
    resumed = submit_prad_plan(
        campaign_spec_path="/campaign/campaign_spec.json",
        spec=spec,
        existing_jobs=ledger["jobs"][:2],
        executor=lambda _: str(next(resumed_counter)),
    )
    assert resumed["jobs"][:2] == ledger["jobs"][:2]
    assert resumed["jobs"][2]["job_id"] == "3002"


def test_production_requests_are_bound_to_exact_smoke_resource_evidence() -> None:
    smoke = create_prad_campaign_spec(
        source_snapshot=_snapshot(), mode="smoke", campaign_root="/smoke"
    )
    counter = iter(range(2000, 2000 + len(smoke["tasks"])))
    ledger = submit_prad_plan(
        campaign_spec_path="/smoke/campaign_spec.json",
        spec=smoke,
        executor=lambda _: str(next(counter)),
    )
    usage = {
        row["job_id"]: {
            "state": "COMPLETED",
            "elapsed_seconds": 10,
            "max_rss_bytes": 1024,
            "allocated_cpus": 4,
        }
        for row in ledger["jobs"]
    }
    requests = {
        task.name: {
            "cpus": task.cpus,
            "memory": task.memory,
            "walltime": task.walltime,
            "gpu": task.gpu,
            "array": task.array,
        }
        for task in prad_tasks(smoke=False)
    }
    requests["E0_baseline"]["memory"] = "128G"
    dry_run = with_content_hash(
        {
            "contract": "hlt_classification_prad_dry_run_v1",
            "schema_version": 1,
            "campaign_spec_sha256": smoke["content_hash"],
            "mutated": False,
            "plan": render_prad_submission_plan(
                campaign_spec_path="/smoke/campaign_spec.json", spec=smoke
            ),
        }
    )
    task_lookup = {row["name"]: row for row in smoke["tasks"]}

    def attestation_count(task: str) -> int:
        value = task_lookup[task]["array"]
        if value is None:
            return 1
        bounds = value.split("%", 1)[0].split("-", 1)
        return int(bounds[1]) - int(bounds[0]) + 1

    monitor = with_content_hash(
        {
            "contract": "hlt_classification_prad_monitor_report_v1",
            "schema_version": 1,
            "campaign_spec_sha256": smoke["content_hash"],
            "submission_ledger_sha256": ledger["content_hash"],
            "jobs": [
                {
                    "task": row["task"],
                    "job_id": row["job_id"],
                    "state": "COMPLETED",
                    "attestations": ["c" * 64] * attestation_count(row["task"]),
                    "reusable": True,
                }
                for row in ledger["jobs"]
            ],
        }
    )
    evidence = build_prad_resource_evidence(
        smoke_spec=smoke,
        submission_ledger=ledger,
        dry_run_report=dry_run,
        monitor_report=monitor,
        usage_by_job_id=usage,
        production_requests=requests,
        campaign_artifact_bytes=4096,
        measurement_host="tigris",
    )
    storage = build_prad_storage_evidence(
        resource_evidence=evidence,
        available_bytes=500 * 1024**3,
        projected_peak_bytes=200 * 1024**3,
        required_free_after_peak_bytes=100 * 1024**3,
        measurement_host="tigris",
        measurement_path="/home/ryreu/atlas/HLT_Classification/artifacts",
    )
    production = create_prad_campaign_spec(
        source_snapshot=_snapshot(),
        mode="production",
        campaign_root="/production",
        production_authorized=True,
        resource_evidence=evidence,
        storage_evidence=storage,
    )
    baseline = next(row for row in production["tasks"] if row["name"] == "E0_baseline")
    assert baseline["memory"] == "128G"
    assert production["production_evidence"]["resource_evidence_sha256"] == evidence["content_hash"]
