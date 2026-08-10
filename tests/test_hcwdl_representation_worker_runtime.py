from __future__ import annotations

import importlib.machinery
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.scouting import hcwdl_representation_production as production
from hlt_classification.scouting import hcwdl_representation_task_runtime as task_runtime
from hlt_classification.scouting import hcwdl_representation_worker_runtime as worker
from hlt_classification.scouting.hcwdl_representation_targets import (
    KERNEL_RESOURCE_NAMES, build_target_forward_spec,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments], cwd=repository, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _clean_repository(tmp_path: Path) -> tuple[Path, dict]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "runtime@example.invalid")
    _git(repository, "config", "user.name", "Runtime Test")
    (repository / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "runtime.py")
    _git(repository, "commit", "-m", "runtime fixture")
    return repository, capture_source_snapshot(repository)


def _patch_nonsource_measurements(
    monkeypatch: pytest.MonkeyPatch, *, weaver_sha256: str,
) -> None:
    monkeypatch.setattr(worker, "_validate_import_origin", lambda _path: None)
    monkeypatch.setattr(worker, "_require_no_user_site", lambda: None)
    monkeypatch.setattr(worker, "_active_conda_environment", lambda _name: {
        "environment": "atlas_kd_tigris",
        "prefix": "/conda/envs/atlas_kd_tigris",
        "python_executable": "/conda/envs/atlas_kd_tigris/bin/python",
    })
    monkeypatch.setattr(worker, "measure_weaver_runtime_source", lambda: {
        "distributions": [{"name": "weaver-core", "version": "fixture"}],
        "files": [{"path": "runtime.py", "bytes": 1, "sha256": "1" * 64}],
        "sha256": weaver_sha256,
    })
    monkeypatch.setattr(worker, "_package_inventory", lambda _torch, _weaver: {
        "python": "3.10.0", "torch": "fixture", "cuda": "unavailable",
        "cudnn": "unavailable", "numpy": "fixture", "awkward": "fixture",
        "uproot": "fixture", "weaver": f"source_sha256={weaver_sha256}",
    })


def test_live_worker_rejects_mutated_clean_checkout_before_runtime_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, snapshot = _clean_repository(tmp_path)
    weaver_sha256 = "a" * 64
    _patch_nonsource_measurements(monkeypatch, weaver_sha256=weaver_sha256)
    monkeypatch.setenv("PROJECT_DIR", str(repository))
    monkeypatch.chdir(repository)

    live = worker.measure_live_worker_runtime(
        project_dir=repository,
        expected_conda_environment="atlas_kd_tigris",
        expected_source_commit=snapshot["git_commit"],
        expected_source_snapshot_sha256=snapshot["source_snapshot_sha256"],
        expected_weaver_runtime_sha256=weaver_sha256,
        row_device="cpu", resource_class="cpu_small",
        deterministic_worker=False,
    )
    assert live["source_snapshot_sha256"] == snapshot["source_snapshot_sha256"]

    with pytest.raises(PermissionError, match="clean source checkout differs"):
        worker.measure_live_worker_runtime(
            project_dir=repository,
            expected_conda_environment="atlas_kd_tigris",
            expected_source_commit=snapshot["git_commit"],
            expected_source_snapshot_sha256="f" * 64,
            expected_weaver_runtime_sha256=weaver_sha256,
            row_device="cpu", resource_class="cpu_small",
            deterministic_worker=False,
        )

    (repository / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is dirty"):
        worker.measure_live_worker_runtime(
            project_dir=repository,
            expected_conda_environment="atlas_kd_tigris",
            expected_source_commit=snapshot["git_commit"],
            expected_source_snapshot_sha256=snapshot["source_snapshot_sha256"],
            expected_weaver_runtime_sha256=weaver_sha256,
            row_device="cpu", resource_class="cpu_small",
            deterministic_worker=False,
        )


def test_active_conda_environment_rejects_wrong_name_before_prefix_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "wrong")
    monkeypatch.setenv("CONDA_PREFIX", "/conda/envs/atlas_kd_tigris")
    with pytest.raises(PermissionError, match="Conda environment differs"):
        worker._active_conda_environment("atlas_kd_tigris")


def test_weaver_source_signature_changes_when_an_importable_byte_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "weaver"
    package.mkdir()
    source = package / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    specification = importlib.machinery.ModuleSpec("weaver", loader=None, is_package=True)
    specification.submodule_search_locations = [str(package)]
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: specification)
    monkeypatch.setattr(
        worker.importlib.metadata, "packages_distributions",
        lambda: {"weaver": ["weaver-core"]},
    )
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: "1.2.3")

    first = worker.measure_weaver_runtime_source()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = worker.measure_weaver_runtime_source()
    assert first["sha256"] != second["sha256"]


def _live_fixture() -> dict:
    return with_content_hash({
        "contract": worker.LIVE_WORKER_RUNTIME_DOMAIN,
        "schema_version": 1,
        "project_dir": "/project", "source_commit": "1" * 40,
        "source_snapshot_sha256": "2" * 64,
        "conda": {
            "environment": "atlas_kd_tigris",
            "prefix": "/conda/envs/atlas_kd_tigris",
            "python_executable": "/conda/envs/atlas_kd_tigris/bin/python",
        },
        "python_no_user_site": True,
        "packages": {
            "python": "3.10.0", "torch": "2.0", "cuda": "12.4",
            "cudnn": "90100", "numpy": "1", "awkward": "2", "uproot": "5",
            "weaver": "weaver-core==1;source_sha256=" + "3" * 64,
        },
        "weaver_runtime_sha256": "3" * 64,
        "resource_class": "gpu_target", "row_device": "cuda",
        "device": {
            "request": "gpu:gh200:1", "architecture": "Hopper",
            "model": "NVIDIA GH200 480GB", "compute_capability": "9.0",
            "driver": "550.54.15", "runtime": "12.4",
        },
        "gpu_uuid": "GPU-live-measured-uuid",
        "deterministic_worker": True,
        "backend": {
            "deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "matmul_tf32": False,
            "cudnn_tf32": False, "reduced_precision_fp32_reduction": False,
            "cublas_workspace_config": ":4096:8", "rng_states_sha256": "4" * 64,
        },
    })


def test_live_row_signature_rejects_wrong_device_policy_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_fixture()
    task = {
        "resource_class": "gpu_target", "deterministic_worker": True,
    }
    spec = {"source_commit": "1" * 40}
    facts = {
        "project_dir": "/project", "conda_environment": "atlas_kd_tigris",
        "source_snapshot_sha256": "2" * 64,
        "weaver_runtime_sha256": "3" * 64, "device": "cpu",
    }
    row = {"device": "cuda", "runtime_signature_sha256": "0" * 64}
    with pytest.raises(PermissionError, match="device policy"):
        worker.validate_live_task_runtime(
            spec=spec, binding={"runtime_facts": facts}, task=task,
            runtime_row=row, deterministic_worker=True,
        )

    facts["device"] = "cuda"
    monkeypatch.setattr(worker, "measure_live_worker_runtime", lambda **_kwargs: live)
    with pytest.raises(PermissionError, match="row runtime signature differs"):
        worker.validate_live_task_runtime(
            spec=spec, binding={"runtime_facts": facts}, task=task,
            runtime_row=row, deterministic_worker=True,
        )
    row["runtime_signature_sha256"] = worker.build_row_runtime_signature(live)[
        "content_hash"
    ]
    assert worker.validate_live_task_runtime(
        spec=spec, binding={"runtime_facts": facts}, task=task,
        runtime_row=row, deterministic_worker=True,
    ) == live


def test_worker_runtime_measurement_is_immutable_and_explicitly_nonauthorizing(
    tmp_path: Path,
) -> None:
    live = _live_fixture()
    request = {
        "cpus": 8, "memory": "320G", "walltime": "24:00:00",
        "gpu": "gpu:gh200:1",
    }
    result = worker.build_worker_runtime_measurement(
        campaign_spec_sha256="5" * 64, data_root=tmp_path,
        resource_class="gpu_target", resource_request=request,
        live_worker_runtime=live,
    )
    assert worker.validate_worker_runtime_measurement(result) == result["content_hash"]
    assert result["resource_request"] == request
    assert result["row_runtime_signature_sha256"] == worker.build_row_runtime_signature(
        live
    )["content_hash"]
    assert result["scheduler_mutated"] is False
    assert result["scientific_authorization"] is False
    assert result["authorizes_tigris_or_pilot"] is False
    assert result["final_role_accessed"] is False


def test_measurement_cli_has_no_scheduler_or_authorization_surface() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = (
        repository / "scripts" / "measure_hcwdl_representation_worker_runtime.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("sbatch", "scancel", "scheduler(", "final_test", "authorization_phrase"):
        assert forbidden not in source


def test_dispatch_checks_live_provenance_before_registered_input_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "artifact_paths": {"runtime_binding": "/campaign/runtime.json"},
        "project_dir": "/project", "source_commit": "1" * 40,
        "tasks": [{
            "task_key": "tap", "kind": "tap_schema", "dependencies": (),
            "resource_class": "cpu_small", "array": None, "graph_node": None,
            "logical_bank": None, "target_purpose": None,
            "deterministic_worker": False, "array_registry": None,
            "registered_inputs": (), "registered_outputs": (),
        }],
    }
    binding = {"runtime_facts": {"project_dir": "/project"}}
    row = {"device": "cpu", "inputs": {"scientific": {"path": "/data/root"}}}
    monkeypatch.setattr(task_runtime, "load_runtime_binding", lambda _spec: binding)
    monkeypatch.setattr(task_runtime, "_validate_environment", lambda _spec, _binding: None)
    monkeypatch.setattr(task_runtime, "resolve_runtime_row", lambda *_args, **_kwargs: row)
    opened = []
    monkeypatch.setattr(
        task_runtime, "_validate_input_bytes",
        lambda *_args, **_kwargs: opened.append(True),
    )
    monkeypatch.setattr(
        worker, "validate_live_task_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("live row runtime signature differs")
        ),
    )
    with pytest.raises(PermissionError, match="runtime signature differs"):
        task_runtime.execute_registered_task(
            spec=spec, task_key="tap", array_index=None,
            deterministic_worker=False,
        )
    assert opened == []


def _target_forward_spec(live: dict) -> dict:
    backend = live["backend"]
    payload = {
        "teacher": {
            "source_kind": "imported_checkpoint",
            "checkpoint_byte_sha256": "5" * 64,
            "checkpoint_logical_sha256": "6" * 64,
            "model_config_sha256": "7" * 64,
            "architecture_sha256": "8" * 64,
            "tap_sha256": "9" * 64,
            "kernel_resources_sha256": "a" * 64,
            "kernel_array_logical_hashes": {
                name: canonical_sha256({"kernel": name})
                for name in KERNEL_RESOURCE_NAMES
            },
        },
        "producer": {
            "source_commit": live["source_commit"],
            "source_snapshot_sha256": live["source_snapshot_sha256"],
            "packages": live["packages"],
        },
        "device": live["device"],
        "precision": {
            "parameters": "float32", "inputs": "float32",
            "activations": "float32", "autocast": False,
            "matmul_tf32": False, "cudnn_tf32": False,
            "reduced_precision_fp32_reduction": False, "output_order": "C",
        },
        "determinism": {
            "deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "rng_states_sha256": backend["rng_states_sha256"],
        },
        "batching": {
            "batch_size": 256, "order": "source_file_id_then_source_entry_v1",
            "cross_source_batches": False, "final_short_batch_per_source": True,
            "padding": False, "row_duplication": False,
        },
        "implementation": {
            "input_decoding_sha256": "b" * 64,
            "feature_layout_sha256": "c" * 64,
            "trimmer_sha256": "d" * 64,
            "family_code_sha256": "e" * 64,
            "surface_capture_sha256": "f" * 64,
            "sketch_arithmetic_sha256": "0" * 64,
            "teacher_input_fields": sorted(
                ["features", "family_codes", "mask", "vectors"]
            ),
        },
        "source_partitions": ["source_000000"],
    }
    return build_target_forward_spec(parents={"logical_bank": "1" * 64}, payload=payload)


def test_target_execution_environment_uses_live_measured_producer_device_and_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_fixture()
    forward_spec = _target_forward_spec(live)
    monkeypatch.setattr(
        worker, "configure_deterministic_worker_backend",
        lambda _torch: live["backend"],
    )
    environment = worker.build_live_target_runtime_environment(
        forward_spec, live_worker_runtime=live,
    )
    assert environment["producer"]["source_commit"] == live["source_commit"]
    assert environment["device"]["gpu_uuid"] == "GPU-live-measured-uuid"
    assert environment["device"] is not forward_spec["payload"]["device"]

    forged_live = with_content_hash({
        **{name: value for name, value in live.items() if name != "content_hash"},
        "source_commit": "f" * 40,
    })
    with pytest.raises(PermissionError, match="measured target producer"):
        worker.build_live_target_runtime_environment(
            forward_spec, live_worker_runtime=forged_live,
        )

    source = inspect.getsource(production.target_build_adapter)
    assert "runtime_environment=live_runtime_environment" in source
    assert "declared_runtime != frozen_runtime" in source


def test_deterministic_backend_measurement_is_stable_and_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    cpu_rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    reductions = {
        name: getattr(torch.backends.cuda.matmul, name)
        for name in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        )
        if hasattr(torch.backends.cuda.matmul, name)
    }
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        first = worker.configure_deterministic_worker_backend(torch)
        torch.rand(5)
        second = worker.configure_deterministic_worker_backend(torch)
        assert first == second
        assert first["rng_states_sha256"] == second["rng_states_sha256"]
        assert first["deterministic_algorithms"] is True
        assert first["reduced_precision_fp32_reduction"] is False
    finally:
        torch.set_rng_state(cpu_rng)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        for name, value in reductions.items():
            setattr(torch.backends.cuda.matmul, name, value)
