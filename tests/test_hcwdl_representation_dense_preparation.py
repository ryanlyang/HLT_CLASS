from __future__ import annotations

from pathlib import Path

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
import hlt_classification.scouting.hcwdl_representation_candidate as candidate
import hlt_classification.scouting.hcwdl_representation_dense_preparation as preparation
import hlt_classification.scouting.hcwdl_representation_resources as resources


H = "1" * 64
MEASURED = "2" * 40
CURRENT = "3" * 40


def test_runtime_target_forward_specs_use_registered_logical_names() -> None:
    forward_specs = {
        "RSET_D100:screen": {"content_hash": "4" * 64},
        "RREL_D95:screen": {"content_hash": "5" * 64},
    }
    assert preparation._runtime_target_forward_specs(forward_specs) == {
        "${target_forward_spec:RSET_D100:screen}": forward_specs["RSET_D100:screen"],
        "${target_forward_spec:RREL_D95:screen}": forward_specs["RREL_D95:screen"],
    }


def _live(resource_class: str) -> dict:
    cuda = resource_class.startswith("gpu_")
    return with_content_hash({
        "contract": "HCWDL_REPRESENTATION_LIVE_WORKER_RUNTIME/v1",
        "schema_version": 1,
        "project_dir": "/project",
        "source_commit": MEASURED,
        "source_snapshot_sha256": "4" * 64,
        "conda": {
            "environment": "atlas_kd_tigris",
            "prefix": "/conda/envs/atlas_kd_tigris",
            "python_executable": "/conda/envs/atlas_kd_tigris/bin/python",
        },
        "python_no_user_site": True,
        "packages": {
            name: "x" for name in (
                "python", "torch", "cuda", "cudnn", "numpy", "awkward",
                "uproot", "weaver",
            )
        },
        "weaver_runtime_sha256": "5" * 64,
        "resource_class": resource_class,
        "row_device": "cuda" if cuda else "cpu",
        "device": ({
            "request": "cuda", "architecture": "hopper", "model": "GH200",
            "compute_capability": "9.0", "driver": "x", "runtime": "x",
        } if cuda else {"type": "cpu"}),
        "gpu_uuid": "GPU-test" if cuda else None,
        "deterministic_worker": resource_class == "gpu_target",
        "backend": {"mode": "test"},
    })


def test_runtime_projection_reuses_jobs_but_binds_current_source(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    measurements = tmp_path / "measurements"
    project.mkdir(); data.mkdir()
    requests = {}
    for resource_class in preparation.DENSE_RESOURCE_CLASSES:
        request = {
            "cpus": 4 if resource_class.startswith("gpu_") else 2,
            "memory": "64G" if resource_class.startswith("gpu_") else "8G",
            "walltime": "02:00:00", "gpu": (
                "gpu:gh200:1" if resource_class.startswith("gpu_") else None
            ),
        }
        requests[resource_class] = request
        path = measurements / resource_class / "worker_runtime_measurement.json"
        path.parent.mkdir(parents=True)
        write_immutable_json(path, with_content_hash({
            "contract": "fixture", "schema_version": 1,
            "resource_class": resource_class, "resource_request": request,
            "runtime_facts": {"data_root": str(data.resolve())},
            "live_worker_runtime": _live(resource_class),
        }))
    monkeypatch.setattr(
        preparation, "capture_source_snapshot",
        lambda *_args, **_kwargs: {
            "git_commit": CURRENT, "source_snapshot_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(preparation, "validate_source_snapshot_payload", lambda _v: "6" * 64)
    monkeypatch.setattr(preparation, "validate_worker_runtime_measurement", lambda _v: H)
    monkeypatch.setattr(preparation, "dense_resource_measurement_source_commit", lambda _v: MEASURED)
    facts, signatures, values = preparation._project_runtime_registry(
        project_dir=project.resolve(), current_source_commit=CURRENT,
        compatible_profile={"requests": requests},
        measurement_root=measurements.resolve(),
        signature_root=tmp_path / "signatures", data_root=data.resolve(),
    )
    assert facts["source_snapshot_sha256"] == "6" * 64
    assert set(signatures) == set(preparation.DENSE_RESOURCE_CLASSES)
    assert all(value["source_commit"] == CURRENT for value in values.values())
    assert all(value["source_snapshot_sha256"] == "6" * 64 for value in values.values())


def test_dense_candidate_uses_measured_source_for_storage(
    tmp_path: Path, monkeypatch,
) -> None:
    inventory_path = tmp_path / "template.json"
    write_immutable_json(inventory_path, {"fixture": True})
    profile = {"measurement_environment": {"source_commit": MEASURED}}
    storage = {"fixture": "storage"}
    binding = {"content_hash": "7" * 64, "tasks": []}
    plan = {
        "content_hash": "8" * 64, "campaign_identity": {},
        "campaign_identity_sha256": "9" * 64,
    }
    spec = {
        "planning_only": True, "live_submission_authorized": False,
        "submission_authorization": None,
        "submission_authorization_sha256": None,
        "runtime_status": "immutable", "mode": "smoke",
        "disposition": "dense_training_only", "source_commit": CURRENT,
        "project_dir": str(tmp_path), "campaign_root": str(tmp_path),
        "resource_profile": profile,
        "storage_estimate": storage,
        "fixed_size_inventory": {"path": str(inventory_path)},
        "tigris_acceptance": None, "representation_recipe_sha256": H,
        "graph_sha256": H, "parent_import_sha256": H,
        "resource_profile_sha256": "a" * 64,
        "storage_estimate_sha256": "b" * 64,
        "fixed_size_inventory_sha256": "c" * 64,
        "tigris_acceptance_sha256": None,
        "command_plan_sha256": plan["content_hash"],
        "runtime_binding_sha256": binding["content_hash"], "tasks": [],
    }
    observed: list[str] = []
    monkeypatch.setattr(resources, "dense_resource_measurement_source_commit", lambda _v: MEASURED)
    monkeypatch.setattr(resources, "validate_dense_measured_profile", lambda *_a, **_k: "a" * 64)
    monkeypatch.setattr(
        resources, "validate_dense_storage_template",
        lambda *_a, **kwargs: observed.append(kwargs["expected_source_commit"]) or "c" * 64,
    )
    monkeypatch.setattr(
        resources, "validate_dense_storage_estimate",
        lambda *_a, **kwargs: observed.append(kwargs["expected_source_commit"]) or "b" * 64,
    )
    monkeypatch.setattr(resources, "validate_dense_storage_availability", lambda *_a, **_k: 1)
    import hlt_classification.scouting.hcwdl_representation_campaign as campaign
    import hlt_classification.scouting.hcwdl_representation_runtime_binding as binding_module
    import hlt_classification.scouting.hcwdl_representation_runtime_rows as rows
    monkeypatch.setattr(campaign, "validate_campaign_spec", lambda *_a, **_k: H)
    monkeypatch.setattr(campaign, "validate_source_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(campaign, "validate_command_plan", lambda *_a, **_k: H)
    monkeypatch.setattr(campaign, "build_command_plan", lambda _spec: plan)
    monkeypatch.setattr(binding_module, "validate_runtime_binding", lambda *_a, **_k: "7" * 64)
    monkeypatch.setattr(rows, "validate_bound_runtime_task_rows", lambda *_a, **_k: None)
    summary = candidate._strict_gate_summary(
        planning_spec=spec, command_plan=plan, runtime_binding=binding,
    )
    assert summary["source_commit"] == CURRENT
    assert observed == [MEASURED, MEASURED]
