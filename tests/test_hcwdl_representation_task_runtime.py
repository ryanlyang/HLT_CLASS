from __future__ import annotations

import importlib.util
from pathlib import Path
import signal
import sys

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, sha256_file, validate_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    LOCAL_SMOKE_CONTRACT, CampaignTask, create_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    LOCAL_PLANNING_WORK_CONTRACT, LOCAL_SEMANTIC_COVERAGE,
    PRODUCTION_ADAPTERS,
    PRODUCTION_ADAPTER_CONTRACT,
    build_local_planning_handlers,
    execute_local_planning_work,
)
from hlt_classification.scouting.hcwdl_representation_smoke import FINAL_ROLE_KINDS
from hlt_classification.scouting.hcwdl_representation_task_runtime import (
    RepresentationPreemptionMonitor, TASK_KINDS, execute_registered_task,
)
from hlt_classification.scouting import hcwdl_representation_task_runtime as task_runtime
from hlt_classification.scouting import hcwdl_representation_runtime_adapters as runtime_adapters
from hlt_classification.scouting.hcwdl_representation_workflow import (
    RepresentationWorkflow, array_indices, exercise_registered_rows,
)
from hlt_classification.scouting import hcwdl_representation_workflow as workflow_module


def _spec(tmp_path: Path, disposition: str = "combined_confirmatory"):
    return create_campaign_spec(
        mode="pilot", campaign_root=tmp_path / "campaign",
        checkpoint_namespace=tmp_path / "checkpoints", project_dir="/project",
        source_commit="1" * 40, source_manifest_sha256="2" * 64,
        split_manifest_sha256="3" * 64, parent_import_sha256="4" * 64,
        representation_recipe_sha256="5" * 64, graph_sha256="6" * 64,
        disposition_sha256="7" * 64, disposition=disposition,
        role_counts={"train": 300_000, "validation": 100_000, "final_test": 100_000},
        final_source_partitions=2, combined_finalist_count=3,
    )


def _task(row) -> CampaignTask:
    return CampaignTask(**{
        **row,
        "dependencies": tuple(row["dependencies"]),
        "registered_inputs": tuple(row["registered_inputs"]),
        "registered_outputs": tuple(row["registered_outputs"]),
    })


def _runtime_row(task: CampaignTask, tmp_path: Path, parameters):
    return {
        "array_index": None,
        "device": "cpu",
        "inputs": {},
        "outputs": {
            name: str(tmp_path / f"output-{index}.json")
            for index, name in enumerate(task.registered_outputs)
        },
        "parameters": parameters,
        "runtime_signature_sha256": canonical_sha256(parameters),
    }


def test_preemption_monitor_preserves_exact_signal_identity() -> None:
    monitor = RepresentationPreemptionMonitor()
    usr1 = getattr(signal, "SIGUSR1", None)
    term = getattr(signal, "SIGTERM", None)
    if usr1 is None:
        pytest.skip("SIGUSR1 is unavailable on this platform")
    monitor._request(usr1, None)
    assert monitor.is_requested()
    assert monitor.observed_signals() == ("SIGUSR1",)
    assert monitor.observed_exact_usr1()
    if term is not None:
        monitor._request(term, None)
        assert monitor.observed_signals() == ("SIGUSR1", "SIGTERM")
        assert not monitor.observed_exact_usr1()


def test_workflow_propagates_the_live_executable_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    monkeypatch.setattr(
        workflow_module,
        "validate_campaign_spec",
        lambda _spec, *, executable: observed.append(executable),
    )
    RepresentationWorkflow({"tasks": []}, handlers={}, executable=True)
    RepresentationWorkflow({"tasks": []}, handlers={}, executable=False)
    assert observed == [True, False]


def test_local_parent_import_gate_exercises_current_v2_shape() -> None:
    work = runtime_adapters._local_parent_import_gate()
    assert work["parent_import_contract"] == (
        "HCWDL_REPRESENTATION_PARENT_IMPORT/v3"
    )
    assert work["parent_import_schema_version"] == 2
    assert work["nonauthorizing_synthetic_v3_fixture"] is True
    assert work["authority_files_reopened"] is False


def test_production_registry_is_closed_and_exhaustive() -> None:
    assert set(PRODUCTION_ADAPTERS) == TASK_KINDS
    assert all(callable(value) for value in PRODUCTION_ADAPTERS.values())


def test_local_semantic_coverage_is_closed_and_has_no_generic_success_surface() -> None:
    assert set(LOCAL_SEMANTIC_COVERAGE) == TASK_KINDS
    forbidden = ("roundtrip", "structural", "mock", "schema_only")
    assert all(
        not any(label in surface for label in forbidden)
        for surface in LOCAL_SEMANTIC_COVERAGE.values()
    )


def test_tap_schema_production_adapter_calls_real_api_and_authenticates_output(
    tmp_path: Path,
) -> None:
    task = CampaignTask(
        "tap", "tap_schema", (), "cpu_small",
        registered_outputs=("architecture/tap.json",),
    )
    parameters = {
        "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
        "task_kind": "tap_schema",
    }
    row = _runtime_row(task, tmp_path, parameters)
    result = PRODUCTION_ADAPTERS["tap_schema"]({}, task, None, row)
    output = Path(next(iter(row["outputs"].values())))
    assert output.is_file()
    assert result["operation"] == "tap_schema"
    from hlt_classification.models.hcwdl_surfaces import tap_schema

    import json
    assert json.loads(output.read_text(encoding="utf-8")) == tap_schema()
    # Exact idempotent reuse is allowed; changed bytes are not.
    assert PRODUCTION_ADAPTERS["tap_schema"]({}, task, None, row) == result
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs"):
        PRODUCTION_ADAPTERS["tap_schema"]({}, task, None, row)


def test_json_builder_adapter_calls_fixed_repository_builder_and_all_outputs(
    tmp_path: Path,
) -> None:
    task = CampaignTask(
        "confirmation-registry", "confirmation_registry", (), "cpu_small",
        registered_outputs=("confirmation/registry.json", "locks/frozen.json"),
    )
    parameters = {
        "adapter_contract": PRODUCTION_ADAPTER_CONTRACT,
        "task_kind": "confirmation_registry",
        "builder_arguments": {
            "screen_sha256": "a" * 64,
            "campaign_sha256": "b" * 64,
            "recipe_sha256": "c" * 64,
            "target_logical_bank_sha256": "d" * 64,
            "objectives": ["RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"],
        },
    }
    row = _runtime_row(task, tmp_path, parameters)
    result = PRODUCTION_ADAPTERS["confirmation_registry"]({}, task, None, row)
    assert set(result["registered_outputs"]) == set(task.registered_outputs)
    artifacts = [load_json(path) for path in map(Path, row["outputs"].values())]
    assert artifacts[0] == artifacts[1]
    validate_content_hash(
        artifacts[0], expected_contract="HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v1",
    )


def test_dispatch_authenticates_registered_input_bytes_before_adapter(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    campaign_path = Path(spec["campaign_root"]) / "campaign_spec.json"
    write_immutable_json(campaign_path, spec)
    parent = tmp_path / "parent.json"
    write_immutable_json(parent, {"value": 1})
    row = {
        "inputs": {
            "${campaign_spec}": {
                "path": str(campaign_path),
                "sha256": "0" * 64,
            },
            "${parent_import}": {
                "path": str(parent),
                "sha256": sha256_file(parent),
            },
        }
    }
    task_runtime._validate_input_bytes(row, spec=spec)
    parent.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="byte/hash identity differs"):
        task_runtime._validate_input_bytes(row, spec=spec)


def test_dispatch_directory_identity_binds_every_member_byte(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    campaign_path = Path(spec["campaign_root"]) / "campaign_spec.json"
    write_immutable_json(campaign_path, spec)
    directory = tmp_path / "registered-directory"
    manifest = directory / "manifest.json"
    payload = directory / "payload.bin"
    write_immutable_json(manifest, {"manifest": 1})
    payload.write_bytes(b"first payload")

    def inventory_sha256() -> str:
        return canonical_sha256([
            {
                "path": member.relative_to(directory).as_posix(),
                "bytes": member.stat().st_size,
                "sha256": sha256_file(member),
            }
            for member in sorted(directory.rglob("*"))
            if member.is_file()
        ])

    row = {
        "inputs": {
            "${campaign_spec}": {
                "path": str(campaign_path), "sha256": "0" * 64,
            },
            "${bundle}": {
                "path": str(directory), "sha256": inventory_sha256(),
            },
        }
    }
    task_runtime._validate_input_bytes(row, spec=spec)
    payload.write_bytes(b"second payload")
    with pytest.raises(ValueError, match="byte/hash identity differs"):
        task_runtime._validate_input_bytes(row, spec=spec)


def test_local_fixture_performs_real_work_for_every_nonfinal_task_and_array_row(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    forbidden = set(FINAL_ROLE_KINDS) | {"reservation", "finalist_lock"}
    results = []
    expected_rows = 0
    for raw in spec["tasks"]:
        task = _task(raw)
        for index in array_indices(task.array):
            if task.kind in forbidden:
                with pytest.raises(PermissionError, match="final/reservation"):
                    execute_local_planning_work(spec, task, index)
                continue
            expected_rows += 1
            result = execute_local_planning_work(spec, task, index)
            validate_content_hash(
                result, expected_contract=LOCAL_PLANNING_WORK_CONTRACT,
                expected_schema_version=1,
            )
            assert result["scientific_authorization"] is False
            assert result["final_role_accessed"] is False
            assert result["production_handler_invoked"] is False
            assert result["semantic_surface"] == LOCAL_SEMANTIC_COVERAGE[task.kind]
            assert result["semantic_fixture_executed"] is True
            assert result["generic_fallback"] is False
            assert result["work"]["work_kind"]
            assert result["work_sha256"] == canonical_sha256(result["work"])
            results.append((task.task_key, index, result["work"]["work_kind"]))
    assert len(results) == expected_rows
    assert len({(key, index) for key, index, _ in results}) == len(results)
    assert {work for _, _, work in results} >= {
        "full_representation_loss_backward",
        "real_target_generation_load_join_cleanup",
        "corrected_parent_loss_forward_backward",
        "fixed_spectral_feature_mean",
        "parent_import_contract_gate",
        "ascent_graph_and_control_registry_gate",
        "frozen_representation_recipe_gate",
        "within_class_shuffle_map_gate",
        "screen_aggregate_builder_gate",
        "confirmation_registry_builder_gate",
        "confirmation_aggregate_builder_gate",
        "zero_coefficient_gradient_disconnect_gate",
    }


def test_validation_only_disposition_runs_its_exact_reporting_builder(tmp_path: Path) -> None:
    spec = _spec(tmp_path, disposition="validation_only_parent_claim_consumed")
    raw = next(row for row in spec["tasks"] if row["kind"] == "validation_only_aggregate")
    result = execute_local_planning_work(spec, _task(raw), None)
    assert result["work"]["work_kind"] == "validation_only_aggregate_builder_gate"
    assert result["semantic_surface"] == LOCAL_SEMANTIC_COVERAGE[
        "validation_only_aggregate"
    ]


def test_public_local_registry_exercises_all_rows_without_final_role_access(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    handlers = build_local_planning_handlers({row["kind"] for row in spec["tasks"]})
    assert set(handlers) == {row["kind"] for row in spec["tasks"]}
    rows = exercise_registered_rows(spec, handlers=handlers)
    expected = sum(
        len(array_indices(row["array"])) for row in spec["tasks"]
    )
    assert len(rows) == expected
    synthetic_final = [
        row for row in rows if row["result"].get("synthetic_final_pipeline")
    ]
    assert synthetic_final
    assert {
        next(task["kind"] for task in spec["tasks"] if task["task_key"] == row["task_key"])
        for row in synthetic_final
    } == set(FINAL_ROLE_KINDS) | {"reservation", "finalist_lock"}
    for row in rows:
        result = row["result"]
        validate_content_hash(
            result, expected_contract=LOCAL_PLANNING_WORK_CONTRACT,
            expected_schema_version=1,
        )
        assert result["final_role_accessed"] is False
        assert result["scientific_authorization"] is False
        assert result["semantic_surface"] == LOCAL_SEMANTIC_COVERAGE[
            next(task["kind"] for task in spec["tasks"] if task["task_key"] == row["task_key"])
        ]
        assert result["semantic_fixture_executed"] is True
        assert result["generic_fallback"] is False
        assert result.get("structural_only") is not True
        if result.get("synthetic_final_pipeline"):
            assert result["full_shared_final_semantics_exercised"] is True
        else:
            assert result["work"]["work_kind"]


def test_registered_task_local_path_uses_real_work_and_preserves_worker_routing(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, disposition="validation_only_parent_claim_consumed")
    result = execute_registered_task(
        spec=spec, task_key="tap_schema", array_index=None,
        deterministic_worker=False, local_planning_fixture=True,
    )
    assert result["work"]["work_kind"] == "tap_schema_materialization"
    target = next(row for row in spec["tasks"] if row["kind"] == "target_build")
    target_result = execute_registered_task(
        spec=spec, task_key=target["task_key"], array_index=None,
        deterministic_worker=True, local_planning_fixture=True,
    )
    assert target_result["work"]["work_kind"] == "real_target_generation_load_join_cleanup"
    assert target_result["work"]["teacher_forward_calls"] == 1
    assert target_result["work"]["cleanup_validated"] is True
    with pytest.raises(PermissionError, match="wrong worker"):
        execute_registered_task(
            spec=spec, task_key=target["task_key"], array_index=None,
            deterministic_worker=False, local_planning_fixture=True,
        )


def test_local_smoke_cli_uses_public_real_work_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "campaign_spec.json"
    output = tmp_path / "local_smoke.json"
    probe = tmp_path / "scientific_probe.json"
    write_immutable_json(spec_path, spec)
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "run_hcwdl_representation_local_smoke.py"
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location("_test_hcwdl_local_smoke", script)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--campaign-spec", str(spec_path),
        "--scientific-probe-output", str(probe),
        "--output", str(output),
    ])
    assert module.main() == 0
    report = load_json(output)
    validate_content_hash(report, expected_contract=LOCAL_SMOKE_CONTRACT)
    assert report["registered_rows_exercised"] == sum(
        len(array_indices(row["array"])) for row in spec["tasks"]
    )
    assert report["all_registered_rows_real_semantics"] is True
    assert report["final_rows_structural_only"] == 0
    assert report["final_rows_synthetic_pipeline"] > 0
    assert report["final_role_accessed"] is False
    assert load_json(probe)["content_hash"] == report[
        "scientific_full_loss_probe_sha256"
    ]
