from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_validation_proxy as proxy
from hlt_classification.scouting.hcwdl_representation_contracts import (
    LIVE_WORKER_RUNTIME_DOMAIN,
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
    VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT,
    VALIDATION_PROXY_PROOF_CONTRACT,
)


def _authority(*, nonfinal_root: Path | None = None) -> dict:
    action_base = {
        "action_id": "validation_proxy",
        "kind": "validation_proxy",
        "dependencies": [],
        "worker_role": "deterministic",
        "resource_class": "gpu_final_prediction",
        "scalar_only": True,
        "array": None,
        "train_rows": 0,
        "validation_rows": 256,
        "final_rows": 0,
        "replicate_seed": None,
        "effective_batch_size": None,
        "maximum_optimizer_updates": 0,
        "execution_id": None,
        "target_identity": None,
        "mode": "acceptance",
        "campaign_task_kind": None,
        "final_role_access_authorized": False,
    }
    action = {
        **action_base,
        "action_spec_sha256": canonical_sha256(action_base),
    }
    payload = {
        "contract": NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        "schema_version": 1,
        "source_commit": "a" * 40,
        "representation_recipe_sha256": "b" * 64,
        "action_inputs_sha256": "c" * 64,
        "role_caps": {"train": 512, "validation": 256, "final_test": 0},
        "actions": {"validation_proxy": action},
        "effective_batch_size": 256,
        "maximum_optimizer_updates": 2,
        "bounded_action_execution_authorized": True,
        "arrays_authorized": False,
        "campaign_training_authorized": False,
        "reservation_authorized": False,
        "shared_final_authorized": False,
        "final_role_access_authorized": False,
        "pilot_submission_authorized": False,
        "scheduler_submission_authorized": False,
        "scheduler_mutated": False,
    }
    if nonfinal_root is not None:
        action_inputs = nonfinal_root / "action_inputs.json"
        write_immutable_json(
            action_inputs,
            with_content_hash({
                "contract": "VALIDATION_PROXY_ACTION_INPUT_FIXTURE/v1",
                "schema_version": 1,
            }),
        )
        payload["action_inputs"] = {
            "path": str(action_inputs.resolve()),
            "sha256": sha256_file(action_inputs),
        }
    return with_content_hash(payload)


def _authority_validator(value) -> str:
    return validate_content_hash(
        value,
        expected_contract=NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        expected_schema_version=1,
    )


def _identities() -> tuple[str, ...]:
    return tuple(
        canonical_sha256({"source_file_sha256": "1" * 64, "source_entry": index})
        for index in range(256)
    )


def _source_access(source_path: str = "validation/class_00.root"):
    return ({
        "source_path": source_path,
        "source_file_sha256": "1" * 64,
        "tree": "tree",
        "entry_start": 0,
        "entry_stop": 256,
    },)


def _bindings():
    lineages = _model_source_lineages()
    return {
        path: {
            "view_id": proxy.VALIDATION_PROXY_VIEW_REGISTRY[path],
            "model_id": proxy.VALIDATION_PROXY_MODEL_IDS[path],
            "checkpoint_sha256": str(index + 2) * 64,
            "model_source_lineage_sha256": lineages[
                proxy.VALIDATION_PROXY_MODEL_IDS[path]
            ]["content_hash"],
        }
        for index, path in enumerate(proxy.VALIDATION_PROXY_PATHS)
    }


def _live_runtime() -> dict:
    return with_content_hash({
        "contract": LIVE_WORKER_RUNTIME_DOMAIN,
        "schema_version": 1,
        "project_dir": "/frozen/nonfinal/project",
        "source_commit": "a" * 40,
        "source_snapshot_sha256": "d" * 64,
        "conda": {
            "environment": "atlas_kd_tigris",
            "prefix": "/frozen/conda",
            "python_executable": "/frozen/conda/bin/python",
        },
        "python_no_user_site": True,
        "packages": {
            "python": "3.10", "torch": "2.13", "cuda": "12.6",
            "cudnn": "9", "numpy": "2", "awkward": "2",
            "uproot": "5", "weaver": "weaver-core;source_sha256=" + "e" * 64,
        },
        "weaver_runtime_sha256": "e" * 64,
        "resource_class": "gpu_final_prediction",
        "row_device": "cuda",
        "device": {
            "request": "gpu:gh200:1", "architecture": "Hopper",
            "model": "NVIDIA GH200", "compute_capability": "9.0",
            "driver": "570", "runtime": "12.6",
        },
        "gpu_uuid": "GPU-test",
        "deterministic_worker": True,
        "backend": {"deterministic_algorithms": True},
    })


def _hex_sha(index: int) -> str:
    return format(index % 16, "x") * 64


def _model_source_lineages() -> dict[str, dict]:
    records = {}
    for index, path in enumerate(proxy.VALIDATION_PROXY_PATHS):
        node_id = proxy.VALIDATION_PROXY_MODEL_IDS[path]
        records[node_id] = with_content_hash({
            "kind": "validation_proxy_model_source",
            "schema_version": 1,
            "node_id": node_id,
            "source_kind": "authenticated_pmard_parent_v1",
            "parent_import_row_sha256": _hex_sha(index + 5),
            "wrapper_report_content_sha256": _hex_sha(index + 8),
            "wrapper_report_byte_sha256": _hex_sha(index + 11),
            "engine_report_content_sha256": _hex_sha(index + 4),
            "engine_report_byte_sha256": _hex_sha(index + 7),
            "wrapper_execution_config_sha256": _hex_sha(index + 10),
            "engine_execution_config_sha256": _hex_sha(index + 10),
            "engine_config_sha256": _hex_sha(index + 10),
            "model_extraction_sha256": _hex_sha(index + 13),
            "checkpoint_sha256": _hex_sha(index + 2),
        })
    return records


def _input_lineage(authority: dict) -> dict:
    source_lineages = _model_source_lineages()
    reports = {
        node_id: row["wrapper_report_content_sha256"]
        for node_id, row in source_lineages.items()
    }
    return proxy.build_validation_proxy_input_lineage(
        authority_sha256=authority["content_hash"],
        action_inputs_sha256=authority["action_inputs_sha256"],
        source_runtime_row_sha256="1" * 64,
        action_assembly_sha256="2" * 64,
        bounded_row_selection_sha256="3" * 64,
        parent_campaign_spec_sha256="4" * 64,
        source_manifest_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        matcher_resources_sha256="7" * 64,
        validation_assignment_manifest_sha256="8" * 64,
        parent_import_sha256="9" * 64,
        registered_input_bytes_sha256={
            name: _hex_sha(index + 10)
            for index, name in enumerate(proxy.VALIDATION_PROXY_REGISTERED_INPUTS)
        },
        model_report_sha256=reports,
        model_source_lineage=source_lineages,
        live_worker_runtime=_live_runtime(),
    )


def _pipeline(
    *, emitted_label_path: str | None = None, omit_class: bool = False,
    source_path: str = "validation/class_00.root",
    nonfinal_root: Path | None = None,
):
    identities = _identities()
    calls: list[tuple] = []
    authority = _authority(nonfinal_root=nonfinal_root)

    def selection_reader(request):
        calls.append(("selection", request))
        labels = [index % 15 for index in range(256)]
        if omit_class:
            labels = [0 if label == 14 else label for label in labels]
        return proxy.ValidationReadResult(
            rows=tuple(
                {
                    "identity_digest": identity,
                    "source_path": source_path,
                    "source_file_sha256": "1" * 64,
                    "source_entry": index,
                    "label": labels[index],
                }
                for index, identity in enumerate(identities)
            ),
            source_accesses=_source_access(source_path),
        )

    def assignment_reader(request):
        calls.append(("assignment", request))
        assert request.labels_allowed is False
        return proxy.ValidationReadResult(
            rows=tuple({
                "identity_digest": identity,
                "assignment": {
                    "offline_index": np.asarray([index % 4], dtype=np.int16),
                    "confidence": np.asarray([1.0], dtype=np.float32),
                },
            } for index, identity in enumerate(identities)),
            source_accesses=_source_access(source_path),
        )

    readers = {}
    predictors = {}
    for path_index, path in enumerate(proxy.VALIDATION_PROXY_PATHS):
        def reader(request, assignments, *, _path=path):
            calls.append(("reader", _path, request, assignments))
            assert request.role == "validation"
            assert request.path == _path
            assert request.labels_allowed is False
            assert not set(request.projected_branches) & set(proxy.LABEL_BRANCHES)
            assert (assignments is not None) is (_path == "shell_exact")
            rows = []
            for index, identity in enumerate(identities):
                model_inputs = {
                    "features": np.asarray([index, len(_path)], dtype=np.float32),
                }
                if emitted_label_path == _path and index == 0:
                    model_inputs["label"] = 0
                rows.append({
                    "identity_digest": identity,
                    "model_inputs": model_inputs,
                })
            return proxy.ValidationReadResult(
                rows=tuple(rows), source_accesses=_source_access(source_path),
            )

        def predictor(request, rows, *, _offset=path_index):
            calls.append(("predictor", request.path, request, rows))
            assert all(isinstance(row, proxy.ValidationModelRow) for row in rows)
            assert all(not hasattr(row, "label") for row in rows)
            logits = np.empty((len(rows), 15), dtype=np.float32)
            for row_index in range(len(rows)):
                logits[row_index] = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
                logits[row_index, (row_index + _offset) % 15] += np.float32(2.0)
            return logits

        readers[path] = reader
        predictors[path] = predictor
    return {
        "authority": authority,
        "authority_validator": _authority_validator,
        "selection_reader": selection_reader,
        "assignment_reader": assignment_reader,
        "stream_readers": readers,
        "predictors": predictors,
        "model_bindings": _bindings(),
        "input_lineage": _input_lineage(authority),
    }, calls


@pytest.fixture
def bounded_bootstrap(monkeypatch):
    # The production constant is asserted separately.  This keeps the focused
    # unit test fast while exercising the real stratified-bootstrap dataflow.
    monkeypatch.setattr(proxy, "VALIDATION_PROXY_BOOTSTRAP_REPLICATES", 4)


def test_validation_proxy_executes_separated_deep_pipeline(
    tmp_path: Path, bounded_bootstrap,
) -> None:
    nonfinal_root = tmp_path / "acceptance" / "nonfinal"
    arguments, calls = _pipeline(nonfinal_root=nonfinal_root)
    result = proxy.run_validation_proxy_action(**arguments)

    assert result["contract"] == VALIDATION_PROXY_PROOF_CONTRACT
    assert result["schema_version"] == 1
    assert result["role"] == "validation"
    assert result["rows"] == 256
    assert result["view_registry"] == {
        "hlt": "D0c", "shell_exact": "D100", "native_offline": "TOFF",
    }
    assert result["selection"]["class_counts"] == [18] + [17] * 14
    assert result["selection"]["raw_labels_published"] is False
    assert all("label" not in row for row in result["selection"]["selected_rows"])
    assert set(result["access_records"]) == {
        "selection", "assignment", "hlt", "shell_exact", "native_offline",
    }
    assert all(
        row["contract"] == VALIDATION_PROXY_BRANCH_ACCESS_CONTRACT
        for row in result["access_records"].values()
    )
    assert all(row["label_free"] for name, row in result["access_records"].items()
               if name != "selection")
    assert [row["comparison_id"] for row in result["paired_bootstraps"]] == [
        "D100_minus_D0c", "TOFF_minus_D0c",
    ]
    assert all(row["replicates"] == 4 for row in result["paired_bootstraps"])
    assert all(
        row["metric_order"][-1] == "acceptance_proxy_nonfinal_marker"
        for row in result["paired_bootstraps"]
    )
    assert all(row["scientific_authorization"] is False
               for row in result["paired_bootstraps"])
    assert result["final_role_accessed"] is False
    assert result["pilot_submission_authorized"] is False
    assert result["scientific_authorization"] is False
    assert result["scheduler_mutated"] is False
    assert result["raw_validation_labels_published"] is False
    assert proxy.validate_validation_proxy_proof_v2(
        result,
        authority=arguments["authority"],
        authority_validator=_authority_validator,
    ) == result["content_hash"]

    destination = nonfinal_root / "validation_proxy" / "result.json"
    reference = proxy.publish_validation_proxy_action_result(
        destination,
        result=result,
        authority=arguments["authority"],
        authority_validator=_authority_validator,
    )
    assert proxy.build_validation_proxy_proof_v2(
        result_reference=reference,
        authority=arguments["authority"],
        authority_validator=_authority_validator,
    ) == result
    assert sorted(path.name for path in (destination.parent / "access").iterdir()) == [
        "assignment.json", "hlt.json", "native_offline.json", "selection.json",
        "shell_exact.json",
    ]

    selection_request = calls[0][1]
    assert selection_request.labels_allowed is True
    assert set(proxy.LABEL_BRANCHES) <= set(selection_request.projected_branches)
    assert sum(call[0] == "predictor" for call in calls) == 3

    def has_raw_label(value) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in {"label", "labels"}
                or has_raw_label(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(has_raw_label(item) for item in value)
        return False

    assert not has_raw_label(result)
    for artifact in destination.parent.rglob("*.json"):
        import json

        assert not has_raw_label(json.loads(artifact.read_text(encoding="utf-8")))

    with pytest.raises(FileExistsError, match="already exists"):
        proxy.publish_validation_proxy_action_result(
            destination,
            result=result,
            authority=arguments["authority"],
            authority_validator=_authority_validator,
        )
    off_route = tmp_path / "occupied" / "validation_proxy" / "result.json"
    with pytest.raises(PermissionError, match="proof route differs"):
        proxy.publish_validation_proxy_action_result(
            off_route,
            result=result,
            authority=arguments["authority"],
            authority_validator=_authority_validator,
        )
    with pytest.raises(PermissionError, match="proof route differs"):
        proxy.build_validation_proxy_proof_v2(
            result_reference={
                "path": str(off_route.resolve()),
                "sha256": reference["sha256"],
            },
            authority=arguments["authority"],
            authority_validator=_authority_validator,
        )


def test_validation_proxy_fails_closed_on_label_leak_or_missing_class(
    bounded_bootstrap,
) -> None:
    leaked, _ = _pipeline(emitted_label_path="hlt")
    with pytest.raises(PermissionError, match="emitted labels"):
        proxy.run_validation_proxy_action(**leaked)

    incomplete, _ = _pipeline(omit_class=True)
    with pytest.raises(ValueError, match="all fifteen classes"):
        proxy.run_validation_proxy_action(**incomplete)

    shared_final, _ = _pipeline(
        source_path="campaign/shared_final/secret.root",
    )
    with pytest.raises(PermissionError, match="final route"):
        proxy.run_validation_proxy_action(**shared_final)


def test_generic_production_registry_adopts_validation_semantic_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.data.cache_contracts import write_immutable_json
    from hlt_classification.scouting import hcwdl_representation_nonfinal_runtime
    from hlt_classification.scouting import hcwdl_representation_validation_runtime
    from hlt_classification.scouting.hcwdl_representation_resources import (
        artifact_reference,
    )

    nonfinal_root = tmp_path / "acceptance" / "nonfinal"
    action_inputs_path = nonfinal_root / "action_inputs.json"
    write_immutable_json(
        action_inputs_path,
        with_content_hash({
            "contract": "VALIDATION_PROXY_ACTION_INPUT_FIXTURE/v1",
            "schema_version": 1,
        }),
    )
    authority = with_content_hash({
        "contract": "VALIDATION_PROXY_DISPATCH_FIXTURE/v1",
        "schema_version": 1,
        "action_inputs": artifact_reference(action_inputs_path),
    })
    authority_path = nonfinal_root / "authority.json"
    write_immutable_json(authority_path, authority)
    semantic = with_content_hash({
        "contract": "VALIDATION_PROXY_SEMANTIC_FIXTURE/v1",
        "schema_version": 1,
    })
    semantic_path = nonfinal_root / "validation_proxy" / "result.json"
    write_immutable_json(semantic_path, semantic)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(
        hcwdl_representation_validation_runtime,
        "execute_validation_proxy_production_action",
        lambda **kwargs: SimpleNamespace(
            semantic_result_path=semantic_path,
            semantic_result=semantic,
            workspace=nonfinal_root / "workspaces" / "validation_proxy",
            dependencies={},
            source_task_key="parent_import",
        ),
    )
    monkeypatch.setattr(
        hcwdl_representation_nonfinal_runtime,
        "_authority_static_context",
        lambda *args, **kwargs: ({"actions": {}}, {}, {}),
    )
    monkeypatch.setattr(
        hcwdl_representation_nonfinal_runtime,
        "_action_descriptor",
        lambda *args, **kwargs: (
            {"source_runtime_row_sha256": "f" * 64}, {}, {},
        ),
    )
    result = hcwdl_representation_nonfinal_runtime.execute_nonfinal_production_action(
        authority=authority,
        authority_path=authority_path,
        action_id="validation_proxy",
        project_dir=tmp_path,
        deterministic_worker=True,
    )
    assert result.semantic_outputs["primary"]["path"] == str(
        semantic_path.resolve()
    )

    off_route = tmp_path / "off-route" / "authority.json"
    write_immutable_json(off_route, authority)
    with pytest.raises(PermissionError, match="authority route differs"):
        hcwdl_representation_nonfinal_runtime.execute_nonfinal_production_action(
            authority=authority,
            authority_path=off_route,
            action_id="validation_proxy",
            project_dir=tmp_path,
            deterministic_worker=True,
        )
    assert result.source_task_key == "parent_import"
    assert result.scheduler_job_id == "12345"


@pytest.mark.parametrize(
    "lineage_field,swap_report",
    (("wrapper_report_content_sha256", True), ("engine_config_sha256", False)),
)
def test_validation_proxy_rejects_swapped_model_report_or_config_lineage(
    lineage_field: str, swap_report: bool, bounded_bootstrap,
) -> None:
    arguments, _ = _pipeline()
    result = proxy.run_validation_proxy_action(**arguments)
    forged = copy.deepcopy(result)
    lineage = forged["input_lineage"]
    model_source = lineage["model_source_lineage"]["D0c"]
    model_source[lineage_field] = "f" * 64
    if swap_report:
        lineage["model_report_sha256"]["D0c"] = "f" * 64
    lineage["model_source_lineage"]["D0c"] = with_content_hash(model_source)
    forged["input_lineage"] = with_content_hash(lineage)
    forged["input_lineage_sha256"] = forged["input_lineage"]["content_hash"]
    forged = with_content_hash(forged)
    with pytest.raises(PermissionError, match="model/input lineage differs"):
        proxy.validate_validation_proxy_proof_v2(
            forged,
            authority=arguments["authority"],
            authority_validator=_authority_validator,
        )


def test_validation_proxy_post_job_validation_is_label_free_and_static(
    monkeypatch: pytest.MonkeyPatch, bounded_bootstrap,
) -> None:
    arguments, _ = _pipeline()
    result = proxy.run_validation_proxy_action(**arguments)

    def forbidden(*args, **kwargs):
        raise AssertionError("post-job validation attempted an execution-time operation")

    monkeypatch.setattr(proxy, "_selection_artifact", forbidden)
    monkeypatch.setattr(proxy, "classification_metrics", forbidden)
    monkeypatch.setattr(proxy, "paired_classification_bootstrap", forbidden)
    assert proxy.validate_validation_proxy_proof_v2(
        result,
        authority=arguments["authority"],
        authority_validator=_authority_validator,
    ) == result["content_hash"]


def test_validation_proxy_production_constants_and_no_final_imports() -> None:
    assert proxy.VALIDATION_PROXY_BOOTSTRAP_REPLICATES == 2_000
    assert proxy.VALIDATION_PROXY_BOOTSTRAP_METRICS[:-1] == proxy.DEFAULT_METRICS
    assert proxy.VALIDATION_PROXY_BOOTSTRAP_METRICS[-1] == (
        "acceptance_proxy_nonfinal_marker"
    )
    assert proxy.BOOTSTRAP_SEED == 8041
    assert proxy.VALIDATION_PROXY_VIEW_REGISTRY == {
        "hlt": "D0c", "shell_exact": "D100", "native_offline": "TOFF",
    }
    parameters = set(inspect.signature(proxy.build_validation_proxy_proof_v2).parameters)
    assert parameters == {"result_reference", "authority", "authority_validator"}
    proof_cli = (
        Path(proxy.__file__).resolve().parents[3]
        / "scripts/build_hcwdl_representation_validation_proxy_proof.py"
    ).read_text(encoding="utf-8")
    assert "validate_nonfinal_acceptance_authority_static" in proof_cli
    assert "authority_validator=validate_nonfinal_acceptance_authority_static" in proof_cli
    assert 'add_argument("--output"' not in proof_cli
    assert "publish(" not in proof_cli

    from hlt_classification.scouting import hcwdl_representation_validation_runtime

    for module_under_test in (proxy, hcwdl_representation_validation_runtime):
        source_path = Path(module_under_test.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            module.endswith(("hcwdl_shared_final", "hcwdl_final_stream"))
            for module in imported
        )
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "class_stratified_selection" not in called_names
        assert "locked_metric_join" not in called_names
        assert not any(name.startswith("iterate_final_") for name in called_names)
        assert "TASK_KINDS" not in source
