from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.scouting.engine import (
    PMARD_TRAINING_REPORT_CONTRACT, PMARD_TRAINING_REPORT_VERSION,
)
from hlt_classification.scouting.hcwdl_parent_loss import (
    HCWDL_PARENT_BASE_LOSS_CONTRACT, HCWDL_PARENT_LOSS_SEMANTICS,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    PARENT_IMPORT_AUTHORITY_ROUTES, PARENT_QUALIFIER_REPORT_ROUTES,
    build_command_plan, build_task_registry,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    CONTROL_REGISTRY, NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_representation_locks import (
    IMPORTED_LOGIT_CONTROLS, IMPORTED_TEACHERS, PARENT_AUTHORITY_FILE_KEYS,
    PARENT_AUTHORITY_PARENT_KEYS,
)
from hlt_classification.scouting.hcwdl_ladder import GRAPH_SHA256
from hlt_classification.scouting.hcwdl_representation_final_policy import (
    project_parent_finalists,
)
from hlt_classification.scouting.hcwdl_representation_reporting import (
    CONFIRMATION_SEEDS, derive_representation_execution_id,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    build_representation_recipe, example_representation_recipe,
)
from hlt_classification.scouting.hcwdl_representation_resources import resource_table
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    PREPUBLISHED_OUTPUT_BINDING, build_runtime_binding, resolve_runtime_row,
    validate_runtime_binding,
)
from hlt_classification.scouting import hcwdl_representation_runtime_rows as rows_module
from hlt_classification.scouting.hcwdl_representation_runtime_rows import (
    RUNTIME_DRY_RUN_AUDIT_CONTRACT, RUNTIME_PREREQUISITES_CONTRACT,
    build_runtime_dry_run_audit, build_runtime_prerequisites,
    build_runtime_task_rows,
    validate_bound_runtime_task_rows, validate_runtime_task_rows,
)
from hlt_classification.scouting.hcwdl_representation_targets import (
    derive_target_generation_id,
)
from hlt_classification.scouting.training import derive_seed


def _split_manifest():
    counts = [10_000] * 15
    return with_content_hash({
        "contract": "hlt_classification_scouting_file_split_v2",
        "schema_version": 2,
        "roles": {
            "final_test": {
                "mapped_entries": sum(counts), "class_counts": counts,
                "files": [{"path": "source_000"}, {"path": "source_001"}],
            },
        },
    })


def _parent_state():
    return with_content_hash({
        "contract": "HCWDL_REPRESENTATION_PARENT_FINAL_STATE/v1",
        "schema_version": 1, "parent_campaign_sha256": "1" * 64,
        "exploratory_campaign_sha256": None, "audited_artifacts": 0,
        "exposures": [], "legacy_jobs": [],
        "pending_or_running_legacy_workers": [],
        "final_population_already_exposed": False,
    })


def _final_disposition(disposition="combined_confirmatory"):
    return with_content_hash({
        "contract": "HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1",
        "schema_version": 1,
        "parent_final_state_sha256": _parent_state()["content_hash"],
        "requested": disposition, "disposition": disposition,
        "reason": None,
        "final_tasks_registered": disposition == "combined_confirmatory",
    })


def _representation_recipe(producer_source: str = "b" * 64):
    fixture = example_representation_recipe()
    return build_representation_recipe(
        parents={
            **fixture["parents"], "producer_source": producer_source,
        },
        kernel_array_logical_hashes=fixture["payload"][
            "kernel_array_logical_hashes"
        ],
        evidence=fixture["payload"]["acceptance_evidence"],
    )


def _spec(*, disposition: str = "combined_confirmatory"):
    tasks = build_task_registry(
        disposition=disposition, final_source_partitions=2,
        combined_finalist_count=17,
    )
    resources = resource_table(mode="pilot")
    spec = {
        "mode": "pilot", "campaign_root": "/campaign",
        "checkpoint_namespace": "/checkpoints", "project_dir": "/project",
        "source_commit": "1" * 40, "source_manifest_sha256": "2" * 64,
        "split_manifest_sha256": _split_manifest()["content_hash"],
        "parent_import_sha256": "4" * 64,
        "representation_recipe_sha256": _representation_recipe()["content_hash"],
        "graph_sha256": "6" * 64,
        "disposition_sha256": _final_disposition(disposition)["content_hash"],
        "disposition": disposition,
        "role_counts": {
            "train": 300_000, "validation": 100_000, "final_test": 100_000,
        },
        "final_source_partitions": 2, "combined_finalist_count": 17,
        "artifact_paths": {
            "runtime_binding": "/campaign/runtime/runtime_binding.json",
        },
        "resources": resources, "array_concurrency_limits": {},
        "resource_request_sha256": canonical_sha256(resources),
        "tasks": [asdict(task) for task in tasks],
        "command_plan_sha256": "8" * 64,
        "content_hash": "9" * 64,
    }
    return spec


def _static_inputs(spec):
    tasks = rows_module._tasks(spec)
    by_key = {task.task_key: task for task in tasks}
    names = set()
    for task in tasks:
        for logical in task.registered_inputs:
            if logical in {"${campaign_spec}", "${submission_ledger}"}:
                continue
            if rows_module._alias_route(logical, task=task, by_key=by_key) is None:
                names.add(logical)
    return {
        logical: {
            "path": f"/inputs/{position:04d}",
            "sha256": canonical_sha256({"logical": logical}),
        }
        for position, logical in enumerate(sorted(names))
    }


def _target_generations(spec, hashes):
    tasks = rows_module._tasks(spec)
    targets = {}
    execution_campaign = "a" * 64
    for task in tasks:
        if task.kind != "target_build":
            continue
        key = f"{task.logical_bank}:{task.target_purpose}"
        logical = f"${{logical_bank:{task.logical_bank}}}"
        registry = (
            f"${{target_consumer_registry:{task.logical_bank}:{task.target_purpose}}}"
        )
        parent = canonical_sha256({"generation": key})
        generation = derive_target_generation_id(
            hashes[logical], hashes[registry], purpose=str(task.target_purpose),
            generation_parent_sha256=parent,
        )
        ids = {"miniature:0": canonical_sha256({"miniature": key})}
        for consumer in tasks:
            if (
                consumer.kind not in {"train_node", "train_control", "confirmation"}
                or consumer.logical_bank != task.logical_bank
                or consumer.target_purpose != task.target_purpose
            ):
                continue
            graph = NODE_REGISTRY.get(str(consumer.graph_node)) or CONTROL_REGISTRY.get(
                str(consumer.graph_node)
            )
            assert graph is not None
            seeds = CONFIRMATION_SEEDS if consumer.kind == "confirmation" else (1337,)
            for seed in seeds:
                execution, _ = derive_representation_execution_id(
                    campaign_sha256=execution_campaign, strategy=graph.strategy,
                    node_id=str(consumer.graph_node), purpose=str(task.target_purpose),
                    seed=seed, initialization_parent=graph.initialization_parent,
                    teacher=graph.representation_logit_teacher,
                    logical_target_bank_sha256=hashes[logical],
                    target_purpose=str(task.target_purpose),
                    recipe_sha256=spec["representation_recipe_sha256"],
                )
                ids[f"{consumer.graph_node}:{seed}"] = execution
        targets[key] = {
            "generation_id": generation,
            "generation_parent_sha256": parent,
            "source_partitions": {
                "source_000": {"rows": 10, "source_file_id": 0},
                "source_001": {"rows": 11, "source_file_id": 1},
            },
            "execution_ids": ids,
            "execution_campaign_sha256": execution_campaign,
        }
    return targets


def _parent_report(node_id, seed, marker):
    return with_content_hash({
        "contract": "TEST_HCWDL_PARENT_REPORT/v1", "schema_version": 1,
        "config": {"master_seed": derive_seed(seed, f"hcwdl/{node_id}")},
        "selected_checkpoint_sha256": canonical_sha256({"checkpoint": marker}),
    })


def _runtime_prerequisites_script(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "build_hcwdl_representation_runtime_prerequisites.py"
    )
    monkeypatch.syspath_prepend(str(script.parent))
    module_spec = importlib.util.spec_from_file_location(
        "_test_hcwdl_representation_runtime_prerequisites", script,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _write_screen_wrapper_and_engine(
    root: Path, *, node_id: str, marker: str,
    wrapper_engine_sha256: str | None = None,
) -> tuple[dict, dict]:
    directory = root / "screen" / node_id
    checkpoint_sha256 = canonical_sha256({"checkpoint": marker})
    execution_sha256 = canonical_sha256({"execution": marker})
    engine = with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": PMARD_TRAINING_REPORT_VERSION,
        "config": {"master_seed": derive_seed(1337, f"hcwdl/{node_id}")},
        "selected_checkpoint_sha256": checkpoint_sha256,
        "execution_config_sha256": execution_sha256,
    })
    semantics = dict(HCWDL_PARENT_LOSS_SEMANTICS)
    wrapper = with_content_hash({
        "contract": "HCWDL_TRAINING_REPORT/v1", "schema_version": 1,
        "node_id": node_id,
        "pmard_engine_report_sha256": (
            engine["content_hash"]
            if wrapper_engine_sha256 is None else wrapper_engine_sha256
        ),
        "pmard_execution_config_sha256": execution_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": semantics,
        "loss_semantics_sha256": canonical_sha256(semantics),
    })
    write_immutable_json(directory / "training_report.json", engine)
    write_immutable_json(directory / "hcwdl_training_report.json", wrapper)
    return wrapper, engine


def test_runtime_prerequisite_projects_authenticated_wrapper_sibling_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_prerequisites_script(monkeypatch)
    parent_root = tmp_path / "parent_reports"
    finalist_root = tmp_path / "finalist_models"
    wrapper, engine = _write_screen_wrapper_and_engine(
        parent_root, node_id="M0", marker="same",
    )
    _write_screen_wrapper_and_engine(
        finalist_root, node_id="M0", marker="same",
    )
    member = module._paired_screen_model_member(
        node_id="M0", parent_reports_root=parent_root,
        parent_wrapper_relative="screen/M0/hcwdl_training_report.json",
        finalist_models_root=finalist_root,
        finalist_wrapper_relative="screen/M0/hcwdl_training_report.json",
    )
    assert member == {
        "relative": "screen/M0/training_report.json", "report": engine,
    }
    assert member["report"]["content_hash"] == wrapper[
        "pmard_engine_report_sha256"
    ]


def test_runtime_prerequisite_rejects_wrapper_engine_or_bundle_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_prerequisites_script(monkeypatch)
    parent_root = tmp_path / "parent_reports"
    finalist_root = tmp_path / "finalist_models"
    _write_screen_wrapper_and_engine(parent_root, node_id="M0", marker="parent")
    _write_screen_wrapper_and_engine(finalist_root, node_id="M0", marker="other")
    with pytest.raises(ValueError, match="wrapper/model source differs"):
        module._paired_screen_model_member(
            node_id="M0", parent_reports_root=parent_root,
            parent_wrapper_relative="screen/M0/hcwdl_training_report.json",
            finalist_models_root=finalist_root,
            finalist_wrapper_relative="screen/M0/hcwdl_training_report.json",
        )

    broken_root = tmp_path / "broken_parent_reports"
    _write_screen_wrapper_and_engine(
        broken_root, node_id="M0", marker="broken",
        wrapper_engine_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="wrapper/engine hash differs"):
        module._paired_wrapper_and_engine(
            broken_root, "screen/M0/hcwdl_training_report.json", node_id="M0",
        )


def _parent_finalist_authority():
    raw_rows = []
    locked = {}
    for node in ("M0", "M6c", "M6w"):
        for seed in (11, 22):
            marker = f"{node}-{seed}"
            report = _parent_report(node, seed, marker)
            raw_rows.append({
                "node_id": node, "seed": seed,
                "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "report_sha256": report["content_hash"],
                "report_path": f"/parent/{marker}/training_report.json",
            })
            locked[f"{node}:{seed}"] = {
                "relative": f"{marker}/training_report.json", "report": report,
            }
    for node, seed in (
        ("NULL_M1_SELF_KD", 11), ("NULL_M6_PREDECESSOR_ONLY", 11),
        ("D100", 1337), ("TOFF", 1337),
    ):
        report = _parent_report(node, seed, node)
        raw_rows.append({
            "node_id": node, "seed": seed,
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            "report_sha256": report["content_hash"],
            "report_path": f"/parent/{node}/training_report.json",
        })
        locked[f"{node}:{seed}"] = {
            "relative": f"{node}/training_report.json", "report": report,
        }
    lock = with_content_hash({
        "contract": "HCWDL_LOCK/v1", "schema_version": 1,
        "campaign": "HCWDL", "level": "finalist",
        "campaign_spec_sha256": "1" * 64,
        "parent_lock_sha256": "e" * 64, "graph_sha256": GRAPH_SHA256,
        "payload": {
            "confirmation_report_sha256": "f" * 64,
            "finalists": raw_rows, "selection_used_validation_only": True,
        },
    })
    paired = {}
    for node in ("M0", "M1c", "M1w"):
        report = _parent_report(node, 1337, f"{node}-screen")
        paired[node] = {
            "relative": f"screen/{node}/training_report.json", "report": report,
        }
    finalists = project_parent_finalists(
        parent_finalist_lock=lock, locked_model_members=locked,
        paired_screen_model_members=paired,
    )
    return lock, locked, paired, finalists


def _finalists():
    return _parent_finalist_authority()[3]


def _parent_import_authority(spec, parent_lock):
    def imported_row(node_id: str, *, teacher: bool):
        track = "cold" if node_id.endswith("c") else (
            "warm" if node_id.endswith("w") else "shared"
        )
        domain = (
            "hlt" if not teacher else
            "native_offline" if node_id == "TOFF" else
            "hlt" if node_id.startswith("D0") else
            f"d{node_id[1:].rstrip('cw').lower()}"
        )
        checkpoint = canonical_sha256({"parent-checkpoint": node_id})
        return {
            "node_id": node_id, "domain": domain, "track": track,
            "report_path": f"/parent/{node_id}/training_report.json",
            "report_sha256": canonical_sha256({"parent-report": node_id}),
            "checkpoint_path": f"/parent/{node_id}/selected.pt",
            "checkpoint_sha256": checkpoint,
            "checkpoint_byte_sha256": checkpoint,
        }

    parents = {
        name: canonical_sha256({"parent-import-parent": name})
        for name in PARENT_AUTHORITY_PARENT_KEYS
    }
    parents.update({
        "parent_campaign_spec": "1" * 64,
        "split_manifest": spec["split_manifest_sha256"],
        "finalist_lock": parent_lock["content_hash"],
        "parent_graph": GRAPH_SHA256,
    })
    return with_content_hash({
        "contract": "HCWDL_REPRESENTATION_PARENT_IMPORT/v3",
        "schema_version": 2,
        "parents": parents,
        "payload": {
            "parent_source_commit": "1" * 40,
            "parent_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v8",
            "parent_campaign_mode": "pilot",
            "parent_execution_scope": "parent_prefix_through_finalist_lock",
            "parent_recipe_contract": "HCWDL_RECIPE/v4",
            "endpoint_continuation": "preauthorized_automatic",
            "training_passes": 60, "validation_every_passes": 1,
            "parent_train_rows": 300000,
            "terminal_task_id": "finalist_lock",
            "execution_lock_authorized": False,
            "final_test_access_authorized": False,
            "registered_final_test_tasks": 0,
            "teachers": [
                imported_row(node, teacher=True) for node in IMPORTED_TEACHERS
            ],
            "logit_controls": [
                imported_row(node, teacher=False)
                for node in IMPORTED_LOGIT_CONTROLS
            ],
            "authority_derived_from_registered_files": True,
            "complete": True,
        },
    })


def _final_authorities(spec):
    from hlt_classification.scouting.highcov_resources import resource_validation_report

    population = with_content_hash({
        "contract": "HCWDL_SHARED_FINAL_POPULATION/v1", "schema_version": 1,
        "population_sha256": "d" * 64,
        "source_snapshot_sha256": "b" * 64,
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "role": "final_test", "row_count": spec["role_counts"]["final_test"],
    })
    lock, locked, paired, _ = _parent_finalist_authority()
    confirmation_lock = with_content_hash({
        "contract": "HCWDL_LOCK/v1", "schema_version": 1,
        "campaign": "HCWDL", "level": "confirmation_registry",
        "campaign_spec_sha256": "1" * 64,
        "parent_lock_sha256": "c" * 64, "graph_sha256": GRAPH_SHA256,
        "payload": {
            "screen_aggregate_sha256": "d" * 64,
            "registry": [{"node_id": "M0", "seed": 11}],
        },
    })
    parent_import = _parent_import_authority(spec, lock)
    parent_import["parents"]["confirmation_registry_lock"] = confirmation_lock[
        "content_hash"
    ]
    parent_import = with_content_hash({
        key: value for key, value in parent_import.items() if key != "content_hash"
    })
    spec["parent_import_sha256"] = parent_import["content_hash"]
    return {
        "shared_final_population": population,
        "parent_final_state": _parent_state(),
        "final_disposition": _final_disposition(spec["disposition"]),
        "parent_import": parent_import,
        "parent_finalist_registry": lock,
        "parent_confirmation_registry": confirmation_lock,
        "locked_model_members": locked,
        "paired_screen_model_members": paired,
        "matcher_resources": resource_validation_report(),
    }


def _target_forward_specs(static):
    environment = {
        "producer": {"source_commit": "1" * 40, "source_snapshot_sha256": "b" * 64},
        "device": {"request": "gpu:gh200:1", "model": "GH200"},
        "precision": {"parameters": "float32", "autocast": False},
        "determinism": {"deterministic_algorithms": True},
    }
    return {
        logical: with_content_hash({
            "contract": "HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v2",
            "schema_version": 1, "payload": environment,
        })
        for logical in static if logical.startswith("${target_forward_spec:")
    }


def _prerequisites(
    spec, *, representation_recipe=None,
    canonical_prebuilt_layout: bool = False,
):
    static = _static_inputs(spec)
    if canonical_prebuilt_layout:
        static = deepcopy(static)
        static["${prebuilt_parent_import}"]["path"] = (
            "/campaign/import/parent_import.json"
        )
        static["${prebuilt_representation_recipe}"]["path"] = (
            "/campaign/recipes/representation_recipe.json"
        )
    authorities = _final_authorities(spec)
    forward_specs = _target_forward_specs(static)
    hashes = {
        logical: canonical_sha256({"content": logical}) for logical in static
    }
    hashes.update({
        "${shared_final_population}": authorities["shared_final_population"]["content_hash"],
        "${parent_final_state}": authorities["parent_final_state"]["content_hash"],
        "${final_disposition}": authorities["final_disposition"]["content_hash"],
        "${matcher_resources}": authorities["matcher_resources"]["content_hash"],
        "${prebuilt_parent_import}": authorities["parent_import"]["content_hash"],
    })
    prerequisites = {
        "contract": RUNTIME_PREREQUISITES_CONTRACT, "schema_version": 1,
        "runtime_facts": {
            "conda_environment": "atlas_kd_tigris", "data_root": "/data",
            "device": "cuda", "project_dir": "/project",
            "python_no_user_site": True, "source_snapshot_sha256": "b" * 64,
            "weaver_runtime_sha256": "c" * 64,
        },
        "runtime_signatures": {
            task["resource_class"]: canonical_sha256({
                "resource_class": task["resource_class"],
            }) for task in spec["tasks"]
        },
        "static_inputs": static,
        "artifact_content_hashes": hashes,
        "target_generations": {},
        "bundle_members": {
            "parent_reports": {
                node: f"reports/{node}/hcwdl_training_report.json"
                for node in (*IMPORTED_TEACHERS, *IMPORTED_LOGIT_CONTROLS)
            },
            "parent_confirmation_reports": {
                "000:M0:11": "confirmation/000_M0_11/hcwdl_training_report.json",
            },
            "parent_model_sources": {
                "D0w": {
                    "relative": "reports/D0w/training_report.json", "kind": "pmard",
                },
                "hcwdl_surfaces": "source/hcwdl_surfaces.py",
                "scouting_particle_transformer": "source/scouting_particle_transformer.py",
            },
            "parent_runtime_sources": {
                "engine": "src/hlt_classification/scouting/engine.py",
                "parent_loss": (
                    "src/hlt_classification/scouting/hcwdl_parent_loss.py"
                ),
                "training": "src/hlt_classification/scouting/hcwdl_training.py",
            },
            "finalist_models": {
                row["finalist_id"]: row["model_relative"] for row in _finalists()
            },
        },
        "settings": {
            "kernel_parent_hashes": None,
            "shuffle_parent_hashes": None,
            "target_budgets": {
                "target_storage_cap_bytes": 10**12,
                "container_overhead_bytes": 1,
                "staging_recovery_reserve_bytes": 1,
                "quarantine_reserve_bytes": 1,
                "filesystem_headroom_bytes": 1,
                "peak_runtime_bytes": 1,
                "slurm_mem_per_node_bytes": 320 * 1024**3,
                "filesystem_available_bytes": 10**13,
            },
            "target_runtime_environment": None,
            "miniature_row_limit": 4096, "view_cache_max_gib": 300.0,
            "synthetic_passes": 60, "training_mode": "scientific",
            "final": {
                "population_sha256": None,
                "assignment_spec_sha256": None,
                "assignment_spec": None,
                "finalist_registry_commitment_sha256": None,
                "finalist_policy": None,
                "legacy_jobs_present": None,
                "parent_campaign_sha256": None,
                "parent_finalist_lock": None,
                "parent_finalist_registry_relative": "registry/registry.json",
                "parent_finalist_registry_sha256": None,
                "parent_confirmation_report_keys": None,
                "rows_per_class": None,
                "selection_rule_sha256": None,
                "step_size": 8192,
                "source_partitions": None,
                "finalists": None,
                "comparison_registry": None,
            },
        },
    }
    prerequisites["target_generations"] = _target_generations(spec, hashes)
    return build_runtime_prerequisites(
        spec,
        **{
            key: prerequisites[key]
            for key in (
                "runtime_facts", "runtime_signatures", "static_inputs",
                "artifact_content_hashes", "target_generations",
                "bundle_members", "settings",
            )
        },
        split_manifest=_split_manifest(),
        final_authorities=authorities,
        target_forward_specs=forward_specs,
        representation_recipe=(
            _representation_recipe()
            if representation_recipe is None else representation_recipe
        ),
    )


def test_all_combined_rows_build_bind_and_validate_without_hand_assembly(
    monkeypatch,
) -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    task_rows = build_runtime_task_rows(spec, prerequisites)
    validate_runtime_task_rows(spec, prerequisites, task_rows)
    expected_rows = sum(
        1 if task["array"] is None else int(task["array"].split("-")[1]) + 1
        for task in spec["tasks"]
    )
    assert len(task_rows) == len(spec["tasks"]) == 295
    assert sum(len(rows) for rows in task_rows.values()) == expected_rows == 361
    binding = build_runtime_binding(
        spec=spec, runtime_facts=prerequisites["runtime_facts"],
        task_rows=task_rows,
    )
    # The executable campaign later embeds this binding hash.  Neither the
    # descriptor owner nor the binding itself may feed that post-build field
    # back into its identity.
    rebound_spec = {**spec, "runtime_binding_sha256": binding["content_hash"]}
    rebound = build_runtime_binding(
        spec=rebound_spec, runtime_facts=prerequisites["runtime_facts"],
        task_rows=task_rows,
    )
    assert rebound["content_hash"] == binding["content_hash"]
    validate_runtime_binding(binding, spec=spec)
    split_reference = prerequisites["static_inputs"]["${split_manifest}"]
    monkeypatch.setattr(
        rows_module, "sha256_file", lambda path: split_reference["sha256"],
    )
    monkeypatch.setattr(
        rows_module, "load_json", lambda path: _split_manifest(),
    )
    validate_bound_runtime_task_rows(spec, binding)
    bound_spec = {**spec, "runtime_binding_sha256": binding["content_hash"]}
    audit = build_runtime_dry_run_audit(
        bound_spec, binding, build_command_plan(bound_spec),
    )
    validate_content_hash(
        audit, expected_contract=RUNTIME_DRY_RUN_AUDIT_CONTRACT,
        expected_schema_version=1,
    )
    assert audit["validated_task_count"] == 295
    assert audit["validated_runtime_row_count"] == 361
    assert audit["scheduler_mutated"] is False


def test_all_rows_bind_with_documented_in_place_prebuilt_layout() -> None:
    spec = _spec()
    prerequisites = _prerequisites(
        spec, canonical_prebuilt_layout=True,
    )
    assert prerequisites["static_inputs"]["${prebuilt_parent_import}"][
        "path"
    ] == "/campaign/import/parent_import.json"
    assert prerequisites["static_inputs"]["${prebuilt_representation_recipe}"][
        "path"
    ] == "/campaign/recipes/representation_recipe.json"

    task_rows = build_runtime_task_rows(spec, prerequisites)
    validate_runtime_task_rows(spec, prerequisites, task_rows)
    assert len(task_rows) == 295
    assert sum(len(rows) for rows in task_rows.values()) == 361
    binding = build_runtime_binding(
        spec=spec, runtime_facts=prerequisites["runtime_facts"],
        task_rows=task_rows,
    )
    validate_runtime_binding(binding, spec=spec)

    expected = {
        "parent_import": (
            "${prebuilt_parent_import}", "import/parent_import.json",
            spec["parent_import_sha256"],
        ),
        "representation_recipe": (
            "${prebuilt_representation_recipe}",
            "recipes/representation_recipe.json",
            spec["representation_recipe_sha256"],
        ),
    }
    for task_key, (logical, output, content_hash) in expected.items():
        reference = resolve_runtime_row(
            binding, spec=spec, task_key=task_key, array_index=None,
        )["inputs"][logical]
        assert reference["path"] == f"/campaign/{output}"
        descriptor = reference[PREPUBLISHED_OUTPUT_BINDING]
        assert descriptor["consumer_task_key"] == task_key
        assert descriptor["consumer_task_kind"] == task_key
        assert descriptor["owner_task_key"] == task_key
        assert descriptor["owner_task_kind"] == task_key
        assert descriptor["registered_input"] == logical
        assert descriptor["registered_output"] == output
        assert descriptor["expected_schema_version"] == (
            2 if task_key == "parent_import" else 1
        )
        assert descriptor["expected_content_hash"] == content_hash

    kernel_reference = resolve_runtime_row(
        binding, spec=spec, task_key="kernel_resources", array_index=None,
    )["inputs"]["${prebuilt_representation_recipe}"]
    assert kernel_reference["path"] == (
        "/campaign/recipes/representation_recipe.json"
    )
    kernel_descriptor = kernel_reference[PREPUBLISHED_OUTPUT_BINDING]
    assert kernel_descriptor["consumer_task_key"] == "kernel_resources"
    assert kernel_descriptor["consumer_task_kind"] == "kernel_resources"
    assert kernel_descriptor["owner_task_key"] == "representation_recipe"
    assert kernel_descriptor["owner_task_kind"] == "representation_recipe"
    assert kernel_descriptor["registered_output"] == (
        "recipes/representation_recipe.json"
    )
    assert kernel_descriptor["expected_content_hash"] == spec[
        "representation_recipe_sha256"
    ]


def test_recipe_producer_source_is_bound_to_measured_runtime_snapshot() -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    task_rows = build_runtime_task_rows(spec, prerequisites)
    parameters = task_rows["representation_recipe"]["single"]["parameters"]
    assert parameters["producer_source_sha256"] == "b" * 64
    assert parameters["representation_graph"] == {
        "registered_reference": "${representation_graph}",
    }
    assert parameters["control_registry"] == {
        "registered_reference": "${control_registry}",
    }
    assert parameters["parent_import"] == {
        "registered_reference": "${parent_import}",
    }

    mismatched = _representation_recipe("e" * 64)
    mismatched_spec = {
        **_spec(),
        "representation_recipe_sha256": mismatched["content_hash"],
    }
    with pytest.raises(PermissionError, match="measured runtime source"):
        _prerequisites(
            mismatched_spec, representation_recipe=mismatched,
        )


def test_runtime_rows_route_parent_sources_through_imported_fresh_evidence() -> None:
    spec = _spec()
    task_rows = build_runtime_task_rows(spec, _prerequisites(spec))
    parent_loss = task_rows["parent_loss_attestation"]["single"]["parameters"]
    assert set(parent_loss) == {
        "adapter_contract", "task_kind", "parent_campaign_spec_path",
        "parent_recipe_path", "parent_report_paths", "runtime_source_paths",
    }
    assert "${active_scientific_plan}" not in next(
        task for task in spec["tasks"]
        if task["task_key"] == "parent_loss_attestation"
    )["registered_inputs"]
    assert "${parent_recipe}" in next(
        task for task in spec["tasks"]
        if task["task_key"] == "parent_loss_attestation"
    )["registered_inputs"]
    assert "${parent_campaign_spec}" in next(
        task for task in spec["tasks"]
        if task["task_key"] == "parent_loss_attestation"
    )["registered_inputs"]
    parent = task_rows["parent_import"]["single"]["parameters"]
    assert set(parent) == {
        "adapter_contract", "task_kind", "artifact", "architecture_attestation",
        "parent_loss_attestation", "parent_report_paths", "model_source_paths",
        "authority_files", "qualifier_report_paths", "confirmation_report_paths",
    }
    assert set(parent["parent_report_paths"]) == set(IMPORTED_TEACHERS) | set(
        IMPORTED_LOGIT_CONTROLS
    )
    assert set(parent["model_source_paths"]) == {
        "D0w", "hcwdl_surfaces", "scouting_particle_transformer",
    }
    assert set(PARENT_IMPORT_AUTHORITY_ROUTES) == (
        set(PARENT_AUTHORITY_FILE_KEYS)
        - {"architecture_attestation", "parent_loss_attestation"}
    )
    assert set(parent["authority_files"]) == set(PARENT_AUTHORITY_FILE_KEYS)
    assert set(parent["qualifier_report_paths"]) == set(
        PARENT_QUALIFIER_REPORT_ROUTES
    )
    assert parent["confirmation_report_paths"] == {
        "000:M0:11": {
            "registered_member": {
                "input": "${parent_confirmation_reports}",
                "relative": "confirmation/000_M0_11/hcwdl_training_report.json",
                "mode": "path",
            },
        },
    }
    parent_task = next(
        task for task in spec["tasks"] if task["task_key"] == "parent_import"
    )
    assert set(PARENT_IMPORT_AUTHORITY_ROUTES.values()) <= set(
        parent_task["registered_inputs"]
    )
    assert set(PARENT_QUALIFIER_REPORT_ROUTES.values()) <= set(
        parent_task["registered_inputs"]
    )
    assert "${parent_confirmation_reports}" in parent_task["registered_inputs"]
    assert parent["authority_files"]["finalist_lock"] == {
        "registered_member": {
            "input": "${parent_finalist_registry}",
            "relative": "registry/registry.json",
            "mode": "path",
        },
    }
    target = task_rows["target_RSET_D75c_screen"]["single"]["parameters"]["assembly"]
    assert target["parent_import"] == {
        "registered_reference": "${task_output:parent_import:0}",
    }
    assert target["architecture_attestation"] == {
        "registered_reference": "${task_output:architecture_attestation:0}",
    }
    assert target["teacher_source"] == {
        "kind": "hcwdl",
        "execution_directory": {
            "registered_path": "${task_output:train_RSET_D75c:3}",
        },
    }
    warm = task_rows["train_RSET_M1w"]["single"]["parameters"]["assembly"]
    assert warm["parent_import"] == target["parent_import"]
    assert set(warm["model_sources"]) == {"RSET_D0w"}
    screen = task_rows["screen_aggregate"]["single"]["parameters"]
    assert set(screen) == {
        "adapter_contract", "task_kind", "parent_import",
        "architecture_attestation", "builder_arguments",
    }


def test_upstream_schema_versions_distinguish_semantic_v2_from_envelope_v2() -> None:
    tasks = {task.task_key: task for task in rows_module._tasks(_spec())}
    expected = {
        "parent_loss_attestation": (
            "HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3", 3,
        ),
        "parent_import": ("HCWDL_REPRESENTATION_PARENT_IMPORT/v3", 2),
        "representation_recipe": ("HCWDL_REPRESENTATION_RECIPE/v3", 1),
    }
    for task_key, (contract, schema_version) in expected.items():
        descriptor = rows_module._upstream_reference(
            tasks[task_key], ordinal=0, array_index=None,
        )["upstream_output"]
        assert descriptor["expected_contract"] == contract
        assert descriptor["expected_schema_version"] == schema_version


def test_final_authority_rejects_cross_campaign_parent_import_splice() -> None:
    spec = _spec()
    static = _static_inputs(spec)
    authorities = _final_authorities(spec)
    forged = deepcopy(authorities["parent_import"])
    forged.pop("content_hash")
    forged["parents"] = dict(forged["parents"])
    forged["parents"]["parent_campaign_spec"] = "2" * 64
    forged = with_content_hash(forged)
    authorities["parent_import"] = forged
    spec["parent_import_sha256"] = forged["content_hash"]
    hashes = {
        logical: canonical_sha256({"content": logical}) for logical in static
    }
    hashes.update({
        "${shared_final_population}": authorities[
            "shared_final_population"
        ]["content_hash"],
        "${parent_final_state}": authorities["parent_final_state"]["content_hash"],
        "${final_disposition}": authorities["final_disposition"]["content_hash"],
        "${matcher_resources}": authorities["matcher_resources"]["content_hash"],
        "${prebuilt_parent_import}": forged["content_hash"],
    })
    with pytest.raises(ValueError, match="parent import/final authority lineage"):
        rows_module._derived_final_authority(
            spec=spec,
            runtime_facts={"source_snapshot_sha256": "b" * 64},
            artifact_content_hashes=hashes,
            split_manifest=_split_manifest(), final_authorities=authorities,
        )


def test_validation_only_rows_are_also_closed_and_exhaustive() -> None:
    spec = _spec(disposition="validation_only_parent_claim_consumed")
    prerequisites = _prerequisites(spec)
    task_rows = build_runtime_task_rows(spec, prerequisites)
    assert set(task_rows) == {task["task_key"] for task in spec["tasks"]}
    assert not any(
        task["kind"] in {"prediction_shard", "metric_join"}
        for task in spec["tasks"]
    )


def test_generator_rejects_missing_exogenous_route_and_hand_edited_assembly() -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    missing = dict(prerequisites)
    missing["static_inputs"] = dict(prerequisites["static_inputs"])
    missing["static_inputs"].pop("${split_manifest}")
    with pytest.raises(ValueError, match="static input registry differs"):
        build_runtime_task_rows(spec, missing)

    task_rows = build_runtime_task_rows(spec, prerequisites)
    task_rows["tap_schema"]["single"]["parameters"]["ad_hoc"] = True
    with pytest.raises(ValueError, match="deterministic assembly"):
        validate_runtime_task_rows(spec, prerequisites, task_rows)


def test_runtime_prerequisites_reject_incomplete_parent_confirmation_bundle() -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    forged = deepcopy(prerequisites)
    forged.pop("content_hash")
    forged["bundle_members"]["parent_confirmation_reports"] = {
        "999:spliced:55": "confirmation/spliced/hcwdl_training_report.json",
    }
    forged = with_content_hash(forged)
    with pytest.raises(
        ValueError, match="parent confirmation-report bundle is incomplete or expanded"
    ):
        build_runtime_task_rows(spec, forged)


def test_pilot_runtime_inputs_cannot_downgrade_scientific_sixty_pass_training() -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    forged = dict(prerequisites)
    forged["settings"] = {
        **prerequisites["settings"],
        "training_mode": "synthetic_test",
        "synthetic_passes": 1,
    }
    with pytest.raises(PermissionError, match="frozen campaign/resource policy"):
        build_runtime_task_rows(spec, forged)


def test_final_population_and_comparison_policy_cannot_be_truncated() -> None:
    spec = _spec()
    prerequisites = _prerequisites(spec)
    final = dict(prerequisites["settings"]["final"])
    forged = dict(prerequisites)
    forged["settings"] = {**prerequisites["settings"], "final": final}

    final["rows_per_class"] = [1] * 15
    with pytest.raises(ValueError, match="cover the frozen final-test budget"):
        build_runtime_task_rows(spec, forged)

    final.update(prerequisites["settings"]["final"])
    final["comparison_registry"] = final["comparison_registry"][:1]
    with pytest.raises(ValueError, match="incomplete or reordered"):
        build_runtime_task_rows(spec, forged)
