from __future__ import annotations

from dataclasses import asdict

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    build_command_plan, build_task_registry,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    CONTROL_REGISTRY, NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_representation_locks import (
    IMPORTED_LOGIT_CONTROLS, IMPORTED_TEACHERS,
)
from hlt_classification.scouting.hcwdl_ladder import GRAPH_SHA256
from hlt_classification.scouting.hcwdl_representation_final_policy import (
    project_parent_finalists,
)
from hlt_classification.scouting.hcwdl_representation_reporting import (
    CONFIRMATION_SEEDS, derive_representation_execution_id,
)
from hlt_classification.scouting.hcwdl_representation_resources import resource_table
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    build_runtime_binding, validate_runtime_binding,
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
        "representation_recipe_sha256": "5" * 64, "graph_sha256": "6" * 64,
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
    for node in ("M0", "M6c", "M6w"):
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
    return {
        "shared_final_population": population,
        "parent_final_state": _parent_state(),
        "final_disposition": _final_disposition(spec["disposition"]),
        "parent_finalist_registry": lock,
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
            "contract": "HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v1",
            "schema_version": 1, "payload": environment,
        })
        for logical in static if logical.startswith("${target_forward_spec:")
    }


def _prerequisites(spec):
    static = _static_inputs(spec)
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
            "parent_model_sources": {
                "D0w": {
                    "relative": "reports/D0w/training_report.json", "kind": "pmard",
                },
                "hcwdl_surfaces": "source/hcwdl_surfaces.py",
                "scouting_particle_transformer": "source/scouting_particle_transformer.py",
            },
            "parent_runtime_sources": ["runtime/engine.py", "runtime/loss.py"],
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
    assert len(task_rows) == len(spec["tasks"]) == 86
    assert sum(len(rows) for rows in task_rows.values()) == expected_rows == 152
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
    assert audit["validated_task_count"] == 86
    assert audit["validated_runtime_row_count"] == 152
    assert audit["scheduler_mutated"] is False


def test_runtime_rows_route_parent_sources_through_imported_fresh_evidence() -> None:
    spec = _spec()
    task_rows = build_runtime_task_rows(spec, _prerequisites(spec))
    parent_loss = task_rows["parent_loss_attestation"]["single"]["parameters"]
    assert set(parent_loss) == {
        "adapter_contract", "task_kind", "parent_report_paths",
        "runtime_source_paths",
    }
    assert "${active_scientific_plan}" not in next(
        task for task in spec["tasks"]
        if task["task_key"] == "parent_loss_attestation"
    )["registered_inputs"]
    parent = task_rows["parent_import"]["single"]["parameters"]
    assert set(parent) == {
        "adapter_contract", "task_kind", "artifact", "architecture_attestation",
        "parent_loss_attestation", "parent_report_paths", "model_source_paths",
    }
    assert set(parent["parent_report_paths"]) == set(IMPORTED_TEACHERS) | set(
        IMPORTED_LOGIT_CONTROLS
    )
    assert set(parent["model_source_paths"]) == {
        "D0w", "hcwdl_surfaces", "scouting_particle_transformer",
    }
    target = task_rows["target_D75c_screen"]["single"]["parameters"]["assembly"]
    assert target["parent_import"] == {
        "registered_reference": "${task_output:parent_import:0}",
    }
    assert target["architecture_attestation"] == {
        "registered_reference": "${task_output:architecture_attestation:0}",
    }
    warm = task_rows["train_RSET_M1w"]["single"]["parameters"]["assembly"]
    assert warm["parent_import"] == target["parent_import"]
    assert set(warm["model_sources"]) == {"D0w"}
    screen = task_rows["screen_aggregate"]["single"]["parameters"]
    assert set(screen) == {
        "adapter_contract", "task_kind", "parent_import",
        "architecture_attestation", "builder_arguments",
    }


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
