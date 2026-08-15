"""Idempotent workers for the all-mapped HCWDL-UB-FULL3 campaign."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_assignment import (
    assignment_recomputer, build_assignment_source, finalize_role_assignments,
)
from .hcwdl_homotopy import HomotopyCoordinate, assert_particle_inputs_equal
from .hcwdl_homotopy_stream import (
    iterate_homotopy_batches, iterate_unified_balanced_batches,
)
from .hcwdl_recipe import (
    CLASS_WEIGHT_POLICY, FULL_DATA_RECIPE_CONTRACT,
    FULL_DATA_RECIPE_SCHEMA_VERSION, validate_recipe,
    validate_recipe_class_weight_lineage,
)
from .hcwdl_unified_balanced_builder import (
    build_balanced_sidecar_for_source, finalize_balanced_role,
)
from .hcwdl_unified_balanced_cache import validate_balanced_manifest
from .hcwdl_unified_balanced_campaign import authenticate_parent_homotopy
from .hcwdl_unified_balanced_contracts import balanced_switch_config_payload
from .hcwdl_unified_balanced_full_campaign import (
    arm_tasks, validate_arm_campaign, validate_foundation_campaign,
)
from .hcwdl_unified_balanced_full_contracts import (
    ARM_AGGREGATE_CONTRACT, ARM_COMPLETION_CONTRACT,
    RESOURCE_PROFILE_CONTRACT, aggregate_payload, assignment_lock_payload,
    completion_payload, endpoint_lock_payload, foundation_lock_payload,
    validate_aggregate, validate_arm_spec, validate_assignment_lock,
    validate_completion, validate_endpoint_lock,
    validate_foundation_lock,
)
from .hcwdl_unified_balanced_full_graph import (
    arm_registry, idealized_u000_ancestry,
)
from .hcwdl_unified_balanced_full_runner import (
    arm_node_output_dir, publish_u000_targets, run_arm_node, run_shared_node,
    shared_node_output_dir,
)
from .hcwdl_unified_balanced_runner import RUNTIME_CONTRACT, _load_common
from .hcwdl_unified_balanced_targets import validate_target_manifest
from .hcwdl_upper_builder import (
    audit_full_roles, build_coupling_source, build_switch_sidecar_for_source,
    calibrate_train_scales, finalize_base_role, finalize_coupling_role,
    freeze_switch_calibration,
)
from .hcwdl_upper_cache import (
    ResidualCouplingStore, build_coupling_lock, load_base_shard,
    load_switch_sidecar, validate_base_manifest, validate_coupling_audit,
    validate_coupling_lock, validate_coupling_manifest,
)
from .hcwdl_upper_coupling import (
    validate_scale_calibration, validate_switch_calibration,
)
from .highcov_cache import (
    DenseAssignmentStore, sampled_recomputation_audit,
    validate_assignment_manifest,
)
from .highcov_resources import RESOURCE_CONTRACT
from .selective_assignment import RowSelection, build_row_selection
from .splits import role_records
from .training import derive_seed


def _task(tasks: list[Mapping[str, Any]], task_id: str) -> Mapping[str, Any]:
    found = [row for row in tasks if row["task_id"] == task_id]
    if len(found) != 1:
        raise ValueError(f"HCWDL-UB-FULL3 task identity differs: {task_id}")
    return found[0]


def _index(task: Mapping[str, Any], array_index: int | None) -> int:
    count = int(task["array_count"])
    if count == 1:
        if array_index not in {None, 0}:
            raise ValueError("HCWDL-UB-FULL3 scalar task received an array index")
        return 0
    if array_index is None or not 0 <= int(array_index) < count:
        raise ValueError("HCWDL-UB-FULL3 array index differs")
    return int(array_index)


def _runtime(
    output: Path, *, scope_spec_sha256: str, canonical_node_id: str,
    training_report_sha256: str, started: float,
) -> Path:
    elapsed = max(0.0, time.monotonic() - started)
    peak_gpu = 0
    try:
        import torch
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    payload = with_content_hash({
        "contract": RUNTIME_CONTRACT, "schema_version": 1,
        "scope_spec_sha256": scope_spec_sha256,
        "canonical_node_id": canonical_node_id,
        "training_report_sha256": training_report_sha256,
        "elapsed_seconds": elapsed, "measured_gpu_hours": elapsed / 3600.0,
        "peak_gpu_memory_bytes": peak_gpu,
        "phase_boundaries_recorded": True, "final_test_accessed": False,
    })
    path = output / "runtime.json"
    write_immutable_json(path, payload)
    return path


def _full_recipe(
    *, parent: Mapping[str, Any], selection: Mapping[str, Any],
    selection_sha256: str, overlay_sha256: str,
) -> dict[str, Any]:
    validate_recipe(parent, require_authorized=True)
    copied = {
        key: parent[key] for key in (
            "repair_family", "batching", "optimizer", "schedule",
            "coefficient_schedule", "single_teacher_coefficients",
            "dual_teacher_coefficients", "controls",
            "single_privileged_temperature", "predecessor_temperature",
            "privileged_temperature", "dual_teacher_peak_learning_rate",
            "amp_dtype",
        )
    }
    result = with_content_hash({
        "contract": FULL_DATA_RECIPE_CONTRACT,
        "schema_version": FULL_DATA_RECIPE_SCHEMA_VERSION,
        "authorized_for_execution": True,
        "recipe_profile": "full_data_scaleup",
        "purpose": "hcwdl_unified_balanced_full_data_three_arm",
        "training_passes": 20, "validation_every_passes": 1,
        **copied,
        "class_weighting": {
            "policy": CLASS_WEIGHT_POLICY,
            "train_class_counts": list(selection["roles"]["train"]["class_counts"]),
            "train_row_selection_sha256": selection_sha256,
        },
        "class_weights": [1.0] * 15,
        "evidence": {
            "parent_recipe_sha256": str(parent["content_hash"]),
            "row_selection_sha256": selection_sha256,
            "recipe_overlay_sha256": overlay_sha256,
        },
    })
    validate_recipe(result, require_authorized=True, expected_profile="full_data_scaleup")
    validate_recipe_class_weight_lineage(result, selection)
    return result


class UnifiedBalancedFullFoundationWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
        recovery_spec_sha256: str | None = None,
    ) -> None:
        validate_foundation_campaign(
            spec, executable=False, verify_source_tree=producer_commit is None,
        )
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.producer_commit = str(producer_commit or spec["source_commit"])
        if len(self.producer_commit) != 40:
            raise ValueError("HCWDL-UB-FULL3 producer commit differs")
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.split = load_json(spec["artifact_paths"]["split_manifest"])
        self.selection_path = Path(spec["artifact_paths"]["selection_manifest"])
        self.resources_path = Path(spec["artifact_paths"]["matcher_resources"])
        self.assignment_root = self.root / "matcher/assignments"

    def _selection(self) -> dict[str, Any]:
        return load_json(self.selection_path)

    def _assignment_manifest(self, role: str) -> Path:
        return Path(self.spec["artifact_paths"][f"{role}_assignment_manifest"])

    def _validated_coupling_lineage(self) -> tuple[dict[str, Any], dict[str, str]]:
        config = load_json(self.root / "coupling/config.json")
        if config["content_hash"] != self.spec["parents"]["coupling_config_sha256"]:
            raise ValueError("HCWDL-UB-FULL3 coupling config differs")
        scale = load_json(self.root / "coupling/scale_calibration.json")
        scale_hash = validate_scale_calibration(
            scale, coupling_config_sha256=config["content_hash"],
        )
        train_base = load_json(self.root / "coupling/train_base_manifest.json")
        train_base_hash = validate_base_manifest(train_base, role="train")
        switch = load_json(self.root / "coupling/switch_calibration.json")
        switch_hash = validate_switch_calibration(
            switch, coupling_config_sha256=config["content_hash"],
            train_base_manifest_sha256=train_base_hash,
        )
        manifests: dict[str, str] = {}
        for role in ("train", "validation"):
            value = load_json(self.root / f"coupling/{role}_manifest.json")
            manifests[role] = validate_coupling_manifest(value, role=role)
            if int(value["rows"]) != int(self.spec["role_counts"][role]):
                raise ValueError("HCWDL-UB-FULL3 coupling role coverage differs")
            ResidualCouplingStore(self.root / f"coupling/{role}_manifest.json")
        audit = load_json(self.root / "coupling/full_role_audit.json")
        audit_hash = validate_coupling_audit(audit)
        return audit, {
            "coupling_config_sha256": config["content_hash"],
            "scale_calibration_sha256": scale_hash,
            "switch_calibration_sha256": switch_hash,
            "train_manifest_sha256": manifests["train"],
            "validation_manifest_sha256": manifests["validation"],
            "audit_sha256": audit_hash,
        }

    def _endpoint_gate(self) -> list[Path]:
        started = time.monotonic()
        balanced_values = {
            role: load_json(self.root / f"balanced/{role}_manifest.json")
            for role in ("train", "validation")
        }
        balanced_hashes = {
            role: validate_balanced_manifest(value, role=role)
            for role, value in balanced_values.items()
        }
        for role, value in balanced_values.items():
            if int(value["rows"]) != int(self.spec["role_counts"][role]):
                raise ValueError("HCWDL-UB-FULL3 balanced role coverage differs")
        split, _, _, selections, assignments, balanced = _load_common(self.spec)
        legacy = {
            role: ResidualCouplingStore(
                self.spec["artifact_paths"][f"legacy_{role}_manifest"]
            ) for role in ("train", "validation")
        }
        repair_seed = derive_seed(
            int(self.spec["replicate_seed"]), "ub_full/repair/v1",
        )
        measured: dict[str, dict[str, int]] = {}
        for role in ("train", "validation"):
            common = {
                "split_manifest": split, "data_root": self.spec["data_root"],
                "role": role, "assignment_store": assignments[role],
                "row_selection": selections[role], "repair_seed": repair_seed,
                "batch_size": min(256, selections[role].rows), "workers": 1,
            }
            old_common = {**common, "coupling_store": legacy[role]}
            new_common = {**common, "coupling_store": balanced[role]}
            measured[role] = {}
            for name, coordinate in (
                ("u000", HomotopyCoordinate(0, 1, 0, 1)),
                ("u100", HomotopyCoordinate(1, 1, 0, 1)),
                ("d0", HomotopyCoordinate(1, 1, 1, 1)),
            ):
                old = next(iterate_homotopy_batches(
                    **old_common, coordinate=coordinate,
                ))["privileged"]
                new = next(iterate_unified_balanced_batches(
                    **new_common, coordinate=coordinate,
                ))["privileged"]
                assert_particle_inputs_equal(new, old, endpoint=name)
                array_bytes = sum(getattr(new, field).nbytes for field in (
                    "features", "vectors", "mask", "raw_lengths",
                ))
                measured[role][name] = math.ceil(array_bytes / len(new.raw_lengths))
        projected_student = sum(
            max(measured[role].values()) * int(self.spec["role_counts"][role])
            for role in ("train", "validation")
        )
        projected_targets = 2 * int(self.spec["role_counts"]["train"]) * 15 * 4
        fixed_allowance = 32 * 1024**3
        projected_peak = projected_student + projected_targets + fixed_allowance
        memory = str(self.spec["resources"]["gpu_training"]["memory"])
        if not memory.endswith("G") or not memory[:-1].isdigit():
            raise ValueError("HCWDL-UB-FULL3 GPU memory format differs")
        requested = int(memory[:-1]) * 1024**3
        if projected_peak > requested * 3 // 4:
            raise MemoryError("HCWDL-UB-FULL3 projected peak exceeds 75% of RAM")
        measurement = with_content_hash({
            "contract": RESOURCE_PROFILE_CONTRACT, "schema_version": 1,
            "foundation_spec_sha256": self.spec["content_hash"],
            "sample_bytes_per_row": measured,
            "projected_simultaneous_student_cache_bytes": projected_student,
            "projected_two_teacher_target_bytes": projected_targets,
            "fixed_runtime_allowance_bytes": fixed_allowance,
            "projected_peak_memory_bytes": projected_peak,
            "requested_memory_bytes": requested,
            "projected_peak_below_75pct_request": True,
            "gpu_class": "gpu:gh200:1", "cpus": 8,
            "walltime": "24:00:00", "durable_repaired_dataset": False,
            "wall_seconds": time.monotonic() - started,
            "final_test_accessed": False,
        })
        measurement_path = self.root / "runtime/resource_profile.json"
        write_immutable_json(measurement_path, measurement)
        assignment_lock = load_json(self.root / "locks/assignment.json")
        coupling_lock = load_json(self.root / "locks/coupling.json")
        lock = endpoint_lock_payload(
            foundation_spec_sha256=self.spec["content_hash"],
            role_rows={
                role: int(self.spec["role_counts"][role])
                for role in ("train", "validation")
            },
            parents={
                "assignment_lock_sha256": validate_assignment_lock(assignment_lock),
                "coupling_lock_sha256": validate_coupling_lock(coupling_lock),
                "train_balanced_manifest_sha256": balanced_hashes["train"],
                "validation_balanced_manifest_sha256": balanced_hashes["validation"],
            },
            resource_measurement_sha256=measurement["content_hash"],
        )
        lock_path = self.root / "locks/endpoint.json"
        write_immutable_json(lock_path, lock)
        return [measurement_path, lock_path]

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec["tasks"], task_id)
        index = _index(task, array_index)
        kind = str(task["kind"])
        if kind == "authenticate":
            validate_foundation_campaign(self.spec, executable=True)
            evidence = authenticate_parent_homotopy(
                self.spec["artifact_paths"]["parent_homotopy_spec"]
            )
            if (
                evidence["spec_hash"]
                != self.spec["parents"]["preparation_template_spec_sha256"]
                or evidence["preparation_lock_hash"]
                != self.spec["parents"]["preparation_template_lock_sha256"]
                or evidence["split_hash"]
                != self.spec["parents"]["split_manifest_sha256"]
            ):
                raise ValueError("HCWDL-UB-FULL3 preparation template drifted")
            payload = with_content_hash({
                "contract": "HCWDL_UNIFIED_BALANCED_FULL_IMPORTED_TEMPLATE/v1",
                "schema_version": 1,
                "foundation_spec_sha256": self.spec["content_hash"],
                "preparation_template_spec_sha256": evidence["spec_hash"],
                "preparation_template_lock_sha256": evidence["preparation_lock_hash"],
                "split_manifest_sha256": evidence["split_hash"],
                "population_policy": "all_authenticated_mapped_rows_v1",
                "final_test_accessed": False,
            })
            output = self.root / "imported_template.json"
            write_immutable_json(output, payload)
            return [output]
        if kind == "row_selection":
            value = build_row_selection(
                self.split, data_root=self.spec["data_root"],
                role_budgets={"train": None, "validation": None},
                seed=int(self.spec["replicate_seed"]),
            )
            if {
                role: int(value["roles"][role]["rows"])
                for role in ("train", "validation")
            } != {
                role: int(self.spec["role_counts"][role])
                for role in ("train", "validation")
            }:
                raise ValueError("HCWDL-UB-FULL3 all-row selection differs")
            write_immutable_json(self.selection_path, value)
            return [self.selection_path]
        if kind == "recipe":
            selection = self._selection()
            selection_hash = validate_content_hash(
                selection, expected_contract=str(selection["contract"]),
                expected_schema_version=int(selection["schema_version"]),
            )
            parent = load_json(self.spec["artifact_paths"]["parent_recipe"])
            recipe = _full_recipe(
                parent=parent, selection=selection,
                selection_sha256=selection_hash,
                overlay_sha256=self.spec["parents"]["recipe_overlay_sha256"],
            )
            output = Path(self.spec["artifact_paths"]["recipe"])
            write_immutable_json(output, recipe)
            return [output]
        if kind == "matcher_resources":
            parent = load_json(self.spec["artifact_paths"]["parent_matcher_resources"])
            digest = validate_content_hash(
                parent, expected_contract=RESOURCE_CONTRACT, expected_schema_version=1,
            )
            if digest != self.spec["parents"]["matcher_resources_sha256"]:
                raise ValueError("HCWDL-UB-FULL3 matcher resources drifted")
            write_immutable_json(self.resources_path, parent)
            return [self.resources_path]
        if kind == "assignment":
            role = "train" if task_id == "assign_train" else "validation"
            return list(build_assignment_source(
                split_manifest=self.split, selection_manifest=self._selection(),
                resources_report=load_json(self.resources_path),
                data_root=self.spec["data_root"], assignment_root=self.assignment_root,
                role=role, source_index=index,
            ))
        if kind == "assignment_manifest":
            outputs: list[Path] = []
            for role in ("train", "validation"):
                manifest_path = self._assignment_manifest(role)
                finalize_role_assignments(
                    split_manifest=self.split, selection_manifest=self._selection(),
                    resources_report=load_json(self.resources_path),
                    assignment_root=self.assignment_root, role=role,
                    output=manifest_path,
                )
                rows = int(self.spec["role_counts"][role])
                audit = sampled_recomputation_audit(
                    manifest_path,
                    recompute=assignment_recomputer(
                        split_manifest=self.split, data_root=self.spec["data_root"],
                        role=role,
                    ),
                    sample_size=min(256, rows), seed=int(self.spec["replicate_seed"]),
                )
                audit_path = self.root / f"matcher/{role}_recomputation_audit.json"
                write_immutable_json(audit_path, audit)
                outputs.extend((manifest_path, audit_path))
            return outputs
        if kind == "assignment_lock":
            selection = self._selection()
            selection_hash = str(selection["content_hash"])
            resources_hash = str(load_json(self.resources_path)["content_hash"])
            parents = {
                "split_manifest_sha256": self.spec["parents"]["split_manifest_sha256"],
                "row_selection_sha256": selection_hash,
                "matcher_resources_sha256": resources_hash,
            }
            manifests: dict[str, str] = {}
            audits: dict[str, str] = {}
            dustbins: dict[str, float] = {}
            for role in ("train", "validation"):
                manifest = validate_assignment_manifest(
                    self._assignment_manifest(role), expected_role=role,
                    expected_mapped_jets=int(self.spec["role_counts"][role]),
                    expected_parents=parents, require_sub10pct_dustbins=True,
                )
                manifests[role] = str(manifest["content_hash"])
                dustbins[role] = float(manifest["dustbin_fraction"])
                audits[role] = str(load_json(
                    self.root / f"matcher/{role}_recomputation_audit.json"
                )["content_hash"])
            payload = assignment_lock_payload(
                foundation_spec_sha256=self.spec["content_hash"],
                role_rows={
                    role: int(self.spec["role_counts"][role])
                    for role in ("train", "validation")
                }, parents=parents, manifests=manifests,
                recomputation_audits=audits, dustbin_fractions=dustbins,
            )
            output = self.root / "locks/assignment.json"
            write_immutable_json(output, payload)
            return [output]
        if kind == "scale_calibration":
            output = self.root / "coupling/scale_calibration.json"
            calibrate_train_scales(
                split_manifest=self.split, selection_manifest=self._selection(),
                assignment_manifest=self._assignment_manifest("train"),
                data_root=self.spec["data_root"],
                coupling_config=load_json(self.root / "coupling/config.json"),
                output=output,
            )
            return [output]
        if kind == "coupling_base":
            role = "train" if task_id == "train_base" else "validation"
            base = self.root / f"coupling/{role}/base/shard_{index:04d}"
            lock = load_json(self.root / "locks/assignment.json")
            build_coupling_source(
                split_manifest=self.split, selection_manifest=self._selection(),
                assignment_manifest=self._assignment_manifest(role),
                data_root=self.spec["data_root"], role=role, source_index=index,
                scale_calibration=load_json(self.root / "coupling/scale_calibration.json"),
                coupling_config_sha256=self.spec["parents"]["coupling_config_sha256"],
                assignment_lock_sha256=validate_assignment_lock(lock),
                qualification_lock_sha256=self.spec["parents"]["parent_shell_lock_sha256"],
                output_base=base, producer_commit=self.producer_commit,
                recovery_spec_sha256=self.recovery_spec_sha256,
            )
            return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if kind == "base_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = Path(self.spec["artifact_paths"][f"{role}_base_manifest"])
            finalize_base_role(
                split_manifest=self.split, selection_manifest=self._selection(),
                role=role, base_root=self.root / "coupling", output=output,
                parents={
                    "foundation_spec_sha256": self.spec["content_hash"],
                    "coupling_config_sha256": self.spec["parents"]["coupling_config_sha256"],
                    "scale_calibration_sha256": load_json(
                        self.root / "coupling/scale_calibration.json"
                    )["content_hash"],
                },
            )
            return [output]
        if kind == "switch_calibration":
            output = self.root / "coupling/switch_calibration.json"
            freeze_switch_calibration(
                train_base_manifest=load_json(
                    self.spec["artifact_paths"]["train_base_manifest"]
                ),
                coupling_config_sha256=self.spec["parents"]["coupling_config_sha256"],
                output=output,
            )
            return [output]
        if kind == "legacy_switch":
            role = "train" if task_id.startswith("train") else "validation"
            base = self.root / f"coupling/{role}/switch/shard_{index:04d}"
            build_switch_sidecar_for_source(
                base_metadata_path=self.root / f"coupling/{role}/base/shard_{index:04d}.json",
                switch_calibration=load_json(self.root / "coupling/switch_calibration.json"),
                coupling_config_sha256=self.spec["parents"]["coupling_config_sha256"],
                output_base=base,
            )
            return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if kind == "legacy_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = Path(self.spec["artifact_paths"][f"legacy_{role}_manifest"])
            finalize_coupling_role(
                role=role,
                base_manifest_path=self.spec["artifact_paths"][f"{role}_base_manifest"],
                sidecar_root=self.root / "coupling",
                switch_calibration_sha256=load_json(
                    self.root / "coupling/switch_calibration.json"
                )["content_hash"],
                output=output,
            )
            return [output]
        if kind == "coupling_audit":
            output = self.root / "coupling/full_role_audit.json"
            audit_full_roles(
                split_manifest=self.split, selection_manifest=self._selection(),
                assignment_manifests={
                    role: self._assignment_manifest(role)
                    for role in ("train", "validation")
                },
                coupling_manifests={
                    role: self.spec["artifact_paths"][f"legacy_{role}_manifest"]
                    for role in ("train", "validation")
                }, data_root=self.spec["data_root"],
                coupling_config_sha256=self.spec["parents"]["coupling_config_sha256"],
                scale_calibration=load_json(self.root / "coupling/scale_calibration.json"),
                switch_calibration=load_json(self.root / "coupling/switch_calibration.json"),
                discrete_seed=derive_seed(
                    int(self.spec["replicate_seed"]), "ub_full/repair/v1",
                ), output=output,
            )
            return [output]
        if kind == "coupling_lock":
            _, expected = self._validated_coupling_lineage()
            payload = build_coupling_lock(
                campaign_spec_sha256=self.spec["content_hash"], **expected,
            )
            output = self.root / "locks/coupling.json"
            write_immutable_json(output, payload)
            return [output]
        if kind == "balanced_config":
            coupling = load_json(self.root / "locks/coupling.json")
            config = balanced_switch_config_payload(
                base_coupling_lock_sha256=validate_coupling_lock(coupling)
            )
            output = self.root / "balanced/config.json"
            write_immutable_json(output, config)
            return [output]
        if kind == "balanced_sidecar":
            role = "train" if task_id.startswith("train") else "validation"
            config = load_json(self.root / "balanced/config.json")
            output = self.root / f"balanced/{role}/shard_{index:04d}"
            build_balanced_sidecar_for_source(
                split_manifest=self.split, selection_manifest=self._selection(),
                assignment_manifest=self._assignment_manifest(role),
                data_root=self.spec["data_root"], role=role, source_index=index,
                base_metadata_path=self.root / f"coupling/{role}/base/shard_{index:04d}.json",
                switch_config_sha256=config["content_hash"], output_base=output,
                producer_commit=self.producer_commit,
            )
            return [output.with_suffix(".npz"), output.with_suffix(".json")]
        if kind == "balanced_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            config = load_json(self.root / "balanced/config.json")
            output = self.root / f"balanced/{role}_manifest.json"
            finalize_balanced_role(
                role=role,
                base_manifest_path=self.spec["artifact_paths"][f"{role}_base_manifest"],
                sidecar_root=self.root / "balanced", output=output,
                switch_config_sha256=config["content_hash"],
            )
            return [output]
        if kind == "endpoint_gate":
            return self._endpoint_gate()
        if kind == "shared_node":
            started = time.monotonic()
            node_id = str(task["node_id"])
            wrapper = run_shared_node(
                foundation_spec=self.spec, node_id=node_id,
            )
            output = shared_node_output_dir(self.root, node_id)
            runtime = _runtime(
                output, scope_spec_sha256=self.spec["content_hash"],
                canonical_node_id=f"shared/{node_id}",
                training_report_sha256=wrapper["pmard_engine_report_sha256"],
                started=started,
            )
            return [
                output / "training_report.json",
                output / "hcwdl_training_report.json", runtime,
            ]
        if kind == "u000_targets":
            return [publish_u000_targets(foundation_spec=self.spec)]
        if kind == "foundation_lock":
            endpoint = load_json(self.root / "locks/endpoint.json")
            assignment = load_json(self.root / "locks/assignment.json")
            coupling = load_json(self.root / "locks/coupling.json")
            balanced = {
                role: load_json(self.root / f"balanced/{role}_manifest.json")
                for role in ("train", "validation")
            }
            recipe = load_json(self.spec["artifact_paths"]["recipe"])
            recipe_hash = validate_recipe(
                recipe, require_authorized=True, expected_profile="full_data_scaleup",
            )
            validate_recipe_class_weight_lineage(recipe, self._selection())
            target = load_json(self.root / "targets/u000_train/manifest.json")
            target_hash = validate_target_manifest(target, teacher_id="shared/U000")
            u000 = load_json(self.root / "training/U000/training_report.json")
            m0 = load_json(self.root / "training/M0paired/training_report.json")
            u000_hash = validate_pmard_training_report(u000)
            m0_hash = validate_pmard_training_report(m0)
            payload = foundation_lock_payload(
                foundation_spec_sha256=self.spec["content_hash"],
                role_counts=self.spec["role_counts"],
                parents={
                    "assignment_lock_sha256": validate_assignment_lock(assignment),
                    "coupling_lock_sha256": validate_coupling_lock(coupling),
                    "endpoint_lock_sha256": validate_endpoint_lock(endpoint),
                    "train_balanced_manifest_sha256": validate_balanced_manifest(
                        balanced["train"], role="train",
                    ),
                    "validation_balanced_manifest_sha256": validate_balanced_manifest(
                        balanced["validation"], role="validation",
                    ),
                    "graph_sha256": self.spec["parents"]["graph_sha256"],
                    "recipe_overlay_sha256": self.spec["parents"]["recipe_overlay_sha256"],
                },
                u000_report_sha256=u000_hash,
                m0paired_report_sha256=m0_hash,
                u000_checkpoint_sha256=u000["selected_checkpoint_sha256"],
                m0paired_checkpoint_sha256=m0["selected_checkpoint_sha256"],
                u000_target_manifest_sha256=target_hash,
                recipe_sha256=recipe_hash,
            )
            output = self.root / "locks/foundation.json"
            write_immutable_json(output, payload)
            return [output]
        raise RuntimeError(f"unhandled HCWDL-UB-FULL3 foundation task kind {kind}")


class UnifiedBalancedFullArmWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
    ) -> None:
        validate_arm_campaign(
            spec, executable=False, verify_source_tree=producer_commit is None,
        )
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.arm_id = str(spec["arm_id"])
        self.producer_commit = str(producer_commit or spec["source_commit"])

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec["tasks"], task_id)
        _index(task, array_index)
        kind = str(task["kind"])
        if kind == "arm_node":
            started = time.monotonic()
            node_id = str(task["node_id"])
            wrapper = run_arm_node(arm_spec=self.spec, node_id=node_id)
            output = arm_node_output_dir(self.root, node_id)
            runtime = _runtime(
                output, scope_spec_sha256=self.spec["content_hash"],
                canonical_node_id=f"{self.arm_id}/{node_id}",
                training_report_sha256=wrapper["pmard_engine_report_sha256"],
                started=started,
            )
            return [
                output / "training_report.json",
                output / "hcwdl_training_report.json", runtime,
            ]
        if kind == "aggregate":
            rows = []
            reports: dict[str, str] = {}
            gpu_hours = 0.0
            ancestry = idealized_u000_ancestry(self.arm_id)
            foundation_root = Path(self.spec["foundation_lock_path"]).parent.parent
            shared = {}
            for node_id in ("U000", "M0paired"):
                report = load_json(
                    foundation_root / f"training/{node_id}/training_report.json"
                )
                shared[node_id] = {
                    "metrics": report["validation"],
                    "report_sha256": validate_pmard_training_report(report),
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                }
            for node_id, node in arm_registry(self.arm_id).items():
                output = self.root / f"training/{node_id}"
                report = load_json(output / "training_report.json")
                report_hash = validate_pmard_training_report(report)
                wrapper = load_json(output / "hcwdl_training_report.json")
                if (
                    wrapper.get("pmard_engine_report_sha256") != report_hash
                    or report.get("scientific_config", {}).get("canonical_node_id")
                    != node.canonical_id
                ):
                    raise ValueError("HCWDL-UB-FULL3 completed node lineage differs")
                runtime = load_json(output / "runtime.json")
                gpu_hours += float(runtime["measured_gpu_hours"])
                rows.append({
                    "node_id": node_id, "canonical_id": node.canonical_id,
                    "parent_id": node.parent_id,
                    "grandparent_id": node.grandparent_id,
                    "coordinate": node.coordinate.payload(),
                    "weights": {
                        "ce": node.ce_weight, "parent_kd": node.parent_kd_weight,
                        "grandparent_kd": node.grandparent_kd_weight,
                    },
                    "idealized_u000_ancestry": ancestry[node_id],
                    "metrics": report["validation"],
                    "selected_update": report["selected_update"],
                    "report_sha256": report_hash,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "runtime_sha256": runtime["content_hash"],
                })
                reports[node_id] = report_hash
            payload = aggregate_payload(
                arm_id=self.arm_id, arm_spec_sha256=self.spec["content_hash"],
                rows=rows, shared=shared, gpu_hours=gpu_hours,
            )
            validate_aggregate(payload)
            output = self.root / "reports/validation_aggregate.json"
            write_immutable_json(output, payload)
            return [output]
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_aggregate(aggregate)
            reports = {
                node_id: load_json(
                    self.root / f"training/{node_id}/training_report.json"
                )["content_hash"]
                for node_id in arm_registry(self.arm_id)
            }
            payload = completion_payload(
                arm_id=self.arm_id, arm_spec_sha256=self.spec["content_hash"],
                aggregate_sha256=aggregate_hash, reports=reports,
                gpu_hours=float(aggregate["gpu_hours"]),
            )
            validate_completion(payload)
            output = self.root / "reports/campaign_complete.json"
            write_immutable_json(output, payload)
            return [output]
        raise RuntimeError(f"unhandled HCWDL-UB-FULL3 arm task kind {kind}")


__all__ = [
    "UnifiedBalancedFullArmWorkflow", "UnifiedBalancedFullFoundationWorkflow",
]
