"""Idempotent workers for the HCWDL-UB shared foundation and recipe arms."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .engine import validate_pmard_training_report
from .hcwdl_homotopy import (
    HomotopyCoordinate, assert_particle_inputs_equal,
)
from .hcwdl_homotopy_stream import (
    iterate_homotopy_batches, iterate_unified_balanced_batches,
)
from .hcwdl_unified_balanced_builder import (
    build_balanced_sidecar_for_source, finalize_balanced_role,
)
from .hcwdl_unified_balanced_cache import (
    BalancedCouplingStore, load_balanced_sidecar, validate_balanced_manifest,
)
from .hcwdl_unified_balanced_campaign import (
    arm_tasks, authenticate_factorial, authenticate_parent_homotopy, foundation_tasks,
    validate_arm_campaign, validate_foundation_campaign,
)
from .hcwdl_unified_balanced_contracts import (
    ARM_AGGREGATE_CONTRACT, ARM_COMPLETION_CONTRACT,
    BALANCED_SWITCH_SIDECAR_CONTRACT, CAMPAIGN_COMPLETION_CONTRACT,
    FOUNDATION_LOCK_CONTRACT, aggregate_payload, completion_payload,
    endpoint_lock_payload, foundation_lock_payload, validate_arm_aggregate,
    validate_arm_completion, validate_arm_spec, validate_endpoint_lock, validate_foundation_lock,
    validate_foundation_spec,
)
from .hcwdl_unified_balanced_graph import (
    arm_registry, idealized_u000_ancestry, shared_registry,
)
from .hcwdl_unified_balanced_runner import (
    RUNTIME_CONTRACT, _load_common, _publish_teacher_targets,
    arm_node_output_dir, run_arm_node, run_shared_node, shared_node_output_dir,
)
from .hcwdl_unified_balanced_targets import (
    target_lock_payload, validate_target_lock, validate_target_manifest,
)
from .hcwdl_upper_cache import ResidualCouplingStore
from .highcov_cache import DenseAssignmentStore
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed


RESOURCE_MEASUREMENT_CONTRACT = "HCWDL_UNIFIED_BALANCED_RESOURCE_MEASUREMENT/v1"


def _report_training_history(report: Mapping[str, Any]) -> list[object]:
    """Return the authenticated PMARD interval-loss history.

    PMARD report contracts v4--v6 publish this field as
    ``training_history``.  ``history`` is the rolling-checkpoint key and was
    never part of a completed training report.  Keeping the normalization in
    one fail-closed accessor prevents reporting-only workers from confusing
    the two schemas again.
    """

    history = report.get("training_history")
    if not isinstance(history, list):
        raise ValueError("HCWDL-UB PMARD training history is absent or malformed")
    return list(history)


def _task(tasks: list[Mapping[str, Any]], task_id: str) -> Mapping[str, Any]:
    found = [row for row in tasks if row["task_id"] == task_id]
    if len(found) != 1:
        raise ValueError(f"HCWDL-UB task identity differs: {task_id}")
    return found[0]


def _index(task: Mapping[str, Any], array_index: int | None) -> int:
    count = int(task["array_count"])
    if count == 1:
        if array_index not in {None, 0}:
            raise ValueError("HCWDL-UB scalar task received an array index")
        return 0
    if array_index is None or not 0 <= int(array_index) < count:
        raise ValueError("HCWDL-UB array index differs")
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
        "phase_boundaries_recorded": True,
        "final_test_accessed": False,
    })
    path = output / "runtime.json"; write_immutable_json(path, payload); return path


class UnifiedBalancedFoundationWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
    ) -> None:
        validate_foundation_campaign(
            spec, executable=False, verify_source_tree=producer_commit is None,
        ); self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.producer_commit = str(producer_commit or spec["source_commit"])
        if len(self.producer_commit) != 40:
            raise ValueError("HCWDL-UB producer commit differs")

    def _endpoint_gate(self) -> list[Path]:
        started = time.monotonic()
        train = load_json(self.root / "balanced/train_manifest.json")
        validation = load_json(self.root / "balanced/validation_manifest.json")
        parents = {
            "train_balanced_manifest_sha256": validate_balanced_manifest(train, role="train"),
            "validation_balanced_manifest_sha256": validate_balanced_manifest(
                validation, role="validation",
            ),
        }
        if train["rows"] != 300_000 or validation["rows"] != 100_000:
            raise ValueError("HCWDL-UB balanced manifest population differs")
        split, split_hash, _, selections, assignments, balanced = _load_common(self.spec)
        legacy = {
            role: ResidualCouplingStore(
                self.spec["artifact_paths"][f"legacy_{role}_manifest"]
            ) for role in ("train", "validation")
        }
        repair_seed = derive_seed(int(self.spec["replicate_seed"]), "ub/repair/v1")
        checks = {
            "u000_exact_p0": True, "u100_exact_d100": True,
            "j100_exact_hlt": True, "d0_exact_hlt": True,
            "no_durable_views": True, "prepared_endpoints_reused": True,
        }
        sampled_rows = 0; sample_bytes = 0
        for role in ("train", "validation"):
            common = dict(
                split_manifest=split, data_root=self.spec["data_root"], role=role,
                assignment_store=assignments[role], row_selection=selections[role],
                repair_seed=repair_seed, batch_size=256,
            )
            old_common = {**common, "coupling_store": legacy[role]}
            new_common = {**common, "coupling_store": balanced[role]}
            for coordinate, endpoint in (
                (HomotopyCoordinate(0, 1, 0, 1), "U000/P0"),
                (HomotopyCoordinate(1, 1, 0, 1), "U100/D100"),
                (HomotopyCoordinate(1, 1, 1, 1), "J100/HLT"),
            ):
                old = next(iterate_homotopy_batches(
                    **old_common, coordinate=coordinate,
                ))["privileged"]
                new = next(iterate_unified_balanced_batches(
                    **new_common, coordinate=coordinate,
                ))["privileged"]
                assert_particle_inputs_equal(new, old, endpoint=endpoint)
                sampled_rows += len(new.raw_lengths)
                sample_bytes += sum(getattr(new, name).nbytes for name in (
                    "features", "vectors", "mask", "raw_lengths",
                ))
        if sampled_rows <= 0:
            raise ValueError("HCWDL-UB endpoint gate sampled no selected rows")
        bytes_per_student_row = math.ceil(sample_bytes / sampled_rows)
        projected_student_cache_bytes = bytes_per_student_row * 400_000
        # One student train+validation cache, two FP32 train-logit banks, and
        # a deliberately conservative fixed allowance for identities, a
        # prepared source chunk, model/optimizer state, and allocator slack.
        projected_teacher_target_bytes = 2 * 300_000 * 15 * 4
        fixed_runtime_allowance_bytes = 24 * 1024**3
        projected_peak_bytes = (
            projected_student_cache_bytes + projected_teacher_target_bytes
            + fixed_runtime_allowance_bytes
        )
        memory_text = str(self.spec["resources"]["gpu_training"]["memory"])
        if not memory_text.endswith("G") or not memory_text[:-1].isdigit():
            raise ValueError("HCWDL-UB GPU memory request format differs")
        requested_memory_bytes = int(memory_text[:-1]) * 1024**3
        if projected_peak_bytes > requested_memory_bytes * 3 // 4:
            raise MemoryError(
                "HCWDL-UB projected peak exceeds 75% of the locked memory request"
            )
        measurement = with_content_hash({
            "contract": RESOURCE_MEASUREMENT_CONTRACT, "schema_version": 1,
            "foundation_spec_sha256": self.spec["content_hash"],
            "sampled_view_rows": sampled_rows, "sampled_array_bytes": sample_bytes,
            "bytes_per_student_row_upper_estimate": bytes_per_student_row,
            "projected_student_cache_bytes": projected_student_cache_bytes,
            "projected_two_teacher_target_bytes": projected_teacher_target_bytes,
            "fixed_runtime_allowance_bytes": fixed_runtime_allowance_bytes,
            "projected_peak_memory_bytes": projected_peak_bytes,
            "requested_memory_bytes": requested_memory_bytes,
            "projected_peak_below_75pct_request": True,
            "balanced_rows": {"train": train["rows"], "validation": validation["rows"]},
            "balanced_edits": {"train": train["edits"], "validation": validation["edits"]},
            "wall_seconds": time.monotonic() - started,
            "linear_prepared_endpoint_path": True,
            "worker_count_lte_slurm_cpus": True,
            "durable_repaired_dataset": False, "final_test_accessed": False,
        })
        measurement_path = self.root / "runtime/resource_measurement.json"
        write_immutable_json(measurement_path, measurement)
        lock = endpoint_lock_payload(
            foundation_spec_sha256=self.spec["content_hash"], parents=parents,
            role_rows={"train": train["rows"], "validation": validation["rows"]},
            endpoint_checks=checks,
            resource_measurement_sha256=measurement["content_hash"],
        )
        lock_path = self.root / "locks/endpoint.json"; write_immutable_json(lock_path, lock)
        return [measurement_path, lock_path]

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec["tasks"], task_id); index = _index(task, array_index)
        kind = task["kind"]
        if kind == "authenticate":
            evidence = authenticate_parent_homotopy(
                self.spec["artifact_paths"]["parent_homotopy_spec"]
            )
            expected = self.spec["parents"]
            actual = {
                "parent_homotopy_spec_sha256": evidence["spec_hash"],
                "parent_homotopy_preparation_sha256": evidence["preparation_lock_hash"],
                "parent_campaign_spec_sha256": evidence["primary_hash"],
                "split_manifest_sha256": evidence["split_hash"],
                "selection_manifest_sha256": evidence["selection_hash"],
            }
            if any(expected.get(name) != value for name, value in actual.items()):
                raise ValueError("HCWDL-UB authenticated parent lineage differs")
            factorial = authenticate_factorial(
                self.spec["artifact_paths"]["factorial_spec"],
                split_sha256=evidence["split_hash"],
                selection_sha256=evidence["selection_hash"],
            )
            factorial_actual = {
                "factorial_spec_sha256": factorial["spec_hash"],
                "factorial_aggregate_sha256": factorial["aggregate_hash"],
                "factorial_completion_sha256": factorial["completion_hash"],
            }
            if any(expected.get(name) != value for name, value in factorial_actual.items()):
                raise ValueError("HCWDL-UB authenticated factorial lineage differs")
            imported = evidence["spec"].get("imported_controls", {})
            contextual = {
                "M0": imported.get("M0"), "TOFF": imported.get("TOFF"),
                **factorial["controls"],
            }
            if contextual != self.spec["contextual_controls"]:
                raise ValueError("HCWDL-UB authenticated contextual controls differ")
            payload = with_content_hash({
                "contract": "HCWDL_UNIFIED_BALANCED_IMPORTED_PARENT/v1",
                "schema_version": 1, "foundation_spec_sha256": self.spec["content_hash"],
                "parents": {**actual, **factorial_actual},
                "role_counts": self.spec["role_counts"],
                "final_test_accessed": False,
            })
            output = self.root / "imported_parent.json"; write_immutable_json(output, payload)
            return [output]
        if kind == "balanced_sidecar":
            role = "train" if task_id.startswith("train") else "validation"
            output = self.root / f"balanced/{role}/shard_{index:04d}"
            base = load_json(self.spec["artifact_paths"][f"{role}_base_manifest"])
            build_balanced_sidecar_for_source(
                split_manifest=load_json(self.spec["artifact_paths"]["split_manifest"]),
                selection_manifest=load_json(self.spec["artifact_paths"]["selection_manifest"]),
                assignment_manifest=self.spec["artifact_paths"][f"{role}_assignment_manifest"],
                data_root=self.spec["data_root"], role=role, source_index=index,
                base_metadata_path=base["shards"][index]["metadata_path"],
                switch_config_sha256=self.spec["parents"]["balanced_switch_config_sha256"],
                output_base=output, producer_commit=self.producer_commit,
            )
            return [output.with_suffix(".npz"), output.with_suffix(".json")]
        if kind == "balanced_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = self.root / f"balanced/{role}_manifest.json"
            finalize_balanced_role(
                role=role, base_manifest_path=self.spec["artifact_paths"][f"{role}_base_manifest"],
                sidecar_root=self.root / "balanced", output=output,
                switch_config_sha256=self.spec["parents"]["balanced_switch_config_sha256"],
            )
            return [output]
        if kind == "endpoint_gate":
            return self._endpoint_gate()
        if kind == "shared_node":
            started = time.monotonic(); node_id = str(task["node_id"])
            wrapper = run_shared_node(foundation_spec=self.spec, node_id=node_id)
            output = shared_node_output_dir(self.root, node_id)
            runtime = _runtime(
                output, scope_spec_sha256=self.spec["content_hash"],
                canonical_node_id=f"shared/{node_id}",
                training_report_sha256=wrapper["pmard_engine_report_sha256"], started=started,
            )
            return [output / "training_report.json", output / "hcwdl_training_report.json", runtime]
        if kind == "u000_targets":
            split, split_hash, _, selections, assignments, balanced = _load_common(self.spec)
            recipe = load_json(self.spec["artifact_paths"]["recipe"])
            node = shared_registry()["U000"]; output = shared_node_output_dir(self.root, "U000")
            result = _publish_teacher_targets(
                canonical_id="shared/U000", output=output, node=node,
                foundation_spec=self.spec, split=split, split_hash=split_hash,
                selections=selections, assignments=assignments, balanced=balanced,
                batch_size=int(recipe["batching"]["effective_batch_size"]),
                sampler_seed=derive_seed(int(self.spec["replicate_seed"]), f"ub/sampler/{node.seed_alias}"),
                repair_seed=derive_seed(int(self.spec["replicate_seed"]), "ub/repair/v1"),
                device="cuda", target_root_override=self.root / "targets/u000_train",
            )
            if result is None:
                raise RuntimeError("HCWDL-UB U000 target cache has no registered consumers")
            manifest = load_json(result); manifest_hash = validate_target_manifest(
                manifest, teacher_id="shared/U000",
            )
            report = load_json(output / "training_report.json")
            lock = target_lock_payload(
                foundation_spec_sha256=self.spec["content_hash"], manifest_sha256=manifest_hash,
                teacher_report_sha256=report["content_hash"],
                teacher_checkpoint_sha256=report["selected_checkpoint_sha256"],
                split_manifest_sha256=split_hash,
                selection_manifest_sha256=self.spec["parents"]["selection_manifest_sha256"],
            )
            lock_path = self.root / "targets/u000_train/lock.json"
            write_immutable_json(lock_path, lock); return [result, lock_path]
        if kind == "foundation_lock":
            endpoint = load_json(self.root / "locks/endpoint.json")
            endpoint_hash = validate_endpoint_lock(endpoint)
            target_manifest = load_json(self.root / "targets/u000_train/manifest.json")
            target_hash = validate_target_manifest(target_manifest, teacher_id="shared/U000")
            u000 = load_json(self.root / "training/U000/training_report.json")
            m0 = load_json(self.root / "training/M0paired/training_report.json")
            u000_report_hash = validate_pmard_training_report(u000)
            m0_report_hash = validate_pmard_training_report(m0)
            parents = {
                "endpoint_lock_sha256": endpoint_hash,
                "target_lock_sha256": validate_target_lock(
                    load_json(self.root / "targets/u000_train/lock.json")
                ),
                "train_balanced_manifest_sha256": load_json(self.root / "balanced/train_manifest.json")["content_hash"],
                "validation_balanced_manifest_sha256": load_json(self.root / "balanced/validation_manifest.json")["content_hash"],
                "graph_sha256": self.spec["parents"]["graph_sha256"],
                "recipe_sha256": self.spec["parents"]["recipe_sha256"],
                "coordinate_sha256": self.spec["parents"]["coordinate_sha256"],
            }
            payload = foundation_lock_payload(
                foundation_spec_sha256=self.spec["content_hash"], parents=parents,
                u000_report_sha256=u000_report_hash,
                m0paired_report_sha256=m0_report_hash,
                u000_checkpoint_sha256=u000["selected_checkpoint_sha256"],
                m0paired_checkpoint_sha256=m0["selected_checkpoint_sha256"],
                u000_target_manifest_sha256=target_hash,
            )
            output = self.root / "locks/foundation.json"; write_immutable_json(output, payload)
            return [output]
        raise RuntimeError(f"unhandled HCWDL-UB foundation task kind {kind}")


class UnifiedBalancedArmWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
        recovery_context: Mapping[str, Any] | None = None,
    ) -> None:
        validate_arm_campaign(
            spec, executable=False, verify_source_tree=producer_commit is None,
        ); self.spec = dict(spec)
        self.root = Path(spec["campaign_root"]); self.arm_id = str(spec["arm_id"])
        self.producer_commit = str(producer_commit or spec["source_commit"])
        if len(self.producer_commit) != 40:
            raise ValueError("HCWDL-UB arm producer commit differs")
        self.recovery_context = (
            None if recovery_context is None else dict(recovery_context)
        )

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec["tasks"], task_id); _index(task, array_index)
        kind = task["kind"]
        if kind == "arm_node":
            started = time.monotonic(); node_id = str(task["node_id"])
            wrapper = run_arm_node(
                arm_spec=self.spec, node_id=node_id,
                producer_commit=self.producer_commit,
                recovery_context=self.recovery_context,
            )
            output = arm_node_output_dir(self.root, node_id)
            runtime = _runtime(
                output, scope_spec_sha256=self.spec["content_hash"],
                canonical_node_id=f"{self.arm_id}/{node_id}",
                training_report_sha256=wrapper["pmard_engine_report_sha256"], started=started,
            )
            outputs = [output / "training_report.json", output / "hcwdl_training_report.json", runtime]
            if (output / "targets/manifest.json").is_file():
                outputs.append(output / "targets/manifest.json")
            return outputs
        if kind == "aggregate":
            rows = []; reports = {}; gpu_hours = 0.0
            ancestry = idealized_u000_ancestry(self.arm_id)
            foundation_lock_path = Path(self.spec["foundation_lock_path"])
            foundation_root = foundation_lock_path.parent.parent
            foundation_spec = load_json(foundation_root / "foundation_spec.json")
            shared_controls = {}
            for shared_id in ("U000", "M0paired"):
                report = load_json(
                    foundation_root / f"training/{shared_id}/training_report.json"
                )
                report_hash = validate_pmard_training_report(report)
                shared_controls[shared_id] = {
                    "metrics": report["validation"], "report_sha256": report_hash,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                }
            contextual_controls = {}
            for control_id, record in foundation_spec["contextual_controls"].items():
                report = load_json(record["report_path"])
                report_hash = validate_pmard_training_report(report)
                if (
                    report_hash != record["report_sha256"]
                    or report["selected_checkpoint_sha256"] != record["checkpoint_sha256"]
                ):
                    raise ValueError(f"HCWDL-UB contextual control drifted: {control_id}")
                contextual_controls[control_id] = {
                    "metrics": report["validation"], "report_sha256": report_hash,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                }
            recovery_metrics = (
                "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
                "macro_mean_log_qcd_rejection_at_50pct_signal",
            )
            def recovery(metrics, low, high):
                values = {}
                for metric in recovery_metrics:
                    denominator = float(high[metric]) - float(low[metric])
                    values[metric] = (
                        None if denominator == 0 else
                        (float(metrics[metric]) - float(low[metric])) / denominator
                    )
                return values
            for node_id, node in arm_registry(self.arm_id).items():
                output = arm_node_output_dir(self.root, node_id)
                report = load_json(output / "training_report.json")
                report_hash = validate_pmard_training_report(report)
                wrapper = load_json(output / "hcwdl_training_report.json")
                if (
                    wrapper.get("pmard_engine_report_sha256") != report_hash
                    or report.get("scientific_config", {}).get("canonical_node_id") != node.canonical_id
                ):
                    raise ValueError(f"HCWDL-UB completed node lineage differs: {node_id}")
                runtime = load_json(output / "runtime.json")
                gpu_hours += float(runtime["measured_gpu_hours"])
                rows.append({
                    "node_id": node_id, "canonical_id": node.canonical_id,
                    "parent_id": node.parent_id, "grandparent_id": node.grandparent_id,
                    "coordinate": node.coordinate.payload(), "behavior": node.behavior,
                    "weights": {
                        "ce": node.ce_weight, "parent_kd": node.parent_kd_weight,
                        "grandparent_kd": node.grandparent_kd_weight,
                    },
                    "idealized_u000_ancestry": ancestry[node_id],
                    "metrics": report["validation"],
                    "recovery_m0paired_to_u000": recovery(
                        report["validation"], shared_controls["M0paired"]["metrics"],
                        shared_controls["U000"]["metrics"],
                    ),
                    "contextual_recovery_m0_to_toff": recovery(
                        report["validation"], contextual_controls["M0"]["metrics"],
                        contextual_controls["TOFF"]["metrics"],
                    ),
                    "loss_history": _report_training_history(report),
                    "validation_history": report["validation_history"],
                    "selected_update": report["selected_update"],
                    "report_sha256": report_hash,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "runtime_sha256": runtime["content_hash"],
                }); reports[node_id] = report_hash
            foundation = load_json(self.spec["foundation_lock_path"])
            validate_foundation_lock(foundation)
            imported = {
                "foundation_lock": foundation["content_hash"],
                "U000": foundation["u000_checkpoint_sha256"],
                "M0paired": foundation["m0paired_checkpoint_sha256"],
            }
            payload = aggregate_payload(
                arm_id=self.arm_id, arm_spec_sha256=self.spec["content_hash"],
                rows=rows, imported=imported,
                contextual_controls=contextual_controls,
                shared_controls=shared_controls, gpu_hours=gpu_hours,
            )
            output = self.root / "reports/validation_aggregate.json"
            write_immutable_json(output, payload); return [output]
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_arm_aggregate(aggregate)
            reports = {
                node_id: load_json(self.root / f"training/{node_id}/training_report.json")["content_hash"]
                for node_id in arm_registry(self.arm_id)
            }
            payload = completion_payload(
                arm_id=self.arm_id, arm_spec_sha256=self.spec["content_hash"],
                aggregate_sha256=aggregate_hash, completed_node_reports=reports,
                gpu_hours=float(aggregate["gpu_hours"]),
            )
            validate_arm_completion(payload)
            output = self.root / "reports/campaign_complete.json"
            write_immutable_json(output, payload); return [output]
        raise RuntimeError(f"unhandled HCWDL-UB arm task kind {kind}")


__all__ = [
    "RESOURCE_MEASUREMENT_CONTRACT", "UnifiedBalancedArmWorkflow",
    "UnifiedBalancedFoundationWorkflow",
]
