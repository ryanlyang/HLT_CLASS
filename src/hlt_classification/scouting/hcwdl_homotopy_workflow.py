"""Filesystem-bound dispatcher for every HCWDL-UJ campaign task."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file, validate_content_hash,
    with_content_hash, write_immutable_json,
)

from .hcwdl_homotopy import HomotopyCoordinate, particle_inputs_sha256
from .hcwdl_homotopy_campaign import (
    build_command_plan, semantic_source_hashes, validate_campaign,
    validate_worker_semantics,
)
from .hcwdl_homotopy_contracts import (
    AGGREGATE_CONTRACT, CACHE_RESOURCE_MEASUREMENT_CONTRACT,
    CACHE_MINIATURE_CONTRACT,
    CAMPAIGN_COMPLETION_CONTRACT,
    COMMAND_PLAN_CONTRACT,
    COUPLING_AUDIT_CONTRACT, COUPLING_LOCK_CONTRACT,
    ENDPOINT_LOCK_CONTRACT, GRAPH_RECIPE_LOCK_CONTRACT,
    TARGET_RESOURCE_MEASUREMENT_CONTRACT, TOFF_TARGET_LOCK_CONTRACT,
    validate_coordinate, validate_coupling_config,
)
from .hcwdl_homotopy_graph import FIT_COUNT
from .hcwdl_homotopy_locks import (
    build_endpoint_equality_lock, build_graph_recipe_lock,
    validate_endpoint_equality_lock, validate_graph_recipe_lock,
)
from .hcwdl_homotopy_reporting import (
    build_campaign_completion, build_validation_aggregate,
)
from .hcwdl_homotopy_runner import run_homotopy_node
from .hcwdl_homotopy_stream import iterate_homotopy_batches
from .hcwdl_toff_targets import (
    build_toff_target_cache, build_toff_target_lock,
    validate_toff_target_lock, validate_toff_target_manifest,
)
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
from .highcov_cache import DenseAssignmentStore
from .selective_assignment import RowSelection
from .splits import role_records
from .training import derive_seed


class HomotopyWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, repository: str | Path,
        producer_commit: str | None = None,
        recovery_spec_sha256: str | None = None,
        execution_semantic_source_sha256: Mapping[str, str] | None = None,
    ) -> None:
        validate_campaign(spec, executable=False)
        if execution_semantic_source_sha256 is None:
            validate_worker_semantics(spec, repository=repository)
        elif (
            dict(execution_semantic_source_sha256)
            != semantic_source_hashes(repository)
        ):
            raise ValueError("HCWDL-UJ recovery execution source differs")
        self.spec = dict(spec); self.root = Path(spec["campaign_root"])
        self.repository = Path(repository)
        self.producer_commit = str(producer_commit or spec["source_commit"])
        if len(self.producer_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.producer_commit
        ):
            raise ValueError("HCWDL-UJ producer commit differs")
        self.recovery_spec_sha256 = recovery_spec_sha256
        if self.producer_commit != spec["source_commit"]:
            require_sha256(recovery_spec_sha256, name="source recovery specification")
            if execution_semantic_source_sha256 is None:
                raise ValueError("HCWDL-UJ source recovery lineage is absent")
        elif execution_semantic_source_sha256 is not None:
            raise ValueError("HCWDL-UJ unchanged-source worker received recovery lineage")
        self.split = load_json(spec["split_manifest_path"])
        self.selection = load_json(spec["selection_manifest_path"])
        self.config = load_json(self.root / "coupling/config.json")

    def _task(self, task_id: str) -> Mapping[str, Any]:
        tasks = {str(row["task_id"]): row for row in self.spec["tasks"]}
        if task_id not in tasks:
            raise ValueError(f"unknown HCWDL-UJ task {task_id!r}")
        return tasks[task_id]

    def _source_index(self, task: Mapping[str, Any], array_index: int | None) -> int:
        count = int(task["array_count"])
        if count == 1:
            if array_index not in {None, 0}:
                raise ValueError("non-array HCWDL-UJ task received an array index")
            return 0
        if array_index is None or not 0 <= int(array_index) < count:
            raise ValueError("HCWDL-UJ array index differs")
        return int(array_index)

    def _load_json_output(
        self, path: Path, contract: str,
        *, expected: Mapping[str, object] | None = None,
    ) -> list[Path] | None:
        if not path.exists():
            return None
        value = load_json(path)
        validate_content_hash(value, expected_contract=contract, expected_schema_version=1)
        if value.get("final_test_accessed", False) is not False:
            raise PermissionError("reused HCWDL-UJ artifact accessed final test")
        if expected is not None and any(value.get(name) != item for name, item in expected.items()):
            raise ValueError(f"reused HCWDL-UJ artifact lineage differs: {path}")
        return [path]

    def _validated_coupling_lineage(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Reopen the complete immutable coupling closure and return lock parents."""

        config = load_json(self.root / "coupling/config.json")
        if validate_coupling_config(config) != self.spec["coupling_config_sha256"]:
            raise ValueError("HCWDL-UJ coupling configuration/spec lineage differs")
        scale = load_json(self.root / "coupling/scale_calibration.json")
        scale_hash = validate_scale_calibration(
            scale, coupling_config_sha256=config["content_hash"],
        )
        train_base = load_json(self.root / "coupling/train_base_manifest.json")
        train_base_hash = validate_base_manifest(train_base, role="train")
        if int(train_base.get("expected_rows", -1)) != int(self.spec["role_counts"]["train"]):
            raise ValueError("HCWDL-UJ train base-manifest coverage differs")
        switch = load_json(self.root / "coupling/switch_calibration.json")
        switch_hash = validate_switch_calibration(
            switch, coupling_config_sha256=config["content_hash"],
            train_base_manifest_sha256=train_base_hash,
        )
        manifests = {}
        for role in ("train", "validation"):
            manifest = load_json(self.root / f"coupling/{role}_manifest.json")
            digest = validate_coupling_manifest(manifest, role=role)
            if int(manifest.get("rows", -1)) != int(self.spec["role_counts"][role]):
                raise ValueError(f"HCWDL-UJ {role} coupling-manifest coverage differs")
            # Opening the store authenticates the manifest shape; the full audit below
            # independently reopens every shard and proves endpoint conservation.
            ResidualCouplingStore(self.root / f"coupling/{role}_manifest.json")
            manifests[role] = digest
        audit = load_json(self.root / "coupling/full_role_audit.json")
        audit_hash = validate_coupling_audit(audit)
        expected_audit = {
            "coupling_config_sha256": config["content_hash"],
            "switch_calibration_sha256": switch_hash,
            "train_manifest_sha256": manifests["train"],
            "validation_manifest_sha256": manifests["validation"],
            "expected_rows": {
                role: int(self.spec["role_counts"][role])
                for role in ("train", "validation")
            },
        }
        if any(audit.get(name) != value for name, value in expected_audit.items()):
            raise ValueError("HCWDL-UJ full-role audit lineage differs")
        return audit, {
            "coupling_config_sha256": config["content_hash"],
            "scale_calibration_sha256": scale_hash,
            "switch_calibration_sha256": switch_hash,
            "train_manifest_sha256": manifests["train"],
            "validation_manifest_sha256": manifests["validation"],
            "audit_sha256": audit_hash,
        }

    def _validated_toff_lineage(self) -> tuple[dict[str, Any], dict[str, str]]:
        manifest = load_json(self.root / "targets/toff_train/manifest.json")
        manifest_hash = validate_toff_target_manifest(manifest)
        imported = self.spec["imported_controls"]["TOFF"]
        expected_parents = {
            "campaign_spec_sha256": self.spec["content_hash"],
            "split_manifest_sha256": self.spec["split_manifest_sha256"],
            "selection_manifest_sha256": self.spec["selection_manifest_sha256"],
            "teacher_report_sha256": imported["report_sha256"],
            "teacher_checkpoint_sha256": imported["checkpoint_sha256"],
        }
        if (
            int(manifest.get("rows", -1)) != int(self.spec["role_counts"]["train"])
            or any(manifest.get("parents", {}).get(name) != value for name, value in expected_parents.items())
        ):
            raise ValueError("HCWDL-UJ TOFF target manifest lineage differs")
        return manifest, {
            "manifest_sha256": manifest_hash,
            "teacher_report_sha256": imported["report_sha256"],
            "teacher_checkpoint_sha256": imported["checkpoint_sha256"],
            "split_manifest_sha256": self.spec["split_manifest_sha256"],
            "selection_manifest_sha256": self.spec["selection_manifest_sha256"],
            "native_adapter_sha256": manifest["parents"]["native_adapter_sha256"],
            "input_projection_sha256": manifest["parents"]["input_projection_sha256"],
            "inference_policy_sha256": manifest["parents"]["inference_policy_sha256"],
        }

    def _cache_miniature(self) -> dict[str, Any]:
        started = time.monotonic()
        split_hash = self.spec["split_manifest_sha256"]
        repair_seed = derive_seed(int(self.spec["replicate_seed"]), "hcwdl_uj/repair/shared_v1")
        rows = {}; total_bytes = 0; bytes_by_view = {}; hashes = {}
        for role in ("train", "validation"):
            selection = RowSelection(self.selection, role=role, split_manifest_sha256=split_hash)
            assignment = DenseAssignmentStore(self.spec["assignment_manifests"][role])
            coupling = ResidualCouplingStore(self.root / f"coupling/{role}_manifest.json")
            for name, coordinate in (
                ("p0", HomotopyCoordinate(0, 1, 0, 1)),
                ("u020", HomotopyCoordinate(1, 5, 0, 1)),
                ("j010", HomotopyCoordinate(1, 10, 1, 10)),
                ("u100", HomotopyCoordinate(1, 1, 0, 1)),
                ("j100", HomotopyCoordinate(1, 1, 1, 1)),
            ):
                batch = next(iterate_homotopy_batches(
                    self.split, data_root=self.spec["data_root"], role=role,
                    assignment_store=assignment, coupling_store=coupling,
                    row_selection=selection, coordinate=coordinate,
                    repair_seed=repair_seed, batch_size=min(256, selection.rows),
                ))
                view = batch["privileged"]
                key = f"{role}:{name}"; rows[key] = len(batch["labels"])
                hashes[key] = particle_inputs_sha256(view)
                current_bytes = sum(getattr(view, field).nbytes for field in (
                    "features", "vectors", "mask", "raw_lengths",
                ))
                bytes_by_view[key] = current_bytes; total_bytes += current_bytes
        # A job holds one train view and one validation view concurrently. Use the
        # largest measured per-row coordinate independently for each role.
        estimated_student_bytes = sum(
            int(np.ceil(max(
                bytes_by_view[key] / rows[key]
                for key in rows if key.startswith(role + ":")
            ) * int(self.spec["role_counts"][role])))
            for role in ("train", "validation")
        )
        peak_gpu = 0
        try:
            import torch
            if torch.cuda.is_available():
                peak_gpu = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass
        payload = with_content_hash({
            "contract": CACHE_MINIATURE_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "sample_rows": rows, "view_sha256": hashes,
            "sample_array_bytes": total_bytes,
            "sample_array_bytes_by_view": bytes_by_view,
            "estimated_simultaneous_student_cache_bytes": estimated_student_bytes,
            "durable_repaired_dataset": False, "matcher_callable_present": False,
            "final_test_accessed": False,
        })
        try:
            import resource
            peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except ImportError:  # pragma: no cover - Windows local contract tests
            peak_rss_kib = 0
        measurement = with_content_hash({
            "contract": CACHE_RESOURCE_MEASUREMENT_CONTRACT, "schema_version": 1,
            "campaign_spec_sha256": self.spec["content_hash"],
            "cache_miniature_sha256": payload["content_hash"],
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": peak_rss_kib,
            "peak_gpu_bytes": peak_gpu,
            "sample_array_bytes": total_bytes,
            "sample_array_bytes_by_view": bytes_by_view,
            "estimated_student_cache_bytes": estimated_student_bytes,
            "final_test_accessed": False,
        })
        write_immutable_json(
            self.root / "runtime/cache_resource_measurement.json", measurement,
        )
        return payload

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = self._task(task_id); kind = str(task["kind"])
        index = self._source_index(task, array_index)
        role = "train" if task_id.startswith("train_") and kind != "train_node" else "validation"
        if kind == "authenticate":
            output = self.root / "imported_parent.json"
            reused = self._load_json_output(
                output, "HCWDL_STRUCTURAL_FEATURE_IMPORTED_PARENT/v1",
                expected={
                    "campaign_spec_sha256": self.spec["content_hash"],
                    "parent_campaign_spec_sha256": self.spec["parent_campaign_spec_sha256"],
                },
            )
            if reused:
                validate_campaign(self.spec, executable=True)
                return reused
            # The campaign validator reopens and authenticates every bound parent.
            validate_campaign(self.spec, executable=True)
            command_plan = build_command_plan(self.spec)
            if command_plan["content_hash"] != self.spec["command_plan_sha256"]:
                raise ValueError("HCWDL-UJ command plan differs from campaign binding")
            payload = with_content_hash({
                "contract": "HCWDL_STRUCTURAL_FEATURE_IMPORTED_PARENT/v1",
                "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "parent_campaign_spec_sha256": self.spec["parent_campaign_spec_sha256"],
                "split_manifest_sha256": self.spec["split_manifest_sha256"],
                "selection_manifest_sha256": self.spec["selection_manifest_sha256"],
                "recipe_sha256": self.spec["recipe_sha256"],
                "assignment_manifest_sha256": self.spec["assignment_manifest_sha256"],
                "imported_controls": self.spec["imported_controls"],
                "final_test_accessed": False,
            })
            write_immutable_json(output, payload); return [output]
        if kind == "upper_calibration":
            output = self.root / "coupling/scale_calibration.json"
            reused = self._load_json_output(
                output, "HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1",
                expected={
                    "coupling_config_sha256": self.spec["coupling_config_sha256"],
                    "split_manifest_sha256": self.spec["split_manifest_sha256"],
                    "selection_manifest_sha256": self.spec["selection_manifest_sha256"],
                },
            )
            if reused: return reused
            calibrate_train_scales(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifest=self.spec["assignment_manifests"]["train"],
                data_root=self.spec["data_root"], coupling_config=self.config,
                output=output,
            ); return [output]
        if kind == "coupling_base":
            role = "train" if task_id == "train_base" else "validation"
            base = self.root / f"coupling/{role}/base/shard_{index:04d}"
            if base.with_suffix(".json").exists():
                metadata, _ = load_base_shard(base.with_suffix(".json"))
                expected_source = role_records(self.split, role)[index].path
                if (
                    metadata.get("role") != role
                    or metadata.get("source_path") != expected_source
                    or metadata.get("producer_commit") not in {
                        self.spec["source_commit"], self.producer_commit,
                    }
                    or metadata.get("parents", {}).get("coupling_config_sha256")
                       != self.spec["coupling_config_sha256"]
                ):
                    raise ValueError("reused HCWDL-UJ coupling base lineage differs")
                return [base.with_suffix(".npz"), base.with_suffix(".json")]
            build_coupling_source(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifest=self.spec["assignment_manifests"][role],
                data_root=self.spec["data_root"], role=role, source_index=index,
                scale_calibration=load_json(self.root / "coupling/scale_calibration.json"),
                coupling_config_sha256=self.spec["coupling_config_sha256"],
                assignment_lock_sha256=self.spec["assignment_lock_sha256"],
                qualification_lock_sha256=self.spec["shell_qualification_lock_sha256"],
                output_base=base, producer_commit=self.producer_commit,
                recovery_spec_sha256=self.recovery_spec_sha256,
            ); return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if kind == "base_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = self.root / f"coupling/{role}_base_manifest.json"
            if output.exists():
                value = load_json(output); validate_base_manifest(value, role=role)
                if value.get("expected_rows") != int(self.spec["role_counts"][role]):
                    raise ValueError("reused HCWDL-UJ base-manifest coverage differs")
                return [output]
            finalize_base_role(
                split_manifest=self.split, selection_manifest=self.selection,
                role=role, base_root=self.root / "coupling", output=output,
                parents={
                    "campaign_spec_sha256": self.spec["content_hash"],
                    "coupling_config_sha256": self.spec["coupling_config_sha256"],
                    "scale_calibration_sha256": load_json(self.root / "coupling/scale_calibration.json")["content_hash"],
                },
            ); return [output]
        if kind == "switch_calibration":
            output = self.root / "coupling/switch_calibration.json"
            reused = self._load_json_output(
                output, "HCWDL_RESIDUAL_SHELL_SWITCH_CALIBRATION/v1",
                expected={"coupling_config_sha256": self.spec["coupling_config_sha256"]},
            )
            if reused: return reused
            freeze_switch_calibration(
                train_base_manifest=load_json(self.root / "coupling/train_base_manifest.json"),
                coupling_config_sha256=self.spec["coupling_config_sha256"], output=output,
            ); return [output]
        if kind == "switch_sidecar":
            role = "train" if task_id.startswith("train") else "validation"
            base = self.root / f"coupling/{role}/switch/shard_{index:04d}"
            if base.with_suffix(".json").exists():
                metadata, _ = load_switch_sidecar(base.with_suffix(".json"))
                base_metadata = load_json(
                    self.root / f"coupling/{role}/base/shard_{index:04d}.json"
                )
                if (
                    metadata.get("role") != role
                    or metadata.get("base_shard_sha256") != base_metadata["content_hash"]
                    or metadata.get("switch_calibration_sha256")
                       != load_json(self.root / "coupling/switch_calibration.json")["content_hash"]
                ):
                    raise ValueError("reused HCWDL-UJ switch sidecar lineage differs")
                return [base.with_suffix(".npz"), base.with_suffix(".json")]
            build_switch_sidecar_for_source(
                base_metadata_path=self.root / f"coupling/{role}/base/shard_{index:04d}.json",
                switch_calibration=load_json(self.root / "coupling/switch_calibration.json"),
                coupling_config_sha256=self.spec["coupling_config_sha256"],
                output_base=base,
            ); return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if kind == "coupling_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = self.root / f"coupling/{role}_manifest.json"
            if output.exists():
                value = load_json(output); validate_coupling_manifest(value, role=role)
                if value.get("rows") != int(self.spec["role_counts"][role]):
                    raise ValueError("reused HCWDL-UJ coupling-manifest coverage differs")
                return [output]
            switch = load_json(self.root / "coupling/switch_calibration.json")
            finalize_coupling_role(
                role=role, base_manifest_path=self.root / f"coupling/{role}_base_manifest.json",
                sidecar_root=self.root / "coupling",
                switch_calibration_sha256=switch["content_hash"], output=output,
            ); return [output]
        if kind == "coupling_audit":
            output = self.root / "coupling/full_role_audit.json"
            if output.exists():
                validate_coupling_audit(load_json(output)); return [output]
            audit_full_roles(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifests=self.spec["assignment_manifests"],
                coupling_manifests={r: self.root / f"coupling/{r}_manifest.json" for r in ("train", "validation")},
                data_root=self.spec["data_root"], coupling_config_sha256=self.spec["coupling_config_sha256"],
                switch_calibration=load_json(self.root / "coupling/switch_calibration.json"),
                scale_calibration=load_json(self.root / "coupling/scale_calibration.json"),
                discrete_seed=derive_seed(int(self.spec["replicate_seed"]), "hcwdl_uj/repair/shared_v1"), output=output,
            ); return [output]
        if kind == "coupling_lock":
            output = self.root / "locks/coupling_lock.json"
            audit, expected = self._validated_coupling_lineage()
            if output.exists():
                validate_coupling_lock(
                    load_json(output), campaign_spec_sha256=self.spec["content_hash"],
                    expected=expected,
                ); return [output]
            payload = build_coupling_lock(
                campaign_spec_sha256=self.spec["content_hash"], **expected,
            ); write_immutable_json(output, payload); return [output]
        if kind == "cache_miniature":
            output = self.root / "runtime/cache_miniature.json"
            reused = self._load_json_output(
                output, CACHE_MINIATURE_CONTRACT,
                expected={"campaign_spec_sha256": self.spec["content_hash"]},
            )
            if reused:
                measurement = self._load_json_output(
                    self.root / "runtime/cache_resource_measurement.json",
                    CACHE_RESOURCE_MEASUREMENT_CONTRACT,
                    expected={
                        "campaign_spec_sha256": self.spec["content_hash"],
                        "cache_miniature_sha256": load_json(output)["content_hash"],
                    },
                )
                if measurement is None:
                    raise FileExistsError("HCWDL-UJ cache miniature lacks its resource measurement")
                return reused
            write_immutable_json(output, self._cache_miniature()); return [output]
        if kind == "endpoint_lock":
            output = self.root / "locks/endpoint_equality_lock.json"
            audit, coupling_expected = self._validated_coupling_lineage()
            coupling_lock = load_json(self.root / "locks/coupling_lock.json")
            coupling_lock_hash = validate_coupling_lock(
                coupling_lock, campaign_spec_sha256=self.spec["content_hash"],
                expected=coupling_expected,
            )
            miniature = load_json(self.root / "runtime/cache_miniature.json")
            miniature_hash = validate_content_hash(
                miniature, expected_contract=CACHE_MINIATURE_CONTRACT,
                expected_schema_version=1,
            )
            expected_views = {
                f"{role}:{view}"
                for role in ("train", "validation")
                for view in ("p0", "u020", "j010", "u100", "j100")
            }
            if (
                miniature.get("campaign_spec_sha256") != self.spec["content_hash"]
                or set(miniature.get("view_sha256", {})) != expected_views
                or set(miniature.get("sample_rows", {})) != expected_views
                or set(miniature.get("sample_array_bytes_by_view", {})) != expected_views
                or any(int(value) <= 0 for value in miniature.get("sample_rows", {}).values())
                or any(int(value) <= 0 for value in miniature.get("sample_array_bytes_by_view", {}).values())
                or miniature.get("durable_repaired_dataset") is not False
                or miniature.get("matcher_callable_present") is not False
                or miniature.get("final_test_accessed") is not False
            ):
                raise ValueError("HCWDL-UJ cache miniature endpoint evidence differs")
            for name, digest in miniature["view_sha256"].items():
                require_sha256(digest, name=f"cache miniature view {name}")
            coordinate = load_json(self.root / "coordinate_table.json")
            coordinate_hash = validate_coordinate(coordinate)
            if coordinate_hash != self.spec["coordinate_sha256"]:
                raise ValueError("HCWDL-UJ endpoint coordinate lineage differs")
            repair_hash = sha256_file(
                self.repository / "src/hlt_classification/scouting/repair.py"
            )
            if (
                self.config.get("projection_sha256") != repair_hash
                or self.config.get("shell_exact_sha256") != repair_hash
            ):
                raise ValueError("HCWDL-UJ endpoint implementation drifted after campaign creation")
            shell_parity_hash = canonical_sha256({
                "public_family": "HIGHCOV_SHELL_EXACT/v1",
                "continuous_alpha": True,
            })
            expected = {
                "coupling_lock_sha256": coupling_lock_hash,
                "full_role_audit_sha256": audit["content_hash"],
                "cache_miniature_sha256": miniature_hash,
                "coordinate_sha256": coordinate_hash,
                "projection_sha256": repair_hash,
                "shell_parity_sha256": shell_parity_hash,
            }
            if output.exists():
                validate_endpoint_equality_lock(
                    load_json(output), campaign_spec_sha256=self.spec["content_hash"],
                    expected=expected,
                ); return [output]
            payload = build_endpoint_equality_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                **expected,
            ); write_immutable_json(output, payload); return [output]
        if kind == "toff_target_shard":
            manifest_path = self.root / "targets/toff_train/manifest.json"
            runtime_path = self.root / "runtime/toff_target_resource_measurement.json"
            if manifest_path.exists():
                manifest = load_json(manifest_path)
                validate_toff_target_manifest(manifest)
                if (
                    manifest.get("rows") != int(self.spec["role_counts"]["train"])
                    or manifest.get("parents", {}).get("campaign_spec_sha256")
                       != self.spec["content_hash"]
                ):
                    raise ValueError("reused HCWDL-UJ TOFF target cache lineage differs")
                measurement = self._load_json_output(
                    runtime_path, TARGET_RESOURCE_MEASUREMENT_CONTRACT,
                    expected={
                        "campaign_spec_sha256": self.spec["content_hash"],
                        "target_manifest_sha256": manifest["content_hash"],
                    },
                )
                if measurement is None:
                    raise FileExistsError("HCWDL-UJ TOFF target cache lacks resource evidence")
                return [manifest_path, runtime_path]
            started = time.monotonic()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:  # pragma: no cover - live target jobs require torch
                pass
            manifest, _ = build_toff_target_cache(
                split_manifest=self.split, selection_manifest=self.selection,
                data_root=self.spec["data_root"],
                teacher_report_path=self.spec["imported_controls"]["TOFF"]["report_path"],
                output_root=self.root / "targets/toff_train",
                producer_commit=self.producer_commit, device="cuda",
                campaign_spec_sha256=self.spec["content_hash"],
            )
            peak_gpu = 0
            try:
                import torch
                if torch.cuda.is_available():
                    peak_gpu = int(torch.cuda.max_memory_reserved())
            except ImportError:  # pragma: no cover
                pass
            try:
                import resource
                peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            except ImportError:  # pragma: no cover
                peak_rss_kib = 0
            write_immutable_json(runtime_path, with_content_hash({
                "contract": TARGET_RESOURCE_MEASUREMENT_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "target_manifest_sha256": manifest["content_hash"],
                "wall_seconds": time.monotonic() - started,
                "peak_rss_kib": peak_rss_kib,
                "peak_gpu_bytes": peak_gpu,
                "final_test_accessed": False,
            }))
            return [manifest_path, runtime_path]
        if kind == "toff_target_manifest":
            output = self.root / "targets/toff_train/manifest.json"
            validate_toff_target_manifest(load_json(output)); return [output]
        if kind == "toff_target_lock":
            output = self.root / "targets/toff_train/lock.json"
            manifest, expected = self._validated_toff_lineage()
            if output.exists():
                validate_toff_target_lock(
                    load_json(output), campaign_spec_sha256=self.spec["content_hash"],
                    expected=expected,
                ); return [output]
            payload = build_toff_target_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                **expected,
            ); write_immutable_json(output, payload); return [output]
        if kind == "graph_recipe_lock":
            output = self.root / "locks/graph_recipe_lock.json"
            audit, coupling_expected = self._validated_coupling_lineage()
            endpoint = load_json(self.root / "locks/endpoint_equality_lock.json")
            coupling_lock = load_json(self.root / "locks/coupling_lock.json")
            coupling_lock_hash = validate_coupling_lock(
                coupling_lock, campaign_spec_sha256=self.spec["content_hash"],
                expected=coupling_expected,
            )
            miniature = load_json(self.root / "runtime/cache_miniature.json")
            miniature_hash = validate_content_hash(
                miniature, expected_contract=CACHE_MINIATURE_CONTRACT,
                expected_schema_version=1,
            )
            coordinate_hash = validate_coordinate(
                load_json(self.root / "coordinate_table.json")
            )
            repair_hash = sha256_file(
                self.repository / "src/hlt_classification/scouting/repair.py"
            )
            endpoint_expected = {
                "coupling_lock_sha256": coupling_lock_hash,
                "full_role_audit_sha256": audit["content_hash"],
                "cache_miniature_sha256": miniature_hash,
                "coordinate_sha256": coordinate_hash,
                "projection_sha256": repair_hash,
                "shell_parity_sha256": canonical_sha256({
                    "public_family": "HIGHCOV_SHELL_EXACT/v1",
                    "continuous_alpha": True,
                }),
            }
            endpoint_hash = validate_endpoint_equality_lock(
                endpoint, campaign_spec_sha256=self.spec["content_hash"],
                expected=endpoint_expected,
            )
            _, toff_expected = self._validated_toff_lineage()
            toff_lock = load_json(self.root / "targets/toff_train/lock.json")
            toff_lock_hash = validate_toff_target_lock(
                toff_lock, campaign_spec_sha256=self.spec["content_hash"],
                expected=toff_expected,
            )
            plan = load_json(self.root / "command_plan.json")
            plan_hash = validate_content_hash(
                plan, expected_contract=COMMAND_PLAN_CONTRACT,
                expected_schema_version=1,
            )
            if plan_hash != self.spec["command_plan_sha256"] or plan != build_command_plan(self.spec):
                raise ValueError("HCWDL-UJ graph-lock command plan differs")
            expected = {
                "endpoint_equality_lock_sha256": endpoint_hash,
                "toff_target_lock_sha256": toff_lock_hash,
                "graph_artifact_sha256": self.spec["graph_artifact_sha256"],
                "graph_semantic_sha256": self.spec["graph_sha256"],
                "recipe_overlay_sha256": self.spec["recipe_overlay_sha256"],
                "parent_recipe_sha256": self.spec["recipe_sha256"],
                "coordinate_sha256": self.spec["coordinate_sha256"],
                "command_plan_sha256": plan_hash,
                "source_commit_sha256": canonical_sha256(self.spec["source_commit"]),
                "weaver_parity_sha256": self.spec["weaver_parity_sha256"],
            }
            if output.exists():
                validate_graph_recipe_lock(
                    load_json(output), campaign_spec_sha256=self.spec["content_hash"],
                    expected=expected,
                ); return [output]
            payload = build_graph_recipe_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                **expected,
            ); write_immutable_json(output, payload); return [output]
        if kind == "train_node":
            node_id = str(task["node_id"])
            run_homotopy_node(spec=self.spec, node_id=node_id, device="cuda")
            from .hcwdl_homotopy_runner import node_output_dir
            output = node_output_dir(self.root, node_id)
            return [
                output / "training_report.json",
                output / "hcwdl_training_report.json",
                output / "runtime.json",
            ]
        if kind == "aggregate":
            output = self.root / "reports/validation_aggregate.json"
            reused = self._load_json_output(
                output, AGGREGATE_CONTRACT,
                expected={"campaign_spec_sha256": self.spec["content_hash"], "fit_count": FIT_COUNT},
            )
            if reused: return reused
            write_immutable_json(output, build_validation_aggregate(self.spec)); return [output]
        if kind == "campaign_complete":
            output = self.root / "reports/campaign_complete.json"
            reused = self._load_json_output(
                output, CAMPAIGN_COMPLETION_CONTRACT,
                expected={"campaign_spec_sha256": self.spec["content_hash"], "fit_count": FIT_COUNT},
            )
            if reused: return reused
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            resource = self.root / "runtime/resource_measurement.json"
            write_immutable_json(output, build_campaign_completion(
                self.spec, aggregate_sha256=aggregate["content_hash"],
                resource_measurement_sha256=load_json(resource)["content_hash"] if resource.exists() else None,
            )); return [output]
        raise RuntimeError(f"unhandled HCWDL-UJ task kind {kind!r}")


__all__ = ["CACHE_MINIATURE_CONTRACT", "HomotopyWorkflow"]
