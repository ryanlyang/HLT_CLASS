"""Fail-closed execution of the bottleneck-pairing foundation overlay."""

from __future__ import annotations

from itertools import zip_longest
import hashlib
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import (
    load_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .hcwdl_fullcard_bottleneck_cache import (
    sampled_recomputation_audit,
    validate_assignment_manifest,
)
from .hcwdl_fullcard_bottleneck_contracts import (
    ASSIGNMENT_AUDIT_CONTRACT,
    ASSIGNMENT_LOCK_CONTRACT,
    COUPLING_LOCK_CONTRACT,
    DIAGNOSTIC_REPORT_CONTRACT,
    FOUNDATION_LOCK_CONTRACT,
    MATCHER_ACCEPTANCE_CONTRACT,
    SCHEMA_VERSION,
    U000_EQUIVALENCE_LOCK_CONTRACT,
    matcher_spec,
    validate_matcher_spec,
)
from .hcwdl_fullcard_bottleneck_foundation import (
    assignment_recomputer,
    build_assignment_source,
    finalize_role_assignments,
    publish_assignment_lock,
)
from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_fullcard_bottleneck_matcher import (
    FullCardinalityBottleneckMatcher,
    production_pairing_from_matrices,
    reference_pairing_from_matrices,
)
from .hcwdl_homotopy import HomotopyCoordinate, assert_particle_inputs_equal
from .hcwdl_homotopy_stream import iterate_unified_balanced_batches
from .hcwdl_unified_balanced_contracts import balanced_switch_config_payload
from .hcwdl_unified_balanced_builder import (
    build_balanced_sidecar_for_source,
    finalize_balanced_role,
)
from .hcwdl_unified_balanced_cache import (
    BalancedCouplingStore,
    validate_balanced_manifest,
)
from .hcwdl_unified_balanced_runner import _load_common
from .hcwdl_upper_builder import (
    build_coupling_source,
    calibrate_train_scales,
    finalize_base_role,
)
from .hcwdl_upper_cache import validate_base_manifest
from .labels import baseline_mask, multiclass_labels
from .particles import decode_particle_sets
from .schema import BASELINE_BRANCHES, LABEL_BRANCHES, matching_required_branches
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks
from .training import derive_seed
from .highcov_matcher import from_scouting_particles


def _task(spec: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    if task_id not in tasks:
        raise KeyError("unknown full-cardinality foundation task")
    return tasks[task_id]


def _index(task: Mapping[str, Any], array_index: int | None) -> int:
    count = int(task["array_count"])
    if count == 1:
        if array_index not in (None, 0):
            raise IndexError("non-array foundation task received an array index")
        return 0
    if array_index is None or not 0 <= array_index < count:
        raise IndexError("foundation array index differs")
    return array_index


def _hash_batch(digest: "hashlib._Hash", batch: Mapping[str, Any]) -> None:
    for identity in np.asarray(batch["identity_keys"]):
        value = str(identity).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little")); digest.update(value)
    digest.update(np.asarray(batch["labels"], dtype="<i8").tobytes())


def _extend_acceptance_candidates(
    *, source_path: str, entry_start: int, indexes: np.ndarray,
    hlt_counts: np.ndarray, offline_counts: np.ndarray,
    generic_candidates: list[tuple[str, int]],
    reference_candidates: list[tuple[str, int]],
    generic_target: int, reference_target: int,
) -> tuple[int, bool]:
    """Register a bounded deterministic real-row acceptance sample."""

    selected = np.asarray(indexes, np.int64)
    hlt = np.asarray(hlt_counts, np.int64)
    offline = np.asarray(offline_counts, np.int64)
    if hlt.ndim != 1 or offline.shape != hlt.shape:
        raise ValueError("matcher acceptance multiplicity arrays differ")
    if np.any(selected < 0) or np.any(selected >= len(hlt)):
        raise ValueError("matcher acceptance selected index differs")
    reference_seen = set(reference_candidates)
    observed = 0
    for row in selected:
        row = int(row)
        if hlt[row] < 0 or offline[row] < 0:
            raise ValueError("matcher acceptance multiplicity is negative")
        ref = (str(source_path), int(entry_start) + row)
        observed += 1
        if len(generic_candidates) < generic_target:
            generic_candidates.append(ref)
        if (
            min(int(hlt[row]), int(offline[row])) <= 9
            and ref not in reference_seen
            and len(reference_candidates) < reference_target
        ):
            reference_candidates.append(ref); reference_seen.add(ref)
        if (
            len(generic_candidates) >= generic_target
            and len(reference_candidates) >= reference_target
        ):
            return observed, True
    return observed, False
    view = batch["privileged"]
    for name in ("features", "vectors", "mask", "raw_lengths"):
        value = np.ascontiguousarray(getattr(view, name))
        digest.update(name.encode("ascii")); digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())


class FullCardinalityFoundationWorkflow:
    def __init__(
        self, spec: Mapping[str, Any], *, producer_commit: str | None = None,
        recovery_spec_sha256: str | None = None,
    ) -> None:
        validate_foundation(spec)
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])
        self.producer_commit = str(producer_commit or spec["source_commit"])
        self.recovery_spec_sha256 = recovery_spec_sha256
        self.split = load_json(spec["artifact_paths"]["split_manifest"])
        self.selection = load_json(spec["artifact_paths"]["selection_manifest"])
        self.assignment_root = self.root / "matcher/assignments"

    def _assignment_manifest(self, role: str) -> Path:
        return Path(self.spec["artifact_paths"][f"{role}_assignment_manifest"])

    def _old_assignment_manifest(self, role: str) -> Path:
        return Path(self.spec["artifact_paths"][f"old_{role}_assignment_manifest"])

    def _matcher_acceptance(self) -> dict[str, Any]:
        started = time.monotonic()
        cpu_started = time.process_time()
        matcher = FullCardinalityBottleneckMatcher()
        generic_target = 64
        reference_target = 8
        generic_candidates: list[tuple[str, int]] = []
        reference_candidates: list[tuple[str, int]] = []
        scan_rows = 0
        selection = RowSelection(
            self.selection, role="train",
            split_manifest_sha256=self.split["content_hash"],
        )
        count_branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | {
            "n_cpfcands", "n_lts", "n_npfcands",
        }

        # Candidate discovery deliberately reads scalar count/selection fields
        # only.  The previous implementation ran the exact matcher on every
        # selected row while searching for rare brute-forceable examples,
        # turning a bounded acceptance miniature into a multi-hour scan.
        scan_started = time.monotonic()
        for record in role_records(self.split, "train"):
            source = Path(self.spec["data_root"]) / record.path
            for chunk in iterate_projected_chunks(
                (source,), count_branches, data_root=self.spec["data_root"],
                role="train", step_size=16_384,
            ):
                labels = multiclass_labels(chunk.arrays)
                indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
                absolute = chunk.entry_start + indexes
                indexes = indexes[selection.mask(chunk.source_path, absolute)]
                if not len(indexes):
                    continue
                hlt_counts = np.asarray(chunk.arrays["n_scoutpfcands"], np.int64)
                offline_counts = (
                    np.asarray(chunk.arrays["n_cpfcands"], np.int64)
                    + np.asarray(chunk.arrays["n_lts"], np.int64)
                    + np.asarray(chunk.arrays["n_npfcands"], np.int64)
                )
                observed, complete = _extend_acceptance_candidates(
                    source_path=chunk.source_path, entry_start=chunk.entry_start,
                    indexes=indexes, hlt_counts=hlt_counts,
                    offline_counts=offline_counts,
                    generic_candidates=generic_candidates,
                    reference_candidates=reference_candidates,
                    generic_target=generic_target,
                    reference_target=reference_target,
                )
                scan_rows += observed
                if complete:
                    break
            if (
                len(generic_candidates) >= generic_target
                and len(reference_candidates) >= reference_target
            ):
                break
        scan_seconds = time.monotonic() - scan_started
        generic_seen = set(generic_candidates)
        candidates = generic_candidates + [
            ref for ref in reference_candidates if ref not in generic_seen
        ]
        if len(generic_candidates) < generic_target or not reference_candidates:
            raise ValueError(
                "matcher acceptance candidate discovery lacks required real rows: "
                f"generic={len(generic_candidates)} reference={len(reference_candidates)}"
            )
        print(
            "HCWDL-FULLCARD phase=matcher_acceptance_candidates "
            f"scanned={scan_rows} generic={len(generic_candidates)} "
            f"reference={len(reference_candidates)} seconds={scan_seconds:.3f}",
            flush=True,
        )

        branches = set(BASELINE_BRANCHES) | set(LABEL_BRANCHES) | set(
            matching_required_branches()
        )
        by_source: dict[str, list[int]] = {}
        for source_path, entry in candidates:
            by_source.setdefault(source_path, []).append(entry)
        loaded: dict[tuple[str, int], Mapping[str, Any]] = {}
        import uproot
        data_root = Path(self.spec["data_root"]).expanduser().resolve()
        for source_path, entries in by_source.items():
            source = (data_root / source_path).resolve()
            try:
                source.relative_to(data_root)
            except ValueError as error:
                raise ValueError("matcher acceptance source escapes data root") from error
            with uproot.open(source) as handle:
                tree = handle["tree"]
                missing = sorted(branches - set(tree.keys()))
                if missing:
                    raise KeyError(f"matcher acceptance source lacks branches: {missing}")
                for entry in entries:
                    arrays = tree.arrays(
                        sorted(branches), entry_start=entry, entry_stop=entry + 1,
                        library="ak", how=dict,
                    )
                    if len(next(iter(arrays.values()))) != 1:
                        raise ValueError("matcher acceptance targeted read differs")
                    loaded[(source_path, entry)] = arrays

        checked = 0
        reference_checked = 0
        max_hlt = 0
        max_offline = 0
        matching_started = time.monotonic()
        for source_path, entry in candidates:
            arrays = loaded[(source_path, entry)]
            hraw, oraw, _ = decode_particle_sets(arrays, 0)
            hlt = from_scouting_particles(hraw, offline=False)
            offline = from_scouting_particles(oraw, offline=True)
            result = matcher.match(hlt, offline)
            max_hlt = max(max_hlt, len(hlt.p4)); max_offline = max(max_offline, len(offline.p4))
            if min(len(hlt.p4), len(offline.p4)) <= 9:
                from .highcov_features import edge_matrices
                from .hcwdl_fullcard_bottleneck_matcher import (
                    canonical_qabs_log_pt_response, canonical_qdr,
                )
                matrices = edge_matrices(hlt, offline)
                native = (
                    np.arange(len(offline.p4)) if offline.native_index is None
                    else offline.native_index
                )
                kwargs = dict(
                    qdr=canonical_qdr(matrices.dr),
                    qresponse=canonical_qabs_log_pt_response(matrices.log_pt),
                    hlt_category=hlt.category, offline_category=offline.category,
                    hlt_charge=hlt.charge, offline_charge=offline.charge,
                    native_offline_index=native,
                )
                expected = reference_pairing_from_matrices(**kwargs)
                actual = production_pairing_from_matrices(**kwargs)
                if not np.array_equal(expected, actual):
                    raise ValueError("production/reference real-row pairing differs")
                reference_checked += 1
            if result.selected_count != min(len(hlt.p4), len(offline.p4)):
                raise ValueError("matcher acceptance cardinality differs")
            checked += 1
            if checked % 8 == 0 or checked == len(candidates):
                print(
                    "HCWDL-FULLCARD phase=matcher_acceptance_match "
                    f"checked={checked}/{len(candidates)} reference={reference_checked}",
                    flush=True,
                )
        matching_seconds = time.monotonic() - matching_started
        if checked < generic_target or reference_checked < 1:
            raise ValueError("matcher acceptance did not inspect real low-multiplicity rows")
        try:
            import resource
            peak_rss_platform_units = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
        except (ImportError, AttributeError):
            peak_rss_platform_units = -1
        report = with_content_hash({
            "contract": MATCHER_ACCEPTANCE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "foundation_spec_sha256": self.spec["content_hash"],
            "matcher_spec_sha256": self.spec["parents"]["matcher_spec"],
            "authenticated_real_rows": checked,
            "reference_checked_rows": reference_checked,
            "production_reference_exact": True,
            "maximum_hlt_multiplicity": max_hlt,
            "maximum_offline_multiplicity": max_offline,
            "candidate_scan_rows": scan_rows,
            "candidate_scan_seconds": scan_seconds,
            "matching_seconds": matching_seconds,
            "generic_candidate_target": generic_target,
            "reference_candidate_target": reference_target,
            "runtime_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_platform_units": peak_rss_platform_units,
            "allocated_cpus": int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "1")),
            "resource_measurement_is_real_worker_observation": True,
            "durable_dense_pair_matrices": False,
            "scientific_metrics_control_execution": False,
            "final_test_accessed": False,
        })
        write_immutable_json(self.root / "locks/matcher_acceptance.json", report)
        return report

    def _u000_equivalence(self) -> dict[str, Any]:
        old = load_json(self.spec["artifact_paths"]["old_foundation_spec"])
        _, _, _, old_selections, old_assignments, old_balanced = _load_common(old)
        _, split_hash, selection_hash, selections, assignments, balanced = _load_common(
            self.spec
        )
        repair_seed = derive_seed(
            int(self.spec["replicate_seed"]), "tri60/repair/shared_v1",
        )
        role_hashes: dict[str, str] = {}
        role_rows: dict[str, int] = {}
        stream_workers = min(
            36, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) // 2),
        )
        coordinate = HomotopyCoordinate(0, 1, 0, 1)
        for role in ("train", "validation"):
            old_stream = iterate_unified_balanced_batches(
                self.split, data_root=self.spec["data_root"], role=role,
                assignment_store=old_assignments[role],
                coupling_store=old_balanced[role], row_selection=old_selections[role],
                coordinate=coordinate, repair_seed=repair_seed, batch_size=1024,
                workers=stream_workers, include_training_metadata=True,
            )
            new_stream = iterate_unified_balanced_batches(
                self.split, data_root=self.spec["data_root"], role=role,
                assignment_store=assignments[role], coupling_store=balanced[role],
                row_selection=selections[role], coordinate=coordinate,
                repair_seed=repair_seed, batch_size=1024,
                workers=stream_workers,
                include_training_metadata=True,
            )
            digest = hashlib.sha256(); observed = 0
            for old_batch, new_batch in zip_longest(old_stream, new_stream):
                if old_batch is None or new_batch is None:
                    raise ValueError("U000 old/new stream length differs")
                if not np.array_equal(old_batch["identity_keys"], new_batch["identity_keys"]):
                    raise ValueError("U000 old/new identities differ")
                if not np.array_equal(old_batch["labels"], new_batch["labels"]):
                    raise ValueError("U000 old/new labels differ")
                assert_particle_inputs_equal(
                    old_batch["privileged"], new_batch["privileged"], endpoint="u000",
                )
                _hash_batch(digest, new_batch)
                observed += len(new_batch["labels"])
            if observed != int(self.spec["role_counts"][role]):
                raise ValueError("U000 equivalence role coverage differs")
            role_hashes[role] = digest.hexdigest(); role_rows[role] = observed
        source = load_json(self.spec["artifact_paths"]["source_lock"])
        report = with_content_hash({
            "contract": U000_EQUIVALENCE_LOCK_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "foundation_spec_sha256": self.spec["content_hash"],
            "parents": {
                "old_foundation": self.spec["parents"]["old_foundation"],
                "new_assignment_lock": load_json(self.root / "locks/assignment.json")["content_hash"],
                "source_u000_report": source["u000"]["report_sha256"],
                "source_u000_checkpoint": source["u000"]["selected_checkpoint_sha256"],
                "source_u000_probability_lock": source["u000_probability"]["lock_sha256"],
                "split_manifest": split_hash,
                "selection_manifest": selection_hash,
            },
            "role_rows": role_rows,
            "complete_p0_stream_sha256": role_hashes,
            "identical_p0_tensors_all_rows": True,
            "identical_labels_and_identity_order": True,
            "workers_per_stream": stream_workers,
            "ordered_parallel_stream_comparison": True,
            "u000_checkpoint_reused_read_only": True,
            "u000_probability_bank_reused_read_only": True,
            "u000_retrained": False,
            "runtime_sibling_worktree_imports": False,
            "final_test_accessed": False,
        })
        write_immutable_json(self.spec["artifact_paths"]["u000_equivalence_lock"], report)
        return report

    def run(self, task_id: str, *, array_index: int | None = None) -> list[Path]:
        task = _task(self.spec, task_id)
        index = _index(task, array_index)
        kind = str(task["kind"])
        if kind == "authenticate":
            validate_foundation(self.spec, executable=True)
            required = int(self.spec["minimum_free_disk_bytes"]) + int(
                self.spec["projected_durable_bytes"]
            )
            if shutil.disk_usage(self.root).free < required:
                raise OSError("foundation free disk cannot preserve reserve after projection")
            output = self.root / "locks/authenticated.json"
            write_immutable_json(output, with_content_hash({
                "contract": "HCWDL_FULLCARD_BOTTLENECK_AUTHENTICATION/v1",
                "schema_version": 1,
                "foundation_spec_sha256": self.spec["content_hash"],
                "source_commit": self.producer_commit,
                "ordinary_final_test_capability": False,
                "minimum_free_disk_bytes": int(self.spec["minimum_free_disk_bytes"]),
                "projected_durable_bytes": int(self.spec["projected_durable_bytes"]),
                "required_free_bytes_at_authentication": required,
                "final_test_accessed": False,
            }))
            return [output]
        if kind == "matcher_acceptance":
            self._matcher_acceptance()
            return [self.root / "locks/matcher_acceptance.json"]
        if kind == "assignment":
            role = "train" if task_id == "assign_train" else "validation"
            return list(build_assignment_source(
                split_manifest=self.split, selection_manifest=self.selection,
                data_root=self.spec["data_root"], assignment_root=self.assignment_root,
                old_assignment_manifest=self._old_assignment_manifest(role),
                role=role, source_index=index,
                matcher_spec_sha256=self.spec["parents"]["matcher_spec"],
            ))
        if kind == "assignment_manifest":
            outputs: list[Path] = []
            for role in ("train", "validation"):
                manifest_path = self._assignment_manifest(role)
                diagnostic_path = self.root / f"matcher/{role}_diagnostics.json"
                finalize_role_assignments(
                    split_manifest=self.split, selection_manifest=self.selection,
                    assignment_root=self.assignment_root,
                    old_assignment_manifest=self._old_assignment_manifest(role),
                    matcher_spec_sha256=self.spec["parents"]["matcher_spec"],
                    role=role, output=manifest_path,
                    diagnostic_output=diagnostic_path,
                )
                audit = sampled_recomputation_audit(
                    manifest_path,
                    recompute=assignment_recomputer(
                        split_manifest=self.split, data_root=self.spec["data_root"], role=role,
                    ),
                    sample_size=min(64, int(self.spec["role_counts"][role])),
                    seed=int(self.spec["replicate_seed"]),
                )
                audit_path = self.root / f"matcher/{role}_recomputation_audit.json"
                write_immutable_json(audit_path, audit)
                outputs.extend((manifest_path, diagnostic_path, audit_path))
            return outputs
        if kind == "assignment_lock":
            manifests = {}; audits = {}; diagnostics = {}
            for role in ("train", "validation"):
                old_hash = load_json(self._old_assignment_manifest(role))["content_hash"]
                parents = {
                    "split_manifest_sha256": self.spec["parents"]["split_manifest"],
                    "row_selection_sha256": self.spec["parents"]["selection_manifest"],
                    "matcher_spec_sha256": self.spec["parents"]["matcher_spec"],
                    "old_assignment_manifest_sha256": old_hash,
                }
                manifests[role] = validate_assignment_manifest(
                    self._assignment_manifest(role), expected_role=role,
                    expected_mapped_jets=int(self.spec["role_counts"][role]),
                    expected_parents=parents,
                )
                audits[role] = load_json(self.root / f"matcher/{role}_recomputation_audit.json")
                validate_content_hash(
                    audits[role], expected_contract=ASSIGNMENT_AUDIT_CONTRACT,
                    expected_schema_version=SCHEMA_VERSION,
                )
                diagnostics[role] = load_json(self.root / f"matcher/{role}_diagnostics.json")
                validate_content_hash(
                    diagnostics[role], expected_contract=DIAGNOSTIC_REPORT_CONTRACT,
                    expected_schema_version=SCHEMA_VERSION,
                )
            output = self.root / "locks/assignment.json"
            publish_assignment_lock(
                output, foundation_spec_sha256=self.spec["content_hash"],
                role_manifests=manifests, role_audits=audits,
                role_diagnostics=diagnostics,
                matcher_spec_sha256=self.spec["parents"]["matcher_spec"],
            )
            return [output]
        if kind == "scale_calibration":
            output = self.root / "coupling/scale_calibration.json"
            calibrate_train_scales(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifest=self._assignment_manifest("train"),
                data_root=self.spec["data_root"],
                coupling_config=load_json(self.root / "coupling/config.json"),
                output=output,
            )
            return [output]
        if kind == "coupling_base":
            role = "train" if task_id == "train_base" else "validation"
            output = self.root / f"coupling/{role}/base/shard_{index:04d}"
            assignment_lock = load_json(self.root / "locks/assignment.json")
            build_coupling_source(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifest=self._assignment_manifest(role),
                data_root=self.spec["data_root"], role=role, source_index=index,
                scale_calibration=load_json(self.root / "coupling/scale_calibration.json"),
                coupling_config_sha256=load_json(self.root / "coupling/config.json")["content_hash"],
                assignment_lock_sha256=assignment_lock["content_hash"],
                qualification_lock_sha256=assignment_lock["content_hash"],
                output_base=output, producer_commit=self.producer_commit,
                recovery_spec_sha256=self.recovery_spec_sha256,
            )
            return [output.with_suffix(".npz"), output.with_suffix(".json")]
        if kind == "base_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = Path(self.spec["artifact_paths"][f"{role}_base_manifest"])
            finalize_base_role(
                split_manifest=self.split, selection_manifest=self.selection,
                role=role, base_root=self.root / "coupling", output=output,
                parents={
                    "foundation_spec_sha256": self.spec["content_hash"],
                    "coupling_config_sha256": load_json(self.root / "coupling/config.json")["content_hash"],
                    "scale_calibration_sha256": load_json(self.root / "coupling/scale_calibration.json")["content_hash"],
                },
            )
            return [output]
        if kind == "coupling_lock":
            base = {
                role: load_json(self.spec["artifact_paths"][f"{role}_base_manifest"])
                for role in ("train", "validation")
            }
            for role in base:
                validate_base_manifest(base[role], role=role)
            payload = with_content_hash({
                "contract": COUPLING_LOCK_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "foundation_spec_sha256": self.spec["content_hash"],
                "assignment_lock_sha256": load_json(self.root / "locks/assignment.json")["content_hash"],
                "coupling_config_sha256": load_json(self.root / "coupling/config.json")["content_hash"],
                "scale_calibration_sha256": load_json(self.root / "coupling/scale_calibration.json")["content_hash"],
                "train_base_manifest_sha256": base["train"]["content_hash"],
                "validation_base_manifest_sha256": base["validation"]["content_hash"],
                "complete_train_validation_coverage": True,
                "final_test_accessed": False,
            })
            output = self.root / "locks/coupling.json"
            write_immutable_json(output, payload); return [output]
        if kind == "balanced_config":
            coupling = load_json(self.root / "locks/coupling.json")
            validate_content_hash(
                coupling, expected_contract=COUPLING_LOCK_CONTRACT,
                expected_schema_version=SCHEMA_VERSION,
            )
            output = self.root / "balanced/config.json"
            write_immutable_json(
                output,
                balanced_switch_config_payload(
                    base_coupling_lock_sha256=coupling["content_hash"]
                ),
            )
            return [output]
        if kind == "balanced_sidecar":
            role = "train" if task_id.startswith("train") else "validation"
            output = self.root / f"balanced/{role}/shard_{index:04d}"
            build_balanced_sidecar_for_source(
                split_manifest=self.split, selection_manifest=self.selection,
                assignment_manifest=self._assignment_manifest(role),
                data_root=self.spec["data_root"], role=role, source_index=index,
                base_metadata_path=self.root / f"coupling/{role}/base/shard_{index:04d}.json",
                switch_config_sha256=load_json(self.root / "balanced/config.json")["content_hash"],
                output_base=output, producer_commit=self.producer_commit,
            )
            return [output.with_suffix(".npz"), output.with_suffix(".json")]
        if kind == "balanced_manifest":
            role = "train" if task_id.startswith("train") else "validation"
            output = Path(self.spec["artifact_paths"][f"{role}_balanced_manifest"])
            finalize_balanced_role(
                role=role,
                base_manifest_path=self.spec["artifact_paths"][f"{role}_base_manifest"],
                sidecar_root=self.root / "balanced", output=output,
                switch_config_sha256=load_json(self.root / "balanced/config.json")["content_hash"],
            )
            return [output]
        if kind == "u000_equivalence":
            self._u000_equivalence()
            return [Path(self.spec["artifact_paths"]["u000_equivalence_lock"])]
        if kind == "foundation_lock":
            balanced = {
                role: load_json(self.spec["artifact_paths"][f"{role}_balanced_manifest"])
                for role in ("train", "validation")
            }
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            equivalence = load_json(self.spec["artifact_paths"]["u000_equivalence_lock"])
            durable_files = tuple(
                path for path in self.root.rglob("*") if path.is_file()
            )
            forbidden_names = tuple(
                str(path.relative_to(self.root)) for path in durable_files
                if "resume" in path.name.lower() or "optimizer" in path.name.lower()
            )
            if forbidden_names:
                raise PermissionError(
                    "full-cardinality foundation contains forbidden durable state: "
                    + ", ".join(forbidden_names)
                )
            durable_bytes = sum(path.stat().st_size for path in durable_files)
            if durable_bytes > int(self.spec["projected_durable_bytes"]):
                raise OSError("foundation durable bytes exceed the authorized projection")
            if shutil.disk_usage(self.root).free < int(self.spec["minimum_free_disk_bytes"]):
                raise OSError("foundation completion cannot preserve free-space reserve")
            payload = with_content_hash({
                "contract": FOUNDATION_LOCK_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "foundation_spec_sha256": self.spec["content_hash"],
                "parents": {
                    "assignment_lock": load_json(self.root / "locks/assignment.json")["content_hash"],
                    "coupling_lock": load_json(self.root / "locks/coupling.json")["content_hash"],
                    "train_balanced_manifest": validate_balanced_manifest(balanced["train"], role="train"),
                    "validation_balanced_manifest": validate_balanced_manifest(balanced["validation"], role="validation"),
                    "u000_equivalence_lock": validate_content_hash(
                        equivalence, expected_contract=U000_EQUIVALENCE_LOCK_CONTRACT,
                        expected_schema_version=SCHEMA_VERSION,
                    ),
                    "u000_report": source["u000"]["report_sha256"],
                    "u000_checkpoint": source["u000"]["selected_checkpoint_sha256"],
                    "u000_probability_lock": source["u000_probability"]["lock_sha256"],
                    "matcher_acceptance": load_json(self.root / "locks/matcher_acceptance.json")["content_hash"],
                },
                "role_counts": self.spec["role_counts"],
                "u000_reused_read_only": True,
                "assignment_dependent_descendants_rebuilt": True,
                "pairing_provenance": "validity_only_not_correspondence_confidence",
                "durable_file_count_before_lock": len(durable_files),
                "durable_bytes_before_lock": durable_bytes,
                "projected_durable_bytes": int(self.spec["projected_durable_bytes"]),
                "minimum_free_disk_bytes_preserved": int(
                    self.spec["minimum_free_disk_bytes"]
                ),
                "durable_dense_pair_matrices": False,
                "rolling_resume_persisted": False,
                "optimizer_state_persisted": False,
                "ordinary_final_test_capability": False,
                "final_test_accessed": False,
            })
            output = Path(self.spec["artifact_paths"]["foundation_lock"])
            write_immutable_json(output, payload); return [output]
        raise RuntimeError(f"unhandled foundation task kind: {kind}")


__all__ = ["FullCardinalityFoundationWorkflow"]
