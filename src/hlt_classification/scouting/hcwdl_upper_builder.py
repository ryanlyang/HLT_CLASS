"""Production coupling calibration, shard construction, finalization, and audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import os
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, validate_content_hash,
    write_immutable_json,
)

from .hcwdl_homotopy import (
    HomotopyCoordinate, assert_particle_inputs_equal, build_homotopy_inputs,
    build_p0_inputs, build_partition_from_arrays,
)
from .inputs import build_hlt_inputs
from .hcwdl_upper_cache import (
    ResidualCouplingStore, build_coupling_audit, build_coupling_lock,
    load_base_shard, publish_base_manifest, publish_base_shard,
    publish_coupling_manifest, publish_switch_sidecar,
)
from .hcwdl_upper_coupling import (
    ResidualEdit, ScaleAccumulator, assign_edit_masses, attach_switches,
    build_switch_calibration, couple_partition, edit_is_active, endpoint_cost,
    validate_scale_calibration, validate_switch_calibration,
)
from .highcov_cache import DenseAssignmentStore
from .repair import (
    FULL_VALIDITY_GROUPS, HIGHCOV_SHELL_EXACT_FAMILY, build_alpha_repaired_inputs,
    full_endpoint_required_branches, project_offline_endpoint_records,
)
from .schema import HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES
from .selective_assignment import RowSelection
from .splits import role_records
from .streaming import iterate_projected_chunks


def coupling_branch_allowlist() -> tuple[str, ...]:
    """Exact label-free branch set for structural coupling and endpoint proof."""

    branches = set(full_endpoint_required_branches())
    branches.update(spec.branch for spec in HLT_FEATURE_SPECS)
    branches.update(HLT_VECTOR_BRANCHES)
    branches.update(("n_scoutpfcands", "n_cpfcands", "n_lts", "n_npfcands"))
    return tuple(sorted(branches))


def branch_allowlist_sha256() -> str:
    return canonical_sha256(list(coupling_branch_allowlist()))


def _slice(arrays: Mapping[str, object], indexes: np.ndarray) -> dict[str, object]:
    return {name: value[indexes] for name, value in arrays.items()}


def _selected_source_chunks(
    *, split_manifest: Mapping[str, Any], selection: RowSelection,
    data_root: str | Path, role: str, source_index: int, step_size: int,
):
    records = role_records(split_manifest, role)
    if role not in {"train", "validation"} or not 0 <= source_index < len(records):
        raise ValueError("HCWDL-UJ source role/index differs")
    record = records[source_index]
    for chunk in iterate_projected_chunks(
        (Path(data_root) / record.path,), coupling_branch_allowlist(),
        data_root=data_root, role=role, step_size=step_size,
    ):
        entries = np.arange(chunk.entry_start, chunk.entry_stop, dtype=np.int64)
        keep = selection.mask(chunk.source_path, entries)
        indexes = np.flatnonzero(keep)
        if len(indexes):
            yield record, chunk.source_path, entries[indexes], _slice(chunk.arrays, indexes)


def _calibrate_source_worker(
    arguments: tuple[
        Mapping[str, Any], Mapping[str, Any], str, str, int, int,
    ],
) -> tuple[int, str, int, str, ScaleAccumulator]:
    """Scan one source in an isolated process; return only integer reducers."""

    split_manifest, selection_manifest, assignment_manifest, data_root, source_index, step_size = arguments
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role="train", split_manifest_sha256=split_hash,
    )
    store = DenseAssignmentStore(assignment_manifest)
    accumulator = ScaleAccumulator(); observed = 0; identity_digest = hashlib.sha256()
    source_path = role_records(split_manifest, "train")[source_index].path
    for _, actual_source, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection, data_root=data_root,
        role="train", source_index=source_index, step_size=step_size,
    ):
        if actual_source != source_path:
            raise ValueError("HCWDL-UJ calibration source identity differs")
        assignment, _ = store.join(source_path, entries)
        for row, entry in enumerate(entries):
            identity = f"{source_path}::tree::{int(entry)}".encode("utf-8")
            identity_digest.update(len(identity).to_bytes(4, "little")); identity_digest.update(identity)
            accumulator.update_partition(build_partition_from_arrays(
                arrays, row=row, assignment=assignment[row],
            ))
        observed += len(entries)
    expected = selection.source_rows(source_path)
    if expected < 0:
        expected = role_records(split_manifest, "train")[source_index].mapped_entries
    if observed != expected:
        raise ValueError("HCWDL-UJ per-source scale-calibration coverage differs")
    return source_index, source_path, observed, identity_digest.hexdigest(), accumulator


def _merge_accumulator(destination: ScaleAccumulator, source: ScaleAccumulator) -> None:
    maximum = np.iinfo(np.uint64).max
    for target, addition in (
        (destination.delta_r, source.delta_r),
        (destination.log_pt, source.log_pt),
        (destination.log_energy, source.log_energy),
        *((destination.fields[channel], source.fields[channel]) for channel in destination.fields),
    ):
        if np.any(target > maximum - addition):
            raise OverflowError("HCWDL-UJ calibration histogram overflow")
        target += addition
    destination.edges += source.edges
    destination.floor_pt += source.floor_pt
    destination.floor_energy += source.floor_energy


def calibrate_train_scales(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifest: str | Path, data_root: str | Path,
    coupling_config: Mapping[str, Any], output: str | Path,
    step_size: int = 4096, workers: int | None = None,
) -> dict[str, Any]:
    """Scan every selected train residual Cartesian edge exactly once."""

    config_hash = validate_content_hash(
        coupling_config, expected_contract="HCWDL_RESIDUAL_SHELL_COUPLING_CONFIG/v1",
        expected_schema_version=1,
    )
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(selection_manifest, role="train", split_manifest_sha256=split_hash)
    records = role_records(split_manifest, "train")
    requested_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) if workers is None else int(workers)
    worker_count = min(len(records), max(1, requested_workers))
    arguments = [(
        split_manifest, selection_manifest, str(assignment_manifest),
        str(data_root), source_index, step_size,
    ) for source_index in range(len(records))]
    if worker_count == 1:
        results = list(map(_calibrate_source_worker, arguments))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(_calibrate_source_worker, arguments))
    accumulator = ScaleAccumulator(); observed = 0
    source_rows: dict[str, int] = {}; source_identity_hashes = []
    for expected_index, result in enumerate(results):
        source_index, source_path, rows, identity_hash, source_accumulator = result
        if source_index != expected_index or source_path != records[expected_index].path:
            raise ValueError("HCWDL-UJ calibration reduction order differs")
        _merge_accumulator(accumulator, source_accumulator)
        observed += rows; source_rows[source_path] = rows
        source_identity_hashes.append({
            "source_path": source_path, "rows": rows,
            "identity_sha256": identity_hash,
        })
    if observed != selection.rows:
        raise ValueError("HCWDL-UJ train scale calibration coverage differs")
    payload = accumulator.payload(
        coupling_config_sha256=config_hash,
        train_identity_sha256=canonical_sha256({
            "method": "canonical_source_order_framed_identity_hashes_v1",
            "sources": source_identity_hashes,
        }),
    )
    payload["split_manifest_sha256"] = split_hash
    payload["selection_manifest_sha256"] = selection.manifest_sha256
    payload["source_rows"] = dict(sorted(source_rows.items()))
    payload["source_identity_hashes"] = source_identity_hashes
    payload["worker_count"] = worker_count
    payload["reduction"] = "canonical_source_order_exact_uint64_addition_v1"
    # Adding lineage requires recomputing content hash.
    from hlt_classification.data.cache_contracts import with_content_hash
    payload = with_content_hash(payload)
    write_immutable_json(output, payload); return payload


def build_coupling_source(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifest: str | Path, data_root: str | Path, role: str,
    source_index: int, scale_calibration: Mapping[str, Any],
    coupling_config_sha256: str, assignment_lock_sha256: str,
    qualification_lock_sha256: str, output_base: str | Path,
    producer_commit: str, recovery_spec_sha256: str | None = None,
    step_size: int = 4096,
) -> tuple[Path, Path]:
    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UJ coupling role differs")
    validate_content_hash(
        scale_calibration, expected_contract="HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1",
        expected_schema_version=1,
    )
    if scale_calibration.get("coupling_config_sha256") != coupling_config_sha256:
        raise ValueError("coupling source scale/config lineage differs")
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(selection_manifest, role=role, split_manifest_sha256=split_hash)
    store = DenseAssignmentStore(assignment_manifest)
    records = role_records(split_manifest, role); record = records[source_index]
    entries_out: list[int] = []; edits_out: list[tuple[ResidualEdit, ...]] = []
    for _, source_path, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection, data_root=data_root,
        role=role, source_index=source_index, step_size=step_size,
    ):
        assignment, _ = store.join(source_path, entries)
        for row, entry in enumerate(entries):
            partition = build_partition_from_arrays(
                arrays, row=row, assignment=assignment[row],
            )
            edits = assign_edit_masses(
                couple_partition(partition, scale_calibration), partition,
            )
            entries_out.append(int(entry)); edits_out.append(edits)
    expected = selection.source_rows(record.path)
    if expected < 0:
        expected = record.mapped_entries
    if len(entries_out) != expected:
        raise ValueError("HCWDL-UJ coupling source coverage differs")
    parents = {
        "split_manifest_sha256": split_hash,
        "row_selection_sha256": selection.manifest_sha256,
        "assignment_manifest_sha256": store.manifest["content_hash"],
        "assignment_lock_sha256": assignment_lock_sha256,
        "qualification_lock_sha256": qualification_lock_sha256,
        "coupling_config_sha256": coupling_config_sha256,
        "scale_calibration_sha256": str(scale_calibration["content_hash"]),
        "source_file_sha256": record.sha256,
        "branch_allowlist_sha256": branch_allowlist_sha256(),
    }
    if recovery_spec_sha256 is not None:
        parents["recovery_spec_sha256"] = require_sha256(
            recovery_spec_sha256, name="recovery specification",
        )
    return publish_base_shard(
        output_base, role=role, source_path=record.path, entries=entries_out,
        edit_rows=edits_out, parents=parents, producer_commit=producer_commit,
    )


def finalize_base_role(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    role: str, base_root: str | Path, output: str | Path,
    parents: Mapping[str, str],
) -> dict[str, Any]:
    selection = RowSelection(
        selection_manifest, role=role,
        split_manifest_sha256=str(split_manifest["content_hash"]),
    )
    records = role_records(split_manifest, role)
    paths = [
        Path(base_root) / role / "base" / f"shard_{index:04d}.json"
        for index in range(len(records))
    ]
    return publish_base_manifest(
        output, role=role, shard_metadata_paths=paths,
        expected_sources=[row.path for row in records], expected_rows=selection.rows,
        parents=parents,
    )


def iter_base_edits(base_manifest: Mapping[str, Any]) -> Iterable[ResidualEdit]:
    for record in base_manifest["shards"]:
        _, arrays = load_base_shard(record["metadata_path"])
        for index in range(len(arrays["edit_kind"])):
            yield ResidualEdit(
                int(arrays["edit_kind"][index]),
                int(arrays["source_native_offline_index"][index]),
                int(arrays["target_hlt_slot"][index]), int(arrays["target_kind"][index]),
                int(arrays["target_native_offline_index"][index]),
                int(arrays["cost_q"][index]), int(arrays["mass_q"][index]),
            )


def freeze_switch_calibration(
    *, train_base_manifest: Mapping[str, Any], coupling_config_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    validate_content_hash(
        train_base_manifest, expected_contract="HCWDL_RESIDUAL_SHELL_BASE_MANIFEST/v1",
        expected_schema_version=1,
    )
    payload = build_switch_calibration(
        iter_base_edits(train_base_manifest),
        coupling_config_sha256=coupling_config_sha256,
        train_base_manifest_sha256=str(train_base_manifest["content_hash"]),
    )
    write_immutable_json(output, payload); return payload


def build_switch_sidecar_for_source(
    *, base_metadata_path: str | Path, switch_calibration: Mapping[str, Any],
    coupling_config_sha256: str, output_base: str | Path,
) -> tuple[Path, Path]:
    validate_content_hash(
        switch_calibration, expected_contract="HCWDL_RESIDUAL_SHELL_SWITCH_CALIBRATION/v1",
        expected_schema_version=1,
    )
    metadata, arrays = load_base_shard(base_metadata_path)
    switches: list[int] = []
    for row, entry in enumerate(arrays["entries"]):
        start, stop = int(arrays["row_offsets"][row]), int(arrays["row_offsets"][row + 1])
        identity = f"{metadata['source_path']}::tree::{int(entry)}"
        edits = [ResidualEdit(
            int(arrays["edit_kind"][i]), int(arrays["source_native_offline_index"][i]),
            int(arrays["target_hlt_slot"][i]), int(arrays["target_kind"][i]),
            int(arrays["target_native_offline_index"][i]), int(arrays["cost_q"][i]),
            int(arrays["mass_q"][i]),
        ) for i in range(start, stop)]
        switches.extend(edit.switch_u16 for edit in attach_switches(
            edits, identity_key=identity, coupling_config_sha256=coupling_config_sha256,
            calibration=switch_calibration,
        ))
    return publish_switch_sidecar(
        output_base, base_metadata_path=base_metadata_path,
        switch_u16=np.asarray(switches, dtype="<u2"),
        switch_calibration_sha256=str(switch_calibration["content_hash"]),
    )


def finalize_coupling_role(
    *, role: str, base_manifest_path: str | Path, sidecar_root: str | Path,
    switch_calibration_sha256: str, output: str | Path,
) -> dict[str, Any]:
    base = load_json(base_manifest_path)
    paths = [
        Path(sidecar_root) / role / "switch" / f"shard_{index:04d}.json"
        for index in range(len(base["shards"]))
    ]
    return publish_coupling_manifest(
        output, role=role, base_manifest_path=base_manifest_path,
        switch_sidecar_paths=paths,
        switch_calibration_sha256=switch_calibration_sha256,
    )


def _update_digest(digest: "hashlib._Hash", *arrays: np.ndarray) -> None:
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode()); digest.update(value.tobytes())


def _row_endpoint_hash(*views: object, row: int) -> str:
    digest = hashlib.sha256()
    for view in views:
        _update_digest(
            digest, view.features[row], view.vectors[row], view.mask[row],
            np.asarray([view.raw_lengths[row]], dtype="<i4"),
        )
    return digest.hexdigest()


def _empty_transition_summary() -> dict[str, Any]:
    return {
        "rows": 0, "active_tokens_sum": 0, "active_tokens_min": 200,
        "active_tokens_max": 0, "edit_count": 0, "switched_edit_count": 0,
        "mass_q": 0, "switched_mass_q": 0,
        "absolute_pt_change_micro": 0, "switched_absolute_pt_change_micro": 0,
        "absolute_energy_change_micro": 0,
        "switched_absolute_energy_change_micro": 0,
        "category_change_count": 0, "switched_category_change_count": 0,
        "track_applicability_change_count": 0,
        "switched_track_applicability_change_count": 0,
        "validity_group_change_count": 0,
        "switched_validity_group_change_count": 0,
        # Exact all-row per-jet distributions.  Counts use one bin per
        # possible visible token/edit count; fractions use integer percent
        # bins 0..100 with half-up rounding.
        "per_jet_distributions": {
            "active_tokens": [0] * 201,
            "edit_count": [0] * 201,
            "switched_edit_count": [0] * 201,
            "switched_mass_fraction_percent": [0] * 101,
            "switched_pt_change_fraction_percent": [0] * 101,
            "switched_energy_change_fraction_percent": [0] * 101,
            "switched_category_change_fraction_percent": [0] * 101,
            "switched_track_change_fraction_percent": [0] * 101,
            "switched_validity_change_fraction_percent": [0] * 101,
        },
    }


def _fraction_percent_bin(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("HCWDL-UJ transition fraction differs")
    if denominator == 0:
        return 0
    return min(100, (200 * numerator + denominator) // (2 * denominator))


def _category(record: object | None) -> int:
    if record is None:
        return -2  # private dummy state, distinct from real unclassified -1
    flags = record.raw_features[2:7]
    if (
        bool(record.validity[2:7].all())
        and bool(np.all((flags == 0) | (flags == 1)))
        and float(flags.sum()) == 1.0
    ):
        return int(np.argmax(flags))
    return -1


def _track_state(record: object | None) -> int:
    if record is None:
        return -2
    charge = None
    if bool(record.validity[1]) and float(record.raw_features[1]) in {-1.0, 0.0, 1.0}:
        charge = int(record.raw_features[1])
    category = _category(record)
    if category < 0 or charge is None:
        return -1
    return 1 if category < 3 else 0


def _transition_attributes(partition: object, edit: ResidualEdit) -> tuple[int, ...]:
    source = next(
        (row for row in partition.source_only if row.native_index == edit.source_native_index),
        None,
    )
    target = next(
        (row for row in partition.target_only if row.hlt_slot == edit.target_hlt_slot),
        None,
    )
    source_pt = 0.0 if source is None else float(np.hypot(source.p4[0], source.p4[1]))
    target_pt = 0.0 if target is None else float(np.hypot(target.p4[0], target.p4[1]))
    source_energy = 0.0 if source is None else float(source.p4[3])
    target_energy = 0.0 if target is None else float(target.p4[3])
    validity = 0
    if source is not None and target is not None:
        validity = sum(
            int(np.any(source.validity[list(channels)] != target.validity[list(channels)]))
            for channels in FULL_VALIDITY_GROUPS.values()
        )
    return (
        int(round(abs(source_pt - target_pt) * 1_000_000)),
        int(round(abs(source_energy - target_energy) * 1_000_000)),
        int(_category(source) != _category(target)),
        int(_track_state(source) != _track_state(target)),
        int(validity),
    )


def audit_full_roles(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifests: Mapping[str, str | Path],
    coupling_manifests: Mapping[str, str | Path], data_root: str | Path,
    coupling_config_sha256: str, scale_calibration: Mapping[str, Any],
    switch_calibration: Mapping[str, Any],
    discrete_seed: int, output: str | Path, step_size: int = 4096,
) -> dict[str, Any]:
    if set(assignment_manifests) != {"train", "validation"} or set(coupling_manifests) != {"train", "validation"}:
        raise ValueError("HCWDL-UJ full-role audit inputs differ")
    validate_scale_calibration(
        scale_calibration, coupling_config_sha256=coupling_config_sha256,
    )
    validate_switch_calibration(
        switch_calibration, coupling_config_sha256=coupling_config_sha256,
    )
    switch_calibration_sha256 = str(switch_calibration["content_hash"])
    scales = scale_calibration["scales"]
    split_hash = str(split_manifest["content_hash"])
    expected: dict[str, int] = {}; observed: dict[str, int] = {}
    counters = {name: 0 for name in (
        "partition_failures", "assignment_injectivity_failures",
        "endpoint_payload_mismatches", "duplicate_endpoint_failures",
        "cardinality_failures", "active_count_overflow", "truncation_events",
        "u000_mismatches", "u100_mismatches", "j100_mismatches",
        "nonfinite_active_values", "forbidden_branch_reads",
        "independent_sample_mismatches", "solver_optimum_failures",
    )}
    digests = {name: hashlib.sha256() for name in ("p0", "d100", "hlt")}
    solver_matrix_digest = hashlib.sha256()
    solver_selection_digest = hashlib.sha256()
    solver_optimum_total = 0
    solver_rows = 0
    partition_totals = {name: 0 for name in ("A", "B", "K", "O", "R", "R_hlt", "R_off")}
    partition_role_summaries = {
        role: {name: 0 for name in (
            "rows", "A_count", "O_count", "B_count", "R_hlt_count",
            "R_off_count", "A_pt_micro", "O_pt_micro", "B_pt_micro",
            "R_hlt_pt_micro", "R_off_pt_micro",
        )} for role in ("train", "validation")
    }
    edit_totals = {name: 0 for name in ("substitution", "removal", "insertion")}
    switch_totals = {
        role: {
            f"s{level:03d}": _empty_transition_summary()
            for level in range(5, 101, 5)
        } for role in ("train", "validation")
    }
    sample_heaps: dict[str, list[tuple[int, str, str]]] = {"train": [], "validation": []}
    for role in ("train", "validation"):
        selection = RowSelection(selection_manifest, role=role, split_manifest_sha256=split_hash)
        assignments = DenseAssignmentStore(assignment_manifests[role])
        couplings = ResidualCouplingStore(coupling_manifests[role])
        expected[role] = selection.rows; observed[role] = 0
        for source_index, _record in enumerate(role_records(split_manifest, role)):
            for _, source_path, entries, arrays in _selected_source_chunks(
                split_manifest=split_manifest, selection=selection, data_root=data_root,
                role=role, source_index=source_index, step_size=step_size,
            ):
                mapping, confidence = assignments.join(source_path, entries)
                edit_rows = [couplings.get(source_path, int(entry)).edits for entry in entries]
                identities = [f"{source_path}::tree::{int(entry)}" for entry in entries]
                for row, edits in enumerate(edit_rows):
                    partition = build_partition_from_arrays(
                        arrays, row=row, assignment=mapping[row],
                    )
                    values = {
                        "A": len(partition.p0), "B": len(partition.d100),
                        "K": len(partition.common), "O": len(partition.source_only),
                        "R": len(partition.target_only), "R_hlt": len(partition.r_hlt),
                        "R_off": len(partition.r_off),
                    }
                    for name, count in values.items():
                        partition_totals[name] += count
                    role_summary = partition_role_summaries[role]
                    role_summary["rows"] += 1
                    for label, records in (
                        ("A", partition.p0), ("O", partition.source_only),
                        ("B", partition.d100), ("R_hlt", partition.r_hlt),
                        ("R_off", partition.r_off),
                    ):
                        role_summary[f"{label}_count"] += len(records)
                        role_summary[f"{label}_pt_micro"] += sum(
                            int(np.floor(float(np.hypot(record.p4[0], record.p4[1])) * 1_000_000 + .5))
                            for record in records
                        )
                    if values["K"] + values["O"] != values["A"] or values["K"] + values["R"] != values["B"]:
                        counters["cardinality_failures"] += 1
                    attributes = [_transition_attributes(partition, edit) for edit in edits]
                    for edit in edits:
                        edit_totals[("substitution", "removal", "insertion")[edit.edit_kind]] += 1
                    for level in range(5, 101, 5):
                        summary = switch_totals[role][f"s{level:03d}"]
                        switched = [
                            edit_is_active(edit, numerator=level, denominator=100)
                            for edit in edits
                        ]
                        active_count = len(partition.common)
                        for edit, active in zip(edits, switched, strict=True):
                            if (
                                edit.edit_kind == 0
                                or (edit.edit_kind == 1 and not active)
                                or (edit.edit_kind == 2 and active)
                            ):
                                active_count += 1
                        summary["rows"] += 1
                        summary["active_tokens_sum"] += active_count
                        summary["active_tokens_min"] = min(
                            summary["active_tokens_min"], active_count,
                        )
                        summary["active_tokens_max"] = max(
                            summary["active_tokens_max"], active_count,
                        )
                        summary["edit_count"] += len(edits)
                        switched_count = 0
                        row_totals = [0, 0, 0, 0, 0, 0]
                        row_switched = [0, 0, 0, 0, 0, 0]
                        for edit, active, diagnostic in zip(
                            edits, switched, attributes, strict=True,
                        ):
                            pt, energy, category, track, validity = diagnostic
                            values_for_fraction = (
                                int(edit.mass_q), pt, energy, category, track,
                                validity,
                            )
                            for item_index, item in enumerate(values_for_fraction):
                                row_totals[item_index] += item
                            summary["mass_q"] += int(edit.mass_q)
                            summary["absolute_pt_change_micro"] += pt
                            summary["absolute_energy_change_micro"] += energy
                            summary["category_change_count"] += category
                            summary["track_applicability_change_count"] += track
                            summary["validity_group_change_count"] += validity
                            if active:
                                switched_count += 1
                                for item_index, item in enumerate(values_for_fraction):
                                    row_switched[item_index] += item
                                summary["switched_edit_count"] += 1
                                summary["switched_mass_q"] += int(edit.mass_q)
                                summary["switched_absolute_pt_change_micro"] += pt
                                summary["switched_absolute_energy_change_micro"] += energy
                                summary["switched_category_change_count"] += category
                                summary["switched_track_applicability_change_count"] += track
                                summary["switched_validity_group_change_count"] += validity
                        distributions = summary["per_jet_distributions"]
                        distributions["active_tokens"][active_count] += 1
                        distributions["edit_count"][len(edits)] += 1
                        distributions["switched_edit_count"][switched_count] += 1
                        for name, numerator_value, denominator_value in zip(
                            (
                                "switched_mass_fraction_percent",
                                "switched_pt_change_fraction_percent",
                                "switched_energy_change_fraction_percent",
                                "switched_category_change_fraction_percent",
                                "switched_track_change_fraction_percent",
                                "switched_validity_change_fraction_percent",
                            ),
                            row_switched, row_totals, strict=True,
                        ):
                            distributions[name][
                                _fraction_percent_bin(
                                    numerator_value, denominator_value,
                                )
                            ] += 1
                    source = tuple(sorted(
                        partition.source_only, key=lambda value: value.source_key,
                    ))
                    target = tuple(sorted(
                        partition.target_only, key=lambda value: value.target_key,
                    ))
                    matrix = np.asarray([
                        [endpoint_cost(left, right, scales)[1] for right in target]
                        for left in source
                    ], dtype="<i8") if source and target else np.empty(
                        (len(source), len(target)), dtype="<i8",
                    )
                    _update_digest(
                        solver_matrix_digest,
                        np.asarray(matrix.shape, dtype="<i8"), matrix,
                    )
                    selected = np.asarray([
                        edit.key for edit in edits
                        if edit.edit_kind == 0
                    ], dtype="<i8").reshape(-1, 5)
                    _update_digest(solver_selection_digest, selected)
                    stored_total = sum(
                        edit.cost_q for edit in edits if edit.edit_kind == 0
                    )
                    if source and target:
                        from scipy.optimize import linear_sum_assignment
                        optimum_rows, optimum_columns = linear_sum_assignment(matrix)
                        optimum_total = sum(
                            int(matrix[row_index, column_index])
                            for row_index, column_index in zip(
                                optimum_rows, optimum_columns, strict=True,
                            )
                        )
                    else:
                        optimum_total = 0
                    if stored_total != optimum_total:
                        counters["solver_optimum_failures"] += 1
                    solver_optimum_total += optimum_total
                    solver_rows += 1
                p0 = build_homotopy_inputs(
                    arrays, assignments=mapping, confidence=confidence,
                    coupling_rows=edit_rows, coordinate=HomotopyCoordinate(0, 1, 0, 1),
                    identity_keys=identities, discrete_seed=discrete_seed,
                )
                u100 = build_homotopy_inputs(
                    arrays, assignments=mapping, confidence=confidence,
                    coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 0, 1),
                    identity_keys=identities, discrete_seed=discrete_seed,
                )
                j100 = build_homotopy_inputs(
                    arrays, assignments=mapping, confidence=confidence,
                    coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 1, 1),
                    identity_keys=identities, discrete_seed=discrete_seed,
                )
                hlt = build_hlt_inputs(arrays)
                expected_p0 = build_p0_inputs(arrays)
                offline_p4 = [project_offline_endpoint_records(arrays, row=i)[2] for i in range(len(entries))]
                d100 = build_alpha_repaired_inputs(
                    arrays, offline_p4, mapping, alpha=1.0,
                    repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
                    confidence_weights=confidence, offline_arrays=arrays,
                    identity_keys=identities, discrete_seed=discrete_seed,
                )
                try:
                    assert_particle_inputs_equal(u100, d100, endpoint="U100/D100")
                except ValueError:
                    counters["u100_mismatches"] += len(entries)
                try:
                    assert_particle_inputs_equal(j100, hlt, endpoint="J100/HLT")
                except ValueError:
                    counters["j100_mismatches"] += len(entries)
                try:
                    # U000 is order-equivalent rather than carrier-byte-identical:
                    # compare each row as a multiset of complete 21+p4 records.
                    for index in range(len(entries)):
                        n_left = int(p0.raw_lengths[index]); n_right = int(expected_p0.raw_lengths[index])
                        if n_left != n_right:
                            raise ValueError("P0 visible length differs")
                        left_tokens = sorted(
                            np.concatenate((p0.features[index, :, token], p0.vectors[index, :, token])).tobytes()
                            for token in range(n_left)
                        )
                        right_tokens = sorted(
                            np.concatenate((expected_p0.features[index, :, token], expected_p0.vectors[index, :, token])).tobytes()
                            for token in range(n_right)
                        )
                        if left_tokens != right_tokens:
                            raise ValueError("P0 projected multiset differs")
                except ValueError:
                    counters["u000_mismatches"] += len(entries)
                if any(np.any(~np.isfinite(value.features)) or np.any(~np.isfinite(value.vectors)) for value in (p0, u100, j100)):
                    counters["nonfinite_active_values"] += len(entries)
                _update_digest(digests["p0"], p0.features, p0.vectors, p0.mask, p0.raw_lengths)
                _update_digest(digests["d100"], d100.features, d100.vectors, d100.mask, d100.raw_lengths)
                _update_digest(digests["hlt"], hlt.features, hlt.vectors, hlt.mask, hlt.raw_lengths)
                for row, identity in enumerate(identities):
                    rank = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
                    item = (-rank, identity, _row_endpoint_hash(p0, u100, j100, row=row))
                    heap = sample_heaps[role]
                    if len(heap) < 128:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                observed[role] += len(entries)
    sample_expected = {
        role: {identity: digest for _, identity, digest in heap}
        for role, heap in sample_heaps.items()
    }
    sample_observed: dict[str, str] = {}
    sampled_displacement = {
        role: {"factorized": {}, "joint": {}}
        for role in ("train", "validation")
    }
    for role in ("train", "validation"):
        selection = RowSelection(selection_manifest, role=role, split_manifest_sha256=split_hash)
        assignments = DenseAssignmentStore(assignment_manifests[role])
        couplings = ResidualCouplingStore(coupling_manifests[role])
        wanted = sample_expected[role]
        for source_index, _record in enumerate(role_records(split_manifest, role)):
            for _, source_path, entries, arrays in _selected_source_chunks(
                split_manifest=split_manifest, selection=selection, data_root=data_root,
                role=role, source_index=source_index, step_size=step_size,
            ):
                selected = [index for index, entry in enumerate(entries) if f"{source_path}::tree::{int(entry)}" in wanted]
                if not selected:
                    continue
                chosen = np.asarray(selected, dtype=np.int64)
                subset = _slice(arrays, chosen); chosen_entries = entries[chosen]
                mapping, confidence = assignments.join(source_path, chosen_entries)
                edits = [couplings.get(source_path, int(entry)).edits for entry in chosen_entries]
                identities = [f"{source_path}::tree::{int(entry)}" for entry in chosen_entries]
                for row, identity in enumerate(identities):
                    partition = build_partition_from_arrays(
                        subset, row=row, assignment=mapping[row],
                    )
                    independently_recomputed = attach_switches(
                        assign_edit_masses(
                            couple_partition(partition, scale_calibration), partition,
                        ),
                        identity_key=identity,
                        coupling_config_sha256=coupling_config_sha256,
                        calibration=switch_calibration,
                    )
                    if tuple(edits[row]) != independently_recomputed:
                        counters["independent_sample_mismatches"] += 1
                views = [build_homotopy_inputs(
                    subset, assignments=mapping, confidence=confidence,
                    coupling_rows=edits, coordinate=coordinate,
                    identity_keys=identities, discrete_seed=discrete_seed,
                ) for coordinate in (
                    HomotopyCoordinate(0, 1, 0, 1),
                    HomotopyCoordinate(1, 1, 0, 1),
                    HomotopyCoordinate(1, 1, 1, 1),
                )]
                factorized_coordinates = [
                    (f"U{index * 10:03d}", HomotopyCoordinate(index, 10, 0, 1))
                    for index in range(1, 11)
                ] + [
                    (
                        f"D{100 - index * 10}F",
                        HomotopyCoordinate(1, 1, index, 10),
                    )
                    for index in range(1, 11)
                ]
                joint_coordinates = [
                    (f"J{index * 5:03d}", HomotopyCoordinate(index, 20, index, 20))
                    for index in range(1, 21)
                ]
                for track, coordinates in (
                    ("factorized", factorized_coordinates),
                    ("joint", joint_coordinates),
                ):
                    previous = views[0]
                    for node_id, coordinate in coordinates:
                        current = build_homotopy_inputs(
                            subset, assignments=mapping, confidence=confidence,
                            coupling_rows=edits, coordinate=coordinate,
                            identity_keys=identities, discrete_seed=discrete_seed,
                        )
                        row = sampled_displacement[role][track].setdefault(
                            node_id,
                            {
                                "sample_rows": 0, "feature_l1_micro": 0,
                                "p4_l1_micro": 0, "mask_change_count": 0,
                            },
                        )
                        row["sample_rows"] += len(identities)
                        row["feature_l1_micro"] += int(np.floor(
                            np.abs(
                                current.features.astype(np.float64)
                                - previous.features.astype(np.float64)
                            ).sum() * 1_000_000 + .5
                        ))
                        row["p4_l1_micro"] += int(np.floor(
                            np.abs(
                                current.vectors.astype(np.float64)
                                - previous.vectors.astype(np.float64)
                            ).sum() * 1_000_000 + .5
                        ))
                        row["mask_change_count"] += int(
                            np.count_nonzero(current.mask != previous.mask)
                        )
                        previous = current
                for row, identity in enumerate(identities):
                    digest = _row_endpoint_hash(*views, row=row)
                    sample_observed[identity] = digest
                    if digest != wanted[identity]:
                        counters["independent_sample_mismatches"] += 1
    if set(sample_observed) != {
        identity for rows in sample_expected.values() for identity in rows
    }:
        counters["independent_sample_mismatches"] += 1
    sample_hash = canonical_sha256({
        "selection": {role: sorted(rows) for role, rows in sample_expected.items()},
        "recomputed": dict(sorted(sample_observed.items())),
        "method": "lowest_sha256_identity_second_root_assignment_reread_v1",
    })
    payload = build_coupling_audit(
        coupling_config_sha256=coupling_config_sha256,
        train_manifest_sha256=ResidualCouplingStore(coupling_manifests["train"]).manifest["content_hash"],
        validation_manifest_sha256=ResidualCouplingStore(coupling_manifests["validation"]).manifest["content_hash"],
        expected_rows=expected, observed_rows=observed, counters=counters,
        endpoint_logical_sha256={name: digest.hexdigest() for name, digest in digests.items()},
        branch_allowlist_sha256=branch_allowlist_sha256(),
        branch_access_trace_sha256=canonical_sha256({"branches": coupling_branch_allowlist(), "roles": ["train", "validation"]}),
        independent_sample_sha256=sample_hash,
    )
    payload["switch_calibration_sha256"] = require_sha256(switch_calibration_sha256, name="switch calibration")
    payload["partition_totals"] = partition_totals
    for role, summary in partition_role_summaries.items():
        summary["O_count_fraction_of_A"] = (
            None if summary["A_count"] == 0
            else summary["O_count"] / summary["A_count"]
        )
        summary["R_hlt_count_fraction_of_B"] = (
            None if summary["B_count"] == 0
            else summary["R_hlt_count"] / summary["B_count"]
        )
        summary["R_off_count_fraction_of_B"] = (
            None if summary["B_count"] == 0
            else summary["R_off_count"] / summary["B_count"]
        )
        summary["O_pt_fraction_of_A"] = (
            None if summary["A_pt_micro"] == 0
            else summary["O_pt_micro"] / summary["A_pt_micro"]
        )
        summary["R_hlt_pt_fraction_of_B"] = (
            None if summary["B_pt_micro"] == 0
            else summary["R_hlt_pt_micro"] / summary["B_pt_micro"]
        )
        summary["R_off_pt_fraction_of_B"] = (
            None if summary["B_pt_micro"] == 0
            else summary["R_off_pt_micro"] / summary["B_pt_micro"]
        )
    payload["partition_role_summaries"] = partition_role_summaries
    payload["edit_totals"] = edit_totals
    payload["transition_summaries"] = switch_totals
    payload["sampled_realized_view_displacement"] = sampled_displacement
    payload["independent_sample_rows"] = {
        role: len(rows) for role, rows in sample_expected.items()
    }
    import scipy
    payload["solver_audit"] = {
        "algorithm": "scipy_hungarian_plus_canonical_edge_fixing_v1",
        "scipy_version": scipy.__version__, "rows": solver_rows,
        "integer_matrix_sha256": solver_matrix_digest.hexdigest(),
        "selected_edge_tuple_sha256": solver_selection_digest.hexdigest(),
        "optimum_total_cost_q": solver_optimum_total,
    }
    from hlt_classification.data.cache_contracts import with_content_hash
    payload = with_content_hash(payload); write_immutable_json(output, payload); return payload


__all__ = [
    "audit_full_roles", "branch_allowlist_sha256", "build_coupling_source",
    "build_switch_sidecar_for_source", "calibrate_train_scales",
    "coupling_branch_allowlist", "finalize_base_role", "finalize_coupling_role",
    "freeze_switch_calibration", "iter_base_edits",
]
