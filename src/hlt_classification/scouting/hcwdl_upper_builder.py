"""Production coupling calibration, shard construction, finalization, and audit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    build_p0_inputs, build_partition_from_arrays, prepare_hlt_endpoints,
    prepare_offline_endpoints,
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
from .hcwdl_assignment_store import load_assignment_shard, open_assignment_store
from .repair import (
    FULL_VALIDITY_GROUPS, HIGHCOV_SHELL_EXACT_FAMILY, build_alpha_repaired_inputs,
    full_endpoint_required_branches,
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


def _prepared_partitions(
    arrays: Mapping[str, object], assignments: np.ndarray,
):
    """Build a chunk's endpoint projections once and reuse them for every row."""

    offline = prepare_offline_endpoints(arrays)
    hlt = prepare_hlt_endpoints(arrays)
    mapping = np.asarray(assignments)
    if mapping.shape[0] != offline.rows or hlt.rows != offline.rows:
        raise ValueError("HCWDL-UJ prepared partition row counts differ")
    partitions = tuple(
        build_partition_from_arrays(
            arrays, row=row, assignment=mapping[row],
            prepared_offline=offline, prepared_hlt=hlt,
        )
        for row in range(offline.rows)
    )
    return partitions, offline, hlt


def _selected_source_chunks(
    *, split_manifest: Mapping[str, Any], selection: RowSelection,
    assignments: object, data_root: str | Path, role: str,
    source_index: int, step_size: int,
):
    """Yield the assignment-locked selected population for one source.

    ``RowSelection.all_rows`` means every *mapped* row in the split, not every
    raw ROOT entry.  Bounded selections encode their mapped identities
    explicitly, whereas the all-mapped representation deliberately omits that
    redundant multi-million-entry list.  The dense assignment shard is the
    authenticated identity index common to both representations, so coupling
    consumers use it as the population carrier and use ``RowSelection`` as an
    independent authorization check.
    """

    records = role_records(split_manifest, role)
    if role not in {"train", "validation"} or not 0 <= source_index < len(records):
        raise ValueError("HCWDL-UJ source role/index differs")
    record = records[source_index]
    matching = [
        row for row in assignments.manifest.get("shards", ())
        if str(row.get("source_path")) == record.path
    ]
    if len(matching) != 1:
        raise ValueError("HCWDL-UJ assignment source identity differs")
    assignment_record = matching[0]
    metadata, assignment_arrays = load_assignment_shard(
        assignments.path.parent / str(assignment_record["metadata_path"]),
    )
    # Retain only the compact identity index.  ``DenseAssignmentStore.join``
    # lazily owns the full ragged shard when endpoint rows are needed; keeping
    # this temporary load alive would otherwise duplicate every assignment
    # array for the duration of a full-source scan.
    assignment_entries = np.asarray(
        assignment_arrays["entries"], dtype=np.int64,
    ).copy()
    if (
        metadata.get("content_hash") != assignment_record.get("metadata_sha256")
        or metadata.get("source_path") != record.path
        or int(assignment_record.get("rows", -1)) != len(assignment_entries)
        or (
            len(assignment_entries) > 1
            and np.any(assignment_entries[1:] <= assignment_entries[:-1])
        )
    ):
        raise ValueError("HCWDL-UJ assignment source-entry index differs")
    del assignment_arrays, metadata
    expected = selection.source_rows(record.path)
    if expected < 0:
        expected = record.mapped_entries
    if len(assignment_entries) != expected:
        raise ValueError("HCWDL-UJ assignment/selection source coverage differs")

    observed = 0
    for chunk in iterate_projected_chunks(
        (Path(data_root) / record.path,), coupling_branch_allowlist(),
        data_root=data_root, role=role, step_size=step_size,
    ):
        if chunk.source_path != record.path:
            raise ValueError("HCWDL-UJ streamed source identity differs")
        left = int(np.searchsorted(assignment_entries, chunk.entry_start, side="left"))
        right = int(np.searchsorted(assignment_entries, chunk.entry_stop, side="left"))
        entries = assignment_entries[left:right]
        if not len(entries):
            continue
        if not np.all(selection.mask(chunk.source_path, entries)):
            raise ValueError("HCWDL-UJ assignment identity is outside row selection")
        indexes = entries - int(chunk.entry_start)
        if np.any(indexes < 0) or np.any(indexes >= chunk.entry_stop - chunk.entry_start):
            raise RuntimeError("HCWDL-UJ assignment/chunk join differs")
        observed += len(entries)
        yield record, chunk.source_path, entries, _slice(chunk.arrays, indexes)
    if observed != len(assignment_entries):
        raise ValueError("HCWDL-UJ assignment identities are absent from ROOT source")


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
    store = open_assignment_store(assignment_manifest)
    accumulator = ScaleAccumulator(); observed = 0; identity_digest = hashlib.sha256()
    source_path = role_records(split_manifest, "train")[source_index].path
    for _, actual_source, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection, assignments=store,
        data_root=data_root, role="train", source_index=source_index,
        step_size=step_size,
    ):
        if actual_source != source_path:
            raise ValueError("HCWDL-UJ calibration source identity differs")
        assignment, _ = store.join(source_path, entries)
        partitions, _, _ = _prepared_partitions(arrays, assignment)
        for row, entry in enumerate(entries):
            identity = f"{source_path}::tree::{int(entry)}".encode("utf-8")
            identity_digest.update(len(identity).to_bytes(4, "little")); identity_digest.update(identity)
            accumulator.update_partition(partitions[row])
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
    store = open_assignment_store(assignment_manifest)
    records = role_records(split_manifest, role); record = records[source_index]
    entries_out: list[int] = []; edits_out: list[tuple[ResidualEdit, ...]] = []
    for _, source_path, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection, assignments=store,
        data_root=data_root, role=role, source_index=source_index,
        step_size=step_size,
    ):
        assignment, _ = store.join(source_path, entries)
        partitions, _, _ = _prepared_partitions(arrays, assignment)
        for row, entry in enumerate(entries):
            partition = partitions[row]
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


_AUDIT_COUNTER_NAMES = (
    "partition_failures", "assignment_injectivity_failures",
    "endpoint_payload_mismatches", "duplicate_endpoint_failures",
    "cardinality_failures", "active_count_overflow", "truncation_events",
    "u000_mismatches", "u100_mismatches", "j100_mismatches",
    "nonfinite_active_values", "forbidden_branch_reads",
    "independent_sample_mismatches", "solver_optimum_failures",
)
_PARTITION_NAMES = ("A", "B", "K", "O", "R", "R_hlt", "R_off")
_ROLE_SUMMARY_NAMES = (
    "rows", "A_count", "O_count", "B_count", "R_hlt_count",
    "R_off_count", "A_pt_micro", "O_pt_micro", "B_pt_micro",
    "R_hlt_pt_micro", "R_off_pt_micro",
)
_EDIT_NAMES = ("substitution", "removal", "insertion")
_TRANSITION_SCALAR_SUMS = (
    "rows", "active_tokens_sum", "edit_count", "switched_edit_count",
    "mass_q", "switched_mass_q", "absolute_pt_change_micro",
    "switched_absolute_pt_change_micro", "absolute_energy_change_micro",
    "switched_absolute_energy_change_micro", "category_change_count",
    "switched_category_change_count", "track_applicability_change_count",
    "switched_track_applicability_change_count",
    "validity_group_change_count", "switched_validity_group_change_count",
)
_AUDIT_DIGEST_REDUCTION = "canonical_role_source_order_framed_subhashes_v1"


def _empty_role_summary() -> dict[str, int]:
    return {name: 0 for name in _ROLE_SUMMARY_NAMES}


def _empty_switch_totals() -> dict[str, dict[str, Any]]:
    return {
        f"s{level:03d}": _empty_transition_summary()
        for level in range(5, 101, 5)
    }


def _merge_transition_summary(
    destination: dict[str, Any], source: Mapping[str, Any],
) -> None:
    for name in _TRANSITION_SCALAR_SUMS:
        destination[name] += int(source[name])
    destination["active_tokens_min"] = min(
        int(destination["active_tokens_min"]), int(source["active_tokens_min"]),
    )
    destination["active_tokens_max"] = max(
        int(destination["active_tokens_max"]), int(source["active_tokens_max"]),
    )
    for name, counts in source["per_jet_distributions"].items():
        target = destination["per_jet_distributions"][name]
        if len(target) != len(counts):
            raise ValueError("HCWDL-UJ transition distribution shape differs")
        for index, count in enumerate(counts):
            target[index] += int(count)


def _merge_displacement(
    destination: dict[str, dict[str, dict[str, int]]],
    source: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> None:
    for track, rows in source.items():
        for node_id, row in rows.items():
            target = destination[track].setdefault(
                node_id,
                {
                    "sample_rows": 0, "feature_l1_micro": 0,
                    "p4_l1_micro": 0, "mask_change_count": 0,
                },
            )
            for name in target:
                target[name] += int(row[name])


def _framed_source_digest(
    results: Sequence[Mapping[str, Any]], *, key: str, domain: str,
) -> str:
    digest = hashlib.sha256()
    encoded_domain = domain.encode("utf-8")
    digest.update(len(encoded_domain).to_bytes(4, "little")); digest.update(encoded_domain)
    for result in results:
        for value in (
            str(result["role"]), str(result["source_path"]),
            str(int(result["observed"])), str(result[key]),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little")); digest.update(encoded)
    return digest.hexdigest()


def _run_audit_workers(
    worker: Callable[[tuple[Any, ...]], Any],
    arguments: Sequence[tuple[Any, ...]], *, workers: int,
    label: str,
) -> list[Any]:
    if workers <= 0:
        raise ValueError("HCWDL-UJ audit worker count differs")
    if workers == 1:
        results = []
        for index, argument in enumerate(arguments, start=1):
            results.append(worker(argument))
            print(
                f"HCWDL-UJ {label}: completed source {index}/{len(arguments)}",
                flush=True,
            )
        return results
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, argument) for argument in arguments]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(
                f"HCWDL-UJ {label}: completed source {index}/{len(arguments)}",
                flush=True,
            )
    return results


def _audit_source_worker(arguments: tuple[Any, ...]) -> dict[str, Any]:
    (
        split_manifest, selection_manifest, assignment_manifest,
        coupling_manifest, data_root, role, source_index, scales,
        discrete_seed, step_size,
    ) = arguments
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    assignments = open_assignment_store(assignment_manifest)
    couplings = ResidualCouplingStore(coupling_manifest)
    record = role_records(split_manifest, role)[source_index]
    counters = {name: 0 for name in _AUDIT_COUNTER_NAMES}
    digests = {name: hashlib.sha256() for name in ("p0", "d100", "hlt")}
    solver_matrix_digest = hashlib.sha256()
    solver_selection_digest = hashlib.sha256()
    solver_optimum_total = 0
    solver_rows = 0
    partition_totals = {name: 0 for name in _PARTITION_NAMES}
    role_summary = _empty_role_summary()
    edit_totals = {name: 0 for name in _EDIT_NAMES}
    switch_totals = _empty_switch_totals()
    sample_heap: list[tuple[int, str, str]] = []
    observed = 0
    for _, source_path, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection,
        assignments=assignments, data_root=data_root, role=role,
        source_index=source_index, step_size=step_size,
    ):
        if source_path != record.path:
            raise ValueError("HCWDL-UJ audit source identity differs")
        mapping, confidence = assignments.join(source_path, entries)
        edit_rows = [couplings.get(source_path, int(entry)).edits for entry in entries]
        identities = [f"{source_path}::tree::{int(entry)}" for entry in entries]
        partitions, prepared_offline, prepared_hlt = _prepared_partitions(
            arrays, mapping,
        )
        for row, edits in enumerate(edit_rows):
            partition = partitions[row]
            values = {
                "A": len(partition.p0), "B": len(partition.d100),
                "K": len(partition.common), "O": len(partition.source_only),
                "R": len(partition.target_only), "R_hlt": len(partition.r_hlt),
                "R_off": len(partition.r_off),
            }
            for name, count in values.items():
                partition_totals[name] += count
            role_summary["rows"] += 1
            for name, records in (
                ("A", partition.p0), ("O", partition.source_only),
                ("B", partition.d100), ("R_hlt", partition.r_hlt),
                ("R_off", partition.r_off),
            ):
                role_summary[f"{name}_count"] += len(records)
                role_summary[f"{name}_pt_micro"] += sum(
                    int(np.floor(
                        float(np.hypot(item.p4[0], item.p4[1])) * 1_000_000 + .5
                    ))
                    for item in records
                )
            if (
                values["K"] + values["O"] != values["A"]
                or values["K"] + values["R"] != values["B"]
            ):
                counters["cardinality_failures"] += 1
            attributes = [_transition_attributes(partition, edit) for edit in edits]
            for edit in edits:
                edit_totals[_EDIT_NAMES[edit.edit_kind]] += 1
            for level in range(5, 101, 5):
                summary = switch_totals[f"s{level:03d}"]
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
                        int(edit.mass_q), pt, energy, category, track, validity,
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
                        _fraction_percent_bin(numerator_value, denominator_value)
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
                solver_matrix_digest, np.asarray(matrix.shape, dtype="<i8"), matrix,
            )
            selected = np.asarray([
                edit.key for edit in edits if edit.edit_kind == 0
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
            prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
        )
        u100 = build_homotopy_inputs(
            arrays, assignments=mapping, confidence=confidence,
            coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 0, 1),
            identity_keys=identities, discrete_seed=discrete_seed,
            prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
        )
        j100 = build_homotopy_inputs(
            arrays, assignments=mapping, confidence=confidence,
            coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 1, 1),
            identity_keys=identities, discrete_seed=discrete_seed,
            prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
        )
        hlt = build_hlt_inputs(arrays)
        expected_p0 = build_p0_inputs(arrays, prepared=prepared_offline)
        offline_p4 = prepared_offline.p4
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
            for index in range(len(entries)):
                n_left = int(p0.raw_lengths[index])
                n_right = int(expected_p0.raw_lengths[index])
                if n_left != n_right:
                    raise ValueError("P0 visible length differs")
                left_tokens = sorted(
                    np.concatenate((
                        p0.features[index, :, token], p0.vectors[index, :, token],
                    )).tobytes()
                    for token in range(n_left)
                )
                right_tokens = sorted(
                    np.concatenate((
                        expected_p0.features[index, :, token],
                        expected_p0.vectors[index, :, token],
                    )).tobytes()
                    for token in range(n_right)
                )
                if left_tokens != right_tokens:
                    raise ValueError("P0 projected multiset differs")
        except ValueError:
            counters["u000_mismatches"] += len(entries)
        if any(
            np.any(~np.isfinite(value.features))
            or np.any(~np.isfinite(value.vectors))
            for value in (p0, u100, j100)
        ):
            counters["nonfinite_active_values"] += len(entries)
        _update_digest(
            digests["p0"], p0.features, p0.vectors, p0.mask, p0.raw_lengths,
        )
        _update_digest(
            digests["d100"], d100.features, d100.vectors, d100.mask,
            d100.raw_lengths,
        )
        _update_digest(
            digests["hlt"], hlt.features, hlt.vectors, hlt.mask, hlt.raw_lengths,
        )
        for row, identity in enumerate(identities):
            rank = int.from_bytes(
                hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big",
            )
            item = (-rank, identity, _row_endpoint_hash(p0, u100, j100, row=row))
            if len(sample_heap) < 128:
                heapq.heappush(sample_heap, item)
            elif item > sample_heap[0]:
                heapq.heapreplace(sample_heap, item)
        observed += len(entries)
    expected = selection.source_rows(record.path)
    if expected < 0:
        expected = record.mapped_entries
    if observed != expected or role_summary["rows"] != expected:
        raise ValueError("HCWDL-UJ per-source full-role audit coverage differs")
    return {
        "role": role, "source_index": source_index, "source_path": record.path,
        "expected": expected, "observed": observed, "counters": counters,
        "endpoint_p0_sha256": digests["p0"].hexdigest(),
        "endpoint_d100_sha256": digests["d100"].hexdigest(),
        "endpoint_hlt_sha256": digests["hlt"].hexdigest(),
        "solver_matrix_sha256": solver_matrix_digest.hexdigest(),
        "solver_selection_sha256": solver_selection_digest.hexdigest(),
        "solver_optimum_total": solver_optimum_total, "solver_rows": solver_rows,
        "partition_totals": partition_totals, "role_summary": role_summary,
        "edit_totals": edit_totals, "switch_totals": switch_totals,
        "sample_heap": sample_heap,
    }


def _audit_sample_source_worker(arguments: tuple[Any, ...]) -> dict[str, Any]:
    (
        split_manifest, selection_manifest, assignment_manifest,
        coupling_manifest, data_root, role, source_index, scale_calibration,
        switch_calibration, coupling_config_sha256, discrete_seed, step_size,
        wanted,
    ) = arguments
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    assignments = open_assignment_store(assignment_manifest)
    couplings = ResidualCouplingStore(coupling_manifest)
    record = role_records(split_manifest, role)[source_index]
    sample_observed: dict[str, str] = {}
    mismatches = 0
    sampled_displacement = {"factorized": {}, "joint": {}}
    for _, source_path, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection,
        assignments=assignments, data_root=data_root, role=role,
        source_index=source_index, step_size=step_size,
    ):
        if source_path != record.path:
            raise ValueError("HCWDL-UJ sample-audit source identity differs")
        selected = [
            index for index, entry in enumerate(entries)
            if f"{source_path}::tree::{int(entry)}" in wanted
        ]
        if not selected:
            continue
        chosen = np.asarray(selected, dtype=np.int64)
        subset = _slice(arrays, chosen)
        chosen_entries = entries[chosen]
        mapping, confidence = assignments.join(source_path, chosen_entries)
        edits = [couplings.get(source_path, int(entry)).edits for entry in chosen_entries]
        identities = [f"{source_path}::tree::{int(entry)}" for entry in chosen_entries]
        partitions, prepared_offline, prepared_hlt = _prepared_partitions(
            subset, mapping,
        )
        for row, identity in enumerate(identities):
            partition = partitions[row]
            independently_recomputed = attach_switches(
                assign_edit_masses(
                    couple_partition(partition, scale_calibration), partition,
                ),
                identity_key=identity,
                coupling_config_sha256=coupling_config_sha256,
                calibration=switch_calibration,
            )
            if tuple(edits[row]) != independently_recomputed:
                mismatches += 1
        views = [build_homotopy_inputs(
            subset, assignments=mapping, confidence=confidence,
            coupling_rows=edits, coordinate=coordinate,
            identity_keys=identities, discrete_seed=discrete_seed,
            prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
        ) for coordinate in (
            HomotopyCoordinate(0, 1, 0, 1),
            HomotopyCoordinate(1, 1, 0, 1),
            HomotopyCoordinate(1, 1, 1, 1),
        )]
        factorized_coordinates = [
            (f"U{index * 20:03d}", HomotopyCoordinate(index, 5, 0, 1))
            for index in range(1, 6)
        ] + [
            (
                f"D{100 - index * 20}F",
                HomotopyCoordinate(1, 1, index, 5),
            )
            for index in range(1, 6)
        ]
        joint_coordinates = [
            (f"J{index * 10:03d}", HomotopyCoordinate(index, 10, index, 10))
            for index in range(1, 11)
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
                    prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
                )
                summary = sampled_displacement[track].setdefault(
                    node_id,
                    {
                        "sample_rows": 0, "feature_l1_micro": 0,
                        "p4_l1_micro": 0, "mask_change_count": 0,
                    },
                )
                summary["sample_rows"] += len(identities)
                summary["feature_l1_micro"] += int(np.floor(
                    np.abs(
                        current.features.astype(np.float64)
                        - previous.features.astype(np.float64)
                    ).sum() * 1_000_000 + .5
                ))
                summary["p4_l1_micro"] += int(np.floor(
                    np.abs(
                        current.vectors.astype(np.float64)
                        - previous.vectors.astype(np.float64)
                    ).sum() * 1_000_000 + .5
                ))
                summary["mask_change_count"] += int(
                    np.count_nonzero(current.mask != previous.mask)
                )
                previous = current
        for row, identity in enumerate(identities):
            digest = _row_endpoint_hash(*views, row=row)
            sample_observed[identity] = digest
            if digest != wanted[identity]:
                mismatches += 1
    if set(sample_observed) != set(wanted):
        mismatches += 1
    return {
        "role": role, "source_index": source_index, "source_path": record.path,
        "sample_observed": sample_observed, "mismatches": mismatches,
        "sampled_displacement": sampled_displacement,
    }


def _audit_full_roles_serial_reference(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifests: Mapping[str, str | Path],
    coupling_manifests: Mapping[str, str | Path], data_root: str | Path,
    coupling_config_sha256: str, scale_calibration: Mapping[str, Any],
    switch_calibration: Mapping[str, Any],
    discrete_seed: int, output: str | Path, step_size: int = 4096,
) -> dict[str, Any]:
    """Frozen single-process reference retained for reducer parity audits."""

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
        assignments = open_assignment_store(assignment_manifests[role])
        couplings = ResidualCouplingStore(coupling_manifests[role])
        expected[role] = selection.rows; observed[role] = 0
        for source_index, _record in enumerate(role_records(split_manifest, role)):
            for _, source_path, entries, arrays in _selected_source_chunks(
                split_manifest=split_manifest, selection=selection,
                assignments=assignments, data_root=data_root, role=role,
                source_index=source_index, step_size=step_size,
            ):
                mapping, confidence = assignments.join(source_path, entries)
                edit_rows = [couplings.get(source_path, int(entry)).edits for entry in entries]
                identities = [f"{source_path}::tree::{int(entry)}" for entry in entries]
                partitions, prepared_offline, prepared_hlt = _prepared_partitions(
                    arrays, mapping,
                )
                for row, edits in enumerate(edit_rows):
                    partition = partitions[row]
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
                    prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
                )
                u100 = build_homotopy_inputs(
                    arrays, assignments=mapping, confidence=confidence,
                    coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 0, 1),
                    identity_keys=identities, discrete_seed=discrete_seed,
                    prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
                )
                j100 = build_homotopy_inputs(
                    arrays, assignments=mapping, confidence=confidence,
                    coupling_rows=edit_rows, coordinate=HomotopyCoordinate(1, 1, 1, 1),
                    identity_keys=identities, discrete_seed=discrete_seed,
                    prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
                )
                hlt = build_hlt_inputs(arrays)
                expected_p0 = build_p0_inputs(arrays, prepared=prepared_offline)
                offline_p4 = prepared_offline.p4
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
        assignments = open_assignment_store(assignment_manifests[role])
        couplings = ResidualCouplingStore(coupling_manifests[role])
        wanted = sample_expected[role]
        for source_index, _record in enumerate(role_records(split_manifest, role)):
            for _, source_path, entries, arrays in _selected_source_chunks(
                split_manifest=split_manifest, selection=selection,
                assignments=assignments, data_root=data_root, role=role,
                source_index=source_index, step_size=step_size,
            ):
                selected = [index for index, entry in enumerate(entries) if f"{source_path}::tree::{int(entry)}" in wanted]
                if not selected:
                    continue
                chosen = np.asarray(selected, dtype=np.int64)
                subset = _slice(arrays, chosen); chosen_entries = entries[chosen]
                mapping, confidence = assignments.join(source_path, chosen_entries)
                edits = [couplings.get(source_path, int(entry)).edits for entry in chosen_entries]
                identities = [f"{source_path}::tree::{int(entry)}" for entry in chosen_entries]
                partitions, prepared_offline, prepared_hlt = _prepared_partitions(
                    subset, mapping,
                )
                for row, identity in enumerate(identities):
                    partition = partitions[row]
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
                    prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
                ) for coordinate in (
                    HomotopyCoordinate(0, 1, 0, 1),
                    HomotopyCoordinate(1, 1, 0, 1),
                    HomotopyCoordinate(1, 1, 1, 1),
                )]
                factorized_coordinates = [
                    (f"U{index * 20:03d}", HomotopyCoordinate(index, 5, 0, 1))
                    for index in range(1, 6)
                ] + [
                    (
                        f"D{100 - index * 20}F",
                        HomotopyCoordinate(1, 1, index, 5),
                    )
                    for index in range(1, 6)
                ]
                joint_coordinates = [
                    (f"J{index * 10:03d}", HomotopyCoordinate(index, 10, index, 10))
                    for index in range(1, 11)
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
                            prepared_offline=prepared_offline, prepared_hlt=prepared_hlt,
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


def audit_full_roles(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifests: Mapping[str, str | Path],
    coupling_manifests: Mapping[str, str | Path], data_root: str | Path,
    coupling_config_sha256: str, scale_calibration: Mapping[str, Any],
    switch_calibration: Mapping[str, Any], discrete_seed: int,
    output: str | Path, step_size: int = 4096,
    workers: int | None = None,
) -> dict[str, Any]:
    """Exhaustively audit each source in parallel and reduce canonically."""

    if (
        set(assignment_manifests) != {"train", "validation"}
        or set(coupling_manifests) != {"train", "validation"}
    ):
        raise ValueError("HCWDL-UJ full-role audit inputs differ")
    validate_scale_calibration(
        scale_calibration, coupling_config_sha256=coupling_config_sha256,
    )
    validate_switch_calibration(
        switch_calibration, coupling_config_sha256=coupling_config_sha256,
    )
    switch_calibration_sha256 = str(switch_calibration["content_hash"])
    split_hash = str(split_manifest["content_hash"])
    role_order = {"train": 0, "validation": 1}
    expected: dict[str, int] = {}
    arguments: list[tuple[Any, ...]] = []
    expected_keys: list[tuple[str, int, str]] = []
    for role in ("train", "validation"):
        selection = RowSelection(
            selection_manifest, role=role, split_manifest_sha256=split_hash,
        )
        expected[role] = selection.rows
        for source_index, record in enumerate(role_records(split_manifest, role)):
            expected_keys.append((role, source_index, record.path))
            arguments.append((
                split_manifest, selection_manifest,
                str(assignment_manifests[role]), str(coupling_manifests[role]),
                str(data_root), role, source_index, scale_calibration["scales"],
                discrete_seed, step_size,
            ))
    requested_workers = (
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if workers is None else int(workers)
    )
    worker_count = min(len(arguments), max(1, requested_workers))
    results = _run_audit_workers(
        _audit_source_worker, arguments, workers=worker_count,
        label="full-role audit",
    )
    results.sort(key=lambda row: (role_order[str(row["role"])], int(row["source_index"])))
    actual_keys = [
        (str(row["role"]), int(row["source_index"]), str(row["source_path"]))
        for row in results
    ]
    if actual_keys != expected_keys:
        raise ValueError("HCWDL-UJ audit source reduction order differs")

    observed = {"train": 0, "validation": 0}
    counters = {name: 0 for name in _AUDIT_COUNTER_NAMES}
    partition_totals = {name: 0 for name in _PARTITION_NAMES}
    partition_role_summaries = {
        role: _empty_role_summary() for role in ("train", "validation")
    }
    edit_totals = {name: 0 for name in _EDIT_NAMES}
    switch_totals = {
        role: _empty_switch_totals() for role in ("train", "validation")
    }
    sample_heaps: dict[str, list[tuple[int, str, str]]] = {
        "train": [], "validation": [],
    }
    solver_optimum_total = 0
    solver_rows = 0
    for result in results:
        role = str(result["role"])
        observed[role] += int(result["observed"])
        for name in counters:
            counters[name] += int(result["counters"][name])
        for name in partition_totals:
            partition_totals[name] += int(result["partition_totals"][name])
        for name in _ROLE_SUMMARY_NAMES:
            partition_role_summaries[role][name] += int(
                result["role_summary"][name]
            )
        for name in edit_totals:
            edit_totals[name] += int(result["edit_totals"][name])
        for level, summary in result["switch_totals"].items():
            _merge_transition_summary(switch_totals[role][level], summary)
        for item in result["sample_heap"]:
            heap = sample_heaps[role]
            if len(heap) < 128:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        solver_optimum_total += int(result["solver_optimum_total"])
        solver_rows += int(result["solver_rows"])
    if observed != expected:
        raise ValueError("HCWDL-UJ full-role audit coverage differs")

    endpoint_hashes = {
        name: _framed_source_digest(
            results, key=f"endpoint_{name}_sha256",
            domain=f"HCWDL-UJ/full-role/{name}/v1",
        )
        for name in ("p0", "d100", "hlt")
    }
    solver_matrix_hash = _framed_source_digest(
        results, key="solver_matrix_sha256",
        domain="HCWDL-UJ/full-role/solver-matrix/v1",
    )
    solver_selection_hash = _framed_source_digest(
        results, key="solver_selection_sha256",
        domain="HCWDL-UJ/full-role/solver-selection/v1",
    )

    sample_expected = {
        role: {identity: digest for _, identity, digest in heap}
        for role, heap in sample_heaps.items()
    }
    sample_arguments: list[tuple[Any, ...]] = []
    for result in results:
        role = str(result["role"])
        wanted = {
            identity: sample_expected[role][identity]
            for _, identity, _ in result["sample_heap"]
            if identity in sample_expected[role]
        }
        if not wanted:
            continue
        sample_arguments.append((
            split_manifest, selection_manifest,
            str(assignment_manifests[role]), str(coupling_manifests[role]),
            str(data_root), role, int(result["source_index"]),
            scale_calibration, switch_calibration, coupling_config_sha256,
            discrete_seed, step_size, wanted,
        ))
    sample_results = _run_audit_workers(
        _audit_sample_source_worker, sample_arguments,
        workers=min(worker_count, max(1, len(sample_arguments))),
        label="independent sample audit",
    ) if sample_arguments else []
    sample_results.sort(
        key=lambda row: (role_order[str(row["role"])], int(row["source_index"])),
    )
    sample_observed: dict[str, str] = {}
    sampled_displacement = {
        role: {"factorized": {}, "joint": {}}
        for role in ("train", "validation")
    }
    for result in sample_results:
        overlap = set(sample_observed).intersection(result["sample_observed"])
        if overlap:
            raise ValueError("HCWDL-UJ independent sample identity is duplicated")
        sample_observed.update(result["sample_observed"])
        counters["independent_sample_mismatches"] += int(result["mismatches"])
        _merge_displacement(
            sampled_displacement[str(result["role"])],
            result["sampled_displacement"],
        )
    expected_sample_identities = {
        identity for rows in sample_expected.values() for identity in rows
    }
    if set(sample_observed) != expected_sample_identities:
        counters["independent_sample_mismatches"] += 1
    sample_hash = canonical_sha256({
        "selection": {role: sorted(rows) for role, rows in sample_expected.items()},
        "recomputed": dict(sorted(sample_observed.items())),
        "method": "lowest_sha256_identity_second_root_assignment_reread_v1",
    })

    payload = build_coupling_audit(
        coupling_config_sha256=coupling_config_sha256,
        train_manifest_sha256=ResidualCouplingStore(
            coupling_manifests["train"],
        ).manifest["content_hash"],
        validation_manifest_sha256=ResidualCouplingStore(
            coupling_manifests["validation"],
        ).manifest["content_hash"],
        expected_rows=expected, observed_rows=observed, counters=counters,
        endpoint_logical_sha256=endpoint_hashes,
        branch_allowlist_sha256=branch_allowlist_sha256(),
        branch_access_trace_sha256=canonical_sha256({
            "branches": coupling_branch_allowlist(),
            "roles": ["train", "validation"],
        }),
        independent_sample_sha256=sample_hash,
    )
    payload["switch_calibration_sha256"] = require_sha256(
        switch_calibration_sha256, name="switch calibration",
    )
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
    payload["audit_execution"] = {
        "source_partition": "one_process_task_per_authenticated_source_unit_v1",
        "reduction": _AUDIT_DIGEST_REDUCTION,
        "integer_reduction_order": "train_then_validation_canonical_source_index_v1",
        "worker_count_not_scientific_identity": True,
    }
    import scipy
    payload["solver_audit"] = {
        "algorithm": "scipy_hungarian_plus_canonical_edge_fixing_v1",
        "scipy_version": scipy.__version__, "rows": solver_rows,
        "integer_matrix_sha256": solver_matrix_hash,
        "selected_edge_tuple_sha256": solver_selection_hash,
        "optimum_total_cost_q": solver_optimum_total,
        "digest_reduction": _AUDIT_DIGEST_REDUCTION,
    }
    from hlt_classification.data.cache_contracts import with_content_hash
    payload = with_content_hash(payload)
    write_immutable_json(output, payload)
    return payload


__all__ = [
    "audit_full_roles", "branch_allowlist_sha256", "build_coupling_source",
    "build_switch_sidecar_for_source", "calibrate_train_scales",
    "coupling_branch_allowlist", "finalize_base_role", "finalize_coupling_role",
    "freeze_switch_calibration", "iter_base_edits",
]
