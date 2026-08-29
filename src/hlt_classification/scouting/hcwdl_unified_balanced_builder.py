"""Build and audit balanced HCWDL-UB sidecars over immutable residual bases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .hcwdl_unified_balanced import balanced_switch_placements
from .hcwdl_unified_balanced_cache import (
    publish_balanced_manifest, publish_balanced_sidecar,
)
from .hcwdl_upper_builder import _prepared_partitions, _selected_source_chunks
from .hcwdl_upper_cache import load_base_shard
from .hcwdl_upper_coupling import ResidualEdit
from .hcwdl_assignment_store import open_assignment_store
from .selective_assignment import RowSelection
from .splits import role_records

# Historical injection seam retained for tests and old callers; the dispatch
# now accepts both established confidence manifests and the new validity-only
# full-cardinality contract.
DenseAssignmentStore = open_assignment_store


def _base_edit_row(arrays, row: int) -> tuple[ResidualEdit, ...]:
    start = int(arrays["row_offsets"][row]); stop = int(arrays["row_offsets"][row + 1])
    return tuple(ResidualEdit(
        int(arrays["edit_kind"][index]),
        int(arrays["source_native_offline_index"][index]),
        int(arrays["target_hlt_slot"][index]),
        int(arrays["target_kind"][index]),
        int(arrays["target_native_offline_index"][index]),
        int(arrays["cost_q"][index]), int(arrays["mass_q"][index]),
    ) for index in range(start, stop))


def build_balanced_sidecar_for_source(
    *, split_manifest: Mapping[str, Any], selection_manifest: Mapping[str, Any],
    assignment_manifest: str | Path, data_root: str | Path, role: str,
    source_index: int, base_metadata_path: str | Path,
    switch_config_sha256: str, output_base: str | Path,
    producer_commit: str, step_size: int = 4096,
):
    """Recompute endpoint strata once and attach frozen coordinates to one base shard."""

    if role not in {"train", "validation"}:
        raise PermissionError("HCWDL-UB balanced switching supports train/validation only")
    base_metadata, base = load_base_shard(base_metadata_path)
    records = role_records(split_manifest, role)
    if not 0 <= source_index < len(records):
        raise ValueError("HCWDL-UB source index differs")
    record = records[source_index]
    if base_metadata["role"] != role or base_metadata["source_path"] != record.path:
        raise ValueError("HCWDL-UB balanced source/base lineage differs")
    split_hash = str(split_manifest["content_hash"])
    selection = RowSelection(
        selection_manifest, role=role, split_manifest_sha256=split_hash,
    )
    assignments = DenseAssignmentStore(assignment_manifest)
    base_lookup = {int(entry): row for row, entry in enumerate(base["entries"])}
    observed: set[int] = set(); placement_rows = [None] * len(base["entries"])
    for _, source_path, entries, arrays in _selected_source_chunks(
        split_manifest=split_manifest, selection=selection,
        assignments=assignments, data_root=data_root, role=role,
        source_index=source_index, step_size=step_size,
    ):
        if source_path != record.path:
            raise ValueError("HCWDL-UB selected source identity differs")
        mapping, _ = assignments.join(source_path, entries)
        partitions, _, _ = _prepared_partitions(arrays, mapping)
        for local_row, entry in enumerate(entries):
            entry_value = int(entry)
            if entry_value not in base_lookup or entry_value in observed:
                raise ValueError("HCWDL-UB balanced selected/base entries differ")
            base_row = base_lookup[entry_value]
            edits = _base_edit_row(base, base_row)
            placement_rows[base_row] = balanced_switch_placements(
                edits, partition=partitions[local_row],
                identity_key=f"{source_path}::tree::{entry_value}",
                switch_config_sha256=switch_config_sha256,
            )
            observed.add(entry_value)
    if observed != set(base_lookup) or any(row is None for row in placement_rows):
        raise ValueError("HCWDL-UB balanced sidecar coverage differs")
    return publish_balanced_sidecar(
        output_base, base_metadata_path=base_metadata_path,
        placement_rows=placement_rows,  # type: ignore[arg-type]
        switch_config_sha256=switch_config_sha256,
        producer_commit=producer_commit,
    )


def finalize_balanced_role(
    *, role: str, base_manifest_path: str | Path,
    sidecar_root: str | Path, output: str | Path,
    switch_config_sha256: str,
) -> dict[str, Any]:
    from hlt_classification.data.cache_contracts import load_json

    base = load_json(base_manifest_path)
    sidecars = [
        Path(sidecar_root) / role / f"shard_{index:04d}.json"
        for index in range(len(base["shards"]))
    ]
    return publish_balanced_manifest(
        output, role=role, base_manifest_path=base_manifest_path,
        sidecar_paths=sidecars, switch_config_sha256=switch_config_sha256,
    )


__all__ = ["build_balanced_sidecar_for_source", "finalize_balanced_role"]
