"""Compact assignment publication and new-dataset foundation acceptance.

No particle views, representation targets, or optimizer states are persisted.
Array shards contain only row identities, offsets, and integer assignments.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import islice
from pathlib import Path
import time

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_matcher import reference_pairing_from_matrices
from .contracts import artifact, relative_file, validate
from .inputs import build_inputs, input_contract
from .inventory import validate_inventory
from .reader import DatasetReader, Jet, Particles
from .splits import validate_splits, is_subset_profile, file_population
from .views import build_view, match_particles, pairing_matrices, view_contract
from .provenance import source_record, validate_source_record


def assignment_source() -> dict:
    return source_record(*[f"src/hlt_classification/jetclass2_delphes/{name}.py" for name in
                          ("foundation", "reader", "schema", "selection", "splits", "split_registry", "inputs", "views", "contracts")],
                         "src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py")


def build_foundation_spec(inventory: dict, splits: dict) -> dict:
    inv_hash = validate_inventory(inventory)
    split_hash = validate_splits(splits, inventory)
    # Count metadata was inspected before sealing model access to final test.
    # Round UP; never reuse the historical 128/200-token truncation.
    maximum = max(max(row["max_selected_particles"].values()) for row in inventory["files"])
    inputs = input_contract(capacity=max(16, ((maximum + 15) // 16) * 16))
    tasks = [dict(file_index=i, path=r["path"], role=g["role"], rows=file_population(splits, r, g["role"])[0])
             for i, (r, g) in enumerate(zip(inventory["files"], splits["groups"]))
             if g["role"] in {"train", "validation"}]
    if is_subset_profile(splits):
        tasks = [t for t in tasks if t["rows"] > 0]
    return artifact(
        "FOUNDATION_SPEC", version=2 if is_subset_profile(splits) else 1,
        inventory=inventory, splits=splits, inputs=inputs, views=view_contract(),
        parents={"inventory": inv_hash, "splits": split_hash}, assignment_tasks=tasks,
        ordinary_roles=["train", "validation"], final_test_accessed=False,
        durable_particle_views=False, rolling_resume=False,
    )


def validate_foundation_spec(spec: dict) -> str:
    digest = validate(spec, "FOUNDATION_SPEC", version=2 if is_subset_profile(spec["splits"]) else 1)
    expected = build_foundation_spec(spec["inventory"], spec["splits"])
    if spec != expected:
        raise ValueError("Foundation semantics differ")
    return digest


def _task(spec: dict, file_index: int) -> dict:
    rows = [r for r in spec["assignment_tasks"] if r["file_index"] == file_index]
    if len(rows) != 1:
        raise PermissionError("Assignment index is not a registered train/validation task")
    return rows[0]


def build_assignment_shard(spec: dict, *, data_root: Path, output_root: Path, file_index: int) -> dict:
    if Path(output_root).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError("Assignment output cannot be inside raw snapshot")
    parent = validate_foundation_spec(spec)
    task = _task(spec, file_index)
    started = time.monotonic()
    reader = DatasetReader(data_root, spec["inventory"], spec["splits"], role=task["role"],
                           include_offline=True, file_paths=(task["path"],))
    identities, offsets, mappings = [], [0], []
    particle_counts = {"hlt": 0, "offline": 0, "matched": 0}
    worst_dr, sum_dr = 0., 0.
    for i, jet in enumerate(reader, 1):
        mapping = match_particles(jet.hlt, jet.offline)
        # All native particles and both exact endpoints are checked by workers,
        # not only the small synthetic acceptance sample.
        for particle_set in (jet.hlt, jet.offline):
            build_inputs(particle_set, capacity=spec["inputs"]["capacity"])
        identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
        offsets.append(offsets[-1] + len(mapping))
        mappings.append(mapping.astype(np.int32))
        common = np.flatnonzero(mapping >= 0)
        matrices = pairing_matrices(jet.hlt, jet.offline)
        dr = matrices["qdr"][common, mapping[common]].astype(np.float64) * 1e-7
        worst_dr = max(worst_dr, float(dr.max(initial=0)))
        sum_dr += float(dr.sum())
        particle_counts["hlt"] += len(jet.hlt)
        particle_counts["offline"] += len(jet.offline)
        particle_counts["matched"] += len(common)
        if i % 1000 == 0:
            print(f"JC2 phase=assign file_index={file_index} rows={i}/{task['rows']} seconds={time.monotonic()-started:.1f}", flush=True)
    if len(identities) != task["rows"]:
        raise ValueError("Assignment shard row coverage differs")
    payload = {
        "identities": np.array(identities, np.uint8).reshape(-1, 32),
        "offsets": np.array(offsets, np.int64),
        "mapping": np.concatenate(mappings) if mappings else np.empty(0, np.int32),
    }
    path = Path(output_root) / "assignments" / f"{file_index:04d}.npz"
    atomic_publish_bytes(path, deterministic_npz_bytes(payload))
    report = artifact(
        "ASSIGNMENT_SHARD", foundation_sha256=parent, file_index=file_index,
        role=task["role"], source_path=task["path"], rows=len(identities),
        array_path=path.relative_to(output_root).as_posix(), array_sha256=sha256_file(path),
        array_bytes=path.stat().st_size, particle_counts=particle_counts,
        mean_selected_dr=sum_dr / particle_counts["matched"] if particle_counts["matched"] else None,
        maximum_dr=worst_dr if particle_counts["matched"] else None,
        producer=assignment_source(),
        final_test_accessed=False,
    )
    write_immutable_json(path.with_suffix(".json"), report)
    return report


def load_assignments(spec: dict, *, root: Path, file_index: int) -> tuple[dict, dict]:
    parent = validate_foundation_spec(spec)
    task = _task(spec, file_index)
    report = load_json(Path(root) / "assignments" / f"{file_index:04d}.json")
    validate(report, "ASSIGNMENT_SHARD")
    validate_source_record(report["producer"])
    if (report["foundation_sha256"] != parent or report["file_index"] != file_index
            or report["role"] != task["role"] or report["source_path"] != task["path"]
            or report["rows"] != task["rows"] or report["final_test_accessed"] is not False):
        raise ValueError("Assignment shard lineage/coverage differs")
    path = relative_file(Path(root), report["array_path"])
    if path.stat().st_size != report["array_bytes"] or sha256_file(path) != report["array_sha256"]:
        raise ValueError("Assignment array checksum differs")
    arrays = load_npz_arrays(path)
    ids, offsets, mapping = arrays["identities"], arrays["offsets"], arrays["mapping"]
    if (ids.dtype != np.uint8 or ids.shape != (task["rows"], 32)
            or offsets.dtype != np.int64 or offsets.shape != (task["rows"] + 1,)
            or mapping.dtype != np.int32 or mapping.ndim != 1 or offsets[0] != 0
            or offsets[-1] != len(mapping) or np.any(np.diff(offsets) <= 0)
            or np.any(mapping < -1)):
        raise ValueError("Assignment array layout differs")
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("Duplicate assignment row identities")
    return report, arrays


def build_foundation_lock(spec: dict, root: Path) -> dict:
    parent = validate_foundation_spec(spec)
    shards = []
    total_bytes = 0
    for task in spec["assignment_tasks"]:
        report, _ = load_assignments(spec, root=root, file_index=task["file_index"])
        shards.append(dict(file_index=task["file_index"], content_hash=report["content_hash"]))
        total_bytes += report["array_bytes"]
    result = artifact(
        "FOUNDATION_LOCK", foundation_sha256=parent, shards=shards,
        durable_array_bytes=total_bytes, particle_views_persisted=False,
        final_test_accessed=False, production_runtime_accepted=False,
    )
    write_immutable_json(Path(root) / "foundation_lock.json", result)
    return result


def audit_sample(spec: dict, *, data_root: Path, rows_per_file: int = 8) -> dict:
    """Bounded native-data audit spanning EVERY ordinary-role file, not final test."""
    parent = validate_foundation_spec(spec)
    if rows_per_file < 1:
        raise ValueError("Positive sample size required")
    seen = set()
    counts = dict(rows=0, hlt_particles=0, offline_particles=0, selected_pairs=0,
                  zero_error_hlt=0, zero_error_offline=0, nonzero_value_zero_error_hlt=0,
                  nonzero_value_zero_error_offline=0, reference_rows=0)
    maximum_dr = 0.
    max_lengths = dict(hlt=0, offline=0)
    for task in spec["assignment_tasks"]:
        reader = DatasetReader(data_root, spec["inventory"], spec["splits"], role=task["role"],
                               include_offline=True, file_paths=(task["path"],), step_size=64)
        for jet in islice(reader, rows_per_file):
            if jet.identity in seen:
                raise ValueError("Duplicate canonical row identity")
            seen.add(jet.identity)
            counts["rows"] += 1
            mapping = match_particles(jet.hlt, jet.offline)
            common = np.flatnonzero(mapping >= 0)
            matrices = pairing_matrices(jet.hlt, jet.offline)
            maximum_dr = max(maximum_dr, float(matrices["qdr"][common, mapping[common]].max(initial=0)) * 1e-7)
            counts["selected_pairs"] += len(common)
            # Small native prefixes give bounded exhaustive solver checks.
            if counts["reference_rows"] < 8:
                h, o = Particles(jet.hlt.values[:4]), Particles(jet.offline.values[:4])
                np.testing.assert_array_equal(match_particles(h, o), reference_pairing_from_matrices(**pairing_matrices(h, o)))
                counts["reference_rows"] += 1
            for view, particles in (("hlt", jet.hlt), ("offline", jet.offline)):
                counts[f"{view}_particles"] += len(particles)
                max_lengths[view] = max(max_lengths[view], len(particles))
                errors = particles.values[:, [11, 13]]
                counts[f"zero_error_{view}"] += int(np.count_nonzero(errors == 0))
                counts[f"nonzero_value_zero_error_{view}"] += int(np.count_nonzero((errors == 0) & (particles.values[:, [10, 12]] != 0)))
            for u, f in ((0, 0), (Fraction(1, 2), 0), (1, 0), (1, Fraction(1, 2)), (1, 1)):
                build_inputs(build_view(jet, u=Fraction(u), f=Fraction(f), mapping=mapping), capacity=spec["inputs"]["capacity"])
            if build_view(jet, u=Fraction(0), f=Fraction(0)) is not jet.offline:
                raise AssertionError("U000 endpoint differs")
            hlt_only = Jet(jet.identity, jet.label, jet.hlt, None)
            if build_view(hlt_only, u=Fraction(1), f=Fraction(1)) is not jet.hlt:
                raise AssertionError("D000 endpoint differs")
        print(f"JC2 phase=sample file_index={task['file_index']} checked_rows={counts['rows']}", flush=True)
    if counts["rows"] == 0:
        raise ValueError("Empty acceptance sample")
    return artifact(
        "SAMPLE_AUDIT", foundation_sha256=parent, counts=counts,
        rows_per_file=rows_per_file, files=len(spec["assignment_tasks"]), max_lengths=max_lengths,
        maximum_selected_dr=maximum_dr, exhaustive_reference="first_four_native_particles_not_full_jets",
        all_ordinary_rows_audited=False, exact_endpoint_checks=True, passed=True,
        zero_error_counts_unit="individual_d0err_and_dzerr_values",
        final_test_accessed=False,
    )
