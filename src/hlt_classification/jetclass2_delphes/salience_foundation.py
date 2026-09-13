"""Compact JetClass2 salience assignments and persistent-HLT foundation."""
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
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    CANDIDATES, matcher_spec, validate_matcher_spec,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    reference_pairing_from_matrices,
)
from .contracts import artifact, relative_file, validate
from .inputs import build_inputs, input_contract
from .inventory import validate_inventory
from .provenance import source_record, validate_source_record
from .reader import DatasetReader, Jet, Particles
from .salience_views import build_view, match_particles, pairing_matrices, view_contract
from .splits import file_population, is_subset_profile, validate_splits


def assignment_source() -> dict:
    return source_record(
        *[f"src/hlt_classification/jetclass2_delphes/{name}.py" for name in (
            "salience_foundation", "salience_views", "reader", "schema",
            "selection", "splits", "split_registry", "inputs", "contracts",
        )],
        "src/hlt_classification/scouting/hcwdl_fullcard_salience_contracts.py",
        "src/hlt_classification/scouting/hcwdl_fullcard_salience_matcher.py",
        "src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py",
        "src/hlt_classification/scouting/highcov_data.py",
        "src/hlt_classification/scouting/highcov_features.py",
    )


def build_foundation_spec(inventory: dict, splits: dict, candidate: str) -> dict:
    if candidate not in CANDIDATES or not is_subset_profile(splits):
        raise ValueError("Salience foundation requires a registered candidate/profile")
    inv_hash = validate_inventory(inventory)
    split_hash = validate_splits(splits, inventory)
    maximum = max(
        max(row["max_selected_particles"].values()) for row in inventory["files"]
    )
    inputs = input_contract(capacity=max(16, ((maximum + 15) // 16) * 16))
    tasks = [
        dict(
            file_index=index, path=row["path"], role=group["role"],
            rows=file_population(splits, row, group["role"])[0],
        )
        for index, (row, group) in enumerate(zip(
            inventory["files"], splits["groups"], strict=True,
        ))
        if group["role"] in {"train", "validation"}
    ]
    tasks = [task for task in tasks if task["rows"] > 0]
    spec = matcher_spec(candidate)
    validate_matcher_spec(spec)
    return artifact(
        "SALIENCE_FOUNDATION_SPEC", inventory=inventory, splits=splits,
        candidate=candidate, matcher=spec, inputs=inputs,
        views=view_contract(candidate),
        parents={"inventory": inv_hash, "splits": split_hash,
                 "matcher": spec["content_hash"]},
        assignment_tasks=tasks, ordinary_roles=["train", "validation"],
        persistent_hlt=True, pure_offline_u000=False,
        final_test_accessed=False, durable_particle_views=False,
        rolling_resume=False,
    )


def validate_foundation_spec(spec: dict) -> str:
    digest = validate(spec, "SALIENCE_FOUNDATION_SPEC")
    expected = build_foundation_spec(
        spec["inventory"], spec["splits"], spec["candidate"],
    )
    if spec != expected:
        raise ValueError("Salience foundation semantics differ")
    return digest


def _task(spec: dict, file_index: int) -> dict:
    rows = [row for row in spec["assignment_tasks"] if row["file_index"] == file_index]
    if len(rows) != 1:
        raise PermissionError("Assignment index is not an ordinary-role task")
    return rows[0]


def build_assignment_shard(
    spec: dict, *, data_root: Path, output_root: Path, file_index: int,
) -> dict:
    if Path(output_root).resolve().is_relative_to(Path(data_root).resolve()):
        raise ValueError("Assignment output cannot be inside raw snapshot")
    parent = validate_foundation_spec(spec)
    task = _task(spec, file_index)
    reader = DatasetReader(
        data_root, spec["inventory"], spec["splits"], role=task["role"],
        include_offline=True, file_paths=(task["path"],),
    )
    started = time.monotonic()
    identities, offsets, mappings = [], [0], []
    counts = {"hlt": 0, "offline": 0, "matched": 0,
              "unmatched_hlt": 0, "unused_offline": 0}
    sum_dr = 0.
    maximum_dr = 0.
    weighted_dr = 0.
    salience_mass = 0.
    for number, jet in enumerate(reader, 1):
        matrices = pairing_matrices(jet.hlt, jet.offline, spec["candidate"])
        mapping = match_particles(jet.hlt, jet.offline, spec["candidate"])
        for endpoint in (jet.hlt, jet.offline):
            build_inputs(endpoint, capacity=spec["inputs"]["capacity"])
        common = np.flatnonzero(mapping >= 0)
        columns = mapping[common]
        qdr = matrices["qdr"][common, columns].astype(np.float64)
        dr = qdr * float(spec["matcher"]["dr_quantum"])
        # This diagnostic deliberately uses one candidate-independent linear
        # endpoint-pT measure so it is a valid cross-candidate tie-breaker.
        hlt_pt = np.hypot(jet.hlt.values[:, 0], jet.hlt.values[:, 1]).astype(np.float64)
        offline_pt = np.hypot(jet.offline.values[:, 0], jet.offline.values[:, 1]).astype(np.float64)
        weights = (
            hlt_pt[common] / hlt_pt.sum(dtype=np.float64)
            + offline_pt[columns] / offline_pt.sum(dtype=np.float64)
        )
        identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
        offsets.append(offsets[-1] + len(mapping))
        mappings.append(mapping.astype(np.int32))
        counts["hlt"] += len(jet.hlt)
        counts["offline"] += len(jet.offline)
        counts["matched"] += len(common)
        counts["unmatched_hlt"] += len(jet.hlt) - len(common)
        counts["unused_offline"] += len(jet.offline) - len(common)
        sum_dr += float(dr.sum())
        maximum_dr = max(maximum_dr, float(dr.max(initial=0.)))
        weighted_dr += float(np.sum(weights * dr))
        salience_mass += float(weights.sum())
        if number % 1000 == 0:
            print(
                f"JC2-SALIENCE phase=assign candidate={spec['candidate']} "
                f"file_index={file_index} rows={number}/{task['rows']} "
                f"seconds={time.monotonic()-started:.1f}", flush=True,
            )
    if len(identities) != task["rows"]:
        raise ValueError("Assignment shard row coverage differs")
    payload = {
        "identities": np.asarray(identities, np.uint8).reshape(-1, 32),
        "offsets": np.asarray(offsets, np.int64),
        "mapping": np.concatenate(mappings) if mappings else np.empty(0, np.int32),
    }
    path = Path(output_root) / "assignments" / f"{file_index:04d}.npz"
    atomic_publish_bytes(path, deterministic_npz_bytes(payload))
    report = artifact(
        "SALIENCE_ASSIGNMENT_SHARD", foundation_sha256=parent,
        matcher_sha256=spec["matcher"]["content_hash"],
        candidate=spec["candidate"], file_index=file_index, role=task["role"],
        source_path=task["path"], rows=len(identities),
        array_path=path.relative_to(output_root).as_posix(),
        array_sha256=sha256_file(path), array_bytes=path.stat().st_size,
        particle_counts=counts,
        mean_selected_dr=sum_dr / counts["matched"] if counts["matched"] else None,
        maximum_dr=maximum_dr if counts["matched"] else None,
        pt_salience_weighted_selected_dr=(
            weighted_dr / salience_mass if salience_mass else None
        ),
        diagnostic_weight="sum_of_endpoint_linear_pt_shares_candidate_independent",
        diagnostic_weight_mass=salience_mass,
        diagnostic_weighted_selected_dr_numerator=weighted_dr,
        pairing_provenance="validity_only_not_correspondence_confidence",
        producer=assignment_source(), final_test_accessed=False,
    )
    write_immutable_json(path.with_suffix(".json"), report)
    return report


def load_assignments(spec: dict, *, root: Path, file_index: int) -> tuple[dict, dict]:
    parent = validate_foundation_spec(spec)
    task = _task(spec, file_index)
    report = load_json(Path(root) / "assignments" / f"{file_index:04d}.json")
    validate(report, "SALIENCE_ASSIGNMENT_SHARD")
    validate_source_record(report["producer"])
    if (
        report["foundation_sha256"] != parent
        or report["matcher_sha256"] != spec["matcher"]["content_hash"]
        or report["candidate"] != spec["candidate"]
        or report["file_index"] != file_index or report["role"] != task["role"]
        or report["source_path"] != task["path"] or report["rows"] != task["rows"]
        or report["final_test_accessed"] is not False
        or report.get("diagnostic_weight") != "sum_of_endpoint_linear_pt_shares_candidate_independent"
    ):
        raise ValueError("Salience assignment shard lineage differs")
    path = relative_file(Path(root), report["array_path"])
    if path.stat().st_size != report["array_bytes"] or sha256_file(path) != report["array_sha256"]:
        raise ValueError("Salience assignment bytes differ")
    arrays = load_npz_arrays(path)
    if set(arrays) != {"identities", "offsets", "mapping"}:
        raise ValueError("Salience assignment arrays differ")
    identities, offsets, mapping = arrays["identities"], arrays["offsets"], arrays["mapping"]
    if (
        identities.dtype != np.uint8 or identities.shape != (task["rows"], 32)
        or offsets.dtype != np.int64 or offsets.shape != (task["rows"] + 1,)
        or mapping.dtype != np.int32 or mapping.ndim != 1 or offsets[0] != 0
        or offsets[-1] != len(mapping) or np.any(np.diff(offsets) <= 0)
        or np.any(mapping < -1) or len(np.unique(identities, axis=0)) != len(identities)
    ):
        raise ValueError("Salience assignment layout differs")
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        row = mapping[int(start):int(stop)]
        accepted = row[row >= 0]
        if len(np.unique(accepted)) != len(accepted):
            raise ValueError("Salience assignment reuses an offline particle")
    return report, arrays


def audit_sample(spec: dict, *, data_root: Path, rows_per_file: int = 8) -> dict:
    parent = validate_foundation_spec(spec)
    if rows_per_file < 1:
        raise ValueError("Positive sample size required")
    counts = {"rows": 0, "hlt": 0, "offline": 0, "matched": 0,
              "reference_rows": 0}
    seen: set[str] = set()
    maximum_dr = 0.
    for task in spec["assignment_tasks"]:
        reader = DatasetReader(
            data_root, spec["inventory"], spec["splits"], role=task["role"],
            include_offline=True, file_paths=(task["path"],), step_size=64,
        )
        for jet in islice(reader, rows_per_file):
            if jet.identity in seen:
                raise ValueError("Duplicate canonical row identity")
            seen.add(jet.identity)
            matrices = pairing_matrices(jet.hlt, jet.offline, spec["candidate"])
            mapping = match_particles(jet.hlt, jet.offline, spec["candidate"])
            common = np.flatnonzero(mapping >= 0)
            if len(common) != min(len(jet.hlt), len(jet.offline)):
                raise AssertionError("Smaller-side coverage differs")
            maximum_dr = max(
                maximum_dr,
                float(matrices["qdr"][common, mapping[common]].max(initial=0)) * 1e-7,
            )
            if counts["reference_rows"] < 8:
                hlt = Particles(jet.hlt.values[:min(4, len(jet.hlt))])
                offline = Particles(jet.offline.values[:min(4, len(jet.offline))])
                small = pairing_matrices(hlt, offline, spec["candidate"])
                np.testing.assert_array_equal(
                    match_particles(hlt, offline, spec["candidate"]),
                    reference_pairing_from_matrices(**small),
                )
                counts["reference_rows"] += 1
            lengths = []
            for u, f in ((0, 0), (Fraction(1, 3), 0), (Fraction(2, 3), 0),
                         (1, 0), (1, Fraction(1, 2)), (1, 1)):
                view = build_view(
                    jet, u=Fraction(u), f=Fraction(f),
                    candidate=spec["candidate"], mapping=mapping,
                )
                build_inputs(view, capacity=spec["inputs"]["capacity"])
                if not f:
                    lengths.append(len(view))
            if lengths != sorted(lengths, reverse=True) or lengths[-1] != len(jet.hlt):
                raise AssertionError("Persistent-HLT U support is not nested")
            if lengths[0] != max(len(jet.hlt), len(jet.offline)):
                raise AssertionError("Persistent U000 cardinality differs")
            hlt_only = Jet(jet.identity, jet.label, jet.hlt, None)
            if build_view(
                hlt_only, u=Fraction(1), f=Fraction(1), candidate=spec["candidate"],
            ) is not jet.hlt:
                raise AssertionError("D000 endpoint differs")
            counts["rows"] += 1
            counts["hlt"] += len(jet.hlt)
            counts["offline"] += len(jet.offline)
            counts["matched"] += len(common)
    if not counts["rows"] or not counts["reference_rows"]:
        raise ValueError("Empty salience acceptance sample")
    return artifact(
        "SALIENCE_SAMPLE_AUDIT", foundation_sha256=parent,
        matcher_sha256=spec["matcher"]["content_hash"], candidate=spec["candidate"],
        counts=counts, rows_per_file=rows_per_file,
        complete_smaller_side_coverage=True, persistent_hlt_endpoints=True,
        exhaustive_reference="bounded_native_prefixes_max_side_at_most_four",
        maximum_selected_dr=maximum_dr, passed=True,
        final_test_accessed=False,
    )


def audit_assignment_sample(
    spec: dict, *, root: Path, data_root: Path, rows_per_file: int = 2,
) -> dict:
    """Recompute a bounded sample from raw rows after durable publication."""
    parent = validate_foundation_spec(spec)
    if type(rows_per_file) is not int or not 1 <= rows_per_file <= 8:
        raise ValueError("Assignment recomputation sample is outside bounds")
    checked = 0
    for task in spec["assignment_tasks"]:
        _, arrays = load_assignments(spec, root=root, file_index=task["file_index"])
        reader = DatasetReader(
            data_root, spec["inventory"], spec["splits"], role=task["role"],
            include_offline=True, file_paths=(task["path"],), step_size=32,
        )
        for row, jet in enumerate(islice(reader, rows_per_file)):
            identity = np.frombuffer(bytes.fromhex(jet.identity), np.uint8)
            start, stop = arrays["offsets"][row:row + 2]
            stored = arrays["mapping"][int(start):int(stop)]
            expected = match_particles(jet.hlt, jet.offline, spec["candidate"])
            if (not np.array_equal(identity, arrays["identities"][row])
                    or not np.array_equal(stored, expected)):
                raise ValueError("Durable salience assignment differs from raw recomputation")
            checked += 1
    expected_checked = sum(
        min(rows_per_file, task["rows"]) for task in spec["assignment_tasks"]
    )
    if checked != expected_checked:
        raise ValueError("Assignment recomputation coverage differs")
    return artifact(
        "SALIENCE_ASSIGNMENT_AUDIT", foundation_sha256=parent,
        matcher_sha256=spec["matcher"]["content_hash"], candidate=spec["candidate"],
        rows_per_file=rows_per_file, files=len(spec["assignment_tasks"]),
        checked_rows=checked, exact_mapping_recomputed=True, passed=True,
        final_test_accessed=False,
    )


def build_foundation_lock(spec: dict, root: Path, *, publish: bool = True) -> dict:
    parent = validate_foundation_spec(spec)
    shards, total_bytes = [], 0
    weighted_numerator, salience_mass = 0., 0.
    totals = {"hlt": 0, "offline": 0, "matched": 0,
              "unmatched_hlt": 0, "unused_offline": 0}
    for task in spec["assignment_tasks"]:
        report, _ = load_assignments(spec, root=root, file_index=task["file_index"])
        shards.append({"file_index": task["file_index"], "content_hash": report["content_hash"]})
        total_bytes += report["array_bytes"]
        weighted_numerator += report["diagnostic_weighted_selected_dr_numerator"]
        salience_mass += report["diagnostic_weight_mass"]
        for key in totals:
            totals[key] += report["particle_counts"][key]
    audit = load_json(Path(root) / "assignment_audit.json")
    validate(audit, "SALIENCE_ASSIGNMENT_AUDIT")
    if (audit["foundation_sha256"] != parent or audit["passed"] is not True
            or audit["exact_mapping_recomputed"] is not True):
        raise ValueError("Salience assignment audit lineage differs")
    result = artifact(
        "SALIENCE_FOUNDATION_LOCK", foundation_sha256=parent,
        matcher_sha256=spec["matcher"]["content_hash"], candidate=spec["candidate"],
        shards=shards, durable_array_bytes=total_bytes, particle_counts=totals,
        pt_salience_weighted_selected_dr=(
            weighted_numerator / salience_mass if salience_mass else None
        ),
        diagnostic_weight="sum_of_endpoint_linear_pt_shares_candidate_independent",
        complete_smaller_side_coverage=True, persistent_hlt=True,
        assignment_audit_sha256=audit["content_hash"],
        particle_views_persisted=False, final_test_accessed=False,
    )
    if publish:
        write_immutable_json(Path(root) / "foundation_lock.json", result)
    return result


def authenticate_preparation(spec: dict, root: Path) -> dict:
    validate_foundation_spec(spec)
    expected = assignment_source()
    for task in spec["assignment_tasks"]:
        report, _ = load_assignments(spec, root=root, file_index=task["file_index"])
        if report["producer"]["file_sha256"] != expected["file_sha256"]:
            raise ValueError("Salience assignment producer differs")
    audit = load_json(Path(root) / "sample_audit.json")
    validate(audit, "SALIENCE_SAMPLE_AUDIT")
    if (
        audit["foundation_sha256"] != spec["content_hash"]
        or audit["matcher_sha256"] != spec["matcher"]["content_hash"]
        or audit["candidate"] != spec["candidate"] or audit["passed"] is not True
        or audit["persistent_hlt_endpoints"] is not True
        or audit["final_test_accessed"] is not False
    ):
        raise ValueError("Salience sample acceptance differs")
    assignment_audit = load_json(Path(root) / "assignment_audit.json")
    validate(assignment_audit, "SALIENCE_ASSIGNMENT_AUDIT")
    if (assignment_audit["foundation_sha256"] != spec["content_hash"]
            or assignment_audit["passed"] is not True
            or assignment_audit["final_test_accessed"] is not False):
        raise ValueError("Salience assignment recomputation audit differs")
    lock = load_json(Path(root) / "foundation_lock.json")
    validate(lock, "SALIENCE_FOUNDATION_LOCK")
    expected_lock = build_foundation_lock(spec, root, publish=False)
    if lock != expected_lock:
        raise ValueError("Salience foundation lock differs")
    return lock


__all__ = [
    "assignment_source", "audit_assignment_sample", "audit_sample", "authenticate_preparation",
    "build_assignment_shard", "build_foundation_lock", "build_foundation_spec",
    "load_assignments", "validate_foundation_spec",
]
