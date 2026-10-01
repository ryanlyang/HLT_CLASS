"""Exact proxy/offline joins and compact full-cardinality assignment bank."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
import hashlib
import multiprocessing
import os

import awkward as ak
import numpy as np
import uproot

from hlt_classification.cms2jc2_production import output
from hlt_classification.cms2jc2_production.campaign import imported, validate_study
from hlt_classification.cms2jc2_production.contracts import load_json, safe as source_safe
from hlt_classification.cms2jc2_response.bridge import JC2_FIELDS, Particles, from_jc2
from hlt_classification.cms2jc2_response.generation_benchmark_data import BRANCHES
from hlt_classification.data.cache_contracts import atomic_publish_bytes, sha256_file
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.jetclass2_delphes.selection import selected_mask
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import matcher_spec

from .contracts import artifact, file_ref, validate, validate_file_ref, write_json
from .inputs import build_inputs, input_contract
from .release import load_bank, validate_release
from .views import CANDIDATE, build_view, match_particles, view_contract

ASSIGNMENT_FIELDS = {"identity", "offsets", "mapping"}


@dataclass(frozen=True)
class PairedRow:
    ordinal: int
    identity: str
    role: str
    label: int
    proxy: Particles
    offline: Particles


def _proxy_particles(values: dict, row: int) -> Particles:
    lo, hi = map(int, values["offsets"][row:row + 2])
    return Particles(
        *(values[name][lo:hi] for name in ("p4", "charge", "category", "tracking", "valid")),
        tuple(f"proxy_native:{index}" for index in range(hi - lo)),
    )


def _role_code(role: str) -> int:
    if role not in {"train", "validation"}:
        raise PermissionError("Proxy-ladder paired reader has no final-test capability")
    return 0 if role == "train" else 1


def iter_paired(
    release: dict, *, release_root: Path, role: str,
    source_file_index: int | tuple[int, ...] | None = None,
):
    """Yield exact identity-joined physical endpoints in frozen release order."""
    validate_release(release, root=release_root)
    arrays = load_bank(release, root=release_root)
    selected = np.flatnonzero(arrays["role"] == _role_code(role))
    role_start = int(selected[0]) if len(selected) else 0
    if source_file_index is not None:
        wanted = (source_file_index,) if type(source_file_index) is int else tuple(source_file_index)
        if (
            not wanted or len(set(wanted)) != len(wanted)
            or any(type(index) is not int or not 0 <= index < len(release["source_files"]) for index in wanted)
        ):
            raise ValueError("Invalid proxy-ladder source-file index")
        selected = selected[np.isin(arrays["source_file"][selected], wanted)]
    study_root = Path(release["study_root"])
    offline_root = Path(release["offline_root"])
    study = load_json(study_root / "study_spec.json")
    validate_study(study)
    inventory = imported(study, "inventory")
    if inventory["content_hash"] != study["population"]["parents"]["inventory"]:
        raise ValueError("Proxy-ladder offline inventory lineage differs")
    cursor = 0
    while cursor < len(selected):
        source_number = int(arrays["source_file"][selected[cursor]])
        stop = cursor + 1
        while stop < len(selected) and int(arrays["source_file"][selected[stop]]) == source_number:
            stop += 1
        indices = selected[cursor:stop]
        source = release["source_files"][source_number]
        path = offline_root / source["path"]
        if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
            raise ValueError("Proxy-ladder offline source bytes differ")
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            if key != source["tree_key"] or tree.num_entries != source["raw_entries"]:
                raise ValueError("Proxy-ladder offline ROOT tree identity differs")
            block_cursor = 0
            while block_cursor < len(indices):
                block_number = int(arrays["proxy_block"][indices[block_cursor]])
                block_stop = block_cursor + 1
                while (block_stop < len(indices)
                       and int(arrays["proxy_block"][indices[block_stop]]) == block_number):
                    block_stop += 1
                group = indices[block_cursor:block_stop]
                block = release["proxy_blocks"][block_number]
                block_path = source_safe(study_root, block["path"])
                if block_path.stat().st_size != block["bytes"] or sha256_file(block_path) != block["sha256"]:
                    raise ValueError("Proxy-ladder proxy block bytes differ")
                proxy_values = output.arrays(block_path)
                entries = arrays["entry"][group]
                start, end = int(entries.min()), int(entries.max()) + 1
                raw = tree.arrays(
                    ["jet_label", "hlt_matched", *BRANCHES],
                    entry_start=start, entry_stop=end, library="ak", how=dict,
                )
                keep, labels = selected_mask(
                    ak.to_numpy(raw["jet_label"]), ak.to_numpy(raw["hlt_matched"]),
                    source["source"], inventory["selection"],
                )
                for index in group:
                    entry = int(arrays["entry"][index])
                    local = entry - start
                    if not keep[local]:
                        raise ValueError("Frozen proxy row is no longer eligible in offline source")
                    count = int(raw["jet_nparticles"][local])
                    columns = {name: ak.to_numpy(raw["part_" + name][local]) for name in JC2_FIELDS}
                    if any(len(value) != count for value in columns.values()):
                        raise ValueError("Proxy-ladder offline jagged counts differ")
                    identity = bytes(arrays["identity"][index]).hex()
                    proxy_row = int(arrays["proxy_row"][index])
                    if bytes(proxy_values["jet_identity"][proxy_row]).hex() != identity:
                        raise ValueError("Proxy-ladder release/proxy identity join differs")
                    yield PairedRow(
                        ordinal=int(index) - role_start,
                        identity=identity, role=role, label=int(labels[local]),
                        proxy=_proxy_particles(proxy_values, proxy_row),
                        offline=from_jc2(
                            columns, study["review"],
                            keys=tuple(f"offline_native:{item}" for item in range(count)),
                        ),
                    )
                block_cursor = block_stop
        if sha256_file(path) != source["sha256"]:
            raise ValueError("Proxy-ladder offline source changed during iteration")
        cursor = stop


def _identity_digest(rows: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes(row))
    return digest.hexdigest()


def _save_npz(path: Path, **arrays) -> None:
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_publish_bytes(path, stream.getvalue())


def _limit_worker_threads():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _match_sources(arguments):
    release, release_root, role, source_indices, capacity = arguments
    identities, offsets, mappings = [], [0], []
    max_proxy = max_offline = max_view = pairs = 0
    for row in iter_paired(
        release, release_root=Path(release_root), role=role,
        source_file_index=tuple(source_indices),
    ):
        mapping = match_particles(row.proxy, row.offline)
        identities.append(np.frombuffer(bytes.fromhex(row.identity), np.uint8))
        mappings.append(mapping.astype(np.int32, copy=False))
        offsets.append(offsets[-1] + len(mapping))
        pairs += int(np.count_nonzero(mapping >= 0))
        max_proxy = max(max_proxy, len(row.proxy))
        max_offline = max(max_offline, len(row.offline))
        max_view = max(max_view, len(row.proxy), len(row.offline))
        if max_view > capacity:
            raise ValueError(
                f"Selected endpoint exceeds capacity {capacity}; truncation is forbidden"
            )
    return {
        "identity": np.asarray(identities, np.uint8).reshape(-1, 32),
        "offsets": np.asarray(offsets, np.int64),
        "mapping": np.concatenate(mappings) if mappings else np.empty(0, np.int32),
        "pairs": pairs, "max_proxy": max_proxy, "max_offline": max_offline,
        "max_view": max_view,
    }


def build_foundation(
    release: dict, *, release_root: Path, output_root: Path, capacity: int = 512,
    workers: int = 1,
) -> dict:
    """Compute the transferred matcher once; persist mappings, never full views."""
    validate_release(release, root=release_root)
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder foundation root must be fresh")
    if type(workers) is not int or not 1 <= workers <= 72:
        raise ValueError("Proxy-ladder foundation workers must be in [1,72]")
    inputs = input_contract(capacity=capacity)
    identities, offsets, mappings = [], [0], []
    role_offsets = {"train": [0, 0], "validation": [0, 0]}
    max_proxy = max_offline = max_view = 0
    total_pairs = 0
    release_arrays = load_bank(release, root=release_root)
    for role in ("train", "validation"):
        role_offsets[role][0] = len(identities)
        code = 0 if role == "train" else 1
        sources = sorted(set(map(int, release_arrays["source_file"][release_arrays["role"] == code])))
        chunks = [
            tuple(map(int, chunk))
            for chunk in np.array_split(sources, min(workers, len(sources)))
            if len(chunk)
        ]
        arguments = [
            (release, str(Path(release_root).resolve()), role, chunk, capacity)
            for chunk in chunks
        ]
        if workers == 1:
            results = map(_match_sources, arguments)
        else:
            pool = ProcessPoolExecutor(
                max_workers=min(workers, len(arguments)),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_limit_worker_threads,
            )
            results = pool.map(_match_sources, arguments)
        try:
            for result in results:
                base = offsets[-1]
                identities.extend(result["identity"])
                mappings.append(result["mapping"])
                offsets.extend((result["offsets"][1:] + base).tolist())
                total_pairs += result["pairs"]
                max_proxy = max(max_proxy, result["max_proxy"])
                max_offline = max(max_offline, result["max_offline"])
                max_view = max(max_view, result["max_view"])
        finally:
            if workers != 1:
                pool.shutdown(wait=True, cancel_futures=True)
        role_offsets[role][1] = len(identities)
    identity_array = np.asarray(identities, np.uint8).reshape(-1, 32)
    mapping_array = np.concatenate(mappings) if mappings else np.empty(0, np.int32)
    offsets_array = np.asarray(offsets, np.int64)
    root.mkdir(parents=True, exist_ok=False)
    bank_path = root / "assignments.npz"
    _save_npz(
        bank_path, identity=identity_array, offsets=offsets_array, mapping=mapping_array,
    )
    # Deterministic endpoint audit over the leading 64 rows of each role.
    checks = 0
    for role in ("train", "validation"):
        lo, hi = role_offsets[role]
        selected_mappings = load_assignment_arrays_from_values(
            identity_array, offsets_array, mapping_array, lo, min(hi, lo + 64),
        )
        role_checks = 0
        for row, (identity, mapping) in zip(
            iter_paired(release, release_root=release_root, role=role), selected_mappings, strict=False,
        ):
            if role_checks >= 64:
                break
            if identity != row.identity:
                raise ValueError("Assignment audit identity differs")
            views = {
                name: build_view(
                    identity=row.identity, proxy=row.proxy, offline=row.offline,
                    coordinate=name, mapping=mapping,
                )
                for name in ("U000", "U100", "D050", "D000", "OFFLINE")
            }
            if (
                len(views["U000"]) != max(len(row.proxy), len(row.offline))
                or len(views["U100"]) != len(row.proxy)
                or not np.array_equal(views["D000"].p4, row.proxy.p4)
                or not np.array_equal(views["OFFLINE"].p4, row.offline.p4)
            ):
                raise ValueError("Persistent-proxy endpoint audit differs")
            for view in views.values():
                build_inputs(view, capacity=capacity)
            checks += 1
            role_checks += 1
    foundation = artifact(
        "FOUNDATION",
        parents={
            "release": release["content_hash"],
            "matcher": matcher_spec(CANDIDATE)["content_hash"],
            "views": view_contract()["content_hash"],
            "inputs": inputs["content_hash"],
        },
        release=release,
        release_root=str(Path(release_root).resolve()),
        matcher=matcher_spec(CANDIDATE), views=view_contract(), inputs=inputs,
        assignments=file_ref(bank_path, root=root),
        role_offsets=role_offsets,
        role_counts=release["counts"],
        identity_sha256={
            role: _identity_digest(identity_array[slice(*role_offsets[role])])
            for role in ("train", "validation")
        },
        mapping_orientation="proxy_slot_to_native_offline_index_or_minus_one",
        selected_pairs=total_pairs,
        max_proxy_particles=max_proxy,
        max_offline_particles=max_offline,
        max_view_particles=max_view,
        exact_endpoint_audit_rows=checks,
        preparation_workers=workers,
        full_views_persisted=False,
        assignments_recomputed_for_proxy_endpoint=True,
    )
    write_json(root / "foundation.json", foundation)
    validate_foundation(foundation, root=root)
    return foundation


def load_assignment_arrays_from_values(identity, offsets, mapping, start, stop):
    result = []
    for index in range(start, stop):
        lo, hi = offsets[index:index + 2]
        result.append((bytes(identity[index]).hex(), mapping[int(lo):int(hi)]))
    return result


def load_assignments(foundation: dict, *, root: Path, role: str):
    validate_foundation(foundation, root=root, check_bank=False)
    path = validate_file_ref(foundation["assignments"], root=root)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ASSIGNMENT_FIELDS:
            raise ValueError("Proxy-ladder assignment bank fields differ")
        identity, offsets, mapping = (archive[name] for name in ("identity", "offsets", "mapping"))
        lo, hi = foundation["role_offsets"][role]
        identity = identity[lo:hi].copy()
        base = int(offsets[lo])
        role_offsets = offsets[lo:hi + 1].copy() - base
        role_mapping = mapping[base:int(offsets[hi])].copy()
    return identity, role_offsets, role_mapping


def validate_foundation(foundation: dict, *, root: Path, check_bank: bool = True) -> str:
    parents = {
        "release": foundation["release"]["content_hash"],
        "matcher": foundation["matcher"]["content_hash"],
        "views": foundation["views"]["content_hash"],
        "inputs": foundation["inputs"]["content_hash"],
    }
    digest = validate(foundation, "FOUNDATION", parents=parents)
    validate_release(foundation["release"], root=Path(foundation["release_root"]))
    if (
        foundation["matcher"] != matcher_spec(CANDIDATE)
        or foundation["views"] != view_contract()
        or foundation["inputs"] != input_contract(capacity=foundation["inputs"]["capacity"])
        or foundation["role_counts"] != foundation["release"]["counts"]
        or foundation["mapping_orientation"] != "proxy_slot_to_native_offline_index_or_minus_one"
        or foundation["full_views_persisted"] is not False
        or foundation["assignments_recomputed_for_proxy_endpoint"] is not True
        or type(foundation["preparation_workers"]) is not int
        or not 1 <= foundation["preparation_workers"] <= 72
        or foundation["max_view_particles"] > foundation["inputs"]["capacity"]
    ):
        raise ValueError("Proxy-ladder foundation semantics differ")
    if check_bank:
        total = 0
        for role in ("train", "validation"):
            identity, offsets, mapping = load_assignments(foundation, root=root, role=role)
            if (
                len(identity) != foundation["role_counts"][role]
                or offsets.dtype != np.int64 or offsets.shape != (len(identity) + 1,)
                or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                or mapping.dtype != np.int32 or len(mapping) != offsets[-1]
                or _identity_digest(identity) != foundation["identity_sha256"][role]
            ):
                raise ValueError("Proxy-ladder assignment population differs")
            total += int(np.count_nonzero(mapping >= 0))
        if total != foundation["selected_pairs"]:
            raise ValueError("Proxy-ladder selected pair count differs")
    return digest


__all__ = [
    "ASSIGNMENT_FIELDS", "PairedRow", "build_foundation", "iter_paired",
    "load_assignments", "validate_foundation",
]
