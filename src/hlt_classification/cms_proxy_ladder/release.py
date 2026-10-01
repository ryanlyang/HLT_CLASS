"""Authenticated label-blind 200k/100k ordinary-role proxy release."""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import hashlib
import heapq

import numpy as np

from hlt_classification.cms2jc2_production import output, population
from hlt_classification.cms2jc2_production.campaign import validate_study
from hlt_classification.cms2jc2_production.contracts import load_json, safe as source_safe
from hlt_classification.data.cache_contracts import atomic_publish_bytes, sha256_file

from .contracts import artifact, file_ref, safe, validate, validate_file_ref, write_json

COUNTS = {"train": 200_000, "validation": 100_000}
SELECTION_DOMAIN = "JETCLASS2_CMS_PROXY_LADDER_200K_100K/v1"
BANK_FIELDS = {
    "identity", "role", "source_file", "entry", "proxy_block", "proxy_row",
}


def release_request(*, study_root: Path, offline_root: Path) -> dict:
    return artifact(
        "RELEASE_REQUEST",
        study_root=str(Path(study_root).resolve()),
        offline_root=str(Path(offline_root).resolve()),
        counts=COUNTS,
        selection_domain=SELECTION_DOMAIN,
        labels_read=False,
        selection_depends_on_labels=False,
        allowed_roles=list(COUNTS),
    )


def validate_request(value: dict) -> str:
    digest = validate(value, "RELEASE_REQUEST")
    if value != release_request(
        study_root=Path(value["study_root"]), offline_root=Path(value["offline_root"]),
    ):
        raise ValueError("Proxy-ladder release request differs")
    return digest


def _rank(study_hash: str, role: str, identity: str) -> tuple[bytes, bytes]:
    payload = b"\0".join((SELECTION_DOMAIN.encode(), study_hash.encode(), role.encode(), bytes.fromhex(identity)))
    return hashlib.sha256(payload).digest(), bytes.fromhex(identity)


def _identity_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for row in values:
        digest.update(bytes(row))
    return digest.hexdigest()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_publish_bytes(path, stream.getvalue())


def build_release(request: dict, *, output_root: Path) -> dict:
    """Freeze a committed ordinary-role snapshot without touching labels/test."""
    validate_request(request)
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("Proxy-ladder release root must be fresh")
    study_root = Path(request["study_root"]).resolve(strict=True)
    offline_root = Path(request["offline_root"]).resolve(strict=True)
    study = load_json(study_root / "study_spec.json")
    validate_study(study)
    if Path(study["data_root"]).resolve() != offline_root:
        raise ValueError("Requested offline root differs from the proxy study source")
    # Do not call output.completed(): it deliberately scans every receipt and
    # would authenticate final-test blocks after a full production completes.
    # This ordinary reader opens only exact registered train/validation paths.
    ordinary = {}
    for shard in study["shards"]:
        if shard["role"] not in COUNTS:
            continue
        path = source_safe(study_root, f"shards/{shard['shard_id']}.json")
        if not path.is_file():
            continue
        receipt = output.verify_shard(study, load_json(path), physical=False)
        if receipt["role"] != shard["role"]:
            raise ValueError("Ordinary proxy receipt role differs")
        ordinary[shard["shard_id"]] = receipt
    if not ordinary:
        raise ValueError("No committed ordinary proxy shards are available")
    shard_specs = {row["shard_id"]: row for row in study["shards"]}
    candidates: dict[str, list[tuple]] = defaultdict(list)
    blocks, block_index = [], {}
    receipts = []
    source_rows = {row["path"]: row for row in study["population"]["files"]}
    for shard_id in sorted(ordinary):
        receipt = ordinary[shard_id]
        shard = shard_specs[shard_id]
        source, entries = population.entries_for(study["population"], shard)
        if source["role"] != receipt["role"] or len(entries) != receipt["jets"]:
            raise ValueError("Committed receipt/source population differs")
        receipts.append({
            "shard_id": shard_id, "role": receipt["role"],
            "content_hash": receipt["content_hash"],
            "ordered_identities": receipt["ordered_identities"],
        })
        cursor = 0
        for block in receipt["blocks"]:
            relative = block["relative"]
            path = source_safe(study_root, relative)
            values = output.arrays(path)
            count = len(values["jet_identity"])
            selected_entries = entries[cursor:cursor + count]
            if len(selected_entries) != count:
                raise ValueError("Proxy block/source-entry alignment differs")
            key = (relative, block["sha256"], block["bytes"])
            if key not in block_index:
                block_index[key] = len(blocks)
                blocks.append({"path": relative, "sha256": block["sha256"], "bytes": block["bytes"]})
            index = block_index[key]
            for row, (raw_identity, entry) in enumerate(zip(values["jet_identity"], selected_entries, strict=True)):
                identity = bytes(raw_identity).hex()
                candidates[receipt["role"]].append((
                    _rank(study["content_hash"], receipt["role"], identity),
                    identity, source["path"], int(entry), index, row,
                ))
            cursor += count
        if cursor != len(entries):
            raise ValueError("Committed shard blocks omit source entries")
    chosen = []
    for role, target in COUNTS.items():
        available = candidates[role]
        if len(available) < target:
            raise ValueError(f"Insufficient committed {role} proxy rows: {len(available)} < {target}")
        chosen.extend((role, *row[1:]) for row in heapq.nsmallest(target, available, key=lambda row: row[0]))
    used_paths = sorted({row[2] for row in chosen})
    source_files = []
    source_index = {}
    for name in used_paths:
        row = source_rows[name]
        path = offline_root / name
        if sha256_file(path) != row["sha256"]:
            raise ValueError("Offline source checksum differs while freezing release")
        source_index[name] = len(source_files)
        source_files.append({
            "path": name, "sha256": row["sha256"], "bytes": path.stat().st_size,
            "tree_key": row["tree_key"], "raw_entries": row["raw_entries"],
            "source": row["source"],
        })
    # Stable physical-read order: role, source file, proxy block, proxy row.
    chosen.sort(key=lambda row: (
        0 if row[0] == "train" else 1, source_index[row[2]], row[4], row[5], row[1],
    ))
    role = np.asarray([0 if row[0] == "train" else 1 for row in chosen], np.uint8)
    identities = np.asarray([np.frombuffer(bytes.fromhex(row[1]), np.uint8) for row in chosen], np.uint8)
    arrays = {
        "identity": identities,
        "role": role,
        "source_file": np.asarray([source_index[row[2]] for row in chosen], np.int32),
        "entry": np.asarray([row[3] for row in chosen], np.int64),
        "proxy_block": np.asarray([row[4] for row in chosen], np.int32),
        "proxy_row": np.asarray([row[5] for row in chosen], np.int32),
    }
    root.mkdir(parents=True, exist_ok=False)
    bank_path = root / "release_index.npz"
    _save_npz(bank_path, arrays)
    manifest = artifact(
        "RELEASE",
        parents={"request": request["content_hash"], "study": study["content_hash"]},
        request=request,
        study_contract=study["contract"],
        study_root=str(study_root), offline_root=str(offline_root),
        scope=("complete_dataset_manifest" if (study_root / "dataset_manifest.json").is_file()
               else "committed_ordinary_receipt_snapshot"),
        counts=COUNTS,
        selection_domain=SELECTION_DOMAIN,
        labels_read=False,
        selection_depends_on_labels=False,
        receipts=receipts,
        source_files=source_files,
        proxy_blocks=blocks,
        bank=file_ref(bank_path, root=root),
        identity_sha256={
            name: _identity_digest(identities[role == code])
            for name, code in (("train", 0), ("validation", 1))
        },
        total_rows=len(chosen),
    )
    write_json(root / "release.json", manifest)
    validate_release(manifest, root=root)
    return manifest


def load_bank(manifest: dict, *, root: Path) -> dict[str, np.ndarray]:
    validate_release(manifest, root=root, check_bank=False)
    path = validate_file_ref(manifest["bank"], root=root)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != BANK_FIELDS:
            raise ValueError("Proxy-ladder release bank fields differ")
        arrays = {name: archive[name] for name in BANK_FIELDS}
    count = manifest["total_rows"]
    if (
        arrays["identity"].dtype != np.uint8 or arrays["identity"].shape != (count, 32)
        or arrays["role"].dtype != np.uint8 or arrays["role"].shape != (count,)
        or not np.isin(arrays["role"], (0, 1)).all()
        or arrays["source_file"].dtype != np.int32
        or arrays["entry"].dtype != np.int64
        or arrays["proxy_block"].dtype != np.int32
        or arrays["proxy_row"].dtype != np.int32
        or any(arrays[name].shape != (count,) for name in BANK_FIELDS - {"identity", "role"})
        or np.any(arrays["source_file"] < 0)
        or np.any(arrays["source_file"] >= len(manifest["source_files"]))
        or np.any(arrays["proxy_block"] < 0)
        or np.any(arrays["proxy_block"] >= len(manifest["proxy_blocks"]))
    ):
        raise ValueError("Proxy-ladder release bank shape/dtype/range differs")
    return arrays


def validate_release(manifest: dict, *, root: Path, check_bank: bool = True) -> str:
    digest = validate(
        manifest, "RELEASE",
        parents={"request": manifest["request"]["content_hash"], "study": manifest["parents"]["study"]},
    )
    validate_request(manifest["request"])
    if (
        manifest["counts"] != COUNTS or manifest["selection_domain"] != SELECTION_DOMAIN
        or manifest["labels_read"] is not False
        or manifest["selection_depends_on_labels"] is not False
        or manifest["total_rows"] != sum(COUNTS.values())
        or manifest["scope"] not in {"complete_dataset_manifest", "committed_ordinary_receipt_snapshot"}
        or len({row["shard_id"] for row in manifest["receipts"]}) != len(manifest["receipts"])
    ):
        raise ValueError("Proxy-ladder release semantics differ")
    if check_bank:
        arrays = load_bank(manifest, root=root)
        for name, code in (("train", 0), ("validation", 1)):
            selected = arrays["identity"][arrays["role"] == code]
            if len(selected) != COUNTS[name] or _identity_digest(selected) != manifest["identity_sha256"][name]:
                raise ValueError("Proxy-ladder release identity population differs")
        if len({bytes(row) for row in arrays["identity"]}) != manifest["total_rows"]:
            raise ValueError("Proxy-ladder release identities overlap")
    return digest


__all__ = [
    "BANK_FIELDS", "COUNTS", "SELECTION_DOMAIN", "build_release", "load_bank",
    "release_request", "validate_release", "validate_request",
]
