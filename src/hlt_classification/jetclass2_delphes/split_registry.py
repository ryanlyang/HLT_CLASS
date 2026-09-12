"""Explicit nested row subsets inside immutable whole-file role reservoirs.

Only label/matched metadata is read, including when freezing test membership.
No particle arrays, model inputs, matching, or inference are exposed here.
Per-file packed masks keep the entire registry small and relocatable.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import write_immutable_json
from .contracts import artifact, validate
from .inventory import validate_inventory, verify_file
from .provenance import source_record, validate_source_record
from .schema import CLASS_NAMES
from .selection import selected_mask
from .splits import ROLES, validate_splits

TRAINING_SIZES = (500_000, 1_000_000, 1_500_000, 2_000_000)
EVALUATION_SIZE = 1_000_000
SELECTION_SEED = 20260911
ENCODING = "base64_packbits_little_entry_index_v1"
ALGORITHM = "sha256_class_prefix_incremental_hamilton_minimum_one_v1"


def profile_name(size: int) -> str:
    return dict(zip(TRAINING_SIZES, ("TRAIN_500K", "TRAIN_1M", "TRAIN_1P5M", "TRAIN_2M"))).get(size, f"TRAIN_{size}")


def _positive_integer(value):
    return type(value) is int and value > 0


def _apportion(total: int, capacities: list[int]) -> list[int]:
    """Exact Hamilton allocation; no floating-point remainder/tie ambiguity."""
    available = sum(capacities)
    if total < 0 or total > available or any(n < 0 for n in capacities):
        raise ValueError("Subset exceeds available class capacity")
    if total == 0:
        return [0] * len(capacities)
    result = [total * n // available for n in capacities]
    order = sorted(range(len(capacities)), key=lambda i: (-(total * capacities[i] % available), i))
    for i in order[:total - sum(result)]:
        result[i] += 1
    return result


def nested_quotas(capacities: list[int], sizes: tuple[int, ...]) -> list[list[int]]:
    """Natural class proportions, exact nested counts, at least one per class."""
    if (len(capacities) != len(CLASS_NAMES) or any(not _positive_integer(n) for n in capacities)
            or not sizes or any(not _positive_integer(n) for n in sizes)
            or list(sizes) != sorted(set(sizes)) or sizes[0] < len(capacities)):
        raise ValueError("Invalid nested sizes or class coverage")
    previous = [1] * len(capacities)
    result = []
    for size in sizes:
        increment = _apportion(size - sum(previous), [n - p for n, p in zip(capacities, previous)])
        previous = [p + n for p, n in zip(previous, increment)]
        result.append(previous)
    return result


def build_design(inventory: dict, reservoirs: dict, *, training_sizes=TRAINING_SIZES,
                 validation_size=EVALUATION_SIZE, test_size=EVALUATION_SIZE,
                 seed=SELECTION_SEED) -> dict:
    inv_hash = validate_inventory(inventory)
    # Profiles cannot recursively redefine their own outer reservoirs.
    validate(reservoirs, "SPLITS")
    reservoir_hash = validate_splits(reservoirs, inventory)
    if type(seed) is not int or seed < 0:
        raise ValueError("Subset seed must be a nonnegative integer")
    sizes = tuple(training_sizes)
    train_quotas = nested_quotas(reservoirs["role_class_counts"]["train"], sizes)
    evaluation = {r: nested_quotas(reservoirs["role_class_counts"][r], (n,))[0]
                  for r, n in (("validation", validation_size), ("final_test", test_size))}
    return artifact(
        "SPLIT_DESIGN", inventory_sha256=inv_hash, reservoirs_sha256=reservoir_hash,
        selection_sha256=inventory["selection"]["content_hash"], class_names=list(CLASS_NAMES),
        seed=seed, algorithm=ALGORITHM, membership_encoding=ENCODING,
        profiles=[dict(name=profile_name(n), train_rows=n, train_class_counts=q) for n, q in zip(sizes, train_quotas)],
        evaluation_class_counts=evaluation,
        evaluation_rows={"validation": validation_size, "final_test": test_size},
        default_profile=profile_name(sizes[0]),
        cross_file_generator_independence="provisional_unverified",
        final_test_metadata_only=True, final_test_accessed=False,
    )


def _validate_design(design, inventory, reservoirs):
    validate(design, "SPLIT_DESIGN")
    expected = build_design(inventory, reservoirs,
                            training_sizes=tuple(p["train_rows"] for p in design["profiles"]),
                            validation_size=design["evaluation_rows"]["validation"],
                            test_size=design["evaluation_rows"]["final_test"], seed=design["seed"])
    if design != expected:
        raise ValueError("Subset design semantics differ")


def pack_entries(entries: np.ndarray, capacity: int) -> str:
    entries = np.asarray(entries)
    if (entries.ndim != 1 or entries.dtype.kind not in "iu" or np.any(entries >= capacity)
            or np.any(entries < 0) or len(np.unique(entries)) != len(entries)):
        raise ValueError("Duplicate/out-of-range selected entries")
    bits = np.zeros(capacity, np.uint8)
    bits[entries] = 1
    return base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode("ascii")


def unpack_entries(encoded: str, capacity: int) -> np.ndarray:
    if type(encoded) is not str or len(encoded) != 4 * (((capacity + 7) // 8 + 2) // 3):
        raise ValueError("Entry mask length differs")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid entry mask encoding") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("Noncanonical entry mask encoding")
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")
    if len(raw) != (capacity + 7) // 8 or np.any(bits[capacity:]):
        raise ValueError("Entry mask has out-of-range padding bits")
    return np.flatnonzero(bits[:capacity])


def _membership(design, inventory, reservoirs, role, quotas, selected_by_file):
    files = []
    for row, group in zip(inventory["files"], reservoirs["groups"]):
        if group["role"] != role:
            continue
        entries, labels = selected_by_file[row["path"]]
        files.append(dict(path=row["path"], rows=len(entries),
                          class_counts=np.bincount(labels.astype(int), minlength=len(CLASS_NAMES)).tolist(),
                          entry_mask=pack_entries(entries, row["entries"])))
    return artifact(
        "ROLE_MEMBERSHIP", inventory_sha256=inventory["content_hash"],
        reservoirs_sha256=reservoirs["content_hash"], design_sha256=design["content_hash"],
        role=role, rows=sum(quotas), class_counts=quotas, files=files,
        encoding=ENCODING, iteration_order="inventory_file_order_then_increasing_entry",
        final_test_metadata_only=True, final_test_accessed=False,
    )


def validate_membership(membership, inventory, reservoirs, design, *, role, quotas):
    digest = validate(membership, "ROLE_MEMBERSHIP")
    if (membership["inventory_sha256"] != inventory["content_hash"]
            or membership["reservoirs_sha256"] != reservoirs["content_hash"]
            or membership["design_sha256"] != design["content_hash"]
            or membership["role"] != role or membership["rows"] != sum(quotas)
            or membership["class_counts"] != quotas or membership["encoding"] != ENCODING
            or membership["iteration_order"] != "inventory_file_order_then_increasing_entry"
            or membership["final_test_metadata_only"] is not True or membership["final_test_accessed"] is not False):
        raise ValueError("Membership lineage/counts/role differ")
    expected = [f for f, g in zip(inventory["files"], reservoirs["groups"]) if g["role"] == role]
    if [f["path"] for f in membership["files"]] != [f["path"] for f in expected]:
        raise ValueError("Membership file coverage escapes reservoir role")
    counts = np.zeros(len(CLASS_NAMES), np.int64)
    for row, source in zip(membership["files"], expected):
        selected = unpack_entries(row["entry_mask"], source["entries"])
        values = row["class_counts"]
        if (len(values) != len(CLASS_NAMES) or any(type(n) is not int or n < 0 for n in values)
                or any(n > cap for n, cap in zip(values, source["selected_class_counts"]))
                or sum(values) != len(selected) or type(row["rows"]) is not int or row["rows"] != len(selected)):
            raise ValueError("Membership file counts or mask coverage differ")
        counts += np.asarray(values, np.int64)
    if counts.tolist() != quotas:
        raise ValueError("Membership class totals differ")
    return digest


def _ranked_metadata(data_root, inventory, reservoirs, role, seed, step_size):
    """One role at a time; memory holds compact row metadata, never particles."""
    dtype = np.dtype([("priority", "V32"), ("file", "<u4"), ("entry", "<u8"), ("label", "u1")])
    blocks = []
    for file_index, (file, group) in enumerate(zip(inventory["files"], reservoirs["groups"])):
        if group["role"] != role:
            continue
        path = verify_file(Path(data_root), file, inventory)
        prefix = f"JC2/subset/v1/{seed}/{inventory['content_hash']}/{role}/{file['path']}/{file['tree_key']}/".encode()
        counts = np.zeros(len(CLASS_NAMES), np.int64)
        with uproot.open(path) as handle:
            tree = handle[file["tree_key"]]
            for start in range(0, file["entries"], step_size):
                # Explicitly limited metadata capability, even for final_test.
                arrays = tree.arrays(["jet_label", "hlt_matched"], entry_start=start,
                                     entry_stop=min(start + step_size, file["entries"]), library="np")
                keep, labels = selected_mask(arrays["jet_label"], arrays["hlt_matched"], file["source"], inventory["selection"])
                entries = start + np.flatnonzero(keep)
                block = np.empty(len(entries), dtype)
                block["file"], block["entry"], block["label"] = file_index, entries, labels[keep]
                block["priority"] = np.fromiter((hashlib.sha256(prefix + str(int(e)).encode()).digest()
                                                 for e in entries), dtype="V32", count=len(entries))
                blocks.append(block)
                counts += np.bincount(labels[keep], minlength=len(CLASS_NAMES))
        if counts.tolist() != file["selected_class_counts"]:
            raise ValueError("Scalar metadata differs from inventory class counts")
        verify_file(Path(data_root), file, inventory)
        print(f"JC2 phase=split_metadata role={role} file_index={file_index} selected={int(counts.sum())}", flush=True)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype)


def build_registry(data_root: Path, inventory: dict, reservoirs: dict, *,
                   training_sizes=TRAINING_SIZES, validation_size=EVALUATION_SIZE,
                   test_size=EVALUATION_SIZE, seed=SELECTION_SEED, step_size=100_000) -> dict:
    """Build metadata only. Nondefault sizes support explicit future/toy designs."""
    if not _positive_integer(step_size):
        raise ValueError("Metadata step_size must be positive")
    design = build_design(inventory, reservoirs, training_sizes=training_sizes,
                          validation_size=validation_size, test_size=test_size, seed=seed)
    memberships = {}
    for role in ROLES:
        rows = _ranked_metadata(data_root, inventory, reservoirs, role, seed, step_size)
        quotas_list = ([p["train_class_counts"] for p in design["profiles"]] if role == "train"
                       else [design["evaluation_class_counts"][role]])
        ranked = []
        for label in range(len(CLASS_NAMES)):
            candidates = rows[rows["label"] == label]
            ranked.append(np.sort(candidates, order=["priority", "file", "entry"]))
        # Free the unsorted role array before materializing masks.
        del rows
        for index, quotas in enumerate(quotas_list):
            selected = np.concatenate([r[:n] for r, n in zip(ranked, quotas)])
            by_file = {}
            for file_index, (file, group) in enumerate(zip(inventory["files"], reservoirs["groups"])):
                if group["role"] == role:
                    values = selected[selected["file"] == file_index]
                    by_file[file["path"]] = (values["entry"], values["label"])
            key = design["profiles"][index]["name"] if role == "train" else role
            memberships[key] = _membership(design, inventory, reservoirs, role, quotas, by_file)
        del ranked, selected
    result = artifact(
        "SPLIT_REGISTRY", inventory_sha256=inventory["content_hash"], reservoirs=reservoirs,
        design=design, memberships=memberships,
        producer=source_record(*[f"src/hlt_classification/jetclass2_delphes/{name}.py"
                                 for name in ("split_registry", "splits", "inventory", "selection", "schema", "contracts")]),
        final_test_metadata_only=True, final_test_accessed=False,
    )
    validate_registry(result, inventory)
    return result


def validate_registry(registry: dict, inventory: dict) -> str:
    digest = validate(registry, "SPLIT_REGISTRY")
    validate_source_record(registry["producer"])
    if (registry["inventory_sha256"] != inventory["content_hash"]
            or registry["final_test_metadata_only"] is not True or registry["final_test_accessed"] is not False):
        raise ValueError("Registry source/test capability differs")
    design, reservoirs, memberships = registry["design"], registry["reservoirs"], registry["memberships"]
    _validate_design(design, inventory, reservoirs)
    profiles = design["profiles"]
    if set(memberships) != {p["name"] for p in profiles} | {"validation", "final_test"}:
        raise ValueError("Registry membership coverage differs")
    previous = None
    for p in profiles:
        member = memberships[p["name"]]
        validate_membership(member, inventory, reservoirs, design, role="train", quotas=p["train_class_counts"])
        if previous is not None:
            for left, right in zip(previous["files"], member["files"]):
                a, b = (np.frombuffer(base64.b64decode(r["entry_mask"]), np.uint8) for r in (left, right))
                if np.any(a & ~b):
                    raise ValueError("Training memberships are not nested")
        previous = member
    for role in ("validation", "final_test"):
        validate_membership(memberships[role], inventory, reservoirs, design,
                            role=role, quotas=design["evaluation_class_counts"][role])
    return digest


def select_profile(registry: dict, inventory: dict, name: str) -> dict:
    parent = validate_registry(registry, inventory)
    matches = [p for p in registry["design"]["profiles"] if p["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unregistered split profile: {name}")
    members = {r: registry["memberships"][name if r == "train" else r] for r in ROLES}
    return artifact(
        "SPLIT_PROFILE", registry_sha256=parent, profile=name, design=registry["design"],
        reservoirs=registry["reservoirs"], inventory_sha256=inventory["content_hash"],
        selection_sha256=inventory["selection"]["content_hash"], groups=registry["reservoirs"]["groups"],
        memberships=members, role_counts={r: m["rows"] for r, m in members.items()},
        role_class_counts={r: m["class_counts"] for r, m in members.items()},
        final_test_accessed=False,
    )


def validate_split_profile(profile: dict, inventory: dict) -> str:
    digest = validate(profile, "SPLIT_PROFILE")
    _validate_design(profile["design"], inventory, profile["reservoirs"])
    if (profile["inventory_sha256"] != inventory["content_hash"]
            or profile["selection_sha256"] != inventory["selection"]["content_hash"]
            or profile["groups"] != profile["reservoirs"]["groups"]
            or profile["final_test_accessed"] is not False
            or not isinstance(profile["registry_sha256"], str) or len(profile["registry_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in profile["registry_sha256"])):
        raise ValueError("Split profile source/roles/registry lineage differ")
    p = next((p for p in profile["design"]["profiles"] if p["name"] == profile["profile"]), None)
    if p is None or set(profile["memberships"]) != set(ROLES):
        raise ValueError("Split profile is not registered")
    for role in ROLES:
        quotas = p["train_class_counts"] if role == "train" else profile["design"]["evaluation_class_counts"][role]
        validate_membership(profile["memberships"][role], inventory, profile["reservoirs"],
                            profile["design"], role=role, quotas=quotas)
        if profile["role_counts"][role] != sum(quotas) or profile["role_class_counts"][role] != quotas:
            raise ValueError("Split profile totals differ")
    return digest


def publish_registry(root: Path, registry: dict, inventory: dict) -> None:
    """Fresh immutable bundle; shared role hashes survive export and relocation."""
    validate_registry(registry, inventory)
    if Path(root).exists():
        raise FileExistsError("Split registry publication requires a new output directory")
    for p in registry["design"]["profiles"]:
        profile = select_profile(registry, inventory, p["name"])
        write_immutable_json(Path(root) / "profiles" / f"{p['name']}.json", profile)
    # Registry is the last publication/commit marker for the complete bundle.
    write_immutable_json(Path(root) / "registry.json", registry)
