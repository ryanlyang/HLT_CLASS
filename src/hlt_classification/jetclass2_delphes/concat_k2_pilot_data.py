"""Frozen parent-row subsets and bounded K2 caches; matching bytes stay intact."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path
import time

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .cache import RamBlock, RamCache, _limit_worker_threads
from .concat_k2_campaign import artifact, validate
from .concat_k2_data import load_assignment
from .concat_k2_pilot import DOMAIN
from .concat_k2_views import build_view, STRENGTHS
from .contracts import artifact as base_artifact, row_identity
from .inputs import build_inputs
from .inventory import verify_file
from .provenance import source_record
from .reader import DatasetReader
from .selection import selected_mask
from .split_registry import (build_design, _membership, unpack_entries, select_profile,
                             validate_registry)

SEED = 20260923


def select_rows(rows, quotas, *, role):
    """Rows are (identity, file_index, entry, label); independent of input order."""
    if role not in {"train", "validation"}:
        raise PermissionError("Pilot selection cannot open final test")
    selected = []
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate pilot candidate identity")
    for label, quota in enumerate(quotas):
        values = [r for r in rows if r[3] == label]
        if len(values) < quota:
            raise ValueError("Pilot class quota exceeds parent capacity")
        values.sort(key=lambda r: (hashlib.sha256(
            (DOMAIN + '/' + role + '/').encode() + bytes.fromhex(r[0])).digest(), r[0]))
        selected.extend(values[:quota])
    return selected


def _metadata(spec, role):
    """Scalar branches only, strictly inside parent's ordinary membership."""
    if role not in {"train", "validation"}:
        raise PermissionError("Final test is sealed")
    f = spec["foundation"]
    parent = f["splits"]
    members = {r["path"]: r for r in parent["memberships"][role]["files"]}
    rows = []
    for index, file in enumerate(f["inventory"]["files"]):
        member = members.get(file["path"])
        if member is None or not member["rows"]:
            continue
        entries = unpack_entries(member["entry_mask"], file["entries"])
        path = verify_file(Path(spec["data_root"]), file, f["inventory"])
        counts = np.zeros(11, np.int64)
        with uproot.open(path) as handle:
            tree = handle[file["tree_key"]]
            for start in range(0, file["entries"], 100_000):
                chosen = entries[np.searchsorted(entries, start):np.searchsorted(entries, start+100_000)]
                if not len(chosen):
                    continue
                a = tree.arrays(["jet_label", "hlt_matched"], entry_start=start,
                    entry_stop=min(start+100_000, file["entries"]), library="np")
                keep, labels = selected_mask(a["jet_label"], a["hlt_matched"],
                    file["source"], f["inventory"]["selection"])
                local = chosen-start
                if not keep[local].all():
                    raise ValueError("Pilot parent membership contains an ineligible row")
                counts += np.bincount(labels[local], minlength=11)
                rows.extend((row_identity(f["inventory"]["content_hash"], file["path"],
                    file["tree_key"], int(entry)), index, int(entry), int(label))
                    for entry, label in zip(chosen, labels[local]))
        if counts.tolist() != member["class_counts"]:
            raise ValueError("Pilot parent metadata class counts differ")
        verify_file(Path(spec["data_root"]), file, f["inventory"])
        print(f"JC2-K2-PILOT phase=select role={role} file={index} rows={len(rows)}", flush=True)
    if len(rows) != parent["role_counts"][role]:
        raise ValueError("Pilot parent metadata coverage differs")
    return rows


def build_population(spec, directory):
    f = spec["foundation"]; inv = f["inventory"]; parent = f["splits"]
    counts = spec["role_counts"]
    design = build_design(inv, parent["reservoirs"], training_sizes=(counts["train"],),
        validation_size=counts["validation"], test_size=parent["role_counts"]["final_test"], seed=SEED)
    members = {}
    for role in ("train", "validation"):
        quotas = (design["profiles"][0]["train_class_counts"] if role == "train"
                  else design["evaluation_class_counts"][role])
        chosen = select_rows(_metadata(spec, role), quotas, role=role)
        by_file = {}
        for index, (file, group) in enumerate(zip(inv["files"], parent["groups"])):
            if group["role"] != role:
                continue
            rows = sorted((r for r in chosen if r[1] == index), key=lambda r:r[2])
            by_file[file["path"]] = (np.asarray([r[2] for r in rows], np.int64),
                                     np.asarray([r[3] for r in rows], np.int64))
        key = design["profiles"][0]["name"] if role == "train" else role
        members[key] = _membership(design, inv, parent["reservoirs"], role, quotas, by_file)
    # Rebind metadata only: identical sealed entries, no final-test ROOT reads.
    old = parent["memberships"]["final_test"]
    fields = {k:v for k,v in old.items() if k not in {"contract", "schema_version", "content_hash", "design_sha256"}}
    members["final_test"] = base_artifact("ROLE_MEMBERSHIP", **fields, design_sha256=design["content_hash"])
    registry = base_artifact("SPLIT_REGISTRY", inventory_sha256=inv["content_hash"],
        reservoirs=parent["reservoirs"], design=design, memberships=members,
        producer=source_record("src/hlt_classification/jetclass2_delphes/concat_k2_pilot_data.py"),
        final_test_metadata_only=True, final_test_accessed=False)
    profile = select_profile(registry, inv, design["profiles"][0]["name"])
    result = artifact("PILOT_POPULATION", campaign_sha256=spec["content_hash"],
        parent_foundation_sha256=f["content_hash"], parent_splits_sha256=parent["content_hash"],
        policy=spec["population_policy"], rank_domain=DOMAIN, registry=registry,
        profile=profile, final_test_accessed=False)
    validate_population(spec, result)
    write_immutable_json(Path(directory)/"population.json", result)
    return result


def validate_population(spec, value):
    digest = validate(value, "PILOT_POPULATION")
    f = spec["foundation"]; parent = f["splits"]
    registry = value["registry"]
    validate_registry(registry, f["inventory"])
    expected = build_design(f["inventory"], parent["reservoirs"],
        training_sizes=(spec["role_counts"]["train"],), validation_size=spec["role_counts"]["validation"],
        test_size=parent["role_counts"]["final_test"], seed=SEED)
    p = select_profile(registry, f["inventory"], expected["profiles"][0]["name"])
    if (value["campaign_sha256"] != spec["content_hash"] or value["profile"] != p
            or value["parent_foundation_sha256"] != f["content_hash"]
            or value["parent_splits_sha256"] != parent["content_hash"]
            or value["policy"] != spec["population_policy"] or value["rank_domain"] != DOMAIN
            or registry["design"] != expected or p["role_counts"] != spec["role_counts"]
            or value["final_test_accessed"] is not False
            or registry["producer"]["file_sha256"] != source_record(
                "src/hlt_classification/jetclass2_delphes/concat_k2_pilot_data.py")["file_sha256"]):
        raise ValueError("Pilot population lineage/recipe differs")
    files = {r["path"]:r for r in f["inventory"]["files"]}
    for role in ("train", "validation", "final_test"):
        for old, new in zip(parent["memberships"][role]["files"], p["memberships"][role]["files"]):
            cap = files[old["path"]]["entries"]
            a, b = (unpack_entries(r["entry_mask"], cap) for r in (old,new))
            if np.any(~np.isin(b, a)) or (role == "final_test" and old != new):
                raise ValueError("Pilot membership escapes parent or changes sealed final test")
    return digest


def population(spec):
    result = load_json(Path(spec["campaign_root"])/"outputs/select_population/population.json")
    validate_population(spec, result)
    return result["profile"]


def selected_tasks(spec, profile):
    members = {r["path"]: r for role in ("train","validation")
               for r in profile["memberships"][role]["files"]}
    return [{**t, "rows":members[t["path"]]["rows"]} for t in spec["foundation"]["assignment_tasks"]
            if members[t["path"]]["rows"]]


def map_rows(foundation, profile, task):
    file = foundation["inventory"]["files"][task["file_index"]]
    def entries(splits):
        member = next(r for r in splits["memberships"][task["role"]]["files"] if r["path"] == task["path"])
        return unpack_entries(member["entry_mask"], file["entries"])
    old, new = entries(foundation["splits"]), entries(profile)
    indices = np.searchsorted(old, new)
    if np.any(indices >= len(old)) or not np.array_equal(old[indices], new):
        raise ValueError("Pilot map lookup escapes parent")
    return indices


def cache_bounds(spec):
    from .concat_k2_data import cache_bounds as parent_bounds
    f = {**spec["foundation"], "assignment_tasks":selected_tasks(spec, population(spec))}
    return parent_bounds({**spec, "foundation":f})


def _file_cache(args):
    f, data_root, root, profile, task, coordinate = args
    privileged = coordinate not in {"HLT_X1", "HLT_X3", "D000"}
    maps = (load_assignment(dict(foundation=f,campaign_root=root), task["file_index"])[1]
            if coordinate in {"D100", "D075", "D050", "D025"} else None)
    lookup = map_rows(f,profile,task) if maps is not None else None
    offsets, features, vectors, ids, labels = [0], [], [], [], []
    for row, jet in enumerate(DatasetReader(Path(data_root),f["inventory"],profile,
            role=task["role"], include_offline=privileged, file_paths=(task["path"],))):
        identity = np.frombuffer(bytes.fromhex(jet.identity),np.uint8)
        mapping = None
        if maps is not None:
            i = lookup[row]; start,end = maps["offsets"][i:i+2]
            if (not np.array_equal(identity,maps["identities"][i]) or end-start != len(jet.hlt)
                    or len(jet.offline) != maps["offline_counts"][i]):
                raise ValueError("Pilot map/native identity join differs")
            mapping = maps["mapping"][start:end]
        view = build_view(jet,coordinate,f["candidate"],mapping)
        value = build_inputs(view,capacity=f["inputs"]["capacity"])
        features.append(value.features[:,:len(view)].T.copy())
        vectors.append(value.vectors[:,:len(view)].T.copy())
        offsets.append(offsets[-1]+len(view)); ids.append(identity); labels.append(jet.label)
    if len(ids) != task["rows"]:
        raise ValueError("Pilot RAM block coverage differs")
    return RamBlock(task["file_index"],np.asarray(offsets,np.int64),np.concatenate(features),
        np.concatenate(vectors),np.asarray(ids,np.uint8).reshape(-1,32),np.asarray(labels,np.int64))


def prepare(spec,role,coordinate):
    if role not in {"train","validation"}:
        raise PermissionError("Final test is sealed")
    if coordinate not in {*STRENGTHS,"HLT_X1","HLT_X3","OFFLINE"}:
        raise ValueError("Unknown pilot view")
    profile = population(spec); f = spec["foundation"]
    tasks = [t for t in selected_tasks(spec,profile) if t["role"] == role]
    args = [(f,spec["data_root"],spec["campaign_root"],profile,t,coordinate) for t in tasks]
    limit = cache_bounds(spec)[role]; workers = spec["resources"]["train"]["cpus"]
    blocks = []; resident = 0; started = time.monotonic()
    def accept(block):
        nonlocal resident
        resident += block.nbytes + len(block.labels)*40
        if resident > limit:
            raise MemoryError("Pilot RAM ceiling exceeded; no disk spill")
        blocks.append(block)
        print(f"JC2-K2-PILOT phase=cache role={role} view={coordinate} files={len(blocks)}/{len(tasks)} seconds={time.monotonic()-started:.1f}",flush=True)
    if workers == 1:
        for arg in args: accept(_file_cache(arg))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=_limit_worker_threads) as pool:
            iterator = iter(args)
            pending = [pool.submit(_file_cache,a) for a in (next(iterator,None) for _ in range(workers)) if a is not None]
            while pending:
                accept(pending.pop(0).result()); a = next(iterator,None)
                if a is not None: pending.append(pool.submit(_file_cache,a))
    result = RamCache(blocks,role=role,foundation_sha256=f["content_hash"],coordinate_name=coordinate)
    if len(result) != spec["role_counts"][role]:
        raise ValueError("Pilot cache population differs")
    return result
